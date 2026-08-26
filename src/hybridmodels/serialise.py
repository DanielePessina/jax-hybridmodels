"""Writing a trained model to disk, and reading it back.

The trainable part of a model here is a pytree of ``eqx.Module`` leaves
(a nested container JAX can flatten and rebuild). By convention it is a
tuple of ``BoundedPredictor`` leaves, but any shape is accepted: tuple,
list, dict, NamedTuple, or a bare ``eqx.Module``. This module writes that
pytree to disk with the configuration needed to reconstruct it, and reads
it back.

Two save/load pairs exist.

``save_predictors`` / ``load_predictors``
    A single binary file, written by ``eqx.tree_serialise_leaves`` and read
    by its inverse. The caller picks the file extension (``save_run`` uses
    ``.eqx``). Loading needs a *template*, a live pytree with the same
    container shape and the same per-leaf static configuration as the one
    that was saved. Its JAX-array leaves are overwritten from the file.

``save_run`` / ``load_run``
    A directory holding a ``predictors.eqx`` binary and a
    ``metadata.json`` describing the run. The JSON carries a UTC ISO-8601
    timestamp, the package version, a structural fingerprint of the
    predictors pytree (a treedef repr plus one ``{path, class}`` entry per
    ``eqx.Module`` leaf), the solver config dict, optional Optax and
    Evosax training configs, an optional loss history, and a free-form
    user-owned ``extras`` dict. Stable formatting
    (``indent=2, sort_keys=True``) keeps diffs reviewable.

The structural fingerprint exists because
``eqx.tree_deserialise_leaves`` raises a generic shape error when a
template does not match the saved tree. The metadata's ``predictors``
block lets a user see the mismatch directly. A wrong container shape
(tuple against dict, wrong arity) shows in ``tree_structure``. A wrong
leaf type (``MLPPredictor`` saved, ``KANPredictor`` in the template) shows
in the per-leaf list.

What is not serialised
----------------------
``simulate_fn``, ``state_to_output``, the ``Dataset``, and the trainable
mask. The framework owns no builder registry that could re-import user
code by name. Loading therefore asks the caller to rebuild a
same-architecture template, which gives
``eqx.tree_deserialise_leaves`` somewhere to put the stored leaves, and to
re-import their own physics functions. That boundary is deliberate.
Dynamic imports inside a load helper fail in confusing ways.

Loss-field handling
-------------------
``OptaxTrainingConfig.loss`` and ``EvosaxTrainingConfig.loss`` accept
either a callable or a string. JSON cannot encode a callable, so
``save_run`` writes a callable as ``"{module}.{qualname}"`` and
``load_run`` returns that string as it stands. The field's
``Callable | str`` annotation accepts it. Turning the string back into a
function is the caller's job, usually with
``importlib.import_module(module)`` and an attribute lookup. Keeping that
boundary explicit avoids the security and brittleness costs of dynamic
imports inside this module.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.metadata as ilm
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from hybridmodels.solver import SolverConfig
from hybridmodels.training.evosax import EvosaxTrainingConfig
from hybridmodels.training.optax import OptaxTrainingConfig

_PREDICTORS_FILENAME = "predictors.eqx"
_METADATA_FILENAME = "metadata.json"


def save_predictors(path: str | Path, predictors: Any) -> None:
    """Write ``predictors`` to ``path`` via ``eqx.tree_serialise_leaves``.

    ``predictors`` is a pytree of ``eqx.Module`` leaves in any container
    shape (tuple, list, dict, NamedTuple, or a bare Module).
    ``eqx.tree_serialise_leaves`` walks the leaves the same way whatever the
    container is, writing a flat binary stream of ``np.save``-encoded
    leaves. The caller picks the file extension. ``save_run`` uses ``.eqx``.
    """
    path = Path(path)
    eqx.tree_serialise_leaves(path, predictors)


def load_predictors(path: str | Path, predictors_template: Any) -> Any:
    """Restore a ``predictors`` pytree from ``path`` using ``predictors_template`` as the skeleton.

    ``predictors_template`` must share the saved pytree's container shape
    and per-leaf static configuration, for example the same
    ``MLPPredictor`` ``in_size``, ``out_size``, ``width_size``, and
    ``depth`` on every predictor leaf. The file's values overwrite the
    template's array leaves. Its static fields stay as they are and supply
    the structure ``equinox`` needs to rebuild the tree.

    Returns the restored pytree. ``predictors_template`` is not mutated.
    """
    path = Path(path)
    return eqx.tree_deserialise_leaves(path, predictors_template)


def _stringify_loss_field(value: Any) -> Any:
    """Replace a callable with ``"{module}.{qualname}"``; pass-through otherwise.

    Used inside ``_serialise_training_config`` to keep ``loss``
    JSON-friendly. Strings pass through unchanged. Callables, the other
    half of the ``Callable | str`` field annotation, become strings.
    Re-importing on load is the caller's job. See the module docstring for
    why that boundary stays explicit.
    """
    if isinstance(value, str):
        return value
    if callable(value):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return f"{module}.{qualname}" if module else qualname
    return value


def _serialise_training_config(config: Any) -> dict[str, Any]:
    """Return a JSON-encodable view of an Optax/Evosax training config.

    ``dataclasses.asdict`` deep-copies the config into a dict tree, turning
    tuples into lists, since JSON has no tuple. One field needs extra care,
    ``loss``, which the dataclasses allow to be a callable. See
    ``_stringify_loss_field``.
    """
    raw = dataclasses.asdict(config)
    if "loss" in raw:
        # Read ``loss`` from the live config rather than the already-asdict'd
        # ``raw`` dict. asdict keeps the callable verbatim, but going through
        # the source object preserves the original identity unambiguously and
        # keeps stringification in one place (``_stringify_loss_field``).
        raw["loss"] = _stringify_loss_field(config.loss)
    return raw


def _resolve_version() -> str:
    """Return the installed ``hybridmodels`` package version, or ``"unknown"``.

    ``importlib.metadata.version`` raises ``PackageNotFoundError`` when the
    package is not installed, for instance when tests run against a checkout
    that has not been ``uv sync``'d. Swallow it, so a save never fails over
    metadata bookkeeping alone.
    """
    try:
        return ilm.version("hybridmodels")
    except ilm.PackageNotFoundError:
        return "unknown"


def _is_module(node: Any) -> bool:
    """Predicate used as ``is_leaf`` to stop pytree traversal at ``eqx.Module`` boundaries.

    ``equinox`` ships no ``is_module`` helper, since its public ``is_…``
    family covers arrays only. A named predicate keeps the intent, treat
    each Module as a leaf, readable at the call site.
    """
    return isinstance(node, eqx.Module)


def _describe_predictors(predictors: Any) -> dict[str, Any]:
    """Walk ``predictors`` once, returning a JSON-friendly structural hint.

    The hint records two views of the pytree.

    - ``tree_structure`` is the ``repr`` of
      ``jax.tree_util.tree_structure``, taken with the Module-stopped
      traversal. One string giving the container shape, in PyTreeDef
      notation: ``[*, *]`` for a 2-tuple, ``{'growth': *, 'nucleation': *}``
      for a dict, ``*`` for a bare Module.
    - ``leaves`` is one ``{path, class}`` entry per ``eqx.Module`` leaf, in
      ``jax.tree_util.tree_flatten_with_path`` order. ``path`` is the
      readable ``jax.tree_util.keystr`` form (``[0]``, ``['growth']``,
      ``.field``, or ``""`` for a single-Module pytree). ``class`` is
      ``"{module}.{qualname}"`` of the leaf's runtime type.

    Each view catches a different mismatch. The structure repr shows a
    shape mismatch, such as tuple against dict or the wrong arity. The
    per-leaf list shows a type mismatch, such as ``MLPPredictor`` saved
    against ``KANPredictor`` in the template. Together they fill the gap
    left by ``eqx.tree_deserialise_leaves`` failing with a generic shape
    error.
    """
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(predictors, is_leaf=_is_module)
    leaves: list[dict[str, str]] = []
    for path, leaf in leaves_with_paths:
        if not _is_module(leaf):
            # Skip non-Module leaves at the top. They are usually static
            # passengers left behind when a user nests modules under a static
            # field. The Module-stopped traversal already stops at Module
            # boundaries for everything reachable, so this guard exists only
            # to stop a stray scalar crashing the metadata write.
            continue
        cls = type(leaf)
        leaves.append(
            {
                "path": jtu.keystr(path),
                "class": f"{cls.__module__}.{cls.__qualname__}",
            }
        )
    return {
        "tree_structure": repr(treedef),
        "leaves": leaves,
    }


def _loss_history_kind(
    optax_config: Any, evosax_config: Any, loss_history: list[float] | None
) -> str | None:
    """Name the semantics of ``loss_history`` from whichever config is present.

    Optax returns the raw data loss at each step, which can go up. Evosax
    returns the best value seen so far, which cannot. Both are
    ``list[float]``, so without this label a reloaded run cannot tell
    whether a rise in the series is a real regression or impossible.
    """
    if loss_history is None:
        return None
    if optax_config is not None and evosax_config is None:
        return "per_step_data"
    if evosax_config is not None and optax_config is None:
        return "best_so_far"
    # Both or neither means the caller composed the two runs, so the series
    # cannot be labelled from here without guessing.
    return "unknown"


def save_run(
    directory: str | Path,
    *,
    predictors: Any,
    solver: SolverConfig,
    optax_config: OptaxTrainingConfig | None = None,
    evosax_config: EvosaxTrainingConfig | None = None,
    loss_history: list[float] | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Persist a complete training run to ``directory``.

    Layout written
    --------------
    ``directory/predictors.eqx``
        Binary leaves of the predictors pytree (``save_predictors`` output).
    ``directory/metadata.json``
        JSON dict with ``timestamp`` (UTC, ISO 8601, microsecond
        precision), ``version`` (``importlib.metadata.version``-resolved),
        ``predictors`` (``{tree_structure, leaves: [{path, class}, ...]}``
        per ``_describe_predictors``), ``solver`` (``solver.to_dict()``),
        ``optax_config`` / ``evosax_config`` (``dataclasses.asdict`` with
        stringified ``loss``, see module docstring), ``loss_history``,
        and ``extras``.

        ``loss_history_kind`` records which training entry point produced
        ``loss_history``. It is ``"per_step_data"`` for Optax (the raw
        value, which can go up) or ``"best_so_far"`` for Evosax (monotone).
        The two series have the same type and mean different things, so a
        saved run that did not say which it held could not be read back
        safely. It is inferred from whichever config was passed.

    The directory is created, parents included, if it does not exist.
    Existing files are overwritten. This saves rather than appends.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    save_predictors(directory / _PREDICTORS_FILENAME, predictors)

    metadata: dict[str, Any] = {
        # ``dt.UTC`` keeps the timestamp explicitly tz-aware, and the
        # microsecond suffix lands in the ISO string by default.
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "version": _resolve_version(),
        "predictors": _describe_predictors(predictors),
        "solver": solver.to_dict(),
        "optax_config": (
            _serialise_training_config(optax_config) if optax_config is not None else None
        ),
        "evosax_config": (
            _serialise_training_config(evosax_config) if evosax_config is not None else None
        ),
        "loss_history": list(loss_history) if loss_history is not None else None,
        "loss_history_kind": _loss_history_kind(optax_config, evosax_config, loss_history),
        "extras": dict(extras) if extras is not None else {},
    }

    with (directory / _METADATA_FILENAME).open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _filter_dataclass_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only ``raw`` keys that are declared fields of ``cls``.

    Lets ``load_run`` rebuild a frozen config even when the saved metadata
    carries fields the current dataclass no longer declares, for example
    after a config rename. Unknown fields are dropped without warning. A
    caller that wants strict loading must check for them itself.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _coerce_tuple_fields(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Re-tuple list-shaped values for fields whose type annotation is a tuple.

    JSON has no tuple, so what ``dataclasses.asdict`` wrote as a list comes
    back as a list. The training-config dataclasses are ``frozen=True`` and
    Python does not coerce in ``__init__``, so ``OptaxTrainingConfig(steps=[5])``
    constructs happily and then differs from a hand-written config in ways
    the annotations say it should not, including no longer being hashable.
    Re-tuple whenever the field's annotation mentions ``tuple``.
    """
    coerced = dict(kwargs)
    for field in dataclasses.fields(cls):
        if field.name not in coerced:
            continue
        value = coerced[field.name]
        if isinstance(value, list):
            type_str = str(field.type)
            if "tuple" in type_str.lower():
                coerced[field.name] = tuple(value)
    return coerced


def _build_training_config(cls: type | None, raw: dict[str, Any] | None) -> Any:
    """Reconstruct a training config from its dict, or pass through if no class given.

    Returns ``None`` when ``raw`` is ``None``, meaning no config was saved.
    When ``cls`` is ``None`` but ``raw`` is populated, the dict passes
    through unchanged, because the caller asked for no instantiation.
    Otherwise unknown keys are dropped (for forward compatibility) and JSON
    lists are re-tupled.
    """
    if raw is None:
        return None
    if cls is None:
        return raw
    kwargs = _filter_dataclass_kwargs(cls, raw)
    kwargs = _coerce_tuple_fields(cls, kwargs)
    return cls(**kwargs)


def load_run(
    directory: str | Path,
    *,
    predictors_template: Any,
    optax_cls: type[OptaxTrainingConfig] | None = None,
    evosax_cls: type[EvosaxTrainingConfig] | None = None,
) -> dict[str, Any]:
    """Reconstruct a run from ``directory``.

    Inverse of :func:`save_run`. ``predictors_template`` is required,
    because the framework owns no builder registry (see the module
    docstring), and it must share the saved pytree's container shape and
    per-leaf static configuration.

    ``optax_cls`` and ``evosax_cls`` are optional. Pass them to get the
    metadata's ``optax_config`` and ``evosax_config`` dicts rebuilt as typed
    dataclass instances. Leave them out and the raw dicts come back.

    Loss-field policy
    -----------------
    A ``loss`` field that was a callable at save time arrives here as the
    string ``"{module}.{qualname}"``. The dataclass field type
    (``Callable | str``) accepts that string as it stands, and no implicit
    re-import happens. Turning it back into a function is the caller's job,
    usually ``importlib.import_module(module).qualname``. That boundary
    stays explicit because dynamic imports inside a load helper fail in
    confusing ways.

    Returns
    -------
    dict
        Keys: ``predictors`` (``PyTree[eqx.Module]``), ``solver``
        (``SolverConfig``), ``optax_config`` (``OptaxTrainingConfig`` |
        dict | None), ``evosax_config`` (``EvosaxTrainingConfig`` | dict |
        None), ``loss_history`` (``list[float] | None``),
        ``loss_history_kind`` (``str | None``), ``extras`` (``dict``).
    """
    directory = Path(directory)
    with (directory / _METADATA_FILENAME).open() as f:
        metadata: dict[str, Any] = json.load(f)

    solver = SolverConfig.from_dict(metadata["solver"])
    optax_config = _build_training_config(optax_cls, metadata.get("optax_config"))
    evosax_config = _build_training_config(evosax_cls, metadata.get("evosax_config"))
    predictors = load_predictors(directory / _PREDICTORS_FILENAME, predictors_template)

    return {
        "predictors": predictors,
        "solver": solver,
        "optax_config": optax_config,
        "evosax_config": evosax_config,
        "loss_history": metadata.get("loss_history"),
        "loss_history_kind": metadata.get("loss_history_kind"),
        "extras": metadata.get("extras", {}),
    }


__all__ = [
    "load_predictors",
    "load_run",
    "save_predictors",
    "save_run",
]
