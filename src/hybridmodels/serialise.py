"""Save and load helpers for a trained ``predictors`` pytree.

The framework's trainable component is a ``PyTree[eqx.Module]`` — by
convention a tuple of ``BoundedPredictor`` leaves, but any pytree shape is
accepted (tuple, list, dict, NamedTuple, or a bare ``eqx.Module``). This
module persists that pytree to disk together with the static configuration
needed to reconstruct it later, and exposes the inverse load operation.

Two save/load pairs are provided:

``save_predictors`` / ``load_predictors``
    Single-file binary round-trip via ``eqx.tree_serialise_leaves`` and
    its inverse. The caller picks the file extension (``.eqx`` is the
    convention used by ``save_run``) and, on load, supplies a *template*
    pytree whose container shape and per-leaf static configuration match
    the original; the dynamic JAX-array leaves are then overwritten from
    the file.

``save_run`` / ``load_run``
    Directory-shaped artifact: a ``predictors.eqx`` binary plus a
    ``metadata.json`` describing the run. The JSON carries a UTC ISO-8601
    timestamp, the package version, a structural fingerprint of the
    predictors pytree (treedef repr + one ``{path, class}`` entry per
    ``eqx.Module`` leaf), the solver config dict, optional Optax / Evosax
    training configs, an optional loss history, and a free-form
    user-owned ``extras`` dict. The JSON is written with stable
    formatting (``indent=2, sort_keys=True``) so diffs are reviewable.

Why a structural fingerprint? ``eqx.tree_deserialise_leaves`` raises a
generic shape error when the template does not match the saved tree. The
metadata's ``predictors`` block lets a user eyeball the mismatch directly
— wrong container shape (tuple-vs-dict, wrong arity) shows in
``tree_structure``; wrong leaf type (``MLPPredictor`` saved,
``KANPredictor`` in the template) shows in the per-leaf list.

What is **not** serialised
--------------------------
``simulate_fn``, ``state_to_output``, the ``Dataset``, and the trainable
mask. The framework does not own a builder registry that could re-import
user code by name; loading therefore requires the caller to reconstruct
a same-architecture template (so ``eqx.tree_deserialise_leaves`` can fill
its dynamic leaves) and to re-import their own physics functions. This is
a deliberate boundary — dynamic imports inside a load helper invite
silent failures.

Loss-field handling
-------------------
``OptaxTrainingConfig.loss`` and ``EvosaxTrainingConfig.loss`` accept
either a callable or a string. JSON cannot encode a callable directly,
so ``save_run`` stringifies a callable to ``"{module}.{qualname}"`` and
``load_run`` returns that string verbatim — the field's
``Callable | str`` annotation accepts it, and re-resolution is the
caller's responsibility (typically ``importlib.import_module(module)``
followed by attribute lookup). Keeping that boundary explicit avoids the
security and brittleness costs of dynamic imports inside this module.
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

    ``predictors`` is a ``PyTree[eqx.Module]`` of any container shape
    (tuple / list / dict / NamedTuple / bare Module). The helper
    delegates to ``eqx.tree_serialise_leaves``, which walks the pytree's
    leaves uniformly regardless of container type, producing a flat
    binary stream of ``np.save``-encoded leaves. The caller picks the
    file extension; ``.eqx`` is the convention used by ``save_run``.
    """
    path = Path(path)
    eqx.tree_serialise_leaves(path, predictors)


def load_predictors(path: str | Path, predictors_template: Any) -> Any:
    """Restore a ``predictors`` pytree from ``path`` using ``predictors_template`` as the skeleton.

    ``predictors_template`` must share the saved pytree's container shape
    and per-leaf static configuration (e.g. matching ``MLPPredictor``
    ``in_size`` / ``out_size`` / ``width_size`` / ``depth`` for every
    predictor leaf). Its dynamic leaves are overwritten by the values
    stored in the file; its static fields are kept and provide the
    structure ``equinox`` needs to reconstruct the tree.

    Returns the restored pytree — does **not** mutate
    ``predictors_template``.
    """
    path = Path(path)
    return eqx.tree_deserialise_leaves(path, predictors_template)


def _stringify_loss_field(value: Any) -> Any:
    """Replace a callable with ``"{module}.{qualname}"``; pass-through otherwise.

    Used inside ``_serialise_training_config`` to keep ``loss`` JSON-friendly.
    Strings flow through unchanged; callables (the other half of the
    ``Callable | str`` field annotation) are stringified. Re-import on
    load is the caller's responsibility — see the module docstring's
    rationale for keeping that boundary explicit.
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

    ``dataclasses.asdict`` deep-copies the config into a dict tree and
    tuples become lists (JSON has no tuple). The single field needing
    extra care is ``loss``, which the dataclasses allow to be a callable
    — see ``_stringify_loss_field``.
    """
    raw = dataclasses.asdict(config)
    if "loss" in raw:
        # Read ``loss`` directly from the live config rather than from the
        # already-asdict'd ``raw`` dict: ``dataclasses.asdict`` keeps the
        # callable verbatim, but going through the source object preserves
        # the original identity unambiguously and keeps the stringification
        # logic in one place (``_stringify_loss_field``).
        raw["loss"] = _stringify_loss_field(config.loss)
    return raw


def _resolve_version() -> str:
    """Return the installed ``hybridmodels`` package version, or ``"unknown"``.

    ``importlib.metadata.version`` raises ``PackageNotFoundError`` when the
    package is not installed (e.g. running tests against a checkout that
    has not been ``uv sync``'d) — we swallow it so save never fails purely
    over metadata bookkeeping.
    """
    try:
        return ilm.version("hybridmodels")
    except ilm.PackageNotFoundError:
        return "unknown"


def _is_module(node: Any) -> bool:
    """Predicate used as ``is_leaf`` to stop pytree traversal at ``eqx.Module`` boundaries.

    ``equinox`` does not ship an ``is_module`` helper (the public ``is_…``
    family covers arrays only), so we define a tiny named predicate
    instead of inlining ``isinstance`` everywhere — calling sites read more
    naturally and the intent ("treat each Module as a leaf") stays
    explicit at the call site.
    """
    return isinstance(node, eqx.Module)


def _describe_predictors(predictors: Any) -> dict[str, Any]:
    """Walk ``predictors`` once, returning a JSON-friendly structural hint.

    The hint records two complementary views of the pytree:

    - ``tree_structure``: the ``repr`` of ``jax.tree_util.tree_structure``
      (with the Module-stopped traversal) — a single string capturing the
      container shape (PyTreeDef notation: ``[*, *]`` for a 2-tuple,
      ``{'growth': *, 'nucleation': *}`` for a dict, ``*`` for a bare
      Module).
    - ``leaves``: one ``{path, class}`` entry per ``eqx.Module`` leaf, in
      ``jax.tree_util.tree_flatten_with_path`` traversal order. ``path``
      is the human-readable ``jax.tree_util.keystr`` form (``[0]``,
      ``['growth']``, ``.field``, or ``""`` for a single-Module pytree);
      ``class`` is ``"{module}.{qualname}"`` of the leaf's runtime type.

    Two views because each is useful in a different debugging scenario:
    the structure repr lets a user eyeball the *shape* mismatch
    (tuple-vs-dict, wrong arity); the per-leaf list lets them eyeball the
    *type* mismatch (``MLPPredictor`` saved, ``KANPredictor`` in the
    template). Together they fill the gap left by
    ``eqx.tree_deserialise_leaves`` failing with a generic shape error.
    """
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(predictors, is_leaf=_is_module)
    leaves: list[dict[str, str]] = []
    for path, leaf in leaves_with_paths:
        if not _is_module(leaf):
            # Skip non-Module leaves at the top — typically static
            # passengers if the user nests modules under a static field.
            # The Module-stopped traversal already keeps us at Module
            # boundaries for everything reachable; this guard exists only
            # so a stray scalar in the pytree doesn't crash metadata
            # writing.
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

    Optax returns the raw per-step data loss; Evosax returns best-so-far.
    Both are ``list[float]``, so without this a reloaded run cannot tell
    whether a rise in the series is a real regression or impossible.
    """
    if loss_history is None:
        return None
    if optax_config is not None and evosax_config is None:
        return "per_step_data"
    if evosax_config is not None and optax_config is None:
        return "best_so_far"
    # Both or neither: the caller composed the two runs, so the series
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

        Which training entry point produced ``loss_history`` is recorded
        alongside it as ``loss_history_kind``: ``"per_step_data"`` for
        Optax (raw, can go up) or ``"best_so_far"`` for Evosax (monotone).
        The two series are the same type and mean different things, so a
        saved run that did not say which it held could not be read back
        safely. Inferred from whichever config was passed.

    The directory is created (parents included) if missing. Pre-existing
    files are overwritten. This is a save, not an append.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    save_predictors(directory / _PREDICTORS_FILENAME, predictors)

    metadata: dict[str, Any] = {
        # ``dt.UTC`` keeps the timestamp explicitly tz-aware; the
        # microsecond-precision suffix lands in the ISO string by default.
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

    Lets ``load_run`` reconstruct a frozen config even if the saved
    metadata carries fields the current dataclass version no longer
    declares (e.g. after a config rename). Unknown fields are dropped
    silently — strict-mode behaviour is the caller's responsibility.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _coerce_tuple_fields(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Re-tuple list-shaped values for fields whose type annotation is a tuple.

    JSON has no tuple, so ``dataclasses.asdict`` lists round-trip as lists.
    The training-config dataclasses are ``frozen=True`` and Python doesn't
    coerce in ``__init__``, so ``OptaxTrainingConfig(steps=[5])`` would
    type-check fine but break downstream code that does ``len(steps)``
    (well, ``len`` works on lists, but ``__post_init__`` uses identity-style
    invariants assuming tuples). We re-tuple when the annotation hints
    ``tuple[...]``.
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

    Returns ``None`` when ``raw`` is ``None`` (no config was saved). When
    ``cls`` is ``None`` but ``raw`` is populated, the dict flows through
    unchanged — the caller asked us not to instantiate. Otherwise we
    filter unknown keys (forward compatibility) and re-tuple JSON lists.
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

    Inverse of :func:`save_run`. ``predictors_template`` is required (the
    framework deliberately does not own a builder registry — see the
    module docstring) and must share the saved pytree's container shape
    and per-leaf static configuration. ``optax_cls`` and ``evosax_cls``
    are optional — pass them only when you want the metadata's
    ``optax_config`` / ``evosax_config`` dicts reconstituted into typed
    dataclass instances. When omitted, the raw dicts flow through.

    Loss-field policy
    -----------------
    If a config's ``loss`` field was a callable at save time, it lands here
    as a string ``"{module}.{qualname}"``. The dataclass field type
    (``Callable | str``) accepts the string verbatim — no implicit re-import
    happens. Re-resolution is the caller's responsibility (typically
    ``importlib.import_module(module).qualname``); we keep that boundary
    explicit because dynamic imports inside a load helper invite confusing
    failure modes.

    Returns
    -------
    dict
        Keys: ``predictors`` (``PyTree[eqx.Module]``), ``solver``
        (``SolverConfig``), ``optax_config`` (``OptaxTrainingConfig`` |
        dict | None), ``evosax_config`` (``EvosaxTrainingConfig`` | dict |
        None), ``loss_history`` (``list[float] | None``), ``extras``
        (``dict``).
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
