"""Writing a trained model to disk, and reading it back.

The trainable part of a model is a pytree of ``eqx.Module`` leaves, by
convention a tuple of ``BoundedPredictor``s but any shape is accepted.
This module writes that pytree to disk with the configuration needed to
reconstruct it, and reads it back. Two save/load pairs exist.

``save_predictors`` / ``load_predictors``
    A single binary file, written by ``eqx.tree_serialise_leaves`` and read
    by its inverse. The caller picks the file extension (``save_run`` uses
    ``.eqx``). Loading needs a *template*, a live pytree with the same
    container shape and the same per-leaf static configuration as the one
    that was saved. Its JAX-array leaves are overwritten from the file.

``save_run`` / ``load_run``
    A directory holding a ``predictors.eqx`` binary and a
    ``metadata.json`` describing the run: UTC ISO-8601 timestamp, package
    version, a structural fingerprint of the predictors pytree, the solver
    config dict, optional Optax and Evosax training configs, an optional
    loss history, and a free-form ``extras`` dict. Stable formatting
    (``indent=2, sort_keys=True``) keeps diffs reviewable.

The structural fingerprint exists because
``eqx.tree_deserialise_leaves`` raises a generic shape error when a
template does not match. ``tree_structure`` shows a wrong container shape,
the per-leaf list shows a wrong leaf type (``MLPPredictor`` saved,
``KANPredictor`` in the template).

What is not serialised
----------------------
``simulate_fn``, ``state_to_output``, the ``Dataset``, and the trainable
mask. The framework owns no builder registry that could re-import user
code by name, so loading asks the caller to rebuild a same-architecture
template and re-import their own physics functions. Dynamic imports
inside a load helper fail in confusing ways.

Loss-field handling
-------------------
``OptaxTrainingConfig.loss`` and ``EvosaxTrainingConfig.loss`` accept a
callable or a string. JSON cannot encode a callable, so ``save_run``
writes one as ``"{module}.{qualname}"`` and ``load_run`` returns that
string as it stands, which the ``Callable | str`` annotation accepts.
Turning it back into a function is the caller's job.
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
import optax

from hybridmodels.predictors.base import Predictor
from hybridmodels.solver import SolverConfig
from hybridmodels.training.evosax import EvosaxTrainingConfig
from hybridmodels.training.optax import OptaxTrainingConfig

_PREDICTORS_FILENAME = "predictors.eqx"
_METADATA_FILENAME = "metadata.json"


def save_predictors(path: str | Path, predictors: Any) -> None:
    """Write ``predictors`` to ``path`` via ``eqx.tree_serialise_leaves``.

    ``eqx.tree_serialise_leaves`` walks the leaves the same way whatever
    the container shape, writing a flat binary stream of ``np.save``-encoded
    leaves. The caller picks the file extension; ``save_run`` uses ``.eqx``.
    """
    path = Path(path)
    eqx.tree_serialise_leaves(path, predictors)


def load_predictors(path: str | Path, predictors_template: Any) -> Any:
    """Restore a ``predictors`` pytree from ``path`` using ``predictors_template`` as the skeleton.

    ``predictors_template`` must share the saved pytree's container shape
    and per-leaf static configuration, such as the same ``in_size`` and
    ``depth`` on every ``MLPPredictor`` leaf. The file's values overwrite
    the template's array leaves; its static fields supply the structure
    ``equinox`` needs. The template is not mutated.
    """
    path = Path(path)
    return eqx.tree_deserialise_leaves(path, predictors_template)


def _stringify_loss_field(value: Any) -> Any:
    """Replace a callable with ``"{module}.{qualname}"``; pass-through otherwise.

    Keeps ``loss`` JSON-friendly. Strings pass through unchanged;
    re-importing on load is the caller's job (module docstring).
    """
    if isinstance(value, str):
        return value
    if callable(value):
        return _callable_path(value)
    return value


def _callable_path(value: Any) -> str:
    """Return the import-style identity of a callable for metadata."""
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", ""))
    return f"{module}.{qualname}" if module else qualname


def _json_safe(value: Any) -> Any:
    """Convert metadata values into JSON-safe values without losing identity hints."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return {"__callable__": _callable_path(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return {"__opaque__": _callable_path(type(value)), "repr": repr(value)}


def _json_safe_optimizer(value: Any) -> Any:
    """Preserve raw Optax transformations as one opaque metadata value."""
    if isinstance(value, optax.GradientTransformation):
        return {
            "__opaque__": _callable_path(type(value)),
            "repr": repr(value),
        }
    return _json_safe(value)


def _serialise_training_config(config: Any) -> dict[str, Any]:
    """Return a JSON-encodable view of an Optax/Evosax training config.

    ``dataclasses.asdict`` deep-copies the config, turning tuples into
    lists since JSON has no tuple. Callable and opaque runtime fields are
    represented by descriptive markers; ``loss`` keeps its historical
    import-style string representation. See ``_json_safe``.
    """
    raw_unprocessed = dataclasses.asdict(config)
    if hasattr(config, "optimizer"):
        raw_unprocessed["optimizer"] = [
            _json_safe_optimizer(spec) for spec in config.optimizer
        ]
    raw = _json_safe(raw_unprocessed)
    if "loss" in raw:
        # Read ``loss`` from the live config, not the asdict'd copy, so the
        # original identity is unambiguous.
        raw["loss"] = _stringify_loss_field(config.loss)
    return raw


def _resolve_version() -> str:
    """Return the installed ``jaxhybridmodels`` version, or ``"unknown"``.

    ``importlib.metadata.version`` raises ``PackageNotFoundError`` when the
    package is not installed, as in a checkout that has not been ``uv
    sync``'d. Swallowed, so a save never fails over bookkeeping alone.
    """
    try:
        return ilm.version("jaxhybridmodels")
    except ilm.PackageNotFoundError:
        return "unknown"


def _is_module(node: Any) -> bool:
    """Predicate used as ``is_leaf`` to stop pytree traversal at ``eqx.Module`` boundaries.

    ``equinox`` ships no ``is_module`` helper; its public ``is_…`` family
    covers arrays only.
    """
    return isinstance(node, eqx.Module)


def _describe_predictors(predictors: Any) -> dict[str, Any]:
    """Walk ``predictors`` once, returning a JSON-friendly structural hint.

    Two views, each catching a different mismatch that
    ``eqx.tree_deserialise_leaves`` would report only as a generic shape
    error.

    - ``tree_structure``: ``repr`` of the Module-stopped
      ``jax.tree_util.tree_structure``, so ``[*, *]`` for a 2-tuple,
      ``{'growth': *, 'nucleation': *}`` for a dict. Catches a wrong
      container shape or arity.
    - ``leaves``: one ``{path, class}`` entry per ``eqx.Module`` leaf in
      flatten order, ``path`` in ``keystr`` form and ``class`` as
      ``"{module}.{qualname}"``. Catches a wrong leaf type.
    """
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(predictors, is_leaf=_is_module)
    leaves: list[dict[str, str]] = []
    for path, leaf in leaves_with_paths:
        if not _is_module(leaf):
            # Non-Module leaves are static passengers, usually from modules
            # nested under a static field. Skipped so a stray scalar cannot
            # crash the metadata write.
            continue
        cls = type(leaf)
        leaves.append(
            {
                "path": jtu.keystr(path),
                "class": f"{cls.__module__}.{cls.__qualname__}",
            }
        )
    static: dict[str, Any] = {}

    def collect_static(node: Any, path: str) -> None:
        if isinstance(node, eqx.Module):
            # Capture static semantics on user-defined Predictor modules too.
            # Dependency internals such as eqx.nn.MLP remain intentionally
            # opaque because their enclosing shipped predictor owns the
            # serialisable architecture metadata.
            if not isinstance(node, Predictor) and not type(node).__module__.startswith(
                "hybridmodels"
            ):
                return
            for field in dataclasses.fields(type(node)):
                value = getattr(node, field.name)
                field_path = f"{path}.{field.name}" if path else field.name
                if field.metadata.get("static", False):
                    static[field_path] = _json_safe(value)
                else:
                    collect_static(value, field_path)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                collect_static(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (tuple, list)):
            for index, value in enumerate(node):
                collect_static(value, f"{path}.{index}" if path else str(index))

    collect_static(predictors, "")
    return {
        "tree_structure": repr(treedef),
        "leaves": leaves,
        "static": static,
    }


def _loss_history_kind(
    optax_config: Any, evosax_config: Any, loss_history: list[float] | None
) -> str | None:
    """Name the semantics of ``loss_history`` from whichever config is present.

    Optax returns the raw data loss at each step, which can go up; Evosax
    returns the best value so far, which cannot. Both are ``list[float]``,
    so without this label a reloaded run cannot read a rise in the series.
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

        ``loss_history_kind``, inferred from whichever config was passed,
        records which entry point produced ``loss_history``:
        ``"per_step_data"`` for Optax (can go up) or ``"best_so_far"`` for
        Evosax (monotone). The two series share a type and mean different
        things.

    The directory is created, parents included. Existing files are
    overwritten: this saves rather than appends.
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
    """Keep only declared fields, refusing metadata the current config cannot represent."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"Saved {cls.__name__} contains unknown configuration fields: {unknown}"
        )
    return {k: v for k, v in raw.items() if k in valid}


def _contains_serialised_callable(value: Any) -> bool:
    """Whether metadata contains a callable or opaque runtime object marker."""
    if isinstance(value, dict):
        if "__callable__" in value or "__opaque__" in value:
            return True
        return any(_contains_serialised_callable(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_serialised_callable(item) for item in value)
    return False


def _coerce_tuple_fields(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Re-tuple list-shaped values for fields whose type annotation is a tuple.

    JSON has no tuple, and the config dataclasses do not coerce in
    ``__init__``, so ``OptaxTrainingConfig(steps=[5])`` constructs happily
    and then differs from a hand-written config, unhashable included.
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

    ``None`` when no config was saved. A populated ``raw`` with no ``cls``
    passes through unchanged. Otherwise unknown keys are rejected and JSON
    lists are re-tupled, so loading cannot silently change the configuration.
    """
    if raw is None:
        return None
    if cls is None:
        return raw
    if _contains_serialised_callable(raw):
        raise ValueError(
            "Saved training config contains callable or opaque fields that cannot be "
            "reconstructed automatically; load it without a config class and rebind "
            "those fields explicitly."
        )
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

    Inverse of :func:`save_run`. ``predictors_template`` is required, since
    the framework owns no builder registry, and must share the saved
    pytree's container shape and per-leaf static configuration.

    Pass ``optax_cls`` / ``evosax_cls`` to get the saved config dicts
    rebuilt as typed dataclass instances; leave them out and the raw dicts
    come back.

    A ``loss`` field that was a callable at save time arrives as the string
    ``"{module}.{qualname}"`` and is returned as it stands. Turning it back
    into a function is the caller's job, usually
    ``importlib.import_module(module).qualname``.

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

    saved_version = metadata.get("version")
    current_version = _resolve_version()
    if (
        saved_version is not None
        and saved_version != "unknown"
        and current_version != "unknown"
        and saved_version != current_version
    ):
        raise ValueError(
            f"Saved run targets hybridmodels version {saved_version!r}, "
            f"but the installed package is {current_version!r}."
        )

    saved_description = metadata.get("predictors", {})
    actual_description = _describe_predictors(predictors_template)
    for field in ("tree_structure", "leaves", "static"):
        if field in saved_description and saved_description[field] != actual_description[field]:
            raise ValueError(
                "predictors_template does not match saved predictor metadata for "
                f"{field}; rebuild the template with the original static configuration."
            )

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
