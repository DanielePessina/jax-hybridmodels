"""Writing a trained model to disk, and reading it back.

``save`` / ``load``
    The workflow pair. ``save(path, run)`` takes a ``HybridModel`` or a
    ``FitResult`` and writes a directory holding a ``predictors.eqx``
    binary and a ``metadata.json``: UTC ISO-8601 timestamp, package
    version, a structural fingerprint of the predictors pytree, the output
    channel names, the solver config, the fit history (for a
    ``FitResult``), and the optional training config, ``Objective`` and
    free-form ``extras``. Stable formatting (``indent=2, sort_keys=True``)
    keeps diffs reviewable. ``load(path, like=model)`` returns a
    :class:`~jaxhybridmodels.results.SavedRun` whose ``model`` is ``like``
    with the saved predictors. The solver is recorded for reference and
    checked on load, but ``like.solver`` is kept: the saved form holds
    only class names, not constructor arguments.

``save_predictors`` / ``load_predictors``
    The predictors pytree alone, as one binary file written by
    ``eqx.tree_serialise_leaves`` and read by its inverse, for code that
    works with a raw predictors pytree (the custom-loop layer). ``like`` is
    a live pytree with the same container shape and per-leaf static
    configuration as the one saved; its array leaves are overwritten from
    the file.

The structural fingerprint exists because
``eqx.tree_deserialise_leaves`` raises a generic shape error when a
template does not match. ``tree_structure`` shows a wrong container shape,
the per-leaf list shows a wrong leaf type (``MLPPredictor`` saved,
``KANPredictor`` in the template), and ``static`` a changed static field.

What is not serialised
----------------------
``simulate_fn``, ``state_to_output``, the ``Dataset``, and the trainable
mask. The framework owns no builder registry that could re-import user
code by name, so ``load`` takes them from the ``like`` model, which the
caller builds with the same architecture and their own physics functions.
Dynamic imports inside a load helper fail in confusing ways.

Callables
---------
JSON cannot encode a callable. A callable anywhere in a training config or
an ``Objective`` (a custom ``loss``, a ``penalty_fn``, an optimizer factory,
an lr schedule) is written as ``{"__callable__": "{module}.{qualname}"}``,
and a raw ``optax.GradientTransformation`` as an ``"__opaque__"`` marker.
``load`` returns such a section as the raw dict rather than a typed
object: turning the name back into a function is the caller's job.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.metadata as ilm
import json
import math
import warnings
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
import optax

from jaxhybridmodels.model import HybridModel
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.predictors.base import Predictor
from jaxhybridmodels.results import FitHistory, FitResult, SavedRun
from jaxhybridmodels.solver import SolverConfig
from jaxhybridmodels.training.evosax import EvosaxTrainingConfig
from jaxhybridmodels.training.optax import OptaxTrainingConfig, TrainingPhase

_PREDICTORS_FILENAME = "predictors.eqx"
_METADATA_FILENAME = "metadata.json"


def save_predictors(path: str | Path, predictors: Any) -> None:
    """Write ``predictors`` to ``path`` via ``eqx.tree_serialise_leaves``.

    ``eqx.tree_serialise_leaves`` walks the leaves the same way whatever
    the container shape, writing a flat binary stream of ``np.save``-encoded
    leaves. The caller picks the file extension; ``save`` uses ``.eqx``.
    """
    path = Path(path)
    eqx.tree_serialise_leaves(path, predictors)


def load_predictors(path: str | Path, *, like: Any) -> Any:
    """Restore a ``predictors`` pytree from ``path``, using ``like`` as the skeleton.

    ``like`` must share the saved pytree's container shape and per-leaf
    static configuration, such as the same ``in_size`` and ``depth`` on
    every ``MLPPredictor`` leaf. The file's values overwrite its array
    leaves; its static fields supply the structure ``equinox`` needs.
    ``like`` is not mutated.
    """
    path = Path(path)
    return eqx.tree_deserialise_leaves(path, like)


def _callable_path(value: Any) -> str:
    """Return the import-style identity of a callable for metadata."""
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", ""))
    return f"{module}.{qualname}" if module else qualname


def _json_safe(value: Any) -> Any:
    """Convert metadata values into JSON-safe values without losing identity hints."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": str(value)}
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
    represented by descriptive markers; see ``_json_safe``.
    """
    raw_unprocessed = dataclasses.asdict(config)
    for phase_raw, phase in zip(
        raw_unprocessed.get("phases", ()), getattr(config, "phases", ()), strict=True
    ):
        phase_raw["optimizer"] = _json_safe_optimizer(phase.optimizer)
    return _json_safe(raw_unprocessed)


def _serialise_objective(objective: Objective) -> dict[str, Any]:
    """Return a JSON-encodable view of an ``Objective``.

    Each ``penalty_points`` array becomes ``{"shape": [...], "data": [...]}``.
    The shape is kept because an empty ``[0, n_inputs]`` grid flattens to
    ``[]`` and would reload as shape ``(0,)``. Callables become
    ``__callable__`` markers.
    """
    raw: dict[str, Any] = {
        f.name: getattr(objective, f.name) for f in dataclasses.fields(objective)
    }
    if objective.penalty_points is not None:
        raw["penalty_points"] = {
            path: {"shape": list(np.shape(points)), "data": np.asarray(points).tolist()}
            for path, points in objective.penalty_points.items()
        }
    return _json_safe(raw)


def _decode_penalty_points(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn saved ``penalty_points`` entries back into arrays of their saved shape.

    A bare nested list (the format before shapes were stored) is read as is.
    """
    points = raw.get("penalty_points")
    if not isinstance(points, dict):
        return raw
    decoded = {}
    for path, entry in points.items():
        if isinstance(entry, dict) and "shape" in entry:
            decoded[path] = np.asarray(entry["data"], dtype=float).reshape(entry["shape"])
        else:
            decoded[path] = np.asarray(entry, dtype=float)
    return {**raw, "penalty_points": decoded}


def _build_objective(raw: dict[str, Any] | None) -> Objective | dict[str, Any] | None:
    """Rebuild an ``Objective`` from its dict; a dict naming a callable comes back raw.

    Penalty grids are decoded to arrays of their saved shape either way.
    """
    if raw is None:
        return None
    raw = _decode_penalty_points(raw)
    if _contains_serialised_callable(raw):
        return raw
    return Objective(**_filter_dataclass_kwargs(Objective, raw))


def _resolve_version() -> str:
    """Return the installed ``jax-hybridmodels`` version, or ``"unknown"``.

    ``importlib.metadata.version`` raises ``PackageNotFoundError`` when the
    package is not installed, as in a checkout that has not been ``uv
    sync``'d. Swallowed, so a save never fails over bookkeeping alone.
    """
    try:
        return ilm.version("jax-hybridmodels")
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
                "jaxhybridmodels"
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


# The config classes ``save`` accepts, by the tag written to metadata.
_CONFIG_TYPES: dict[str, type] = {
    "optax": OptaxTrainingConfig,
    "evosax": EvosaxTrainingConfig,
}


def _serialise_config(config: Any) -> dict[str, Any]:
    """Tag a training config with its kind so ``load`` knows which class to rebuild."""
    for kind, cls in _CONFIG_TYPES.items():
        if isinstance(config, cls):
            return {"kind": kind, "fields": _serialise_training_config(config)}
    raise TypeError(
        "save: config must be an OptaxTrainingConfig or EvosaxTrainingConfig; "
        f"got {type(config).__name__}"
    )


def _serialise_history(history: FitHistory) -> dict[str, Any]:
    """Write each ``FitHistory`` series as a JSON list, NaN and inf as markers."""
    return {
        f.name: _json_safe([float(v) for v in getattr(history, f.name)])
        for f in dataclasses.fields(history)
    }


def _decode_float(value: Any) -> float:
    """Inverse of ``_json_safe`` on one float: ``{"__float__": "nan"}`` back to NaN."""
    if isinstance(value, dict) and "__float__" in value:
        return float(value["__float__"])
    return float(value)


def _build_history(raw: dict[str, Any] | None) -> FitHistory:
    """Rebuild a ``FitHistory`` from its saved series; an empty one when none was saved."""
    if raw is None:
        return FitHistory()
    return FitHistory(
        loss=tuple(_decode_float(v) for v in raw.get("loss", ())),
        penalty=tuple(_decode_float(v) for v in raw.get("penalty", ())),
        phase=tuple(int(v) for v in raw.get("phase", ())),
        validation_loss=tuple(_decode_float(v) for v in raw.get("validation_loss", ())),
    )


def save(
    path: str | Path,
    run: HybridModel | FitResult,
    *,
    config: OptaxTrainingConfig | EvosaxTrainingConfig | None = None,
    objective: Objective | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Write a model, or a fit with its history, to the directory ``path``.

    ``run`` is a ``HybridModel`` or the ``FitResult`` from ``fit``; a
    ``FitResult`` also saves its ``history``, ``best_step`` and
    ``stop_reason``. ``config`` and ``objective`` record how the model was
    trained, and ``extras`` is any JSON-encodable dict of your own.
    Read it back with :func:`load`.

    Layout written
    --------------
    ``path/predictors.eqx``
        Binary leaves of the predictors pytree (``save_predictors`` output).
    ``path/metadata.json``
        ``timestamp`` (UTC, ISO 8601), ``version``, ``predictors`` (the
        structural fingerprint ``load`` checks ``like`` against),
        ``output_channel_names``, ``solver`` (``SolverConfig.to_dict()``),
        ``config`` (``{"kind": "optax" | "evosax", "fields": {...}}``),
        ``objective``, ``history`` (the four ``FitHistory`` series),
        ``best_step``, ``stop_reason`` and ``extras``. Callables are
        described, not pickled (see the module docstring).

    The directory is created, parents included. Existing files are
    overwritten.
    """
    if isinstance(run, FitResult):
        model, result = run.model, run
    elif isinstance(run, HybridModel):
        model, result = run, None
    else:
        raise TypeError(f"save: run must be a HybridModel or FitResult; got {type(run).__name__}")
    # The metadata is built before anything is written, so a value that
    # cannot be encoded fails without leaving a half-written directory.
    metadata: dict[str, Any] = {
        # ``dt.UTC`` keeps the timestamp explicitly tz-aware, and the
        # microsecond suffix lands in the ISO string by default.
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "version": _resolve_version(),
        "predictors": _describe_predictors(model.predictors),
        "output_channel_names": list(model.output_channel_names),
        "solver": _solver_metadata(model.solver),
        "config": _serialise_config(config) if config is not None else None,
        "objective": _serialise_objective(objective) if objective is not None else None,
        "history": _serialise_history(result.history) if result is not None else None,
        "best_step": result.best_step if result is not None else None,
        "stop_reason": result.stop_reason if result is not None else None,
        "extras": dict(extras) if extras is not None else {},
    }
    encoded = json.dumps(metadata, indent=2, sort_keys=True)

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    save_predictors(directory / _PREDICTORS_FILENAME, model.predictors)
    (directory / _METADATA_FILENAME).write_text(encoded)


def _solver_metadata(solver: SolverConfig) -> dict[str, Any]:
    """``solver.to_dict()``, or its ``repr`` when a class is unregistered.

    The record is informational (``load`` keeps ``like.solver``), so a
    solver outside the registries, such as ``diffrax.Kvaerno5``, still
    saves instead of failing the whole run.
    """
    try:
        return solver.to_dict()
    except ValueError:
        return {"unregistered": repr(solver)}


def _filter_dataclass_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only declared fields, refusing metadata the current config cannot represent."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"Saved {cls.__name__} contains unknown configuration fields: {unknown}")
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


def _build_training_config(raw: dict[str, Any] | None) -> Any:
    """Rebuild a training config from its tagged dict.

    ``None`` when no config was saved. A config naming a callable comes
    back as its raw ``fields`` dict. Otherwise unknown keys are rejected
    and JSON lists are re-tupled, so loading cannot silently change the
    configuration.
    """
    if raw is None:
        return None
    fields = raw["fields"]
    if _contains_serialised_callable(fields):
        return fields
    cls = _CONFIG_TYPES[raw["kind"]]
    kwargs = _filter_dataclass_kwargs(cls, fields)
    kwargs = _coerce_tuple_fields(cls, kwargs)
    if "phases" in kwargs:
        kwargs["phases"] = tuple(TrainingPhase(**phase) for phase in kwargs["phases"])
    return cls(**kwargs)


def load(path: str | Path, *, like: HybridModel) -> SavedRun:
    """Read back a run written by :func:`save`.

    ``like`` is a model built the same way as the saved one: same
    predictor architecture and static configuration, same
    ``output_channel_names``. It supplies ``simulate_fn`` and
    ``state_to_output``, which are not saved, and the solver. The returned
    ``SavedRun.model`` is ``like`` with the saved predictors. The saved
    solver record holds class names only (constructor arguments such as
    ``RecursiveCheckpointAdjoint(checkpoints=7)`` are not kept), so
    ``like.solver`` is used as is, with a ``RuntimeWarning`` when it
    differs from the record.

    Raises ``ValueError`` when the saved package version differs from the
    installed one, when ``like`` does not match the saved predictors or
    channel names, or when a saved config has a field the current class
    does not know.
    """
    directory = Path(path)
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
            f"Saved run targets jax-hybridmodels version {saved_version!r}, "
            f"but the installed package is {current_version!r}."
        )

    saved_description = metadata.get("predictors", {})
    actual_description = _describe_predictors(like.predictors)
    for field in ("tree_structure", "leaves", "static"):
        if field in saved_description and saved_description[field] != actual_description[field]:
            raise ValueError(
                f"load: like.predictors does not match the saved predictors in {field}; "
                "rebuild it with the original static configuration."
            )
    saved_names = tuple(metadata["output_channel_names"])
    if saved_names != like.output_channel_names:
        raise ValueError(
            f"load: saved output_channel_names {saved_names} differ from "
            f"like.output_channel_names {like.output_channel_names}"
        )

    saved_solver = metadata.get("solver")
    if saved_solver is not None and saved_solver != _solver_metadata(like.solver):
        warnings.warn(
            f"load: the run was saved with solver {saved_solver}, but like.solver is "
            f"{_solver_metadata(like.solver)}; keeping like.solver",
            RuntimeWarning,
            stacklevel=2,
        )

    model = like._with_predictors(
        load_predictors(directory / _PREDICTORS_FILENAME, like=like.predictors)
    )
    return SavedRun(
        model=model,
        history=_build_history(metadata.get("history")),
        best_step=metadata.get("best_step"),
        stop_reason=metadata.get("stop_reason"),
        config=_build_training_config(metadata.get("config")),
        objective=_build_objective(metadata.get("objective")),
        extras=metadata.get("extras", {}),
    )


__all__ = [
    "load",
    "load_predictors",
    "save",
    "save_predictors",
]
