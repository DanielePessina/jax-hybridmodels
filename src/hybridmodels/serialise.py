"""Save / load helpers for trained predictors and runs (SPEC §5.11 / R-A5 / R-S2).

Phase 16 of the build plan — the *last shipped* module per CONTEXT.md
"Serialisation". The R-A5 round-trip cleanliness has been a hard design
constraint since day one; every ``Predictor`` was built so its dynamic
leaves are JAX arrays and its static fields are JSON-encodable. This module
ships the user-facing helpers that compose those guarantees into a
directory-shaped artifact.

Public surface
--------------
``save_predictor`` / ``load_predictor``
    Single-file binary round-trip via ``eqx.tree_serialise_leaves`` /
    ``eqx.tree_deserialise_leaves``. The caller picks the file extension
    (we conventionally use ``.eqx``) and supplies a *template* predictor
    matching the original's class plus static configuration on load.

``save_run`` / ``load_run``
    Directory-shaped artifact — ``predictor.eqx`` plus a JSON
    ``metadata.json`` carrying timestamp, package version, the predictor's
    fully-qualified class path, the solver dict, optional optax / evosax
    training configs (as ``dataclasses.asdict``), an optional loss history,
    and a free-form ``extras`` dict the user owns. The JSON is written with
    ``indent=2, sort_keys=True`` so diffs are stable.

What is *not* serialised
------------------------
``simulate_fn``, ``state_to_output``, the ``Dataset``, and the trainable
mask. CONTEXT.md spells this out: the framework re-imports user code, and
no builder registry is wired in v1 (deferred until friction proves real;
see SPEC §2.3). Loading a predictor therefore requires the caller to
reconstruct a same-architecture template — exactly the contract of
``eqx.tree_deserialise_leaves`` — and re-import their own physics code.

Loss-callable handling
----------------------
``OptaxTrainingConfig.loss`` and ``EvosaxTrainingConfig.loss`` accept either
a callable or a string. JSON cannot encode a callable directly; instead of
attempting an importlib-based round-trip, ``save_run`` stringifies a
callable to ``"{module}.{qualname}"``. ``load_run`` returns that string
verbatim — the field type ``Callable | str`` already accepts it, and the
caller is expected to re-resolve the callable themselves if they need one
(typically by importing the module and looking up the attribute, or by
threading a builder dict). This avoids the security and brittleness costs
of dynamic imports inside a load helper.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.metadata as ilm
import json
from pathlib import Path
from typing import Any, cast

import equinox as eqx

from hybridmodels.predictors.base import Predictor
from hybridmodels.solver import SolverConfig
from hybridmodels.training.evosax import EvosaxTrainingConfig
from hybridmodels.training.optax import OptaxTrainingConfig

_PREDICTOR_FILENAME = "predictor.eqx"
_METADATA_FILENAME = "metadata.json"


def save_predictor(path: str | Path, predictor: Predictor) -> None:
    """Write ``predictor`` to ``path`` via ``eqx.tree_serialise_leaves``.

    The file format is whatever ``eqx.tree_serialise_leaves`` writes — a
    binary stream of ``np.save``-encoded leaves driven by the predictor's
    pytree structure. The caller chooses the extension; ``.eqx`` is the
    project convention (see ``save_run``).
    """
    path = Path(path)
    eqx.tree_serialise_leaves(path, predictor)


def load_predictor(path: str | Path, template: Predictor) -> Predictor:
    """Restore a predictor from ``path`` using ``template`` as the pytree skeleton.

    ``template`` must be an instance of the same concrete predictor class
    as the saved one, with matching static configuration (e.g. matching
    ``MLPPredictor`` ``in_size`` / ``out_size`` / ``width_size`` / ``depth``).
    Its dynamic leaves are overwritten by the values stored in the file;
    its static fields are kept and provide the structure ``equinox`` needs
    to reconstruct the tree.

    Returns the restored predictor — does **not** mutate ``template``.
    """
    path = Path(path)
    return cast(Predictor, eqx.tree_deserialise_leaves(path, template))


def _stringify_loss_field(value: Any) -> Any:
    """Replace a callable with ``"{module}.{qualname}"``; pass-through otherwise.

    Used inside ``_serialise_training_config`` to keep ``loss`` JSON-friendly.
    Strings flow through unchanged; everything else (the spec-allowed
    ``Callable | str`` union) gets stringified. We deliberately do **not**
    attempt to re-import on load — that decision is the caller's per the
    module docstring rationale.
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


def save_run(
    directory: str | Path,
    *,
    predictor: Predictor,
    solver: SolverConfig,
    optax_config: OptaxTrainingConfig | None = None,
    evosax_config: EvosaxTrainingConfig | None = None,
    loss_history: list[float] | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Persist a complete training run to ``directory``.

    Layout written
    --------------
    ``directory/predictor.eqx``
        Binary predictor leaves (``save_predictor`` output).
    ``directory/metadata.json``
        JSON dict with ``timestamp`` (UTC, ISO 8601, microsecond
        precision), ``version`` (``importlib.metadata.version``-resolved),
        ``predictor_class_path`` (``"{module}.{qualname}"``), ``solver``
        (``solver.to_dict()``), ``optax_config`` /
        ``evosax_config`` (``dataclasses.asdict`` with stringified ``loss``
        — see module docstring), ``loss_history``, and ``extras``.

    The directory is created (parents included) if missing. Pre-existing
    files are overwritten — this is a save, not an append.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    save_predictor(directory / _PREDICTOR_FILENAME, predictor)

    predictor_cls = type(predictor)
    metadata: dict[str, Any] = {
        # ``dt.UTC`` keeps the timestamp explicitly tz-aware; the
        # microsecond-precision suffix lands in the ISO string by default.
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "version": _resolve_version(),
        "predictor_class_path": (
            f"{predictor_cls.__module__}.{predictor_cls.__qualname__}"
        ),
        "solver": solver.to_dict(),
        "optax_config": (
            _serialise_training_config(optax_config) if optax_config is not None else None
        ),
        "evosax_config": (
            _serialise_training_config(evosax_config) if evosax_config is not None else None
        ),
        "loss_history": list(loss_history) if loss_history is not None else None,
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


def _build_training_config(
    cls: type | None, raw: dict[str, Any] | None
) -> Any:
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
    predictor_template: Predictor,
    optax_cls: type[OptaxTrainingConfig] | None = None,
    evosax_cls: type[EvosaxTrainingConfig] | None = None,
) -> dict[str, Any]:
    """Reconstruct a run from ``directory``.

    Inverse of :func:`save_run`. ``predictor_template`` is required (per the
    no-builder-registry policy in CONTEXT.md "Serialisation"); ``optax_cls``
    and ``evosax_cls`` are optional — pass them only when you want the
    metadata's ``optax_config`` / ``evosax_config`` dicts reconstituted into
    typed dataclass instances. When omitted, the raw dicts flow through.

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
        Keys: ``predictor`` (``Predictor``), ``solver`` (``SolverConfig``),
        ``optax_config`` (``OptaxTrainingConfig`` | dict | None),
        ``evosax_config`` (``EvosaxTrainingConfig`` | dict | None),
        ``loss_history`` (``list[float] | None``), ``extras`` (``dict``).
    """
    directory = Path(directory)
    with (directory / _METADATA_FILENAME).open() as f:
        metadata: dict[str, Any] = json.load(f)

    solver = SolverConfig.from_dict(metadata["solver"])
    optax_config = _build_training_config(optax_cls, metadata.get("optax_config"))
    evosax_config = _build_training_config(evosax_cls, metadata.get("evosax_config"))
    predictor = load_predictor(directory / _PREDICTOR_FILENAME, predictor_template)

    return {
        "predictor": predictor,
        "solver": solver,
        "optax_config": optax_config,
        "evosax_config": evosax_config,
        "loss_history": metadata.get("loss_history"),
        "extras": metadata.get("extras", {}),
    }


__all__ = [
    "load_predictor",
    "load_run",
    "save_predictor",
    "save_run",
]
