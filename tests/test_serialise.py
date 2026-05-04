"""Tests for ``hybridmodels.serialise``.

Pins the four public helpers (``save_predictors``,
``load_predictors``, ``save_run``, ``load_run``) and the on-disk
metadata contract.

Coverage strategy
-----------------
- Round-trip is parametrised over the four pytree shapes the runtime
  contract permits: a bare ``BoundedPredictor`` (one-leaf pytree), the
  conventional ``(BP,)`` one-tuple, a multi-rate
  ``(growth_BP, nucleation_BP)`` two-tuple, and a ``dict[str, BP]``
  mapping. Each shape goes through the public ``save_predictors`` /
  ``load_predictors`` entry points so the helper plumbing (path
  coercion, file mode) is covered alongside
  ``eqx.tree_serialise_leaves`` itself.
- ``save_run`` / ``load_run`` get one minimal happy path plus targeted
  variants for each load-bearing branch: Optax + Evosax both
  populated, optional ``loss_history`` / ``extras`` propagation,
  missing builder classes, directory creation, overwrite semantics,
  loss-callable stringification, and the per-leaf ``predictors``
  description fields.
- The KAN-specific forward-pass equality check guards the
  static/dynamic split documented in ``predictors/kan.py`` — the
  highest-risk predictor for round-trip drift because of the
  ``jaxkan`` / ``nnx.Param`` wiring.
"""

# ruff: noqa: F722

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from jaxtyping import Array

from hybridmodels.losses import masked_mse
from hybridmodels.predictors import (
    BoundedPredictor,
    BoundScaler,
    KANPredictor,
    MLPPredictor,
)
from hybridmodels.serialise import (
    load_predictors,
    load_run,
    save_predictors,
    save_run,
)
from hybridmodels.solver import SolverConfig
from hybridmodels.training.evosax import EvosaxTrainingConfig
from hybridmodels.training.optax import OptaxTrainingConfig

# -- predictor + pytree factories ------------------------------------------


def _mlp(key: Array | None = None) -> MLPPredictor:
    return MLPPredictor(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        activation_name="tanh",
        key=jr.PRNGKey(0) if key is None else key,
    )


def _kan(key: Array | None = None) -> KANPredictor:
    return KANPredictor(
        in_size=2,
        out_size=1,
        hidden_widths=(6,),
        grid_size=4,
        basis="spline",
        key=jr.PRNGKey(0) if key is None else key,
    )


def _bounded(key: Array | None = None) -> BoundedPredictor:
    inner = MLPPredictor(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        activation_name="tanh",
        key=jr.PRNGKey(0) if key is None else key,
    )
    return BoundedPredictor(
        input_keys=("a", "b"),
        in_scaler=BoundScaler(
            bounds=((0.0, 1.0), (0.0, 2.0)),
            transform="sigmoid",
            temperature=1.0,
        ),
        inner=inner,
        out_scaler=BoundScaler(
            bounds=((0.0, 5.0), (-1.0, 1.0)),
            transform="sigmoid",
        ),
    )


def _bare_predictor() -> BoundedPredictor:
    """One-leaf pytree — a single ``BoundedPredictor`` in isolation."""
    return _bounded(jr.PRNGKey(1))


def _one_tuple() -> tuple[BoundedPredictor, ...]:
    """Canonical convention for the single-predictor case: ``(BP,)``."""
    return (_bounded(jr.PRNGKey(2)),)


def _two_tuple() -> tuple[BoundedPredictor, BoundedPredictor]:
    """Multi-rate convention: ``(growth, nucleation)`` unpacked in vector field."""
    return (_bounded(jr.PRNGKey(3)), _bounded(jr.PRNGKey(4)))


def _dict_shape() -> dict[str, BoundedPredictor]:
    """Runtime-permissive alternative shape — a named dict of predictors."""
    return {
        "growth": _bounded(jr.PRNGKey(5)),
        "nucleation": _bounded(jr.PRNGKey(6)),
    }


PYTREE_FACTORIES: list[tuple[str, Callable[[], Any]]] = [
    ("bare_module", _bare_predictor),
    ("one_tuple", _one_tuple),
    ("two_tuple", _two_tuple),
    ("dict_shape", _dict_shape),
]


def _template_for(predictors: Any) -> Any:
    """Return a same-shape template pytree with different-weights leaves.

    The deserialise template must match container shape and per-leaf static
    config but should *not* share any inexact-array leaf values with the
    saved pytree — otherwise the round-trip equality check would be
    vacuous (template leaves would already equal saved leaves).
    """
    fresh = jr.PRNGKey(99)
    template: Any
    if isinstance(predictors, BoundedPredictor):
        template = _bounded(fresh)
    elif isinstance(predictors, tuple):
        template = tuple(_bounded(jr.fold_in(fresh, i)) for i in range(len(predictors)))
    elif isinstance(predictors, dict):
        template = {k: _bounded(jr.fold_in(fresh, hash(k) & 0xFFFF)) for k in predictors}
    else:
        raise AssertionError(f"no template factory for {type(predictors).__name__}")
    return template


# -- shared run fixtures ---------------------------------------------------


def _solver() -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=(1e-5, 1e-5),
        max_steps=10_000,
        dt0=None,
    )


def _optax_config(loss: Callable[..., Array] | str = "mse") -> OptaxTrainingConfig:
    return OptaxTrainingConfig(
        steps=(5,),
        lr=(1e-3,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss=loss,
        verbose=False,
    )


def _evosax_config() -> EvosaxTrainingConfig:
    return EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=8,
        num_generations=3,
        init="warm",
        sigma_init=0.1,
        loss="mse",
        verbose=False,
    )


def _assert_pytree_leaves_equal(a: Any, b: Any) -> None:
    leaves_a = jtu.tree_leaves(a)
    leaves_b = jtu.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for left, right in zip(leaves_a, leaves_b, strict=True):
        if hasattr(left, "shape"):
            assert np.array_equal(np.asarray(left), np.asarray(right))
        else:
            assert left == right


# -- save_predictors / load_predictors -------------------------------------


@pytest.mark.parametrize(
    "factory",
    [f for _, f in PYTREE_FACTORIES],
    ids=[name for name, _ in PYTREE_FACTORIES],
)
def test_save_load_predictors_round_trip(factory: Callable[[], Any], tmp_path: Path) -> None:
    predictors = factory()
    path = tmp_path / "predictors.eqx"
    save_predictors(path, predictors)
    assert path.exists()

    template = _template_for(predictors)
    restored = load_predictors(path, template)
    _assert_pytree_leaves_equal(predictors, restored)


def test_save_load_predictors_accepts_str_path(tmp_path: Path) -> None:
    predictors = _one_tuple()
    path = str(tmp_path / "predictors.eqx")
    save_predictors(path, predictors)
    restored = load_predictors(path, _template_for(predictors))
    _assert_pytree_leaves_equal(predictors, restored)


# -- save_run / load_run ---------------------------------------------------


def test_save_run_writes_expected_layout(tmp_path: Path) -> None:
    predictors = _two_tuple()
    solver = _solver()
    optax_cfg = _optax_config()

    run_dir = tmp_path / "run0"
    save_run(
        run_dir,
        predictors=predictors,
        solver=solver,
        optax_config=optax_cfg,
    )

    assert (run_dir / "predictors.eqx").exists()
    metadata_path = run_dir / "metadata.json"
    assert metadata_path.exists()
    with metadata_path.open() as f:
        metadata = json.load(f)

    expected_keys = {
        "timestamp",
        "version",
        "predictors",
        "solver",
        "optax_config",
        "evosax_config",
        "loss_history",
        "extras",
    }
    assert set(metadata.keys()) == expected_keys
    assert isinstance(metadata["timestamp"], str)
    assert isinstance(metadata["version"], str)
    assert metadata["solver"] == solver.to_dict()
    assert metadata["evosax_config"] is None
    assert metadata["loss_history"] is None
    assert metadata["extras"] == {}


@pytest.mark.parametrize(
    "factory,expected_paths",
    [
        ("bare_module", [""]),
        ("one_tuple", ["[0]"]),
        ("two_tuple", ["[0]", "[1]"]),
        ("dict_shape", ["['growth']", "['nucleation']"]),
    ],
)
def test_save_run_records_per_leaf_description(
    factory: str, expected_paths: list[str], tmp_path: Path
) -> None:
    """Metadata records one ``{path, class}`` entry per ``eqx.Module`` leaf.

    The leaf list is what gives a load-time mismatch a human-readable
    diagnosis — without it, the user only sees ``eqx`` complain about a
    shape mismatch.
    """
    factory_fn = dict(PYTREE_FACTORIES)[factory]
    predictors = factory_fn()
    run_dir = tmp_path / f"run_{factory}"
    save_run(run_dir, predictors=predictors, solver=_solver())

    with (run_dir / "metadata.json").open() as f:
        metadata = json.load(f)

    desc = metadata["predictors"]
    assert "tree_structure" in desc
    assert isinstance(desc["tree_structure"], str)

    leaves = desc["leaves"]
    assert [leaf["path"] for leaf in leaves] == expected_paths
    for leaf in leaves:
        assert leaf["class"] == "hybridmodels.predictors.base.BoundedPredictor"


def test_load_run_round_trips_predictors_solver_optax(tmp_path: Path) -> None:
    predictors = _two_tuple()
    solver = _solver()
    optax_cfg = _optax_config()

    run_dir = tmp_path / "run0"
    save_run(
        run_dir,
        predictors=predictors,
        solver=solver,
        optax_config=optax_cfg,
    )

    template = _template_for(predictors)
    loaded = load_run(
        run_dir,
        predictors_template=template,
        optax_cls=OptaxTrainingConfig,
    )

    assert set(loaded.keys()) == {
        "predictors",
        "solver",
        "optax_config",
        "evosax_config",
        "loss_history",
        "extras",
    }
    _assert_pytree_leaves_equal(predictors, loaded["predictors"])
    assert loaded["solver"].to_dict() == solver.to_dict()
    assert isinstance(loaded["optax_config"], OptaxTrainingConfig)
    for field in dataclasses.fields(OptaxTrainingConfig):
        if field.name == "loss":
            # Loss is stringified at save time; re-resolution is the user's job.
            continue
        assert getattr(loaded["optax_config"], field.name) == getattr(optax_cfg, field.name)
    assert loaded["evosax_config"] is None
    assert loaded["loss_history"] is None
    assert loaded["extras"] == {}


def test_save_run_with_optax_and_evosax_configs(tmp_path: Path) -> None:
    predictors = _one_tuple()
    solver = _solver()
    optax_cfg = _optax_config()
    evosax_cfg = _evosax_config()

    run_dir = tmp_path / "run_both"
    save_run(
        run_dir,
        predictors=predictors,
        solver=solver,
        optax_config=optax_cfg,
        evosax_config=evosax_cfg,
    )

    loaded = load_run(
        run_dir,
        predictors_template=_template_for(predictors),
        optax_cls=OptaxTrainingConfig,
        evosax_cls=EvosaxTrainingConfig,
    )

    assert isinstance(loaded["optax_config"], OptaxTrainingConfig)
    assert isinstance(loaded["evosax_config"], EvosaxTrainingConfig)
    for field in dataclasses.fields(EvosaxTrainingConfig):
        if field.name == "loss":
            continue
        assert getattr(loaded["evosax_config"], field.name) == getattr(evosax_cfg, field.name)


def test_save_run_propagates_loss_history_and_extras(tmp_path: Path) -> None:
    predictors = _one_tuple()
    solver = _solver()
    history = [0.5, 0.3, 0.1]
    extras = {"experiment_name": "smoke", "n_iter": 3}

    run_dir = tmp_path / "run_extras"
    save_run(
        run_dir,
        predictors=predictors,
        solver=solver,
        loss_history=history,
        extras=extras,
    )

    loaded = load_run(run_dir, predictors_template=_template_for(predictors))
    assert loaded["loss_history"] == history
    assert loaded["extras"] == extras


def test_load_run_returns_raw_dict_when_classes_missing(tmp_path: Path) -> None:
    predictors = _one_tuple()
    solver = _solver()
    optax_cfg = _optax_config()
    evosax_cfg = _evosax_config()

    run_dir = tmp_path / "run_raw"
    save_run(
        run_dir,
        predictors=predictors,
        solver=solver,
        optax_config=optax_cfg,
        evosax_config=evosax_cfg,
    )

    loaded = load_run(
        run_dir,
        predictors_template=_template_for(predictors),
        optax_cls=None,
        evosax_cls=None,
    )

    assert isinstance(loaded["optax_config"], dict)
    assert isinstance(loaded["evosax_config"], dict)
    assert loaded["optax_config"]["steps"] == list(optax_cfg.steps)
    assert loaded["evosax_config"]["population_size"] == evosax_cfg.population_size


def test_save_run_creates_missing_directory(tmp_path: Path) -> None:
    predictors = _one_tuple()
    solver = _solver()
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    save_run(nested, predictors=predictors, solver=solver)
    assert nested.is_dir()
    assert (nested / "predictors.eqx").exists()
    assert (nested / "metadata.json").exists()


def test_save_run_overwrites_existing_directory(tmp_path: Path) -> None:
    predictors_a = (_bounded(jr.PRNGKey(1)),)
    predictors_b = (_bounded(jr.PRNGKey(2)),)
    solver = _solver()

    run_dir = tmp_path / "run_overwrite"
    save_run(run_dir, predictors=predictors_a, solver=solver, extras={"v": 1})
    save_run(run_dir, predictors=predictors_b, solver=solver, extras={"v": 2})

    loaded = load_run(run_dir, predictors_template=_template_for(predictors_b))
    _assert_pytree_leaves_equal(predictors_b, loaded["predictors"])
    assert loaded["extras"] == {"v": 2}


def test_save_run_stringifies_loss_callable(tmp_path: Path) -> None:
    predictors = _one_tuple()
    solver = _solver()
    optax_cfg = _optax_config(loss=masked_mse)

    run_dir = tmp_path / "run_loss_callable"
    save_run(run_dir, predictors=predictors, solver=solver, optax_config=optax_cfg)

    with (run_dir / "metadata.json").open() as f:
        metadata = json.load(f)
    assert metadata["optax_config"]["loss"] == "hybridmodels.losses.masked_mse"

    loaded = load_run(
        run_dir,
        predictors_template=_template_for(predictors),
        optax_cls=OptaxTrainingConfig,
    )
    assert isinstance(loaded["optax_config"], OptaxTrainingConfig)
    assert loaded["optax_config"].loss == "hybridmodels.losses.masked_mse"


def test_kan_predictor_round_trip_forward_pass(tmp_path: Path) -> None:
    """KAN's static/dynamic split is the highest round-trip risk — pin it.

    Wrapped in a 1-tuple to also exercise the tuple-pytree path through
    the new pytree-aware helpers.
    """
    predictors = (_kan(),)
    x = jnp.asarray([0.25, -0.5])
    expected = predictors[0](x)

    path = tmp_path / "kan.eqx"
    save_predictors(path, predictors)
    restored = load_predictors(path, (_kan(jr.PRNGKey(99)),))
    actual = restored[0](x)

    assert np.allclose(np.asarray(expected), np.asarray(actual), atol=1e-6)


def test_save_run_describes_bare_predictor_with_empty_path(tmp_path: Path) -> None:
    """A single-Module pytree should be describable too — empty keystr path.

    The is_module-stopped traversal terminates at the root, giving a
    single-entry leaves list with ``path == ""`` and the runtime class.
    """
    predictor = _bare_predictor()
    run_dir = tmp_path / "run_bare"
    save_run(run_dir, predictors=predictor, solver=_solver())

    with (run_dir / "metadata.json").open() as f:
        metadata = json.load(f)

    leaves = metadata["predictors"]["leaves"]
    assert len(leaves) == 1
    assert leaves[0]["path"] == ""
    assert leaves[0]["class"] == "hybridmodels.predictors.base.BoundedPredictor"


def test_predictors_eqx_filename_matches_context_md(tmp_path: Path) -> None:
    """CONTEXT.md "Save format" commits to the literal filename ``predictors.eqx``."""
    predictors = _two_tuple()
    run_dir = tmp_path / "run_filename"
    save_run(run_dir, predictors=predictors, solver=_solver())
    assert (run_dir / "predictors.eqx").exists()
    # And not the legacy single-predictor name:
    assert not (run_dir / "predictor.eqx").exists()


def test_save_predictors_written_file_has_content(tmp_path: Path) -> None:
    """``save_predictors`` must emit a non-empty binary file the matching
    ``load_predictors`` can read back to identical inexact-array leaves.

    Path object input is exercised here; the str-path case is covered
    separately in ``test_save_load_predictors_accepts_str_path``.
    """
    predictors = _two_tuple()
    path = tmp_path / "predictors.eqx"
    save_predictors(path, predictors)
    assert path.stat().st_size > 0

    restored = load_predictors(path, _template_for(predictors))
    leaves_orig = jtu.tree_leaves(eqx.filter(predictors, eqx.is_inexact_array))
    leaves_back = jtu.tree_leaves(eqx.filter(restored, eqx.is_inexact_array))
    assert len(leaves_orig) == len(leaves_back)
    for a, b in zip(leaves_orig, leaves_back, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
