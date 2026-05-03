"""Tests for ``hybridmodels.serialise`` (SPEC §5.11 / R-A5 / R-S2).

Phase 16 of the build plan. Pins the four public helpers
(``save_predictor`` / ``load_predictor`` / ``save_run`` / ``load_run``) and
the on-disk metadata contract documented in CONTEXT.md "Serialisation".

Coverage strategy
-----------------
- The single-predictor round-trip is parametrised over every concrete
  ``Predictor`` shape supported in v1, mirroring (but not duplicating) the
  R-A5 enforcement gate at ``tests/test_predictors_serialise.py``. Where
  that test exercises ``eqx.tree_serialise_leaves`` directly, this one
  goes through the public ``save_predictor`` / ``load_predictor`` entry
  points so the helper plumbing (path coercion, file mode) is covered.
- ``save_run`` / ``load_run`` get one minimal happy path plus targeted
  variants for each load-bearing branch: optax+evosax both populated,
  optional ``loss_history`` / ``extras`` propagation, missing builder
  classes, directory creation, overwrite semantics, and loss-callable
  stringification.
- The KAN-specific forward-pass equality check guards the static/dynamic
  split documented in ``predictors/kan.py`` — the highest-risk predictor
  for round-trip drift because of the jaxkan / Param wiring.
"""

# ruff: noqa: F722

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

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
    CovariateSelector,
    KANPredictor,
    MLPPredictor,
    NeuralNPolynomial,
    RatePair,
)
from hybridmodels.serialise import (
    load_predictor,
    load_run,
    save_predictor,
    save_run,
)
from hybridmodels.solver import SolverConfig
from hybridmodels.training.evosax import EvosaxTrainingConfig
from hybridmodels.training.optax import OptaxTrainingConfig

# -- predictor factories ---------------------------------------------------


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
        selector=CovariateSelector(keys=("a", "b")),
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


def _rate_pair() -> RatePair:
    return RatePair(
        nucleation=_bounded(jr.PRNGKey(11)),
        growth=_bounded(jr.PRNGKey(22)),
    )


def _neural_npoly(key: Array | None = None) -> NeuralNPolynomial:
    coeff_net = MLPPredictor(
        in_size=3,
        out_size=6,
        width_size=8,
        depth=2,
        activation_name="tanh",
        key=jr.PRNGKey(0) if key is None else key,
    )
    return NeuralNPolynomial(
        coeff_net=coeff_net,
        exponents=(0.0, 1.0, 2.0),
        in_size=3,
        out_size=2,
    )


PREDICTOR_FACTORIES: list[tuple[str, Callable[[], eqx.Module]]] = [
    ("mlp_predictor", _mlp),
    ("kan_predictor", _kan),
    ("bounded_predictor", _bounded),
    ("rate_pair", _rate_pair),
    ("neural_npoly", _neural_npoly),
]


def _template_for(predictor: eqx.Module) -> eqx.Module:
    """Return a same-architecture, different-weights ``predictor`` to deserialise into.

    The deserialise template must match class + static config but should
    *not* share any inexact-array leaf values with the saved predictor —
    otherwise the round-trip equality check would be vacuous (the template
    leaves would already equal the saved leaves).
    """
    if isinstance(predictor, MLPPredictor):
        return _mlp(jr.PRNGKey(99))
    if isinstance(predictor, KANPredictor):
        return _kan(jr.PRNGKey(99))
    if isinstance(predictor, BoundedPredictor):
        return _bounded(jr.PRNGKey(99))
    if isinstance(predictor, RatePair):
        return RatePair(
            nucleation=_bounded(jr.PRNGKey(91)),
            growth=_bounded(jr.PRNGKey(92)),
        )
    if isinstance(predictor, NeuralNPolynomial):
        return _neural_npoly(jr.PRNGKey(99))
    raise AssertionError(f"no template factory for {type(predictor).__name__}")


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


def _assert_pytree_leaves_equal(a: eqx.Module, b: eqx.Module) -> None:
    leaves_a = jtu.tree_leaves(a)
    leaves_b = jtu.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for left, right in zip(leaves_a, leaves_b, strict=True):
        if hasattr(left, "shape"):
            assert np.array_equal(np.asarray(left), np.asarray(right))
        else:
            assert left == right


# -- save_predictor / load_predictor ---------------------------------------


@pytest.mark.parametrize(
    "factory",
    [f for _, f in PREDICTOR_FACTORIES],
    ids=[name for name, _ in PREDICTOR_FACTORIES],
)
def test_save_load_predictor_round_trip(
    factory: Callable[[], eqx.Module], tmp_path: Path
) -> None:
    predictor = factory()
    path = tmp_path / "predictor.eqx"
    save_predictor(path, predictor)
    assert path.exists()

    template = _template_for(predictor)
    restored = load_predictor(path, template)
    _assert_pytree_leaves_equal(predictor, restored)


def test_save_load_predictor_accepts_str_path(tmp_path: Path) -> None:
    predictor = _mlp()
    path = str(tmp_path / "predictor.eqx")
    save_predictor(path, predictor)
    restored = load_predictor(path, _mlp(jr.PRNGKey(99)))
    _assert_pytree_leaves_equal(predictor, restored)


# -- save_run / load_run ---------------------------------------------------


def test_save_run_writes_expected_layout(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    optax_cfg = _optax_config()

    run_dir = tmp_path / "run0"
    save_run(
        run_dir,
        predictor=predictor,
        solver=solver,
        optax_config=optax_cfg,
    )

    assert (run_dir / "predictor.eqx").exists()
    metadata_path = run_dir / "metadata.json"
    assert metadata_path.exists()
    with metadata_path.open() as f:
        metadata = json.load(f)

    expected_keys = {
        "timestamp",
        "version",
        "predictor_class_path",
        "solver",
        "optax_config",
        "evosax_config",
        "loss_history",
        "extras",
    }
    assert set(metadata.keys()) == expected_keys
    assert isinstance(metadata["timestamp"], str)
    assert isinstance(metadata["version"], str)
    assert (
        metadata["predictor_class_path"]
        == "hybridmodels.predictors.base.BoundedPredictor"
    )
    assert metadata["solver"] == solver.to_dict()
    assert metadata["evosax_config"] is None
    assert metadata["loss_history"] is None
    assert metadata["extras"] == {}


def test_load_run_round_trips_predictor_solver_optax(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    optax_cfg = _optax_config()

    run_dir = tmp_path / "run0"
    save_run(
        run_dir,
        predictor=predictor,
        solver=solver,
        optax_config=optax_cfg,
    )

    template = _bounded(jr.PRNGKey(123))
    loaded = load_run(
        run_dir,
        predictor_template=template,
        optax_cls=OptaxTrainingConfig,
    )

    assert set(loaded.keys()) == {
        "predictor",
        "solver",
        "optax_config",
        "evosax_config",
        "loss_history",
        "extras",
    }
    _assert_pytree_leaves_equal(predictor, loaded["predictor"])
    assert loaded["solver"].to_dict() == solver.to_dict()
    assert isinstance(loaded["optax_config"], OptaxTrainingConfig)
    for field in dataclasses.fields(OptaxTrainingConfig):
        if field.name == "loss":
            # Loss is stringified at save time; re-resolution is the user's job.
            continue
        assert getattr(loaded["optax_config"], field.name) == getattr(
            optax_cfg, field.name
        )
    assert loaded["evosax_config"] is None
    assert loaded["loss_history"] is None
    assert loaded["extras"] == {}


def test_save_run_with_optax_and_evosax_configs(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    optax_cfg = _optax_config()
    evosax_cfg = _evosax_config()

    run_dir = tmp_path / "run_both"
    save_run(
        run_dir,
        predictor=predictor,
        solver=solver,
        optax_config=optax_cfg,
        evosax_config=evosax_cfg,
    )

    loaded = load_run(
        run_dir,
        predictor_template=_bounded(jr.PRNGKey(7)),
        optax_cls=OptaxTrainingConfig,
        evosax_cls=EvosaxTrainingConfig,
    )

    assert isinstance(loaded["optax_config"], OptaxTrainingConfig)
    assert isinstance(loaded["evosax_config"], EvosaxTrainingConfig)
    for field in dataclasses.fields(EvosaxTrainingConfig):
        if field.name == "loss":
            continue
        assert getattr(loaded["evosax_config"], field.name) == getattr(
            evosax_cfg, field.name
        )


def test_save_run_propagates_loss_history_and_extras(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    history = [0.5, 0.3, 0.1]
    extras = {"experiment_name": "smoke", "n_iter": 3}

    run_dir = tmp_path / "run_extras"
    save_run(
        run_dir,
        predictor=predictor,
        solver=solver,
        loss_history=history,
        extras=extras,
    )

    loaded = load_run(run_dir, predictor_template=_bounded(jr.PRNGKey(5)))
    assert loaded["loss_history"] == history
    assert loaded["extras"] == extras


def test_load_run_returns_raw_dict_when_classes_missing(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    optax_cfg = _optax_config()
    evosax_cfg = _evosax_config()

    run_dir = tmp_path / "run_raw"
    save_run(
        run_dir,
        predictor=predictor,
        solver=solver,
        optax_config=optax_cfg,
        evosax_config=evosax_cfg,
    )

    loaded = load_run(
        run_dir,
        predictor_template=_bounded(jr.PRNGKey(5)),
        optax_cls=None,
        evosax_cls=None,
    )

    assert isinstance(loaded["optax_config"], dict)
    assert isinstance(loaded["evosax_config"], dict)
    assert loaded["optax_config"]["steps"] == list(optax_cfg.steps)
    assert loaded["evosax_config"]["population_size"] == evosax_cfg.population_size


def test_save_run_creates_missing_directory(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    save_run(nested, predictor=predictor, solver=solver)
    assert nested.is_dir()
    assert (nested / "predictor.eqx").exists()
    assert (nested / "metadata.json").exists()


def test_save_run_overwrites_existing_directory(tmp_path: Path) -> None:
    predictor_a = _bounded(jr.PRNGKey(1))
    predictor_b = _bounded(jr.PRNGKey(2))
    solver = _solver()

    run_dir = tmp_path / "run_overwrite"
    save_run(run_dir, predictor=predictor_a, solver=solver, extras={"v": 1})
    save_run(run_dir, predictor=predictor_b, solver=solver, extras={"v": 2})

    loaded = load_run(run_dir, predictor_template=_bounded(jr.PRNGKey(99)))
    _assert_pytree_leaves_equal(predictor_b, loaded["predictor"])
    assert loaded["extras"] == {"v": 2}


def test_save_run_stringifies_loss_callable(tmp_path: Path) -> None:
    predictor = _bounded()
    solver = _solver()
    optax_cfg = _optax_config(loss=masked_mse)

    run_dir = tmp_path / "run_loss_callable"
    save_run(run_dir, predictor=predictor, solver=solver, optax_config=optax_cfg)

    with (run_dir / "metadata.json").open() as f:
        metadata = json.load(f)
    assert metadata["optax_config"]["loss"] == "hybridmodels.losses.masked_mse"

    loaded = load_run(
        run_dir,
        predictor_template=_bounded(jr.PRNGKey(5)),
        optax_cls=OptaxTrainingConfig,
    )
    assert isinstance(loaded["optax_config"], OptaxTrainingConfig)
    assert loaded["optax_config"].loss == "hybridmodels.losses.masked_mse"


def test_kan_predictor_round_trip_forward_pass(tmp_path: Path) -> None:
    """KAN's static/dynamic split is the highest round-trip risk — pin it."""
    predictor = _kan()
    x = jnp.asarray([0.25, -0.5])
    expected = predictor(x)

    path = tmp_path / "kan.eqx"
    save_predictor(path, predictor)
    restored = load_predictor(path, _kan(jr.PRNGKey(99)))
    actual = restored(x)

    assert np.allclose(np.asarray(expected), np.asarray(actual), atol=1e-6)
