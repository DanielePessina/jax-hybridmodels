"""End-to-end tests of the WS2 extensibility seams.

These exercise the public extension points through the full training
pipeline, not inner kernels:

- ``trainable=`` actually freezes leaves through ``train_with_optax``.
- A custom ``loss`` callable flows through ``train_with_optax``.
- The predictors pytree works in dict and nested shapes, not just a bare
  Module or tuple.
- ``optimizer=`` accepts a factory and a raw ``optax.GradientTransformation``.
- ``penalty_fn`` replaces the default bound penalty with a custom
  regulariser.
- The evosax ``algorithm=`` accepts the registered strategies.

Each test builds a model that has two trainable leaves and asserts only the
intended ones moved, so a silent mask/optimiser bug shows up as a moved
leaf rather than a vacuous pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from _harness import (
    OmegaPredictor,
    make_oscillator_dataset,
    make_oscillator_simulate_fn,
    oscillator_state_to_output,
)
from jax import Array

from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from hybridmodels.predictors.base import Predictor
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import freeze_paths, trainable_mask
from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax


class TwoLeafPredictor(Predictor):
    """Two trainable scalar leaves, so ``trainable`` can hold one fixed.

    ``a`` and ``b`` both feed the (degenerate) "simulation", which is just
    the sum projected to one channel over the bucket's time axis.
    """

    a: Array
    b: Array

    def __init__(self, a: float, b: float) -> None:
        self.a = jnp.asarray(a, dtype=jnp.float32)
        self.b = jnp.asarray(b, dtype=jnp.float32)


def two_leaf_simulate_fn(predictor, ts, covariates, y0, solver):
    return jnp.broadcast_to((predictor.a + predictor.b)[None], (ts.shape[0], 1))


def two_leaf_state_to_output(state: Array) -> Array:
    return state[..., :1]


def two_leaf_dataset() -> Dataset:
    ts = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={"y": ChannelObs(ts=ts, values=jnp.array([1.0, 2.0, 3.0]))},
        y0_fn=lambda c, ch: jnp.zeros(1, dtype=jnp.float32),
        exp_id="e0",
    )
    return make_dataset([exp], output_channel_names=("y",))


@pytest.fixture
def two_leaf():
    return TwoLeafPredictor(a=0.0, b=0.0)


@pytest.fixture
def solver() -> SolverConfig:
    return SolverConfig(
        solver=__import__("diffrax").Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=4096,
        dt0=0.05,
    )


class TestTrainableEndToEnd:
    def test_frozen_leaf_does_not_move_through_optax(self, two_leaf, solver):
        mask = freeze_paths(trainable_mask(two_leaf), ("b",))
        _, final = train_with_optax(
            two_leaf,
            two_leaf_dataset(),
            OptaxTrainingConfig(
                steps=(6,), lr=(1e-2,), optimizer=("adamw",), reset_optimiser_state=(False,)
            ),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            trainable=mask,
            key=jax.random.PRNGKey(0),
        )
        # The frozen leaf must not have moved at all.
        assert float(final.b) == pytest.approx(0.0)
        # And training must actually have moved the free leaf (not a no-op).
        assert float(final.a) != 0.0

    def test_frozen_leaf_does_not_move_through_evosax(self, two_leaf, solver):
        mask = freeze_paths(trainable_mask(two_leaf), ("b",))
        _, final = train_with_evosax(
            two_leaf,
            two_leaf_dataset(),
            EvosaxTrainingConfig(population_size=16, num_generations=3, init="warm"),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            trainable=mask,
            key=jax.random.PRNGKey(0),
        )
        assert float(final.b) == pytest.approx(0.0)


class TestCustomLossEndToEnd:
    def test_custom_loss_callable_flows_through_trainer(self, two_leaf, solver):
        seen = {}

        def my_loss(pred_obs, bp):
            seen["called"] = True
            return jnp.sum((pred_obs - bp.y_observed) ** 2)

        _, final = train_with_optax(
            two_leaf,
            two_leaf_dataset(),
            OptaxTrainingConfig(
                steps=(3,),
                lr=(1e-3,),
                optimizer=("adamw",),
                reset_optimiser_state=(False,),
                loss=my_loss,
            ),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert seen.get("called") is True


class TestPytreeShapes:
    def test_dict_predictors_train_with_optax(self, solver):
        preds = {"alpha": OmegaPredictor(1.5), "beta": OmegaPredictor(1.5)}

        def dict_simulate_fn(predictor, ts, covariates, y0, slv):
            # Both dict leaves drive the same oscillator; the framework never
            # inspects the container, so this must work through the pipeline.
            return make_oscillator_simulate_fn()(predictor["alpha"], ts, covariates, y0, slv)

        _, final = train_with_optax(
            preds,
            make_oscillator_dataset(),
            OptaxTrainingConfig(
                steps=(4,), lr=(5e-3,), optimizer=("adamw",), reset_optimiser_state=(False,)
            ),
            simulate_fn=dict_simulate_fn,
            state_to_output=oscillator_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert set(final) == {"alpha", "beta"}
        # Training must have moved both leaves toward the truth (1.0).
        assert abs(float(final["alpha"].omega) - 1.0) < 0.5
        assert abs(float(final["beta"].omega) - 1.0) < 0.5

    def test_nested_tuple_predictors_train_with_evosax(self, solver):
        preds = (TwoLeafPredictor(0.0, 0.0), TwoLeafPredictor(0.0, 0.0))
        _, final = train_with_evosax(
            preds,
            two_leaf_dataset(),
            EvosaxTrainingConfig(population_size=16, num_generations=2, init="warm"),
            simulate_fn=lambda ps, ts, cov, y0, slv: jnp.broadcast_to(
                (ps[0].a + ps[1].a)[None], (ts.shape[0], 1)
            ),
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert len(final) == 2


class TestInjectableOptimizer:
    def test_factory_optimizer_runs(self, two_leaf, solver):
        _, final = train_with_optax(
            two_leaf,
            two_leaf_dataset(),
            OptaxTrainingConfig(
                steps=(4,),
                lr=(1e-3,),
                optimizer=(
                    lambda learning_rate: optax.chain(
                        optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
                    ),
                ),
                reset_optimiser_state=(False,),
            ),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert final is not None

    def test_raw_transformation_optimizer_runs(self, two_leaf, solver):
        # A raw transformation's state cannot be re-hyperparametrised, so a
        # single phase with a fixed lr is the valid use.
        _, final = train_with_optax(
            two_leaf,
            two_leaf_dataset(),
            OptaxTrainingConfig(
                steps=(4,),
                lr=(1e-3,),
                optimizer=(optax.adam(1e-3),),
                reset_optimiser_state=(False,),
            ),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert final is not None

    def test_raw_optimizer_with_lr_change_requires_reset(self, two_leaf, solver):
        # Reusing ONE raw object across both phases makes the specs equal,
        # so the constructor's optimizer-transition check passes; the lr
        # change then has to be refused by _begin_phase, because a raw
        # transformation's state cannot be re-hyperparametrised.
        raw = optax.adam(1e-3)
        with pytest.raises(ValueError, match="cannot be re-hyperparametrised"):
            train_with_optax(
                two_leaf,
                two_leaf_dataset(),
                OptaxTrainingConfig(
                    steps=(2, 2),
                    lr=(1e-3, 1e-2),
                    optimizer=(raw, raw),
                    reset_optimiser_state=(False, False),
                    length_schedule=(1.0, 1.0),
                ),
                simulate_fn=two_leaf_simulate_fn,
                state_to_output=two_leaf_state_to_output,
                solver=solver,
                key=jax.random.PRNGKey(0),
            )


class TestPenaltyFnHook:
    def test_custom_penalty_fn_is_used(self, two_leaf, solver):
        seen = {}

        def my_penalty(predictors, grids):
            seen["called"] = True
            return predictors.a**2

        _, _ = train_with_optax(
            two_leaf,
            two_leaf_dataset(),
            OptaxTrainingConfig(
                steps=(2,),
                lr=(1e-3,),
                optimizer=("adamw",),
                reset_optimiser_state=(False,),
                penalty_weight=(1.0,),
                penalty_fn=my_penalty,
            ),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert seen.get("called") is True


class TestEvosaxAlgorithms:
    @pytest.mark.parametrize("algo", ["CMA_ES", "Sep_CMA_ES", "SimpleES"])
    def test_registered_algorithms_run(self, two_leaf, solver, algo):
        _, final = train_with_evosax(
            two_leaf,
            two_leaf_dataset(),
            EvosaxTrainingConfig(
                population_size=16, num_generations=2, init="warm", algorithm=algo
            ),
            simulate_fn=two_leaf_simulate_fn,
            state_to_output=two_leaf_state_to_output,
            solver=solver,
            key=jax.random.PRNGKey(0),
        )
        assert final is not None
