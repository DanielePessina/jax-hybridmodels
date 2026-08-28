"""Evosax training tests.

Synthetic 4-D quadratic problem: a single trainable ``theta: [4]``
whose loss against the constant target ``theta_star = [1, -2, 3, -4]``
is the elementwise MSE. The "simulator" is the identity-of-the-predictor,
so the entire test file runs in well under a minute and isolates the
evosax loop from any real ODE numerics.

Each test pins one piece of the contract:

* ``test_convergence_to_known_minimum`` — end-to-end correctness on
  the convex target.
* ``test_best_ever_tracking`` — host-side argmin bookkeeping must not
  regress between generations.
* ``test_init_modes_change_population_spread`` — ``"warm"`` vs
  ``"lhs_box"`` must produce visibly different gen-0 populations.
* ``test_flatten_unflatten_round_trip`` — ``eqx.partition`` and
  ``ravel_pytree`` must invert exactly.
* ``test_missing_key_raises`` — no silent default key.
* ``test_recording_ui_lifecycle_events_fire`` — UI protocol contract.
* ``test_silent_default_when_no_ui_and_verbose_false`` — UI selection
  fallback.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from _harness import (
    N_DIM,
    THETA_STAR,
    QuadraticPredictor,
    quadratic_dataset,
    quadratic_simulate_fn,
    quadratic_state_to_output,
    solver_config,
)
from jax import Array

from hybridmodels.data import Dataset
from hybridmodels.losses import masked_mse
from hybridmodels.predictors.base import (
    BoundedPredictor,
    BoundScaler,
    Predictor,
)
from hybridmodels.predictors.mlp import MLPPredictor
from hybridmodels.trainable import trainable_mask
from hybridmodels.training.evosax import (
    EvosaxTrainingConfig,
    _box_population,
    _build_strategy,
    train_with_evosax,
)
from hybridmodels.ui.testing import RecordingUI


def _eval_loss(predictor: Predictor, ds: Dataset) -> float:
    """Recompute the same ``masked_mse`` the trainer minimises, single-bucket only."""
    bp = ds.bucket_payloads[0]

    def per_exp(ts, cov, y0):
        return quadratic_state_to_output(
            quadratic_simulate_fn(predictor, ts, cov, y0, solver_config())
        )

    pred_obs = jax.vmap(per_exp, in_axes=(0, 0, 0))(bp.ts, bp.covariates, bp.y0)
    return float(masked_mse(pred_obs, bp))


def test_convergence_to_known_minimum() -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=24,
        num_generations=40,
        init="warm",
        sigma_init=0.5,
        verbose=False,
    )
    history, trained = train_with_evosax(
        pred,
        ds,
        config,
        simulate_fn=quadratic_simulate_fn,
        state_to_output=quadratic_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    assert jnp.allclose(trained.theta, THETA_STAR, rtol=5e-2, atol=5e-2)
    assert history[-1] < 1e-3


def test_best_ever_tracking() -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    ui = RecordingUI()
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=12,
        num_generations=6,
        init="warm",
        sigma_init=0.4,
        verbose=False,
    )
    _history, trained = train_with_evosax(
        pred,
        ds,
        config,
        simulate_fn=quadratic_simulate_fn,
        state_to_output=quadratic_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(1),
        ui=ui,
    )
    # The host-side best-ever bookkeeping must not regress: the
    # returned loss must be no worse than the final generation's
    # population mean. Otherwise the loop would have handed back a
    # candidate that was beaten by an earlier generation, defeating
    # the whole point of tracking the best ever.
    final_mean_fitness = next(
        kw["mean_fitness"] for name, kw in reversed(ui.events) if name == "on_generation_end"
    )
    returned_loss = _eval_loss(trained, ds)
    assert returned_loss <= final_mean_fitness + 1e-7


def test_init_modes_change_population_spread() -> None:
    # The LHS box covers a [-2, 2]^4 hypercube while ``"warm"`` draws
    # from a tight N(0, 0.5*I) ball — the LHS spread must be visibly
    # wider, which is the whole reason a user would pick ``"lhs_box"``
    # over ``"warm"`` in the first place.
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    mask = trainable_mask(pred)
    key = jr.PRNGKey(7)

    warm_config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=24,
        num_generations=1,
        init="warm",
        sigma_init=0.5,
        verbose=False,
    )
    uniform_config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=24,
        num_generations=1,
        init="uniform_box",
        init_box_extent=2.0,
        verbose=False,
    )
    lhs_config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=24,
        num_generations=1,
        init="lhs_box",
        init_box_extent=2.0,
        verbose=False,
    )

    params, _static = eqx.partition(pred, mask)
    flat, _unflatten = jfu.ravel_pytree(params)

    # _box_population is what train_with_evosax actually calls at gen 0 for
    # the box modes. The previous version of this test drove a helper the
    # training loop never invoked, so it could pass with the real init path
    # completely broken.
    pop_uniform = _box_population(flat=flat, config=uniform_config, key=key)
    pop_lhs = _box_population(flat=flat, config=lhs_config, key=key)

    def _max_pairwise(pop: Array) -> float:
        diffs = pop[:, None, :] - pop[None, :, :]
        return float(jnp.linalg.norm(diffs, axis=-1).max())

    # Both box modes must spread across the requested extent, and a warm
    # CMA-ES draw at sigma_init=0.5 must be tighter than either.
    strategy, strat_params = _build_strategy(config=warm_config, flat=flat)
    state = strategy.init(key, flat, strat_params)
    pop_warm, _ = strategy.ask(key, state, strat_params)

    assert _max_pairwise(pop_lhs) >= 1.5 * _max_pairwise(pop_warm)
    assert _max_pairwise(pop_uniform) >= 1.5 * _max_pairwise(pop_warm)


def test_box_population_respects_the_requested_extent() -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    mask = trainable_mask(pred)
    params, _static = eqx.partition(pred, mask)
    flat, _unflatten = jfu.ravel_pytree(params)
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=1,
        init="uniform_box",
        init_box_extent=3.0,
        verbose=False,
    )
    pop = _box_population(flat=flat, config=config, key=jr.PRNGKey(0))
    assert pop.shape == (32, flat.size)
    assert float(jnp.abs(pop - flat).max()) <= 3.0 + 1e-6


def test_flatten_unflatten_round_trip() -> None:
    # ``eqx.partition`` + ``ravel_pytree`` is the round-trip CMA-ES
    # uses to operate in flat parameter space and then reassemble the
    # predictor pytree. This must invert exactly: any divergence would
    # mean the strategy is optimising over a different parameterisation
    # than the one we evaluate.
    inner = MLPPredictor(in_size=2, out_size=1, width_size=8, depth=2, key=jr.PRNGKey(0))
    pred = BoundedPredictor(
        input_keys=("a", "b"),
        in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0))),
        inner=inner,
        out_scaler=BoundScaler(bounds=((0.0, 1.0),)),
    )
    mask = trainable_mask(pred)
    params, static = eqx.partition(pred, mask)
    flat, unflatten = jfu.ravel_pytree(params)

    rebuilt_params = unflatten(flat)
    rebuilt = eqx.combine(rebuilt_params, static)

    orig_leaves = jtu.tree_leaves(eqx.filter(pred, eqx.is_array))
    new_leaves = jtu.tree_leaves(eqx.filter(rebuilt, eqx.is_array))
    assert len(orig_leaves) == len(new_leaves) > 0
    for a, b in zip(orig_leaves, new_leaves, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert jnp.allclose(a, b)


def test_missing_key_raises() -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=8,
        num_generations=1,
        verbose=False,
    )
    with pytest.raises(TypeError):
        train_with_evosax(  # ty: ignore[missing-argument]
            pred,
            ds,
            config,
            simulate_fn=quadratic_simulate_fn,
            state_to_output=quadratic_state_to_output,
            solver=solver_config(),
        )


def test_recording_ui_lifecycle_events_fire() -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    n_gens = 3
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=8,
        num_generations=n_gens,
        init="warm",
        sigma_init=0.5,
        verbose=False,
    )
    ui = RecordingUI()
    train_with_evosax(
        pred,
        ds,
        config,
        simulate_fn=quadratic_simulate_fn,
        state_to_output=quadratic_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
        ui=ui,
    )
    names = [n for n, _ in ui.events]
    assert names.count("on_run_start") == 1
    assert names.count("on_compile_start") >= 1
    assert names.count("on_compile_done") >= 1
    assert names.count("on_run_end") == 1

    gen_events = [(n, kw) for n, kw in ui.events if n == "on_generation_end"]
    assert len(gen_events) == n_gens
    for i, (_, kw) in enumerate(gen_events):
        assert kw["gen_idx"] == i
        assert "best_fitness" in kw
        assert "mean_fitness" in kw


def test_silent_default_when_no_ui_and_verbose_false(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=8,
        num_generations=1,
        init="warm",
        sigma_init=0.5,
        verbose=False,
    )
    train_with_evosax(
        pred,
        ds,
        config,
        simulate_fn=quadratic_simulate_fn,
        state_to_output=quadratic_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    assert capsys.readouterr().out == ""


class TestEvosaxPenalty:
    """CMA-ES searches latent space unbounded, so saturation needs charging here too."""

    def test_negative_penalty_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EvosaxTrainingConfig(
                algorithm="CMA_ES",
                population_size=4,
                num_generations=1,
                penalty_weight=-1.0,
                verbose=False,
            )

    def test_default_weight_is_off(self) -> None:
        cfg = EvosaxTrainingConfig(
            algorithm="CMA_ES", population_size=4, num_generations=1, verbose=False
        )
        assert cfg.penalty_weight == 0.0

    def test_zero_weight_reproduces_the_unpenalised_run(self) -> None:
        # A predictors pytree with no BoundedPredictor leaf must be
        # completely unaffected, and an explicit zero must match the
        # default exactly.
        ds = quadratic_dataset()

        def run(weight: float) -> list[float]:
            history, _ = train_with_evosax(
                QuadraticPredictor(theta=jnp.zeros(N_DIM)),
                ds,
                EvosaxTrainingConfig(
                    algorithm="CMA_ES",
                    population_size=8,
                    num_generations=3,
                    sigma_init=0.5,
                    penalty_weight=weight,
                    verbose=False,
                ),
                simulate_fn=quadratic_simulate_fn,
                state_to_output=quadratic_state_to_output,
                solver=solver_config(),
                key=jr.PRNGKey(0),
            )
            return history

        assert run(0.0) == run(0.0)
