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

import diffrax
import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jax import Array

from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from hybridmodels.losses import masked_mse
from hybridmodels.predictors.base import (
    BoundedPredictor,
    BoundScaler,
    CovariateSelector,
    Predictor,
)
from hybridmodels.predictors.mlp import MLPPredictor
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.training.evosax import (
    EvosaxTrainingConfig,
    _initial_population,
    train_with_evosax,
)
from hybridmodels.ui.testing import RecordingUI

THETA_STAR = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)
N_DIM = 4


class _QuadraticPredictor(Predictor):
    """Single inexact-array leaf ``theta: [4]`` — covariates are ignored on call."""

    theta: Array

    def __init__(self, theta: Array) -> None:
        # Float32 to keep CMA-ES (whose state is float32 by default) and the
        # predictor's leaf in a single dtype, avoiding silent upcasts that
        # would force a re-trace inside population_eval.
        self.theta = jnp.asarray(theta, dtype=jnp.float32)

    def __call__(self, covariates):  # type: ignore[override]
        # The covariates dict is irrelevant here — the "model" is just the
        # constant theta. Returning the leaf directly lets simulate_fn fold it
        # into [T, S] = [1, 4] without an ODE call.
        return self.theta


def _quadratic_dataset() -> Dataset:
    """One bucket containing one experiment, T=1, D=4, mask all True."""
    ts = jnp.array([0.0], dtype=jnp.float32)
    channels = {f"c{i}": ChannelObs(ts=ts, values=THETA_STAR[i : i + 1]) for i in range(N_DIM)}
    exp = make_experiment(
        covariates={"id": 0.0},
        channels=channels,
        y0_fn=lambda _c, _ch: jnp.zeros(N_DIM, dtype=jnp.float32),
        exp_id="exp_0",
    )
    return make_dataset(
        [exp],
        state_to_output=lambda state: state,
        output_channel_names=tuple(f"c{i}" for i in range(N_DIM)),
    )


def _simulate_fn(predictor, ts, covariates, y0, solver):
    """Identity-of-predictor "simulator": returns ``[T, 4]`` constant in time."""
    # predictor(covariates) -> [4]; broadcasting to [T, 4] gives the per-experiment
    # full-state trajectory expected by the framework.
    return jnp.broadcast_to(predictor(covariates)[None, :], (ts.shape[0], N_DIM))


def _solver() -> SolverConfig:
    """A ``SolverConfig`` for the simulate_fn signature; values are unused here."""
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=4096,
        dt0=0.05,
    )


def _eval_loss(predictor: Predictor, ds: Dataset) -> float:
    """Recompute the same ``masked_mse`` the trainer minimises, single-bucket only."""
    bp = ds.bucket_payloads[0]

    def per_exp(ts, cov, y0):
        return ds.state_to_output(_simulate_fn(predictor, ts, cov, y0, _solver()))

    pred_obs = jax.vmap(per_exp, in_axes=(0, 0, 0))(bp.ts, bp.covariates, bp.y0)
    return float(masked_mse(pred_obs, bp))


def test_convergence_to_known_minimum() -> None:
    pred = _QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = _quadratic_dataset()
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
        simulate_fn=_simulate_fn,
        solver=_solver(),
        key=jr.PRNGKey(0),
    )
    assert jnp.allclose(trained.theta, THETA_STAR, rtol=5e-2, atol=5e-2)
    assert history[-1] < 1e-3


def test_best_ever_tracking() -> None:
    pred = _QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = _quadratic_dataset()
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
        simulate_fn=_simulate_fn,
        solver=_solver(),
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
    pred = _QuadraticPredictor(theta=jnp.zeros(N_DIM))
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
    lhs_config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=24,
        num_generations=1,
        init="lhs_box",
        init_box_extent=2.0,
        verbose=False,
    )

    pop_warm = _initial_population(pred, warm_config, trainable=mask, key=key)
    pop_lhs = _initial_population(pred, lhs_config, trainable=mask, key=key)

    def _max_pairwise(pop: Array) -> float:
        diffs = pop[:, None, :] - pop[None, :, :]
        d = jnp.linalg.norm(diffs, axis=-1)
        return float(d.max())

    assert _max_pairwise(pop_lhs) >= 1.5 * _max_pairwise(pop_warm)


def test_flatten_unflatten_round_trip() -> None:
    # ``eqx.partition`` + ``ravel_pytree`` is the round-trip CMA-ES
    # uses to operate in flat parameter space and then reassemble the
    # predictor pytree. This must invert exactly: any divergence would
    # mean the strategy is optimising over a different parameterisation
    # than the one we evaluate.
    inner = MLPPredictor(in_size=2, out_size=1, width_size=8, depth=2, key=jr.PRNGKey(0))
    pred = BoundedPredictor(
        selector=CovariateSelector(keys=("a", "b")),
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
    pred = _QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = _quadratic_dataset()
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
            simulate_fn=_simulate_fn,
            solver=_solver(),
        )


def test_recording_ui_lifecycle_events_fire() -> None:
    pred = _QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = _quadratic_dataset()
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
        simulate_fn=_simulate_fn,
        solver=_solver(),
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
    pred = _QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = _quadratic_dataset()
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
        simulate_fn=_simulate_fn,
        solver=_solver(),
        key=jr.PRNGKey(0),
    )
    assert capsys.readouterr().out == ""
