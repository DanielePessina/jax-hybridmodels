"""Exact checks a workflow cannot make, each against an oracle outside the framework.

Adjoint gradients against the analytic derivative, masked cells against poison,
bounds against a round trip, and compiled kernels against a trace counter. None
trains for long, and each guards a failure that would otherwise show up only as
a slightly wrong fit or a slow run.
"""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _models import (
    CHANNELS,
    SOLVER,
    TRUE_RATE,
    decay_dataset,
    identity,
    make_model,
    parameter_predictors,
    quiet,
    simulate_parameter,
)
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.losses import bal_mle, bal_mse, masked_mle, masked_mse
from jaxhybridmodels.solver import ADJOINT_REGISTRY

GRID = jnp.array([0.0, 0.5, 1.0, 2.0])


def solution_sum(rate: Array, adjoint: str, *, through_args: bool) -> Array:
    """Sum of the decay solution over ``GRID``, differentiable in ``rate``.

    Backsolve re-integrates an adjoint ODE and differentiates only what passes
    through ``args``, so ``through_args`` routes the rate there.
    """
    solver = hm.SolverConfig(
        rtol=1e-6, atol=1e-8, max_steps=32, dt0=0.1, adjoint=ADJOINT_REGISTRY[adjoint]()
    )
    y0 = jnp.array([1.0])
    if through_args:
        term = diffrax.ODETerm(lambda t, y, k: -k * y)
        return solver.diffeqsolve(term, GRID, y0, args=rate).ys.sum()
    term = diffrax.ODETerm(lambda t, y, args: -rate * y)
    return solver.diffeqsolve(term, GRID, y0).ys.sum()


@pytest.mark.parametrize(
    "adjoint,through_args",
    [("Direct", False), ("RecursiveCheckpoint", False), ("Backsolve", True)],
)
def test_adjoint_gradients_match_the_analytic_derivative(adjoint, through_args):
    # y = exp(-k t), so d/dk sum y = -sum t exp(-k t). An exact anchor, so
    # adjoints sharing a bug would still fail.
    rate = jnp.asarray(TRUE_RATE)
    expected = float(-(GRID * jnp.exp(-rate * GRID)).sum())
    grad = jax.grad(lambda k: solution_sum(k, adjoint, through_args=through_args))(rate)
    assert float(grad) == pytest.approx(expected, rel=1e-4)


def test_backsolve_refuses_a_closed_over_value():
    # An error, not a silently wrong gradient, when someone switches adjoint.
    with pytest.raises(Exception, match="closed-over"):
        jax.grad(lambda k: solution_sum(k, "Backsolve", through_args=False))(jnp.asarray(1.0))


@pytest.mark.parametrize("loss_fn", [masked_mse, masked_mle, bal_mse, bal_mle])
@pytest.mark.parametrize("poison", [jnp.nan, jnp.inf, 1e6], ids=["nan", "inf", "huge"])
def test_masked_cells_reach_neither_value_nor_gradient(loss_fn, poison):
    # The t0 row of every bucket is masked. A diverged solve leaves garbage
    # there, which must not move the loss or put NaN into the optimiser.
    bp = decay_dataset().bucket_payloads[0]
    prediction = jnp.exp(-0.5 * bp.ts)[..., None]
    poisoned = prediction.at[:, 0].set(poison)
    np.testing.assert_allclose(loss_fn(poisoned, bp), loss_fn(prediction, bp), rtol=1e-6)
    grad = jax.grad(lambda p: loss_fn(p, bp))(poisoned)
    assert bool(jnp.isfinite(grad).all())
    assert not grad[:, 0].any()


@pytest.mark.parametrize(
    "bounds,value",
    [((0.0, 5.0), 2.5), ((0.0, None), 3.0), ((None, 2.0), -1.0), (None, 7.0), ((-1.0, 1.0), 0.999)],
    ids=["closed", "lower-only", "upper-only", "open", "near-edge"],
)
def test_parameter_holds_its_physical_value_and_stays_in_bounds(bounds, value):
    parameter = hm.Parameter(value, bounds=bounds)
    assert float(parameter.value) == pytest.approx(value, rel=1e-3)
    # However far the optimiser pushes the latent, the value stays in the box.
    low, high = bounds if bounds is not None else (None, None)
    for push in (-1e3, 1e3):
        pushed = eqx.tree_at(lambda p: p.latent, parameter, parameter.latent + push)
        assert low is None or float(pushed.value) >= low
        assert high is None or float(pushed.value) <= high


def _counting(traces: list[Any]) -> Any:
    def simulate(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
        traces.append(ts.shape)
        return simulate_parameter(predictors, ts, covariates, y0, solver)

    return simulate


def test_prediction_compiles_once_per_bucket_shape():
    traces: list[Any] = []
    data = decay_dataset()
    simulate = _counting(traces)

    def model_with(predictors: Any) -> hm.HybridModel:
        return hm.HybridModel(
            predictors=predictors,
            simulate_fn=simulate,
            state_to_output=identity,
            output_channel_names=CHANNELS,
            solver=SOLVER,
        )

    model_with(parameter_predictors(0.3)).predict(data)
    assert len(traces) == len(data.bucket_payloads) == 2
    # Same shapes and new parameter values reuse the compiled kernels.
    model_with(parameter_predictors(0.3)).predict(data)
    model_with(parameter_predictors(0.5)).predict(data)
    assert len(traces) == 2


def test_a_horizon_change_does_not_recompile_training():
    # The horizon is a traced fraction. One trace per bucket, plus the abstract
    # shape check before training; anything more is a retrace at a phase boundary.
    traces: list[Any] = []
    config = quiet(phases=(hm.TrainingPhase(5, lr=0.1, horizon=0.5), hm.TrainingPhase(5, lr=0.05)))
    make_model(parameter_predictors(), _counting(traces)).fit(
        decay_dataset(), config, key=jr.key(0)
    )
    assert len(traces) <= len(decay_dataset().bucket_payloads) + 1
