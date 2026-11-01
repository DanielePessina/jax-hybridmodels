"""Exact checks a workflow cannot make, each against an oracle outside the framework.

Adjoint gradients against the analytic derivative, masked cells against poison,
losses and metrics against values worked out by hand, bounds against a round
trip, and compiled kernels against a trace counter. None trains for long, and
each guards a failure that would otherwise show up only as a slightly wrong fit
or a slow run.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pandas as pd
import pytest
from _models import (
    CHANNELS,
    SOLVER,
    TRUE_RATE,
    decay_dataset,
    identity,
    make_model,
    network_predictors,
    parameter_predictors,
    quiet,
    simulate_parameter,
)
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.data import (
    BucketPayload,
    Dataset,
    experiment_from_array,
    experiments_from_dataframe,
)
from jaxhybridmodels.losses import (
    bal_mle,
    bal_mse,
    masked_mle,
    masked_mse,
    resolve_loss_fn,
)
from jaxhybridmodels.metrics import _worst_first
from jaxhybridmodels.penalties import (
    _bounded_leaves,
    attach_penalty_state,
    bound_penalty,
    box_grid,
    box_violation,
    clip_ste,
    data_penalty_points,
    length_mask_keep,
    penalty_integral,
    penalty_vector_field,
    soft_logit,
    softclip,
    strip_penalty_state,
    trajectory_saturation_penalty,
)
from jaxhybridmodels.predictors import (
    BoundedPredictor,
    KANPredictor,
    MLPPredictor,
    NeuralNPolynomial,
    reinitialize_pytree_with_key,
    reinitialize_with_key,
)
from jaxhybridmodels.predictors.kan import _scaffold_parts
from jaxhybridmodels.profiles import (
    constant_profile,
    piecewise_linear_profile,
    ramp_profile,
    step_profile,
)
from jaxhybridmodels.rng import fold
from jaxhybridmodels.schedules import annealing_schedule
from jaxhybridmodels.solver import (
    ADJOINT_REGISTRY,
    SOLVER_REGISTRY,
    register_adjoint,
    register_solver,
)
from jaxhybridmodels.trainable import freeze_mask, trainable_mask
from jaxhybridmodels.training.kernels import apply_length_mask, build_score_bucket
from jaxhybridmodels.transforms import (
    BOUND_TRANSFORMS,
    WARPS,
    BoundTransform,
    Warp,
    register_bound_transform,
    register_warp,
)

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


# ---------------------------------------------------------------------------
# Scalers and parameters
# ---------------------------------------------------------------------------


TRANSFORMS = sorted(BOUND_TRANSFORMS)


INF = math.inf


# (id, bounds pair, warp, physical values inside, physical values outside)
BOXES = [
    ("finite", (0.0, 10.0), "linear", [0.5, 3.5, 9.9], [-3.0, 13.0]),
    ("log", (1e-6, 1e2), "log", [1e-5, 1e-2, 50.0], [200.0]),
    ("log10", (1e-6, 1e2), "log10", [1e-5, 1e-2, 50.0], [200.0]),
    ("lower", (0.0, INF), "linear", [1e-3, 0.5, 50.0], [-2.0]),
    ("upper", (-INF, 1.0), "linear", [-50.0, 0.0, 0.99], [3.0]),
    ("open", (-INF, INF), "linear", [-50.0, 0.0, 50.0], []),
]


@pytest.fixture
def restore_registries():
    transforms, warps = dict(BOUND_TRANSFORMS), dict(WARPS)
    yield
    BOUND_TRANSFORMS.clear()
    BOUND_TRANSFORMS.update(transforms)
    WARPS.clear()
    WARPS.update(warps)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("box", BOXES, ids=[b[0] for b in BOXES])
def test_scaler_properties(transform, box):
    _, (low, high), warp, inside, outside = box
    scaler = hm.BoundScaler(((low, high),), transform=transform, warp=warp)

    # Round trip is the identity inside the box.
    x = jnp.asarray(inside)
    np.testing.assert_allclose(scaler.from_latent(scaler.to_latent(x)), x, rtol=1e-4)

    # Any finite latent lands inside the box.
    out = scaler.from_latent(jnp.array([-1e6, -20.0, 0.0, 20.0, 1e6]))
    assert jnp.all(jnp.isfinite(out))
    assert float(out.min()) >= low * (1 - 1e-5) and float(out.max()) <= high * (1 + 1e-5)

    # Outside the box the inverse keeps a finite, non-zero gradient, so a
    # state-derived input that strays does not cut the adjoint.
    if outside:
        grads = jax.vmap(jax.grad(scaler.to_latent))(jnp.asarray(outside))
        assert jnp.all(jnp.isfinite(grads)) and jnp.all(grads != 0.0)

    # Zero latent is the midpoint of the warped box.
    if math.isfinite(low) and math.isfinite(high):
        lo, hi = WARPS[warp].forward(low), WARPS[warp].forward(high)
        midpoint = float(WARPS[warp].inverse((lo + hi) / 2))
        np.testing.assert_allclose(scaler.from_latent(jnp.array([0.0])), [midpoint], rtol=1e-4)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ((270.0, 350.0), ((270.0, 350.0),)),
        ([270, None], ((270.0, INF),)),
        (((270.0, 350.0),), ((270.0, 350.0),)),
        (((0.0, 1.0), (None, 5.0)), ((0.0, 1.0), (-INF, 5.0))),
        ([(0.0, 1.0), (2.0, 3.0)], ((0.0, 1.0), (2.0, 3.0))),
    ],
    ids=["pair", "list-pair", "one-of-many", "pair-of-pairs", "list-of-pairs"],
)
def test_every_bounds_argument_reads_one_pair_or_a_sequence_of_pairs(spec, expected):
    """A pair of numbers or None is one pair; a pair of pairs is per component."""
    assert hm.BoundScaler(spec).bounds == expected
    size = len(expected)
    bounded = hm.BoundedPredictor(
        inner=_Identity(), input_bounds=spec, output_bounds=spec, input_keys=("a", "b")[:size]
    )
    assert bounded.in_scaler.bounds == bounded.out_scaler.bounds == expected
    inside = [
        (lo + hi) / 2 if math.isfinite(hi - lo) else lo + 1 if math.isfinite(lo) else hi - 1
        for lo, hi in expected
    ]
    assert hm.Parameter(inside, bounds=spec).scaler.bounds == expected


class _Identity(hm.Predictor):
    def __call__(self, x):
        return x


def test_polynomial_tails_keep_gradient_where_sigmoid_dies():
    def tail_gradient(name, z):
        scaler = hm.BoundScaler(((0.0, 10.0),), transform=name)
        return float(jax.grad(lambda v: scaler.from_latent(v)[0])(jnp.array([z]))[0])

    assert tail_gradient("sigmoid", 25.0) == 0.0
    assert tail_gradient("algebraic", 25.0) > 0.0 and tail_gradient("softsign", 25.0) > 0.0
    g = {name: tail_gradient(name, 12.0) for name in ("sigmoid", "algebraic", "softsign")}
    assert g["sigmoid"] < g["algebraic"] < g["softsign"]


def test_saturation_gradient_survives_where_the_physical_gradient_underflows():
    scaler = hm.BoundScaler(((0.0, 10.0),))
    z = jnp.array([40.0])
    assert float(jax.grad(lambda v: scaler.saturation(v))(z)[0]) > 0.0
    assert float(jax.grad(lambda v: scaler.from_latent(v)[0])(z)[0]) == 0.0
    assert float(scaler.saturation(jnp.array([0.0, 1.0, -2.0]))) == 0.0


def test_input_violation_is_a_width_normalised_hinge():
    scaler = hm.BoundScaler(((0.0, 10.0), (-1.0, 1.0)))
    assert float(scaler.input_violation(jnp.array([5.0, 0.0]))) == 0.0
    near = scaler.input_violation(jnp.array([11.0, 0.0]))
    far = scaler.input_violation(jnp.array([20.0, 0.0]))
    assert 0.0 < float(near) < float(far)
    assert float(jax.grad(lambda x: scaler.input_violation(x))(jnp.array([13.0, 0.0]))[0]) > 0.0
    narrow = hm.BoundScaler(((0.0, 1.0),)).input_violation(jnp.array([1.5]))
    wide = hm.BoundScaler(((0.0, 1000.0),)).input_violation(jnp.array([1500.0]))
    np.testing.assert_allclose(narrow, wide)
    # A one-sided box has no width, so the overshoot is not normalised.
    assert float(hm.BoundScaler(((0.0, None),)).input_violation(jnp.array([-2.0]))) == 4.0


def test_array_parameter_takes_one_bound_per_element():
    parameter = hm.Parameter(
        [[0.2, 0.4], [-2.0, 0.8]], bounds=[(0.0, None), (None, 1.0), (None, None), (0.0, 1.0)]
    )
    assert parameter.value.shape == (2, 2)
    np.testing.assert_allclose(parameter.value, [[0.2, 0.4], [-2.0, 0.8]], rtol=1e-5)
    assert jnp.isfinite(eqx.filter_grad(lambda p: p.value.sum())(parameter).latent).all()
    unbounded = hm.Parameter([-100.0, 100.0])
    assert float(unbounded.scaler.saturation(unbounded.latent)) == 0.0


def test_a_registered_transform_and_warp_work_end_to_end(restore_registries):
    # arctan, which the shipped set omits as dominated by softsign.
    register_bound_transform(
        "arctan",
        BoundTransform(
            forward=lambda z: 0.5 + jnp.arctan(z) / jnp.pi,
            inverse=lambda s: jnp.tan(jnp.pi * (s - 0.5)),
            inverse_slope=lambda s: jnp.pi / jnp.cos(jnp.pi * (s - 0.5)) ** 2,
            knee=6.313752,
        ),
    )
    # Square root: more resolution toward the low end, and zero is allowed.
    register_warp(
        "sqrt",
        Warp(forward=jnp.sqrt, inverse=lambda w: jnp.asarray(w) ** 2, requires_positive=False),
    )
    parameter = hm.Parameter(9.0, bounds=(0.0, 100.0), transform="arctan", warp="sqrt")
    np.testing.assert_allclose(parameter.value, 9.0, rtol=1e-4)
    scaler = parameter.scaler
    assert float(scaler.from_latent(jnp.array([0.0]))[0]) == pytest.approx(25.0, rel=1e-3)
    knee_value = float(scaler.from_latent(jnp.array([scaler.z_knee]))[0])
    assert knee_value == pytest.approx(9.5**2, rel=1e-2)


# ---------------------------------------------------------------------------
# Configuration, schedules, profiles and solvers
# ---------------------------------------------------------------------------


_TSIT5 = diffrax.Tsit5()


def _solver(
    *,
    solver: diffrax.AbstractSolver = _TSIT5,
    rtol: float = 1e-4,
    atol: float | tuple[float, ...] = 1e-6,
    max_steps: int = 100,
    dt0: float | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
) -> hm.SolverConfig:
    return hm.SolverConfig(
        solver=solver,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        dt0=dt0,
        adjoint=adjoint,
        pcoeff=pcoeff,
        icoeff=icoeff,
    )


class _Unregistered(diffrax.Heun):
    pass


class _UnregisteredAdjoint(diffrax.DirectAdjoint):
    pass


@pytest.mark.parametrize(
    "config",
    [
        _solver(max_steps=500_000),
        _solver(solver=diffrax.Dopri5(), atol=(1e-5, 1e-6, 1e-7), dt0=0.01),
        _solver(
            solver=diffrax.Kvaerno3(),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            pcoeff=0.4,
            icoeff=0.3,
        ),
    ],
    ids=["tsit5", "dopri5-tuple-atol", "kvaerno3-recursive-pid"],
)
def test_solver_config_round_trips_through_json(config):
    # SolverConfig is static configuration: no array leaves, so changing it
    # recompiles rather than reusing a stale kernel.
    assert jax.tree.leaves(eqx.filter(config, eqx.is_array)) == []
    assert jax.tree.leaves(config) == []
    restored = hm.SolverConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert type(restored.solver) is type(config.solver)
    assert type(restored.adjoint) is type(config.adjoint)
    for field in ("rtol", "atol", "max_steps", "dt0", "pcoeff", "icoeff", "dcoeff"):
        assert getattr(restored, field) == getattr(config, field)


def test_registering_a_solver_and_adjoint_makes_them_serialisable(monkeypatch):
    monkeypatch.setattr("jaxhybridmodels.solver.SOLVER_REGISTRY", dict(SOLVER_REGISTRY))
    monkeypatch.setattr("jaxhybridmodels.solver.ADJOINT_REGISTRY", dict(ADJOINT_REGISTRY))
    register_solver("Custom", _Unregistered)
    register_adjoint("CustomAdjoint", _UnregisteredAdjoint)
    config = _solver(solver=_Unregistered(), adjoint=_UnregisteredAdjoint())
    data = config.to_dict()
    assert (data["solver"], data["adjoint"]) == ("Custom", "CustomAdjoint")
    restored = hm.SolverConfig.from_dict(data)
    assert isinstance(restored.solver, _Unregistered)
    assert isinstance(restored.adjoint, _UnregisteredAdjoint)


@pytest.mark.parametrize(
    "kind, kwargs, values",
    [
        ("cosine", {}, {0: 1.0, 50: 0.5, 100: 0.0}),
        ("cosine", {"end_value": 0.1}, {0: 1.0, 100: 0.1}),
        ("linear", {"end_value": 0.2}, {0: 1.0, 50: 0.6, 100: 0.2, 500: 0.2}),
        ("warmup_cosine", {"warmup_epochs": 20}, {0: 0.0, 20: 1.0, 100: 0.0}),
        # Decay spans the remaining 50 steps, so the midpoint of the decay is step 75.
        ("warmup_cosine", {"warmup_epochs": 50}, {50: 1.0, 75: 0.5, 99: 0.0, 100: 0.0}),
        ("exponential", {"end_value": 0.01}, {0: 1.0, 100: 0.01, 500: 0.01}),
        ("exponential", {"end_value": 0.0}, {0: 1.0, 100: 0.0, 500: 0.0}),
    ],
)
def test_annealing_schedule_values(kind, kwargs, values):
    # Steps past total_epochs hold the end value, so a loop that overruns
    # its budget cannot drive the multiplier negative.
    schedule = annealing_schedule(kind, total_epochs=100, **kwargs)
    jitted = jax.jit(schedule)
    for step, expected in values.items():
        assert float(schedule(step)) == pytest.approx(expected, abs=1e-2)
        assert float(jitted(step)) == pytest.approx(expected, abs=1e-2)


PROFILES = {
    "constant": (constant_profile(3.5), {-2.0: 3.5, 9.0: 3.5}),
    "step": (step_profile(before=1.0, after=4.0, jump_at=2.0), {1.9: 1.0, 2.0: 4.0, 5.0: 4.0}),
    "ramp-up": (
        ramp_profile(t0=1.0, t1=3.0, v0=10.0, v1=30.0),
        {0.0: 10.0, 1.0: 10.0, 2.0: 20.0, 3.0: 30.0, 7.0: 30.0},
    ),
    "ramp-down": (ramp_profile(t0=1.0, t1=3.0, v0=30.0, v1=10.0), {0.0: 30.0, 2.0: 20.0}),
    "piecewise": (
        piecewise_linear_profile(((0.0, 0.0), (1.0, 2.0), (3.0, 6.0))),
        {-1.0: 0.0, 0.5: 1.0, 2.0: 4.0, 4.0: 6.0},
    ),
}


@pytest.mark.parametrize("name", PROFILES)
def test_profiles_evaluate_inside_traced_vector_fields(name):
    profile, values = PROFILES[name]
    ts = jnp.asarray(list(values), dtype=jnp.float32)
    expected = list(values.values())
    # diffrax calls the profile with a traced scalar t; vmap under jit is
    # the same trace, batched. A Python float must work on the host too.
    np.testing.assert_allclose(jax.jit(jax.vmap(profile))(ts), expected, rtol=1e-6)
    t, value = next(iter(values.items()))
    out = profile(t)
    assert isinstance(out, Array) and out.dtype == jnp.float32
    assert float(out) == pytest.approx(value)
    assert jnp.isfinite(jax.grad(profile)(jnp.asarray(2.5)))


def test_profile_parameters_can_be_traced_covariates():
    # Per-experiment ramp parameters arrive as traced covariates, and the
    # host-side knot check must step aside instead of raising.
    def ramp_at(t, lo, hi, t0, t1):
        return ramp_profile(t0=t0, t1=t1, v0=lo, v1=hi)(t)

    covariates = jnp.array([[10.0, 30.0, 0.0, 4.0], [20.0, 40.0, 1.0, 5.0]])
    at_2 = jax.vmap(lambda c: ramp_at(2.0, *c))(covariates)
    np.testing.assert_allclose(at_2, [20.0, 25.0])

    def piecewise_at_zero(offset):
        knots = ((offset, 0.0), (1.0 + offset, 1.0), (2.0 + offset, 2.0))
        return piecewise_linear_profile(knots)(jnp.asarray(0.0))

    assert jnp.isfinite(jax.jit(piecewise_at_zero)(jnp.asarray(0.5)))


def test_rng_fold_is_named_stable_and_pinned():
    root = jr.PRNGKey(0)
    # Pinned bytes: a change here silently changes every seeded run.
    np.testing.assert_array_equal(
        np.asarray(fold(root, "init")), np.array([3020537540, 2946840664], dtype=np.uint32)
    )
    names = ("init", "tournament", "phase_0", "evosax_ask_0")
    forward = [fold(root, n) for n in names]
    backward = [fold(root, n) for n in reversed(names)][::-1]
    for a, b in zip(forward, backward, strict=True):
        np.testing.assert_array_equal(a, b)
    assert len({tuple(np.asarray(k).tolist()) for k in forward}) == len(names)
    assert not jnp.array_equal(fold(jr.PRNGKey(1), "init"), forward[0])


def test_python_lr_schedule_is_evaluated_on_the_host():
    """A schedule that branches on the step still gives one lr per step."""
    from jaxhybridmodels.training.optax import _scheduled_lrs

    def stepwise(step: Any) -> float:
        return 0.1 if step < 2 else 0.01

    lrs = _scheduled_lrs(hm.TrainingPhase(4, lr=stepwise))
    np.testing.assert_allclose(np.asarray(lrs), [0.1, 0.1, 0.01, 0.01], rtol=1e-6)


# ---------------------------------------------------------------------------
# Data, splits, metrics and losses
# ---------------------------------------------------------------------------


def _obs(ts, values, **kwargs) -> hm.ChannelObs:
    return hm.ChannelObs(ts=jnp.asarray(ts), values=jnp.asarray(values), **kwargs)


def _exp(exp_id: str = "e", covariates: Any = None, **channels) -> hm.Experiment:
    """One experiment with a single-state ``y0``; channels as ``name=(ts, values)``."""
    return hm.make_experiment(
        channels={name: _obs(*spec) for name, spec in channels.items()},
        covariates={"a": 1.0} if covariates is None else covariates,
        y0=[0.0],
        exp_id=exp_id,
    )


def _two_channel() -> Dataset:
    """Two experiments in one bucket: ``x`` dense, ``y`` sparse in the second."""
    full = _exp("full", x=([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]), y=([0.0, 1.0, 2.0], [5.0, 6.0, 7.0]))
    sparse = _exp("sparse", x=([0.0, 1.0, 2.0], [2.0, 1.0, 0.5]), y=([1.0], [4.0]))
    return hm.make_dataset([full, sparse], output_channel_names=("x", "y"))


def _ids(ds: Dataset) -> list[str]:
    return [e.exp_id for e in ds.experiments]


def _analytic(bp: BucketPayload, rate: float) -> jax.Array:
    """Decay from ``y0 = 1`` on the bucket's grid, shaped ``[N, T, 1]``."""
    return jnp.exp(-rate * bp.ts)[..., None]


def test_sparse_channels_scatter_onto_a_sorted_union_grid():
    exp = hm.make_experiment(
        channels={
            "x": _obs([2.0, 0.0], [20.0, 10.0]),
            "y": _obs([1.0, 3.0], [100.0, 300.0], variance=jnp.array([0.1, 0.2])),
        },
        y0=[0.0],
    )
    bp = hm.make_dataset([exp], output_channel_names=("x", "y")).bucket_payloads[0]
    np.testing.assert_allclose(bp.ts[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        bp.mask[0], [[True, False], [False, True], [True, False], [False, True]]
    )
    observed = np.where(bp.mask[0], bp.y_observed[0], np.nan)
    np.testing.assert_allclose(
        observed, [[10.0, np.nan], [np.nan, 100.0], [20.0, np.nan], [np.nan, 300.0]]
    )
    # Default variance is one; an array variance follows its timestamps.
    assert float(bp.yvar[0, 0, 0]) == 1.0
    assert float(bp.yvar[0, 1, 1]) == pytest.approx(0.1)
    assert float(bp.yvar[0, 3, 1]) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "ts_dtype, values_dtype, variance_dtype",
    [
        (jnp.float32, jnp.float32, jnp.float32),
        (jnp.float16, jnp.float16, jnp.float16),
        (jnp.float32, jnp.float16, jnp.float32),
    ],
)
def test_dtypes_survive_bucketing(ts_dtype, values_dtype, variance_dtype):
    channel = hm.ChannelObs(
        ts=jnp.asarray([0.0, 1.0], dtype=ts_dtype),
        values=jnp.asarray([1.0, 2.0], dtype=values_dtype),
        variance=jnp.asarray([0.1, 0.2], dtype=variance_dtype),
    )
    exp = hm.make_experiment(channels={"x": channel}, y0=[0.0])
    bp = hm.make_dataset([exp], output_channel_names=("x",)).bucket_payloads[0]
    assert (bp.ts.dtype, bp.y_observed.dtype, bp.yvar.dtype) == (
        ts_dtype,
        values_dtype,
        variance_dtype,
    )


def test_equal_timestamps_in_mixed_float_dtypes_share_one_row():
    # 0.1 has different exact values in float32 and float64; the union grid
    # must canonicalise before using timestamps as set keys.
    with jax.enable_x64():
        exp = hm.make_experiment(
            channels={
                "x": _obs(jnp.asarray([0.1], dtype=jnp.float32), [1.0]),
                "y": _obs(jnp.asarray([0.1], dtype=jnp.float64), [2.0]),
            },
            y0=[0.0],
        )
        bp = hm.make_dataset([exp], output_channel_names=("x", "y")).bucket_payloads[0]
    assert bp.ts.shape == (1, 1)
    assert bp.mask.tolist() == [[[True, True]]]


def test_experiment_ids_are_unique_and_empty_ones_are_named_by_position():
    named = _exp("kept", x=([0.0], [1.0]))
    anonymous = [_exp("", x=([0.0, 1.0], [1.0, 2.0])) for _ in range(2)]
    ds = hm.make_dataset([anonymous[0], named, anonymous[1]], output_channel_names=("x",))
    assert _ids(ds) == ["experiment_0", "kept", "experiment_2"]
    with pytest.raises(ValueError, match=r"duplicate exp_id values \['kept'\]"):
        hm.make_dataset([named, named], output_channel_names=("x",))


def test_experiment_from_array_drops_nan_cells_per_channel():
    exp = experiment_from_array(
        ts=[0.0, 1.0, 2.0],
        values=[[1.0, np.nan], [2.0, 20.0], [np.nan, 30.0]],
        channel_names=("x", "y"),
        variance=(0.1, 0.2),
        covariates={"a": 1.0},
        y0=[0.0],
        exp_id="run",
    )
    np.testing.assert_allclose(exp.channels["x"].ts, [0.0, 1.0])
    np.testing.assert_allclose(exp.channels["y"].values, [20.0, 30.0])
    np.testing.assert_allclose(exp.channels["y"].variance, [0.2, 0.2])
    assert exp.exp_id == "run"
    # A [T, D] variance is read cell by cell; one channel may be rank-1.
    single = experiment_from_array(
        ts=[0.0, 1.0],
        values=[1.0, 2.0],
        channel_names=("x",),
        variance=[[0.5], [0.25]],
        y0=[0.0],
        exp_id="one",
    )
    np.testing.assert_allclose(single.channels["x"].variance, [0.5, 0.25])


def _lab_table() -> pd.DataFrame:
    """Two runs in the shape of a lab spreadsheet: one row per sample time."""
    return pd.DataFrame(
        {
            "Exp": [7, 7, 7, 3, 3],
            "Time": [0.0, 30.0, 60.0, 0.0, 45.0],
            "Temp": [21.0, 21.0, 21.0, 25.0, 25.0],
            "Conc": [19.1, 18.4, 13.5, 12.3, 11.9],
            "Conc_var": [0.03, 0.16, 2.4, 0.06, 0.07],
            "PS": [np.nan, np.nan, 13.0, np.nan, 15.0],
        }
    )


def test_experiments_from_dataframe_builds_one_experiment_per_run():
    experiments = experiments_from_dataframe(
        _lab_table(),
        time="Time",
        experiment="Exp",
        channels={"conc": "Conc", "d43": "PS"},
        covariates={"temperature": "Temp"},
        variance={"conc": "Conc_var", "d43": 0.5},
        t0=0.0,
        y0_fn=lambda cov, ch: jnp.array([ch["conc"].values[0], cov["temperature"]]),
    )
    # First-appearance order, IDs as strings.
    assert [e.exp_id for e in experiments] == ["7", "3"]
    first = experiments[0]
    np.testing.assert_allclose(first.channels["conc"].variance, [0.03, 0.16, 2.4], rtol=1e-6)
    np.testing.assert_allclose(first.channels["d43"].ts, [60.0])
    np.testing.assert_allclose(first.channels["d43"].variance, [0.5])
    np.testing.assert_allclose(first.y0, [19.1, 21.0], rtol=1e-6)
    assert float(experiments[1].covariates["temperature"]) == 25.0
    ds = hm.make_dataset(experiments, output_channel_names=("conc", "d43"))
    assert int(sum(int(bp.n_obs) for bp in ds.bucket_payloads)) == 7

    # A list names channels after their columns; y0 may be keyed by run.
    plain = experiments_from_dataframe(
        _lab_table(),
        time="Time",
        experiment="Exp",
        channels=["Conc"],
        y0={"7": [1.0], "3": [2.0]},
    )
    assert list(plain[1].channels) == ["Conc"] and float(plain[1].y0[0]) == 2.0
    np.testing.assert_allclose(plain[0].channels["Conc"].variance, 1.0)


def _numbered(n: int) -> Dataset:
    """``n`` experiments with grids of 2, 3 and 4 points in rotation."""
    experiments = []
    for i in range(n):
        ts = jnp.linspace(0.0, 1.0, 2 + i % 3)
        experiments.append(_exp(f"e{i}", {"a": float(i)}, c=(ts, ts * i)))
    return hm.make_dataset(experiments, output_channel_names=("c",))


@pytest.mark.parametrize(
    "n, fractions, counts",
    [
        (10, {}, (8, 1, 1)),
        # Floor for train and val; the remainder goes to test.
        (13, {"train": 0.7, "val": 0.2, "test": 0.1}, (9, 2, 2)),
        (17, {}, (13, 1, 3)),
        # An explicit zero stays zero, even when flooring leaves a remainder.
        (10, {"train": 1.0, "val": 0.0, "test": 0.0}, (10, 0, 0)),
        (4, {"train": 0.7, "val": 0.3, "test": 0.0}, (3, 1, 0)),
    ],
)
def test_split_partitions_experiments(n, fractions, counts):
    source = _numbered(n)
    splits = hm.split_dataset(source, key=jr.key(0), **fractions)
    assert tuple(len(s.experiments) for s in splits) == counts
    ids = [set(_ids(s)) for s in splits]
    assert set().union(*ids) == {f"e{i}" for i in range(n)}
    assert sum(len(i) for i in ids) == n
    for split, count in zip(splits, counts, strict=True):
        assert split.output_channel_names == source.output_channel_names
        if count == 0:
            assert split.bucket_payloads == ()
            continue
        # Re-bucketed from the split's own experiments.
        assert sum(bp.ts.shape[0] for bp in split.bucket_payloads) == count
        lengths = [bp.ts.shape[1] for bp in split.bucket_payloads]
        assert len(lengths) == len(set(lengths))


def _two_channel_scored() -> tuple[jax.Array, BucketPayload]:
    bp = _two_channel().bucket_payloads[0]
    offsets = jnp.arange(bp.y_observed.size, dtype=jnp.float32).reshape(bp.y_observed.shape)
    return bp.y_observed + 0.1 * offsets, bp


def test_channel_idx_and_channel_weights_compose_linearly():
    pred, bp = _two_channel_scored()
    mask = bp.mask
    # Same numerator, different denominators: channel_idx divides by the
    # selected channel's count, channel_weights by the full mask count.
    via_idx = masked_mse(pred, bp, channel_idx=(0,))
    via_weights = masked_mse(pred, bp, channel_weights=(1.0, 0.0))
    np.testing.assert_allclose(via_idx * mask[..., 0].sum(), via_weights * mask.sum(), rtol=1e-6)
    # Weighted masked_mse is homogeneous and additive over channels.
    one = masked_mse(pred, bp, channel_weights=(1.0, 1.0))
    np.testing.assert_allclose(masked_mse(pred, bp, channel_weights=(3.0, 3.0)), 3 * one, rtol=1e-6)
    parts = [masked_mse(pred, bp, channel_weights=w) for w in [(1.0, 0.0), (0.0, 1.0)]]
    both = masked_mse(pred, bp, channel_weights=(2.0, 1.0))
    np.testing.assert_allclose(both, 2 * parts[0] + parts[1], rtol=1e-6)
    # bal_mse weights are linear in the per-channel losses.
    single = [bal_mse(pred, bp, channel_idx=(i,)) for i in (0, 1)]
    weighted = bal_mse(pred, bp, channel_weights=(2.0, 1.0))
    np.testing.assert_allclose(weighted, 2 * single[0] + single[1], rtol=1e-6)


@pytest.mark.parametrize("loss_fn", [bal_mse, bal_mle])
def test_balanced_losses_weight_experiments_not_observations(loss_fn, decay_data):
    base = decay_data
    copies = [
        exp
        if k == 0
        else hm.make_experiment(
            channels=exp.channels,
            covariates=exp.covariates,
            y0=exp.y0,
            t0=exp.t0,
            exp_id=f"{exp.exp_id}-copy",
        )
        for k in range(2)
        for exp in base.experiments
    ]
    doubled = hm.make_dataset(copies, output_channel_names=("conc",))
    for bp, bp2 in zip(base.bucket_payloads, doubled.bucket_payloads, strict=True):
        np.testing.assert_allclose(
            loss_fn(_analytic(bp, 0.5), bp), loss_fn(_analytic(bp2, 0.5), bp2), rtol=1e-6
        )

        # An all-masked probe row adds no data and must not dilute the mean.
        def extend(a):
            return jnp.concatenate([a, a[:1]])

        probe = bp._replace(
            ts=extend(bp.ts),
            y_observed=extend(bp.y_observed),
            yvar=extend(bp.yvar),
            mask=extend(bp.mask).at[-1].set(False),
            covariates={k: extend(v) for k, v in bp.covariates.items()},
            y0=extend(bp.y0),
        )
        pred = jnp.concatenate([_analytic(bp, 0.5), jnp.full_like(_analytic(bp, 0.5)[:1], 99.0)])
        np.testing.assert_allclose(loss_fn(pred, probe), loss_fn(_analytic(bp, 0.5), bp), rtol=1e-6)


def test_likelihood_losses_read_the_observation_variance():
    pred, bp = _two_channel_scored()
    low = bp._replace(yvar=jnp.full_like(bp.yvar, 0.5))
    high = bp._replace(yvar=jnp.full_like(bp.yvar, 1.0))
    # Gaussian NLL: doubling the variance adds n log(2) / 2 and halves the
    # squared-error term.
    se = jnp.where(bp.mask, (pred - bp.y_observed) ** 2, 0.0).sum()
    expected = 0.5 * bp.mask.sum() * jnp.log(2.0) - 0.5 * se
    np.testing.assert_allclose(masked_mle(pred, high) - masked_mle(pred, low), expected, atol=1e-4)
    assert not jnp.allclose(bal_mle(pred, low), bal_mle(pred, high))


def test_resolve_loss_fn_binds_channel_arguments():
    pred, bp = _two_channel_scored()
    big = pred.at[..., 1].add(100.0)
    # Registry losses take the channel arguments directly.
    assert float(resolve_loss_fn("mse", (0,), None)(big, bp)) == pytest.approx(
        float(masked_mse(pred, bp, channel_idx=(0,)))
    )
    doubled = resolve_loss_fn("mse", None, (2.0, 2.0))(pred, bp)
    np.testing.assert_allclose(doubled, 2 * resolve_loss_fn("mse", None, (1.0, 1.0))(pred, bp))

    # A custom loss without channel arguments passes through untouched.
    def plain(pred_obs, bp):
        return jnp.sum(jnp.where(bp.mask, (pred_obs - bp.y_observed) ** 2, 0.0))

    assert resolve_loss_fn(plain, None, None) is plain
    # With channel_idx the channels are sliced out before the call.
    assert float(resolve_loss_fn(plain, (0,), None)(big, bp)) == pytest.approx(
        float(
            plain(
                pred[..., :1], bp._replace(y_observed=bp.y_observed[..., :1], mask=bp.mask[..., :1])
            )
        )
    )

    # A custom loss that declares the keywords receives them.
    seen = {}

    def keyword(pred_obs, bp, *, channel_idx=None, channel_weights=None):
        seen.update(channel_idx=channel_idx, channel_weights=channel_weights)
        return jnp.asarray(0.0)

    resolve_loss_fn(keyword, (1,), (0.5,))(pred, bp)
    assert seen == {"channel_idx": (1,), "channel_weights": (0.5,)}


# ---------------------------------------------------------------------------
# Predictors and masks
# ---------------------------------------------------------------------------

IN, OUT = 2, 1


def mlp(key: Array) -> MLPPredictor:
    return MLPPredictor(in_size=IN, out_size=OUT, width_size=4, depth=2, key=key)


def kan(key: Array) -> KANPredictor:
    return KANPredictor(in_size=IN, out_size=OUT, hidden_widths=(3,), grid_size=3, key=key)


def npoly(key: Array) -> NeuralNPolynomial:
    exponents = (0.0, 1.0, 2.0)
    coeff_net = MLPPredictor(
        in_size=IN, out_size=OUT * len(exponents), width_size=4, depth=1, key=key
    )
    return NeuralNPolynomial(coeff_net=coeff_net, exponents=exponents, in_size=IN, out_size=OUT)


FAMILIES: dict[str, Callable[[Array], hm.Predictor]] = {"mlp": mlp, "kan": kan, "npoly": npoly}


family = pytest.mark.parametrize("build", FAMILIES.values(), ids=FAMILIES.keys())


def arrays(module: eqx.Module) -> list[Array]:
    return jtu.tree_leaves(eqx.filter(module, eqx.is_array))


def static_part(module: eqx.Module) -> eqx.Module:
    return eqx.filter(module, eqx.is_array, inverse=True)


@family
def test_restart_keeps_the_architecture_and_redraws_the_weights(build):
    predictor = build(jr.key(0))
    fresh = predictor.initialized_with_key(jr.key(11))
    # Same architecture: same type and leaf shapes. (KAN derives a new
    # static seed from the key, so the static fields need not match.)
    assert type(fresh) is type(predictor)
    assert [a.shape for a in arrays(fresh)] == [a.shape for a in arrays(predictor)]
    assert any(
        not jnp.array_equal(a, b) for a, b in zip(arrays(predictor), arrays(fresh), strict=True)
    )
    again = predictor.initialized_with_key(jr.key(11))
    assert all(jnp.array_equal(a, b) for a, b in zip(arrays(fresh), arrays(again), strict=True))


@pytest.mark.parametrize("build", [mlp, kan], ids=["mlp", "kan"])
def test_zero_final_head_zeroes_only_the_readout(build):
    predictor = build(jr.key(0))
    zeroed = predictor.with_zero_final_head()
    for x in (jnp.zeros(IN), jnp.array([0.5, -1.7])):
        assert jnp.array_equal(zeroed(x), jnp.zeros(OUT))
    assert eqx.tree_equal(static_part(zeroed), static_part(predictor))
    before, after = arrays(predictor), arrays(zeroed)
    changed = [not jnp.array_equal(a, b) for a, b in zip(before, after, strict=True)]
    # Some leaves were zeroed, the rest (the hidden layers) are untouched,
    # and the input module was not mutated.
    assert any(changed) and not all(changed)
    assert all(
        jnp.array_equal(a, jnp.zeros_like(a)) for a, c in zip(after, changed, strict=True) if c
    )
    assert not any(
        jnp.array_equal(a, jnp.zeros_like(a)) for a, c in zip(before, changed, strict=True) if c
    )


class FixedCoefficients(hm.Predictor):
    """Returns the same coefficients for any input, isolating the polynomial."""

    coefficients: Array

    def __call__(self, x: Array) -> Array:
        return self.coefficients


def test_npoly_evaluates_its_polynomial():
    exponents = (0.0, 1.0, 2.0)
    coefficients = jnp.arange(6.0)
    poly = NeuralNPolynomial(
        coeff_net=FixedCoefficients(coefficients), exponents=exponents, in_size=3, out_size=2
    )
    x = jnp.array([0.5, 1.5, 2.0])
    expected = coefficients.reshape(2, 3) @ (jnp.sum(x) ** jnp.asarray(exponents))
    np.testing.assert_allclose(poly(x), expected, rtol=1e-6)
    # 0 ** 0 at a zero basis must stay finite.
    assert jnp.all(jnp.isfinite(poly(jnp.zeros(3))))


class Identity(hm.Predictor):
    """Latent in, latent out, so the bounded output equals the input."""

    def __call__(self, x: Array) -> Array:
        return x


def test_open_input_bounds_pass_inputs_through_unscaled():
    bp = BoundedPredictor(
        inner=Identity(),
        input_keys=("a", "b"),
        input_bounds=((0.0, None), (None, None)),
        output_bounds=((None, None),) * 2,
    )
    np.testing.assert_allclose(bp({"a": -3.0, "b": 4.0}), [-3.0, 4.0])


class Linear(hm.Predictor):
    """No ``initialized_with_key``, so restarts take the generic leaf branch."""

    weights: Array
    counter: Array

    def __call__(self, x: Array) -> Array:
        return x @ self.weights


def test_generic_restart_redraws_floats_and_keeps_integers():
    p = Linear(weights=jnp.ones((3, 2)), counter=jnp.asarray(5, dtype=jnp.int32))
    a = reinitialize_with_key(p, jr.key(0))
    assert isinstance(a, Linear)
    assert not jnp.allclose(a.weights, p.weights)
    assert jnp.array_equal(a.counter, p.counter) and a.counter.dtype == jnp.int32
    assert jnp.array_equal(reinitialize_with_key(p, jr.key(0)).weights, a.weights)


def test_kan_scaffold_warmed_inside_jit_survives_a_second_trace():
    """The cached scaffold must hold concrete arrays, not tracers.

    ``_scaffold_parts`` memoises the jaxkan scaffold per architecture. The
    first call usually happens inside ``jit``; built carelessly, the cache
    then stores that trace's tracers and the next trace dies with
    ``UnexpectedTracerError``.
    """
    _scaffold_parts.cache_clear()
    predictor = kan(jr.key(0))
    x = jnp.array([0.3, -0.2])
    # Two distinct jitted callables, so two traces.
    first = eqx.filter_jit(lambda p, v: p(v))(predictor, x)
    second = eqx.filter_jit(lambda p, v: 2.0 * p(v))(predictor, x)
    np.testing.assert_allclose(second, 2.0 * first, rtol=1e-6)
    _, rest = _scaffold_parts(IN, OUT, (3,), 3, predictor.basis, predictor.seed)
    leaves = [leaf for leaf in jtu.tree_leaves(rest) if hasattr(leaf, "shape")]
    assert leaves and not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _model_predictors() -> dict[str, eqx.Module]:
    return {
        "rate": BoundedPredictor(inner=mlp(jr.key(0)), input_keys=("a", "b"), output_bounds=(0, 1)),
        "offset": hm.Parameter(0.5, bounds=(0.0, 1.0)),
        "gain": hm.Parameter(2.0, bounds=(0.0, None)),
        "fixed": hm.Constant(1.0),
    }


def _flags(tree: object) -> list[bool]:
    return [bool(leaf) for leaf in jtu.tree_leaves(tree)]


def test_freeze_mask_takes_paths_classes_and_predicates():
    predictors = _model_predictors()
    base = hm.trainable_mask(predictors)
    assert any(_flags(base["rate"].inner)) and base["offset"].latent and base["gain"].latent

    # A path may stop at a subtree: "rate" freezes the whole network.
    by_path = freeze_mask(predictors, ("rate",))
    assert not any(_flags(by_path["rate"])) and by_path["offset"].latent
    # A class freezes every instance, and a bare entry needs no tuple.
    by_class = freeze_mask(predictors, hm.Parameter)
    assert not by_class["offset"].latent and not by_class["gain"].latent
    assert _flags(by_class["rate"]) == _flags(base["rate"])
    # Predicates see the real modules, so they may read values.
    by_value = freeze_mask(
        predictors, (lambda m: isinstance(m, hm.Parameter) and float(m.value) > 1.0,)
    )
    assert by_value["offset"].latent and not by_value["gain"].latent
    # Entries compose, on top of an explicit starting mask if given.
    both = freeze_mask(predictors, ("offset", MLPPredictor), mask=by_value)
    assert not any(_flags(both))
    # The input mask is never mutated.
    assert base["offset"].latent and any(_flags(base["rate"].inner))


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def saturate(rate: hm.BoundedPredictor, scale: float = 50.0) -> hm.BoundedPredictor:
    """Scale the readout weights so the output squash is pinned against its bound."""
    return eqx.tree_at(
        lambda r: r.inner.mlp.layers[-1].weight, rate, rate.inner.mlp.layers[-1].weight * scale
    )


def grid_penalty(predictors: Any) -> float:
    """Bound penalty on a sweep of each leaf's input box, for before/after comparisons."""
    points = tuple(box_grid(leaf.in_scaler, 5) for leaf in _bounded_leaves(predictors))
    return float(bound_penalty(predictors, points))


class Latent(hm.Predictor):
    """One trainable latent ``c``, ignoring its input.

    Restarts keep ``c``, so "training moved c" cannot be satisfied by a
    tournament re-draw alone.
    """

    c: Array

    def __init__(self, c: float) -> None:
        self.c = jnp.asarray(c, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:
        return self.c

    def initialized_with_key(self, key: Array) -> Latent:
        return self


def latent_predictors(c: float) -> tuple[hm.BoundedPredictor]:
    return (
        hm.BoundedPredictor(
            inner=Latent(c), input_keys=("u",), input_bounds=(-3.0, 3.0), output_bounds=(0.0, 1.0)
        ),
    )


def test_penalty_state_helpers():
    y0 = jnp.array([1.0, 2.0])
    full = jnp.stack([attach_penalty_state(y0, 2), jnp.array([1.0, 2.0, 5.0, 6.0])])
    assert full[0].tolist() == [1.0, 2.0, 0.0, 0.0]
    assert strip_penalty_state(full, 2).shape == (2, 2)
    assert penalty_integral(full, 2).tolist() == [5.0, 6.0]

    # The base field sees only the physical state; the rates are appended.
    seen = []

    def base(t, y, args):
        seen.append(y.shape)
        return -y

    vf = penalty_vector_field(base, lambda t, y, args: jnp.array([y[0] ** 2, 7.0]))
    assert vf(0.0, jnp.array([3.0, 0.0, 0.0]), None).tolist() == [-3.0, 9.0, 7.0]
    assert seen == [(1,)]

    for helper, arg in (
        (attach_penalty_state, y0),
        (strip_penalty_state, full),
        (penalty_integral, full),
    ):
        with pytest.raises(ValueError, match="at least one"):
            helper(arg, n=0)


def test_bound_penalty_finds_a_predictor_nested_inside_another():
    # An is_leaf traversal that stops at the outer BoundedPredictor would
    # skip the inner one's box entirely.
    inner = saturate(network_predictors()["rate"])
    outer = hm.BoundedPredictor(
        inner=inner, input_keys=("temp",), input_bounds=(0.0, 2.0), output_bounds=(0.05, 3.0)
    )
    assert len(_bounded_leaves(outer)) == 2
    with pytest.raises(ValueError, match="points"):
        bound_penalty(outer, (box_grid(outer.in_scaler, 3),))
    fresh = hm.BoundedPredictor(
        inner=network_predictors()["rate"],
        input_keys=("temp",),
        input_bounds=(0.0, 2.0),
        output_bounds=(0.05, 3.0),
    )
    assert grid_penalty(outer) > grid_penalty(fresh)


def test_bound_penalty_gradient_reaches_the_inner_weights():
    rate = saturate(network_predictors()["rate"])
    points = (box_grid(rate.in_scaler, 5),)
    grads = eqx.filter_grad(lambda p: bound_penalty(p, points))(rate)
    w = grads.inner.mlp.layers[-1].weight
    assert jnp.all(jnp.isfinite(w)) and float(jnp.linalg.norm(w)) > 1.0
    jitted = eqx.filter_jit(lambda p: bound_penalty(p, points))(rate)
    np.testing.assert_allclose(jitted, bound_penalty(rate, points), rtol=1e-6)


class Latent1(Latent):
    """``Latent`` with a rank-1 output, the shape before one-output predictors went scalar."""

    def __init__(self, c: float) -> None:
        super().__init__(c)

    def __call__(self, x: Array) -> Array:
        return self.c[None]


def test_penalties_accept_scalar_and_rank_1_outputs():
    """One-output predictors may return ``()`` or ``(1,)``; the penalties agree on both.

    ``saturation`` reads the trailing axis as outputs, so a ``[G]`` batch of
    scalar latents read as ``G`` outputs of one point would charge only the
    first point.
    """
    points = (jnp.linspace(-3.0, 3.0, 7)[:, None],)

    def bounded(inner: hm.Predictor) -> hm.BoundedPredictor:
        return hm.BoundedPredictor(
            inner=inner, input_keys=("u",), input_bounds=(-3.0, 3.0), output_bounds=(0.0, 1.0)
        )

    scalar = float(bound_penalty(bounded(Latent(6.0)), points))
    rank_1 = float(bound_penalty(bounded(Latent1(6.0)), points))
    assert scalar > 0.0 and scalar == pytest.approx(rank_1)

    scaler = latent_predictors(0.0)[0].out_scaler
    outputs = scaler.from_latent(jnp.linspace(0.0, 8.0, 5))
    np.testing.assert_allclose(
        trajectory_saturation_penalty(outputs, scaler),
        trajectory_saturation_penalty(outputs[:, None], scaler),
        rtol=1e-6,
    )


def test_soft_logit_is_exact_inside_and_linear_outside():
    s = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(soft_logit(s), jax.scipy.special.logit(s), atol=1e-5)
    grad = jax.vmap(jax.grad(soft_logit))
    # The stop_gradient on the clamp is load-bearing: without it the
    # interior derivative picks up a spurious term.
    np.testing.assert_allclose(grad(s), 1.0 / (s * (1.0 - s)), rtol=1e-4)
    outside = jnp.array([-5.0, -0.01, 1.01, 7.0])
    assert jnp.all(jnp.isfinite(soft_logit(outside))) and jnp.all(grad(outside) > 0.0)
    # The hard clip this replaced has exactly zero gradient there.
    hard = jax.grad(lambda x: jax.scipy.special.logit(jnp.clip(x, 1e-3, 1 - 1e-3)))
    assert float(hard(jnp.asarray(2.0))) == 0.0
    # eps sets the continuation slope.
    slope = lambda eps: float(jax.grad(lambda v: soft_logit(v, eps))(jnp.asarray(3.0)))  # noqa: E731
    assert slope(1e-4) > slope(1e-2)


def test_box_violation_pushes_back_from_outside_only():
    lows, highs = jnp.array([0.0]), jnp.array([10.0])
    grad = jax.grad(lambda v: box_violation(v, lows, highs))
    assert float(box_violation(jnp.array([4.0]), lows, highs)) == 0.0
    assert float(grad(jnp.array([4.0]))[0]) == 0.0
    assert float(grad(jnp.array([13.0]))[0]) > 0.0 > float(grad(jnp.array([-3.0]))[0])
    pushes = [float(grad(jnp.array([x]))[0]) for x in (20.0, 200.0, 2000.0)]
    assert pushes[0] < pushes[1] < pushes[2]
    # Width-normalised, so one weight works across channels.
    np.testing.assert_allclose(
        box_violation(jnp.array([2.0]), jnp.array([0.0]), jnp.array([1.0])),
        box_violation(jnp.array([2000.0]), jnp.array([0.0]), jnp.array([1000.0])),
    )


def test_clip_ste_and_softclip():
    xs = jnp.array([-2.0, 0.4, 3.0])
    np.testing.assert_allclose(clip_ste(xs, 0.0, 1.0), [0.0, 0.4, 1.0])
    assert jax.vmap(jax.grad(lambda v: clip_ste(v, 0.0, 1.0)))(xs).tolist() == [1.0, 1.0, 1.0]

    out = softclip(jnp.linspace(-10.0, 10.0, 41), 0.0, 1.0)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert 0.0 < float(jax.grad(lambda v: softclip(v, 0.0, 1.0))(jnp.asarray(1.05))) < 1.0
    # Interior distortion is O(1 / beta), which is why softclip is not the
    # logit guard. Measured near an edge; the shoulders cancel at the centre.
    error = lambda beta: abs(float(softclip(jnp.asarray(0.05), 0.0, 1.0, beta)) - 0.05)  # noqa: E731
    assert error(20.0) > error(200.0) and error(20.0) > 1e-3


def test_vector_scaler_maps_each_component_against_its_own_box():
    scaler = hm.BoundScaler(((0.0, 10.0), (-1.0, 1.0), (0.0, None)))
    x = jnp.array([3.5, 0.25, 7.0])
    np.testing.assert_allclose(scaler.from_latent(scaler.to_latent(x)), x, rtol=1e-5)
    out = scaler.from_latent(jnp.array([100.0, -100.0, -100.0]))
    assert 0.0 <= out[0] <= 10.0 and -1.0 <= out[1] <= 1.0 and out[2] >= 0.0
    # Network inputs: finite boxes go to latent, open ones pass through.
    np.testing.assert_allclose(scaler.scale_inputs(x)[2], 7.0)
    np.testing.assert_allclose(scaler.scale_inputs(x)[:2], scaler.to_latent(x)[:2])


def test_stepsize_controller_drives_a_real_solve():
    # diffrax needs atol as an array to broadcast it per state; a tuple
    # atol used to be coerced by hand in every example.
    config = _solver(atol=(1e-8, 1e-8), rtol=1e-6, max_steps=32)
    controller = config.stepsize_controller()
    assert isinstance(controller, diffrax.PIDController)
    assert jnp.asarray(controller.atol).shape == (2,)
    assert jnp.asarray(_solver().stepsize_controller().atol).shape == ()
    ys = config.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -y), jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0])
    ).ys
    assert ys is not None
    np.testing.assert_allclose(ys[-1], jnp.array([1.0, 2.0]) * math.exp(-1.0), rtol=1e-4)


ALL_LOSSES = [masked_mse, masked_mle, bal_mse, bal_mle]


def test_vector_covariates_stack_and_y0_fn_sees_covariates_and_channels():
    seen = {}

    def y0_fn(covariates, channels):
        seen["keys"] = (sorted(covariates), sorted(channels))
        return jnp.array([covariates["f"][0], channels["x"].values[0]])

    experiments = [
        hm.make_experiment(
            channels={"x": _obs([0.0], [float(i)])},
            covariates={"f": jnp.array([1.0 + i, 2.0])},
            y0_fn=y0_fn,
            exp_id=f"e{i}",
        )
        for i in range(2)
    ]
    assert seen["keys"] == (["f"], ["x"])
    bp = hm.make_dataset(experiments, output_channel_names=("x",)).bucket_payloads[0]
    np.testing.assert_array_equal(bp.covariates["f"], [[1.0, 2.0], [2.0, 2.0]])
    np.testing.assert_array_equal(bp.y0, [[1.0, 0.0], [2.0, 1.0]])


def test_make_dataset_is_deterministic_and_silent():
    experiments = [_exp("e0", x=([0.0, 1.0], [1.0, 2.0])), _exp("e1", x=([0.0], [3.0]))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = hm.make_dataset(experiments, output_channel_names=("x",))
    second = hm.make_dataset(experiments, output_channel_names=("x",))
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)


def test_dataset_constructs_from_its_public_fields_alone():
    ds = Dataset(bucket_payloads=(), output_channel_names=("c",), covariate_names=("a",))
    assert ds.experiments == ()


def test_split_is_deterministic_in_the_key():
    source = _numbered(20)
    first = hm.split_dataset(source, key=jr.key(42))
    again = hm.split_dataset(source, key=jr.key(42))
    other = hm.split_dataset(source, key=jr.key(1))
    assert [_ids(s) for s in first] == [_ids(s) for s in again]
    assert _ids(first[0]) != _ids(other[0])


def _prediction(observed: dict[str, list[float]], predicted: dict[str, list[float]]):
    """A ``PredictionResult`` with one experiment per entry, NaN marking unobserved cells."""
    experiments = {}
    for exp_id, values in observed.items():
        obs = jnp.asarray(values)
        experiments[exp_id] = hm.ExperimentPrediction(
            times=jnp.arange(obs.shape[0], dtype=jnp.float32),
            outputs={"x": jnp.asarray(predicted[exp_id])},
            observed={"x": obs},
            observation_mask={"x": ~jnp.isnan(obs)},
        )
    return hm.PredictionResult(experiments)


def test_metrics_read_a_prediction_result_and_pool_experiments(capsys):
    # Two experiments observed at 0 and 10, predicted 0: R^2 uses the pooled
    # variance, so it is -1, not undefined per experiment. NaN cells (a t0
    # row, say) count for nothing, whatever the prediction there.
    result = _prediction({"a": [jnp.nan, 0.0], "b": [10.0]}, {"a": [99.0, 0.0], "b": [0.0]})
    metrics = hm.compute_metrics(result)
    x = metrics["x"]
    assert int(x.n) == 2
    assert float(x.mse) == pytest.approx(50.0) and float(x.mae) == pytest.approx(5.0)
    assert float(x.r2) == pytest.approx(-1.0)
    flat = _prediction({"a": [2.0], "b": [2.0]}, {"a": [0.0], "b": [0.0]})
    assert np.isnan(float(hm.compute_metrics(flat)["x"].r2))

    hm.print_metrics(result, header="RESULTS")
    from_result = capsys.readouterr().out
    hm.print_metrics(metrics, header="RESULTS")
    assert capsys.readouterr().out == from_result
    assert "RESULTS" in from_result and "5.0000e+00" in from_result


def test_a_failed_experiment_is_listed_as_the_worst():
    result = _prediction(
        {"a": [1.0], "b": [1.0], "c": [1.0]}, {"a": [2.0], "b": [jnp.nan], "c": [6.0]}
    )
    rows = _worst_first(hm.compute_metrics(result, by="experiment"), "x")
    assert [exp_id for exp_id, _ in rows] == ["b", "c", "a"]


@pytest.mark.parametrize("loss_fn", ALL_LOSSES)
def test_losses_score_the_decay_fit(loss_fn, decay_data):
    for bp in decay_data.bucket_payloads:
        wrong = loss_fn(_analytic(bp, 0.5), bp)
        right = loss_fn(_analytic(bp, TRUE_RATE), bp)
        assert wrong.shape == ()
        assert float(right) < float(wrong)
        if loss_fn in (masked_mse, bal_mse):
            assert float(right) == pytest.approx(0.0, abs=1e-10)


def test_bootstrap_renaming_skips_ids_already_taken():
    """A source ID shaped like ``a#2`` never clashes with a renamed repeat."""
    source = hm.make_dataset(
        [_exp(i, x=([0.0, 1.0], [1.0, 2.0])) for i in ("a", "a#2", "b")],
        output_channel_names=("x",),
    )
    for seed in range(60):
        resampled = hm.make_bootstrap_dataset(source, key=jr.key(seed))
        ids = [exp.exp_id for exp in resampled.experiments]
        assert len(set(ids)) == len(ids)


def identity_bounded(keys: tuple[str, ...] | None = ("a", "b")) -> BoundedPredictor:
    return BoundedPredictor(
        inner=Identity(),
        input_keys=keys,
        input_bounds=((0.0, 1.0),) * 2,
        output_bounds=((0.0, 1.0),) * 2,
    )


def test_bounded_predictor_reads_named_or_positional_inputs():
    bp = identity_bounded()
    mapping = {"a": 0.4, "b": 0.7, "unused": 99.0}
    np.testing.assert_allclose(bp(mapping), [0.4, 0.7], atol=1e-5)
    np.testing.assert_allclose(bp(jnp.array([0.4, 0.7])), [0.4, 0.7], atol=1e-5)
    np.testing.assert_allclose(identity_bounded(("b", "a"))(mapping), [0.7, 0.4], atol=1e-5)
    jitted = eqx.filter_jit(lambda p, v: p(v))
    np.testing.assert_allclose(jitted(bp, {"a": 0.4, "b": 0.7}), [0.4, 0.7], atol=1e-5)
    np.testing.assert_allclose(jitted(bp, jnp.array([0.4, 0.7])), [0.4, 0.7], atol=1e-5)
    assert identity_bounded(None).input_keys == ("x1", "x2")


class Scalar(hm.Predictor):
    """Returns a rank-0 latent, which a one-output wrapper must also accept."""

    def __call__(self, x: Array) -> Array:
        return jnp.sum(x)


def test_one_output_is_a_scalar_whatever_the_inner_returns():
    for inner in (Scalar(), Identity()):
        bp = BoundedPredictor(
            inner=inner, input_keys=("x",), input_bounds=(0.0, 1.0), output_bounds=(0.0, 1.0)
        )
        out = bp({"x": 0.25})
        assert out.shape == ()
        np.testing.assert_allclose(out, 0.25, rtol=1e-5)


def test_evaluate_predictor_broadcasts_named_inputs():
    bp = identity_bounded()
    scalar = hm.evaluate_predictor(bp, {"a": 0.4, "b": 0.7})
    np.testing.assert_allclose(scalar, [0.4, 0.7], atol=1e-5)
    sweep = hm.evaluate_predictor(bp, {"a": jnp.linspace(0.1, 0.9, 5), "b": 0.5})
    assert sweep.shape == (5, 2)
    np.testing.assert_allclose(sweep[:, 0], jnp.linspace(0.1, 0.9, 5), atol=1e-5)
    np.testing.assert_allclose(sweep[:, 1], 0.5, atol=1e-5)
    grid = hm.evaluate_predictor(bp, {"a": jnp.array([0.2, 0.3])[:, None], "b": jnp.ones(3) / 4})
    assert grid.shape == (2, 3, 2)
    # One output gives just the broadcast shape; ints are promoted to float.
    rate = BoundedPredictor(inner=Scalar(), input_keys=("T",), output_bounds=(0.0, None))
    assert hm.evaluate_predictor(rate, {"T": jnp.arange(4)}).shape == (4,)
    assert hm.evaluate_predictor(rate, {"T": 1}).shape == ()


def test_pytree_restart_keeps_scalers_and_constants():
    def bounded(key):
        return BoundedPredictor(
            inner=mlp(key),
            input_keys=("a", "b"),
            input_bounds=((0, 10), (0, 5)),
            output_bounds=(-2, 2),
        )

    predictors = {
        "twins": (bounded(jr.key(0)), bounded(jr.key(0))),
        "fixed": hm.Constant((1.0, [2.0, 3.0])),
    }
    fresh = reinitialize_pytree_with_key(predictors, jr.key(2))
    a, b = fresh["twins"]
    # Identical siblings get independent subkeys.
    assert not jnp.allclose(a.inner.mlp.layers[0].weight, b.inner.mlp.layers[0].weight)
    # Scalers hold geometry, not state. A temperature redrawn from N(0, 1)
    # can go negative and invert the squash.
    for leaf in (a, b):
        assert float(leaf.in_scaler.temperature) == 1.0
        assert float(leaf.out_scaler.temperature) == 1.0
    # The restart uses the MLP's own init, not leaf-level N(0, 1).
    assert float(jnp.std(a.inner.mlp.layers[0].weight)) < 0.7
    assert eqx.tree_equal(fresh["fixed"], predictors["fixed"])


N_ACC = 2  # saturation accumulator, input-violation accumulator


def simulate_embedded(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """Decay whose state ``x`` feeds the predictor; two accumulators ride along.

    The accumulators integrate the predictor's output saturation and its
    input violation, so the final state holds both time integrals.
    """
    (p,) = predictors
    rate = jnp.reshape(p({"u": jnp.asarray(1.0)}), ())

    def physics(t, y, args):
        return -rate * y

    def penalty_rates(t, y, args):
        u = y[0]
        z = p.inner(p.in_scaler.to_latent(u))
        return jnp.stack([p.out_scaler.saturation(z), p.in_scaler.input_violation(u)])

    term = diffrax.ODETerm(penalty_vector_field(physics, penalty_rates))
    return solver.diffeqsolve(term, ts, y0).ys


def strip_accumulators(state: Array) -> Array:
    return strip_penalty_state(state, N_ACC)


def accumulated_penalty(full_state: Array, bp: Any) -> Array:
    return jnp.sum(penalty_integral(full_state, N_ACC))


EMBEDDED_KWARGS: dict[str, Any] = dict(
    simulate_fn=simulate_embedded, state_to_output=strip_accumulators, solver=SOLVER
)


def embedded_dataset(*, probe: bool = False) -> hm.Dataset:
    """One experiment observed at four times, or a probe with no observations.

    A probe keeps its time grid, so the solver still integrates over it, but
    its mask is all False: the data loss is exactly zero and only a
    trajectory penalty can produce a gradient.
    """
    ts = jnp.array([0.5, 1.0, 1.5, 2.0])
    values = jnp.array([]) if probe else jnp.exp(-0.5 * ts)
    experiment = hm.make_experiment(
        channels={"x": hm.ChannelObs(ts=ts, values=values)},
        y0=attach_penalty_state(jnp.array([1.0]), N_ACC),
        t0=0.0,
        exp_id="probe" if probe else "real",
    )
    return hm.make_dataset([experiment], output_channel_names=("x",))


def test_the_score_charges_the_trajectory_penalty():
    data_only = build_score_bucket(loss_fn=masked_mse, **EMBEDDED_KWARGS)
    penalised = build_score_bucket(
        loss_fn=masked_mse,
        trajectory_penalty_fn=accumulated_penalty,
        trajectory_penalty_weight=1.0,
        **EMBEDDED_KWARGS,
    )
    predictors = latent_predictors(5.0)
    full = jnp.asarray(1.0)
    # Both datasets share one grid shape, so each kernel compiles once.
    for probe in (False, True):
        bp = embedded_dataset(probe=probe).bucket_payloads[0]
        data = float(data_only(predictors, bp, full))
        if probe:
            assert int(bp.mask.sum()) == 0
            assert data == pytest.approx(0.0, abs=1e-7)
        assert float(penalised(predictors, bp, full)) > data + 1.0


def test_data_penalty_points_sit_at_observed_cells(temp_data):
    preds = network_predictors()
    (source,) = data_penalty_points(preds, temp_data)
    assert source is not None
    # Ten observed cells across the four experiments, one temp column.
    assert source.points.shape == (10, 1)
    assert set(np.asarray(source.points[:, 0]).tolist()) == {0.5, 1.0, 1.5}
    # Horizon masking follows the loss's own prefix mask, cell for cell.
    for fraction in (1.0, 0.5, 0.05):
        expected = jnp.concatenate(
            [
                apply_length_mask(bp, jnp.asarray(fraction)).mask.any(-1)[bp.mask.any(-1)]
                for bp in temp_data.bucket_payloads
            ]
        )
        assert jnp.array_equal(length_mask_keep(source, fraction), expected)


def test_data_penalty_points_resolution():
    ts = jnp.array([0.5, 1.0])
    experiments = [
        hm.make_experiment(
            channels={"y": hm.ChannelObs(ts=ts, values=ts)},
            covariates={"a": 1.0, "b": 10.0},
            y0=[0.0],
        ),
        hm.make_experiment(
            channels={"y": hm.ChannelObs(ts=ts, values=jnp.array([]))},
            covariates={"a": 9.0, "b": 99.0},
            y0=[0.0],
        ),
    ]
    dataset = hm.make_dataset(experiments, output_channel_names=("y",))

    def leaf(keys):
        inner = hm.MLPPredictor(in_size=2, out_size=1, width_size=2, depth=1, key=jr.key(0))
        return hm.BoundedPredictor(inner=inner, input_keys=keys, input_bounds=(0.0, 100.0))

    swapped, missing = data_penalty_points((leaf(("b", "a")), leaf(("a", "c"))), dataset)
    assert swapped is not None and missing is None
    # Columns follow input_keys order, and the probe contributes nothing.
    assert swapped.points.tolist() == [[10.0, 1.0], [10.0, 1.0]]
    assert data_penalty_points(hm.Constant(1.0), dataset) == ()


def flags(mask: Any) -> list[bool]:
    return [bool(leaf) for leaf in jtu.tree_leaves(mask)]


def test_default_mask_trains_networks_and_parameters_only():
    predictors = network_predictors()
    predictors["k"] = hm.Parameter(0.5, bounds=(0.0, 1.0))
    mask = trainable_mask(predictors)
    assert jtu.tree_structure(mask) == jtu.tree_structure(predictors)
    rate = mask["rate"]
    assert rate.in_scaler.temperature is False and rate.out_scaler.temperature is False
    assert not any(flags(mask["offset"]))
    assert mask["k"].latent is True and mask["k"].scaler.temperature is False
    inner_leaves = jtu.tree_leaves(predictors["rate"].inner)
    assert flags(rate.inner) == [eqx.is_inexact_array(leaf) for leaf in inner_leaves]

    scalers_too = trainable_mask(predictors, train_scalers=True)
    assert scalers_too["rate"].in_scaler.temperature is True
    assert not any(flags(scalers_too["offset"])), "a Constant is never trainable"
    assert not any(flags(trainable_mask(predictors, predicate=lambda _: False)))
