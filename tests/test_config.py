"""Configuration objects: what they accept, what they refuse, and how they serialise.

Every constructor that validates its arguments has its rejections in one
table, ``test_invalid_configuration_raises``. A rejected config is the
contract: each one would otherwise be accepted and then quietly do the
wrong thing. The rest covers the valid side: defaults, the single-phase
shorthand, solver round trips, schedules, profiles and the RNG anchor.
"""

from __future__ import annotations

import json
import math

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import jaxhybridmodels as hm
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
from jaxhybridmodels.training import EvosaxTrainingConfig, OptaxTrainingConfig, TrainingPhase

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


def _trajectory_penalty(state: Array, bp: object) -> Array:
    del bp
    return jnp.sum(state)


class _Unregistered(diffrax.Heun):
    pass


class _UnregisteredAdjoint(diffrax.DirectAdjoint):
    pass


_NAN = float("nan")
_INF = float("inf")

INVALID = [
    # TrainingPhase
    pytest.param(lambda: TrainingPhase(0), "steps", id="phase-steps-0"),
    pytest.param(lambda: TrainingPhase(-1), "steps", id="phase-steps-negative"),
    pytest.param(lambda: TrainingPhase(2, lr=-1.0), "lr", id="phase-lr-negative"),
    pytest.param(lambda: TrainingPhase(2, lr=_NAN), "lr", id="phase-lr-nan"),
    # The horizon indexes a prefix of the observation times: zero scores
    # nothing, above one claims more timestamps than exist.
    pytest.param(lambda: TrainingPhase(2, horizon=0.0), "horizon", id="phase-horizon-0"),
    pytest.param(lambda: TrainingPhase(2, horizon=-0.1), "horizon", id="phase-horizon-neg"),
    pytest.param(lambda: TrainingPhase(2, horizon=1.5), "horizon", id="phase-horizon-1.5"),
    pytest.param(lambda: TrainingPhase(2, penalty_weight=-1.0), "non-negative", id="phase-pw-neg"),
    pytest.param(lambda: TrainingPhase(2, penalty_weight=_NAN), "finite", id="phase-pw-nan"),
    pytest.param(lambda: TrainingPhase(2, penalty_weight=_INF), "finite", id="phase-pw-inf"),
    # OptaxTrainingConfig: the phase schedule. Zero phases would run nothing
    # and return an empty history with no hint why.
    pytest.param(lambda: OptaxTrainingConfig(), "steps= for one phase", id="optax-no-phases"),
    pytest.param(lambda: OptaxTrainingConfig(phases=()), "steps= for one phase", id="optax-empty"),
    pytest.param(
        lambda: OptaxTrainingConfig(steps=2, phases=(TrainingPhase(3),)),
        "steps= or phases=",
        id="optax-steps-and-phases",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(phases=(TrainingPhase(2),), penalty_weight=1e-2),
        "per TrainingPhase",
        id="optax-phases-with-shorthand-weight",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(phases=(TrainingPhase(2),), lr=1e-2),
        "per TrainingPhase",
        id="optax-phases-with-shorthand-lr",
    ),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, penalty_weight=-1.0), "non-negative"),
    pytest.param(lambda: OptaxTrainingConfig(steps=0), "steps", id="optax-steps-0"),
    # Optimiser state belongs to the optimiser that built it; switching
    # without a reset used to be accepted and silently keep the old one.
    pytest.param(
        lambda: OptaxTrainingConfig(
            phases=(TrainingPhase(2), TrainingPhase(2, optimizer="adabelief"))
        ),
        "reset_optimizer",
        id="optax-switch-without-reset",
    ),
    # Penalty points are [G, n_inputs] physical input vectors.
    pytest.param(
        lambda: OptaxTrainingConfig(steps=2, penalty_points=(jnp.zeros(3),)),
        "penalty_points",
        id="optax-points-rank-1",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(steps=2, penalty_points=(jnp.zeros((2, 2, 1)),)),
        "penalty_points",
        id="optax-points-rank-3",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(steps=2, trajectory_penalty_weight=-1.0),
        "non-negative",
        id="optax-traj-weight-neg",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(
            steps=2, trajectory_penalty_fn=_trajectory_penalty, trajectory_penalty_weight=_NAN
        ),
        "finite",
        id="optax-traj-weight-nan",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(steps=2, trajectory_penalty_weight=1.0),
        "trajectory_penalty_fn is None",
        id="optax-traj-weight-without-fn",
    ),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, tournament_attempts=0), "attempts"),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, tournament_steps=-1), "tournament_steps"),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, tournament_lr=_NAN), "tournament_lr"),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, patience=-1), "patience"),
    # EvosaxTrainingConfig
    pytest.param(lambda: EvosaxTrainingConfig(algorithm="nope"), "available", id="evo-algorithm"),
    pytest.param(lambda: EvosaxTrainingConfig(init="nope"), "available", id="evo-init"),  # ty: ignore[invalid-argument-type]
    pytest.param(lambda: EvosaxTrainingConfig(penalty_weight=-1.0), "non-negative", id="evo-pw"),
    pytest.param(lambda: EvosaxTrainingConfig(penalty_weight=_NAN), "finite", id="evo-pw-nan"),
    pytest.param(
        lambda: EvosaxTrainingConfig(trajectory_penalty_weight=-1.0),
        "non-negative",
        id="evo-traj-weight-neg",
    ),
    pytest.param(
        lambda: EvosaxTrainingConfig(trajectory_penalty_weight=1.0),
        "trajectory_penalty_fn is None",
        id="evo-traj-weight-without-fn",
    ),
    pytest.param(lambda: EvosaxTrainingConfig(population_size=0), "population_size"),
    pytest.param(lambda: EvosaxTrainingConfig(num_generations=0), "num_generations"),
    pytest.param(
        lambda: EvosaxTrainingConfig(penalty_points=(jnp.zeros(3),)),
        "penalty_points",
        id="evo-points-rank-1",
    ),
    pytest.param(lambda: EvosaxTrainingConfig(init_box_extent=-1.0), "init_box_extent"),
    pytest.param(lambda: EvosaxTrainingConfig(sigma_init=0.0), "sigma_init"),
    pytest.param(lambda: EvosaxTrainingConfig(log_every=0), "log_every"),
    # SolverConfig
    pytest.param(lambda: _solver(rtol=0.0), "rtol", id="solver-rtol-0"),
    pytest.param(lambda: _solver(rtol=-1.0), "rtol", id="solver-rtol-neg"),
    pytest.param(lambda: _solver(atol=0.0), "atol", id="solver-atol-0"),
    pytest.param(lambda: _solver(atol=(1e-6, 0.0)), "atol", id="solver-atol-tuple"),
    pytest.param(lambda: _solver(max_steps=0), "max_steps", id="solver-max-steps"),
    pytest.param(lambda: _solver(dt0=0.0), "dt0", id="solver-dt0-0"),
    pytest.param(lambda: _solver(dt0=-0.1), "dt0", id="solver-dt0-neg"),
    pytest.param(
        lambda: hm.SolverConfig.from_dict(
            {"solver": "DoesNotExist", "rtol": 1e-4, "atol": 1e-5, "max_steps": 10, "dt0": None}
        ),
        "solver",
        id="solver-from-dict-unknown",
    ),
    pytest.param(
        lambda: _solver(solver=_Unregistered()).to_dict(),
        "SOLVER_REGISTRY",
        id="solver-unregistered",
    ),
    pytest.param(
        lambda: _solver(adjoint=_UnregisteredAdjoint()).to_dict(),
        "ADJOINT_REGISTRY",
        id="adjoint-unregistered",
    ),
    # Schedules and profiles
    pytest.param(lambda: annealing_schedule("cosine", total_epochs=0), "total_epochs"),
    pytest.param(
        lambda: annealing_schedule("warmup_cosine", total_epochs=10, warmup_epochs=10), "warmup"
    ),
    pytest.param(lambda: annealing_schedule("cosine", total_epochs=10, init_value=0.0), "init"),
    pytest.param(lambda: annealing_schedule("cosine", total_epochs=10, end_value=1.5), "end"),
    pytest.param(lambda: annealing_schedule("nope", total_epochs=10), "kind", id="schedule-kind"),
    pytest.param(lambda: ramp_profile(t0=3.0, t1=3.0, v0=0.0, v1=1.0), "t1", id="ramp-t1"),
    pytest.param(lambda: piecewise_linear_profile(((0.0, 1.0),)), "at least two", id="pw-one"),
    pytest.param(
        lambda: piecewise_linear_profile(((0.0, 1.0), (1.0, 2.0), (0.5, 3.0))),
        "strictly increasing",
        id="pw-decreasing",
    ),
]


@pytest.mark.parametrize("build, match", INVALID)
def test_invalid_configuration_raises(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_training_configs_normalise_their_schedule():
    # The shorthand builds one phase from the TrainingPhase defaults.
    single = OptaxTrainingConfig(steps=5, lr=1e-2)
    assert single.phases == (TrainingPhase(5, lr=1e-2),)
    assert single.phases[0].optimizer == "adamw" and single.phases[0].horizon == 1.0

    # A list of phases is stored as a tuple, and the step budget sums.
    multi = OptaxTrainingConfig(
        phases=[  # ty: ignore[invalid-argument-type]
            TrainingPhase(2, horizon=0.5),
            TrainingPhase(3, optimizer="adabelief", reset_optimizer=True, penalty_weight=1e-2),
            TrainingPhase(4, optimizer="adabelief"),
        ]
    )
    assert isinstance(multi.phases, tuple)
    assert multi.total_steps == 9
    assert [p.penalty_weight for p in multi.phases] == [0.0, 1e-2, 0.0]

    # A zero trajectory weight with a function attached is a valid no-op.
    off = OptaxTrainingConfig(steps=1, trajectory_penalty_fn=_trajectory_penalty)
    assert off.trajectory_penalty_weight == 0.0
    assert EvosaxTrainingConfig().penalty_weight == 0.0


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------


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


def test_solver_defaults_and_registries():
    config = _solver()
    # The examples all wrote DirectAdjoint by hand before the field existed.
    assert isinstance(config.adjoint, diffrax.DirectAdjoint)
    assert (config.pcoeff, config.icoeff, config.dcoeff) == (0.0, 1.0, 0.0)
    for name, cls in [("Tsit5", diffrax.Tsit5), ("Kvaerno3", diffrax.Kvaerno3)]:
        assert SOLVER_REGISTRY[name] is cls
    for name in ("RecursiveCheckpoint", "Direct", "Backsolve", "ForwardMode"):
        assert name in ADJOINT_REGISTRY


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


# ---------------------------------------------------------------------------
# Schedules and profiles
# ---------------------------------------------------------------------------


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


def test_schedule_shapes_between_endpoints():
    cosine = annealing_schedule("cosine", total_epochs=100)
    values = [float(cosine(i)) for i in range(0, 101, 10)]
    assert values == sorted(values, reverse=True)
    exponential = annealing_schedule("exponential", total_epochs=10, end_value=0.01)
    ratios = [float(exponential(i + 1)) / float(exponential(i)) for i in range(3)]
    assert ratios == pytest.approx([ratios[0]] * 3, abs=1e-3)


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
