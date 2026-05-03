"""Optax training tests (SPEC §5.7 / R-T1..R-T8 / R-J1..R-J3 / R-R1..R-R3 / R-A2 / R-L1).

Synthetic harmonic-oscillator setup: state = [position, velocity], dy/dt = [v, -omega^2 x].
Four experiments share ten timestamps; only position is observed. The trainable
predictor holds a single scalar ``omega`` JAX leaf so convergence and tournament
behaviour can be checked against the ground-truth ``omega = 1.0``.
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from hybridmodels.predictors.base import Predictor
from hybridmodels.solver import SolverConfig
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax
from hybridmodels.ui.testing import RecordingUI

OMEGA_TRUE: float = 1.0
N_TIMESTEPS: int = 10
T_MAX: float = 5.0
INITIAL_STATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, -0.5),
    (1.0, 1.0),
)


class _OmegaPredictor(Predictor):
    omega: Array

    def __init__(self, omega):
        # Strong-type the leaf so the predictor's pytree weak_type does not flip
        # after the first apply_update (which would force a make_step retrace).
        self.omega = jnp.asarray(omega, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:  # type: ignore[override]
        return self.omega


def _y0_fn_factory(y0: Array):
    def _y0_fn(_cov, _channels):
        return y0

    return _y0_fn


def _solver_config() -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=4096,
        dt0=0.05,
    )


def _true_position(omega: float, t: Array, x0: float, v0: float) -> Array:
    return x0 * jnp.cos(omega * t) + (v0 / omega) * jnp.sin(omega * t)


def _state_to_output(state: Array) -> Array:
    return state[..., 0:1]


def _make_oscillator_dataset() -> Dataset:
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments = []
    for i, (x0, v0) in enumerate(INITIAL_STATES):
        x_obs = _true_position(OMEGA_TRUE, ts, x0, v0)
        y0 = jnp.asarray([x0, v0])
        experiments.append(
            make_experiment(
                covariates={"id": float(i)},
                channels={"position": ChannelObs(ts=ts, values=x_obs)},
                y0_fn=_y0_fn_factory(y0),
                exp_id=f"exp_{i}",
            )
        )
    return make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=("position",),
    )


def _make_simulate_fn():
    def simulate_fn(predictor, ts, covariates, y0, solver):
        omega = predictor.omega

        def vector_field(t, y, args):
            return jnp.stack([y[1], -(omega**2) * y[0]])

        term = diffrax.ODETerm(vector_field)
        controller = diffrax.PIDController(rtol=solver.rtol, atol=solver.atol)
        sol = diffrax.diffeqsolve(
            term,
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=solver.max_steps,
        )
        return sol.ys

    return simulate_fn


def _nan_simulate_fn(predictor, ts, covariates, y0, solver):
    return jnp.full((ts.shape[0], 2), jnp.nan)


def test_convergence_recovers_omega():
    # Initial omega is inside the convex basin around omega=1.0; the harmonic
    # oscillator's loss is multimodal further out (a local minimum sits near
    # omega=2.4) so a single-phase optax run from omega=2 lands in that basin
    # rather than the global optimum.
    pred = _OmegaPredictor(omega=1.5)
    ds = _make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(150,),
        lr=(5e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=False,
    )
    history, trained = train_with_optax(
        pred,
        ds,
        config,
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )
    assert abs(float(trained.omega) - OMEGA_TRUE) <= 2e-2 * OMEGA_TRUE
    assert history[-1] < 1e-2


def test_multi_phase_runs_and_improves():
    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()

    config_phase0 = OptaxTrainingConfig(
        steps=(40,),
        lr=(5e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(0.5,),
        verbose=False,
    )
    _, after_phase0 = train_with_optax(
        pred,
        ds,
        config_phase0,
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )

    config_two_phase = OptaxTrainingConfig(
        steps=(40, 40),
        lr=(5e-2, 1e-2),
        optimizer=("adamw", "adamw"),
        reset_optimiser_state=(False, True),
        length_schedule=(0.5, 1.0),
        verbose=False,
    )
    _, after_two = train_with_optax(
        pred,
        ds,
        config_two_phase,
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )
    err_phase0 = abs(float(after_phase0.omega) - OMEGA_TRUE)
    err_two = abs(float(after_two.omega) - OMEGA_TRUE)
    assert err_two < err_phase0


def test_length_schedule_does_not_recompile():
    trace_count = [0]
    base_simulate = _make_simulate_fn()

    def counted_simulate(predictor, ts, covariates, y0, solver):
        trace_count[0] += 1
        return base_simulate(predictor, ts, covariates, y0, solver)

    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()

    config = OptaxTrainingConfig(
        steps=(2, 2),
        lr=(1e-2, 1e-2),
        optimizer=("adamw", "adamw"),
        reset_optimiser_state=(False, False),
        length_schedule=(0.5, 1.0),
        verbose=False,
    )
    train_with_optax(
        pred,
        ds,
        config,
        simulate_fn=counted_simulate,
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )
    assert trace_count[0] == 1


def test_missing_key_raises():
    pred = _OmegaPredictor(omega=1.0)
    ds = _make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(1,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=False,
    )
    with pytest.raises(TypeError):
        train_with_optax(  # type: ignore[call-arg]
            pred,
            ds,
            config,
            simulate_fn=_make_simulate_fn(),
            solver=_solver_config(),
        )


def test_tournament_finds_good_init_at_least_once():
    seeds = (0, 1, 2, 3, 4)
    base_losses: list[float] = []
    tournament_losses: list[float] = []

    config_no_tournament = OptaxTrainingConfig(
        steps=(1,),
        lr=(0.0,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=False,
    )
    config_tournament = OptaxTrainingConfig(
        steps=(1,),
        lr=(0.0,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        tournament_attempts=4,
        tournament_steps=20,
        verbose=False,
    )

    for seed in seeds:
        pred = _OmegaPredictor(omega=2.0)
        ds = _make_oscillator_dataset()

        history_no_tourn, _ = train_with_optax(
            pred,
            ds,
            config_no_tournament,
            simulate_fn=_make_simulate_fn(),
            solver=_solver_config(),
            key=jr.PRNGKey(seed),
        )
        base_losses.append(history_no_tourn[0])

        history_tourn, _ = train_with_optax(
            pred,
            ds,
            config_tournament,
            simulate_fn=_make_simulate_fn(),
            solver=_solver_config(),
            key=jr.PRNGKey(seed),
        )
        tournament_losses.append(history_tourn[0])

    assert any(t <= b for t, b in zip(tournament_losses, base_losses, strict=True))


def test_tournament_falls_back_when_all_attempts_fail():
    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(1,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        tournament_attempts=2,
        tournament_steps=2,
        verbose=False,
    )
    with pytest.warns(RuntimeWarning):
        train_with_optax(
            pred,
            ds,
            config,
            simulate_fn=_nan_simulate_fn,
            solver=_solver_config(),
            key=jr.PRNGKey(0),
        )


def test_recording_ui_lifecycle_events_fire():
    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(2,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=False,
    )
    ui = RecordingUI()
    train_with_optax(
        pred,
        ds,
        config,
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
        ui=ui,
    )
    names = [n for n, _ in ui.events]
    assert names.count("on_run_start") >= 1
    assert names.count("on_phase_start") >= 1
    assert names.count("on_step_end") == 2
    assert names.count("on_phase_end") >= 1
    assert names.count("on_run_end") >= 1
    assert names.count("on_compile_start") >= 1
    assert names.count("on_compile_done") >= 1


def test_silent_default_when_no_ui_and_verbose_false(capsys):
    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(1,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=False,
    )
    train_with_optax(
        pred,
        ds,
        config,
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )
    assert capsys.readouterr().out == ""
