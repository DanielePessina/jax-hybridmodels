"""Optax training tests.

Synthetic harmonic-oscillator setup: state ``[position, velocity]``,
``dy/dt = [v, -omega^2 * x]``. Four experiments share ten timestamps;
only position is observed. The trainable predictor holds a single
scalar ``omega`` JAX leaf so convergence and tournament behaviour can
be checked against the ground-truth ``omega = 1.0``.
"""

from __future__ import annotations

import math
import statistics

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from _harness import (
    INITIAL_STATES,
    N_TIMESTEPS,
    OMEGA_TRUE,
    T_MAX,
    OmegaPredictor,
    make_oscillator_dataset,
    make_oscillator_simulate_fn,
    oscillator_state_to_output,
    solver_config,
    true_position,
    y0_fn_factory,
)
from jax import Array

from jaxhybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.penalties import bound_penalty, box_grid
from jaxhybridmodels.prediction import predict_dataset
from jaxhybridmodels.predictors import BoundedPredictor, BoundScaler, MLPPredictor
from jaxhybridmodels.trainable import trainable_mask
from jaxhybridmodels.training.kernels import apply_length_mask
from jaxhybridmodels.training.optax import OptaxTrainingConfig, _shared_tournament, train_with_optax
from jaxhybridmodels.ui.testing import RecordingUI

# The ``_bounded_predictors`` leaf reads ``input_keys=("omega_input",)``,
# which the oscillator dataset's only covariate (``"id"``) does not
# resolve, so penalty-on runs must cover it with penalty-only points. One
# interior point inside the leaf's ``(0.5, 2.0)`` box.
_EXTRA_POINTS = jnp.asarray([[1.0]])


def _bounded_simulate_fn():
    """Same oscillator, but omega comes from a BoundedPredictor tuple.

    The penalty tests need a predictors pytree that actually holds a
    ``BoundedPredictor`` leaf; ``OmegaPredictor`` is a bare scalar module
    and would give the penalty nothing to find.
    """

    def simulate_fn(predictors, ts, covariates, y0, solver):
        (bp,) = predictors
        omega = jnp.squeeze(bp({"omega_input": jnp.asarray(1.0)}))

        def vector_field(t, y, args):
            return jnp.stack([y[1], -(omega**2) * y[0]])

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
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
    pred = OmegaPredictor(omega=1.5)
    ds = make_oscillator_dataset()
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
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    assert abs(float(trained.omega) - OMEGA_TRUE) <= 2e-2 * OMEGA_TRUE
    assert history[-1] < 1e-2


def test_multi_phase_runs_and_improves():
    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset()

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
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
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
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    err_phase0 = abs(float(after_phase0.omega) - OMEGA_TRUE)
    err_two = abs(float(after_two.omega) - OMEGA_TRUE)
    assert err_two < err_phase0


def test_length_schedule_does_not_recompile():
    trace_count = [0]
    base_simulate = make_oscillator_simulate_fn()

    def counted_simulate(predictor, ts, covariates, y0, solver):
        trace_count[0] += 1
        return base_simulate(predictor, ts, covariates, y0, solver)

    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset()

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
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    assert trace_count[0] == 1


def test_weak_zero_d_scalar_trains_via_settle_path():
    """A user model with a weak-typed scalar leaf still trains cleanly.

    ``jnp.asarray(1.0)`` produces a weak-typed 0-d array, and JAX keys its
    compile cache on weak type, so the first optimiser update would force
    a one-time retrace of every training kernel. The warm-up settle absorbs
    that (``training/optax.py:_warmup_compile``). This guards the settle
    path with the case that triggers it: the trainer must complete with
    finite losses. The framework's own scalars are strong-typed, so only a
    user-supplied tree like this one reaches the settle.
    """

    class WeakScalarPredictor(eqx.Module):
        omega: Array
        scale: Array

        def __call__(self) -> Array:
            return self.omega

    pred = WeakScalarPredictor(
        omega=jnp.asarray(2.0, dtype=jnp.float32),
        scale=jnp.asarray(1.0),  # weak-typed 0-d leaf, deliberately
    )
    assert jax.typeof(pred.scale).weak_type
    ds = make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(6,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        verbose=False,
    )
    history, _trained = train_with_optax(
        pred,
        ds,
        config,
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    assert len(history) == 6
    assert all(math.isfinite(v) for v in history)


def test_missing_key_raises():
    pred = OmegaPredictor(omega=1.0)
    ds = make_oscillator_dataset()
    config = OptaxTrainingConfig(
        steps=(1,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=False,
    )
    with pytest.raises(TypeError):
        train_with_optax(  # ty: ignore[missing-argument]
            pred,
            ds,
            config,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
        )


def test_tournament_reduces_across_seed_variance():
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
        pred = OmegaPredictor(omega=2.0)
        ds = make_oscillator_dataset()

        history_no_tourn, _ = train_with_optax(
            pred,
            ds,
            config_no_tournament,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(seed),
        )
        base_losses.append(history_no_tourn[0])

        history_tourn, _ = train_with_optax(
            pred,
            ds,
            config_tournament,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(seed),
        )
        tournament_losses.append(history_tourn[0])

    # The old assertion was "at least one of five seeds is no worse",
    # which is near-certain under the null and so could not fail.
    #
    # The baseline here is seed-independent by construction: a fixed
    # omega=2.0 start with lr=0.0 gives the same loss every time, which
    # this pins first. That also rules out the obvious replacement claim,
    # variance reduction: there is no baseline variance to reduce.
    #
    # What the tournament does claim is that restarting from several
    # random inits beats the fixed start. A tournament that silently fell
    # back to the original predictors would score exactly the baseline on
    # every seed and fail both assertions below.
    assert len(set(base_losses)) == 1, base_losses
    baseline = base_losses[0]
    assert statistics.median(tournament_losses) < baseline
    assert sum(loss < baseline for loss in tournament_losses) >= 3


def test_tournament_falls_back_when_all_attempts_fail():
    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset()
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
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(0),
        )


def test_recording_ui_lifecycle_events_fire():
    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset()
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
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
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
    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset()
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
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    assert capsys.readouterr().out == ""


def test_length_mask_keeps_n_obs_consistent_with_mask():
    bp = make_oscillator_dataset().bucket_payloads[0]
    masked = apply_length_mask(bp, jnp.asarray(0.5))
    assert int(masked.n_obs) == int(masked.mask.sum())


class TestPenaltyWiring:
    """The bound penalty reaches the optimiser without disturbing the data path."""

    def _bounded_predictors(self, *, scale: float = 1.0):
        inner = MLPPredictor(
            in_size=1,
            out_size=1,
            width_size=8,
            depth=1,
            activation_name="tanh",
            key=jr.PRNGKey(0),
        )
        if scale != 1.0:
            inner = eqx.tree_at(
                lambda m: m.mlp.layers[-1].weight,
                inner,
                inner.mlp.layers[-1].weight * scale,
            )
        return (
            BoundedPredictor(
                input_keys=("omega_input",),
                in_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
                inner=inner,
                out_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
            ),
        )

    def _config(
        self,
        *,
        steps: tuple[int, ...] = (3,),
        lr: tuple[float, ...] = (1e-2,),
        optimizer: tuple[str, ...] = ("adamw",),
        reset_optimiser_state: tuple[bool, ...] = (False,),
        length_schedule: tuple[float, ...] = (1.0,),
        penalty_weight: tuple[float, ...] = (0.0,),
        penalty_points: tuple[Array, ...] | None = None,
        restore_best: bool = True,
    ) -> OptaxTrainingConfig:
        return OptaxTrainingConfig(
            steps=steps,
            lr=lr,
            optimizer=optimizer,
            reset_optimiser_state=reset_optimiser_state,
            length_schedule=length_schedule,
            penalty_weight=penalty_weight,
            penalty_points=penalty_points,
            restore_best=restore_best,
            verbose=False,
        )

    def _run(self, preds, ds, config, *, bounded: bool = True, ui=None):
        return train_with_optax(
            preds,
            ds,
            config,
            simulate_fn=_bounded_simulate_fn() if bounded else make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(0),
            ui=ui,
        )

    def test_zero_weight_reproduces_the_unpenalised_run(self):
        # The default must be bit-for-bit inert, or every existing run
        # silently changes meaning. Run this against a predictors pytree
        # that actually holds a BoundedPredictor: with no such leaf the
        # penalty is zero for reasons unrelated to the weight, and the
        # test passes whether or not the feature works.
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        # Anti-vacuity guard: without this the test would also pass if the
        # penalty never reached the loss at all.
        assert float(bound_penalty(preds, (box_grid(preds[0].in_scaler, 5),))) > 0.0
        h_off, _ = self._run(preds, ds, self._config())
        h_zero, _ = self._run(preds, ds, self._config(penalty_weight=(0.0,)))
        assert h_off == h_zero

    def test_penalty_weight_broadcasts_across_phases(self):
        cfg = self._config(
            steps=(2, 2),
            lr=(1e-2, 1e-2),
            optimizer=("adamw", "adamw"),
            reset_optimiser_state=(False, False),
            length_schedule=(1.0, 1.0),
            penalty_weight=(1e-3,),
        )
        assert cfg.penalty_weight_for_phase(0) == 1e-3
        assert cfg.penalty_weight_for_phase(1) == 1e-3

    def test_per_phase_penalty_weight_is_respected(self):
        cfg = self._config(
            steps=(2, 2),
            lr=(1e-2, 1e-2),
            optimizer=("adamw", "adamw"),
            reset_optimiser_state=(False, False),
            length_schedule=(1.0, 1.0),
            penalty_weight=(0.0, 1e-2),
        )
        assert cfg.penalty_weight_for_phase(0) == 0.0
        assert cfg.penalty_weight_for_phase(1) == 1e-2

    def test_wrong_length_penalty_weight_raises(self):
        with pytest.raises(ValueError, match="penalty_weight"):
            self._config(
                steps=(2, 2),
                lr=(1e-2, 1e-2),
                optimizer=("adamw", "adamw"),
                reset_optimiser_state=(False, False),
                length_schedule=(1.0, 1.0),
                penalty_weight=(0.0, 1e-2, 1e-2),
            )

    def test_negative_penalty_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._config(penalty_weight=(-1.0,))

    @pytest.mark.parametrize("weight", [float("nan"), float("inf")])
    def test_nonfinite_penalty_weight_raises(self, weight):
        with pytest.raises(ValueError, match="finite"):
            self._config(penalty_weight=(weight,))

    def test_history_tracks_the_data_term_not_the_combined_objective(self):
        # A large penalty must not inflate the reported loss; otherwise
        # runs with different weights are incomparable and restore_best
        # chases the schedule rather than the fit.
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        h_off, _ = self._run(preds, ds, self._config())
        h_on, _ = self._run(
            preds,
            ds,
            self._config(penalty_weight=(1e3,), penalty_points=(_EXTRA_POINTS,)),
        )
        # Step 0 is evaluated at identical parameters in both runs, so the
        # data term must agree exactly even though the objectives differ.
        assert h_off[0] == pytest.approx(h_on[0], rel=1e-6)

    def test_penalty_changes_the_trajectory(self):
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        # restore_best=False so we observe where the optimiser actually
        # walked. With it on, both runs would return the step-0 snapshot
        # whenever the data loss never improves, and the comparison would
        # pass vacuously.
        _, off = self._run(preds, ds, self._config(restore_best=False))
        _, on = self._run(
            preds,
            ds,
            self._config(
                penalty_weight=(1e2,), penalty_points=(_EXTRA_POINTS,), restore_best=False
            ),
        )
        w_off = off[0].inner.mlp.layers[-1].weight
        w_on = on[0].inner.mlp.layers[-1].weight
        assert not jnp.allclose(w_off, w_on)

    def test_penalty_shrinks_saturation(self):
        # The point of the whole exercise: turning the penalty on must pull
        # a saturated predictor back toward its usable range.
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        points = (box_grid(preds[0].in_scaler, 5),)
        before = float(bound_penalty(preds, points))
        _, after_preds = self._run(
            preds,
            ds,
            self._config(
                steps=(25,),
                penalty_weight=(1e2,),
                penalty_points=(_EXTRA_POINTS,),
                restore_best=False,
            ),
        )
        after = float(bound_penalty(after_preds, points))
        assert after < before

    def test_ui_receives_the_penalty(self):
        ds = make_oscillator_dataset()
        ui = RecordingUI()
        self._run(
            self._bounded_predictors(scale=50.0),
            ds,
            self._config(penalty_weight=(1e-1,), penalty_points=(_EXTRA_POINTS,)),
            ui=ui,
        )
        steps = [kw for name, kw in ui.events if name == "on_step_end"]
        assert steps and all("penalty" in kw for kw in steps)

    def test_uncovered_leaf_with_penalty_on_raises(self):
        # An enabled penalty must never silently skip a leaf: input_keys
        # that do not resolve to dataset covariates leave the leaf with no
        # points unless the user supplies them.
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        with pytest.raises(ValueError, match="trajectory_penalty_fn"):
            self._run(preds, ds, self._config(penalty_weight=(1e-2,)))

    def test_extras_cover_an_unresolvable_leaf(self):
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        h, _ = self._run(
            preds,
            ds,
            self._config(penalty_weight=(1e-2,), penalty_points=(_EXTRA_POINTS,)),
        )
        assert h  # ran to completion

    def test_penalty_disabled_needs_no_coverage(self):
        # The off state stays fully inert: an unresolvable leaf is fine
        # when no phase charges the penalty.
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        h, _ = self._run(preds, ds, self._config())
        assert h

    def test_penalty_points_length_mismatch_raises(self):
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        with pytest.raises(ValueError, match="penalty_points has"):
            self._run(
                preds,
                ds,
                self._config(
                    penalty_weight=(1e-2,),
                    penalty_points=(_EXTRA_POINTS, _EXTRA_POINTS),
                ),
            )

    def test_penalty_points_wrong_input_columns_raises(self):
        ds = make_oscillator_dataset()
        preds = self._bounded_predictors(scale=50.0)
        with pytest.raises(ValueError, match="input columns"):
            self._run(
                preds,
                ds,
                self._config(penalty_weight=(1e-2,), penalty_points=(jnp.zeros((2, 3)),)),
            )


class TestPhaseOptimiserSwitch:
    def _cfg(self, optimizer, reset):
        return OptaxTrainingConfig(
            steps=(2, 2),
            lr=(1e-2, 1e-2),
            optimizer=optimizer,
            reset_optimiser_state=reset,
            length_schedule=(1.0, 1.0),
            verbose=False,
        )

    def test_switching_optimizer_without_a_reset_raises(self):
        # It used to be accepted and then ignored: without a reset only the
        # learning rate is pushed into the existing opt_state, so phase 1
        # kept running adamw.
        with pytest.raises(ValueError, match="reset_optimiser_state"):
            self._cfg(("adamw", "adabelief"), (False, False))

    def test_switching_optimizer_with_a_reset_is_allowed(self):
        cfg = self._cfg(("adamw", "adabelief"), (False, True))
        assert cfg.optimizer == ("adamw", "adabelief")

    def test_keeping_the_same_optimizer_needs_no_reset(self):
        cfg = self._cfg(("adamw", "adamw"), (False, False))
        assert cfg.reset_optimiser_state == (False, False)


class TestConfigValidation:
    """The ``__post_init__`` rules that had no test.

    Each is a guard against a config that would otherwise be accepted and
    then quietly do the wrong thing, so the raise is the contract.
    """

    def _kwargs(self, **overrides):
        base = dict(
            steps=(2,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            length_schedule=(1.0,),
            verbose=False,
        )
        base.update(overrides)
        return base

    def test_empty_steps_raises(self):
        # The phase count is len(steps), so zero phases means the run loop
        # never executes and the caller gets an empty history with no hint why.
        with pytest.raises(ValueError, match="at least one phase"):
            OptaxTrainingConfig(
                **self._kwargs(
                    steps=(), lr=(), optimizer=(), reset_optimiser_state=(), length_schedule=()
                )
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("lr", (1e-2, 1e-3)),
            ("optimizer", ("adamw", "adamw")),
            ("reset_optimiser_state", (False, False)),
            ("length_schedule", (1.0, 1.0)),
        ],
    )
    def test_phase_keyed_length_mismatch_raises(self, field, value):
        # These four do not broadcast: a length that disagrees with steps
        # would silently truncate or index out of range at a phase boundary.
        with pytest.raises(ValueError, match="phase-keyed field"):
            OptaxTrainingConfig(**self._kwargs(**{field: value}))

    @pytest.mark.parametrize(
        "points",
        [(jnp.zeros(3),), (jnp.zeros((2, 2, 1)),)],
    )
    def test_non_rank2_penalty_points_raises(self, points):
        # Penalty-only points are [G, n_inputs] physical input vectors; a
        # different rank cannot align with any leaf's input_keys.
        with pytest.raises(ValueError, match="penalty_points"):
            OptaxTrainingConfig(**self._kwargs(penalty_points=points))

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_length_schedule_outside_the_unit_interval_raises(self, fraction):
        # The fraction indexes a prefix of the observation times. Zero would
        # score nothing and above one would claim more timestamps than exist.
        with pytest.raises(ValueError, match="length_schedule"):
            OptaxTrainingConfig(**self._kwargs(length_schedule=(fraction,)))

    def test_length_schedule_of_exactly_one_is_allowed(self):
        # The interval is (0, 1], so the default has to pass.
        assert OptaxTrainingConfig(**self._kwargs(length_schedule=(1.0,))).length_schedule == (1.0,)


class TestBestSnapshotIsPreUpdate:
    """``restore_best`` must return the parameters the winning loss was measured at.

    ``avg_data`` is computed from the pre-update predictors, then the update
    is applied. Snapshotting after the update would hand back a model whose
    loss is not the reported minimum, and ``loss_history`` would no longer
    describe the returned model.
    """

    STEPS = 25
    LR = 0.35  # high enough to overshoot, so the minimum lands mid-run

    def _run(self, dataset, *, restore_best):
        return train_with_optax(
            OmegaPredictor(0.55),
            dataset,
            OptaxTrainingConfig(
                steps=(self.STEPS,),
                lr=(self.LR,),
                optimizer=("adamw",),
                reset_optimiser_state=(False,),
                length_schedule=(1.0,),
                restore_best=restore_best,
                verbose=False,
            ),
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(0),
        )

    def _loss_of(self, dataset, predictor) -> float:
        predictions = predict_dataset(
            predictor,
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
        )
        total = sum(
            float(masked_mse(pred, bp))
            for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True)
        )
        return total / len(dataset.bucket_payloads)

    def test_returned_model_scores_the_minimum_of_the_history(self):
        dataset = make_oscillator_dataset()
        history, trained = self._run(dataset, restore_best=True)
        # Anti-vacuity: if the run were monotone the argmin would be the last
        # step and a post-update snapshot would be nearly indistinguishable.
        argmin = min(range(len(history)), key=history.__getitem__)
        assert argmin < len(history) - 1, "lr was meant to overshoot the minimum"
        assert self._loss_of(dataset, trained) == pytest.approx(min(history), rel=1e-5)

    def test_restore_best_false_returns_the_final_step_instead(self):
        dataset = make_oscillator_dataset()
        best_history, best_model = self._run(dataset, restore_best=True)
        last_history, last_model = self._run(dataset, restore_best=False)

        # restore_best is a choice about what to hand back, not about how to
        # train, so the trajectory must be identical.
        assert best_history == last_history
        assert len(last_history) == self.STEPS

        # The argmin is mid-run, so the final parameters are a different point
        # in the trajectory. No ordering is asserted between their losses: the
        # last update can land anywhere, and often lands better.
        assert float(last_model.omega) != float(best_model.omega)


class TestRestoreBestAcrossHorizons:
    """``restore_best`` must not compare losses measured over different horizons.

    ``length_schedule`` masks the loss to a prefix of each trajectory, so
    a phase at 0.2 and a phase at 1.0 are scoring different quantities.
    Run one running minimum across both and it lands in the short phase
    whenever the model cannot fit the long horizon as tightly, which is
    the situation a curriculum exists to handle. The user then gets the
    least-trained model in the run, silently.

    The dataset here is deliberately unfittable: four oscillators with
    four different true frequencies against a model holding one shared
    ``omega``. Over the first two timestamps every frequency looks alike,
    so the short phase reaches a low loss and carries almost no signal.
    Over the full window no single ``omega`` works and the loss has a
    floor well above it.
    """

    PHASE_1_STEPS = 12
    PHASE_2_STEPS = 20
    SHORT_HORIZON = 0.2
    OMEGAS = (0.6, 0.9, 1.3, 1.7)

    def _dataset(self) -> Dataset:
        ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
        experiments = []
        for i, ((x0, v0), omega) in enumerate(zip(INITIAL_STATES, self.OMEGAS, strict=True)):
            experiments.append(
                make_experiment(
                    covariates={"id": float(i)},
                    channels={
                        "position": ChannelObs(ts=ts, values=true_position(omega, ts, x0, v0))
                    },
                    y0_fn=y0_fn_factory(jnp.asarray([x0, v0])),
                    exp_id=f"exp_{i}",
                )
            )
        return make_dataset(
            experiments,
            output_channel_names=("position",),
        )

    def _full_length_loss(self, dataset: Dataset, predictor) -> float:
        """Data loss over the whole window, the quantity a user cares about."""
        predictions = predict_dataset(
            predictor,
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
        )
        total = sum(
            float(masked_mse(pred, bp))
            for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True)
        )
        return total / len(dataset.bucket_payloads)

    def _run(self, dataset: Dataset, *, steps, length_schedule, restore_best, lr=3e-2):
        history, trained = train_with_optax(
            OmegaPredictor(0.55),
            dataset,
            OptaxTrainingConfig(
                steps=steps,
                lr=(lr,) * len(steps) if isinstance(lr, float) else lr,
                optimizer=("adamw",) * len(steps),
                reset_optimiser_state=(False,) * len(steps),
                length_schedule=length_schedule,
                restore_best=restore_best,
                verbose=False,
            ),
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(0),
        )
        return history, trained

    def test_short_phase_holds_the_global_minimum(self):
        # Anti-vacuity. Without this the next test could pass because the
        # two horizons happened to produce comparable losses, and the bug
        # it guards would never be reachable.
        history, _ = self._run(
            self._dataset(),
            steps=(self.PHASE_1_STEPS, self.PHASE_2_STEPS),
            length_schedule=(self.SHORT_HORIZON, 1.0),
            restore_best=True,
        )
        argmin = min(range(len(history)), key=history.__getitem__)
        assert argmin < self.PHASE_1_STEPS
        assert min(history[: self.PHASE_1_STEPS]) < min(history[self.PHASE_1_STEPS :])

    def test_restored_model_comes_from_the_final_horizon(self):
        dataset = self._dataset()
        _, two_phase = self._run(
            dataset,
            steps=(self.PHASE_1_STEPS, self.PHASE_2_STEPS),
            length_schedule=(self.SHORT_HORIZON, 1.0),
            restore_best=True,
        )
        # What the pre-fix code handed back: the best point of the short phase.
        _, short_only = self._run(
            dataset,
            steps=(self.PHASE_1_STEPS,),
            length_schedule=(self.SHORT_HORIZON,),
            restore_best=True,
        )
        assert self._full_length_loss(dataset, two_phase) < self._full_length_loss(
            dataset, short_only
        )

    def test_a_constant_horizon_still_compares_across_phases(self):
        # The reset is keyed on a *change* in length_schedule. Held flat,
        # the two phases measure the same thing and the running minimum has
        # to survive the boundary.
        #
        # Constructed so the two behaviours are distinguishable: phase 1
        # runs at a learning rate high enough to overshoot, so its best
        # point is mid-phase and its endpoint is worse. Phase 2 runs at
        # zero, so it cannot improve on anything. Carrying the minimum
        # across returns the good mid-phase-1 point; resetting at the
        # boundary would return the phase-1 endpoint instead.
        dataset = self._dataset()
        history, restored = self._run(
            dataset,
            steps=(self.PHASE_1_STEPS, 3),
            length_schedule=(1.0, 1.0),
            restore_best=True,
            lr=(0.4, 0.0),
        )
        _, endpoint = self._run(
            dataset,
            steps=(self.PHASE_1_STEPS, 3),
            length_schedule=(1.0, 1.0),
            restore_best=False,
            lr=(0.4, 0.0),
        )
        argmin = min(range(len(history)), key=history.__getitem__)
        assert argmin < self.PHASE_1_STEPS - 1, "phase 1 was meant to overshoot its own best"
        assert self._full_length_loss(dataset, restored) < self._full_length_loss(dataset, endpoint)
        assert len(history) == self.PHASE_1_STEPS + 3


class TestTournamentSelection:
    """The tournament returns the best candidate, not the first survivor.

    It used to compute a score for every attempt and then return whichever
    one happened to finish first, so ``tournament_attempts`` was a retry
    count and the score was dead work. Nothing caught it, because a
    re-initialised candidate given a short warm-up already beats an
    untrained baseline whether or not selection happens.

    The property tested here is independent of how selection is
    implemented. Attempt keys come from ``fold(key, f"..._{i}")``, so the
    candidate pool for ``n + 1`` attempts contains the pool for ``n``.
    The best achievable score is therefore non-increasing in
    ``tournament_attempts``. Under first-survivor-wins the sequence is
    flat instead.
    """

    ATTEMPT_COUNTS = (2, 3, 4, 5, 6)
    # Chosen because attempt 0 is a poor candidate under it, so the pool
    # has somewhere to improve. Under the default key attempt 0 happens to
    # be the best of six, which would make the test below pass whether or
    # not selection happens. The running best here is
    # 0.561 -> 0.019 -> 0.002, two strict improvements.
    KEY_SEED = 8

    def _config(self, attempts: int) -> OptaxTrainingConfig:
        return OptaxTrainingConfig(
            # One main-loop step at zero learning rate, so what comes back
            # is the tournament's choice and nothing else.
            steps=(1,),
            lr=(0.0,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            length_schedule=(1.0,),
            tournament_attempts=attempts,
            tournament_steps=15,
            tournament_lr=1e-2,
            restore_best=False,
            verbose=False,
        )

    def _winner_loss(self, dataset: Dataset, attempts: int) -> float:
        _history, trained = train_with_optax(
            OmegaPredictor(0.55),
            dataset,
            self._config(attempts),
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
            key=jr.PRNGKey(self.KEY_SEED),
        )
        predictions = predict_dataset(
            trained,
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver_config(),
        )
        return sum(
            float(masked_mse(pred, bp))
            for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True)
        ) / len(dataset.bucket_payloads)

    def test_more_attempts_never_gives_a_worse_winner(self):
        dataset = make_oscillator_dataset()
        losses = [self._winner_loss(dataset, n) for n in self.ATTEMPT_COUNTS]
        for smaller, larger in zip(losses, losses[1:], strict=False):
            assert larger <= smaller + 1e-9, (
                f"a larger attempt pool returned a worse winner: {losses}"
            )

    def test_the_pool_is_not_degenerate(self):
        # Anti-vacuity for the test above. If every candidate scored the
        # same, non-increasing would hold trivially and would also hold
        # under first-survivor-wins. At least one larger pool has to do
        # strictly better.
        dataset = make_oscillator_dataset()
        losses = [self._winner_loss(dataset, n) for n in self.ATTEMPT_COUNTS]
        assert min(losses) < losses[0] - 1e-9, (
            f"no larger pool improved on two attempts, so selection is untested: {losses}"
        )

    def test_selection_is_deterministic_in_the_key(self):
        dataset = make_oscillator_dataset()
        assert self._winner_loss(dataset, 4) == self._winner_loss(dataset, 4)


def test_tournament_retries_equinox_runtime_failures():
    predictors = OmegaPredictor(1.0)
    dataset = make_oscillator_dataset()
    mask = trainable_mask(predictors)

    def score_bucket(_predictors, _bp, _fraction):
        raise eqx.EquinoxRuntimeError("solver failed")

    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        ranked = _shared_tournament(
            predictors,
            dataset,
            bucket_step=lambda *_args: (jnp.asarray(0.0), predictors),
            score_bucket=score_bucket,
            apply_update=lambda *args: args[:2],
            optimizer=optax.sgd(1e-3),
            trainable=mask,
            tournament_attempts=2,
            tournament_steps=0,
            tournament_lr=1e-3,
            key=jr.PRNGKey(0),
            length_mask_fraction=jnp.asarray(1.0),
            top_k=1,
        )
    assert ranked[0][0] == float("inf")
