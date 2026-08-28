"""Integration tests for trajectory-aware penalties (embedded hybrid models).

Two modes, both driven by the same ``trajectory_penalty_fn(full_state, bp)``
hook on the training configs:

- **Embedded** (the predictor runs inside the vector field): the penalty
  rides in the ODE state as extra accumulators
  (``attach_penalty_state`` / ``penalty_vector_field`` /
  ``strip_penalty_state`` / ``penalty_integral``), so the charged value is
  the *time-integral* of ``saturation(z)`` and ``input_violation(x)``
  along the trajectory. Two accumulators, separately tunable.
- **Parallel** (the predictor's output *is* a predicted channel):
  ``trajectory_saturation_penalty`` inverts the outputs back to latents
  and charges saturation over time.

Also covered: probe experiments (a grid with no observations at all — the
data loss is zero, the trajectory penalty still fires), and config
validation.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import hybridmodels as hm
from hybridmodels.data import (
    ChannelObs,
    Dataset,
    make_dataset,
    make_experiment,
)
from hybridmodels.penalties import (
    attach_penalty_state,
    penalty_integral,
    penalty_vector_field,
    strip_penalty_state,
    trajectory_saturation_penalty,
)
from hybridmodels.predictors.base import BoundedPredictor, BoundScaler, Predictor
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.training.kernels import build_bucket_step
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

N_PENALTY = 2  # saturation accumulator + input-violation accumulator


class LatentLeafPredictor(Predictor):
    """One trainable scalar latent ``c``; ignores its input."""

    c: Array

    def __init__(self, c: float) -> None:
        self.c = jnp.asarray(c, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:
        return self.c

    def initialized_with_key(self, key: Array) -> LatentLeafPredictor:
        # Toy leaf: restart at the same scalar. The tournament otherwise
        # re-samples ``c`` from N(0, 1), which would make an assertion
        # like "training moved c away from saturation" vacuous — the
        # restart alone would satisfy it.
        return self


def make_predictor(c: float) -> BoundedPredictor:
    """A bounded predictor whose latent is a single trainable scalar."""
    return BoundedPredictor(
        input_keys=("u",),
        in_scaler=BoundScaler(bounds=((-3.0, 3.0),)),
        inner=LatentLeafPredictor(c),
        out_scaler=BoundScaler(bounds=((0.0, 1.0),)),
    )


def solver() -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-6,
        atol=1e-8,
        max_steps=4096,
        dt0=0.05,
    )


def make_physics_simulate_fn(predictor: BoundedPredictor, probe: bool = False):
    """Embedded hybrid model: predictor inside the vector field, 2 accumulators.

    Physics: a decaying ``x`` with a small ``v``; the predictor reads ``x``
    as its input ``u``. The penalty rates charge output ``saturation`` of
    the predictor's latent and ``input_violation`` of ``x`` against the
    declared box.
    """

    def simulate_fn(predictors, ts, covariates, y0, solver):
        p = predictors[0]

        def physics(t, y, args):
            x, v = y[0], y[1]
            return jnp.stack([-x + 0.1 * v, -0.5 * v])

        def penalty_rates(t, y, args):
            u = y[0]
            z = p.inner(p.in_scaler.to_latent(jnp.asarray(u)))
            sat = p.out_scaler.saturation(z)
            viol = p.in_scaler.input_violation(jnp.asarray(u))
            return jnp.stack([sat, viol])

        term = diffrax.ODETerm(penalty_vector_field(physics, penalty_rates))
        sol = solver.diffeqsolve(term, ts, y0)
        return jnp.asarray(sol.ys)

    return simulate_fn


def embedded_state_to_output(state: Array) -> Array:
    """Drop the 2 penalty accumulators; the observed channels are x and v."""
    return strip_penalty_state(state, N_PENALTY)


def trajectory_penalty_fn(full_state: Array, bp) -> Array:
    """Charge the time-integrals of both accumulators."""
    return jnp.sum(penalty_integral(full_state, N_PENALTY))


def build_embedded_dataset(predictor: BoundedPredictor) -> Dataset:
    """One real experiment: x observed, mask all-True."""

    def y0_fn(c, ch):
        return attach_penalty_state(jnp.array([1.0, 0.0]), N_PENALTY)

    ts = jnp.linspace(0.0, 5.0, 20)
    # Truth trajectory of the physics without the predictor: x decays.
    x_obs = jnp.exp(-ts) + 0.05 * jnp.sin(ts)
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={
            "x": ChannelObs(ts=ts, values=x_obs),
            "v": ChannelObs(ts=ts, values=jnp.zeros_like(ts)),
        },
        y0_fn=y0_fn,
        exp_id="real",
    )
    return make_dataset([exp], output_channel_names=("x", "v"))


def build_probe_dataset(predictor: BoundedPredictor) -> Dataset:
    """A probe experiment: full time grid, zero observations (all-False mask)."""

    def y0_fn(c, ch):
        return attach_penalty_state(jnp.array([3.0, 0.0]), N_PENALTY)

    ts = jnp.linspace(0.0, 5.0, 20)
    exp = make_experiment(
        covariates={"id": 0.0},
        # Empty values: the grid defines where to integrate; nothing is
        # scored, so the data loss is exactly zero and only the
        # trajectory penalty fires.
        channels={
            "x": ChannelObs(ts=ts, values=jnp.array([])),
            "v": ChannelObs(ts=ts, values=jnp.array([])),
        },
        y0_fn=y0_fn,
        exp_id="probe",
    )
    return make_dataset([exp], output_channel_names=("x", "v"))


def bucket_step_loss(predictors, ds: Dataset, cfg: OptaxTrainingConfig):
    """Public-kernel loss with the config's trajectory penalty applied."""
    bp = ds.bucket_payloads[0]
    step = build_bucket_step(
        simulate_fn=make_physics_simulate_fn(predictors[0]),
        state_to_output=embedded_state_to_output,
        solver=solver(),
        loss_fn=hm.masked_mse,
        trainable=trainable_mask(predictors),
        trajectory_penalty_fn=cfg.trajectory_penalty_fn,
        trajectory_penalty_weight=cfg.trajectory_penalty_weight,
    )
    loss, grads = step(predictors, bp, jnp.asarray(1.0))
    return float(loss), grads


class TestPenaltyStateHelpers:
    def test_attach_strip_roundtrip(self):
        y0 = jnp.array([1.0, 2.0])
        assert attach_penalty_state(y0, 2).tolist() == [1.0, 2.0, 0.0, 0.0]
        full = jnp.stack([attach_penalty_state(y0, 2), attach_penalty_state(y0, 2)])
        assert strip_penalty_state(full, 2).shape == (2, 2)

    def test_penalty_integral_reads_final_accumulators(self):
        full = jnp.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # t0
                [1.0, 2.0, 5.0, 6.0],  # t1
            ]
        )
        assert penalty_integral(full, 2).tolist() == [5.0, 6.0]

    def test_penalty_vector_field_concatenates(self):
        def base(t, y, args):
            return jnp.array([-y[0]])

        def pen(t, y, args):
            return jnp.array([y[0] ** 2, jnp.abs(y[1])])

        vf = penalty_vector_field(base, pen)
        out = vf(0.0, jnp.array([1.0, 2.0, 0.0, 0.0]), None)
        assert out.tolist() == [-1.0, 1.0, 2.0]


class TestEmbeddedTrajectoryPenalty:
    def test_penalty_is_zero_for_unsaturated_and_large_for_saturated(self):
        ds_sat = build_embedded_dataset(make_predictor(c=5.0))  # deep saturation
        ds_mild = build_embedded_dataset(make_predictor(c=0.1))
        cfg = OptaxTrainingConfig(
            steps=(1,), lr=(1e-3,), optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=1.0,
        )
        loss_sat, _ = bucket_step_loss((make_predictor(c=5.0),), ds_sat, cfg)
        loss_mild, _ = bucket_step_loss((make_predictor(c=0.1),), ds_mild, cfg)
        assert loss_sat > loss_mild  # saturated latent is penalised more

    def test_gradient_pulls_the_latent_down(self):
        # With only the trajectory penalty charged (weight 1, data loss is
        # small), the gradient on the latent leaf must be non-zero and in
        # the direction that reduces saturation (negative for c > 0).
        ds = build_embedded_dataset(make_predictor(c=3.0))
        cfg = OptaxTrainingConfig(
            steps=(1,), lr=(1e-3,), optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=1.0,
        )
        _, grads = bucket_step_loss((make_predictor(c=3.0),), ds, cfg)
        grad_c = float(grads[0].inner.c)
        assert grad_c != 0.0
        # Positive c, saturation grows with c -> gradient should be positive
        # (increasing the loss), so Adam steps c down.
        assert grad_c > 0.0

    def test_training_reduces_the_penalty(self):
        ds = build_embedded_dataset(make_predictor(c=4.0))
        cfg = OptaxTrainingConfig(
            steps=(15,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=0.5,
            restore_best=False,
        )
        start = (make_predictor(c=4.0),)

        _, trained = train_with_optax(
            start,
            ds,
            cfg,
            simulate_fn=make_physics_simulate_fn(start[0]),
            state_to_output=embedded_state_to_output,
            solver=solver(),
            key=jax.random.PRNGKey(0),
        )
        c_after = float(trained[0].inner.c)
        # The latent moved down (toward 0), away from saturation.
        assert c_after < 4.0


class TestParallelTrajectoryPenalty:
    def test_output_saturation_penalty_fires_and_has_gradient(self):
        # Parallel hybrid: the predictor's output IS the measured channel.
        # No penalty state needed; the hook inverts the outputs.
        def parallel_simulate_fn(predictors, ts, covariates, y0, solver):
            p = predictors[0]
            # The predictor output is constant over time (the "channel").
            out = p({"u": 0.0})  # returns the physical output in (0, 1)
            return jnp.broadcast_to(out[None], (ts.shape[0], 1))

        def parallel_state_to_output(state):
            return state  # the state IS the channel

        out_scaler = make_predictor(0.0).out_scaler

        def sat_penalty(full_state, bp):
            return trajectory_saturation_penalty(full_state, out_scaler)

        ts = jnp.linspace(0.0, 5.0, 10)
        exp = make_experiment(
            covariates={"id": 0.0},
            channels={"y": ChannelObs(ts=ts, values=jnp.zeros_like(ts))},
            y0_fn=lambda c, ch: jnp.zeros(1),
            exp_id="e0",
        )
        ds = make_dataset([exp], output_channel_names=("y",))

        sat_pred = (make_predictor(c=6.0),)  # latent 6 -> output pinned at 1.0
        step = build_bucket_step(
            simulate_fn=parallel_simulate_fn,
            state_to_output=parallel_state_to_output,
            solver=solver(),
            loss_fn=hm.masked_mse,
            trainable=trainable_mask(sat_pred),
            trajectory_penalty_fn=sat_penalty,
            trajectory_penalty_weight=1.0,
        )
        loss, grads = step(sat_pred, ds.bucket_payloads[0], jnp.asarray(1.0))
        assert float(loss) > 0.0
        grad_c = float(grads[0].inner.c)
        assert grad_c != 0.0
        assert grad_c > 0.0  # pulls the latent down, away from the bound

    def test_parallel_penalty_through_train_with_optax(self):
        # End to end through the stock trainer: the saturation penalty must
        # actually move the latent, not just register a gradient.
        def parallel_simulate_fn(predictors, ts, covariates, y0, solver):
            p = predictors[0]
            out = p({"u": 0.0})
            return jnp.broadcast_to(out[None], (ts.shape[0], 1))

        def parallel_state_to_output(state):
            return state

        out_scaler = make_predictor(0.0).out_scaler

        def sat_penalty(full_state, bp):
            return trajectory_saturation_penalty(full_state, out_scaler)

        ts = jnp.linspace(0.0, 5.0, 10)
        exp = make_experiment(
            covariates={"id": 0.0},
            channels={"y": ChannelObs(ts=ts, values=jnp.zeros_like(ts))},
            y0_fn=lambda c, ch: jnp.zeros(1),
            exp_id="e0",
        )
        ds = make_dataset([exp], output_channel_names=("y",))

        start = (make_predictor(c=6.0),)
        cfg = OptaxTrainingConfig(
            steps=(15,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=sat_penalty,
            trajectory_penalty_weight=1.0,
            restore_best=False,
        )
        _, trained = train_with_optax(
            start,
            ds,
            cfg,
            simulate_fn=parallel_simulate_fn,
            state_to_output=parallel_state_to_output,
            solver=solver(),
            key=jax.random.PRNGKey(0),
        )
        # The latent starts deep in saturation (output pinned at 1.0) while
        # the observations are all zero; both the data loss and the penalty
        # pull it down toward 0.
        assert float(trained[0].inner.c) < 6.0


class TestProbeExperiments:
    def test_probe_mask_is_all_false(self):
        ds = build_probe_dataset(make_predictor(c=1.0))
        bp = ds.bucket_payloads[0]
        assert int(jnp.sum(bp.mask)) == 0
        assert bp.ts.shape[1] == 20  # grid still defined

    def test_probe_scores_zero_data_loss_but_fires_penalty(self):
        ds = build_probe_dataset(make_predictor(c=5.0))
        cfg = OptaxTrainingConfig(
            steps=(1,), lr=(1e-3,), optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=1.0,
        )
        # Data-only loss on the probe is exactly zero (all-False mask)...
        no_pen_cfg = OptaxTrainingConfig(
            steps=(1,), lr=(1e-3,), optimizer=("adamw",),
            reset_optimiser_state=(False,),
        )
        data_loss, _ = bucket_step_loss((make_predictor(c=5.0),), ds, no_pen_cfg)
        assert data_loss == pytest.approx(0.0, abs=1e-6)
        # ...and the trajectory penalty still fires on the probe grid.
        pen_loss, _ = bucket_step_loss((make_predictor(c=5.0),), ds, cfg)
        assert pen_loss > data_loss

    def test_probe_drives_training_through_train_with_optax(self):
        # A probe is a pure steering scenario: no measurements, so the only
        # gradient in the loop comes from the trajectory penalty. The stock
        # trainer must still make progress on it.
        ds = build_probe_dataset(make_predictor(c=5.0))
        start = (make_predictor(c=5.0),)
        cfg = OptaxTrainingConfig(
            steps=(20,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=1.0,
            restore_best=False,
        )
        _, trained = train_with_optax(
            start,
            ds,
            cfg,
            simulate_fn=make_physics_simulate_fn(start[0]),
            state_to_output=embedded_state_to_output,
            solver=solver(),
            key=jax.random.PRNGKey(0),
        )
        assert float(trained[0].inner.c) < 5.0  # penalty pulled it out of saturation


class TestConfigValidation:
    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            OptaxTrainingConfig(
                steps=(1,), lr=(1e-3,), optimizer=("adamw",),
                reset_optimiser_state=(False,),
                trajectory_penalty_weight=-1.0,
            )
        with pytest.raises(ValueError, match="non-negative"):
            hm.EvosaxTrainingConfig(trajectory_penalty_weight=-1.0)

    def test_weight_without_fn_raises(self):
        with pytest.raises(ValueError, match="trajectory_penalty_fn is None"):
            OptaxTrainingConfig(
                steps=(1,), lr=(1e-3,), optimizer=("adamw",),
                reset_optimiser_state=(False,),
                trajectory_penalty_weight=1.0,
            )
        with pytest.raises(ValueError, match="trajectory_penalty_fn is None"):
            hm.EvosaxTrainingConfig(trajectory_penalty_weight=1.0)

    def test_zero_weight_with_fn_is_a_noop(self):
        cfg = OptaxTrainingConfig(
            steps=(1,), lr=(1e-3,), optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=0.0,
        )
        assert cfg.trajectory_penalty_weight == 0.0


class TestTrajectoryPenaltyEndToEnd:
    """Coverage for the stock-trainer combinations the unit hooks miss.

    The kernel-level tests above prove the penalty has a gradient; these
    prove it survives the full pipelines — evosax ranking and bootstrap
    bagging — that a user would actually run.
    """

    def test_evosax_trajectory_penalty_moves_the_latent(self):
        ds = build_embedded_dataset(make_predictor(c=4.0))
        start = (make_predictor(c=4.0),)
        cfg = hm.EvosaxTrainingConfig(
            num_generations=20,
            population_size=16,
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=1.0,
            sigma_init=0.05,
        )
        history, best = hm.train_with_evosax(
            start,
            ds,
            cfg,
            simulate_fn=make_physics_simulate_fn(start[0]),
            state_to_output=embedded_state_to_output,
            solver=solver(),
            key=jax.random.PRNGKey(0),
        )
        assert len(history) == 20
        # The saturated latent (4.0) is penalised; the search must land
        # closer to the unsaturated region than it started.
        assert float(best[0].inner.c) < 4.0

    def test_bootstrap_ensemble_with_trajectory_penalty(self):
        from hybridmodels.training.optax import train_bootstrap_ensemble

        ds = build_embedded_dataset(make_predictor(c=4.0))
        start = (make_predictor(c=4.0),)
        cfg = OptaxTrainingConfig(
            steps=(10,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            trajectory_penalty_fn=trajectory_penalty_fn,
            trajectory_penalty_weight=0.5,
            restore_best=False,
        )
        members = train_bootstrap_ensemble(
            start,
            ds,
            cfg,
            simulate_fn=make_physics_simulate_fn(start[0]),
            state_to_output=embedded_state_to_output,
            solver=solver(),
            n_bootstraps=2,
            n_seeds=1,
            key=jax.random.PRNGKey(0),
        )
        assert len(members) == 2
        for final_loss, member in members:
            assert float(member[0].inner.c) < 4.0  # penalty pulled out of saturation
            assert jnp.isfinite(final_loss)
