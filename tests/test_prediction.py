"""Tests for the public inference path.

``prediction.py`` had no test file, which is why it sat at 65% coverage
while being the entry point every user calls after training.

Each claim is checked against an oracle that does not share the
implementation's machinery: an un-vmapped Python loop for the numerics,
the bucket payloads' own shapes for the layout, and a trace-counting
``simulate_fn`` for the compile-caching contract.
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import pytest
from _harness import OMEGA_TRUE, OmegaPredictor, solver_config, y0_fn_factory
from jax import Array

from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from hybridmodels.prediction import predict_bucket, predict_dataset
from hybridmodels.solver import SolverConfig


def _solver() -> SolverConfig:
    """Tighter than the training tests on purpose.

    The oracle here compares an un-vmapped Python loop against the vmapped
    path, so integration error has to sit well below the comparison
    tolerance.
    """
    return solver_config(rtol=1e-6, atol=1e-8)


def _state_to_output(full_state: Array) -> Array:
    return full_state[:, :1]


def _simulate_fn(predictor, ts, covariates, y0, solver):
    omega = predictor.omega

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
        stepsize_controller=solver.stepsize_controller(),
        max_steps=solver.max_steps,
        adjoint=solver.adjoint,
    )
    return sol.ys


def _experiment(n_steps: int, x0: float, v0: float, exp_id: str):
    ts = jnp.linspace(0.0, 4.0, n_steps)
    values = x0 * jnp.cos(OMEGA_TRUE * ts) + (v0 / OMEGA_TRUE) * jnp.sin(OMEGA_TRUE * ts)
    return make_experiment(
        covariates={"x0": x0},
        channels={"position": ChannelObs(ts=ts, values=values)},
        y0_fn=y0_fn_factory(jnp.asarray([x0, v0])),
        exp_id=exp_id,
    )


def _dataset(shapes=((6, 1.0, 0.0), (6, 0.0, 1.0), (9, 0.5, -0.5))) -> Dataset:
    """Two experiments share a T; the third differs, so there are two buckets."""
    exps = [_experiment(n, x0, v0, f"e{i}") for i, (n, x0, v0) in enumerate(shapes)]
    return make_dataset(exps, output_channel_names=("position",))


class TestPredictBucket:
    def test_vector_covariates_reach_each_vmapped_simulation(self):
        ts = jnp.array([0.0, 1.0])
        experiments = [
            make_experiment(
                covariates={"features": jnp.array([1.0, 2.0])},
                channels={"position": ChannelObs(ts=ts, values=jnp.zeros_like(ts))},
                y0_fn=lambda _cov, _channels: jnp.zeros((1,)),
            ),
            make_experiment(
                covariates={"features": jnp.array([3.0, 4.0])},
                channels={"position": ChannelObs(ts=ts, values=jnp.zeros_like(ts))},
                y0_fn=lambda _cov, _channels: jnp.zeros((1,)),
            ),
        ]
        bp = make_dataset(experiments, output_channel_names=("position",)).bucket_payloads[0]

        def simulate_fn(_predictors, ts, covariates, _y0, _solver):
            value = covariates["features"].sum()
            return jnp.broadcast_to(value, (ts.shape[0], 1))

        got = predict_bucket(
            OmegaPredictor(OMEGA_TRUE),
            bp,
            simulate_fn=simulate_fn,
            state_to_output=lambda state: state,
            solver=_solver(),
        )
        assert jnp.array_equal(got[:, 0, 0], jnp.array([3.0, 7.0]))

    def test_matches_an_unvmapped_python_loop(self):
        # The oracle: call simulate_fn once per experiment and project, with
        # no vmap anywhere. Independent of the batching under test.
        ds = _dataset()
        pred = OmegaPredictor(OMEGA_TRUE)
        solver = _solver()
        bp = ds.bucket_payloads[0]

        got = predict_bucket(
            pred,
            bp,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=solver,
        )

        expected = jnp.stack(
            [
                _state_to_output(
                    _simulate_fn(
                        pred,
                        bp.ts[i],
                        {k: v[i] for k, v in bp.covariates.items()},
                        bp.y0[i],
                        solver,
                    )
                )
                for i in range(bp.ts.shape[0])
            ]
        )
        assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_output_shape_is_n_t_d(self):
        ds = _dataset()
        bp = ds.bucket_payloads[0]
        out = predict_bucket(
            OmegaPredictor(OMEGA_TRUE),
            bp,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        assert out.shape == bp.y_observed.shape

    def test_recovers_the_generating_trajectory_at_the_true_parameter(self):
        # An oracle outside the framework entirely: the analytic solution
        # the data was generated from.
        ds = _dataset()
        bp = ds.bucket_payloads[0]
        out = predict_bucket(
            OmegaPredictor(OMEGA_TRUE),
            bp,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        assert jnp.allclose(out, bp.y_observed, atol=1e-4)


class TestPredictDataset:
    def test_returns_one_array_per_bucket_in_payload_order(self):
        ds = _dataset()
        outs = predict_dataset(
            OmegaPredictor(OMEGA_TRUE),
            ds,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        assert isinstance(outs, tuple)
        assert len(outs) == len(ds.bucket_payloads)
        for out, bp in zip(outs, ds.bucket_payloads, strict=True):
            assert out.shape == bp.y_observed.shape

    def test_buckets_differ_in_t_so_the_result_cannot_be_stacked(self):
        # Documents why a tuple is returned rather than one concatenated
        # array: the per-bucket T genuinely differs.
        ds = _dataset()
        widths = {bp.ts.shape[1] for bp in ds.bucket_payloads}
        assert len(widths) > 1
        outs = predict_dataset(
            OmegaPredictor(OMEGA_TRUE),
            ds,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        with pytest.raises(Exception):  # noqa: B017
            jnp.stack(outs)

    def test_agrees_with_predict_bucket_called_directly(self):
        ds = _dataset()
        pred = OmegaPredictor(OMEGA_TRUE)
        solver = _solver()
        outs = predict_dataset(
            pred,
            ds,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=solver,
        )
        for out, bp in zip(outs, ds.bucket_payloads, strict=True):
            direct = predict_bucket(
                pred,
                bp,
                simulate_fn=_simulate_fn,
                state_to_output=_state_to_output,
                solver=solver,
            )
            assert jnp.allclose(out, direct)

    def test_state_to_output_is_an_explicit_parameter(self):
        # predict_dataset takes the projection as a keyword parameter rather
        # than reading it off the Dataset, so the caller owns the swap
        # (ADR-0008).
        ds = _dataset()
        ts = jnp.linspace(0.0, 4.0, 6)
        two_channel = make_experiment(
            covariates={"x0": 1.0},
            channels={
                "position": ChannelObs(ts=ts, values=jnp.cos(ts)),
                "velocity": ChannelObs(ts=ts, values=-jnp.sin(ts)),
            },
            y0_fn=y0_fn_factory(jnp.asarray([1.0, 0.0])),
            exp_id="two",
        )
        both_channels = make_dataset(
            [two_channel],
            output_channel_names=("position", "velocity"),
        )
        out = predict_dataset(
            OmegaPredictor(OMEGA_TRUE),
            both_channels,
            simulate_fn=_simulate_fn,
            state_to_output=lambda s: s,
            solver=_solver(),
        )
        assert out[0].shape[-1] == 2
        assert ds.bucket_payloads[0].y_observed.shape[-1] == 1


class TestCompileCaching:
    """One compiled trace per bucket *shape*, and the dispatch loop stays in Python."""

    def test_one_trace_per_bucket_shape(self):
        traces = {"n": 0}

        def counting_simulate_fn(predictor, ts, covariates, y0, solver):
            traces["n"] += 1
            return _simulate_fn(predictor, ts, covariates, y0, solver)

        ds = _dataset()
        assert len({bp.ts.shape for bp in ds.bucket_payloads}) == 2
        pred = OmegaPredictor(OMEGA_TRUE)

        predict_dataset(
            pred,
            ds,
            simulate_fn=counting_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        first = traces["n"]
        assert first == 2, "expected one trace per distinct bucket shape"

        # Re-running the same shapes must hit the cache and trace nothing.
        predict_dataset(
            pred,
            ds,
            simulate_fn=counting_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        assert traces["n"] == first

    def test_changing_only_parameter_values_does_not_retrace(self):
        traces = {"n": 0}

        def counting_simulate_fn(predictor, ts, covariates, y0, solver):
            traces["n"] += 1
            return _simulate_fn(predictor, ts, covariates, y0, solver)

        ds = _dataset()
        predict_dataset(
            OmegaPredictor(1.0),
            ds,
            simulate_fn=counting_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        before = traces["n"]
        predict_dataset(
            OmegaPredictor(1.3),
            ds,
            simulate_fn=counting_simulate_fn,
            state_to_output=_state_to_output,
            solver=_solver(),
        )
        assert traces["n"] == before

    def test_prediction_and_training_kernels_do_not_share_a_cache(self):
        # prediction.py jits separately from the training kernels precisely
        # so the two graphs cannot collide. Build a training step over the
        # same bucket and check prediction still traces its own.
        from hybridmodels.trainable import trainable_mask
        from hybridmodels.training.kernels import build_bucket_step

        traces = {"n": 0}

        def counting_simulate_fn(predictor, ts, covariates, y0, solver):
            traces["n"] += 1
            return _simulate_fn(predictor, ts, covariates, y0, solver)

        ds = _dataset()
        pred = OmegaPredictor(OMEGA_TRUE)
        solver = _solver()
        bp = ds.bucket_payloads[0]

        bucket_step = build_bucket_step(
            simulate_fn=counting_simulate_fn,
            state_to_output=_state_to_output,
            solver=solver,
            loss_fn=lambda p, b: jnp.sum(jnp.where(b.mask, (p - b.y_observed) ** 2, 0.0)),
            trainable=trainable_mask(pred),
        )
        bucket_step(pred, bp, jnp.asarray(1.0))
        after_training = traces["n"]
        assert after_training > 0

        predict_bucket(
            pred,
            bp,
            simulate_fn=counting_simulate_fn,
            state_to_output=_state_to_output,
            solver=solver,
        )
        assert traces["n"] > after_training, "prediction reused the training trace"
