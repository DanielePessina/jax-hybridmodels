"""Integration tests for the WS3 helper layer folded into the library.

Covers the four fold-in candidates from the examples:

- ``hybridmodels.metrics`` — masked per-channel MSE/RMSE/MAE/R^2.
- ``SolverConfig.diffeqsolve`` — the invocation boilerplate, forwarding
  ``solver.adjoint`` (the field most hand-written examples ignored).
- ``describe_buckets`` / ``count_trainable_params`` / ``evaluate_predictor``
  / ``frozen_default_mask`` — the small conveniences.

The metrics tests build a real dataset and predictions through the public
``predict_bucket`` path, asserting the numbers match hand-computed values
under the mask.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import pytest
from _harness import (
    OmegaPredictor,
    make_oscillator_dataset,
    make_oscillator_simulate_fn,
    oscillator_state_to_output,
)

from hybridmodels.data import Dataset
from hybridmodels.metrics import compute_metrics, print_metrics
from hybridmodels.prediction import evaluate_predictor, predict_bucket
from hybridmodels.solver import ADJOINT_REGISTRY, SolverConfig
from hybridmodels.trainable import (
    count_trainable_params,
    freeze_modules_of_type,
    frozen_default_mask,
    trainable_mask,
)


def describe_buckets(dataset: Dataset) -> str:
    from hybridmodels.data import describe_buckets as _db

    return _db(dataset)


def test_compute_metrics_matches_hand_computed_values():
    ds = make_oscillator_dataset()
    pred = OmegaPredictor(1.0)  # omega == truth -> zero error
    predictions = tuple(
        predict_bucket(
            pred,
            bp,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=SolverConfig(
                solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=0.05
            ),
        )
        for bp in ds.bucket_payloads
    )
    metrics = compute_metrics(predictions, ds)
    assert set(metrics) == {"position"}
    m = metrics["position"]
    assert m.n == sum(int(jnp.sum(bp.mask)) for bp in ds.bucket_payloads)
    # At the truth, all errors are ~0 and R^2 ~ 1.
    assert float(m.mse) < 1e-6
    assert float(m.rmse) < 1e-3
    assert float(m.r2) > 0.99

    # A deliberately wrong predictor gives a large, finite error.
    bad = OmegaPredictor(2.0)
    predictions_bad = tuple(
        predict_bucket(
            bad,
            bp,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=SolverConfig(
                solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=0.05
            ),
        )
        for bp in ds.bucket_payloads
    )
    m_bad = compute_metrics(predictions_bad, ds)["position"]
    assert float(m_bad.mse) > float(m.mse)


def test_compute_metrics_respects_the_mask():
    # Build a dataset whose union mask is genuinely PARTIAL, so an
    # implementation that ignored the mask would report a different n and
    # different error than the mask-respecting one.
    from hybridmodels.data import ChannelObs, make_dataset, make_experiment

    ts_full = jnp.linspace(0.0, 5.0, 10)
    # "position" is observed only on the first 4 timestamps; "velocity" is
    # observed on all 10, so the union axis is 10 and position's mask is
    # genuinely partial (4 of 10).
    ts_sparse = ts_full[:4]
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={
            "position": ChannelObs(ts=ts_sparse, values=jnp.array([1.0, 0.9, 0.6, 0.1])),
            "velocity": ChannelObs(ts=ts_full, values=jnp.cos(ts_full)),
        },
        y0_fn=lambda c, ch: jnp.array([1.0, 0.0]),
        exp_id="e0",
    )
    ds = make_dataset([exp], output_channel_names=("position", "velocity"))
    bp = ds.bucket_payloads[0]
    assert int(bp.mask.shape[1]) == 10  # union axis spans both channels
    assert int(jnp.sum(bp.mask[..., 0])) == 4  # position genuinely partial
    assert int(jnp.sum(bp.mask[..., 1])) == 10  # velocity fully observed

    predictions = tuple(
        predict_bucket(
            OmegaPredictor(1.0),
            bp,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=SolverConfig(
                solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=0.05
            ),
        )
        for bp in ds.bucket_payloads
    )
    metrics = compute_metrics(predictions, ds)
    assert metrics["position"].n == 4
    assert metrics["velocity"].n == 10


def test_compute_metrics_constant_observations_give_nan_r2():
    # R^2 is undefined when the observed values are constant (SS_tot = 0).
    constant_ds = _constant_observation_dataset()
    pred = OmegaPredictor(1.0)
    predictions = tuple(
        predict_bucket(
            pred,
            bp,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=SolverConfig(
                solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=0.05
            ),
        )
        for bp in constant_ds.bucket_payloads
    )
    m = compute_metrics(predictions, constant_ds)["position"]
    assert float(m.r2) != float(m.r2)  # NaN


def _constant_observation_dataset() -> Dataset:
    from hybridmodels.data import ChannelObs, make_dataset, make_experiment

    ts = jnp.linspace(0.0, 5.0, 10)
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={"position": ChannelObs(ts=ts, values=jnp.full(ts.shape, 2.0))},
        y0_fn=lambda c, ch: jnp.array([2.0, 0.0]),
        exp_id="e0",
    )
    return make_dataset([exp], output_channel_names=("position",))


def test_print_metrics_runs_and_captures_header(capsys):
    ds = make_oscillator_dataset()
    pred = OmegaPredictor(1.0)
    predictions = tuple(
        predict_bucket(
            pred,
            bp,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=SolverConfig(
                solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=0.05
            ),
        )
        for bp in ds.bucket_payloads
    )
    print_metrics(compute_metrics(predictions, ds), header="RESULTS")
    out = capsys.readouterr().out
    assert "RESULTS" in out
    assert "position" in out


def test_describe_buckets_reports_shape_and_mask():
    ds = make_oscillator_dataset()
    text = describe_buckets(ds)
    assert "1 bucket(s)" in text
    assert "N=" in text and "T=" in text and "D=" in text
    assert "mask" in text


def test_count_trainable_params_counts_mask_selected_leaves():
    preds = (OmegaPredictor(1.0), OmegaPredictor(1.0))
    full = trainable_mask(preds)
    assert count_trainable_params(preds, full) == 2
    frozen = frozen_default_mask(preds)  # nothing to freeze here, so 2 remain
    assert count_trainable_params(preds, frozen) == 2


def test_frozen_default_mask_freezes_module_types():
    from hybridmodels.predictors.base import BoundedPredictor, BoundScaler
    from hybridmodels.predictors.mlp import MLPPredictor

    inner = MLPPredictor(in_size=1, out_size=1, width_size=4, depth=1, key=jax.random.PRNGKey(0))
    bp = BoundedPredictor(
        input_keys=("x",),
        in_scaler=BoundScaler(bounds=((0.0, 1.0),)),
        inner=inner,
        out_scaler=BoundScaler(bounds=((0.0, 1.0),)),
    )
    mask = frozen_default_mask(bp, BoundScaler)
    # Everything in the BoundScalers (their temperature) is frozen.
    frozen_mask = freeze_modules_of_type(mask, bp, BoundScaler)
    assert count_trainable_params(bp, mask) == count_trainable_params(bp, frozen_mask)
    # And at least something is still trainable (the MLP weights).
    assert count_trainable_params(bp, mask) > 0


def test_evaluate_predictor_reads_scalar():
    pred = OmegaPredictor(1.234)
    value = evaluate_predictor(pred, {})
    assert isinstance(value, float)
    assert value == pytest.approx(1.234)


def test_diffeqsolve_forwards_adjoint_and_matches_handwritten():
    # The whole point: SolverConfig.diffeqsolve must forward solver.adjoint,
    # so Backsolve (which needs args-threading) and Direct both work.
    from hybridmodels.data import ChannelObs, make_dataset, make_experiment

    ts = jnp.linspace(0.0, 5.0, 10)
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={"x": ChannelObs(ts=ts, values=jnp.cos(ts))},
        y0_fn=lambda c, ch: jnp.array([float(ch["x"].values[0]), 0.0]),
        exp_id="e0",
    )
    ds = make_dataset([exp], output_channel_names=("x",))
    bp = ds.bucket_payloads[0]

    def args_threaded_simulate_fn(predictor, ts, covariates, y0, solver):
        def vf(t, y, args):
            omega = args[0].omega
            return jnp.stack([y[1], -(omega**2) * y[0]])

        sol = solver.diffeqsolve(
            diffrax.ODETerm(vf), ts, y0, args=predictor
        )
        return jnp.asarray(sol.ys)

    for name in ("Direct", "RecursiveCheckpoint", "Backsolve"):
        cfg = SolverConfig(
            solver=diffrax.Tsit5(),
            rtol=1e-6,
            atol=1e-8,
            max_steps=4096,
            dt0=0.05,
            adjoint=ADJOINT_REGISTRY[name](),
        )
        ys = args_threaded_simulate_fn((OmegaPredictor(1.0),), ts, {}, bp.y0[0], cfg)
        assert ys.shape == (10, 2)
        # Position should be ~cos(t).
        assert float(ys[0, 0]) == pytest.approx(1.0, abs=1e-3)
        assert float(ys[-1, 0]) == pytest.approx(float(jnp.cos(5.0)), abs=1e-3)


def test_diffeqsolve_forwards_adjoint_at_gradient_level():
    # Proves the method really forwards self.adjoint: Backsolve cannot
    # differentiate through values closed over in the vector field, so if
    # diffeqsolve hardcoded DirectAdjoint (the old example bug) this would
    # silently succeed; with the adjoint forwarded it raises. The forward
    # solution is adjoint-independent, so a forward-only check cannot tell
    # the difference — only a gradient does.
    import equinox as eqx

    from hybridmodels.data import ChannelObs, make_dataset, make_experiment

    ts = jnp.linspace(0.0, 5.0, 10)
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={"x": ChannelObs(ts=ts, values=jnp.cos(ts))},
        y0_fn=lambda c, ch: jnp.array([float(ch["x"].values[0]), 0.0]),
        exp_id="e0",
    )
    ds = make_dataset([exp], output_channel_names=("x",))
    bp = ds.bucket_payloads[0]
    back_solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-6,
        atol=1e-8,
        max_steps=4096,
        dt0=0.05,
        adjoint=ADJOINT_REGISTRY["Backsolve"](),
    )

    def closed_over_simulate_fn(predictor, ts, covariates, y0, solver):
        # NOTE: predictors closed over in the vector field, not threaded
        # through args — the canonical (wrong-for-Backsolve) shape.
        omega = predictor[0].omega

        def vf(t, y, args):
            return jnp.stack([y[1], -(omega**2) * y[0]])

        sol = solver.diffeqsolve(diffrax.ODETerm(vf), ts, y0)
        return jnp.asarray(sol.ys)

    def loss(predictors):
        pred = predict_bucket(
            predictors,
            bp,
            simulate_fn=closed_over_simulate_fn,
            state_to_output=oscillator_state_to_output,
            solver=back_solver,
        )
        return jnp.sum((pred - bp.y_observed) ** 2)

    with pytest.raises(Exception, match="closed-over"):
        eqx.filter_grad(loss)((OmegaPredictor(1.0),))
