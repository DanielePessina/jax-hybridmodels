"""End-to-end smoke run for losses + prediction (Phase 5).

Builds a tiny harmonic-oscillator dataset, defines a SPEC §4.2-shaped
``simulate_fn``, runs ``predict_dataset`` with a fixed scalar predictor
(MLP -> sigmoid-bounded omega), and prints per-bucket prediction shapes
and ``masked_mse`` losses against ``bp.y_observed``.

Run:
    uv run python scripts/smoke_predict.py
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import jax.random as jr

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    CovariateSelector,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    make_experiment,
    masked_mse,
    predict_dataset,
)


def _y0_fn(covariates: dict[str, jnp.ndarray], channels: dict[str, ChannelObs]) -> jnp.ndarray:
    initial_x = float(channels["x"].values[0])
    return jnp.array([initial_x, 0.0])


def _state_to_output(state: jnp.ndarray) -> jnp.ndarray:
    return state[..., :1]


def _simulate_fn(
    predictor: BoundedPredictor,
    ts: jnp.ndarray,
    covariates: dict[str, jnp.ndarray],
    y0: jnp.ndarray,
    solver: SolverConfig,
) -> jnp.ndarray:
    omega = predictor(covariates).reshape(())
    omega_sq = omega * omega

    def vector_field(t: jnp.ndarray, y: jnp.ndarray, args: object) -> jnp.ndarray:
        return jnp.stack([y[1], -omega_sq * y[0]])

    term = diffrax.ODETerm(vector_field)
    sol = diffrax.diffeqsolve(
        term,
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


def _build_experiments() -> list:
    rng = jr.PRNGKey(0)
    specs = [
        ("exp_A", 1.0, [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("exp_B", 1.5, [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]),
        ("exp_C", 0.7, [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("exp_D", 1.2, [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]),
    ]
    experiments = []
    for exp_id, omega_in, ts in specs:
        rng, k = jr.split(rng)
        ts_arr = jnp.asarray(ts)
        x_vals = jnp.cos(omega_in * ts_arr) + 0.01 * jr.normal(k, (len(ts),))
        channels = {"x": ChannelObs(ts=ts_arr, values=x_vals, variance=1e-3)}
        experiments.append(
            make_experiment(
                covariates={"omega_input": omega_in},
                channels=channels,
                y0_fn=_y0_fn,
                exp_id=exp_id,
            )
        )
    return experiments


def main() -> None:
    experiments = _build_experiments()
    dataset = make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=("x",),
    )
    print(f"dataset has {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(f"  bucket {i}: ts {bp.ts.shape}, y_observed {bp.y_observed.shape}")

    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-6,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.01,
    )

    mlp = MLPPredictor(
        in_size=1,
        out_size=1,
        width_size=8,
        depth=1,
        activation_name="tanh",
        key=jr.PRNGKey(7),
    )
    predictor = BoundedPredictor(
        selector=CovariateSelector(keys=("omega_input",)),
        in_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
        inner=mlp,
        out_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
    )

    preds = predict_dataset(predictor, dataset, simulate_fn=_simulate_fn, solver=solver)

    for i, (bp, pred) in enumerate(zip(dataset.bucket_payloads, preds, strict=True)):
        loss = masked_mse(pred, bp)
        loss_value = float(loss)
        print(
            f"bucket {i}: pred shape={pred.shape}, "
            f"mask shape={bp.mask.shape}, masked_mse={loss_value:.6f}"
        )
        assert jnp.isfinite(loss), f"non-finite loss at bucket {i}"
        assert loss_value >= 0.0, f"negative loss at bucket {i}: {loss_value}"

    print("smoke OK")


if __name__ == "__main__":
    main()
