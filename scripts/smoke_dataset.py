"""End-to-end smoke run over the currently-shipped surface (data + solver + predictors).

Run with:

    uv run python scripts/smoke_dataset.py
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import jax.random as jr

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Dataset,
    Experiment,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    make_experiment,
    split_dataset,
)

OUTPUT_CHANNELS = ("conc", "d43")


def _y0_fn(covariates: dict[str, jnp.ndarray], channels: dict[str, ChannelObs]) -> jnp.ndarray:
    initial_conc = float(channels["conc"].values[0])
    return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, initial_conc])


def _build_experiments() -> list[Experiment]:
    rng = jr.PRNGKey(0)
    experiments: list[Experiment] = []

    specs = [
        ("exp_A", 25.0, 0.10, [0.0, 1.0, 2.0, 3.0], [0.0, 2.0]),
        ("exp_B", 25.0, 0.20, [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
        ("exp_C", 30.0, 0.10, [0.0, 0.5, 1.0, 1.5, 2.0], [0.5, 1.5]),
        ("exp_D", 30.0, 0.20, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0]),
        ("exp_E", 35.0, 0.15, [0.0, 1.0, 2.5, 4.0], [1.0, 2.5, 4.0]),
        ("exp_F", 35.0, 0.25, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0], [0.5, 4.0]),
    ]

    for exp_id, temp, loading, conc_ts, d43_ts in specs:
        rng, k_conc, k_d43 = jr.split(rng, 3)
        conc_vals = 1.0 - 0.1 * jnp.asarray(conc_ts) + 0.01 * jr.normal(k_conc, (len(conc_ts),))
        d43_vals = 5.0 + 2.0 * jnp.asarray(d43_ts) + 0.05 * jr.normal(k_d43, (len(d43_ts),))
        channels = {
            "conc": ChannelObs(ts=jnp.asarray(conc_ts), values=conc_vals, variance=1e-4),
            "d43": ChannelObs(ts=jnp.asarray(d43_ts), values=d43_vals, variance=1e-3),
        }
        experiments.append(
            make_experiment(
                covariates={"temperature_C": temp, "loading": loading},
                channels=channels,
                y0_fn=_y0_fn,
                exp_id=exp_id,
            )
        )
    return experiments


def _state_to_output(state: jnp.ndarray) -> jnp.ndarray:
    conc = state[..., 5:6]
    d43 = state[..., 4:5]
    return jnp.concatenate([conc, d43], axis=-1)


def _print_buckets(label: str, dataset: Dataset) -> None:
    print(f"\n[{label}] {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(
            f"  bucket {i}: ts {bp.ts.shape} | y_observed {bp.y_observed.shape} "
            f"| mask {bp.mask.shape} | y0 {bp.y0.shape} | n_obs={int(bp.n_obs)}"
        )


def main() -> None:
    print("=== Phase 1: data ===")
    experiments = _build_experiments()
    dataset = make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    _print_buckets("full", dataset)

    train, val, test = split_dataset(dataset, train=0.6, val=0.2, test=0.2, key=jr.PRNGKey(42))
    print(
        f"\nsplit sizes: train={len(train._experiments)}, "
        f"val={len(val._experiments)}, test={len(test._experiments)}"
    )
    _print_buckets("train", train)
    _print_buckets("val", val)
    _print_buckets("test", test)

    print("\n=== Phase 2: solver ===")
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=(1e-5,) * 6,
        max_steps=500_000,
        dt0=None,
    )
    print(f"solver.to_dict() = {solver.to_dict()}")

    print("\n=== Phase 3/4: predictors ===")
    mlp = MLPPredictor(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        activation_name="tanh",
        key=jr.PRNGKey(7),
    )
    raw_input = jnp.array([0.5, -0.3])
    print(f"MLP raw forward: in={raw_input} → out={mlp(raw_input)} (shape {mlp(raw_input).shape})")

    bounded = BoundedPredictor(
        input_keys=("temperature_C", "loading"),
        in_scaler=BoundScaler(bounds=((20.0, 40.0), (0.05, 0.30)), transform="sigmoid"),
        inner=mlp,
        out_scaler=BoundScaler(
            bounds=((1e8, 1e14), (1e-8, 1e-4)),
            transform="sigmoid",
        ),
    )
    cov = {"temperature_C": jnp.array(28.0), "loading": jnp.array(0.18)}
    bounded_out = bounded(cov)
    print(f"BoundedPredictor on cov={cov}: out={bounded_out} (shape {bounded_out.shape})")


if __name__ == "__main__":
    main()
