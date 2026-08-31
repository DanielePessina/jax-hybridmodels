"""Recover the frequency of a harmonic oscillator from noisy positions.

The hidden physics is one trainable scalar, the angular frequency ``omega``
of ``dx/dt = v``, ``dv/dt = -omega^2 x``. Each experiment is one oscillator
with a different ``(x0, v0)`` and the same true ``omega = 1.0``. Only
position is observed; velocity is latent state. A ``BoundedPredictor``
confines ``omega`` to ``[0.5, 2.0]`` and training has to find it.

Because the answer is known, this is the example to run to check the
pipeline is wired correctly rather than whether a model is any good.

Run: ``uv run python examples/pendulum/train_harmonic.py``
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
    SolverConfig,
    compute_metrics,
    evaluate_predictor,
    make_dataset,
    make_experiment,
    predict_dataset,
    print_metrics,
)
from hybridmodels.predictors.base import Predictor
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# ``examples/_shared`` is a sibling directory; put it on sys.path so the
# helpers import without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
    trajectory_plot,
)

OMEGA_TRUE: float = 1.0
OMEGA_BOUNDS: tuple[float, float] = (0.5, 2.0)

INITIAL_STATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, -0.5),
    (1.0, 1.0),
    (-0.7, 0.4),
    (0.3, 0.9),
)

# Five time units is about 0.8 of a period at omega=1, enough phase coverage
# to fit omega without aliasing into the wrong basin.
T_MAX: float = 5.0
N_TIMESTEPS: int = 12
NOISE_STD: float = 0.02
OUTPUT_CHANNELS: tuple[str, ...] = ("position",)


class OmegaPredictor(Predictor):
    """One trainable scalar, ignoring its input.

    Every experiment shares the same ground-truth ``omega``, so the same
    value comes back whatever covariate arrives. ``initialized_with_key``
    is what the tournament calls to draw a fresh starting point.
    """

    omega_lat: Array

    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)

    def __call__(self, x: Array) -> Float[Array, " 1"]:
        return self.omega_lat[None]

    def initialized_with_key(self, key: Array) -> OmegaPredictor:
        return OmegaPredictor(jr.normal(key))


def _state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Project full state ``[x, v]`` to the observed channel ``[x]``."""
    return state[..., :1]


def _simulate_fn(
    predictor: BoundedPredictor,
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate the harmonic oscillator for one experiment.

    The framework's ``simulate_fn`` contract:
    ``(predictor, ts, covariates, y0, solver) -> [T, S]``. Read ``omega``
    from the predictor in physical units, build the vector field
    ``f(t, [x, v]) = [v, -omega^2 x]``, and solve it over ``ts``.
    """
    omega = predictor(covariates).reshape(())
    omega_sq = omega * omega

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        return jnp.stack([y[1], -omega_sq * y[0]])

    term = diffrax.ODETerm(vector_field)
    return jnp.asarray(solver.diffeqsolve(term, ts, y0).ys)


def _read_omega(predictor: BoundedPredictor) -> float:
    """Pull the bounded ``omega`` value out of the predictor for reporting."""
    return evaluate_predictor(predictor, {"dummy": 0.0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to write parity + trajectory PNGs into.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (still prints diagnostics).",
    )
    args = parser.parse_args()

    apply_default_style()

    root_key = jr.PRNGKey(args.seed)
    k_data, k_init, k_train = jr.split(root_key, 3)

    print("[build] synthetic harmonic-oscillator dataset")
    # Positions come from the closed form ``x(t) = x0 cos(ωt) + (v0/ω) sin(ωt)``,
    # sampled evenly on ``[0, T_MAX]`` and perturbed by ``NOISE_STD`` Gaussian
    # noise. Each experiment's ``y0_fn`` closes over its own ``[x0, v0]``,
    # captured at synthesis time rather than read off the data.
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments: list[Experiment] = []
    rng = k_data
    for i, (x0, v0) in enumerate(INITIAL_STATES):
        rng, k = jr.split(rng)
        clean = x0 * jnp.cos(OMEGA_TRUE * ts) + (v0 / OMEGA_TRUE) * jnp.sin(OMEGA_TRUE * ts)
        noisy = clean + NOISE_STD * jr.normal(k, ts.shape)
        y0 = jnp.asarray([x0, v0], dtype=jnp.float32)
        experiments.append(
            make_experiment(
                covariates={"dummy": 0.0},
                channels={
                    "position": ChannelObs(
                        ts=ts,
                        values=noisy,
                        variance=jnp.full(ts.shape, NOISE_STD**2),
                    )
                },
                y0_fn=lambda _cov, _channels, y0=y0: y0,
                exp_id=f"osc_{i}_x0={x0}_v0={v0}",
            )
        )
    dataset = make_dataset(
        experiments,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(experiments)} experiments, {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(
            f"    bucket {i}: ts={tuple(bp.ts.shape)}, "
            f"y_observed={tuple(bp.y_observed.shape)}, n_obs={int(bp.n_obs)}"
        )

    print("\n[build] solver + predictor")
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.05,
    )
    # ``BoundedPredictor`` needs at least one input slot, so a constant
    # ``"dummy"`` covariate satisfies the named-input contract. The
    # ``in_scaler`` is a no-op in practice, since the inner predictor ignores
    # what it receives.
    predictor = BoundedPredictor(
        input_keys=("dummy",),
        in_scaler=BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid"),
        inner=OmegaPredictor(jr.normal(k_init)),
        out_scaler=BoundScaler(bounds=(OMEGA_BOUNDS,), transform="sigmoid"),
    )
    print(f"  initial omega = {_read_omega(predictor):.4f} (target {OMEGA_TRUE})")

    print("\n[train] optax (single phase, mse loss)")
    config = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=False,
    )
    history, trained = train_with_optax(
        predictor,
        dataset,
        config,
        simulate_fn=_simulate_fn,
        state_to_output=_state_to_output,
        solver=solver,
        key=k_train,
    )
    print(f"  {len(history)} steps; final loss {history[-1]:.6f}")
    sample_every = max(1, len(history) // 10)
    sampled = [f"{loss:.4f}" for loss in history[::sample_every]]
    print(f"  loss every ~{sample_every} steps: {sampled}")

    final_omega = _read_omega(trained)
    print(f"\n  recovered omega = {final_omega:.4f} (target {OMEGA_TRUE})")
    print(f"  absolute error  = {abs(final_omega - OMEGA_TRUE):.4f}")

    # predict_dataset returns one [N, T, D] array per bucket; the helpers
    # walk it in lockstep with the dataset.
    print("\n[diagnostics] per-channel parity stats over the training set")
    predictions = predict_dataset(
        trained, dataset, simulate_fn=_simulate_fn, state_to_output=_state_to_output, solver=solver
    )
    metrics = compute_metrics(predictions, dataset)
    print_metrics(metrics)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        # ``compute_metrics`` keeps only summary stats; the scatter needs the
        # raw value pairs, so re-walk the buckets' masks here.
        parity_data = parity_diagnostics(predictions, dataset)
        parity_plot(
            parity_data,
            title="Pendulum parity (trained model)",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=solver,
            max_experiments=len(experiments),
            title="Pendulum trajectories (trained model)",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
