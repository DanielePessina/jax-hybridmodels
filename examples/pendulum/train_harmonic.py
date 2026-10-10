"""Recover the frequency of a harmonic oscillator from noisy positions.

The model is ``dx/dt = v``, ``dv/dt = -omega^2 x`` with one unknown, the
angular frequency ``omega``. Six oscillators start from different
``(x0, v0)`` and share the true ``omega = 1``. Only position is measured;
velocity is state the solver tracks and no instrument sees.

The answer is known, so run this when you want to check that the data,
ODE, loss and training path are wired correctly. If the recovered
``omega`` is not within about 1% of 1.0, look first at ``state_to_output``
and the initial states.

Uses ``Parameter`` for a bounded physical scalar, ``HybridModel.fit``, and
``compute_metrics`` on the bucketed predictions.

Run::

    uv run python examples/pendulum/train_harmonic.py
    uv run python examples/pendulum/train_harmonic.py --no-plot

Prints the dataset layout, the loss curve, the recovered ``omega`` and
per-channel metrics. Writes ``figures/parity.png`` and
``figures/trajectories.png`` unless ``--no-plot`` is given.
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
from jax.typing import ArrayLike
from jaxtyping import Float

import jaxhybridmodels as hm

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
OMEGA_INIT: float = 1.4
OMEGA_BOUNDS: tuple[float, float] = (0.5, 2.0)

INITIAL_STATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, -0.5),
    (1.0, 1.0),
    (-0.7, 0.4),
    (0.3, 0.9),
)

# Five time units is about 0.8 of a period at omega = 1: enough phase
# coverage to pin omega without a second basin at a multiple of it.
T_MAX: float = 5.0
N_TIMESTEPS: int = 12
NOISE_STD: float = 0.02
OUTPUT_CHANNELS: tuple[str, ...] = ("position",)


def simulate_fn(
    predictors: dict[str, hm.Parameter],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: hm.SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate one oscillator and return the full state ``[x, v]``."""
    omega = predictors["omega"].value

    def vector_field(t: ArrayLike, y: Float[Array, " 2"], args: object) -> Array:
        return jnp.stack([y[1], -(omega**2) * y[0]])

    return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Only position is observed."""
    return state[..., :1]


def build_dataset(key: Array) -> hm.Dataset:
    """Noisy positions from the closed form ``x0 cos(wt) + (v0 / w) sin(wt)``."""
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments = []
    for i, (x0, v0) in enumerate(INITIAL_STATES):
        clean = x0 * jnp.cos(OMEGA_TRUE * ts) + (v0 / OMEGA_TRUE) * jnp.sin(OMEGA_TRUE * ts)
        noisy = clean + NOISE_STD * jr.normal(jr.fold_in(key, i), ts.shape)
        experiments.append(
            hm.make_experiment(
                channels={
                    "position": hm.ChannelObs(
                        ts=ts, values=noisy, variance=jnp.full(ts.shape, NOISE_STD**2)
                    )
                },
                y0=[x0, v0],
                exp_id=f"osc_{i}",
            )
        )
    return hm.make_dataset(experiments, output_channel_names=OUTPUT_CHANNELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the figures.")
    args = parser.parse_args()

    apply_default_style()
    k_data, k_train = jr.split(jr.key(args.seed))

    dataset = build_dataset(k_data)
    print(hm.describe_buckets(dataset))

    model = hm.HybridModel(
        predictors={"omega": hm.Parameter(OMEGA_INIT, bounds=OMEGA_BOUNDS)},
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
        solver=hm.SolverConfig(
            solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05
        ),
    )
    print(f"\ninitial omega = {float(model.predictors['omega'].value):.4f}")

    config = hm.OptaxTrainingConfig(steps=args.steps, lr=args.lr, loss="mse", verbose=False)
    result = model.fit(dataset, config, key=k_train)
    loss = result.history.loss
    every = max(1, len(loss) // 10)
    print(f"loss every {every} steps: {[f'{v:.4f}' for v in loss[::every]]}")

    omega = float(result.model.predictors["omega"].value)
    print(f"recovered omega = {omega:.4f} (target {OMEGA_TRUE}, best step {result.best_step})")

    # ``predict`` returns per-experiment results by name; the metrics and
    # plotting helpers take the per-bucket arrays it also carries.
    predictions = result.model.predict(dataset).buckets
    print()
    hm.print_metrics(hm.compute_metrics(predictions, dataset))

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            parity_diagnostics(predictions, dataset),
            title="Harmonic oscillator parity",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=result.model.predictors,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=model.solver,
            max_experiments=len(INITIAL_STATES),
            title="Harmonic oscillator trajectories",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\nfigures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
