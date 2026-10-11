"""Recover the frequency of a harmonic oscillator: the quickstart example.

The model is ``dx/dt = v``, ``dv/dt = -omega^2 x - c v`` with one unknown,
the angular frequency ``omega`` (true value 1). The damping ``c`` is known
to be zero. Six oscillators start from different ``(x0, v0)`` at ``t = 0``
and are measured in position only, from ``t = 0.2`` on, with noise.
Velocity is state the solver tracks and no instrument sees.

The answer is known, so this is also the script to adapt when you want to
check that your data, ODE, loss and training path are wired correctly.

Shows:

- ``HybridModel.from_vector_field`` for a model that is one ODE solve
- ``Parameter`` (trained, bounded) and ``Constant`` (fixed), both read
  through ``.value``
- ``make_experiment(t0=..., y0=...)``: an initial state before the first
  measurement
- a two-phase ``OptaxTrainingConfig`` built from ``TrainingPhase``
- ``FitResult.best_step`` and ``FitResult.stop_reason``
- ``predict`` indexed by experiment and channel, ``simulate`` for new
  conditions, and ``compute_metrics``

Run::

    uv run python examples/pendulum/train_harmonic.py
    uv run python examples/pendulum/train_harmonic.py --no-plot

Prints the dataset layout, the fit summary, the recovered ``omega`` and
per-channel metrics. Writes ``figures/parity.png`` and
``figures/trajectories.png`` unless ``--no-plot`` is given. About 10 s on
CPU.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

import jaxhybridmodels as hm

# ``examples/_shared`` is a sibling directory; put it on sys.path so the
# plot helpers import without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
    trajectory_plot,
)

OMEGA_TRUE: float = 1.0
OMEGA_INIT: float = 1.4
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
TS = jnp.linspace(0.2, 5.0, 12)
NOISE_STD: float = 0.02
CHANNELS: tuple[str, ...] = ("position",)


# region model-functions
def vector_field(
    t: ArrayLike,
    y: Float[Array, " 2"],
    predictors: dict[str, hm.Parameter | hm.Constant],
    covariates: dict[str, Array],
) -> Float[Array, " 2"]:
    """``d[x, v]/dt`` for one oscillator. Both predictor kinds are read through ``.value``."""
    omega = predictors["omega"].value
    damping = predictors["damping"].value
    return jnp.stack([y[1], -(omega**2) * y[0] - damping * y[1]])


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Only position is observed."""
    return state[..., :1]
    # endregion model-functions


def build_dataset(key: Array) -> hm.Dataset:
    """Noisy positions from the closed form ``x0 cos(wt) + (v0 / w) sin(wt)``."""
    experiments = []
    for i, (x0, v0) in enumerate(INITIAL_STATES):
        clean = x0 * jnp.cos(OMEGA_TRUE * TS) + (v0 / OMEGA_TRUE) * jnp.sin(OMEGA_TRUE * TS)
        noisy = clean + NOISE_STD * jr.normal(jr.fold_in(key, i), TS.shape)
        experiments.append(
            # region experiment
            hm.make_experiment(
                channels={"position": hm.ChannelObs(ts=TS, values=noisy, variance=NOISE_STD**2)},
                # The state is known at t0 = 0; the first measurement is at 0.2.
                t0=0.0,
                y0=[x0, v0],
                exp_id=f"osc_{i}",
            )
            # endregion experiment
        )
    return hm.make_dataset(experiments, output_channel_names=CHANNELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the figures.")
    args = parser.parse_args()
    k_data, k_train = jr.split(jr.key(args.seed))

    print("[data]")
    dataset = build_dataset(k_data)
    print(hm.describe_buckets(dataset))

    # region model
    model = hm.HybridModel.from_vector_field(
        vector_field=vector_field,
        predictors={
            # Trained, and kept inside (0.5, 2.0) at every step.
            "omega": hm.Parameter(OMEGA_INIT, bounds=(0.5, 2.0)),
            # Known and fixed: the default training mask never updates it.
            "damping": hm.Constant(0.0),
        },
        state_to_output=state_to_output,
        output_channel_names=CHANNELS,
        # Tsit5 with rtol 1e-5 and atol 1e-7 by default; only the first
        # step size differs here.
        solver=hm.SolverConfig(dt0=0.05),
    )
    # endregion model

    print("\n[fit]")
    # A fast phase to get close, then a slower one to settle.
    # region fit
    config = hm.OptaxTrainingConfig(
        phases=(
            hm.TrainingPhase(steps=200, lr=0.05),
            hm.TrainingPhase(steps=100, lr=0.01),
        ),
        verbose=False,
    )
    result = model.fit(dataset, config, key=k_train)
    # endregion fit
    omega = float(result.model.predictors["omega"].value)
    print(f"omega {OMEGA_INIT} -> {omega:.4f} (true {OMEGA_TRUE})")
    print(f"best step {result.best_step}, stop reason {result.stop_reason}")

    print("\n[results]")
    # region predict-simulate
    predictions = result.model.predict(dataset)
    # Indexed by experiment, then channel. ``times`` starts at t0, where
    # nothing was measured.
    first = predictions["osc_0"]
    print(
        f"osc_0 at t = {float(first.times[1]):.2f}: x = {float(first.outputs['position'][1]):.4f}"
    )
    # A new oscillator, run past the end of the training window.
    new = result.model.simulate(ts=jnp.linspace(0.5, 10.0, 50), t0=0.0, y0=[0.5, 0.0])
    print(f"new oscillator at t = 10: x = {float(new.outputs['position'][-1]):.4f}")
    # endregion predict-simulate
    print()
    hm.print_metrics(predictions)

    if args.no_plot:
        return
    print("\n[plot]")
    apply_default_style()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(
        parity_diagnostics(predictions),
        title="Harmonic oscillator parity",
        save_path=args.plot_dir / "parity.png",
    )
    trajectory_plot(
        result.model,
        dataset,
        predictions=predictions,
        max_experiments=3,
        title="Harmonic oscillator trajectories",
        save_path=args.plot_dir / "trajectories.png",
    )
    print(f"figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
