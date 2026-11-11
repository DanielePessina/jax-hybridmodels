"""Multiple shooting rescues a frequency fit that single shooting gets wrong.

A lightly damped oscillator, ``x'' = -w^2 x - c x'``, with only the position
measured: 61 samples over five cycles, noise of standard deviation 0.05. The
damping is known and the frequency is fitted from a poor guess.

Single shooting solves once over the whole horizon. From a wrong frequency
the model falls out of step with the data within a cycle or two, and past
that the loss measures how two out-of-step waves happen to overlap, not how
wrong the frequency is. Gradient descent settles in the nearest dip.

Multiple shooting splits each trajectory into segments that start from
trainable states (nodes), so a wrong frequency has only a fraction of a cycle
to go wrong before the next segment starts afresh. The schedule goes from
ten segments to four to single shooting, so the returned model is an
ordinary one, fitted as one solve.

Shows:

- ``TrainingPhase(shooting_segments=...)`` for a schedule that ends in single shooting
- ``history.gap_term``, the gap term, and the summary's phase table

Run::

    uv run python examples/multiple_shooting/oscillator.py
    uv run python examples/multiple_shooting/oscillator.py --no-plot

Writes ``figures/multiple_shooting.png``. About 40 seconds on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import diffrax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import jaxhybridmodels as hm

# ``examples/_shared`` is a sibling directory; put it on sys.path so the
# helpers import without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import apply_default_style  # noqa: E402

TRUE_FREQUENCY = 1.0
DAMPING = 0.1  # known
NOISE = 0.05
TIMES = np.linspace(0.0, 30.0, 61)
STARTS = (0.4, 1.75, 1.95)
FIGURES = Path(__file__).resolve().parent / "figures"


# region model
def simulate_fn(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
    w = predictors["w"].value

    def field(t: Any, y: Any, args: Any) -> Any:
        return jnp.stack([y[1], -(w**2) * y[0] - DAMPING * y[1]])

    return solver.diffeqsolve(diffrax.ODETerm(field), ts, y0)


def oscillator(frequency: float) -> hm.HybridModel:
    """The model at a starting frequency; the state is ``(x, v)`` and only ``x`` is measured."""
    return hm.HybridModel(
        predictors={"w": hm.Parameter(frequency, bounds=(0.1, 3.0))},
        simulate_fn=simulate_fn,
        state_to_output=lambda state: state[:, :1],
        output_channel_names=("x",),
    )


# endregion model


def make_data(key: Any) -> list[hm.Experiment]:
    """One run from rest at x = 1, solved at the true frequency, plus noise."""
    probe = hm.make_experiment(
        channels={"x": hm.ChannelObs(ts=TIMES, values=np.zeros_like(TIMES))}, y0=[1.0, 0.0]
    )
    clean = np.asarray(oscillator(TRUE_FREQUENCY).simulate(probe).outputs["x"])
    noisy = clean + NOISE * np.asarray(jr.normal(key, clean.shape))
    return [
        hm.make_experiment(
            channels={"x": hm.ChannelObs(ts=TIMES, values=noisy)}, y0=[1.0, 0.0], exp_id="run"
        )
    ]


# region configs
SINGLE = hm.OptaxTrainingConfig(steps=900, lr=0.03)
SEGMENTED = hm.OptaxTrainingConfig(
    phases=(
        hm.TrainingPhase(steps=300, lr=0.03, shooting_segments=10),
        hm.TrainingPhase(steps=300, lr=0.03, shooting_segments=4),
        hm.TrainingPhase(steps=300, lr=0.03),  # segments=1: single shooting
    )
)
# endregion configs


def frequency(result: hm.FitResult) -> float:
    return float(result.model.predictors["w"].value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiple shooting on a damped oscillator")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = parser.parse_args()
    data = make_data(jr.key(0))

    print(f"Fitted frequency (truth {TRUE_FREQUENCY}), {SINGLE.total_steps} steps each:")
    print(f"  {'start':>5}  {'single':>8}  {'segmented':>9}")
    fits = {}
    for start in STARTS:
        single = oscillator(start).fit(data, SINGLE, key=jr.key(1), verbose=False)
        segmented = oscillator(start).fit(data, SEGMENTED, key=jr.key(1), verbose=False)
        fits[start] = (single, segmented)
        print(f"  {start:>5.2f}  {frequency(single):>8.4f}  {frequency(segmented):>9.4f}")

    single, segmented = fits[1.75]
    print("\nFrom 1.75, segmented:")
    print(segmented.summary().split("Parameters")[0].rstrip())

    if not all(abs(frequency(fit) - TRUE_FREQUENCY) < 0.01 for _, fit in fits.values()):
        raise SystemExit("multiple shooting should recover the frequency from every start")
    if all(abs(frequency(fit) - TRUE_FREQUENCY) < 0.01 for fit, _ in fits.values()):
        raise SystemExit("single shooting was expected to miss from at least one start")

    if not args.no_plot:
        plot(data[0], single, segmented)


def plot(experiment: hm.Experiment, single: hm.FitResult, segmented: hm.FitResult) -> None:
    """The two fits from 1.75 against the data, and their losses by step."""
    import matplotlib.pyplot as plt

    apply_default_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    grid = jnp.linspace(0.0, 30.0, 601)
    observed = experiment.channels["x"]
    left.plot(observed.ts, observed.values, "o", color="0.45", ms=3.5, label="measurements")
    for result, label in ((single, "single shooting"), (segmented, "segments, then single")):
        x = result.model.simulate(experiment, ts=grid).outputs["x"]
        left.plot(grid, x, lw=1.6, label=f"{label}, ω = {frequency(result):.3f}")
    # Headroom above the first peak keeps the legend off the data.
    left.set(xlabel="time", ylabel="position x", title="Fits from ω = 1.75", ylim=(-1.0, 1.7))
    left.legend(frameon=False, loc="upper right", fontsize=8)

    steps = np.arange(len(single.history.loss))
    right.semilogy(steps, single.history.loss, lw=1.4, label="single shooting")
    total = np.asarray(segmented.history.loss) + np.asarray(segmented.history.gap_term)
    right.semilogy(steps, total, lw=1.4, label="segments: data + gap term")
    for boundary in (300, 600):
        right.axvline(boundary, color="0.7", lw=0.8, ls="--")
    right.set(xlabel="training step", ylabel="loss", title="10 segments, 4, then single shooting")
    right.legend(frameon=False)
    FIGURES.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / "multiple_shooting.png", dpi=150)
    print(f"[plot] {FIGURES / 'multiple_shooting.png'}")


if __name__ == "__main__":
    main()
