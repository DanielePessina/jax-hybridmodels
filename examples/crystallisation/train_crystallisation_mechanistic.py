"""Fit four mechanistic kinetic constants to the crystallisation data with CMA-ES.

The parametric baseline for ``train_kinetic.py``. The data, moment ODE and
observation model are imported from it; only the rate laws change. The two
networks are replaced by classical nucleation theory for ``J`` and a power
law for ``G``, with four constants shared by every experiment: ``logA``,
``gamma``, ``Ag`` and ``g``. Four bounded scalars on a surface with local
minima near the metastable limit suit a population search.

Shows:

- ``Parameter`` scalars with physical bounds, read through ``.value``
- ``HybridModel.fit`` with an ``EvosaxTrainingConfig`` (CMA-ES)
- ``init="lhs_box"`` to spread generation 0 over the bounded box
- an implicit solver (``Kvaerno3``) for stiff candidates

Run::

    uv run python examples/crystallisation/train_crystallisation_mechanistic.py
    uv run python examples/crystallisation/train_crystallisation_mechanistic.py \
        --generations 2 --population-size 8 --no-plot

Prints the buckets, the four constants before and after the fit, and
per-channel metrics. Writes ``parity_mechanistic.png`` and
``trajectories_mechanistic.png`` to ``figures/``. Under a minute on CPU
at the default 80 generations of 64.
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

# Importing train_kinetic also enables float64, before any array is made.
from train_kinetic import (
    META_EPS,
    OUTPUT_CHANNELS,
    integrate_minutes,
    make_dataset,
    moment_rhs,
    saturation_conc,
    state_to_output,
)

import jaxhybridmodels as hm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
    trajectory_plot,
)

M_V = 2.97e-26  # molecular volume [m^3]
K_B = 1.38064852e-23  # Boltzmann constant [J/K]

# Bounds from the thesis package, so the fitted constants are comparable
# with its runs. Each Parameter starts at the midpoint of its bounds.
PARAMETER_BOUNDS = {
    "logA": (20.0, 65.0),  # ln of the CNT pre-exponential factor
    "gamma": (0.15, 1.0),  # interfacial energy [mJ/m^2]
    "Ag": (-20.0, -5.0),  # log10 of the growth pre-factor [m/min]
    "g": (1.0, 3.5),  # growth exponent
}

# Tolerances as in train_kinetic.py. The CNT exponential makes the ODE
# stiff for many of the candidates CMA-ES tries, so the solver is implicit.
SOLVER = hm.SolverConfig(
    solver=diffrax.Kvaerno3(),
    rtol=1e-4,
    atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
    max_steps=500_000,
    dt0=None,
)


def simulate_fn(
    predictors: dict[str, hm.Parameter],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 6"],
    solver: hm.SolverConfig,
) -> Float[Array, "T 6"]:
    """Integrate the moment ODE with CNT nucleation and power-law growth."""
    logA = predictors["logA"].value
    gamma = predictors["gamma"].value * 1e-3  # mJ/m^2 to J/m^2
    Ag = predictors["Ag"].value
    g = predictors["g"].value

    temperature_C = covariates["temperature_C"]
    T_K = temperature_C + 273.15
    conc_sat = saturation_conc(temperature_C)

    # region vector-field
    def vector_field(t: ArrayLike, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
        S = y[5] / conc_sat
        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

        # Keep S just above 1 inside the log. The mask already zeros the
        # rate below saturation, but ln(1) = 0 in the denominator would
        # still put a nan into the gradient of every step.
        S_safe = jnp.clip(S, min=1.0 + 1e-12)
        cnt_exponent = (
            -16.0 * jnp.pi * gamma**3 * M_V**2 / (3.0 * (K_B * T_K) ** 3 * jnp.log(S_safe) ** 2)
        )
        J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exponent)

        # Ag is per minute, as in the thesis package; /60 gives per second.
        G = meta_mask * 10.0**Ag / 60.0 * jnp.maximum(S - 1.0, 0.0) ** g
        return moment_rhs(y, G, J)

    # endregion vector-field

    return integrate_minutes(vector_field, ts, y0, solver)


def print_constants(stage: str, predictors: dict[str, hm.Parameter]) -> None:
    values = ", ".join(f"{name}={float(p.value):.3f}" for name, p in predictors.items())
    print(f"  {stage}: {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--sigma-init", type=float, default=0.5)
    parser.add_argument(
        "--init",
        choices=("warm", "uniform_box", "lhs_box"),
        default="lhs_box",
        help="How CMA-ES draws generation 0; lhs_box covers the 4-D box most evenly.",
    )
    parser.add_argument("--init-box-extent", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    apply_default_style()

    dataset = make_dataset()
    print("[data]")
    print(hm.describe_buckets(dataset))

    # region model
    model = hm.HybridModel(
        predictors={
            name: hm.Parameter((low + high) / 2, bounds=(low, high))
            for name, (low, high) in PARAMETER_BOUNDS.items()
        },
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
        solver=SOLVER,
    )
    # endregion model
    # region config
    config = hm.EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=args.population_size,
        num_generations=args.generations,
        init=args.init,
        init_box_extent=args.init_box_extent,
        sigma_init=args.sigma_init,
    )
    # endregion config
    print("\n[fit] CMA-ES")
    print_constants("init", model.predictors)
    result = model.fit(dataset, config, key=jr.key(args.seed))
    print_constants("fitted", result.model.predictors)
    print(f"  best loss {min(result.history.best_fitness):.6f}")

    print("\n[results]")
    predictions = result.model.predict(dataset)
    hm.print_metrics(hm.compute_metrics(predictions.buckets, dataset))

    if args.no_plot:
        return
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(
        parity_diagnostics(predictions.buckets, dataset),
        title="Crystallisation parity (mechanistic CNT + power-law)",
        save_path=args.plot_dir / "parity_mechanistic.png",
    )
    trajectory_plot(
        predictions.buckets,
        dataset,
        predictors=result.model.predictors,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=SOLVER,
        title="Crystallisation trajectories (mechanistic)",
        save_path=args.plot_dir / "trajectories_mechanistic.png",
    )
    print(f"[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
