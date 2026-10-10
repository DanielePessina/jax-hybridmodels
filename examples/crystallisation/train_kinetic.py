"""Learn crystallisation kinetics with two networks inside a moment ODE.

Two ``BoundedPredictor`` networks read ``(temperature_C, supersaturation)``
and return bounded log-rates, ``log10(G)`` for growth and ``log10(J)`` for
nucleation. The vector field exponentiates them and drives a
method-of-moments population balance with a mass balance on the solute.
Supersaturation is computed from the state at every solver step, so the
networks see a time-varying input.

The script fits the same model twice, once with MLPs and once with KANs,
and compares the final losses. Four experiments from the thesis
``Unseeded_LowData4`` cut are inlined, so it needs no data file. Each has
an irregular concentration series and one terminal ``d43`` size.

Run:
    uv run python examples/crystallisation/train_kinetic.py [--steps N] [--no-plot]

Prints the dataset buckets, the rates at a reference condition before and
after training, and per-channel metrics for each family. Writes
``parity.png`` and ``trajectories.png`` to ``figures/mlp/`` and
``figures/kan/``.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jax
from jax.typing import ArrayLike

# Before any other JAX-touching import. The moments span about 18 decades
# during integration, and a float32 mass balance drifts visibly in one run.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import Array  # noqa: E402
from jaxtyping import Float  # noqa: E402

import jaxhybridmodels as hm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
    trajectory_plot,
)

# Values rounded from the thesis sheet. The single terminal d43 point makes
# that channel's time axis much sparser than conc's, which is the irregular
# layout make_dataset buckets.
EXPERIMENTS_DATA: tuple[dict[str, Any], ...] = (
    {
        "exp_id": "LowData4_3",
        "temperature_C": 17.0,
        "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
        "conc": (14.7, 13.7, 7.7, 7.0, 5.7, 5.5, 5.4, 5.1, 5.2),
        "d43_time_min": 270.0,
        "d43": 9.2,
        "d43_var": 5.345,
    },
    {
        "exp_id": "LowData4_4",
        "temperature_C": 17.0,
        "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
        "conc": (11.6, 10.8, 10.8, 10.5, 10.9, 9.5, 6.8, 6.1, 5.7),
        "d43_time_min": 270.0,
        "d43": 7.7,
        "d43_var": 3.734,
    },
    {
        "exp_id": "LowData4_7",
        "temperature_C": 21.0,
        "time_min": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
        "conc": (16.8, 12.1, 8.6, 7.6, 7.2, 6.8, 6.4),
        "d43_time_min": 360.0,
        "d43": 10.5,
        "d43_var": 1.421,
    },
    {
        "exp_id": "LowData4_9",
        "temperature_C": 21.0,
        "time_min": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 375.0),
        "conc": (14.4, 14.2, 13.6, 10.1, 9.3, 7.6, 7.0),
        "d43_time_min": 375.0,
        "d43": 11.9,
        "d43_var": 0.267,
    },
)
CONC_VAR = 0.1  # made up; the thesis per-row variances are noisy and unneeded

# An untrained network outputs the midpoint of its output box, so the
# midpoint has to be a physically reasonable rate. Wider bounds put it
# decades off and made the moment ODE intractably stiff at init.
TEMPERATURE_BOUNDS = (13.0, 27.0)  # °C, slightly wider than the data
SUPERSATURATION_BOUNDS = (0.0, 12.0)  # S = conc / conc_sat
LOG10_GROWTH_BOUNDS = (-15.0, -5.0)  # log10(G [m/s])
LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)  # log10(J [#/(m³·s)])

RHO_C = 1370.0  # crystal density [kg/m^3]
K_V = 0.81  # volumetric shape factor
D43_MAX = 55.0  # upper guard on d43 [um]
D43_MU3_EPS = 1e-6  # mu3 floor for the d43 ratio
META_EPS = 1e-5  # rates are zero unless S > 1 + META_EPS

OUTPUT_CHANNELS = ("conc", "d43")
INPUT_KEYS = ("temperature_C", "supersaturation")

# Per-state atol matched to each state's natural magnitude (mu0 ~ 1e11 down
# to mu4 ~ 1e-7, conc ~ 1). A uniform atol over-resolves the small moments,
# under-resolves the large ones, and runs out of max_steps. dt0=None lets
# integrate_minutes pick a first step from each experiment's span.
SOLVER = hm.SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-4,
    atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
    max_steps=500_000,
    dt0=None,
)


def y0_fn(covariates: dict[str, Array], channels: dict[str, hm.ChannelObs]) -> Float[Array, " 6"]:
    """Initial state ``[mu0..mu4, conc]``: no crystals, conc at its first observation."""
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])


def state_to_output(state: Float[Array, "T 6"]) -> Float[Array, "T 2"]:
    """``[mu0..mu4, conc]`` to the observed ``[conc, d43]``."""
    mu3 = state[..., 3]
    mu4 = state[..., 4]
    conc = state[..., 5]
    # d43 = mu4 / mu3 in um, guarded while the moments are near zero. The
    # divisor is replaced before dividing: guarding only the result still
    # evaluates mu4 / 0, and its nan gradient flows back through where.
    safe_mu3 = jnp.where(mu3 > D43_MU3_EPS, mu3, 1.0)
    ratio = jnp.where(mu3 > D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
    d43 = jnp.clip(jnp.where(jnp.isfinite(ratio) & (ratio > 0.0), ratio, 0.0), 0.0, D43_MAX)
    return jnp.stack([conc, d43], axis=-1)


def saturation_conc(temperature_C: Array) -> Array:
    """Empirical solubility, in the units of ``conc``, at a temperature in °C."""
    return (
        0.3705
        + 7.171e-2 * temperature_C
        - 1.924e-3 * temperature_C**2
        + 17.97e-5 * temperature_C**3
    )


def moment_rhs(y: Float[Array, " 6"], G: Array, J: Array) -> Float[Array, " 6"]:
    """Moment equations: nucleation at zero size, linear growth, solute mass balance."""
    mu0, mu1, mu2, mu3 = y[0], y[1], y[2], y[3]
    return jnp.stack(
        [
            J,
            G * mu0,
            2.0 * G * mu1,
            3.0 * G * mu2,
            4.0 * G * mu3,
            -3.0 * K_V * RHO_C * G * mu2,
        ]
    )


def integrate_minutes(
    vector_field: Any, ts: Float[Array, " T"], y0: Float[Array, " 6"], solver: hm.SolverConfig
) -> Float[Array, "T 6"]:
    """Solve on a time grid given in minutes, with the rates in per-second units.

    ``solver.diffeqsolve`` would integrate in the data's minutes, so this
    calls diffrax directly and still takes the step controller and adjoint
    from the config. With ``dt0=None`` the first step is 1/1000 of the
    span, at least 1 s.
    """
    times_sec = ts * 60.0
    span = times_sec[-1] - times_sec[0]
    dt0 = jnp.maximum(span / 1000.0, 1.0) if solver.dt0 is None else solver.dt0
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=times_sec[0],
        t1=times_sec[-1],
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=times_sec),
        stepsize_controller=solver.stepsize_controller(),
        max_steps=solver.max_steps,
        adjoint=solver.adjoint,
    )
    return jnp.asarray(sol.ys)


def simulate_fn(
    predictors: dict[str, hm.BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 6"],
    solver: hm.SolverConfig,
) -> Float[Array, "T 6"]:
    """Integrate the moment ODE with growth and nucleation rates from the networks."""
    temperature_C = covariates["temperature_C"]
    conc_sat = saturation_conc(temperature_C)

    def vector_field(t: ArrayLike, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
        S = y[5] / conc_sat
        # Both rates switch off at the metastable limit.
        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)
        # Temperature is a covariate and S comes from the state; the
        # predictors read both as named inputs.
        inputs = {"temperature_C": temperature_C, "supersaturation": S}
        G = meta_mask * 10.0 ** predictors["growth"](inputs)[0]
        J = meta_mask * 10.0 ** predictors["nucleation"](inputs)[0]
        return moment_rhs(y, G, J)

    return integrate_minutes(vector_field, ts, y0, solver)


def make_dataset() -> hm.Dataset:
    experiments = []
    for data in EXPERIMENTS_DATA:
        conc = jnp.asarray(data["conc"], dtype=float)
        experiments.append(
            hm.make_experiment(
                covariates={"temperature_C": data["temperature_C"]},
                channels={
                    "conc": hm.ChannelObs(
                        ts=jnp.asarray(data["time_min"], dtype=float),
                        values=conc,
                        variance=jnp.full_like(conc, CONC_VAR),
                    ),
                    "d43": hm.ChannelObs(
                        ts=jnp.asarray([data["d43_time_min"]], dtype=float),
                        values=jnp.asarray([data["d43"]], dtype=float),
                        variance=jnp.asarray([data["d43_var"]], dtype=float),
                    ),
                },
                y0_fn=y0_fn,
                exp_id=data["exp_id"],
            )
        )
    return hm.make_dataset(experiments, output_channel_names=OUTPUT_CHANNELS)


def rate_predictors(make_inner, key: Array) -> dict[str, hm.BoundedPredictor]:
    """Wrap two inner networks as bounded log10 growth and nucleation rates."""
    input_bounds = (TEMPERATURE_BOUNDS, SUPERSATURATION_BOUNDS)
    k_growth, k_nucleation = jr.split(key)
    return {
        "growth": hm.BoundedPredictor(
            inner=make_inner(k_growth),
            input_keys=INPUT_KEYS,
            input_bounds=input_bounds,
            output_bounds=LOG10_GROWTH_BOUNDS,
        ),
        "nucleation": hm.BoundedPredictor(
            inner=make_inner(k_nucleation),
            input_keys=INPUT_KEYS,
            input_bounds=input_bounds,
            output_bounds=LOG10_NUCLEATION_BOUNDS,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    # Spline resolution per KAN edge. 5 is the jaxkan default and suits the
    # smooth log-rate surfaces here. basis="base" drops the spline, as an
    # ablation.
    parser.add_argument("--kan-grid-size", type=int, default=5)
    parser.add_argument("--kan-basis", choices=("spline", "base"), default="spline")
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    apply_default_style()

    dataset = make_dataset()
    print(hm.describe_buckets(dataset))

    k_init, k_train = jr.split(jr.key(args.seed))
    k_mlp, k_kan = jr.split(k_init)

    def mlp(key: Array) -> hm.MLPPredictor:
        return hm.MLPPredictor(
            in_size=2, out_size=1, width_size=64, depth=1, activation_name="relu", key=key
        )

    def kan(key: Array) -> hm.KANPredictor:
        # The zero head starts the output at the midpoint of each log-bound.
        # Without it jaxkan's init lands decades below the midpoint of the
        # asymmetric nucleation box, the moments never move, and the loss
        # stays flat. A random MLP already starts near zero output.
        return hm.KANPredictor(
            in_size=2,
            out_size=1,
            hidden_widths=(64,),  # matches the MLP's one hidden layer of 64
            grid_size=args.kan_grid_size,
            basis=args.kan_basis,
            key=key,
        ).with_zero_final_head()

    config = hm.OptaxTrainingConfig(steps=args.steps, lr=args.lr)
    # A reference condition for reading the learned rates.
    reference = {"temperature_C": 17.0, "supersaturation": 1.5}

    def report_rates(stage: str, predictors: dict[str, hm.BoundedPredictor]) -> None:
        log10_G = hm.evaluate_predictor(predictors["growth"], reference)
        log10_J = hm.evaluate_predictor(predictors["nucleation"], reference)
        print(
            f"  {stage} at T=17°C, S=1.5: G={10.0**log10_G:.2e} m/s, J={10.0**log10_J:.2e} #/m³/s"
        )

    final_losses = {}
    for label, make_inner, k_family in (("mlp", mlp, k_mlp), ("kan", kan, k_kan)):
        model = hm.HybridModel(
            predictors=rate_predictors(make_inner, k_family),
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
            solver=SOLVER,
        )
        print(f"\n[{label}]")
        report_rates("init", model.predictors)
        result = model.fit(dataset, config, key=k_train)
        report_rates("trained", result.model.predictors)
        final_losses[label] = min(result.history.loss)

        predictions = result.model.predict(dataset)
        hm.print_metrics(hm.compute_metrics(predictions.buckets, dataset))

        if args.no_plot:
            continue
        family_dir = args.plot_dir / label
        family_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            parity_diagnostics(predictions.buckets, dataset),
            title=f"Crystallisation parity ({label})",
            save_path=family_dir / "parity.png",
        )
        trajectory_plot(
            predictions.buckets,
            dataset,
            predictors=result.model.predictors,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=SOLVER,
            title=f"Crystallisation trajectories ({label})",
            save_path=family_dir / "trajectories.png",
        )
        print(f"  figures written to {family_dir}")

    print(f"\nbest loss: mlp {final_losses['mlp']:.6f}, kan {final_losses['kan']:.6f}")


if __name__ == "__main__":
    main()
