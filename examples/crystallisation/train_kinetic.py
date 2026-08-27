"""Learn crystallisation kinetics with two networks inside a moment ODE.

Two predictors read ``(temperature_C, supersaturation)`` and emit bounded
log-rates, ``log10(G)`` for growth and ``log10(J)`` for nucleation, that
feed a method-of-moments ODE. Supersaturation is state-derived inside the
vector field; the population balance and mass balance around it stay
mechanistic. The script trains an MLP pair and a KAN pair on the same data
and prints both final losses.

Four experiments from the thesis ``Unseeded_LowData4`` cut are inlined
below, so the script runs from a clean checkout with no data file.

Run: ``uv run python examples/crystallisation/train_kinetic.py``
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax

# Before any other JAX-touching import. The moments span ~18 decades during
# integration, and a float32 mass balance drifts visibly within one run.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import Array  # noqa: E402
from jaxtyping import Float  # noqa: E402

from hybridmodels import (  # noqa: E402
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
    KANPredictor,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    make_experiment,
    predict_dataset,
)
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
    trajectory_plot,
)

# Values rounded from the thesis sheet. Each experiment carries one terminal
# d43 measurement, so that channel's ``ts`` axis is far sparser than conc's,
# which is exactly the irregular layout ``make_dataset`` buckets.
EXPERIMENTS_DATA: tuple[dict[str, object], ...] = (
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

# Centring matters: a random-init network sits near the sigmoid midpoint, so
# that midpoint has to be a physically reasonable rate. Wider bounds put it
# several decades off and made the moment ODE intractably stiff at init.
TEMPERATURE_BOUNDS = (13.0, 27.0)  # °C, slightly wider than data span
SUPERSATURATION_BOUNDS = (0.0, 12.0)  # S = conc / conc_sat
LOG10_GROWTH_BOUNDS = (-15.0, -5.0)  # log10(G [m/s])
LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)  # log10(J [#/(m³·s)])

# ODE constants (from hybridcrystals/mechanistic.py).
RHO_C = 1370.0  # crystal density [kg/m^3]
K_V = 0.81  # volumetric shape factor
D43_MAX = 55.0  # upper guard on d43 [um]
D43_MU3_EPS = 1e-6  # mu3 floor for the d43 ratio
META_EPS = 1e-5  # supersaturation must exceed 1 + eps for nucleation/growth

OUTPUT_CHANNELS = ("conc", "d43")
INPUT_KEYS = ("temperature_C", "supersaturation")


# The three framework hooks, at module level so their identities are stable
# across calls: y0_fn, state_to_output, simulate_fn.


def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 6"]:
    """Initial state ``[mu0..mu4, conc]``: moments at zero, conc at first observation."""
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])


def state_to_output(state: Float[Array, "T 6"]) -> Float[Array, "T 2"]:
    """Project full state ``[mu0..mu4, conc]`` to observed channels ``[conc, d43]``."""
    mu3 = state[..., 3]
    mu4 = state[..., 4]
    conc = state[..., 5]
    # d43 = (mu4 / mu3) * 1e6 [um], guarded for the early near-zero moments.
    # The double-``where`` is the grad-safe form: a naive
    # ``jnp.where(cond, mu4/mu3, 0.0)`` still evaluates the division on the
    # masked branch, and its inf/nan gradient flows back through ``where``.
    safe_mu3 = jnp.where(mu3 > D43_MU3_EPS, mu3, 1.0)
    ratio = jnp.where(mu3 > D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
    d43 = jnp.clip(jnp.where(jnp.isfinite(ratio) & (ratio > 0.0), ratio, 0.0), 0.0, D43_MAX)
    return jnp.stack([conc, d43], axis=-1)


def simulate_fn(
    predictors: tuple[BoundedPredictor, BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 6"],
    solver: SolverConfig,
) -> Float[Array, "T 6"]:
    """Integrate the method-of-moments ODE using two direct-rate predictors.

    ``predictors = (growth_BP, nucleation_BP)``, each emitting one bounded
    log-rate that the vector field exponentiates back to physical units.
    Temperature is constant per experiment, supersaturation is state-derived
    each step, and both rates are gated by ``(S > 1 + eps)`` so the ODE
    stops moving below the metastable limit.
    """
    growth_bp, nucleation_bp = predictors
    temperature_C = covariates["temperature_C"]

    # Empirical saturation polynomial in °C (from hybridcrystals/mechanistic.py).
    conc_sat = (
        0.3705
        + 7.171e-2 * temperature_C
        - 1.924e-3 * temperature_C**2
        + 17.97e-5 * temperature_C**3
    )

    def vector_field(t: Array, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat
        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

        inputs = {"temperature_C": temperature_C, "supersaturation": S}
        log10_G = jnp.squeeze(growth_bp(inputs))
        log10_J = jnp.squeeze(nucleation_bp(inputs))
        G = meta_mask * jnp.power(10.0, log10_G)
        J = meta_mask * jnp.power(10.0, log10_J)

        # Hulburt-Katz form for nucleation at zero size and linear growth,
        # plus a mass balance: solute lost equals crystal volume gained.
        dmu0 = J
        dmu1 = G * mu0
        dmu2 = 2.0 * G * mu1
        dmu3 = 3.0 * G * mu2
        dmu4 = 4.0 * G * mu3
        dconc = -3.0 * K_V * RHO_C * G * mu2
        return jnp.stack([dmu0, dmu1, dmu2, dmu3, dmu4, dconc])

    # Dataset stores time in minutes; rate constants are SI-second-based.
    times_sec = ts * 60.0
    if solver.dt0 is None:
        # 1/1000 of the span, floored at 1 s so short trajectories do not
        # underflow on the first step.
        span = times_sec[-1] - times_sec[0]
        dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
    else:
        dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)

    # PIDController needs atol as a scalar or array; a tuple does not
    # broadcast against the y_error pytree leaves.
    atol = (
        jnp.asarray(solver.atol, dtype=times_sec.dtype)
        if isinstance(solver.atol, tuple)
        else solver.atol
    )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=times_sec[0],
        t1=times_sec[-1],
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=times_sec),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    # ``grid_size`` is the spline resolution per edge; 5 is the jaxkan
    # default and suits the smooth log-rate surfaces here. ``basis="base"``
    # drops the spline and is kept as an ablation.
    parser.add_argument("--kan-grid-size", type=int, default=5)
    parser.add_argument("--kan-basis", choices=("spline", "base"), default="spline")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    k_init, k_train = jr.split(jr.PRNGKey(args.seed), 2)

    # One Experiment per entry, each with two channels on their own ``ts``
    # axes. ``CONC_VAR`` broadcasts to a per-row vector; d43 has a single
    # terminal observation and a scalar variance.
    experiments: list[Experiment] = []
    for data in EXPERIMENTS_DATA:
        time_min = jnp.asarray(data["time_min"], dtype=float)
        conc = jnp.asarray(data["conc"], dtype=float)
        d43_ts = jnp.asarray((data["d43_time_min"],), dtype=float)
        d43_vals = jnp.asarray((data["d43"],), dtype=float)
        d43_var = jnp.asarray((data["d43_var"],), dtype=float)

        experiments.append(
            make_experiment(
                covariates={"temperature_C": float(data["temperature_C"])},  # type: ignore[arg-type]
                channels={
                    "conc": ChannelObs(
                        ts=time_min,
                        values=conc,
                        variance=jnp.full_like(conc, CONC_VAR),
                    ),
                    "d43": ChannelObs(ts=d43_ts, values=d43_vals, variance=d43_var),
                },
                y0_fn=y0_fn,
                exp_id=str(data["exp_id"]),
            )
        )

    print(f"[load] {len(experiments)} hardcoded experiments")
    for exp in experiments:
        print(
            f"    {exp.exp_id}: T={float(exp.covariates['temperature_C']):.1f}°C, "
            f"conc obs={exp.channels['conc'].values.shape[0]}, "
            f"d43 obs={exp.channels['d43'].values.shape[0]}"
        )

    print("\n[build] dataset")
    dataset = make_dataset(
        experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(
            f"    bucket {i}: ts={tuple(bp.ts.shape)}, "
            f"y_observed={tuple(bp.y_observed.shape)}, "
            f"mask={tuple(bp.mask.shape)}, n_obs={int(bp.n_obs)}"
        )

    # Per-state atol matched to the natural moment magnitudes (``mu0 ~ 1e11``
    # down to ``mu4 ~ 1e-7``, ``conc ~ 1``). A uniform atol over-resolves the
    # small components and under-resolves the large ones, and the integrator
    # hits ``max_steps`` before finishing one trajectory. Nine decades below
    # each magnitude leaves rtol dominant and atol a near-zero floor.
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )

    # Both pairs share the input scaler, and the growth and nucleation
    # branches get independent inner weights by key splitting. The KAN's
    # ``hidden_widths=(64,)`` mirrors the MLP's single width-64 hidden layer,
    # so the comparison varies only the inner-network family.
    in_scaler = BoundScaler(
        bounds=(TEMPERATURE_BOUNDS, SUPERSATURATION_BOUNDS),
        transform="sigmoid",
    )
    k_mlp, k_kan = jr.split(k_init, 2)
    k_mlp_growth, k_mlp_nucleation = jr.split(k_mlp, 2)
    k_kan_growth, k_kan_nucleation = jr.split(k_kan, 2)

    def _wrap(inner_growth, inner_nucleation):
        growth_bp = BoundedPredictor(
            input_keys=INPUT_KEYS,
            in_scaler=in_scaler,
            inner=inner_growth,
            out_scaler=BoundScaler(bounds=(LOG10_GROWTH_BOUNDS,), transform="sigmoid"),
        )
        nucleation_bp = BoundedPredictor(
            input_keys=INPUT_KEYS,
            in_scaler=in_scaler,
            inner=inner_nucleation,
            out_scaler=BoundScaler(bounds=(LOG10_NUCLEATION_BOUNDS,), transform="sigmoid"),
        )
        return (growth_bp, nucleation_bp)

    mlp_predictors = _wrap(
        MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=1,
            activation_name="relu",
            key=k_mlp_growth,
        ),
        MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=1,
            activation_name="relu",
            key=k_mlp_nucleation,
        ),
    )
    # ``with_zero_final_head`` lands the KAN's init at the physical midpoint
    # of each log-bound. Without it, jaxkan's spline+residual init gives a
    # non-zero inner output that the asymmetric out-scaler over
    # ``(-6.5, 20.0)`` parks decades below the midpoint. At ``J ~ 2`` the
    # moments never evolve, the gradient through the integrator vanishes,
    # and Adam sits at a flat loss for the whole run. The MLP is unaffected:
    # random linear init already gives a near-zero output.
    kan_predictors = _wrap(
        KANPredictor(
            in_size=2,
            out_size=1,
            hidden_widths=(64,),
            grid_size=args.kan_grid_size,
            basis=args.kan_basis,
            key=k_kan_growth,
        ).with_zero_final_head(),
        KANPredictor(
            in_size=2,
            out_size=1,
            hidden_widths=(64,),
            grid_size=args.kan_grid_size,
            basis=args.kan_basis,
            key=k_kan_nucleation,
        ).with_zero_final_head(),
    )

    # Sanity check at the first experiment's covariates and a mid-range
    # supersaturation, in the shape the vector field builds each step.
    sample_cov = experiments[0].covariates
    sample_inputs = {
        "temperature_C": jnp.asarray(sample_cov["temperature_C"]),
        "supersaturation": jnp.asarray(1.5),
    }

    config = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=True,
    )

    def run_family(label: str, predictors_init):
        """Train one predictor family end-to-end and write its plots.

        ``label`` becomes the plot subdirectory and goes into the figure
        titles, so MLP and KAN outputs land side by side.
        """
        growth_bp, nucleation_bp = predictors_init
        log10_G_init = float(jnp.squeeze(growth_bp(sample_inputs)))
        log10_J_init = float(jnp.squeeze(nucleation_bp(sample_inputs)))
        print(
            f"\n[{label}][init] T={float(sample_cov['temperature_C']):.1f}°C, S=1.5 -> "
            f"log10_G={log10_G_init:.2f} (G={10.0**log10_G_init:.2e} m/s), "
            f"log10_J={log10_J_init:.2f} (J={10.0**log10_J_init:.2e} #/m³/s)"
        )

        print(f"\n[{label}][train] optax (single phase, mse loss)")
        history, trained_predictors = train_with_optax(
            predictors_init,
            dataset,
            config,
            simulate_fn=simulate_fn,
            solver=solver,
            key=k_train,
        )
        print(f"  {len(history)} steps; final loss {history[-1]:.6f}")
        sample_every = max(1, len(history) // 10)
        print(
            f"  loss every ~{sample_every} steps: "
            f"{[f'{loss:.4f}' for loss in history[::sample_every]]}"
        )

        trained_growth, trained_nucleation = trained_predictors
        log10_G_final = float(jnp.squeeze(trained_growth(sample_inputs)))
        log10_J_final = float(jnp.squeeze(trained_nucleation(sample_inputs)))
        print(
            f"  trained -> "
            f"log10_G={log10_G_final:.2f} (G={10.0**log10_G_final:.2e} m/s), "
            f"log10_J={log10_J_final:.2f} (J={10.0**log10_J_final:.2e} #/m³/s)"
        )

        print(f"\n[{label}][diagnostics] per-channel parity stats over the training set")
        predictions = predict_dataset(
            trained_predictors,
            dataset,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        diag = compute_diagnostics(predictions, dataset)
        print_diagnostics(diag)

        if not args.no_plot:
            family_dir = args.plot_dir / label
            family_dir.mkdir(parents=True, exist_ok=True)
            parity_plot(
                diag,
                title=f"Crystallisation parity ({label})",
                save_path=family_dir / "parity.png",
            )
            trajectory_plot(
                predictions,
                dataset,
                predictors=trained_predictors,
                simulate_fn=simulate_fn,
                solver=solver,
                max_experiments=6,
                title=f"Crystallisation trajectories ({label}, first 6 experiments)",
                save_path=family_dir / "trajectories.png",
            )
            print(f"\n[{label}][plot] figures written to {family_dir}")

        return float(history[-1])

    final_loss_mlp = run_family("mlp", mlp_predictors)
    final_loss_kan = run_family("kan", kan_predictors)
    print(f"\n[summary] final loss — mlp: {final_loss_mlp:.6f}, kan: {final_loss_kan:.6f}")


if __name__ == "__main__":
    main()
