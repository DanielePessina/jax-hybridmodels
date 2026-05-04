"""Crystallisation kinetic-MLP training example.

Trains a hybrid model on four hardcoded crystallisation experiments
(reproduced from the thesis ``Unseeded_LowData4`` cut), where two MLPs
predict reaction rates that feed a method-of-moments ODE.

Each MLP consumes ``(temperature_C, supersaturation)`` and emits a bounded
log-rate — ``log10(G)`` for crystal growth, ``log10(J)`` for nucleation.
``supersaturation`` is state-derived inside the vector field; the
population-balance moment ODEs and mass balance around them stay mechanistic.

The four experiments are inlined as a tuple of dicts at module level so the
script runs from a clean checkout with no Excel/CSV dependency. Concentration
variance is a single made-up scalar (the thesis cuts carry per-row variances
that are noisy and not needed to demonstrate the framework); the d43 variances
are the rounded thesis values.

Run: ``uv run python examples/crystallisation/train_kinetic.py``
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax

# x64 must be enabled before any other JAX-touching import. The population-
# balance moments span ~18 decades during integration; float32 mass balance
# drifts visibly within a single experiment.
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

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Four hardcoded experiments from the thesis ``Unseeded_LowData4`` sheet.
# Conc and d43 are rounded to 1 dp; d43 variance to 3 dp; concentration
# variance is a single made-up scalar (``CONC_VAR``) applied per row. Each
# experiment carries one terminal d43 measurement, so the d43 channel's ``ts``
# axis is sparser than the conc channel's — exactly the irregular layout
# ``make_dataset`` is designed to bucket. The thesis ``Loading`` column is
# uniformly zero across LowData4 and is intentionally omitted as a covariate.
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
# Made-up uniform concentration variance, broadcast across every row.
CONC_VAR = 0.1

# Predictor input/output bounds. Centring matters: a random-init network sits
# near the sigmoid midpoint, so the midpoint must be a physically reasonable
# rate. ``G ~ 1e-10 m/s`` and ``J ~ 1.8e3 #/m³/s`` are matched to the source-
# package's typical ranges; earlier wider bounds put the midpoint several
# decades off and made the moment ODE intractably stiff at random init.
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


# --------------------------------------------------------------------------- #
# Hook callables consumed by the framework                                    #
# --------------------------------------------------------------------------- #
# These three functions are passed into ``make_experiment``, ``make_dataset``,
# and ``train_with_optax`` respectively. They have to live at module level so
# their identities are stable across calls.


def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 6"]:
    """Initial state ``[mu0..mu4, conc]``: moments at zero, conc at first observation."""
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])


def state_to_output(state: Float[Array, "T 6"]) -> Float[Array, "T 2"]:
    """Project full state ``[mu0..mu4, conc]`` to observed channels ``[conc, d43]``."""
    mu3 = state[..., 3]
    mu4 = state[..., 4]
    conc = state[..., 5]
    # d43 = (mu4 / mu3) * 1e6 [um], guarded against the early-time near-zero-
    # moments regime. The double-``where`` pattern is the canonical JAX-grad-
    # safe guarded division: a naive ``jnp.where(cond, mu4/mu3, 0.0)`` still
    # evaluates ``mu4/mu3`` on the masked branch, producing inf/nan whose
    # gradient flows back through ``where`` and poisons the loss. Replacing
    # the divisor on the masked branch gives a finite gradient on both sides.
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

    ``predictors = (growth_BP, nucleation_BP)``. Each consumes
    ``(temperature_C, supersaturation)`` and emits one bounded log-rate;
    the vector field exponentiates with ``10**(.)`` to recover physical units.
    Temperature is constant per experiment; supersaturation is state-derived
    each step. Both rates are gated by ``(S > 1 + eps)`` so the ODE stops
    moving below the metastable limit.
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

        # Hulburt-Katz form for size-independent nucleation at zero size and
        # pure linear growth, plus a mass balance: solute lost equals crystal
        # volume gained.
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
        # 1/1000 of the full span, floored at 1 s so the first step doesn't
        # underflow on short trajectories.
        span = times_sec[-1] - times_sec[0]
        dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
    else:
        dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)

    # diffrax.PIDController needs atol as a scalar/array — tuples don't
    # broadcast against the y_error PyTree leaves.
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


# --------------------------------------------------------------------------- #
# Script body                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    # KAN-specific hyperparameters. ``grid_size`` is the spline resolution
    # per edge — 5 is the jaxkan default and is appropriate for the smooth
    # log-rate surfaces we expect here. ``basis`` selects the layer
    # parameterisation: ``spline`` is the canonical KAN (learnable spline
    # + residual), ``base`` drops the spline and is kept as an ablation.
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

    # ---- Build experiments from the hardcoded LowData4 cut ----------------- #
    # Each entry in ``EXPERIMENTS_DATA`` is converted into one
    # ``hybridmodels.Experiment`` carrying two channels with their own ``ts``
    # axes — the d43 channel has a single terminal observation, so
    # ``make_dataset`` will form per-experiment union axes and bucket on
    # length parity. ``CONC_VAR`` is broadcast to a per-row variance vector;
    # the d43 variance is scalar (one observation per experiment).
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

    # ---- Build dataset (bucket + union-axis logic) ------------------------ #
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

    # ---- Solver ----------------------------------------------------------- #
    # Per-state atol matched to the natural moment magnitudes (``mu0 ~ 1e11``,
    # ``mu1 ~ 1e6``, ..., ``mu4 ~ 1e-7``, ``conc ~ 1``). A uniform atol
    # forces the PIDController to over-resolve small components and
    # under-resolve large ones, and the integrator hits ``max_steps`` before
    # finishing one trajectory. Setting atol per-component ~9 decades below
    # each natural magnitude lets ``rtol`` dominate once values are
    # appreciable, with atol acting as a near-zero floor.
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )

    # ---- Build predictor pairs (MLP first, then KAN) ---------------------- #
    # Each BoundedPredictor branch:
    #     dict -> [2] (INPUT_KEYS order)
    #          -> in_scaler  : physical -> latent (logit-of-normalised)
    #          -> inner      : [2] -> [1] (MLPPredictor or KANPredictor)
    #          -> out_scaler : latent -> physical (sigmoid into log-bounds)
    # Both pairs share the input scaler; growth/nucleation branches inside
    # each pair get independent inner weights via key splitting. KAN
    # ``hidden_widths=(64,)`` mirrors the MLP's single hidden layer of width
    # 64 so the comparison varies only the inner-network family.
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
    # ``with_zero_final_head`` zeroes the inner KAN's readout layer so the
    # bounded predictor's init lands at the physical midpoint of each
    # log-bound. Without this warm start, jaxkan's default spline+residual
    # init produces a non-zero inner output that, composed with the
    # asymmetric out-scaler ``low + (high-low)*sigmoid(z)`` for
    # ``LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)``, parks log10_J several
    # decades below the midpoint. With ``J ~ 2`` the moment ODE never
    # evolves over the trajectory and ``∂loss/∂params`` through the
    # integrator is numerically vanishing — Adam stays at a flat loss for
    # the entire run. The MLP path is not affected because random linear
    # init naturally produces near-zero output for a 2->64->1 net.
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

    # Sanity-check evaluation at the first experiment's covariates and a
    # plausible mid-range supersaturation. Same input shape the vector field
    # constructs each timestep.
    sample_cov = experiments[0].covariates
    sample_inputs = {
        "temperature_C": jnp.asarray(sample_cov["temperature_C"]),
        "supersaturation": jnp.asarray(1.5),
    }

    # ---- Train + diagnose + plot, once per family ------------------------ #
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

        Reuses the outer-scope ``dataset``, ``solver``, ``config``,
        ``k_train``, ``sample_inputs``, ``sample_cov``, and ``args``.
        ``label`` becomes the plot subdirectory and is interpolated into
        figure titles so MLP and KAN outputs land side by side under
        ``args.plot_dir``.
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
            f"  loss every ~{sample_every} steps: {[f'{loss:.4f}' for loss in history[::sample_every]]}"
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
