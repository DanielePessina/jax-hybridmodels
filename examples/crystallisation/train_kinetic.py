"""Crystallisation kinetic-MLP training example.

Loads the thesis crystallisation Excel dataset (irregular concentration + d43
observations across many experiments), wraps each experiment into an
``hybridmodels.Experiment``, and trains a hybrid model where two MLPs predict
reaction rates that feed a method-of-moments ODE.

Each MLP consumes ``(temperature_C, supersaturation)`` and emits a bounded
log-rate — ``log10(G)`` for crystal growth, ``log10(J)`` for nucleation.
``supersaturation`` is state-derived inside the vector field; the
population-balance moment ODEs and mass balance around them stay mechanistic.

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
import pandas as pd  # noqa: E402
from jax import Array  # noqa: E402
from jaxtyping import Float  # noqa: E402

from hybridmodels import (  # noqa: E402
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
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

EXCEL_PATH_DEFAULT = (
    Path(__file__).parent / "data" / "NODE_fullExperimental_dataset_ps3-dec2025__thesis.xlsx"
)
# ``Unseeded_thesis`` is the full thesis cut: per-row variances and a
# particle-size column (``d43`` in newer cuts, ``PS`` in legacy thesis cuts)
# with ``-1`` as the missing-row sentinel. The bare ``Unseeded`` sheet has
# only concentration; choosing it would silently skip every experiment.
DEFAULT_SHEET = "Unseeded_thesis"

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
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH_DEFAULT)
    parser.add_argument("--sheets", nargs="+", default=[DEFAULT_SHEET])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    k_init, k_train = jr.split(jr.PRNGKey(args.seed), 2)

    # ---- Load experiments from the thesis Excel --------------------------- #
    # Excel column contract: ``Exp_ID`` (group key), ``Time`` (min),
    # ``Concentration`` (+ optional ``Concentration_var``), ``Temperature``
    # (°C, per-experiment scalar), ``Loading`` (per-experiment scalar), and a
    # particle-size column — ``d43`` in newer cuts, ``PS`` in legacy ones.
    # The size column uses ``-1`` (and any non-finite value) as a "no
    # observation at this row" sentinel; filtering those rows recovers the
    # per-channel sparsity that ``make_dataset`` then turns into a union axis.
    print(f"[load] {args.excel}")
    sheet_dict = pd.read_excel(args.excel, sheet_name=list(args.sheets))
    if not isinstance(sheet_dict, dict):
        sheet_dict = {args.sheets[0]: sheet_dict}

    experiments: list[Experiment] = []
    for sheet_name, df in sheet_dict.items():
        d43_col = "d43" if "d43" in df.columns else ("PS" if "PS" in df.columns else None)
        d43_var_col = (
            "d43_var" if "d43_var" in df.columns else ("PS_var" if "PS_var" in df.columns else None)
        )
        for exp_id, df_exp in df.groupby("Exp_ID"):
            time_min = jnp.asarray(df_exp["Time"].to_numpy(dtype=float))
            conc = jnp.asarray(df_exp["Concentration"].to_numpy(dtype=float))

            if "Concentration_var" in df_exp.columns:
                cv_raw = jnp.asarray(df_exp["Concentration_var"].to_numpy(dtype=float))
                conc_var = jnp.where(jnp.isfinite(cv_raw) & (cv_raw > 0.0), cv_raw, 1e-4)
            else:
                conc_var = jnp.full_like(conc, 1e-4)

            channels: dict[str, ChannelObs] = {
                "conc": ChannelObs(ts=time_min, values=conc, variance=conc_var),
            }

            if d43_col is not None:
                d43_arr = jnp.asarray(df_exp[d43_col].to_numpy(dtype=float))
                valid_mask = (d43_arr > 0.0) & jnp.isfinite(d43_arr)
                if bool(jnp.any(valid_mask)):
                    valid_idx = jnp.where(valid_mask)[0]
                    d43_ts = time_min[valid_idx]
                    d43_vals = d43_arr[valid_idx]
                    if d43_var_col is not None:
                        var_raw = jnp.asarray(df_exp[d43_var_col].to_numpy(dtype=float))[valid_idx]
                        d43_var = jnp.where(jnp.isfinite(var_raw) & (var_raw > 0.0), var_raw, 1e-2)
                    else:
                        d43_var = jnp.full(d43_vals.shape, 1e-2)
                    channels["d43"] = ChannelObs(ts=d43_ts, values=d43_vals, variance=d43_var)

            # ``make_dataset`` requires every experiment to define every
            # channel listed in ``OUTPUT_CHANNELS``; skip experiments with no
            # particle-size data rather than fabricate empty channels.
            if "d43" not in channels:
                continue

            experiments.append(
                make_experiment(
                    covariates={
                        "temperature_C": float(df_exp["Temperature"].iloc[0]),
                        "loading": float(df_exp["Loading"].iloc[0]),
                    },
                    channels=channels,
                    y0_fn=y0_fn,
                    exp_id=f"{sheet_name}_{int(exp_id)}",
                )
            )

    if not experiments:
        raise RuntimeError(
            f"no experiments loaded from {args.excel} (sheets={args.sheets}); "
            "check the file path and that PS/d43 observations exist."
        )

    print(f"  {len(experiments)} experiments loaded from {args.sheets}")
    for exp in experiments[:3]:
        print(
            f"    {exp.exp_id}: T={float(exp.covariates['temperature_C']):.1f}°C, "
            f"L={float(exp.covariates['loading']):.2f}, "
            f"conc obs={exp.channels['conc'].values.shape[0]}, "
            f"d43 obs={exp.channels['d43'].values.shape[0]}"
        )
    if len(experiments) > 3:
        print(f"    ... and {len(experiments) - 3} more")

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

    # ---- Build predictors (two BoundedPredictor branches) ----------------- #
    # Each branch:
    #     dict -> [2] (INPUT_KEYS order)
    #          -> in_scaler  : physical -> latent (logit-of-normalised)
    #          -> MLPPredictor: [2] -> [1] (relu, depth=1, width=64)
    #          -> out_scaler : latent -> physical (sigmoid into log-bounds)
    # Branches share the input scaler but get independent MLP weights via key
    # splitting.
    k_growth, k_nucleation = jr.split(k_init, 2)
    in_scaler = BoundScaler(
        bounds=(TEMPERATURE_BOUNDS, SUPERSATURATION_BOUNDS),
        transform="sigmoid",
    )

    growth_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=in_scaler,
        inner=MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=1,
            activation_name="relu",
            key=k_growth,
        ),
        out_scaler=BoundScaler(bounds=(LOG10_GROWTH_BOUNDS,), transform="sigmoid"),
    )
    nucleation_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=in_scaler,
        inner=MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=1,
            activation_name="relu",
            key=k_nucleation,
        ),
        out_scaler=BoundScaler(bounds=(LOG10_NUCLEATION_BOUNDS,), transform="sigmoid"),
    )
    predictors = (growth_bp, nucleation_bp)

    # Sanity-check evaluation at the first experiment's covariates and a
    # plausible mid-range supersaturation. Same input shape the vector field
    # constructs each timestep.
    sample_cov = experiments[0].covariates
    sample_inputs = {
        "temperature_C": jnp.asarray(sample_cov["temperature_C"]),
        "supersaturation": jnp.asarray(1.5),
    }
    log10_G_init = float(jnp.squeeze(growth_bp(sample_inputs)))
    log10_J_init = float(jnp.squeeze(nucleation_bp(sample_inputs)))
    print(
        f"\n[init] T={float(sample_cov['temperature_C']):.1f}°C, S=1.5 -> "
        f"log10_G={log10_G_init:.2f} (G={10.0**log10_G_init:.2e} m/s), "
        f"log10_J={log10_J_init:.2f} (J={10.0**log10_J_init:.2e} #/m³/s)"
    )

    # ---- Train ------------------------------------------------------------ #
    print("\n[train] optax (single phase, mse loss)")
    config = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        log_every=max(1, args.steps // 10),
        verbose=True,
    )
    history, trained_predictors = train_with_optax(
        predictors,
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

    # ---- Diagnostics + plots --------------------------------------------- #
    # ``predict_dataset`` returns one ``[N, T, D]`` array per bucket; the
    # helpers walk it in lockstep with the dataset.
    print("\n[diagnostics] per-channel parity stats over the training set")
    predictions = predict_dataset(
        trained_predictors,
        dataset,
        simulate_fn=simulate_fn,
        solver=solver,
    )
    diag = compute_diagnostics(predictions, dataset)
    print_diagnostics(diag)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            diag,
            title="Crystallisation parity (trained model)",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            max_experiments=6,
            title="Crystallisation trajectories (first 6 experiments)",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
