"""Crystallisation kinetic-MLP training (single-file end-to-end example).

Overview
--------
Loads the thesis crystallisation Excel dataset (irregular concentration + d43
observations across many experiments), wraps it into ``hybridmodels.Experiment``
objects, defines the method-of-moments ODE in the same form as
``hybridcrystals.mechanistic.vector_ode``, and trains an MLP that maps
``(temperature_C, loading)`` to four kinetic parameters ``(logA, gamma, Ag, g)``.

The MLP outputs are sigmoid-bounded into physical units via ``BoundedPredictor``;
those parameters then drive CNT-nucleation and power-law-growth rate laws inside
``simulate_fn``.

Initial state convention
------------------------
``y0 = [0, 0, 0, 0, 0, init_conc]`` — the five population-balance moments
``mu0..mu4`` start at zero, and the dissolved-solute concentration starts at
the experiment's first observed concentration value.

Phase status
------------
Uses every module shipped through Phase 9 (optax training):

- Phase 1 ``data`` — ``ChannelObs``, ``Experiment``, ``make_dataset``
- Phase 2 ``solver`` — ``SolverConfig``
- Phase 3/4 ``predictors`` — ``BoundedPredictor``, ``BoundScaler``,
  ``CovariateSelector``, ``MLPPredictor``
- Phase 5 ``losses`` (``masked_mse``) / ``prediction``
- Phase 6 ``trainable`` (default mask via ``train_with_optax``)
- Phase 7 ``rng`` (consumed inside training internals)
- Phase 8 ``ui`` (``SilentUI`` selected via ``verbose=False``)
- Phase 9 ``training/optax`` — ``train_with_optax``

How to run
----------
``uv run python examples/crystallisation/train_kinetic.py``

The thesis Excel is bundled at ``examples/crystallisation/data/`` so the
script needs no external path. Override with ``--excel /path/to/file.xlsx``.

Bounds reference
----------------
Covariate and parameter bounds are reproduced from
``hybridcrystals.regressor_constants`` so the trained predictor lives in the
same physical-units box as the source-package thesis runs.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from jax import Array
from jaxtyping import Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    CovariateSelector,
    Experiment,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    make_experiment,
)
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

EXCEL_PATH_DEFAULT: Path = (
    Path(__file__).parent / "data" / "NODE_fullExperimental_dataset_ps3-dec2025__thesis.xlsx"
)
"""Default thesis dataset (irregular concentration + d43 over ~50 experiments).

Bundled into the repo at ``examples/crystallisation/data/`` so the script
runs out-of-the-box; the path is resolved relative to this file."""

DEFAULT_SHEETS: tuple[str, ...] = ("Unseeded",)
"""Excel sheets to load. Restricted to a single system to keep training short
and to dodge multi-system conditioning (out of scope per SPEC §2.2)."""

COVARIATE_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_C": (13.0, 27.0),
    "loading": (0.0, 30.0),
}
"""``(low, high)`` pairs for input covariates; consumed by ``in_scaler``.
Slightly wider than the data's actual span so sigmoid saturation is rare."""

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "logA": (20.0, 65.0),    # ln of CNT pre-exponential
    "gamma": (0.15, 1.0),    # interfacial energy [mJ/m^2]
    "Ag": (-10.0, 6.0),      # log10 of growth pre-factor [m/s], converted to /min via /60
    "g": (1.0, 3.5),         # power-law growth exponent
}
"""Output bounds for the kinetic parameters; consumed by ``out_scaler``."""

# ODE constants — from hybridcrystals/mechanistic.py
_RHO_C: float = 1370.0   # crystal density [kg/m^3]
_K_V: float = 0.81       # volumetric shape factor
_M_V: float = 2.97e-26   # molecular volume [m^3]
_K_B: float = 1.38064852e-23  # Boltzmann constant [J/K]
_D43_MAX: float = 55.0   # upper guard on d43 [um]
_D43_MU3_EPS: float = 1e-6
_META_EPS: float = 1e-5  # supersaturation must exceed 1 + eps for nucleation/growth

OUTPUT_CHANNELS: tuple[str, ...] = ("conc", "d43")
"""Channel order on the trailing ``D`` axis of every ``BucketPayload``."""

INPUT_KEYS: tuple[str, ...] = ("temperature_C", "loading")
"""Covariate order consumed by the predictor's ``CovariateSelector``."""


# --------------------------------------------------------------------------- #
# Dataset loader                                                              #
# --------------------------------------------------------------------------- #


def _load_experiments(
    excel_path: Path, *, sheet_names: Sequence[str] = DEFAULT_SHEETS
) -> list[Experiment]:
    """Load experiments from the thesis Excel into ``hybridmodels.Experiment`` records.

    Excel column contract (per ``hybridcrystals/data/_core.py::get_experiment_list``):

    - ``Exp_ID`` (int)              — groups rows belonging to one experiment
    - ``System`` (str)              — chemical system name; carried into ``exp_id``
    - ``Time`` (float, minutes)     — observation time
    - ``Concentration`` (float)     — solute concentration (dense across rows)
    - ``Concentration_var`` (float) — per-row variance (optional; defaults to ``1e-4``)
    - ``Temperature`` (°C)          — per-experiment scalar (constant within ``Exp_ID``)
    - ``Loading`` (float)           — per-experiment scalar
    - ``PS`` (float, optional)      — particle size (d43, ``[um]``); ``-1`` marks missing
    - ``PS_var`` (float, optional)  — per-row variance for ``PS``

    Per-channel sparsity is recovered by filtering rows where ``PS == -1`` or
    ``PS`` is non-finite; those timestamps are dropped from the d43 channel's
    ``ts`` axis but still appear on the dense concentration channel. The
    framework's union-axis logic in ``make_dataset`` builds the per-experiment
    ``[T, D]`` mask without any user mask code.

    Time stays in **minutes** in the dataset; ``simulate_fn`` converts to seconds
    before integration so the rate constants (CNT exponent uses K and seconds,
    growth pre-factor is ``10**Ag / 60`` to convert per-minute) line up.
    """
    sheet_dict = pd.read_excel(excel_path, sheet_name=list(sheet_names))
    if not isinstance(sheet_dict, dict):
        sheet_dict = {sheet_names[0]: sheet_dict}

    experiments: list[Experiment] = []
    for sheet_name, df in sheet_dict.items():
        for exp_id, df_exp in df.groupby("Exp_ID"):
            time_min = jnp.asarray(df_exp["Time"].to_numpy(dtype=float))
            conc = jnp.asarray(df_exp["Concentration"].to_numpy(dtype=float))

            if "Concentration_var" in df_exp.columns:
                cv_raw = jnp.asarray(df_exp["Concentration_var"].to_numpy(dtype=float))
                conc_var = jnp.where(
                    jnp.isfinite(cv_raw) & (cv_raw > 0.0), cv_raw, 1e-4
                )
            else:
                conc_var = jnp.full_like(conc, 1e-4)

            channels: dict[str, ChannelObs] = {
                "conc": ChannelObs(ts=time_min, values=conc, variance=conc_var),
            }

            if "PS" in df_exp.columns:
                ps_raw = df_exp["PS"].to_numpy(dtype=float)
                # -1 sentinel and NaN both mean "no PS at this row"; drop them
                # so the d43 channel's ts axis is sparse.
                ps_arr = jnp.asarray(ps_raw)
                valid_mask = (ps_arr > 0.0) & jnp.isfinite(ps_arr)
                if bool(jnp.any(valid_mask)):
                    valid_idx = jnp.where(valid_mask)[0]
                    d43_ts = time_min[valid_idx]
                    d43_vals = ps_arr[valid_idx]
                    if "PS_var" in df_exp.columns:
                        psv_raw = jnp.asarray(
                            df_exp["PS_var"].to_numpy(dtype=float)
                        )[valid_idx]
                        d43_var = jnp.where(
                            jnp.isfinite(psv_raw) & (psv_raw > 0.0), psv_raw, 1e-2
                        )
                    else:
                        d43_var = jnp.full(d43_vals.shape, 1e-2)
                    channels["d43"] = ChannelObs(
                        ts=d43_ts, values=d43_vals, variance=d43_var
                    )

            if "d43" not in channels:
                # make_dataset requires every experiment to define every output
                # channel listed in OUTPUT_CHANNELS. Skip experiments with no
                # particle-size data rather than fabricate empty channels.
                continue

            experiments.append(
                make_experiment(
                    covariates={
                        "temperature_C": float(df_exp["Temperature"].iloc[0]),
                        "loading": float(df_exp["Loading"].iloc[0]),
                    },
                    channels=channels,
                    y0_fn=_y0_fn,
                    exp_id=f"{sheet_name}_{int(exp_id)}",
                )
            )

    if not experiments:
        raise RuntimeError(
            f"no experiments loaded from {excel_path} (sheets={list(sheet_names)}); "
            "check the file path and that PS observations exist."
        )
    return experiments


# --------------------------------------------------------------------------- #
# Hooks: y0_fn, state_to_output                                               #
# --------------------------------------------------------------------------- #


def _y0_fn(
    covariates: dict[str, Array], channels: dict[str, ChannelObs]
) -> Float[Array, " 6"]:
    """Build the full initial state ``[mu0, mu1, mu2, mu3, mu4, conc]``.

    All five population-balance moments start at zero (the suspension is
    nominally clear at ``t = 0``); the dissolved-solute concentration starts
    at the first observed value of the ``conc`` channel for this experiment.
    Matches ``hybridcrystals/mechanistic.py::simulate_ode_from_arrays`` line 331.
    """
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate(
        [jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]]
    )


def _d43_from_moments(
    mu3: Float[Array, " T"], mu4: Float[Array, " T"]
) -> Float[Array, " T"]:
    """Volume-weighted mean diameter ``d43 = (mu4 / mu3) * 1e6`` [um], guarded.

    The ``1e6`` factor converts metres to micrometres. Outputs are clamped to
    ``[0, _D43_MAX]`` and ``mu3 < eps`` is replaced with ``0`` to avoid
    division blow-ups during the early-time near-zero-moments regime.
    Mirrors ``hybridcrystals/mechanistic.py::d43_from_moments``.
    """
    raw = jnp.where(mu3 > _D43_MU3_EPS, (mu4 / mu3) * 1e6, 0.0)
    finite_positive = jnp.isfinite(raw) & (raw > 0.0)
    guarded = jnp.where(finite_positive, raw, 0.0)
    return jnp.clip(guarded, 0.0, _D43_MAX)


def _state_to_output(
    state: Float[Array, "T 6"],
) -> Float[Array, "T 2"]:
    """Project full state ``[mu0..mu4, conc]`` to observed channels ``[conc, d43]``.

    Channel order must match ``OUTPUT_CHANNELS``. Channel 0 is concentration
    (state index 5); channel 1 is the volume-weighted diameter d43 derived
    from moments 3 and 4.
    """
    mu3 = state[..., 3]
    mu4 = state[..., 4]
    conc = state[..., 5]
    d43 = _d43_from_moments(mu3, mu4)
    return jnp.stack([conc, d43], axis=-1)


# --------------------------------------------------------------------------- #
# simulate_fn — method-of-moments ODE                                         #
# --------------------------------------------------------------------------- #


def _conc_sat(temperature_C: Array) -> Array:
    """Empirical saturation-concentration polynomial in °C.

    From ``hybridcrystals/mechanistic.py::simulate_ode_from_arrays`` line 328;
    physical units match ``Concentration`` in the dataset.
    """
    return (
        0.3705
        + 7.171e-2 * temperature_C
        - 1.924e-3 * temperature_C**2
        + 17.97e-5 * temperature_C**3
    )


def _simulate_fn(
    predictor: BoundedPredictor,
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 6"],
    solver: SolverConfig,
) -> Float[Array, "T 6"]:
    """Integrate the method-of-moments ODE for one experiment.

    Mandatory ``simulate_fn`` shape per SPEC §4.2 / R-A2: returns the full
    state at every timestamp in ``ts``. The user supplies physics; the
    framework owns vmap, jit, and gradient flow.

    Pipeline
    --------
    1. Pull ``temperature_C`` from ``covariates`` (constant in time).
    2. Evaluate ``predictor(covariates) -> [logA, gamma, Ag, g]``; these are
       the *bounded* kinetic parameters in physical units already (the
       ``BoundedPredictor`` has applied its sigmoid output scaler).
    3. Define the RHS: nucleation rate ``J`` from CNT, growth rate ``G`` from
       power-law. Both are gated by a ``(supersaturation > 1 + 1e-5)`` mask so
       the ODE stops moving below the metastable limit.
    4. Convert ``ts`` from minutes to seconds (the dataset stores minutes;
       the rate constants are SI-second-based) and call ``diffrax.diffeqsolve``.

    Shape conventions
    -----------------
    ``ts``: ``[T]`` (per-experiment union axis, in minutes).
    ``covariates``: dict of 0-d arrays.
    ``y0``: ``[6] = [mu0, mu1, mu2, mu3, mu4, conc]`` per ``_y0_fn``.
    Returns ``sol.ys`` of shape ``[T, 6]`` aligned with ``ts``.
    """
    temperature_C = covariates["temperature_C"]
    conc_sat = _conc_sat(temperature_C)
    T_K = temperature_C + 273.15

    params = predictor(covariates)  # [4] in physical units (bounded)
    logA = params[0]
    gamma_mJ_m2 = params[1]
    Ag = params[2]
    g_exp = params[3]
    gamma_J_m2 = gamma_mJ_m2 * 1e-3  # [mJ/m^2] -> [J/m^2]

    def vector_field(
        t: Array, y: Float[Array, " 6"], args: object
    ) -> Float[Array, " 6"]:
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat
        meta_mask = (S > 1.0 + _META_EPS).astype(y.dtype)

        # CNT nucleation J = exp(logA) * S * exp(-16π γ³ v² / (3 (k_B T)³ ln²S))
        S_safe = jnp.clip(S, min=1.0 + 1e-12)
        logS = jnp.log(S_safe)
        cnt_exp = (
            -16.0 * jnp.pi * gamma_J_m2**3 * _M_V**2
            / (3.0 * (_K_B * T_K) ** 3 * logS**2)
        )
        J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exp)

        # Power-law growth G = (10**Ag / 60) * max(S - 1, 0)**g
        growth_drive = jnp.maximum(S - 1.0, 0.0)
        G = meta_mask * jnp.power(10.0, Ag) / 60.0 * jnp.power(growth_drive, g_exp)

        # Population-balance moment ODEs (Hulburt-Katz form for size-independent
        # nucleation at zero size and pure linear growth).
        dmu0 = J
        dmu1 = G * mu0
        dmu2 = 2.0 * G * mu1
        dmu3 = 3.0 * G * mu2
        dmu4 = 4.0 * G * mu3
        # Mass balance: solute lost to growing crystal volume.
        dconc = -3.0 * _K_V * _RHO_C * G * mu2

        return jnp.stack([dmu0, dmu1, dmu2, dmu3, dmu4, dconc])

    times_sec = ts * 60.0
    term = diffrax.ODETerm(vector_field)
    saveat = diffrax.SaveAt(ts=times_sec)
    # Heuristic dt0 if user did not pin one: 1/1000 of the full span,
    # floored at 1 second to keep the first step from underflowing.
    if solver.dt0 is None:
        span = times_sec[-1] - times_sec[0]
        dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
    else:
        dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)

    # diffrax.PIDController arithmetic requires atol as an array (or scalar) —
    # tuples don't broadcast against the y_error PyTree leaves.
    atol = (
        jnp.asarray(solver.atol, dtype=times_sec.dtype)
        if isinstance(solver.atol, tuple)
        else solver.atol
    )

    sol = diffrax.diffeqsolve(
        term,
        solver.solver,
        t0=times_sec[0],
        t1=times_sec[-1],
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


# --------------------------------------------------------------------------- #
# Predictor                                                                   #
# --------------------------------------------------------------------------- #


def _build_predictor(key: Array) -> BoundedPredictor:
    """Construct the covariate-conditioned kinetic-parameter predictor.

    Pipeline ``(temperature_C, loading) -> [logA, gamma, Ag, g]``::

        CovariateSelector  : dict -> [2] in declared INPUT_KEYS order
        in_scaler          : [2] physical -> [2] latent (logit-of-normalised)
        MLPPredictor       : [2] -> [4]   (tanh, depth=2, width=16)
        out_scaler         : [4] latent -> [4] physical (sigmoid into bounds)

    Output is in physical units already; ``simulate_fn`` consumes it directly.
    """
    selector = CovariateSelector(keys=INPUT_KEYS)
    in_scaler = BoundScaler(
        bounds=tuple(COVARIATE_BOUNDS[k] for k in INPUT_KEYS),
        transform="sigmoid",
    )
    inner = MLPPredictor(
        in_size=len(INPUT_KEYS),
        out_size=4,
        width_size=16,
        depth=2,
        activation_name="tanh",
        key=key,
    )
    out_scaler = BoundScaler(
        bounds=(
            PARAMETER_BOUNDS["logA"],
            PARAMETER_BOUNDS["gamma"],
            PARAMETER_BOUNDS["Ag"],
            PARAMETER_BOUNDS["g"],
        ),
        transform="sigmoid",
    )
    return BoundedPredictor(
        selector=selector,
        in_scaler=in_scaler,
        inner=inner,
        out_scaler=out_scaler,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH_DEFAULT)
    parser.add_argument("--sheets", nargs="+", default=list(DEFAULT_SHEETS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Enable float64 to match source-package thesis runs; mass-balance in the
    # moment ODE is stiff enough that float32 sometimes drifts.
    jax.config.update("jax_enable_x64", True)

    root_key = jr.PRNGKey(args.seed)
    k_init, k_train = jr.split(root_key, 2)

    print(f"[load] {args.excel}")
    experiments = _load_experiments(args.excel, sheet_names=tuple(args.sheets))
    print(f"  {len(experiments)} experiments loaded from {args.sheets}")
    for exp in experiments[:3]:
        n_conc = exp.channels["conc"].values.shape[0]
        n_d43 = exp.channels["d43"].values.shape[0]
        print(
            f"    {exp.exp_id}: T={float(exp.covariates['temperature_C']):.1f}°C, "
            f"L={float(exp.covariates['loading']):.2f}, "
            f"conc obs={n_conc}, d43 obs={n_d43}"
        )
    if len(experiments) > 3:
        print(f"    ... and {len(experiments) - 3} more")

    print("\n[build] dataset + bucketing")
    dataset = make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(
            f"    bucket {i}: ts={tuple(bp.ts.shape)}, "
            f"y_observed={tuple(bp.y_observed.shape)}, "
            f"mask={tuple(bp.mask.shape)}, n_obs={int(bp.n_obs)}"
        )

    print("\n[build] solver + predictor")
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=(1e-5,) * 6,
        max_steps=500_000,
        dt0=None,
    )
    predictor = _build_predictor(k_init)
    sample_cov = experiments[0].covariates
    sample_params = predictor(sample_cov)
    print(
        f"  predictor sample on T={float(sample_cov['temperature_C']):.1f}°C, "
        f"L={float(sample_cov['loading']):.2f} -> "
        f"logA={float(sample_params[0]):.2f}, "
        f"gamma={float(sample_params[1]):.3f} mJ/m^2, "
        f"Ag={float(sample_params[2]):.2f}, "
        f"g={float(sample_params[3]):.2f}"
    )

    print("\n[train] optax (single phase, mse loss)")
    config = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        log_every=max(1, args.steps // 10),
        verbose=False,
    )
    history, trained = train_with_optax(
        predictor,
        dataset,
        config,
        simulate_fn=_simulate_fn,
        solver=solver,
        key=k_train,
    )

    print(f"  {len(history)} steps; final loss {history[-1]:.6f}")
    sample_every = max(1, len(history) // 10)
    sampled = [f"{loss:.4f}" for loss in history[::sample_every]]
    print(f"  loss every ~{sample_every} steps: {sampled}")

    final_params = trained(sample_cov)
    print(
        f"  trained predictor on same input -> "
        f"logA={float(final_params[0]):.2f}, "
        f"gamma={float(final_params[1]):.3f} mJ/m^2, "
        f"Ag={float(final_params[2]):.2f}, "
        f"g={float(final_params[3]):.2f}"
    )


if __name__ == "__main__":
    main()
