"""Crystallisation kinetic-MLP training (single-file end-to-end example).

Overview
--------
Loads the thesis crystallisation Excel dataset (irregular concentration + d43
observations across many experiments), wraps it into ``hybridmodels.Experiment``
objects, and trains a hybrid model where MLPs predict reaction rates feeding
the method-of-moments ODE.

This file demonstrates **two predictor parameterisations** of the same
crystallisation hybrid model. Only the *direct-rate* path (1) is wired into
``main()``; the *kinetic-parameter* path (2) is preserved as commented-out
reference code so the comparison stays explicit.

(1) Direct-rate predictors — **active**
    Two ``BoundedPredictor``s consuming a 3-key input dict
    ``(temperature_C, loading, supersaturation)`` and emitting bounded
    log-rates::

        predictors[0]: inputs -> log10_G   (growth log-rate, [m/s])
        predictors[1]: inputs -> log10_J   (nucleation log-rate, [#/(m^3·s)])

    Inside ``simulate_fn``'s vector field, ``supersaturation`` is computed
    per timestep from the state (``S = conc/conc_sat(T)``) and mixed into
    the inputs dict. Each call yields a physical rate after a single
    ``power(10, log_rate)``. This collapses kinetic mechanism into the
    network — the user no longer commits to a CNT / power-law form, the
    network learns whatever ``(T, L, S) -> rate`` mapping the data implies.

(2) Kinetic-parameter predictor — **commented-out reference**
    A single ``BoundedPredictor`` consuming a 2-key dict
    ``(temperature_C, loading)`` and emitting four bounded scalars
    ``(logA, gamma, Ag, g)``. The vector field then plugs these into
    classical CNT (nucleation) and power-law (growth) forms with
    ``exp(logA)`` / ``10**Ag`` scalings done inside the user's vector field
    code. Closer to the source-package thesis script
    ``hybridcrystals/thesis_training/sharedgrowth.py``; preserved here so a
    user can flip back by uncommenting and switching ``main()``'s builder
    selection.

The two paths share the same data loader, ``y0_fn``, ``state_to_output``,
solver config, dataset shape, and training entry-point — only the predictors
pytree and the rate-derivation logic inside ``simulate_fn`` differ.

Initial state convention
------------------------
``y0 = [0, 0, 0, 0, 0, init_conc]`` — the five population-balance moments
``mu0..mu4`` start at zero, and the dissolved-solute concentration starts at
the experiment's first observed concentration value.

Modules exercised by this example
---------------------------------
The script exercises the framework end-to-end:

- ``hybridmodels.data`` — ``ChannelObs``, ``Experiment``, ``make_dataset``
- ``hybridmodels.solver`` — ``SolverConfig``
- ``hybridmodels.predictors`` — ``BoundedPredictor``, ``BoundScaler``,
  ``CovariateSelector``, ``MLPPredictor``
- ``hybridmodels.losses`` (``masked_mse``) and ``hybridmodels.prediction``
- ``hybridmodels.trainable`` (default mask via ``train_with_optax``)
- ``hybridmodels.rng`` (consumed inside training internals)
- ``hybridmodels.ui`` (``SilentUI`` selected via ``verbose=False``)
- ``hybridmodels.training.optax`` — ``train_with_optax``

How to run
----------
``uv run python examples/crystallisation/train_kinetic.py``

The thesis Excel is bundled at ``examples/crystallisation/data/`` so the
script needs no external path. Override with ``--excel /path/to/file.xlsx``.
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
"""Excel sheets to load. Restricted to a single system so the example
trains in a few minutes and avoids the multi-system conditioning
problem, which is out of scope for this example."""

COVARIATE_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_C": (13.0, 27.0),
    "loading": (0.0, 30.0),
}
"""``(low, high)`` pairs for input covariates; consumed by ``in_scaler``.
Slightly wider than the data's actual span so sigmoid saturation is rare."""

# Direct-rate path (1) — bounds on predictor inputs and outputs.
SUPERSATURATION_BOUNDS: tuple[float, float] = (1.0, 3.0)
"""Physical bounds on supersaturation ``S = conc / conc_sat`` fed as the
third input key to each direct-rate predictor. The lower bound is at the
nucleation/growth threshold (``S = 1``); the upper end is generous for the
thesis dataset's typical ``S`` range."""

LOG10_GROWTH_BOUNDS: tuple[float, float] = (-12.0, -3.0)
"""Bounds on ``log10(G)`` where ``G`` is growth velocity in m/s. The exponent
range covers ~9 decades; the vector field exponentiates with ``10**log10_G``
to recover physical units. Wide enough to subsume the source-package
power-law output range (``10**Ag * (S-1)**g`` with ``Ag ∈ [-10, 6]``)."""

LOG10_NUCLEATION_BOUNDS: tuple[float, float] = (0.0, 15.0)
"""Bounds on ``log10(J)`` where ``J`` is nucleation rate in #/(m^3·s).
The CNT form in the kinetic-parameter path produces values up to ~1e15
when supersaturation is high; this bound keeps the direct path commensurate."""

# Kinetic-parameter path (2) — bounds reproduced from the source-package
# thesis runs; consumed only by the commented-out ``_build_kinetic_predictor``
# below. Kept here so flipping the path requires only one uncomment.
KINETIC_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "logA": (20.0, 65.0),  # ln of CNT pre-exponential
    "gamma": (0.15, 1.0),  # interfacial energy [mJ/m^2]
    "Ag": (-10.0, 6.0),  # log10 of growth pre-factor [m/s], converted to /min via /60
    "g": (1.0, 3.5),  # power-law growth exponent
}

# ODE constants — from hybridcrystals/mechanistic.py
_RHO_C: float = 1370.0  # crystal density [kg/m^3]
_K_V: float = 0.81  # volumetric shape factor
_M_V: float = 2.97e-26  # molecular volume [m^3]
_K_B: float = 1.38064852e-23  # Boltzmann constant [J/K]
_D43_MAX: float = 55.0  # upper guard on d43 [um]
_D43_MU3_EPS: float = 1e-6
_META_EPS: float = 1e-5  # supersaturation must exceed 1 + eps for nucleation/growth

OUTPUT_CHANNELS: tuple[str, ...] = ("conc", "d43")
"""Channel order on the trailing ``D`` axis of every ``BucketPayload``."""

# Direct-rate path (1) — three-key input dict: two covariates + state-derived S.
INPUT_KEYS_DIRECT: tuple[str, ...] = ("temperature_C", "loading", "supersaturation")

# Kinetic-parameter path (2) — two covariates only (S is reconstructed in
# the vector field from the predicted gamma / Ag / g but is not a predictor
# input in this path).
INPUT_KEYS_KINETIC: tuple[str, ...] = ("temperature_C", "loading")


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
                conc_var = jnp.where(jnp.isfinite(cv_raw) & (cv_raw > 0.0), cv_raw, 1e-4)
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
                        psv_raw = jnp.asarray(df_exp["PS_var"].to_numpy(dtype=float))[valid_idx]
                        d43_var = jnp.where(jnp.isfinite(psv_raw) & (psv_raw > 0.0), psv_raw, 1e-2)
                    else:
                        d43_var = jnp.full(d43_vals.shape, 1e-2)
                    channels["d43"] = ChannelObs(ts=d43_ts, values=d43_vals, variance=d43_var)

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


def _y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 6"]:
    """Build the full initial state ``[mu0, mu1, mu2, mu3, mu4, conc]``.

    All five population-balance moments start at zero (the suspension is
    nominally clear at ``t = 0``); the dissolved-solute concentration starts
    at the first observed value of the ``conc`` channel for this experiment.
    Matches ``hybridcrystals/mechanistic.py::simulate_ode_from_arrays`` line 331.
    """
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])


def _d43_from_moments(mu3: Float[Array, " T"], mu4: Float[Array, " T"]) -> Float[Array, " T"]:
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


def _state_to_output(state: Float[Array, "T 6"]) -> Float[Array, "T 2"]:
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
# Saturation curve (shared by both predictor paths)                           #
# --------------------------------------------------------------------------- #


def _conc_sat(temperature_C: Array) -> Array:
    """Empirical saturation-concentration polynomial in °C.

    From ``hybridcrystals/mechanistic.py::simulate_ode_from_arrays`` line 328;
    physical units match ``Concentration`` in the dataset. Used by both
    predictor paths to compute supersaturation ``S = conc / conc_sat``.
    """
    return (
        0.3705
        + 7.171e-2 * temperature_C
        - 1.924e-3 * temperature_C**2
        + 17.97e-5 * temperature_C**3
    )


# --------------------------------------------------------------------------- #
# simulate_fn (1) — direct-rate predictors  (ACTIVE)                          #
# --------------------------------------------------------------------------- #


def _simulate_fn(
    predictors: tuple[BoundedPredictor, BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 6"],
    solver: SolverConfig,
) -> Float[Array, "T 6"]:
    """Integrate the method-of-moments ODE using two direct-rate predictors.

    Conforms to the framework's ``simulate_fn`` signature
    ``(predictors, ts, covariates, y0, solver) -> [T, S]`` — i.e. it
    returns the full simulator state at every timestamp in ``ts``. The
    user supplies the physics here; the framework owns the surrounding
    ``vmap``, ``jit``, and gradient plumbing.

    Predictor pytree
    ----------------
    ``predictors = (growth_BP, nucleation_BP)`` — a tuple of two
    ``BoundedPredictor``s, the convention this framework uses for
    multi-rate hybrid models. Each consumes a 3-key input dict
    ``(temperature_C, loading, supersaturation)`` and emits one
    bounded log-rate (``log10(G)`` or ``log10(J)``). Two of those
    inputs are constant covariates supplied by the experiment;
    ``supersaturation`` is *time-varying* — derived from the simulator
    state at each integrator step — and is mixed into the input dict
    alongside the covariates inside ``vector_field`` below.

    Pipeline
    --------
    1. Pull ``temperature_C`` from ``covariates`` (constant in time) and
       compute ``conc_sat(T)`` once.
    2. Inside ``vector_field``: compute ``S = conc / conc_sat`` from state,
       construct the per-call inputs dict, evaluate both predictors, then
       exponentiate the log-rates back to physical units (``10**log_rate``).
       Both rates are gated by a ``(S > 1 + 1e-5)`` mask so the ODE stops
       moving below the metastable limit.
    3. Convert ``ts`` from minutes to seconds (the dataset stores minutes;
       the rate constants are SI-second-based) and call
       ``diffrax.diffeqsolve``.

    Shape conventions
    -----------------
    ``ts``: ``[T]`` (per-experiment union axis, in minutes).
    ``covariates``: dict of 0-d arrays.
    ``y0``: ``[6] = [mu0, mu1, mu2, mu3, mu4, conc]`` per ``_y0_fn``.
    Returns ``sol.ys`` of shape ``[T, 6]`` aligned with ``ts``.
    """
    growth_bp, nucleation_bp = predictors

    temperature_C = covariates["temperature_C"]
    conc_sat = _conc_sat(temperature_C)

    def vector_field(t: Array, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat
        meta_mask = (S > 1.0 + _META_EPS).astype(y.dtype)

        # Construct the per-call predictor input dict. Key collision is
        # intentional: covariates' "temperature_C" / "loading" pass through
        # unchanged, "supersaturation" is the state-derived time-varying
        # value (CONTEXT.md "Predictor inputs"). Each predictor's selector
        # picks all three keys in its declared order.
        inputs = {
            "temperature_C": covariates["temperature_C"],
            "loading": covariates["loading"],
            "supersaturation": S,
        }

        # Bounded log-rates -> physical rates via 10**(.) inside the vector
        # field. The squeeze handles the [1] output shape from MLPPredictor
        # configured with out_size=1.
        log10_G = jnp.squeeze(growth_bp(inputs))
        log10_J = jnp.squeeze(nucleation_bp(inputs))
        G = meta_mask * jnp.power(10.0, log10_G)
        J = meta_mask * jnp.power(10.0, log10_J)

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
# simulate_fn (2) — kinetic-parameter predictor  (REFERENCE, COMMENTED-OUT)   #
# --------------------------------------------------------------------------- #
#
# The original kinetic-parameter path is preserved verbatim below. To
# reactivate it: uncomment the block, swap the names ``_simulate_fn`` and
# ``_simulate_fn_kinetic_params`` (or equivalent), and switch ``main()`` to
# call ``_build_kinetic_predictor`` instead of ``_build_direct_rate_predictors``.
#
# def _simulate_fn_kinetic_params(
#     predictors: tuple[BoundedPredictor],
#     ts: Float[Array, " T"],
#     covariates: dict[str, Array],
#     y0: Float[Array, " 6"],
#     solver: SolverConfig,
# ) -> Float[Array, "T 6"]:
#     """Integrate the method-of-moments ODE using a single kinetic-parameter predictor.
#
#     The single predictor is wrapped in a one-tuple ``(BP,)`` to keep
#     the ``predictors`` argument shape consistent with the multi-rate
#     case. The predictor maps
#     ``(temperature_C, loading) -> [logA, gamma, Ag, g]`` and the
#     vector field plugs the four scalars into a CNT (nucleation) and
#     power-law (growth) form, applying the ``exp(logA)`` /
#     ``10**Ag / 60`` scalings here.
#     """
#     (predictor,) = predictors
#     temperature_C = covariates["temperature_C"]
#     conc_sat = _conc_sat(temperature_C)
#     T_K = temperature_C + 273.15
#
#     params = predictor(covariates)  # [4] in physical units (bounded)
#     logA = params[0]
#     gamma_mJ_m2 = params[1]
#     Ag = params[2]
#     g_exp = params[3]
#     gamma_J_m2 = gamma_mJ_m2 * 1e-3  # [mJ/m^2] -> [J/m^2]
#
#     def vector_field(t: Array, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
#         mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
#         S = conc / conc_sat
#         meta_mask = (S > 1.0 + _META_EPS).astype(y.dtype)
#
#         # CNT nucleation J = exp(logA) * S * exp(-16π γ³ v² / (3 (k_B T)³ ln²S))
#         S_safe = jnp.clip(S, min=1.0 + 1e-12)
#         logS = jnp.log(S_safe)
#         cnt_exp = (
#             -16.0 * jnp.pi * gamma_J_m2**3 * _M_V**2
#             / (3.0 * (_K_B * T_K) ** 3 * logS**2)
#         )
#         J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exp)
#
#         # Power-law growth G = (10**Ag / 60) * max(S - 1, 0)**g
#         growth_drive = jnp.maximum(S - 1.0, 0.0)
#         G = meta_mask * jnp.power(10.0, Ag) / 60.0 * jnp.power(growth_drive, g_exp)
#
#         dmu0 = J
#         dmu1 = G * mu0
#         dmu2 = 2.0 * G * mu1
#         dmu3 = 3.0 * G * mu2
#         dmu4 = 4.0 * G * mu3
#         dconc = -3.0 * _K_V * _RHO_C * G * mu2
#         return jnp.stack([dmu0, dmu1, dmu2, dmu3, dmu4, dconc])
#
#     times_sec = ts * 60.0
#     term = diffrax.ODETerm(vector_field)
#     saveat = diffrax.SaveAt(ts=times_sec)
#     if solver.dt0 is None:
#         span = times_sec[-1] - times_sec[0]
#         dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
#     else:
#         dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)
#     atol = (
#         jnp.asarray(solver.atol, dtype=times_sec.dtype)
#         if isinstance(solver.atol, tuple)
#         else solver.atol
#     )
#     sol = diffrax.diffeqsolve(
#         term, solver.solver,
#         t0=times_sec[0], t1=times_sec[-1], dt0=dt0,
#         y0=y0, saveat=saveat,
#         stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=atol),
#         max_steps=solver.max_steps, adjoint=diffrax.DirectAdjoint(),
#     )
#     return jnp.asarray(sol.ys)


# --------------------------------------------------------------------------- #
# Predictors (1) — direct-rate tuple  (ACTIVE)                                #
# --------------------------------------------------------------------------- #


def _build_direct_rate_predictors(
    key: Array,
) -> tuple[BoundedPredictor, BoundedPredictor]:
    """Build the canonical predictors tuple ``(growth_BP, nucleation_BP)``.

    Each ``BoundedPredictor`` consumes a 3-key dict
    ``(temperature_C, loading, supersaturation)`` and emits one bounded
    log-rate. The vector field exponentiates with ``10**(.)`` to recover
    physical rates. Construction is symmetric across the two branches
    except for the output bound (growth vs nucleation log-magnitudes).

    Pipeline per branch::

        CovariateSelector  : dict -> [3] in declared INPUT_KEYS_DIRECT order
        in_scaler          : [3] physical -> [3] latent (logit-of-normalised)
        MLPPredictor       : [3] -> [1]   (tanh, depth=2, width=16)
        out_scaler         : [1] latent -> [1] physical (sigmoid into log-bounds)

    The two branches receive *independent* MLP weights via key splitting so
    the network architecture is identical but the initial parameters differ.
    """
    k_growth, k_nucleation = jr.split(key, 2)

    selector = CovariateSelector(keys=INPUT_KEYS_DIRECT)
    in_scaler = BoundScaler(
        bounds=(
            COVARIATE_BOUNDS["temperature_C"],
            COVARIATE_BOUNDS["loading"],
            SUPERSATURATION_BOUNDS,
        ),
        transform="sigmoid",
    )

    growth_inner = MLPPredictor(
        in_size=len(INPUT_KEYS_DIRECT),
        out_size=1,
        width_size=16,
        depth=2,
        activation_name="tanh",
        key=k_growth,
    )
    growth_out_scaler = BoundScaler(
        bounds=(LOG10_GROWTH_BOUNDS,),
        transform="sigmoid",
    )
    growth_bp = BoundedPredictor(
        selector=selector,
        in_scaler=in_scaler,
        inner=growth_inner,
        out_scaler=growth_out_scaler,
    )

    nucleation_inner = MLPPredictor(
        in_size=len(INPUT_KEYS_DIRECT),
        out_size=1,
        width_size=16,
        depth=2,
        activation_name="tanh",
        key=k_nucleation,
    )
    nucleation_out_scaler = BoundScaler(
        bounds=(LOG10_NUCLEATION_BOUNDS,),
        transform="sigmoid",
    )
    nucleation_bp = BoundedPredictor(
        selector=selector,
        in_scaler=in_scaler,
        inner=nucleation_inner,
        out_scaler=nucleation_out_scaler,
    )

    return (growth_bp, nucleation_bp)


# --------------------------------------------------------------------------- #
# Predictors (2) — kinetic-parameter single  (REFERENCE, COMMENTED-OUT)       #
# --------------------------------------------------------------------------- #
#
# The original kinetic-parameter builder is preserved verbatim below. To
# reactivate it: uncomment, and switch ``main()`` to call this builder and
# the matching ``_simulate_fn_kinetic_params``.
#
# def _build_kinetic_predictor(key: Array) -> tuple[BoundedPredictor]:
#     """Construct a single-element predictors tuple holding the kinetic-parameter MLP.
#
#     Pipeline ``(temperature_C, loading) -> [logA, gamma, Ag, g]``::
#
#         CovariateSelector  : dict -> [2] in declared INPUT_KEYS_KINETIC order
#         in_scaler          : [2] physical -> [2] latent (logit-of-normalised)
#         MLPPredictor       : [2] -> [4]   (tanh, depth=2, width=16)
#         out_scaler         : [4] latent -> [4] physical (sigmoid into bounds)
#
#     Output is in physical units; the matching
#     ``_simulate_fn_kinetic_params`` consumes it directly. Returned as
#     a one-tuple to match the framework's ``predictors`` convention,
#     so the same training entry-point handles single-rate and
#     multi-rate cases uniformly.
#     """
#     selector = CovariateSelector(keys=INPUT_KEYS_KINETIC)
#     in_scaler = BoundScaler(
#         bounds=tuple(COVARIATE_BOUNDS[k] for k in INPUT_KEYS_KINETIC),
#         transform="sigmoid",
#     )
#     inner = MLPPredictor(
#         in_size=len(INPUT_KEYS_KINETIC),
#         out_size=4,
#         width_size=16,
#         depth=2,
#         activation_name="tanh",
#         key=key,
#     )
#     out_scaler = BoundScaler(
#         bounds=(
#             KINETIC_PARAMETER_BOUNDS["logA"],
#             KINETIC_PARAMETER_BOUNDS["gamma"],
#             KINETIC_PARAMETER_BOUNDS["Ag"],
#             KINETIC_PARAMETER_BOUNDS["g"],
#         ),
#         transform="sigmoid",
#     )
#     bp = BoundedPredictor(
#         selector=selector,
#         in_scaler=in_scaler,
#         inner=inner,
#         out_scaler=out_scaler,
#     )
#     return (bp,)


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

    print("\n[build] solver + predictors (direct-rate path)")
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=(1e-5,) * 6,
        max_steps=500_000,
        dt0=None,
    )
    predictors = _build_direct_rate_predictors(k_init)
    growth_bp, nucleation_bp = predictors

    # Demo: evaluate both predictors at T, L of the first experiment with a
    # plausible mid-range supersaturation. This is the same input shape the
    # vector field constructs each timestep.
    sample_cov = experiments[0].covariates
    sample_inputs = {
        "temperature_C": jnp.asarray(sample_cov["temperature_C"]),
        "loading": jnp.asarray(sample_cov["loading"]),
        "supersaturation": jnp.asarray(1.5),
    }
    sample_log10_G = float(jnp.squeeze(growth_bp(sample_inputs)))
    sample_log10_J = float(jnp.squeeze(nucleation_bp(sample_inputs)))
    print(
        f"  predictors sample on T={float(sample_cov['temperature_C']):.1f}°C, "
        f"L={float(sample_cov['loading']):.2f}, S=1.5 -> "
        f"log10_G={sample_log10_G:.2f} (G={10.0**sample_log10_G:.2e} m/s), "
        f"log10_J={sample_log10_J:.2f} (J={10.0**sample_log10_J:.2e} #/(m^3·s))"
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
    history, trained_predictors = train_with_optax(
        predictors,
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

    trained_growth, trained_nucleation = trained_predictors
    final_log10_G = float(jnp.squeeze(trained_growth(sample_inputs)))
    final_log10_J = float(jnp.squeeze(trained_nucleation(sample_inputs)))
    print(
        f"  trained predictors on same input -> "
        f"log10_G={final_log10_G:.2f} (G={10.0**final_log10_G:.2e} m/s), "
        f"log10_J={final_log10_J:.2f} (J={10.0**final_log10_J:.2e} #/(m^3·s))"
    )


if __name__ == "__main__":
    main()
