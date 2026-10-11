"""Hidden truth and synthetic experiments for the batch reactor scripts.

``train_hybrid.py`` and ``compare_baselines.py`` build fresh-catalyst runs
from this module, and ``train_rl_deactivation.py`` reuses the same truth for
its aged-catalyst runs. Nothing here is visible to the models: it generates
data and draws truth overlays.

Every run heats along a flat-ramp-flat profile around a midpoint
temperature, at a fixed pH, starting from ``Ca = CA0``. Like ``_model``,
this module is imported by scripts that run with this directory on
``sys.path``.
"""

# ruff: noqa: F722

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _model import R_GAS, T_REF, y0_fn
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float
from scipy.stats import qmc

import jaxhybridmodels as hm

EA_TRUE: float = 30.0  # kJ/mol; the rate doubles about every 10 °C near T_REF

# The hidden pH dependence, a Hill-type saturation curve: fast below pH 5,
# a plateau above pH 7, and a knee at K_SAT_PH50.
K_SAT_BASELINE: float = 0.14
K_SAT_AMPLITUDE: float = 1.05
K_SAT_PH50: float = 5.85
K_SAT_HILL: float = 5.0

# Design space. Ramp midpoints are sampled in T_C_RANGE.
T_C_RANGE: tuple[float, float] = (15.0, 35.0)
PH_RANGE: tuple[float, float] = (4.5, 7.5)
N_TRAIN_EXPERIMENTS: int = 9
VALIDATION_POINTS: tuple[tuple[float, float], ...] = ((20.0, 5.3), (30.0, 6.8))

# Heating ramp: the jacket goes from T_C - 3 to T_C + 3 between RAMP_T0 and
# RAMP_T1, flat before and after, so each run sweeps a band of temperatures.
RAMP_HALF_WIDTH: float = 3.0
RAMP_T0: float = 0.5
RAMP_T1: float = 4.5

# Observations: 12 evenly spaced samples of Ca on [0, T_MAX], with noise
# sigma = NOISE_REL * max(|Ca|, NOISE_FLOOR) so near-zero Ca stays noisy.
T_MAX: float = 5.0
N_TIMESTEPS: int = 12
CA0: float = 1.0
NOISE_REL: float = 0.03
NOISE_FLOOR: float = 0.02
OUTPUT_CHANNELS: tuple[str, ...] = ("Ca",)


def k_true(temperature_C: ArrayLike, pH: ArrayLike) -> Array:
    """Fresh-catalyst rate constant: pH saturation times Arrhenius centred at ``T_REF``."""
    T_K = jnp.asarray(temperature_C) + 273.15
    arrhenius = jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))
    pH_arr = jnp.asarray(pH)
    k_sat = K_SAT_BASELINE + K_SAT_AMPLITUDE / (
        1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
    )
    return k_sat * arrhenius


def ramp_covariates(T_C: float) -> dict[str, float]:
    """The four ramp parameters for a run centred on ``T_C``."""
    return {
        "ramp_t0": RAMP_T0,
        "ramp_t1": RAMP_T1,
        "T_lo": T_C - RAMP_HALF_WIDTH,
        "T_hi": T_C + RAMP_HALF_WIDTH,
    }


def true_ca(ts: Float[Array, " T"], *, T_C: float, pH: float) -> Float[Array, " T"]:
    """True ``Ca(t)`` under the ramp centred on ``T_C``.

    No closed form once ``T`` varies, so it is integrated at tolerances well
    below the training solver's. ``T(t)`` is the same ``ramp_profile`` the
    model evaluates, so model and truth differ only in the rate law.
    """
    ramp = ramp_covariates(T_C)
    T_profile = hm.ramp_profile(
        t0=ramp["ramp_t0"], t1=ramp["ramp_t1"], v0=ramp["T_lo"], v1=ramp["T_hi"]
    )

    def field(t: ArrayLike, y: Array, args: object) -> Array:
        return -k_true(T_profile(t), pH) * y

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=0.01,
        y0=jnp.asarray(CA0, dtype=jnp.float32),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=10_000,
    )
    return jnp.asarray(sol.ys)


def lhs_design(seed: int, n: int = N_TRAIN_EXPERIMENTS) -> list[tuple[float, float]]:
    """Latin hypercube samples of ``(T_C, pH)`` over the design space."""
    unit = qmc.LatinHypercube(d=2, seed=seed).random(n=n)
    lo = np.array([T_C_RANGE[0], PH_RANGE[0]])
    hi = np.array([T_C_RANGE[1], PH_RANGE[1]])
    return [(float(t), float(ph)) for t, ph in lo + (hi - lo) * unit]


def make_ramp_experiment(*, T_C: float, pH: float, noise_key: Array, exp_id: str) -> hm.Experiment:
    """One noisy run ramping through ``T_C ± RAMP_HALF_WIDTH`` at fixed pH.

    Noise is clipped at zero, and ``sigma**2`` is stored as the channel
    variance.
    """
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    clean = true_ca(ts, T_C=T_C, pH=pH)
    sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
    noisy = jnp.clip(clean + sigma * jr.normal(noise_key, clean.shape), 0.0, None)
    return hm.make_experiment(
        covariates={**ramp_covariates(T_C), "pH": pH},
        channels={"Ca": hm.ChannelObs(ts=ts, values=noisy, variance=sigma**2)},
        y0_fn=y0_fn,
        exp_id=exp_id,
    )


def ramp_midpoint(experiment: hm.Experiment) -> float:
    """The midpoint temperature of an experiment's ramp."""
    return (float(experiment.covariates["T_lo"]) + float(experiment.covariates["T_hi"])) / 2.0
