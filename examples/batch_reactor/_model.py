"""Model definitions shared by the batch reactor scripts.

``train_hybrid.py`` and ``compare_baselines.py`` fit these predictors, and
``train_rl_deactivation.py`` loads the fitted ones with ``load_predictors``.
Loading needs a template whose leaves have the same classes and static
configuration as the saved tree, so every script builds the predictors from
this one definition. The hidden truth and data generation are in
``_data.py``.

Not an importable package module. The scripts run as ``__main__`` with
their own directory on ``sys.path``, which is what resolves ``from _model
import ...``.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

import jaxhybridmodels as hm

T_REF: float = 298.15  # K (25 °C); centring temperature for the Arrhenius form
R_GAS: float = 8.314e-3  # kJ/(mol·K); pair with Ea in kJ/mol

# Predictor input boxes, wider than the data span so the sigmoid input
# scaler has headroom outside the training box.
TEMPERATURE_BOUNDS: tuple[float, float] = (0.0, 50.0)
PH_BOUNDS: tuple[float, float] = (3.0, 9.0)

# Trunk output boxes. ``k`` is typically in [0.05, 7.4], so its log sits in
# [-3, 2]; Ea is 30 kJ/mol in truth, with headroom on both sides.
LOG_KREF_BOUNDS: tuple[float, float] = (-3.0, 2.0)
EA_BOUNDS: tuple[float, float] = (0.0, 80.0)

# Residual box, symmetric so a fresh MLP contributes no correction. Two
# decades is generous against the truth's one-decade pH modulation.
RES_LOG10_BOUNDS: tuple[float, float] = (-2.0, 2.0)

INPUT_KEYS: tuple[str, ...] = ("temperature_C", "pH")

# Every fit and prediction in these examples integrates with this solver:
# the default Tsit5 at rtol 1e-5, atol 1e-7, with a larger step budget and a
# fixed first step.
SOLVER = hm.SolverConfig(max_steps=10_000, dt0=0.05)


def y0_fn(covariates: dict[str, Array], channels: dict[str, hm.ChannelObs]) -> Float[Array, " 2"]:
    """Initial state ``[Ca, Cb] = [Ca0_observed, 0.0]``.

    Reads the first ``Ca`` observation rather than the constant ``CA0``, so a
    variant that varies ``Ca0`` per experiment still works.
    """
    ca0 = jnp.asarray(channels["Ca"].values[0])
    return jnp.stack([ca0, jnp.zeros_like(ca0)])


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Project ``[Ca, Cb]`` to the observed channel ``[Ca]``."""
    return state[..., :1]


class ArrheniusKinetics(eqx.Module):
    """Centred-Arrhenius trunk: ``k_param(T) = exp(log_k_ref - Ea/R * (1/T - 1/T_ref))``.

    Two trainable scalars stored as a latent vector and mapped into
    ``LOG_KREF_BOUNDS`` and ``EA_BOUNDS`` by a sigmoid ``BoundScaler``, so
    CMA-ES searches an unbounded space while the simulator only sees values
    inside the boxes. It takes no covariates; :func:`log10_k_trunk` combines
    the parameters with ``T``.

    A ``Parameter`` per scalar would also work. This class stays because the
    committed ``artefacts/trunk_fresh.eqx`` was saved with it, and because
    its latent initialisation around zero is what ``init="lhs_box"`` samples
    around.
    """

    latent: Float[Array, " 2"]
    out_scaler: hm.BoundScaler

    def __init__(self, *, key: Array) -> None:
        # A latent near zero puts both parameters near their box midpoints.
        self.latent = jr.normal(key, (2,)) * 0.1
        self.out_scaler = hm.BoundScaler((LOG_KREF_BOUNDS, EA_BOUNDS), transform="sigmoid")

    def __call__(self) -> Float[Array, " 2"]:
        """Return ``(log_k_ref, Ea)`` in physical units."""
        return self.out_scaler.from_latent(self.latent)


def log10_k_trunk(parametric: ArrheniusKinetics, temperature_C: ArrayLike) -> Array:
    """``log10 k_param(T)`` from the trunk alone, pH-blind."""
    log_k_ref, Ea = parametric()
    T_K = jnp.asarray(temperature_C) + 273.15
    # Arrhenius is base e; one division converts to decades.
    return (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)


def log10_k(
    predictors: tuple[ArrheniusKinetics, hm.BoundedPredictor],
    temperature_C: ArrayLike,
    pH: ArrayLike,
) -> Array:
    """Hybrid rate in decades: ``log10 k_param(T) + delta_log10(T, pH)``.

    Scalars inside the vector field; array inputs broadcast, so a plot can
    pass a whole pH sweep at one temperature.
    """
    parametric, residual = predictors
    delta = hm.evaluate_predictor(residual, {"temperature_C": temperature_C, "pH": pH})
    return log10_k_trunk(parametric, temperature_C) + delta


def _integrate_ramp(
    log10_k_at: Callable[[Array], Array],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: hm.SolverConfig,
) -> diffrax.Solution:
    """Integrate ``A -> B`` with rate ``10 ** log10_k_at(T(t))`` along the run's ramp.

    The covariates carry the four parameters of a flat-ramp-flat heating
    profile (``ramp_t0``, ``ramp_t1``, ``T_lo``, ``T_hi``), and
    ``ramp_profile`` turns them into ``T(t)``, evaluated inside the vector
    field at the solver's ``t``. The profile is pure JAX, so it
    differentiates and vmaps like the rest of the model.
    """
    T_profile = hm.ramp_profile(
        t0=covariates["ramp_t0"],
        t1=covariates["ramp_t1"],
        v0=covariates["T_lo"],
        v1=covariates["T_hi"],
    )

    def vector_field(t: ArrayLike, y: Float[Array, " 2"], args: object) -> Array:
        k = jnp.power(10.0, log10_k_at(T_profile(t)))
        # Clipping guards rare negative solver excursions near the asymptote.
        rate = k * jnp.maximum(y[0], 0.0)
        return jnp.stack([-rate, rate])

    # The framework reads the state off the solution's ``ys``.
    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0)


def simulate_fn(
    predictors: tuple[ArrheniusKinetics, hm.BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: hm.SolverConfig,
) -> diffrax.Solution:
    """The hybrid model: trunk plus residual, :func:`log10_k`.

    ``T(t)`` changes with ``t``, so the residual is re-evaluated every step.
    """
    pH = covariates["pH"]
    return _integrate_ramp(lambda T: log10_k(predictors, T, pH), ts, covariates, y0, solver)


def simulate_trunk_fn(
    predictors: tuple[ArrheniusKinetics, hm.BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: hm.SolverConfig,
) -> diffrax.Solution:
    """The classical baseline: the Arrhenius trunk alone, pH-blind.

    Takes the same predictor tuple as :func:`simulate_fn` and ignores the
    residual, so one ``freeze`` choice serves both models. A fresh residual
    is close to zero but not exactly zero; letting it into the rate would add
    a spurious temperature dependence and blur the fitted ``Ea``.
    """
    return _integrate_ramp(lambda T: log10_k_trunk(predictors[0], T), ts, covariates, y0, solver)


def build_predictors(*, key: Array) -> tuple[ArrheniusKinetics, hm.BoundedPredictor]:
    """Build the ``(parametric_trunk, residual_bp)`` predictors tuple.

    The residual's scalers come from the bounds with the default sigmoid
    squash, and its one output comes back as a scalar.
    """
    k_param, k_residual = jr.split(key, 2)
    parametric = ArrheniusKinetics(key=k_param)
    residual = hm.BoundedPredictor(
        input_keys=INPUT_KEYS,
        input_bounds=(TEMPERATURE_BOUNDS, PH_BOUNDS),
        inner=hm.MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=16,
            depth=1,
            activation_name="relu",
            key=k_residual,
        ),
        output_bounds=RES_LOG10_BOUNDS,
    )
    return (parametric, residual)
