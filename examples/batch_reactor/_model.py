"""Model definitions shared by the two batch reactor scripts.

``train_hybrid.py`` fits these predictors and ``train_rl_deactivation.py``
loads the result with ``load_predictors``. Loading needs a template whose
leaves have the same classes and static configuration as the saved tree, so
both scripts must build the predictors from this one definition. Data
generation stays in each script: they build different datasets (fresh and
aged catalyst) and share only the physics.

Not an importable package module. Both scripts run as ``__main__`` with
their own directory on ``sys.path``, which is what resolves ``from _model
import ...``.
"""

# ruff: noqa: F722

from __future__ import annotations

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

from jaxhybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    MLPPredictor,
    SolverConfig,
    ramp_profile,
)

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


def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 2"]:
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
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        # A latent near zero puts both parameters near their box midpoints.
        self.latent = jr.normal(key, (2,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
            transform="sigmoid",
        )

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
    predictors: tuple[ArrheniusKinetics, BoundedPredictor],
    temperature_C: ArrayLike,
    pH: ArrayLike,
) -> Array:
    """Hybrid rate in decades: ``log10 k_param(T) + delta_log10(T, pH)``."""
    parametric, residual = predictors
    delta = jnp.squeeze(residual({"temperature_C": temperature_C, "pH": pH}))
    return log10_k_trunk(parametric, temperature_C) + delta


def simulate_fn(
    predictors: tuple[ArrheniusKinetics, BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate ``A -> B`` first-order kinetics under the hybrid rate :func:`log10_k`.

    The temperature changes during each run. The covariates carry the
    parameters of a flat-ramp-flat heating profile (``ramp_t0``,
    ``ramp_t1``, ``T_lo``, ``T_hi``) and ``ramp_profile`` turns them into
    ``T(t)``, evaluated inside the vector field at the solver's ``t``. Profile
    parameters travel as ordinary scalar covariates; the profile callable is
    pure JAX, so it differentiates and vmaps like the rest of the model.
    """
    pH = covariates["pH"]
    T_profile = ramp_profile(
        t0=covariates["ramp_t0"],
        t1=covariates["ramp_t1"],
        v0=covariates["T_lo"],
        v1=covariates["T_hi"],
    )

    def vector_field(t: ArrayLike, y: Float[Array, " 2"], args: object) -> Array:
        # T(t) changes with t, so the residual is re-evaluated every step.
        k = jnp.power(10.0, log10_k(predictors, T_profile(t), pH))
        # Clipping guards rare negative solver excursions near the asymptote.
        rate = k * jnp.maximum(y[0], 0.0)
        return jnp.stack([-rate, rate])

    term = diffrax.ODETerm(vector_field)
    return jnp.asarray(solver.diffeqsolve(term, ts, y0).ys)


def build_predictors(*, key: Array) -> tuple[ArrheniusKinetics, BoundedPredictor]:
    """Build the ``(parametric_trunk, residual_bp)`` predictors tuple."""
    k_param, k_residual = jr.split(key, 2)
    parametric = ArrheniusKinetics(key=k_param)
    residual = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=BoundScaler(
            bounds=(TEMPERATURE_BOUNDS, PH_BOUNDS),
            transform="sigmoid",
        ),
        inner=MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=16,
            depth=1,
            activation_name="relu",
            key=k_residual,
        ),
        out_scaler=BoundScaler(
            bounds=(RES_LOG10_BOUNDS,),
            transform="sigmoid",
        ),
    )
    return (parametric, residual)
