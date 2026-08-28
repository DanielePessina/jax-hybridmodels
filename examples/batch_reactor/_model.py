"""Model definitions shared by the two batch reactor examples.

The state hooks, the parametric trunk, the vector field and the predictor
builder, which ``train_hybrid.py`` and ``train_rl_deactivation.py`` must
agree on exactly. ``load_predictors`` rebuilds a saved pytree against a
template and needs every leaf's static configuration to match, so a second
definition of ``ArrheniusKinetics`` in the RL script would deserialise as a
different type and fail.

Data generation is deliberately elsewhere: the two scripts build different
datasets, fresh catalyst and aged catalyst, and share only the physics.

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
from jaxtyping import Float

from hybridmodels import (
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
    """Centred-Arrhenius parametric trunk: ``k_param(T) = exp(log_k_ref - Ea/R · (1/T - 1/T_ref))``.

    The two trainable scalars live in latent space and pass through a
    sigmoid ``BoundScaler``, so the search is unbounded while the simulator
    only ever sees values inside ``(LOG_KREF_BOUNDS, EA_BOUNDS)``.

    Not a ``BoundedPredictor``: the trunk takes no covariates at all. It
    returns the parameters and the vector field combines them with ``T``.
    """

    latent: Float[Array, " 2"]
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        # Small Gaussian latent init starts the physical parameters near
        # each bound's midpoint; CMA-ES expands from there.
        self.latent = jr.normal(key, (2,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self) -> Float[Array, " 2"]:
        """Return ``(log_k_ref, Ea)`` in physical units."""
        return self.out_scaler.from_latent(self.latent)


def simulate_fn(
    predictors: tuple[ArrheniusKinetics, BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate ``A -> B`` first-order kinetics with a hybrid rate constant.

    The rate is decomposed log-additively::

        log10(k(T, pH)) = log10(k_param(T)) + Δlog10(T, pH)

    where ``k_param`` is the centred-Arrhenius trunk and ``Δlog10`` a
    bounded MLP residual.

    The temperature is **not** constant in time: each experiment's
    covariates carry the parameters of a flat-ramp-flat heating profile
    (``ramp_t0``, ``ramp_t1``, ``T_lo``, ``T_hi``), and the profile is
    evaluated inside the vector field at the solver's continuous ``t``.
    That is the ``hybridmodels.profiles`` pattern: profile parameters
    ride as ordinary scalar covariates, the pure-JAX profile callable
    produces the time-varying value, and the predictor's input dict is
    mixed at every step (``T(t)`` overrides the ``temperature_C`` key).
    """
    parametric, residual = predictors
    log_k_ref, Ea = parametric()  # [2] in physical units (sigmoid-bounded)

    pH = covariates["pH"]
    T_profile = ramp_profile(
        t0=covariates["ramp_t0"],
        t1=covariates["ramp_t1"],
        v0=covariates["T_lo"],
        v1=covariates["T_hi"],
    )

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        T_C = T_profile(t)
        T_K = T_C + 273.15

        # log10 conversion: Arrhenius is naturally base-e; one /ln(10) at
        # the end.
        log10_k_param = (
            log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)
        ) / jnp.log(10.0)

        # The input dict is rebuilt at every solver step: T(t) changes
        # with t, so the residual must be re-read at the current value.
        delta_log10_k = jnp.squeeze(residual({"temperature_C": T_C, "pH": pH}))
        k = jnp.power(10.0, log10_k_param + delta_log10_k)

        # Guards the integrator's rare negative excursions near the
        # asymptote. Mass conservation is exact analytically.
        Ca = jnp.maximum(y[0], 0.0)
        rate = k * Ca
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
