"""Model definitions shared by the two batch reactor examples.

Owns the parts that ``train_hybrid.py`` and ``train_rl_deactivation.py`` must
agree on exactly: the state hooks, the parametric trunk class, the vector
field, and the predictor builder.

The sharing is not a convenience. ``load_predictors`` rebuilds a saved pytree
against a template and needs every leaf's static configuration to match, so a
second definition of ``ArrheniusKinetics`` in the RL script would produce a
different type and fail to deserialise. One class, one module.

Data generation is deliberately *not* here. The two scripts generate different
datasets (fresh catalyst and aged catalyst) and share only the physics.

Not an importable package module. Both scripts run as ``__main__`` with their
own directory on ``sys.path``, which is what makes the bare ``from _model
import ...`` resolve.
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
)

# --------------------------------------------------------------------------- #
# Physics constants                                                           #
# --------------------------------------------------------------------------- #

T_REF: float = 298.15  # K (25 °C); centring temperature for the Arrhenius form
R_GAS: float = 8.314e-3  # kJ/(mol·K); pair with Ea in kJ/mol

# --------------------------------------------------------------------------- #
# Predictor bounds                                                            #
# --------------------------------------------------------------------------- #

# Predictor input bounds (slightly wider than data span; matches train_kinetic.py
# convention of giving the sigmoid input scaler headroom outside the training box)
TEMPERATURE_BOUNDS: tuple[float, float] = (0.0, 50.0)
PH_BOUNDS: tuple[float, float] = (3.0, 9.0)

# Parametric trunk bounds — centred Arrhenius
# log_k_ref = ln(k(T_REF, _)); k typically in [0.05, 7.4] ⇒ ln in [-3, 2]
LOG_KREF_BOUNDS: tuple[float, float] = (-3.0, 2.0)
# Ea in kJ/mol; truth at 30, bounds give ample headroom on both sides
EA_BOUNDS: tuple[float, float] = (0.0, 80.0)

# Residual MLP output bounds — symmetric around 0 so a fresh-init MLP contributes
# 0 decades of correction. ±2 decades is generous for the truth's pH-modulation
# range (k_sat spans ~1 decade across pH 4-8).
RES_LOG10_BOUNDS: tuple[float, float] = (-2.0, 2.0)

INPUT_KEYS: tuple[str, ...] = ("temperature_C", "pH")


# --------------------------------------------------------------------------- #
# Hooks: y0_fn, state_to_output, simulate_fn                                  #
# --------------------------------------------------------------------------- #


def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 2"]:
    """Initial state ``[Ca, Cb] = [Ca0_observed, 0.0]``.

    Reads the first ``Ca`` observation rather than the constant ``CA0`` so this
    helper stays correct if a future variant varies ``Ca0`` per experiment.
    """
    ca0 = jnp.asarray(channels["Ca"].values[0])
    return jnp.stack([ca0, jnp.zeros_like(ca0)])


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Project ``[Ca, Cb]`` to the observed channel ``[Ca]``."""
    return state[..., :1]


# --------------------------------------------------------------------------- #
# Parametric trunk: two scalars (log_k_ref, Ea) with sigmoid-bounded latent.  #
# Modelled on ``train_crystallisation_mechanistic.py::KineticParameters``.    #
# --------------------------------------------------------------------------- #


class ArrheniusKinetics(eqx.Module):
    """Centred-Arrhenius parametric trunk: ``k_param(T) = exp(log_k_ref - Ea/R · (1/T - 1/T_ref))``.

    The two trainable scalars live in latent space and pass through a sigmoid
    :class:`BoundScaler` so the simulator always sees physical-units values
    inside ``(LOG_KREF_BOUNDS, EA_BOUNDS)``. CMA-ES sees a 2-D unbounded
    search; the ``out_scaler`` keeps every candidate inside the box.

    Not a ``BoundedPredictor`` — the trunk has no covariate inputs at all
    (it returns the parameters; the vector field combines them with ``T``).
    """

    latent: Float[Array, " 2"]
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        # Small Gaussian init in latent space puts the physical parameters
        # near each bound's midpoint at gen 0; CMA-ES expands from there.
        self.latent = jr.normal(key, (2,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self) -> Float[Array, " 2"]:
        """Return ``(log_k_ref, Ea)`` in physical units."""
        return self.out_scaler.from_latent(self.latent)


# --------------------------------------------------------------------------- #
# Simulate function — the user-owned physics                                  #
# --------------------------------------------------------------------------- #


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

    where ``k_param`` is the centred-Arrhenius parametric trunk and
    ``Δlog10`` is a bounded MLP residual. ``k`` is constant across the
    integration (covariates are constant in time per R-D5), so we evaluate
    it once at the top of the call and close over the scalar inside the
    vector field.
    """
    parametric, residual = predictors
    log_k_ref, Ea = parametric()  # [2] in physical units (sigmoid-bounded)

    T_C = covariates["temperature_C"]
    pH = covariates["pH"]
    T_K = T_C + 273.15

    # log10 conversion: Arrhenius is naturally base-e; one /ln(10) at the end.
    log10_k_param = (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)

    inputs = {"temperature_C": T_C, "pH": pH}
    delta_log10_k = jnp.squeeze(residual(inputs))

    log10_k = log10_k_param + delta_log10_k
    k = jnp.power(10.0, log10_k)

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        # Clipping ``Ca`` at zero guards against rare negative excursions of
        # the integrator near the asymptote; mass conservation is exact
        # analytically, so this is purely numerical hygiene.
        Ca = jnp.maximum(y[0], 0.0)
        rate = k * Ca
        return jnp.stack([-rate, rate])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=solver.dt0 if solver.dt0 is not None else 0.05,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


# --------------------------------------------------------------------------- #
# Predictor builder                                                           #
# --------------------------------------------------------------------------- #


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
