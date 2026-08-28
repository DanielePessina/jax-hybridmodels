"""Batch reactor, part two: a bounded parameter controller trained by PPO.

Reads the frozen trunk that ``train_hybrid.py --save-predictors`` wrote,
then learns the one thing that trunk cannot represent: a catalyst whose
activity decays as the batch runs.

Mowbray et al. (Biotechnol Bioeng 120:154, 2023) reframe kinetic parameter
estimation as control. Parameters become actions, a policy maps the current
model state to those actions, and reward is the negative fit error at the
next measurement. Carrying that onto the batch reactor shows two things.

**Bounds by reparameterisation, inside an RL actor.** Continuous-control RL
normally bounds actions with a ``tanh`` squash and corrects the
log-probability by its log-det-Jacobian. Here the policy distribution lives
in the *latent* space instead: the network emits a Gaussian mean over ``z``
and ``BoundScaler.from_latent`` carries it into the physical box inside the
rollout. The action is ``z``, so the log-probability is a plain diagonal
Gaussian with no correction term, the bound is structural rather than
enforced, and ``BoundScaler.saturation`` becomes a reward term keeping the
policy off the dead flat region of the squash. The pieces recombine into an
ordinary ``BoundedPredictor`` at the end, which is why the trained policy
drops straight into ``predict_dataset``.

**No gradient through the solve.** PPO differentiates only the policy's
log-probability and the value head. The rollout is never differentiated, so
``SolverConfig.adjoint`` is irrelevant here and the solver could be stiff or
nonsmooth without consequence. That is the honest reason to reach for RL,
and the same argument the evosax trainer makes by another route. A
gradient-trained version of the identical model runs as a baseline, and on a
problem this smooth it is expected to be competitive: the conclusion is
about the adjoint, not about accuracy.

PPO rather than the paper's SAC, because the MDP is deterministic, the
horizon is 11 steps and rollouts are nearly free, so off-policy replay buys
little against roughly twice the code.

Run::

    uv run python examples/batch_reactor/train_hybrid.py --no-plot \\
        --save-predictors examples/batch_reactor/artefacts/trunk_fresh.eqx
    uv run python examples/batch_reactor/train_rl_deactivation.py
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import rlax
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float
from scipy.stats import qmc

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
    MLPPredictor,
    SolverConfig,
    compute_metrics,
    frozen_default_mask,
    load_predictors,
    make_dataset,
    make_experiment,
    predict_dataset,
    print_metrics,
)
from hybridmodels.training import OptaxTrainingConfig, train_with_optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ``_model`` sits in this script's own directory and owns everything the two
# batch reactor examples must agree on exactly. A second definition of
# ArrheniusKinetics here would fail to deserialise against the saved trunk.
from _model import (  # noqa: E402
    PH_BOUNDS,
    R_GAS,
    T_REF,
    TEMPERATURE_BOUNDS,
    build_predictors,
)
from _shared import (  # noqa: E402
    apply_default_style,
    parity_plot,
)

# Truth, duplicated from train_hybrid.py rather than shared: the two scripts
# generate different datasets and agree only on the physics in _model.py.
EA_TRUE: float = 30.0  # kJ/mol
K_SAT_BASELINE: float = 0.14
K_SAT_AMPLITUDE: float = 1.05
K_SAT_PH50: float = 5.85
K_SAT_HILL: float = 5.0

# Deactivation truth. Sigmoidal in time with a pH-dependent onset, so acid
# ages the catalyst sooner. The cubic exponent gives a flat plateau then a
# sharp fall, which no single decay scalar can represent, which is what makes
# a time-varying policy necessary rather than decorative.
TAU_REF: float = 1.2  # time units, against a batch length of 5
PH_REF: float = 6.0
DEACT_HILL: float = 3.0

# DOE, identical to train_hybrid.py so activity is the only difference between
# the fresh and aged datasets.
T_C_RANGE: tuple[float, float] = (15.0, 35.0)
PH_RANGE: tuple[float, float] = (4.5, 7.5)
N_TRAIN_EXPERIMENTS: int = 9
VALIDATION_POINTS: tuple[tuple[float, float], ...] = ((20.0, 5.3), (30.0, 6.8))

# Per-experiment observations, identical to train_hybrid.py.
T_MAX: float = 5.0
N_TIMESTEPS: int = 12
CA0: float = 1.0
NOISE_REL: float = 0.03
NOISE_FLOOR: float = 0.02

# Horizon: 12 observation times give 11 zero-order-hold intervals.
HORIZON: int = N_TIMESTEPS - 1

OUTPUT_CHANNELS: tuple[str, ...] = ("Ca",)

# Policy observation box. Ca is a concentration in [0, 1] by construction; the
# temperature and pH boxes match the residual predictor in _model.py.
CA_BOUNDS: tuple[float, float] = (0.0, 1.0)
OBS_BOUNDS: tuple[tuple[float, float], ...] = (CA_BOUNDS, TEMPERATURE_BOUNDS, PH_BOUNDS)
OBS_KEYS: tuple[str, ...] = ("Ca", "temperature_C", "pH")

# The upper edge is 1.05, not 1.0: a fresh catalyst has activity exactly 1,
# and a sigmoid only approaches its edge asymptotically. Under a hard ceiling
# the policy would have to saturate to represent t=0 and the saturation
# reward term would fight the fit.
ACTIVITY_BOUNDS: tuple[tuple[float, float], ...] = ((0.0, 1.05),)

# Deactivation-rate bound for the exponential baseline. Log-warped because a
# decay constant is a rate: the interesting range spans decades.
KD_BOUNDS: tuple[tuple[float, float], ...] = ((1e-3, 3.0),)

# Policy and critic architecture. tanh rather than relu on the inner MLP: a
# piecewise-linear activity curve reads badly against a smooth truth.
POLICY_WIDTH: int = 32
POLICY_DEPTH: int = 2

# PPO hyperparameters.
GAE_LAMBDA: float = 0.95
CLIP_EPS: float = 0.2
VF_COEF: float = 0.5
ENT_COEF: float = 1e-3
INIT_LOG_STD: float = -1.0
N_PPO_EPOCHS: int = 4
N_MINIBATCHES: int = 4
MAX_GRAD_NORM: float = 0.5

# Per-dimension constants in a Gaussian's differential entropy and density,
# pulled out so those expressions stay one line each.
_HALF_LOG_2PI_E: float = 0.5 * math.log(2.0 * math.pi * math.e)
_HALF_LOG_2PI: float = 0.5 * math.log(2.0 * math.pi)


# Truth helpers. Data generation and plot overlays only, never the model path.


def _k_sat_from_ph(pH: ArrayLike) -> Array:
    """Hidden truth for the pH dependence of the fresh-catalyst rate."""
    pH_arr = jnp.asarray(pH)
    return K_SAT_BASELINE + K_SAT_AMPLITUDE / (
        1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
    )


def _k_true(temperature_C: ArrayLike, pH: ArrayLike) -> Array:
    """Fresh-catalyst rate constant, unchanged from ``train_hybrid.py``."""
    T_K = jnp.asarray(temperature_C) + 273.15
    arrhenius = jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))
    return _k_sat_from_ph(pH) * arrhenius


def _tau_true(pH: ArrayLike) -> Array:
    """Deactivation onset time. Shorter at low pH, so acid ages the catalyst sooner."""
    return TAU_REF * (jnp.asarray(pH) / PH_REF) ** 2


def _activity_true(t: ArrayLike, pH: ArrayLike) -> Array:
    """Hidden truth for catalyst activity: ``1 / (1 + (t/tau(pH))^3)``.

    Exactly 1.0 at ``t = 0``, which is what the fresh-catalyst calibration in
    ``train_hybrid.py`` measured, and what makes the activity bound in
    ``ACTIVITY_BOUNDS`` need headroom above 1.
    """
    return 1.0 / (1.0 + (jnp.asarray(t) / _tau_true(pH)) ** DEACT_HILL)


def _activity_interval_mean(t0: ArrayLike, t1: ArrayLike, pH: ArrayLike, n: int = 257) -> Array:
    """Mean activity over ``[t0, t1]``, which is the *exact* zero-order-hold target.

    For ``dCa/dt = -k a(t) Ca`` the solution over one interval is
    ``Ca(t1) = Ca(t0) exp(-k * integral of a)``. Only the integral of ``a``
    enters, so holding activity at its interval mean reproduces the continuous
    truth exactly, while holding it at the left endpoint does not.

    That matters twice: it is the reference the pre-learning checkpoint
    scores against, and it is what the policy is recovering. The step
    function it emits is not sampling ``a_true(t_i)``, it is the
    piecewise-constant activity that best represents each interval.
    """
    grid = jnp.linspace(t0, t1, n)
    return jnp.trapezoid(_activity_true(grid, pH), grid) / (t1 - t0)


def _aged_ca_trajectory(
    ts: Float[Array, " T"], temperature_C: float, pH: float
) -> Float[Array, " T"]:
    """Dense truth for an aged run, integrated rather than solved in closed form.

    ``dCa/dt = -a_true(t, pH) * k_true(T, pH) * Ca`` has no tidy antiderivative
    for a cubic-Hill activity, so this integrates it tightly. The tolerances are
    two orders below the training solver's, so the data is not limited by the
    integrator that later fits it.
    """
    k = float(_k_true(temperature_C, pH))

    def vector_field(t: Array, y: Array, args: object) -> Array:
        rate = k * _activity_true(t, pH) * jnp.maximum(y[0], 0.0)
        return jnp.stack([-rate, rate])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=0.01,
        y0=jnp.array([CA0, 0.0]),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-11),
        max_steps=100_000,
    )
    return jnp.asarray(sol.ys)[:, 0]


def _add_heteroscedastic_noise(
    values: Array, *, key: Array, rel: float = NOISE_REL, floor: float = NOISE_FLOOR
) -> Array:
    """``sigma = rel * max(|values|, floor)``, clipped at zero. As in the fresh runs."""
    scale = rel * jnp.maximum(jnp.abs(values), floor)
    noisy = values + scale * jr.normal(key, values.shape)
    return jnp.clip(noisy, 0.0, None)


def _lhs_design(seed: int, n: int = N_TRAIN_EXPERIMENTS) -> list[tuple[float, float]]:
    """The same Latin hypercube as ``train_hybrid.py``, so the designs coincide."""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n)
    lo = np.array([T_C_RANGE[0], PH_RANGE[0]])
    hi = np.array([T_C_RANGE[1], PH_RANGE[1]])
    scaled = lo + (hi - lo) * unit
    return [(float(t), float(ph)) for t, ph in scaled]


def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 2"]:
    """Initial state ``[Ca, Cb] = [Ca0_observed, 0]``. Same rule as the fresh runs."""
    ca0 = jnp.asarray(channels["Ca"].values[0])
    return jnp.stack([ca0, jnp.zeros_like(ca0)])


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Project ``[Ca, Cb]`` to the observed channel ``[Ca]``."""
    return state[..., :1]


def _make_aged_experiment(
    *, temperature_C: float, pH: float, noise_key: Array, exp_id: str
) -> Experiment:
    """One aged run at fixed ``(T, pH)``, integrated with the decaying activity."""
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    clean = _aged_ca_trajectory(ts, temperature_C=temperature_C, pH=pH)
    noisy = _add_heteroscedastic_noise(clean, key=noise_key)
    sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
    channels = {"Ca": ChannelObs(ts=ts, values=noisy, variance=sigma**2)}
    return make_experiment(
        covariates={"temperature_C": float(temperature_C), "pH": float(pH)},
        channels=channels,
        y0_fn=y0_fn,
        exp_id=exp_id,
    )


def _build_aged_datasets(
    *, doe_seed: int, noise_key: Array
) -> tuple[list[Experiment], list[Experiment]]:
    """The 9 LHS aged runs plus the 2 off-grid aged validation runs.

    The noise keys are folded from a different root than ``train_hybrid.py``
    uses, so the fresh and aged datasets are independent noise realisations at
    the same ``(T, pH)`` points rather than correlated ones.
    """
    design = _lhs_design(seed=doe_seed)
    train = [
        _make_aged_experiment(
            temperature_C=T_C,
            pH=pH,
            noise_key=jr.fold_in(noise_key, i),
            exp_id=f"aged_train_{i:02d}_T{T_C:.1f}_pH{pH:.2f}",
        )
        for i, (T_C, pH) in enumerate(design)
    ]
    val = [
        _make_aged_experiment(
            temperature_C=T_C,
            pH=pH,
            noise_key=jr.fold_in(noise_key, 1000 + j),
            exp_id=f"aged_val_{j:02d}_T{T_C:.1f}_pH{pH:.2f}",
        )
        for j, (T_C, pH) in enumerate(VALIDATION_POINTS)
    ]
    return train, val


def load_trunk(path: Path, *, key: Array) -> tuple[Any, BoundedPredictor]:
    """Restore the ``(ArrheniusKinetics, BoundedPredictor)`` tuple from ``path``.

    ``load_predictors`` needs a template whose per-leaf static configuration
    matches the saved tree, which ``build_predictors`` supplies. The key only
    seeds the template's array leaves; every one of them is overwritten by the
    file's values.
    """
    if not path.exists():
        raise SystemExit(
            f"No trunk artefact at {path}. Produce one first with:\n"
            f"  uv run python examples/batch_reactor/train_hybrid.py --no-plot "
            f"--save-predictors {path}"
        )
    return load_predictors(path, build_predictors(key=key))


def trunk_rate(trunk: tuple[Any, BoundedPredictor], temperature_C: Array, pH: Array) -> Array:
    """Fresh-catalyst rate constant predicted by the frozen hybrid trunk.

    ``log10 k = log10 k_param(T) + delta_log10(T, pH)``, the decomposition
    ``train_hybrid.py`` fitted. A function of the covariates only, so every
    caller evaluates it once per experiment and hoists it above the rollout.
    """
    parametric, residual = trunk
    log_k_ref, Ea = parametric()
    T_K = temperature_C + 273.15
    log10_k_param = (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)
    delta = jnp.squeeze(residual({"temperature_C": temperature_C, "pH": pH}))
    return jnp.power(10.0, log10_k_param + delta)


def _integrate_interval(
    k_eff: Array, t0: Array, t1: Array, y: Float[Array, " 2"], solver: SolverConfig
) -> Float[Array, " 2"]:
    """Integrate one interval at a fixed effective rate, returning the end state.

    This is the MDP's transition ``x_{t+1} = f(x_t, p_t)``. The rate is constant
    across the interval by construction (zero-order hold), so the vector field
    closes over a scalar.
    """

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        # Clipping at zero is numerical hygiene near the asymptote; mass
        # conservation is exact analytically.
        rate = k_eff * jnp.maximum(y[0], 0.0)
        return jnp.stack([-rate, rate])

    # Solve over the single interval and keep the endpoint state.
    sol = solver.diffeqsolve(diffrax.ODETerm(vector_field), jnp.stack([t0, t1]), y)
    return jnp.asarray(sol.ys)[-1]


def zoh_states(
    activity_fn: Any,
    ts: Float[Array, " T"],
    k_fresh: Array,
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Roll the reactor forward holding activity constant across each interval.

    ``activity_fn(t, Ca) -> a`` is evaluated once at the start of each interval,
    on the model's own state rather than the measurement, which is the paper's
    free-running formulation (its Eq 4c and 4e).

    Every model reported on this page goes through this function, so the
    zero-order hold is not a handicap applied only to the RL policy. The truth
    is continuous, so the hold is an approximation in all four cases equally.
    """

    def step(y: Array, i: Array) -> tuple[Array, Array]:
        t0 = ts[i]
        a = activity_fn(t0, y[0])
        y_next = _integrate_interval(a * k_fresh, t0, ts[i + 1], y, solver)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, jnp.arange(ts.shape[0] - 1))
    return jnp.concatenate([y0[None, :], ys], axis=0)


class ExponentialDecay(eqx.Module):
    """Baseline 2: one trainable scalar, ``a(t) = exp(-k_d t)``.

    The strongest fixed-form competitor to a learned policy. ``k_d`` is a rate,
    so its bound is log-warped: the interesting range spans decades and a linear
    sigmoid would spend most of its resolution in the wrong place.
    """

    latent: Float[Array, " 1"]
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        self.latent = jr.normal(key, (1,)) * 0.1
        self.out_scaler = BoundScaler(bounds=KD_BOUNDS, transform="sigmoid", warp="log10")

    def rate(self) -> Array:
        """Return ``k_d`` in physical units."""
        return self.out_scaler.from_latent(self.latent)[0]


def build_policy(*, key: Array) -> BoundedPredictor:
    """The activity policy: ``(Ca, T, pH) -> a`` in ``ACTIVITY_BOUNDS``.

    An ordinary ``BoundedPredictor``. PPO trains it by splitting it at
    ``out_scaler`` (see :func:`rollout_batch`); the gradient baseline and the
    final evaluation call it whole. The class knows neither.

    The covariates are in the observation because without them the problem is
    not identifiable: only ``Ca`` is measured and ``Ca0`` is fixed, so a
    policy on conversion alone would return the same action for a hot acidic
    batch and a cool neutral one.
    """
    return BoundedPredictor(
        input_keys=OBS_KEYS,
        in_scaler=BoundScaler(bounds=OBS_BOUNDS, transform="sigmoid"),
        inner=MLPPredictor(
            in_size=len(OBS_KEYS),
            out_size=1,
            width_size=POLICY_WIDTH,
            depth=POLICY_DEPTH,
            activation_name="tanh",
            key=key,
        ),
        out_scaler=BoundScaler(bounds=ACTIVITY_BOUNDS, transform="sigmoid"),
    )


def make_simulate_fn(trunk: tuple[Any, BoundedPredictor], *, kind: str) -> Any:
    """Build a framework-shaped ``simulate_fn`` for one of the activity models.

    ``kind`` selects how activity is produced:

    ``"frozen"``
        Activity pinned at 1. The static hybrid model, evaluated on aged runs.
        ``predictors`` is ignored.
    ``"exponential"``
        ``predictors`` is an :class:`ExponentialDecay`.
    ``"policy"``
        ``predictors`` is a :class:`BoundedPredictor` over ``OBS_KEYS``.

    The trunk is closed over rather than passed through ``predictors``, so it
    can never reach a gradient transformation. Freezing it is structural, not
    a matter of getting a mask right.
    """

    def simulate_fn(
        predictors: Any,
        ts: Float[Array, " T"],
        covariates: dict[str, Array],
        y0: Float[Array, " 2"],
        solver: SolverConfig,
    ) -> Float[Array, "T 2"]:
        T_C = covariates["temperature_C"]
        pH = covariates["pH"]
        k_fresh = trunk_rate(trunk, T_C, pH)

        if kind == "frozen":

            def activity_fn(t: Array, ca: Array) -> Array:
                return jnp.asarray(1.0)

        elif kind == "exponential":

            def activity_fn(t: Array, ca: Array) -> Array:
                return jnp.exp(-predictors.rate() * t)

        elif kind == "policy":

            def activity_fn(t: Array, ca: Array) -> Array:
                return predictors({"Ca": ca, "temperature_C": T_C, "pH": pH})[0]

        else:
            raise ValueError(f"Unknown activity kind {kind!r}.")

        return zoh_states(activity_fn, ts, k_fresh, y0, solver)

    return simulate_fn


class EpisodeData(NamedTuple):
    """Everything one episode needs, with a leading axis to vmap over.

    One entry per experiment. Because the trunk depends only on covariates,
    ``k_fresh`` is precomputed here and the rollout never touches the trunk.
    """

    ts: Float[Array, "N T"]
    ca_obs: Float[Array, "N T"]
    variance: Float[Array, "N T"]
    k_fresh: Float[Array, " N"]
    temperature_C: Float[Array, " N"]
    pH: Float[Array, " N"]
    y0: Float[Array, "N 2"]


class Rollout(NamedTuple):
    """One batch of episodes, flattened to transitions where PPO wants them."""

    obs: Float[Array, "B Tp1 3"]
    latent: Float[Array, "B T 1"]
    log_prob: Float[Array, "B T"]
    reward: Float[Array, "B T"]
    value: Float[Array, "B Tp1"]
    activity: Float[Array, "B T"]


class Agent(NamedTuple):
    """The trainable trio. A plain pytree, so ``eqx.partition`` walks it."""

    policy: BoundedPredictor
    log_std: Float[Array, " 1"]
    critic: MLPPredictor


def build_agent(*, key: Array) -> Agent:
    """Policy, exploration spread, and value head."""
    k_policy, k_critic = jr.split(key, 2)
    return Agent(
        policy=build_policy(key=k_policy),
        log_std=jnp.full((1,), INIT_LOG_STD),
        critic=MLPPredictor(
            in_size=len(OBS_KEYS),
            out_size=1,
            width_size=POLICY_WIDTH,
            depth=POLICY_DEPTH,
            activation_name="tanh",
            key=k_critic,
        ),
    )


def agent_trainable(agent: Agent) -> Any:
    """Trainability mask: everything except the scalers' internal arrays.

    ``BoundScaler`` carries ``temperature`` as an array leaf. It is
    configuration, not a parameter, and freezing it is the same
    ``frozen_default_mask`` idiom that ``train_hybrid.py`` uses.
    """
    return frozen_default_mask(agent, BoundScaler)


def _parity_diagnostics(predictions, dataset):
    """Masked obs/pred pairs per channel for ``parity_plot``.

    ``compute_metrics`` keeps only the summary stats; the scatter needs the
    raw value pairs, so re-walk the mask here.
    """
    metrics = compute_metrics(predictions, dataset)
    out: dict[str, SimpleNamespace] = {}
    for d, name in enumerate(dataset.output_channel_names):
        obs_chunks: list = []
        pred_chunks: list = []
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
            mask = bp.mask[..., d]
            obs_chunks.append(bp.y_observed[..., d][mask])
            pred_chunks.append(pred[..., d][mask])
        obs = jnp.concatenate(obs_chunks) if obs_chunks else jnp.empty(0)
        pred = jnp.concatenate(pred_chunks) if pred_chunks else jnp.empty(0)
        m = metrics[name]
        out[name] = SimpleNamespace(
            name=name, n=m.n, obs=obs, pred=pred, r2=float(m.r2), rmse=float(m.rmse)
        )
    return out


def _gaussian_log_prob(z: Array, mean: Array, log_std: Array) -> Array:
    """Diagonal-Gaussian log density, summed over the action dimension.

    No squash correction appears here, and that is the point. The action *is*
    the latent ``z``; ``BoundScaler.from_latent`` is part of the environment's
    dynamics, not part of the policy distribution. A ``tanh``-squashed actor
    would owe a log-det-Jacobian term at this line.
    """
    std = jnp.exp(log_std)
    return jnp.sum(-0.5 * ((z - mean) / std) ** 2 - log_std - _HALF_LOG_2PI)


def _gaussian_entropy(log_std: Array) -> Array:
    """Differential entropy of the diagonal Gaussian.

    ``rlax.entropy_loss`` is categorical: it takes unnormalised logits and
    reduces a softmax entropy. A continuous Gaussian's entropy is a closed form
    in ``log_std`` alone, so it is computed here rather than borrowed.
    """
    return jnp.sum(log_std + _HALF_LOG_2PI_E)


def _policy_mean(agent: Agent, obs: Array) -> Array:
    """Latent mean of the policy at one observation.

    Splits the ``BoundedPredictor`` at ``out_scaler``: the input scaler and the
    inner network produce the Gaussian's mean in latent space, and the output
    scaler is applied later, inside the rollout. Recombined, the two halves are
    exactly ``BoundedPredictor.__call__``.
    """
    return agent.policy.inner(agent.policy.in_scaler.to_latent(obs))


def _value(agent: Agent, obs: Array) -> Array:
    """Value estimate at one observation, sharing the policy's input scaling."""
    return jnp.squeeze(agent.critic(agent.policy.in_scaler.to_latent(obs)), axis=-1)


def _rollout_episode(
    agent: Agent,
    episode: EpisodeData,
    key: Array,
    *,
    solver: SolverConfig,
    penalty_weight: float,
    deterministic: bool,
) -> Rollout:
    """One free-running episode of ``HORIZON`` zero-order-hold intervals.

    The reward is the paper's bounded form, with the framework's own noise model
    supplying the weight::

        r_t = exp(-(Ca_model - Ca_data)^2 / variance) - w * saturation(z)

    The first term lies in ``[0, 1]``, so the undiscounted return over
    ``HORIZON`` intervals has a known ceiling. The penalty reads the *latent*,
    not the physical activity: ``from_latent``'s derivative carries a
    ``sigma'(z)`` factor that underflows to zero exactly where saturation is
    worst, so a penalty written against the output would die where it is needed.
    """
    std = jnp.exp(agent.log_std)

    def step(carry: tuple[Array, Array], i: Array) -> tuple[tuple[Array, Array], tuple[Array, ...]]:
        y, rng = carry
        rng, sub = jr.split(rng)
        obs = jnp.stack([y[0], episode.temperature_C, episode.pH])
        mean = _policy_mean(agent, obs)
        noise = jnp.where(deterministic, 0.0, jr.normal(sub, mean.shape))
        z = mean + std * noise
        log_prob = _gaussian_log_prob(z, mean, agent.log_std)
        activity = agent.policy.out_scaler.from_latent(z)[0]
        y_next = _integrate_interval(
            activity * episode.k_fresh, episode.ts[i], episode.ts[i + 1], y, solver
        )
        err = y_next[0] - episode.ca_obs[i + 1]
        fit = jnp.exp(-(err**2) / episode.variance[i + 1])
        reward = fit - penalty_weight * agent.policy.out_scaler.saturation(z)
        value = _value(agent, obs)
        return (y_next, rng), (obs, z, log_prob, reward, value, activity)

    (y_final, _), (obs, z, log_prob, reward, value, activity) = jax.lax.scan(
        step, (episode.y0, key), jnp.arange(HORIZON)
    )

    # GAE wants values at all HORIZON+1 visited states. The scan emits the first
    # HORIZON; the terminal one comes from the final carry.
    obs_terminal = jnp.stack([y_final[0], episode.temperature_C, episode.pH])
    return Rollout(
        obs=jnp.concatenate([obs, obs_terminal[None, :]], axis=0),
        latent=z,
        log_prob=log_prob,
        reward=reward,
        value=jnp.concatenate([value, _value(agent, obs_terminal)[None]], axis=0),
        activity=activity,
    )


@eqx.filter_jit
def rollout_batch(
    agent: Agent,
    episodes: EpisodeData,
    key: Array,
    *,
    solver: SolverConfig,
    penalty_weight: float,
    n_samples: int,
    deterministic: bool = False,
) -> Rollout:
    """Roll every experiment out ``n_samples`` times, flattened to ``B`` episodes.

    Nothing here is differentiated. The result is data: observations, sampled
    latents, log-probabilities and rewards. The PPO loss recomputes
    log-probabilities from ``obs`` and ``latent`` without re-entering the ODE,
    which is what keeps the solver off the tape entirely.
    """
    keys = jr.split(key, n_samples)
    per_episode = jax.vmap(  # over experiments
        jax.vmap(  # over noise samples
            lambda ep, k: _rollout_episode(
                agent,
                ep,
                k,
                solver=solver,
                penalty_weight=penalty_weight,
                deterministic=deterministic,
            ),
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )(episodes, keys)
    # [n_experiments, n_samples, ...] -> [B, ...]
    return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), per_episode)


def _advantages_and_returns(rollout: Rollout) -> tuple[Array, Array]:
    """GAE advantages and value targets, one row per episode.

    ``discount`` is 1 everywhere except the final step. The return is a fit
    criterion over a finite horizon, not a control return, so discounting it
    would down-weight late measurements for no modelling reason. The trailing
    zero terminates the episode, which stops GAE bootstrapping off a state that
    has no successor.
    """
    discount = jnp.ones((HORIZON,)).at[-1].set(0.0)
    advantages = jax.vmap(
        lambda r, v: rlax.truncated_generalized_advantage_estimation(r, discount, GAE_LAMBDA, v)
    )(rollout.reward, rollout.value)
    returns = advantages + rollout.value[:, :-1]
    return advantages, returns


class Batch(NamedTuple):
    """Flattened transitions, the granularity PPO minibatches at."""

    obs: Float[Array, "M 3"]
    latent: Float[Array, "M 1"]
    log_prob: Float[Array, " M"]
    advantage: Float[Array, " M"]
    target: Float[Array, " M"]


def _ppo_loss(agent: Agent, batch: Batch) -> tuple[Array, tuple[Array, Array, Array]]:
    """Clipped surrogate plus value loss minus entropy bonus.

    No ODE solve happens in here. The rollout already produced the rewards; this
    only re-evaluates the policy and the value head at stored observations.
    """
    mean = jax.vmap(_policy_mean, in_axes=(None, 0))(agent, batch.obs)
    log_prob = jax.vmap(_gaussian_log_prob, in_axes=(0, 0, None))(batch.latent, mean, agent.log_std)
    ratio = jnp.exp(log_prob - batch.log_prob)
    pg_loss = rlax.clipped_surrogate_pg_loss(ratio, batch.advantage, CLIP_EPS)

    value = jax.vmap(_value, in_axes=(None, 0))(agent, batch.obs)
    value_loss = jnp.mean((value - batch.target) ** 2)

    entropy = _gaussian_entropy(agent.log_std)
    total = pg_loss + VF_COEF * value_loss - ENT_COEF * entropy
    return total, (pg_loss, value_loss, entropy)


@eqx.filter_jit
def ppo_update(
    agent: Agent,
    opt_state: Any,
    batch: Batch,
    key: Array,
    *,
    optimiser: Any,
    trainable: Any,
) -> tuple[Agent, Any, Array]:
    """One pass of ``N_PPO_EPOCHS`` epochs over ``N_MINIBATCHES`` shuffled minibatches.

    The agent is split into ``params`` and ``static`` before the scans. Only
    ``params`` rides the carry: ``MLPPredictor`` holds its activation as a
    callable leaf, and a ``lax.scan`` carry has to be arrays. The mask that does
    the splitting is the same one that decides trainability, so the scalers'
    internal arrays land in ``static`` and never see an update.
    """
    params, static = eqx.partition(agent, trainable)
    n = batch.obs.shape[0]
    minibatch_size = n // N_MINIBATCHES

    def loss_on_params(params: Any, mb: Batch) -> tuple[Array, tuple[Array, Array, Array]]:
        return _ppo_loss(eqx.combine(params, static), mb)

    def epoch(carry: tuple[Any, Any, Array], _: Any) -> tuple[tuple[Any, Any, Array], Array]:
        params, opt_state, rng = carry
        rng, sub = jr.split(rng)
        perm = jr.permutation(sub, n)[: minibatch_size * N_MINIBATCHES]
        shuffled = jax.tree.map(lambda x: x[perm], batch)
        reshaped = jax.tree.map(
            lambda x: x.reshape((N_MINIBATCHES, minibatch_size) + x.shape[1:]), shuffled
        )

        def minibatch(carry: tuple[Any, Any], mb: Batch) -> tuple[tuple[Any, Any], Array]:
            params, opt_state = carry
            (loss, _), grads = jax.value_and_grad(loss_on_params, has_aux=True)(params, mb)
            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(minibatch, (params, opt_state), reshaped)
        return (params, opt_state, rng), jnp.mean(losses)

    (params, opt_state, _), losses = jax.lax.scan(
        epoch, (params, opt_state, key), None, length=N_PPO_EPOCHS
    )
    return eqx.combine(params, static), opt_state, jnp.mean(losses)


def train_ppo(
    agent: Agent,
    episodes: EpisodeData,
    *,
    solver: SolverConfig,
    n_updates: int,
    n_samples: int,
    lr: float,
    penalty_weight: float,
    truth_return: float,
    val_episodes: EpisodeData,
    key: Array,
) -> tuple[Agent, Agent, list[float], list[float], list[float]]:
    """Run PPO, returning the best agent and the training, validation and loss curves.

    "Best" is the mean deterministic return on the *training* episodes, which is
    the only signal an honest RL loop has. Keeping the argmax rather than the
    last iterate matters for the same reason ``restore_best`` exists in the
    framework's optax loop: the return is noisy and the final update is not
    reliably the best one.

    Two agents come back. The first is selected on training return, which is the
    only signal an honest RL loop has, and it is the one that overfits. The
    second is selected on validation return, which is ordinary early stopping.
    Reporting both makes the over-parameterisation visible instead of implied:
    once the training return climbs past ``truth_return`` the policy is fitting
    observation noise, and validation is what prices that.
    """
    trainable = agent_trainable(agent)
    optimiser = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(lr),
    )
    opt_state = optimiser.init(eqx.filter(agent, trainable))

    returns: list[float] = []
    val_returns: list[float] = []
    losses: list[float] = []
    best_return = -math.inf
    best_agent = agent
    best_val_return = -math.inf
    best_val_agent = agent

    for update in range(n_updates):
        key, k_roll, k_upd, k_eval, k_val = jr.split(key, 5)
        rollout = rollout_batch(
            agent,
            episodes,
            k_roll,
            solver=solver,
            penalty_weight=penalty_weight,
            n_samples=n_samples,
        )
        advantages, targets = _advantages_and_returns(rollout)
        # Normalising per batch is standard PPO practice; it decouples the step
        # size from the reward scale, which here is set by the noise variance.
        flat_adv = advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        batch = Batch(
            obs=rollout.obs[:, :-1].reshape(-1, len(OBS_KEYS)),
            latent=rollout.latent.reshape(-1, 1),
            log_prob=rollout.log_prob.reshape(-1),
            advantage=flat_adv,
            target=targets.reshape(-1),
        )
        agent, opt_state, loss = ppo_update(
            agent, opt_state, batch, k_upd, optimiser=optimiser, trainable=trainable
        )

        # ``_ag`` bound as a default argument, not captured: the loop rebinds
        # ``agent`` every iteration and ruff's B023 is right to object.
        def deterministic_return(eps: EpisodeData, k: Array, _ag: Agent = agent) -> float:
            evaluation = rollout_batch(
                _ag,
                eps,
                k,
                solver=solver,
                penalty_weight=0.0,
                n_samples=1,
                deterministic=True,
            )
            return float(jnp.mean(jnp.sum(evaluation.reward, axis=-1)))

        mean_return = deterministic_return(episodes, k_eval)
        val_return = deterministic_return(val_episodes, k_val)
        returns.append(mean_return)
        val_returns.append(val_return)
        losses.append(float(loss))
        if mean_return > best_return:
            best_return = mean_return
            best_agent = agent
        if val_return > best_val_return:
            best_val_return = val_return
            best_val_agent = agent

        if update % 10 == 0 or update == n_updates - 1:
            print(
                f"  update {update:4d}  train {mean_return:7.4f}"
                f"  val {val_returns[-1]:7.4f}  (truth scores {truth_return:.4f})"
                f"  loss {float(loss):+.4f}  log_std {float(agent.log_std[0]):+.3f}"
            )

    if not returns:  # --updates 0, used to check the baselines without training
        return best_agent, best_val_agent, returns, val_returns, losses

    peak = int(np.argmax(val_returns))
    print(f"  best training return   {best_return:.4f} (truth scores {truth_return:.4f})")
    print(f"  best validation return {best_val_return:.4f} at update {peak}")
    if best_return > truth_return:
        print(
            f"  the train-selected policy scores {best_return / truth_return:.2f}x the true "
            "law on its own training data, so it is fitting observation noise"
        )
    return best_agent, best_val_agent, returns, val_returns, losses


def episodes_from_experiments(
    experiments: list[Experiment], trunk: tuple[Any, BoundedPredictor]
) -> EpisodeData:
    """Stack experiments into a vmappable :class:`EpisodeData`.

    Every experiment shares the same 12-point time grid, so the stack is
    rectangular and one compiled rollout kernel covers the whole set.
    """
    ts = jnp.stack([exp.channels["Ca"].ts for exp in experiments])
    ca_obs = jnp.stack([exp.channels["Ca"].values for exp in experiments])
    variance = jnp.stack([jnp.asarray(exp.channels["Ca"].variance) for exp in experiments])
    temperature_C = jnp.array([float(exp.covariates["temperature_C"]) for exp in experiments])
    pH = jnp.array([float(exp.covariates["pH"]) for exp in experiments])
    k_fresh = jax.vmap(lambda t, p: trunk_rate(trunk, t, p))(temperature_C, pH)
    y0 = jnp.stack([ca_obs[:, 0], jnp.zeros_like(ca_obs[:, 0])], axis=-1)
    return EpisodeData(
        ts=ts,
        ca_obs=ca_obs,
        variance=variance,
        k_fresh=k_fresh,
        temperature_C=temperature_C,
        pH=pH,
        y0=y0,
    )


def _verify_truth() -> None:
    """Check the deactivation truth is severe enough to be worth learning.

    The point of the aged runs is that the batch *stalls*: fouling shuts the
    reaction down long before the substrate is consumed, so a static rate
    constant is not slightly wrong, it is wrong about the shape of the curve.
    These assertions pin that, rather than pinning particular activity values.
    """
    a0 = float(_activity_true(0.0, PH_REF))
    assert abs(a0 - 1.0) < 1e-12, f"activity at t=0 must be exactly 1, got {a0}"
    print(f"  activity at t=0: {a0:.6f}")
    print(
        f"  tau(pH) spans {float(_tau_true(PH_RANGE[0])):.2f} to "
        f"{float(_tau_true(PH_RANGE[1])):.2f}"
    )

    assert float(_activity_true(T_MAX, PH_RANGE[0])) < float(_activity_true(T_MAX, PH_RANGE[1])), (
        "low pH must age the catalyst faster"
    )

    print(f"  {'pH':>5}{'a(T_MAX)':>10}{'Ca(T_MAX) aged':>16}{'Ca(T_MAX) static':>18}")
    grid = jnp.linspace(0.0, T_MAX, 4001)
    for pH in (PH_RANGE[0], PH_REF, PH_RANGE[1]):
        k = float(_k_true(25.0, pH))
        integral = float(jnp.trapezoid(_activity_true(grid, pH), grid))
        aged = math.exp(-k * integral)
        static = math.exp(-k * T_MAX)
        print(f"  {pH:>5.1f}{float(_activity_true(T_MAX, pH)):>10.4f}{aged:>16.4f}{static:>18.4f}")
        # An absolute gap, not a ratio: the static prediction goes to nearly zero
        # at low pH, where any ratio is huge and says nothing.
        assert aged - static > 0.25, (
            f"at pH {pH} the aged batch must stall well short of the static prediction, "
            f"got Ca={aged:.4f} against {static:.4f} (gap {aged - static:.4f})"
        )
        assert aged < 0.75, (
            f"at pH {pH} the aged batch must still make real progress, got Ca={aged:.4f}; "
            "a batch that barely reacts carries no information about k"
        )

    # An exponential fitted through (0, 1) and the endpoint predicts the midpoint
    # monotonically; the truth's plateau means it does not. The gap is the room
    # the policy has to beat the fixed-form baseline in.
    for pH in (PH_RANGE[0], PH_RANGE[1]):
        end = float(_activity_true(T_MAX, pH))
        k_d = -math.log(end) / T_MAX
        print(
            f"  pH {pH}: midpoint truth {float(_activity_true(T_MAX / 2, pH)):.4f} vs "
            f"best-exponential {math.exp(-k_d * T_MAX / 2):.4f}"
        )


def reference_returns(episodes: EpisodeData, solver: SolverConfig) -> tuple[float, float]:
    """Score the exact zero-order-hold target and the do-nothing model.

    Returns ``(truth, static)``. ``truth`` is what the hidden deactivation law
    itself scores against this noisy data: the interval-mean activity reproduces
    the continuous truth exactly, so the only residue is observation noise and
    whatever the frozen trunk gets wrong about ``k``.

    A reference, not a ceiling. A model scoring *above* it is fitting noise
    rather than signal, the over-parameterisation failure mode hybrid models
    are prone to, and reading the return against this number is how that
    shows up.

    The undiscounted return does *not* top out at ``HORIZON``. With
    ``r = exp(-err^2 / sigma^2)`` and residuals ``N(0, sigma^2)`` at the true
    model, ``E[r] = 1/sqrt(3) = 0.577``, so the noise-limited optimum is about
    ``0.577 * HORIZON`` and reporting against ``HORIZON`` would understate a
    converged policy by nearly a factor of two.
    """

    def scored(activity_fn: Any) -> Array:
        def one(ep: EpisodeData) -> Array:
            def step(carry: tuple[Array, Array], i: Array) -> tuple[tuple[Array, Array], None]:
                y, total = carry
                a = activity_fn(ep.ts[i], ep.ts[i + 1], ep.pH)
                y_next = _integrate_interval(a * ep.k_fresh, ep.ts[i], ep.ts[i + 1], y, solver)
                err = y_next[0] - ep.ca_obs[i + 1]
                return (y_next, total + jnp.exp(-(err**2) / ep.variance[i + 1])), None

            (_, total), _ = jax.lax.scan(step, (ep.y0, jnp.asarray(0.0)), jnp.arange(HORIZON))
            return total

        return jax.vmap(one)(episodes)

    truth = float(jnp.mean(scored(_activity_interval_mean)))
    static = float(jnp.mean(scored(lambda t0, t1, pH: jnp.asarray(1.0))))
    return truth, static


def _verify_rollout_ceiling(episodes: EpisodeData, solver: SolverConfig) -> tuple[float, float]:
    """The checkpoint the whole script rests on (SPEC_RL.md step 5).

    If the exact target does not clearly beat the do-nothing model, the reward
    scaling or the zero-order hold is wrong and no amount of PPO tuning fixes
    it. Both numbers are reported so the learning curves later have a scale.
    """
    truth, static = reference_returns(episodes, solver)
    noise_floor = HORIZON / math.sqrt(3.0)
    print(f"  the true deactivation law scores: {truth:7.4f}")
    print(f"  no deactivation at all:           {static:7.4f}")
    print(f"  pure-noise optimum (HORIZON/sqrt 3): {noise_floor:.4f}")
    print(f"  range to learn in: {static:.2f} to {truth:.2f}, above which is noise-fitting")
    assert truth > 0.6 * noise_floor, (
        "the exact target should approach the noise-limited optimum; if it does not, "
        "the reward scaling or the zero-order hold is wrong"
    )
    assert truth > 2.0 * static, (
        "deactivation must matter, or there is nothing for the policy to learn"
    )
    return truth, static


def _activity_recovery_plot(
    agent: Agent,
    episodes: EpisodeData,
    experiments: list[Experiment],
    exponential: ExponentialDecay,
    gradient_policy: BoundedPredictor,
    *,
    solver: SolverConfig,
    save_path: Path,
) -> None:
    """Recovered activity against the truth, one panel per experiment."""
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    dense = jnp.linspace(0.0, T_MAX, 200)

    for j, (ax, exp) in enumerate(zip(axes[0], experiments, strict=True)):
        pH = float(exp.covariates["pH"])
        T_C = float(exp.covariates["temperature_C"])
        ax.plot(dense, _activity_true(dense, pH), color="black", lw=2, label="truth a(t)")
        ts_exp = np.asarray(exp.channels["Ca"].ts)
        ax.step(
            ts_exp[:-1],
            [float(_activity_interval_mean(ts_exp[i], ts_exp[i + 1], pH)) for i in range(HORIZON)],
            where="post",
            color="black",
            ls="--",
            lw=1.2,
            label="exact hold target",
        )

        episode = jax.tree.map(lambda x, j=j: x[j], episodes)
        rollout = rollout_batch(
            agent,
            jax.tree.map(lambda x: x[None], episode),
            jr.PRNGKey(0),
            solver=solver,
            penalty_weight=0.0,
            n_samples=1,
            deterministic=True,
        )
        ts = np.asarray(episode.ts)
        ax.step(
            ts[:-1],
            np.asarray(rollout.activity[0]),
            where="post",
            color="tab:red",
            lw=1.8,
            label="PPO policy",
        )

        # Default-argument binding, not closure capture: ruff's B023 is right
        # that a bare `episode` here would resolve to the loop's last value.
        grad_states = zoh_states(
            lambda t, ca, _ep=episode: gradient_policy(
                {"Ca": ca, "temperature_C": _ep.temperature_C, "pH": _ep.pH}
            )[0],
            episode.ts,
            episode.k_fresh,
            episode.y0,
            solver,
        )
        grad_activity = [
            float(gradient_policy({"Ca": grad_states[i, 0], "temperature_C": T_C, "pH": pH})[0])
            for i in range(HORIZON)
        ]
        ax.step(
            ts[:-1],
            grad_activity,
            where="post",
            color="tab:blue",
            lw=1.4,
            ls="--",
            label="optax policy",
        )

        k_d = float(exponential.rate())
        ax.plot(
            dense,
            np.exp(-k_d * np.asarray(dense)),
            color="tab:green",
            lw=1.4,
            ls=":",
            label="exp decay",
        )

        ax.set_title(f"T = {T_C:.1f} °C, pH = {pH:.2f}", fontsize=10)
        ax.set_xlabel("time")
        ax.set_ylim(-0.05, 1.15)
        if j == 0:
            ax.set_ylabel("catalyst activity")
            ax.legend(fontsize=8)

    fig.suptitle("Recovered catalyst activity against the hidden truth")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _trajectory_plot(
    experiments: list[Experiment],
    predictions: dict[str, list[np.ndarray]],
    *,
    save_path: Path,
    title: str,
) -> None:
    """Observed Ca against every model on the page, one panel per experiment."""
    n = len(experiments)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    colours = {
        "frozen trunk": "tab:grey",
        "exp decay": "tab:green",
        "optax policy": "tab:blue",
        "PPO train-sel": "tab:orange",
        "PPO val-sel": "tab:red",
    }

    for i, exp in enumerate(experiments):
        ax = axes[i // ncols][i % ncols]
        ts = np.asarray(exp.channels["Ca"].ts)
        ax.plot(ts, np.asarray(exp.channels["Ca"].values), "o", ms=4, color="black", label="data")
        for name, series in predictions.items():
            ax.plot(ts, series[i], lw=1.6, color=colours.get(name), label=name)
        ax.set_title(
            f"T = {float(exp.covariates['temperature_C']):.1f} °C, "
            f"pH = {float(exp.covariates['pH']):.2f}",
            fontsize=10,
        )
        ax.set_xlabel("time")
        ax.set_ylabel("Ca")
        if i == 0:
            ax.legend(fontsize=8)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _learning_curve_plot(
    returns: list[float],
    val_returns: list[float],
    losses: list[float],
    *,
    truth_return: float,
    static_return: float,
    save_path: Path,
) -> None:
    """PPO return against update, with the true law's own score marked.

    The reference is not ``HORIZON``. Observation noise caps a perfect model at
    about ``0.577`` per step, so the true deactivation law scores roughly
    ``0.577 * HORIZON`` against its own noisy data. Training return climbing
    above that line is the policy fitting noise, which is why the validation
    curve is plotted alongside it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(returns, color="tab:red", lw=1.4, label="training")
    axes[0].plot(val_returns, color="tab:blue", lw=1.4, label="validation")
    axes[0].axhline(
        truth_return,
        color="black",
        ls="--",
        lw=1.0,
        label=f"true law scores {truth_return:.2f}",
    )
    axes[0].axhline(
        static_return,
        color="tab:grey",
        ls=":",
        lw=1.0,
        label=f"no deactivation = {static_return:.2f}",
    )
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("mean deterministic return")
    axes[0].legend(fontsize=8)
    axes[1].plot(losses, color="tab:purple", lw=1.4)
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("PPO loss")
    fig.suptitle("PPO learning curve")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _saturation_plot(
    agent: Agent, episodes: EpisodeData, *, solver: SolverConfig, save_path: Path
) -> None:
    """Sampled latents against the squash knee, showing where the policy operates."""
    rollout = rollout_batch(
        agent, episodes, jr.PRNGKey(7), solver=solver, penalty_weight=0.0, n_samples=64
    )
    z = np.asarray(rollout.latent).reshape(-1)
    knee = float(agent.policy.out_scaler.z_knee)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(z, bins=60, color="tab:red", alpha=0.75)
    for sign in (-1.0, 1.0):
        ax.axvline(sign * knee, color="black", ls="--", lw=1.0)
    ax.set_xlabel("sampled latent z")
    ax.set_ylabel("count")
    ax.set_title(
        f"Policy latents against the sigmoid knee at ±{knee:.3f}\n"
        f"{100.0 * float(np.mean(np.abs(z) > knee)):.1f}% of samples past the knee",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doe-seed", type=int, default=0, help="LHS sampler seed")
    parser.add_argument("--seed", type=int, default=0, help="JAX root key")
    parser.add_argument(
        "--trunk",
        type=Path,
        default=Path(__file__).resolve().parent / "artefacts" / "trunk_fresh.eqx",
        help="Frozen trunk written by train_hybrid.py --save-predictors",
    )
    parser.add_argument("--updates", type=int, default=600, help="PPO updates")
    parser.add_argument(
        "--rollouts", type=int, default=64, help="Policy noise samples per experiment per update"
    )
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--penalty-weight", type=float, default=1e-3)
    parser.add_argument("--baseline-steps", type=int, default=600)
    parser.add_argument("--baseline-lr", type=float, default=3e-3)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures_rl"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    root = jr.PRNGKey(args.seed)
    k_noise, k_template, k_agent, k_exp, k_grad, k_ppo = jr.split(root, 6)

    # ---- Truth ------------------------------------------------------------- #
    print("[verify] deactivation truth")
    _verify_truth()

    # ---- Aged dataset ------------------------------------------------------ #
    print("\n[build] aged dataset")
    train_experiments, val_experiments = _build_aged_datasets(
        doe_seed=args.doe_seed, noise_key=k_noise
    )
    train_dataset = make_dataset(
        train_experiments,
        output_channel_names=OUTPUT_CHANNELS,
    )
    val_dataset = make_dataset(
        val_experiments,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(train_experiments)} aged training runs, {len(val_experiments)} validation")
    print(f"  buckets: {[bp.ts.shape for bp in train_dataset.bucket_payloads]}")

    # ---- Frozen trunk ------------------------------------------------------ #
    print("\n[load] frozen trunk from the fresh-catalyst calibration")
    trunk = load_trunk(args.trunk, key=k_template)
    log_k_ref, Ea = trunk[0]()
    print(f"  trunk parametric: log_k_ref {float(log_k_ref):+.4f}, Ea {float(Ea):.2f} kJ/mol")
    for name, pts in (("train", _lhs_design(seed=args.doe_seed)), ("val", VALIDATION_POINTS)):
        rows = []
        for T_C, pH in pts:
            k_trunk = float(trunk_rate(trunk, jnp.asarray(T_C), jnp.asarray(pH)))
            rows.append(
                f"T={T_C:5.1f} pH={pH:4.2f}  k_trunk={k_trunk:.4f}"
                f"  k_true={float(_k_true(T_C, pH)):.4f}"
            )
        print(f"  {name}: " + "\n         ".join(rows))

    solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05)

    episodes = episodes_from_experiments(train_experiments, trunk)
    val_episodes = episodes_from_experiments(val_experiments, trunk)

    # ---- Step 5 checkpoint ------------------------------------------------- #
    print("\n[verify] rollout and reward, before any learning")
    truth_return, static_return = _verify_rollout_ceiling(episodes, solver)

    # ---- Baseline 2: fitted exponential decay ------------------------------ #
    print("\n[baseline] trunk + fitted exponential decay scalar (optax)")
    exponential = ExponentialDecay(key=k_exp)
    mask_exp = frozen_default_mask(exponential, BoundScaler)
    history_exp, exponential = train_with_optax(
        exponential,
        train_dataset,
        OptaxTrainingConfig(
            steps=(args.baseline_steps,),
            lr=(args.baseline_lr,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            length_schedule=(1.0,),
            loss="mse",
            verbose=False,
        ),
        simulate_fn=make_simulate_fn(trunk, kind="exponential"),
        state_to_output=state_to_output,
        solver=solver,
        trainable=mask_exp,
        key=k_exp,
    )
    print(f"  final loss {history_exp[-1]:.6f}, fitted k_d {float(exponential.rate()):.4f}")

    # ---- Baseline 3: the same policy, trained by backprop ------------------ #
    print("\n[baseline] trunk + the activity BoundedPredictor (optax backprop through the solve)")
    gradient_policy = build_policy(key=k_grad)
    mask_grad = frozen_default_mask(gradient_policy, BoundScaler)
    history_grad, gradient_policy = train_with_optax(
        gradient_policy,
        train_dataset,
        OptaxTrainingConfig(
            steps=(args.baseline_steps,),
            lr=(args.baseline_lr,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            length_schedule=(1.0,),
            loss="mse",
            verbose=False,
        ),
        simulate_fn=make_simulate_fn(trunk, kind="policy"),
        state_to_output=state_to_output,
        solver=solver,
        trainable=mask_grad,
        key=k_grad,
    )
    print(f"  final loss {history_grad[-1]:.6f}")

    # ---- PPO --------------------------------------------------------------- #
    print("\n[ppo] the same policy, trained without differentiating the solve")
    agent = build_agent(key=k_agent)
    print(
        f"  {args.rollouts} noise samples x {len(train_experiments)} experiments = "
        f"{args.rollouts * len(train_experiments)} episodes per update, "
        f"horizon {HORIZON}"
    )
    agent, val_agent, returns, val_returns, losses = train_ppo(
        agent,
        episodes,
        solver=solver,
        n_updates=args.updates,
        n_samples=args.rollouts,
        lr=args.ppo_lr,
        penalty_weight=args.penalty_weight,
        truth_return=truth_return,
        val_episodes=val_episodes,
        key=k_ppo,
    )

    # ---- Results ----------------------------------------------------------- #
    print("\n[results] aged validation set")
    models: dict[str, tuple[Any, str]] = {
        "frozen trunk": (None, "frozen"),
        "exp decay": (exponential, "exponential"),
        "optax policy": (gradient_policy, "policy"),
        "PPO train-sel": (agent.policy, "policy"),
        "PPO val-sel": (val_agent.policy, "policy"),
    }
    table: list[tuple[str, float, float, float]] = []
    val_predictions: dict[str, list[np.ndarray]] = {}
    train_predictions: dict[str, list[np.ndarray]] = {}
    for name, (predictors, kind) in models.items():
        simulate = make_simulate_fn(trunk, kind=kind)
        val_pred = predict_dataset(
            predictors,
            val_dataset,
            simulate_fn=simulate,
            state_to_output=state_to_output,
            solver=solver,
        )
        train_pred = predict_dataset(
            predictors,
            train_dataset,
            simulate_fn=simulate,
            state_to_output=state_to_output,
            solver=solver,
        )
        diag_val = compute_metrics(val_pred, val_dataset)["Ca"]
        diag_train = compute_metrics(train_pred, train_dataset)["Ca"]
        table.append(
            (name, float(diag_train.r2), float(diag_val.r2), float(diag_val.rmse))
        )
        val_predictions[name] = [
            np.asarray(val_pred[0][i, :, 0]) for i in range(len(val_experiments))
        ]
        train_predictions[name] = [
            np.asarray(train_pred[0][i, :, 0]) for i in range(len(train_experiments))
        ]

    print(f"  {'model':<16}{'train R2':>10}{'val R2':>10}{'val RMSE':>11}{'gap':>9}")
    for name, r2_train, r2_val, rmse_val in table:
        print(
            f"  {name:<16}{r2_train:>10.4f}{r2_val:>10.4f}{rmse_val:>11.4f}"
            f"{r2_train - r2_val:>+9.4f}"
        )
    print(
        "  gap is train R2 minus val R2. A large positive gap on a learned policy is the "
        "over-parameterisation\n  failure mode: 11 free actions per run against 12 noisy "
        "observations leaves room to fit the noise."
    )

    print("\n  recovered activity at t = T_MAX against the truth")
    print(f"  {'T (K)':>7}{'pH':>6}{'truth':>9}{'PPO':>9}{'optax':>9}{'exp':>9}")
    for exp in val_experiments:
        T_C = float(exp.covariates["temperature_C"])
        pH = float(exp.covariates["pH"])
        truth = float(_activity_true(T_MAX, pH))
        ca_end_ppo = float(val_predictions["PPO val-sel"][val_experiments.index(exp)][-1])
        ca_end_grad = float(val_predictions["optax policy"][val_experiments.index(exp)][-1])
        ppo_a = float(val_agent.policy({"Ca": ca_end_ppo, "temperature_C": T_C, "pH": pH})[0])
        grad_a = float(gradient_policy({"Ca": ca_end_grad, "temperature_C": T_C, "pH": pH})[0])
        exp_a = float(jnp.exp(-exponential.rate() * T_MAX))
        print(f"  {T_C:>7.1f}{pH:>6.2f}{truth:>9.4f}{ppo_a:>9.4f}{grad_a:>9.4f}{exp_a:>9.4f}")

    # ---- Plots ------------------------------------------------------------- #
    if not args.no_plot:
        _learning_curve_plot(
            returns,
            val_returns,
            losses,
            truth_return=truth_return,
            static_return=static_return,
            save_path=args.plot_dir / "01_ppo_learning.png",
        )
        _trajectory_plot(
            val_experiments,
            val_predictions,
            save_path=args.plot_dir / "02_val_trajectories.png",
            title="Aged validation runs: Ca against every model",
        )
        _trajectory_plot(
            train_experiments,
            train_predictions,
            save_path=args.plot_dir / "03_train_trajectories.png",
            title="Aged training runs: Ca against every model",
        )
        _activity_recovery_plot(
            val_agent,
            val_episodes,
            val_experiments,
            exponential,
            gradient_policy,
            solver=solver,
            save_path=args.plot_dir / "04_activity_recovery.png",
        )
        _saturation_plot(
            val_agent, episodes, solver=solver, save_path=args.plot_dir / "05_latent_saturation.png"
        )
        val_policy_pred = predict_dataset(
            val_agent.policy,
            val_dataset,
            simulate_fn=make_simulate_fn(trunk, kind="policy"),
            state_to_output=state_to_output,
            solver=solver,
        )
        diag = compute_metrics(val_policy_pred, val_dataset)
        print_metrics(diag)
        parity_plot(
            _parity_diagnostics(val_policy_pred, val_dataset),
            title="PPO policy parity (aged validation)",
            save_path=args.plot_dir / "06_parity_ppo.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
