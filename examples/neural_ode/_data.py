"""Synthetic spiral datasets in the two shapes this framework has to handle.

The generators here are deliberate ports of the ones in the Equinox and
diffrax example galleries, so a reader who already knows those files can
see exactly what changes when the data goes through ``make_dataset``.

``rectangular_spiral`` is the Equinox ``neural_ode`` dataset: a cubic
spiral ``dy/dt = A y^3``, every experiment sampled on the same time grid,
both coordinates observed at every sample. One shared ``T`` means
``make_dataset`` produces a **single bucket** whose mask is all ``True``.
This is the easy case, and the point of including it is that the
framework does not treat it specially. It is the degenerate bucketing,
not a separate code path.

``irregular_spiral`` is the diffrax ``latent_ode`` dataset: per-experiment
end times and randomly drawn sample times, here pushed one step further
by thinning each channel independently. Two coordinates measured at
different instants is the normal situation in a laboratory, and it is
what the union timestamp axis exists for. Experiments then disagree on
``T``, so ``make_dataset`` groups them into **several buckets** and the
mask is genuinely sparse.

Both generators use the same underlying physics so the two scripts stay
comparable:

    dy/dt = -k y_rot + omega R y + C y^3

with ``R`` a 90-degree rotation, ``C`` a cubic coupling, and ``k`` a decay
rate that varies across experiments with a temperature covariate. The
neural ODE script throws all of that away and learns the whole field. The
hybrid script keeps ``omega`` and the rotation, learns ``k`` from
temperature with one network, and learns ``C y^3`` with another.

``t = 0`` is always retained in every channel, so ``y0_fn`` can read the
initial state off the first observation of each channel the way the
framework's docstring suggests. Dropping it would need an encoder, which
is the latent-ODE problem and not this one.
"""

# ruff: noqa: F722

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from hybridmodels import ChannelObs, Experiment, make_experiment

CUBIC_A: Float[Array, "2 2"] = jnp.array([[-0.1, 1.3], [-1.0, -0.1]])
"""Coefficient matrix of the Equinox spiral ``dy/dt = A y^3``."""

OMEGA_TRUE: float = 1.0
"""Rotation frequency of the hybrid system. Known to the model, not fitted."""

COUPLING: Float[Array, "2 2"] = jnp.array([[0.0, 0.6], [-0.6, 0.0]])
"""Cubic coupling of the hybrid system. Unknown to the model; the residual
network has to reproduce its effect from data alone."""

K_REF: float = 0.05
"""Decay rate at ``T_REF``, in inverse time units."""

EA_OVER_R: float = 6000.0
"""Arrhenius slope, in kelvin. Gives ``k`` a range of roughly 1.6 decades
over ``TEMPERATURES``, which is the reason the hybrid script bounds ``k``
with ``warp="log10"``."""

T_REF: float = 310.0
"""Reference temperature of the Arrhenius law, in kelvin."""

TEMPERATURES: tuple[float, ...] = (280.0, 292.0, 304.0, 316.0, 328.0, 340.0)
"""Covariate levels. Each experiment is run at one of these."""

CHANNELS: tuple[str, ...] = ("y1", "y2")
"""Observed channel names, in state order."""


def true_k(temperature: float | Array) -> Array:
    """Arrhenius decay rate at ``temperature`` (kelvin).

    The ground truth the hybrid script's rate network is asked to recover
    from six temperature levels. It is never shown to the model.
    """
    t = jnp.asarray(temperature)
    return K_REF * jnp.exp(-EA_OVER_R * (1.0 / t - 1.0 / T_REF))


def cubic_field(t: Array, y: Float[Array, " 2"], args: object) -> Float[Array, " 2"]:
    """Equinox spiral vector field ``A y^3``, elementwise cube."""
    return CUBIC_A @ y**3


def hybrid_field(k: Array, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
    """Damped rotation plus cubic coupling, the hybrid script's ground truth.

    Split the way the model will split it: the first term is the part the
    hybrid model keeps in closed form up to the unknown ``k``, the second
    is the part the residual network has to learn.
    """
    rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]]) @ y
    return rotation + COUPLING @ y**3


def _solve(field, ts: Float[Array, " T"], y0: Float[Array, " 2"]) -> Float[Array, "T 2"]:
    """Reference integration, tighter than anything training will use."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=100_000,
    )
    return jnp.asarray(sol.ys)


def _y0_from_first_observations(cov: dict[str, Array], channels: dict[str, ChannelObs]) -> Array:
    """Initial state from the ``t = 0`` sample of each channel.

    Both generators keep ``t = 0`` in every channel, so index 0 is the
    initial observation and the noise on it is the noise the model has to
    live with. This is the hook the framework's ``make_experiment``
    docstring describes for systems whose observed channels are the state.
    """
    return jnp.stack([channels[name].values[0] for name in CHANNELS])


def _experiment(
    ts_per_channel: tuple[Array, Array],
    values_per_channel: tuple[Array, Array],
    noise_std: float,
    temperature: float,
    exp_id: str,
) -> Experiment:
    channels = {
        name: ChannelObs(ts=ts, values=values, variance=jnp.full(ts.shape, noise_std**2))
        for name, ts, values in zip(CHANNELS, ts_per_channel, values_per_channel, strict=True)
    }
    return make_experiment(
        covariates={"temperature": temperature},
        channels=channels,
        y0_fn=_y0_from_first_observations,
        exp_id=exp_id,
    )


def rectangular_spiral(
    *,
    n_experiments: int = 16,
    n_times: int = 40,
    t1: float = 8.0,
    noise_std: float = 0.02,
    key: Array,
) -> list[Experiment]:
    """Equinox ``neural_ode`` dataset, one shared time grid, one bucket.

    ``y0`` is drawn uniformly from ``[-0.6, 1]^2`` exactly as in the
    upstream example. Every experiment is sampled on ``linspace(0, t1,
    n_times)`` and both coordinates are observed at every sample, so the
    union axis is that grid, the mask is all ``True``, and all
    ``n_experiments`` land in one ``BucketPayload``.

    The ``temperature`` covariate is carried anyway, at a constant value.
    It is unused by the pure neural ODE and keeps the two datasets
    interchangeable if a reader wants to swap them.
    """
    ts = jnp.linspace(0.0, t1, n_times)
    y0_key, noise_key = jr.split(key)
    y0s = jr.uniform(y0_key, (n_experiments, 2), minval=-0.6, maxval=1.0)
    ys = jax.vmap(lambda y0: _solve(cubic_field, ts, y0))(y0s)
    noise = noise_std * jr.normal(noise_key, ys.shape)
    observed = ys + noise
    return [
        _experiment(
            (ts, ts),
            (observed[i, :, 0], observed[i, :, 1]),
            noise_std,
            T_REF,
            exp_id=f"rect_{i:02d}",
        )
        for i in range(n_experiments)
    ]


def irregular_spiral(
    *,
    n_experiments: int = 24,
    samples_per_channel: tuple[int, ...] = (8, 12),
    t1_range: tuple[float, float] = (6.0, 9.0),
    noise_std: float = 0.03,
    key: Array,
) -> list[Experiment]:
    """diffrax ``latent_ode`` dataset, thinned per channel, several buckets.

    Each experiment gets its own end time and its own randomly drawn
    sample times, and each channel is thinned independently. The number of
    samples a channel receives is drawn from ``samples_per_channel``, so
    the union length is ``n1 + n2 - 1`` (the shared ``t = 0`` counts once)
    and takes one of a small set of values. That set is the bucket
    structure: with the default two-element tuple there are three distinct
    ``T`` values and therefore three buckets, so training compiles three
    times and no experiment is padded.

    Every experiment also carries a ``temperature`` covariate that sets its
    true decay rate. The hybrid script fits a network to that mapping.
    """
    keys = jr.split(key, n_experiments)
    experiments: list[Experiment] = []
    for i, exp_key in enumerate(keys):
        k_end, k_y0, k_n, k_t1, k_t2, k_noise = jr.split(exp_key, 6)
        temperature = TEMPERATURES[i % len(TEMPERATURES)]
        k_decay = true_k(temperature)
        end = jr.uniform(k_end, (), minval=t1_range[0], maxval=t1_range[1])

        counts = jr.choice(k_n, jnp.asarray(samples_per_channel), shape=(2,))
        ts_channels = []
        for count_key, count in zip((k_t1, k_t2), counts, strict=True):
            interior = jr.uniform(count_key, (int(count) - 1,), minval=0.0, maxval=1.0) * end
            ts_channels.append(jnp.concatenate([jnp.zeros((1,)), jnp.sort(interior)]))

        # One dense solve on the union of both channels' times, then read
        # each channel off it. Solving twice would give the two channels
        # independently truncated trajectories of the same system.
        union = jnp.unique(jnp.concatenate(ts_channels), size=int(counts.sum()) - 1)
        y0 = jr.uniform(k_y0, (2,), minval=-0.6, maxval=1.0)
        ys = _solve(lambda t, y, args, _k=k_decay: hybrid_field(_k, y), union, y0)

        values = []
        noise_keys = jr.split(k_noise, 2)
        for d, (ts_c, nkey) in enumerate(zip(ts_channels, noise_keys, strict=True)):
            idx = jnp.searchsorted(union, ts_c)
            clean = ys[idx, d]
            values.append(clean + noise_std * jr.normal(nkey, clean.shape))

        experiments.append(
            _experiment(
                (ts_channels[0], ts_channels[1]),
                (values[0], values[1]),
                noise_std,
                float(temperature),
                exp_id=f"irr_{i:02d}_T{int(temperature)}",
            )
        )
    return experiments


def describe_buckets(dataset) -> str:
    """One line per bucket: how many experiments, how long, how full the mask is."""
    lines = [f"{len(dataset.bucket_payloads)} bucket(s)"]
    for i, bp in enumerate(dataset.bucket_payloads):
        n, t, d = bp.y_observed.shape
        density = float(bp.mask.mean())
        lines.append(f"  bucket {i}: N={n:2d}  T={t:3d}  D={d}  mask density {density:.2f}")
    return "\n".join(lines)
