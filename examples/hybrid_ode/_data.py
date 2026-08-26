"""Synthetic data for the hybrid example, in the two layouts the library handles.

One system, sampled two ways.

``rectangular_dataset`` samples every experiment on the same time grid and
measures both channels at every point. ``make_dataset`` turns that into a
single bucket whose mask is entirely ``True``.

``irregular_dataset`` gives every experiment its own end time and its own
randomly drawn sample times, then thins each channel independently. Union
lengths then differ across experiments, so ``make_dataset`` builds several
buckets and the mask is about half full.

The same model in ``train_hybrid_ode.py`` trains on either without a line
of difference, which is the point of the pair.

The system
----------
A damped rotation with an unknown cubic coupling::

    dy/dt = [[-k, w], [-w, -k]] y + C y^3

``w`` is known to the model. ``k`` varies with an experiment's temperature
through an Arrhenius law and spans a factor of 44 across the levels used
here, which is why the model bounds it with ``warp="log10"``. ``C`` is
unknown to the model, and the residual network has to reproduce its effect
from trajectories alone.

``t = 0`` stays in every channel so ``y0_fn`` can read the initial state
off the first observation. Relaxing that needs an encoder, which is the
latent-ODE problem rather than this one.
"""

# ruff: noqa: F722

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from hybridmodels import ChannelObs, Dataset, Experiment, make_dataset, make_experiment

OMEGA_TRUE: float = 1.0
"""Rotation frequency. Known to the model, not fitted."""

COUPLING: Float[Array, "2 2"] = jnp.array([[0.0, 0.6], [-0.6, 0.0]])
"""Cubic coupling. Unknown to the model; the residual network learns it."""

K_REF: float = 0.05
"""Decay rate at ``T_REF``, in inverse time units."""

EA_OVER_R: float = 6000.0
"""Arrhenius slope, in kelvin."""

T_REF: float = 310.0
"""Reference temperature of the Arrhenius law, in kelvin."""

TEMPERATURES: tuple[float, ...] = (280.0, 292.0, 304.0, 316.0, 328.0, 340.0)
"""Covariate levels. Each experiment runs at one of these."""

CHANNELS: tuple[str, ...] = ("y1", "y2")
"""Observed channel names, in state order."""


def true_k(temperature: float | Array) -> Array:
    """Arrhenius decay rate at ``temperature`` (kelvin). Ground truth."""
    t = jnp.asarray(temperature)
    return K_REF * jnp.exp(-EA_OVER_R * (1.0 / t - 1.0 / T_REF))


def hybrid_field(k: Array, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
    """Damped rotation plus cubic coupling.

    Split the way the model splits it. The first term is what the hybrid
    keeps in closed form up to the unknown ``k``; the second is what the
    residual network has to learn.
    """
    rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]]) @ y
    return rotation + COUPLING @ y**3


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 2"]:
    """Both state components are observed, in order, so this is the identity."""
    return state


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
    """Initial state from the ``t = 0`` sample of each channel."""
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


def rectangular_experiments(
    *,
    n_experiments: int = 24,
    n_times: int = 20,
    t1: float = 8.0,
    noise_std: float = 0.03,
    key: Array,
) -> list[Experiment]:
    """One shared time grid, both channels measured at every point.

    Produces a single bucket with an all-``True`` mask. Included so the
    regular case is visible next to the general one. The library has no
    separate path for it.
    """
    ts = jnp.linspace(0.0, t1, n_times)
    y0_key, noise_key = jr.split(key)
    y0s = jr.uniform(y0_key, (n_experiments, 2), minval=-0.6, maxval=1.0)
    temperatures = jnp.asarray([TEMPERATURES[i % len(TEMPERATURES)] for i in range(n_experiments)])
    clean = jax.vmap(
        lambda y0, temp: _solve(lambda t, y, args: hybrid_field(true_k(temp), y), ts, y0)
    )(y0s, temperatures)
    observed = clean + noise_std * jr.normal(noise_key, clean.shape)
    return [
        _experiment(
            (ts, ts),
            (observed[i, :, 0], observed[i, :, 1]),
            noise_std,
            float(temperatures[i]),
            exp_id=f"rect_{i:02d}_T{int(temperatures[i])}",
        )
        for i in range(n_experiments)
    ]


def irregular_experiments(
    *,
    n_experiments: int = 24,
    samples_per_channel: tuple[int, ...] = (8, 12),
    t1_range: tuple[float, float] = (6.0, 9.0),
    noise_std: float = 0.03,
    key: Array,
) -> list[Experiment]:
    """Per-experiment end times, drawn sample times, channels thinned separately.

    The number of samples a channel receives is drawn from
    ``samples_per_channel``, so the union length is ``n1 + n2 - 1`` (the
    shared ``t = 0`` counts once) and takes one of a small set of values.
    That set is the bucket structure: with the default two-element tuple
    there are three distinct lengths, so three buckets and three
    compilations, with nothing padded and nothing dropped.
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
        # independently truncated views of the same system.
        union = jnp.unique(jnp.concatenate(ts_channels), size=int(counts.sum()) - 1)
        y0 = jr.uniform(k_y0, (2,), minval=-0.6, maxval=1.0)
        ys = _solve(lambda t, y, args, _k=k_decay: hybrid_field(_k, y), union, y0)

        values = []
        noise_keys = jr.split(k_noise, 2)
        for d, (ts_c, nkey) in enumerate(zip(ts_channels, noise_keys, strict=True)):
            clean = ys[jnp.searchsorted(union, ts_c), d]
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


def build_dataset(experiments: list[Experiment]) -> Dataset:
    """Bucket a list of experiments with this example's projector and channels."""
    return make_dataset(
        experiments,
        state_to_output=state_to_output,
        output_channel_names=CHANNELS,
    )


def describe_buckets(dataset: Dataset) -> str:
    """One line per bucket: how many experiments, how long, how full the mask is."""
    lines = [f"{len(dataset.bucket_payloads)} bucket(s)"]
    for i, bp in enumerate(dataset.bucket_payloads):
        n, t, d = bp.y_observed.shape
        lines.append(
            f"  bucket {i}: N={n:2d} experiments, T={t:3d} timestamps, "
            f"D={d} channels, mask {float(bp.mask.mean()):.2f} full"
        )
    return "\n".join(lines)
