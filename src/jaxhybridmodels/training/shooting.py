"""Multiple shooting: solve each trajectory as segments that start from trainable nodes.

Single shooting solves each experiment once, from ``y0`` to its last sample.
With a poor starting guess, an oscillating or stiff system, or a long
horizon, that trajectory drifts away from the data early, and the loss stops
measuring how wrong the parameters are. Multiple shooting splits each
trajectory into segments. Segment 0 starts from ``y0``; every later segment
starts from its own state at its first sample, a **node**, which is trained
alongside the predictors. A penalty on the **gap** where one segment ends
and the next begins pulls the segments back into one trajectory.

A ``TrainingPhase(segments=K)`` turns it on for one phase, and the schedule
must end in single shooting, so the returned model has been fitted as one
solve. The nodes belong to the run and are dropped when it ends.

Layout
------
Segments are counted in samples. A bucket's ``T`` union timestamps are split
into segments of ``L`` samples that overlap by one: segment ``k`` covers
indices ``k(L-1)`` to ``(k+1)(L-1)``. Every experiment in a bucket has the
same ``T``, so the layout is fixed when the step compiles. The last segment
is padded by repeating its final time, which diffrax accepts, so every
segment has the same length and all of them solve as one ``vmap``.

The **assembled** trajectory takes each index from the segment that starts
at or before it: a node's own sample is scored at the node value. The result
is an ordinary ``[N, T, S]`` state, so ``state_to_output``, the loss, the
horizon mask and trajectory penalties run unchanged.

Nodes
-----
A node is the full state, measured or not. When a segmented phase starts,
the bucket is solved once at the current predictors (``throw=False``) and
each node is read off that solve at its time, so the segments start joined
and the first loss equals the single-shooting one. An experiment whose solve
fails falls back to ``y0`` at every node.

The optimiser trains ``z`` with ``node = scale * z``. ``scale`` is, per
experiment and state, the largest absolute value along that starting solve,
floored at a thousandth of the bucket's largest for that state, and 1 for a
state that is zero everywhere. Adam then moves each node by a fraction of
its own size whatever its units. The same scale divides the gaps, so the
continuity term ``weight * mean((gap / scale)^2)`` has no units.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from jaxhybridmodels.data import BucketPayload, Dataset
from jaxhybridmodels.solver import SolverConfig, simulation_state
from jaxhybridmodels.training.kernels import apply_length_mask

__all__ = [
    "Nodes",
    "SegmentLayout",
    "ShootingKernels",
    "bucket_nodes",
    "build_checked_simulation",
    "build_shooting_kernels",
    "initial_nodes",
    "segment_layout",
    "shoot_bucket",
]


class SegmentLayout(NamedTuple):
    """Where each segment starts and which segment supplies each sample.

    ``segments`` can be fewer than asked for: a bucket with few samples
    cannot hold that many segments of at least two samples each.
    ``samples[k, j]`` is the union index of segment ``k``'s ``j``-th sample,
    clamped at ``T - 1`` to pad the last segment. ``segment[i]`` and
    ``position[i]`` say which segment supplies union index ``i``, and where in
    it. ``boundaries`` are the union indices of the nodes.
    """

    segments: int
    length: int
    samples: np.ndarray
    segment: np.ndarray
    position: np.ndarray
    boundaries: np.ndarray


def segment_layout(n_times: int, segments: int) -> SegmentLayout:
    """Split ``n_times`` samples into at most ``segments`` overlapping segments.

    Each segment gets ``L = ceil((T-1)/K) + 1`` samples, and the count is
    then recomputed from ``L``, so the last segment always starts before the
    final sample and holds at least two distinct times. One sample, or one
    segment, is single shooting.
    """
    if n_times < 2 or segments < 2:
        index = np.arange(n_times)[None, :]
        zeros = np.zeros(n_times, dtype=np.int32)
        return SegmentLayout(
            1, n_times, index, zeros, np.arange(n_times), np.zeros(0, dtype=np.int32)
        )
    steps = n_times - 1
    length = math.ceil(steps / min(segments, steps)) + 1
    count = math.ceil(steps / (length - 1))
    starts = np.arange(count) * (length - 1)
    index = np.minimum(starts[:, None] + np.arange(length)[None, :], steps)
    segment = np.minimum(np.arange(n_times) // (length - 1), count - 1)
    position = np.arange(n_times) - segment * (length - 1)
    return SegmentLayout(count, length, index, segment, position, starts[1:])


def shoot_bucket(
    predictors: Any,
    bp: BucketPayload,
    nodes: Array,
    layout: SegmentLayout,
    *,
    simulate_fn: Callable[..., Any],
    solver: SolverConfig,
) -> tuple[Array, Array]:
    """Solve every segment of every experiment and assemble the trajectory.

    ``nodes`` are physical, ``[N, K-1, S]``. Returns the assembled state
    ``[N, T, S]`` and the gaps ``[N, K-1, S]``, each segment's end state minus
    the next node.
    """
    samples = jnp.asarray(layout.samples)
    segment, position = jnp.asarray(layout.segment), jnp.asarray(layout.position)

    def per_experiment(ts: Array, covariates: Any, y0: Array, own: Array) -> tuple[Array, Array]:
        starts = jnp.concatenate([y0[None, :], own.astype(y0.dtype)], axis=0)

        def one_segment(seg_ts: Array, start: Array) -> Array:
            return simulation_state(simulate_fn(predictors, seg_ts, covariates, start, solver))

        states = jax.vmap(one_segment)(ts[samples], starts)  # [K, L, S]
        gaps = states[:-1, layout.length - 1] - own
        return states[segment, position], gaps

    return jax.vmap(per_experiment)(bp.ts, bp.covariates, bp.y0, nodes)


class ShootingKernels(NamedTuple):
    """The jitted pieces of one segment count.

    ``step(predictors, z, scale, bp, horizon, weight)`` returns
    ``((total, (data, continuity)), (predictor_grads, z_grads))``. ``value``
    takes the same arguments and returns ``(data, continuity)`` with no
    backward pass. ``data`` is the loss plus any trajectory penalty on the
    assembled trajectory, and ``continuity`` the weighted gap term.
    """

    segments: int
    step: Callable[..., tuple[tuple[Array, tuple[Array, Array]], Any]]
    value: Callable[..., tuple[Array, Array]]


def build_shooting_kernels(
    *,
    segments: int,
    simulate_fn: Callable[..., Any],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
    trainable: Any,
    trajectory_penalty_fn: Callable[[Array, BucketPayload], Array] | None = None,
    trajectory_penalty_weight: float = 0.0,
) -> ShootingKernels:
    """Build the step and the forward-only value for ``segments`` segments.

    One trace per bucket shape, as for ``build_bucket_step``: the layout is
    read from the bucket's static ``T`` at trace time. ``horizon`` and the
    continuity ``weight`` are traced, so a phase that changes either does
    not recompile.
    """

    def terms(
        predictors: Any, z: Array, scale: Array, bp: BucketPayload, weight: Array
    ) -> tuple[Array, Array]:
        layout = segment_layout(int(bp.ts.shape[1]), segments)
        full, gaps = shoot_bucket(
            predictors, bp, z * scale, layout, simulate_fn=simulate_fn, solver=solver
        )
        data = loss_fn(jax.vmap(state_to_output)(full), bp)
        if trajectory_penalty_fn is not None and trajectory_penalty_weight != 0.0:
            data = data + trajectory_penalty_weight * trajectory_penalty_fn(full, bp)
        if layout.segments == 1:
            return data, jnp.zeros_like(data)
        return data, weight * jnp.mean((gaps / scale) ** 2)

    def objective(
        diff: tuple[Any, Array], static: Any, scale: Array, bp: BucketPayload, weight: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        predictors = eqx.combine(diff[0], static)
        data, continuity = terms(predictors, diff[1], scale, bp, weight)
        return data + continuity, (data, continuity)

    grad_fn = eqx.filter_value_and_grad(objective, has_aux=True)

    @eqx.filter_jit
    def jitted_step(
        predictors: Any, z: Array, scale: Array, bp: BucketPayload, horizon: Array, weight: Array
    ) -> tuple[tuple[Array, tuple[Array, Array]], Any]:
        masked = apply_length_mask(bp, horizon)
        diff, static = eqx.partition(predictors, trainable)
        return grad_fn((diff, z), static, scale, masked, weight)

    @eqx.filter_jit
    def jitted_value(
        predictors: Any, z: Array, scale: Array, bp: BucketPayload, horizon: Array, weight: Array
    ) -> tuple[Array, Array]:
        return terms(predictors, z, scale, apply_length_mask(bp, horizon), weight)

    def step(
        predictors: Any,
        z: Array,
        scale: Array,
        bp: BucketPayload,
        horizon: ArrayLike = 1.0,
        weight: ArrayLike = 1.0,
    ) -> tuple[tuple[Array, tuple[Array, Array]], Any]:
        # Arrays, because filter_jit treats a Python float as static.
        return jitted_step(
            predictors, z, scale, bp, jnp.asarray(horizon, float), jnp.asarray(weight, float)
        )

    def value(
        predictors: Any,
        z: Array,
        scale: Array,
        bp: BucketPayload,
        horizon: ArrayLike = 1.0,
        weight: ArrayLike = 1.0,
    ) -> tuple[Array, Array]:
        return jitted_value(
            predictors, z, scale, bp, jnp.asarray(horizon, float), jnp.asarray(weight, float)
        )

    return ShootingKernels(segments, step, value)


def build_checked_simulation(
    *, simulate_fn: Callable[..., Any], solver: SolverConfig
) -> Callable[[Any, BucketPayload], tuple[Array, Array]]:
    """``simulate(predictors, bp) -> (states [N, T, S], solved [N])``, never raising.

    The nodes start on this solve. It runs with ``throw=False``, so an
    experiment whose solve fails comes back unsolved instead of ending the
    run, and its nodes fall back to ``y0``.
    """
    quiet = dataclasses.replace(solver, throw=False)

    @eqx.filter_jit
    def simulate(predictors: Any, bp: BucketPayload) -> tuple[Array, Array]:
        def one(ts: Array, covariates: Any, y0: Array) -> Array:
            return simulation_state(simulate_fn(predictors, ts, covariates, y0, quiet))

        states = jax.vmap(one)(bp.ts, bp.covariates, bp.y0)
        return states, jnp.all(jnp.isfinite(states), axis=(1, 2))

    return simulate


class Nodes(NamedTuple):
    """Per-bucket node latents and their scales: ``z [N, K-1, S]``, ``scale [N, 1, S]``."""

    z: tuple[Array, ...]
    scale: tuple[Array, ...]


def bucket_nodes(
    predictors: Any,
    bp: BucketPayload,
    segments: int,
    simulate: Callable[[Any, BucketPayload], tuple[Array, Array]],
) -> tuple[Array, Array]:
    """One bucket's ``(z, scale)``, on a solve at ``predictors`` or at ``y0`` where it fails."""
    states, solved = simulate(predictors, bp)
    y0 = bp.y0.astype(states.dtype)
    filled = jnp.where(solved[:, None, None], states, y0[:, None, :])
    own = jnp.max(jnp.abs(filled), axis=1)  # [N, S]
    floor = 1e-3 * jnp.max(own, axis=0, keepdims=True)
    scale = jnp.maximum(own, floor)
    scale = jnp.where(scale > 0, scale, 1.0)[:, None, :]
    layout = segment_layout(int(bp.ts.shape[1]), segments)
    nodes = filled[:, jnp.asarray(layout.boundaries, dtype=jnp.int32)]
    return nodes / scale, scale


def initial_nodes(
    predictors: Any,
    dataset: Dataset,
    segments: int,
    simulate: Callable[[Any, BucketPayload], tuple[Array, Array]],
) -> Nodes:
    """Every bucket's nodes, from :func:`bucket_nodes`."""
    pairs = [bucket_nodes(predictors, bp, segments, simulate) for bp in dataset.bucket_payloads]
    return Nodes(tuple(z for z, _ in pairs), tuple(scale for _, scale in pairs))
