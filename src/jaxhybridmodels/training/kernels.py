"""Public gradient-kernel builders for writing your own training loop.

The stock trainers (:func:`jaxhybridmodels.training.optax.train_with_optax`,
:func:`jaxhybridmodels.training.evosax.train_with_evosax`) are assembled from
these pieces. They are public so that a user who wants a custom loop —
different accumulation, a custom regulariser, per-bucket weighting, a
custom schedule — can compose the same kernels the framework uses, instead
of forking the trainer.

A training *step* in this library is one full pass over every bucket,
accumulating per-bucket gradients, then a single optimiser update. The
kernels here are the compiled pieces that step makes:

- :func:`build_bucket_step` — one jitted ``bucket_step(predictors, bp,
  fraction) -> (loss, grads)`` per bucket shape. The **bound** penalty is
  *not* here; a configured **trajectory** penalty is, added to the loss
  inside the same forward pass (one charge per bucket, since each bucket's
  trajectories differ).
- :func:`build_score_bucket` — a forward-only scorer (no backward pass),
  for sweeps like the tournament where gradients would be wasted.
- :func:`build_penalty_step` — one jitted ``penalty_step(predictors,
  weight, points=()) -> (penalty, weighted_grads)`` per step, evaluated
  outside the bucket loop because it reads only the predictors tree and
  its point arrays. This is the bound penalty's slot: once per step, not
  once per bucket, at the per-leaf point arrays selected host-side by
  :func:`jaxhybridmodels.penalties.select_penalty_points`.
- :func:`build_apply_update` — one jitted ``apply_update(predictors,
  grads, opt_state)``; the single optimiser update per step.

``bucket_step`` is ``eqx.filter_jit``-decorated and closed over the
``trainable`` mask, so ``eqx.partition`` and ``eqx.filter_value_and_grad``
never leak into the caller. The pieces compose over any pytree shape that
``eqx.partition`` can walk.

One trace per bucket shape (R-T5): the caller loops over
``dataset.bucket_payloads`` in Python, calling ``bucket_step`` once per
bucket, so each distinct shape compiles exactly once. Keep that Python
loop *outside* any jit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from jaxhybridmodels.data import BucketPayload
from jaxhybridmodels.solver import SolverConfig


def apply_length_mask(bp: BucketPayload, length_mask_fraction: Array) -> BucketPayload:
    """Narrow the bucket's mask to the first ``fraction`` of its timestamps.

    This masks the **loss**, never the integration: the solver still runs the
    full trajectory, and only the leading prefix of the observation times is
    scored. That is what keeps a long-horizon divergence from drowning the
    gradient early in a run.

    ``length_mask_fraction`` stays traced rather than becoming a Python
    branch, so a phase that changes the fraction costs no recompile.

    The cutoff is clamped at 1. A fraction small enough to floor to zero
    would otherwise give an all-false mask, and every loss here divides by a
    count clamped at 1, so the step would silently score nothing.
    """
    T = bp.ts.shape[1]
    cutoff = jnp.maximum(
        jnp.ceil(jnp.float32(T) * length_mask_fraction).astype(jnp.int32),
        jnp.int32(1),
    )
    sched_mask = (jnp.arange(T) < cutoff)[None, :, None]
    mask = bp.mask & sched_mask
    return bp._replace(mask=mask, n_obs=mask.sum().astype(bp.n_obs.dtype))


def simulate_bucket(
    predictors: Any,
    bp: BucketPayload,
    *,
    simulate_fn: Callable[..., Array],
    solver: SolverConfig,
) -> Array:
    """Simulate every experiment in the bucket, returning the **full state**.

    ``[N, T, S]`` — the state *before* ``state_to_output``. Needed by
    trajectory-aware penalties, which read extra ODE components (see
    :mod:`jaxhybridmodels.penalties`). Uncompiled, like :func:`predict_bucket_obs`;
    each caller traces it into its own kernel.
    """

    def per_experiment(ts: Array, covariates: dict[str, Array], y0: Array) -> Array:
        return simulate_fn(predictors, ts, covariates, y0, solver)

    return jax.vmap(per_experiment, in_axes=(0, 0, 0))(bp.ts, bp.covariates, bp.y0)


def predict_bucket_obs(
    predictors: Any,
    bp: BucketPayload,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> Array:
    """Simulate every experiment in the bucket and project to ``[N, T, D]``.

    The uncompiled shared core behind both :func:`predict_bucket
    <jaxhybridmodels.prediction.predict_bucket>` and the training kernels.
    Each caller wraps it in its own ``eqx.filter_jit``, which is what keeps
    the training and prediction jit caches separate (R-J1): this body is
    traced into whichever kernel calls it.
    """
    full = simulate_bucket(predictors, bp, simulate_fn=simulate_fn, solver=solver)
    return jax.vmap(state_to_output)(full)


def build_bucket_step(
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
    trainable: Any,
    trajectory_penalty_fn: Callable[[Array, BucketPayload], Array] | None = None,
    trajectory_penalty_weight: float = 0.0,
) -> Callable[[Any, BucketPayload, Array], tuple[Array, Any]]:
    """Return a jitted ``bucket_step(predictors, bp, fraction) -> (loss, grads)``.

    One trace per bucket shape (R-T5). Takes no ``opt_state``: the optimiser
    update lives in a separate jitted ``apply_update``, and the **bound**
    penalty is charged once per step by
    :func:`build_penalty_step`, outside the bucket loop — this kernel
    charges the data loss (plus any configured trajectory penalty, below).

    ``trajectory_penalty_fn`` is the trajectory-aware counterpart, and the
    exception to that: it reads the **full state** ``[N, T, S]`` (penalty
    accumulators included) and the bucket payload, and returns a scalar
    added to the data loss inside the same forward pass. Because it reads
    per-bucket trajectories, it is charged **per bucket** — every bucket
    in a step contributes its own trajectory penalty — unlike the bound
    penalty's once-per-step charge. When it is ``None`` (the default) this
    kernel is byte-for-byte what it was before — no extra simulate, no
    behaviour change.
    """

    def loss_eval(
        diff_predictors: Any,
        static_predictors: Any,
        bp_masked: BucketPayload,
    ) -> Array:
        # ``eqx.combine`` walks any pytree shape, so the container is never
        # inspected before the recombined tree goes to simulate_fn.
        predictors = eqx.combine(diff_predictors, static_predictors)
        full_state = simulate_bucket(
            predictors,
            bp_masked,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        pred_obs = jax.vmap(state_to_output)(full_state)
        loss = loss_fn(pred_obs, bp_masked)
        if trajectory_penalty_fn is not None and trajectory_penalty_weight != 0.0:
            loss = loss + trajectory_penalty_weight * trajectory_penalty_fn(full_state, bp_masked)
        return loss

    grad_fn = eqx.filter_value_and_grad(loss_eval)

    @eqx.filter_jit
    def bucket_step(
        predictors: Any, bp: BucketPayload, length_mask_fraction: Array
    ) -> tuple[Array, Any]:
        bp_masked = apply_length_mask(bp, length_mask_fraction)
        diff_part, static_part = eqx.partition(predictors, trainable)
        return grad_fn(diff_part, static_part, bp_masked)

    return bucket_step


def build_score_bucket(
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
    trajectory_penalty_fn: Callable[[Array, BucketPayload], Array] | None = None,
    trajectory_penalty_weight: float = 0.0,
) -> Callable[[Any, BucketPayload, Array], Array]:
    """Return a forward-only ``score_bucket(predictors, bp, fraction) -> loss``.

    Scoring through ``bucket_step`` would run a full ``value_and_grad`` and
    discard the gradients, roughly tripling the cost of a scoring sweep.
    This costs one extra compile per bucket shape and pays for itself above
    two attempts. The returned score includes the data loss and, when
    configured, the trajectory penalty; the bound penalty remains
    outside this per-bucket scorer.
    """

    @eqx.filter_jit
    def score_bucket(predictors: Any, bp: BucketPayload, length_mask_fraction: Array) -> Array:
        bp_masked = apply_length_mask(bp, length_mask_fraction)
        full_state = simulate_bucket(
            predictors, bp_masked, simulate_fn=simulate_fn, solver=solver
        )
        pred_obs = jax.vmap(state_to_output)(full_state)
        loss = loss_fn(pred_obs, bp_masked)
        if trajectory_penalty_fn is not None and trajectory_penalty_weight != 0.0:
            loss = loss + trajectory_penalty_weight * trajectory_penalty_fn(
                full_state, bp_masked
            )
        return loss

    return score_bucket


def build_penalty_step(
    *,
    penalty_fn: Callable[[Any, tuple[Array, ...]], Array],
    trainable: Any,
) -> Callable[[Any, Array, tuple[Array, ...]], tuple[Array, Any]]:
    """Return ``penalty_step(predictors, weight, points=()) -> (penalty, weighted_grads)``.

    Evaluated once per training step, outside the bucket loop: the penalty
    reads only the predictors tree and its point arrays, so computing it
    inside ``bucket_step`` would repeat one identical evaluation per
    bucket.

    ``points`` are the per-leaf point arrays the penalty is evaluated at —
    the output of :func:`jaxhybridmodels.penalties.select_penalty_points` —
    passed as traced arrays, so a shape change (a phase boundary) retraces
    this small kernel and nothing else. The default ``()`` suits a custom
    ``penalty_fn`` that ignores points.

    ``penalty_fn`` is the regulariser, required here and defaulted to
    :func:`jaxhybridmodels.penalties.bound_penalty` by the stock trainers.
    Passing a different callable (weight decay on inner weights, a
    monotonicity term, ...) is how a custom regulariser composes with the
    loop. It must take ``(predictors, points)`` and return a scalar; a
    custom term that does not need points just ignores them.

    Returns the gradient of ``weight * penalty``, to add straight onto the
    averaged data gradient. ``penalty`` comes back unweighted, since that
    is what gets reported.
    """

    @eqx.filter_jit
    def penalty_step(
        predictors: Any, weight: Array, points: tuple[Array, ...] = ()
    ) -> tuple[Array, Any]:
        diff_part, static_part = eqx.partition(predictors, trainable)

        def weighted(diff: Any, static: Any, w: Array) -> Array:
            preds = eqx.combine(diff, static)
            return w * penalty_fn(preds, points)

        weighted_value, grads = eqx.filter_value_and_grad(weighted)(diff_part, static_part, weight)
        # Report the raw penalty; the weight is a scheduling choice and
        # folding it into the number would make phases incomparable.
        unweighted = jnp.where(weight > 0.0, weighted_value / jnp.maximum(weight, 1e-30), 0.0)
        return unweighted, grads

    return penalty_step


def build_apply_update(
    optimizer: optax.GradientTransformation, trainable: Any
) -> Callable[[Any, Any, Any], tuple[Any, Any]]:
    """Return a jitted ``apply_update(predictors, grads, opt_state)``.

    The single optimiser update per training step. Build once per
    optimiser; reuse its returned state across steps, rebuilding only at a
    reset/phase boundary.
    """

    @eqx.filter_jit
    def apply_update(predictors: Any, grads: Any, opt_state: Any) -> tuple[Any, Any]:
        params = eqx.filter(predictors, trainable)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_predictors = eqx.apply_updates(predictors, updates)
        return new_predictors, new_opt_state

    return apply_update
