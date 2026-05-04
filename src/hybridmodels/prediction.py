"""Prediction entry points.

Two layers, separated so the JIT cache stays predictable:

``predict_bucket``
    The jitted kernel. Vmaps the user's ``simulate_fn`` across a
    bucket's ``N`` axis, then applies ``state_to_output`` per experiment
    to project the full simulator state ``[T, S]`` onto the observed
    channels ``[T, D]``. It is jitted independently from any training
    kernel so prediction-time graphs do not collide with training-time
    graphs in the cache.

``predict_dataset``
    A plain Python ``for`` loop over ``dataset.bucket_payloads`` that
    dispatches to ``predict_bucket`` once per bucket. The dispatch loop
    lives outside the jitted region, so each *bucket shape* compiles
    exactly once and re-using the same shape (e.g. across epochs) is
    free.

Both entry points take ``predictors`` as a ``PyTree[eqx.Module]`` of any
container shape (the convention is a tuple of ``BoundedPredictor``
leaves, but a dict, NamedTuple, or bare Module is equally valid). This
module never inspects the pytree structure — it is forwarded to the
user's ``simulate_fn`` unchanged.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, Float

from hybridmodels.data import BucketPayload, Dataset
from hybridmodels.solver import SolverConfig


@eqx.filter_jit
def predict_bucket(
    predictors: Any,
    bp: BucketPayload,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> Float[Array, "N T D"]:
    """Vmap ``simulate_fn`` over the bucket's ``N`` axis and project to observed channels.

    The inner ``_per_experiment`` runs the user's ``simulate_fn`` once for
    one experiment to produce a full state trajectory ``[T, S]`` and then
    projects to the observed channels ``[T, D]`` via ``state_to_output``.
    ``jax.vmap`` lifts this over ``(ts, covariates, y0)`` along the ``N``
    axis to produce ``[N, T, D]``. ``predictors`` and ``solver`` are
    closed over (no vmap axis) — they are constant across the bucket.

    JIT caching: one compiled trace per bucket *shape*. The Python
    dispatch over ``dataset.bucket_payloads`` lives in
    ``predict_dataset``, never inside the jitted region — that boundary
    is what keeps the cache predictable.

    Parameters
    ----------
    predictors : PyTree[eqx.Module]
        Trainable component, typically a tuple of ``BoundedPredictor``
        leaves but accepted as any pytree shape. Forwarded to
        ``simulate_fn`` as-is; this module does not inspect the
        container.
    bp : BucketPayload
        One bucket; ``ts``, ``covariates``, ``y0`` are vmapped along ``N``.
    simulate_fn
        User-supplied integrator with signature
        ``(predictors, ts, covariates, y0, solver) -> [T, S]``.
    state_to_output
        Pure ``[T, S] -> [T, D]`` projector held on ``Dataset``.
    solver
        Static ``SolverConfig``.

    Returns
    -------
    Float[Array, "N T D"]
        Predicted output channels for every experiment in the bucket.
    """

    def _per_experiment(ts: Array, covariates: dict[str, Array], y0: Array) -> Array:
        full_state = simulate_fn(predictors, ts, covariates, y0, solver)
        return state_to_output(full_state)

    return jax.vmap(_per_experiment, in_axes=(0, 0, 0))(bp.ts, bp.covariates, bp.y0)


def predict_dataset(
    predictors: Any,
    dataset: Dataset,
    *,
    simulate_fn: Callable[..., Array],
    solver: SolverConfig,
) -> tuple[Float[Array, "N T D"], ...]:
    """Run ``predict_bucket`` over every bucket in ``dataset`` and return the stack tuple.

    The Python ``for`` loop over ``dataset.bucket_payloads`` is the dispatch
    driver: each bucket shape compiles ``predict_bucket`` exactly once.
    Returns a tuple aligned with ``dataset.bucket_payloads`` order, *not*
    a flat concatenation — each entry has its own ``[N_b, T_b, D]``
    shape and cannot be stacked into a single tensor (the buckets differ
    precisely in ``T``).

    Returns
    -------
    tuple[Float[Array, "N T D"], ...]
        One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order.
    """
    state_to_output = dataset.state_to_output
    return tuple(
        predict_bucket(
            predictors,
            bp,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=solver,
        )
        for bp in dataset.bucket_payloads
    )
