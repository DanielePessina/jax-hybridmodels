"""Run a trained model forward on data, without any gradient machinery.

Two layers, kept apart so the compilation cache stays predictable.

``predict_bucket``
    The compiled kernel. ``jax.vmap`` runs the user's ``simulate_fn`` for
    every experiment in a bucket at once along the leading ``N`` axis, then
    ``state_to_output`` maps each state ``[T, S]`` to observed channels
    ``[T, D]``. Compiled separately from any training kernel.

``predict_dataset``
    A Python ``for`` loop over ``dataset.bucket_payloads`` calling
    ``predict_bucket`` once per bucket. The loop stays outside the compiled
    region, so each distinct bucket shape compiles exactly once.

Both take ``predictors`` as a pytree of ``eqx.Module`` leaves in any
container shape, and forward it to ``simulate_fn`` unchanged.
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

    ``_per_experiment`` runs ``simulate_fn`` for one experiment, giving a
    state trajectory ``[T, S]``, and maps it to observed channels ``[T, D]``.
    ``jax.vmap`` lifts that over ``(ts, covariates, y0)`` along ``N``.
    ``predictors`` and ``solver`` are closed over with no vmap axis, being
    the same for every experiment in the bucket.

    One compiled kernel per bucket shape. Python dispatch over buckets lives
    in ``predict_dataset``, never inside the compiled region.

    Parameters
    ----------
    predictors : PyTree[eqx.Module]
        The trainable part of the model, typically a tuple of
        ``BoundedPredictor`` leaves. Forwarded to ``simulate_fn`` unchanged.
    bp : BucketPayload
        One bucket. Its ``ts``, ``covariates``, and ``y0`` are vmapped
        along ``N``.
    simulate_fn
        User-supplied integrator with signature
        ``(predictors, ts, covariates, y0, solver) -> [T, S]``.
    state_to_output
        Pure ``[T, S] -> [T, D]`` map from full state to observed channels,
        held on the ``Dataset``.
    solver
        ``SolverConfig``. All its fields are static, so it enters the
        compiled kernel as configuration rather than as data.

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

    Each bucket shape compiles ``predict_bucket`` exactly once. The result
    is a tuple rather than one array, because buckets differ precisely in
    ``T`` and cannot be stacked.

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
