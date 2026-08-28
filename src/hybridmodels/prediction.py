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

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from hybridmodels.data import BucketPayload, Dataset
from hybridmodels.solver import SolverConfig
from hybridmodels.training.kernels import predict_bucket_obs


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

    ``predict_bucket_obs`` runs ``simulate_fn`` for one experiment, giving a
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
        Pure ``[T, S] -> [T, D]`` map from full state to observed channels.
        A property of the model, passed explicitly (ADR-0008).
    solver
        ``SolverConfig``. All its fields are static, so it enters the
        compiled kernel as configuration rather than as data.

    Returns
    -------
    Float[Array, "N T D"]
        Predicted output channels for every experiment in the bucket.
    """
    return predict_bucket_obs(
        predictors,
        bp,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
    )


def predict_dataset(
    predictors: Any,
    dataset: Dataset,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> tuple[Float[Array, "N T D"], ...]:
    """Run ``predict_bucket`` over every bucket in ``dataset`` and return the stack tuple.

    Each bucket shape compiles ``predict_bucket`` exactly once. The result
    is a tuple rather than one array, because buckets differ precisely in
    ``T`` and cannot be stacked.

    Parameters
    ----------
    state_to_output
        Pure mapping ``[T, S] -> [T, D]`` from full simulator state to the
        observed channels. A property of the model, passed here rather than
        stored on the ``Dataset`` (ADR-0008).

    Returns
    -------
    tuple[Float[Array, "N T D"], ...]
        One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order.
    """
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


def evaluate_predictor(predictor: Any, covariates: dict[str, float]) -> float:
    """Evaluate a scalar-valued predictor at named inputs, as a Python float.

    Shortcut for the recovered-physics readout every example writes by hand
    (``float(predictor({"k": jnp.asarray(v)}).reshape(()))``). Takes the
    predictor's named inputs as plain Python floats, calls it, and flattens
    the scalar result to a ``float``.
    """
    x = {k: jnp.asarray(v) for k, v in covariates.items()}
    return float(predictor(x).reshape(()))


def predict_dense(
    predictors: Any,
    dataset: Dataset,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    ts_grid: Array | None = None,
    n_points: int = 100,
) -> tuple[Float[Array, "N T_d D"], ...]:
    """Evaluate a trained model on a dense time grid, one array per bucket.

    ``predict_dataset`` returns predictions only at the *measured*
    timestamps. For smooth trajectory plots or dense evaluation you usually
    want more points than that. This builds a fine grid per experiment and
    reuses the same compiled forward pass, so no new kernel or dependency
    is needed (the diffraxtra ``VectorizedDenseInterpolation`` equivalent,
    folded in ~20 lines).

    Parameters
    ----------
    ts_grid
        Optional shared grid ``[T_d]`` to evaluate every experiment on. If
        ``None``, each experiment gets its own ``linspace`` from its first
        to its last measured time with ``n_points`` points.
    n_points
        Points per experiment when ``ts_grid`` is ``None``. Ignored
        otherwise.

    Returns
    -------
    tuple[Float[Array, "N T_d D"], ...]
        One ``[N, T_d, D]`` array per bucket, in bucket-payload order.
    """
    out = []
    for bp in dataset.bucket_payloads:
        n = int(bp.ts.shape[0])
        if ts_grid is not None:
            grid = jnp.asarray(ts_grid, dtype=bp.ts.dtype)
            dense_ts = jnp.broadcast_to(grid, (n, int(grid.shape[0])))
        else:
            dense_ts = jnp.stack(
                [jnp.linspace(bp.ts[i, 0], bp.ts[i, -1], n_points) for i in range(n)]
            )
        dense_bp = bp._replace(ts=dense_ts)
        out.append(
            predict_bucket(
                predictors,
                dense_bp,
                simulate_fn=simulate_fn,
                state_to_output=state_to_output,
                solver=solver,
            )
        )
    return tuple(out)


def ensemble_predictions(
    members: Sequence[Any],
    dataset: Dataset,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> tuple[Float[Array, "N T D"], ...]:
    """Average per-bucket predictions across an ensemble of models.

    ``members`` is a sequence of predictor pytrees — each the ``predictors``
    argument you would pass to :func:`predict_dataset` alone. Each member is
    run forward and the per-bucket predictions are averaged, so the result
    is the same shape as a single ``predict_dataset`` return.

    Parameters
    ----------
    members
        Non-empty sequence of predictor pytrees. Every member must be
        compatible with the same ``simulate_fn``.

    Returns
    -------
    tuple[Float[Array, "N T D"], ...]
        The member-mean prediction per bucket, in bucket-payload order.
    """
    if not members:
        raise ValueError("ensemble_predictions requires at least one member")
    per_member = tuple(
        predict_dataset(
            m,
            dataset,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=solver,
        )
        for m in members
    )
    n_buckets = len(dataset.bucket_payloads)
    return tuple(
        jnp.mean(jnp.stack([p[b] for p in per_member]), axis=0) for b in range(n_buckets)
    )
