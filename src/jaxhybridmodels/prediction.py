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

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from jaxhybridmodels.data import BucketPayload, Dataset, _require_dataset
from jaxhybridmodels.solver import SolverConfig, simulation_state
from jaxhybridmodels.training.kernels import predict_bucket_obs


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
        A property of the model, passed explicitly.
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
        stored on the ``Dataset``.

    Returns
    -------
    tuple[Float[Array, "N T D"], ...]
        One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order.
    """
    _require_dataset(dataset, "predict_dataset")
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


def evaluate_predictor(predictor: Any, inputs: Mapping[str, ArrayLike]) -> Array:
    """Evaluate a predictor at named inputs, vectorised over array-valued ones.

    The inputs are broadcast against each other, so scalars, a 1-D sweep,
    or a 2-D grid built from ``x[:, None]`` and ``y[None, :]`` all work.
    The predictor is called once per broadcast point with a dict of scalars,
    the way a ``simulate_fn`` calls it, and the result has shape
    ``broadcast_shape + output_shape``. A one-output ``BoundedPredictor``
    has output shape ``()``, so a sweep over 50 temperatures gives ``[50]``.

    Examples
    --------
    ::

        hm.evaluate_predictor(predictors["rate"], {"T": 300.0})  # shape ()
        hm.evaluate_predictor(predictors["rate"], {"T": temps, "pH": 6.5})  # [len(temps)]

    Returns
    -------
    Array
        Call ``float(...)`` on a scalar result for a Python number.
    """
    arrays = {
        key: jnp.asarray(value, dtype=jnp.result_type(jnp.asarray(value).dtype, float))
        for key, value in inputs.items()
    }
    shape = jnp.broadcast_shapes(*(a.shape for a in arrays.values())) if arrays else ()
    if not shape:
        return jnp.asarray(predictor(arrays))
    flat = {key: jnp.broadcast_to(a, shape).reshape(-1) for key, a in arrays.items()}
    out = jax.vmap(predictor)(flat)
    return out.reshape(*shape, *out.shape[1:])


@eqx.filter_jit
def _predict_dense_bucket(
    predictors: Any,
    bp: BucketPayload,
    dense_ts: Array,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> Array:
    """Preserve the initial time with a reusable compiled dense-grid kernel."""

    def per_experiment(ts, start, covariates, y0):
        integration_ts = jnp.concatenate((start[None], ts))
        state = simulation_state(simulate_fn(predictors, integration_ts, covariates, y0, solver))
        return state_to_output(state)[1:]

    return jax.vmap(per_experiment)(dense_ts, bp.ts[:, 0], bp.covariates, bp.y0)


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
    _require_dataset(dataset, "predict_dense")
    if ts_grid is None and n_points < 1:
        raise ValueError("n_points must be at least 1")
    if ts_grid is not None:
        grid = jnp.asarray(ts_grid)
        if grid.ndim != 1 or grid.size == 0 or not bool(jnp.all(jnp.isfinite(grid))):
            raise ValueError("ts_grid must be a nonempty finite rank-1 array")
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
        if bool(jnp.any(dense_ts[:, 0] < bp.ts[:, 0])):
            raise ValueError("Dense prediction times must not precede the initial state time")
        if dense_ts.shape[1] == 0 or bool(jnp.any(jnp.diff(dense_ts, axis=1) <= 0)):
            raise ValueError("Dense prediction times must be nonempty and strictly increasing")

        out.append(
            _predict_dense_bucket(
                predictors,
                bp,
                dense_ts,
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

    ``members`` is a sequence of predictor pytrees, each one the ``predictors``
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
    _require_dataset(dataset, "ensemble_predictions")
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
    return tuple(jnp.mean(jnp.stack([p[b] for p in per_member]), axis=0) for b in range(n_buckets))
