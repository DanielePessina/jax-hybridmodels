"""Run trained models forward on data, without any gradient machinery.

``HybridModel.predict`` and :func:`predict_ensemble` are the entry points.
Both return a :class:`~jaxhybridmodels.PredictionResult` keyed by
experiment ID and channel name.

Underneath, one compiled kernel per bucket shape: ``jax.vmap`` runs the
user's ``simulate_fn`` for every experiment in a bucket at once along the
leading ``N`` axis, and ``state_to_output`` maps each state ``[T, S]`` to
observed channels ``[T, D]``. The loop over buckets stays in Python,
outside the compiled region, so each distinct bucket shape compiles once,
separately from any training kernel.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from jaxhybridmodels.data import BucketPayload, Data, Dataset, _per_experiment_arrays
from jaxhybridmodels.results import ExperimentPrediction, FitResult, PredictionResult
from jaxhybridmodels.solver import SolverConfig, simulation_state
from jaxhybridmodels.training.kernels import predict_bucket_obs

if TYPE_CHECKING:
    from jaxhybridmodels.model import HybridModel


@eqx.filter_jit
def _predict_bucket(
    predictors: Any,
    bp: BucketPayload,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> Float[Array, "N T D"]:
    """Predict one bucket on its measured grid, ``[N, T, D]``.

    ``predictors`` and ``solver`` are closed over with no vmap axis, being
    the same for every experiment in the bucket. Its own ``filter_jit``
    keeps the prediction cache apart from the training kernels.
    """
    return predict_bucket_obs(
        predictors,
        bp,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
    )


@eqx.filter_jit
def _predict_grid_bucket(
    predictors: Any,
    bp: BucketPayload,
    grid: Array,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> Float[Array, "N T_g D"]:
    """Predict one bucket on per-experiment grids ``[N, T_g]``.

    Each experiment still integrates from its own start, ``bp.ts[:, 0]``
    (its ``t0`` when set), so ``y0`` stays where it belongs. That start is
    prepended to the grid and its row dropped from the output.
    """

    def per_experiment(ts: Array, start: Array, covariates: Any, y0: Array) -> Array:
        integration_ts = jnp.concatenate((start[None], ts))
        state = simulation_state(simulate_fn(predictors, integration_ts, covariates, y0, solver))
        return state_to_output(state)[1:]

    return jax.vmap(per_experiment)(grid, bp.ts[:, 0], bp.covariates, bp.y0)


def _bucket_grid(bp: BucketPayload, ts: Array | None, n_points: int | None) -> Array | None:
    """The ``[N, T_g]`` prediction times for one bucket, or None for the measured grid."""
    if ts is not None:
        grid = jnp.asarray(ts, dtype=bp.ts.dtype)
        return jnp.broadcast_to(grid, (bp.ts.shape[0], grid.shape[0]))
    if n_points is not None:
        return jnp.linspace(bp.ts[:, 0], bp.ts[:, -1], n_points, axis=1)
    return None


def _check_grid_args(ts: Any, n_points: int | None, caller: str) -> Array | None:
    """Validate ``ts`` and ``n_points``; return ``ts`` as an array."""
    if ts is not None and n_points is not None:
        raise ValueError(f"{caller}: pass ts= or n_points=, not both")
    if n_points is not None and (isinstance(n_points, bool) or int(n_points) != n_points):
        raise TypeError(f"{caller}: n_points must be an int; got {n_points!r}")
    if n_points is not None and n_points < 1:
        raise ValueError(f"{caller}: n_points must be at least 1; got {n_points}")
    if ts is None:
        return None
    grid = jnp.asarray(ts)
    values = np.asarray(grid)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{caller}: ts must be a nonempty finite rank-1 array; got {values.shape}")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{caller}: ts must be strictly increasing")
    return grid


def _predict(
    model: HybridModel,
    dataset: Dataset,
    *,
    ts: Array | None,
    n_points: int | None,
    caller: str,
) -> PredictionResult:
    """Predict every experiment of ``dataset``, keyed by experiment ID.

    With neither ``ts`` nor ``n_points`` each experiment is predicted on its
    measured grid and keeps its observations. Otherwise every experiment
    gets the requested grid and no observations.
    """
    if not dataset.experiments:
        raise ValueError(
            f"{caller}: the dataset holds no source experiments; build it with make_dataset"
        )
    names = model.output_channel_names
    buckets: list[Array] = []
    grids: list[Array | None] = []
    for bp in dataset.bucket_payloads:
        grid = _bucket_grid(bp, ts, n_points)
        if grid is None:
            values = _predict_bucket(
                model.predictors,
                bp,
                simulate_fn=model.simulate_fn,
                state_to_output=model.state_to_output,
                solver=model.solver,
            )
            expected = bp.y_observed.shape
        else:
            if bool(jnp.any(grid[:, 0] < bp.ts[:, 0])):
                raise ValueError(
                    f"{caller}: ts starts at {float(grid[0, 0])}, before the start of an "
                    "experiment (its t0, or its first observation time when t0 is unset)"
                )
            values = _predict_grid_bucket(
                model.predictors,
                bp,
                grid,
                simulate_fn=model.simulate_fn,
                state_to_output=model.state_to_output,
                solver=model.solver,
            )
            expected = (*grid.shape, len(names))
        if values.shape != expected:
            # Re-trace one row abstractly to say which function is at fault;
            # the batched shape alone cannot tell.
            model._check_shapes(bp, caller)
            raise ValueError(
                f"{caller}: bucket predictions have shape {values.shape}; expected "
                f"[N, T, D] = {tuple(expected)}"
            )
        buckets.append(values)
        grids.append(grid)

    # make_dataset groups experiments by union-grid length, keeping input
    # order inside each group. Walking the experiments in input order and
    # counting rows per length recovers each one's bucket row.
    bucket_of_length = {bp.ts.shape[1]: i for i, bp in enumerate(dataset.bucket_payloads)}
    next_row: dict[int, int] = {}
    experiments: dict[str, ExperimentPrediction] = {}
    for exp in dataset.experiments:
        arrays = _per_experiment_arrays(exp, names)
        length = arrays.ts.shape[0]
        row = next_row.get(length, 0)
        next_row[length] = row + 1
        b = bucket_of_length[length]
        outputs = dict(zip(names, buckets[b][row].T, strict=True))
        grid = grids[b]
        if grid is None:
            observed = jnp.where(arrays.mask, arrays.y_observed, jnp.nan)
            experiments[exp.exp_id] = ExperimentPrediction(
                times=arrays.ts,
                outputs=outputs,
                observed=dict(zip(names, observed.T, strict=True)),
                observation_mask=dict(zip(names, arrays.mask.T, strict=True)),
            )
        else:
            experiments[exp.exp_id] = ExperimentPrediction(times=grid[row], outputs=outputs)
    return PredictionResult(experiments, tuple(buckets))


def predict_ensemble(
    models: Sequence[HybridModel | FitResult],
    data: Data,
    *,
    ts: ArrayLike | None = None,
    n_points: int | None = None,
) -> PredictionResult:
    """Predict with every model and return the member mean, with the spread.

    ``outputs`` (and ``buckets``) hold the mean across members, so
    ``compute_metrics`` scores the ensemble mean. Each experiment's ``std``
    holds the per-channel standard deviation across members (population,
    ``ddof=0``; zero for one member), and ``members`` holds each model's
    own :class:`~jaxhybridmodels.PredictionResult`.

    ``models`` holds ``HybridModel``\ s or ``FitResult``\ s, so the list
    ``fit_ensemble`` returns can be passed as is. They must share
    ``output_channel_names``. ``data``, ``ts`` and ``n_points`` are as for
    :meth:`HybridModel.predict`.
    """
    models = [_member_model(m, i) for i, m in enumerate(models)]
    if not models:
        raise ValueError("predict_ensemble: pass at least one model")
    names = models[0].output_channel_names
    for i, model in enumerate(models[1:], start=1):
        if model.output_channel_names != names:
            raise ValueError(
                f"predict_ensemble: model {i} has channels {model.output_channel_names}, "
                f"model 0 has {names}; every member must predict the same channels"
            )
    dataset = models[0]._dataset(data, "predict_ensemble")
    grid = _check_grid_args(ts, n_points, "predict_ensemble")
    members = tuple(
        _predict(m, dataset, ts=grid, n_points=n_points, caller="predict_ensemble") for m in models
    )
    experiments: dict[str, ExperimentPrediction] = {}
    for exp_id, first in members[0].items():
        stacked = {name: jnp.stack([m[exp_id].outputs[name] for m in members]) for name in names}
        experiments[exp_id] = ExperimentPrediction(
            times=first.times,
            outputs={name: jnp.mean(values, axis=0) for name, values in stacked.items()},
            observed=first.observed,
            observation_mask=first.observation_mask,
            std={name: jnp.std(values, axis=0) for name, values in stacked.items()},
        )
    buckets = tuple(
        jnp.mean(jnp.stack([m.buckets[b] for m in members]), axis=0)
        for b in range(len(dataset.bucket_payloads))
    )
    return PredictionResult(experiments, buckets, members)


def _member_model(member: Any, index: int) -> HybridModel:
    """A ``predict_ensemble`` member as a model, unwrapping a ``FitResult``."""
    # model.py imports this module, so HybridModel is imported at call time.
    from jaxhybridmodels.model import HybridModel

    if isinstance(member, FitResult):
        return member.model
    if isinstance(member, HybridModel):
        return member
    raise TypeError(
        f"predict_ensemble: member {index} is a {type(member).__name__}; pass "
        "HybridModels or the FitResults from fit_ensemble"
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
