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

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
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


@contextmanager
def _note_user_code(
    caller: str, predictors: Any, covariates: Mapping[str, Any], y0_shape: tuple[int, ...]
) -> Iterator[None]:
    """Add what the user's functions were given to an exception they raised.

    A typo in ``simulate_fn`` surfaces as a bare ``KeyError: 'K'`` from deep
    inside a trace. The note lists the names that exist, without changing the
    exception's type or message.
    """
    try:
        yield
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        names = list(predictors) if isinstance(predictors, Mapping) else type(predictors).__name__
        exc.add_note(
            f"While {caller} ran simulate_fn and state_to_output: predictors are {names}, "
            f"covariates are {sorted(covariates)}, and y0 has shape {y0_shape}."
        )
        raise


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
            with _note_user_code(caller, model.predictors, bp.covariates, bp.y0.shape[1:]):
                values = _predict_bucket(
                    model.predictors,
                    bp,
                    simulate_fn=model.simulate_fn,
                    state_to_output=model.state_to_output,
                    solver=model.solver,
                )
            expected = bp.observed.shape
        else:
            if bool(jnp.any(grid[:, 0] < bp.ts[:, 0])):
                raise ValueError(
                    f"{caller}: ts starts at {float(grid[0, 0])}, before the start of an "
                    "experiment (its t0, or its first observation time when t0 is unset)"
                )
            with _note_user_code(caller, model.predictors, bp.covariates, bp.y0.shape[1:]):
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

    experiments: dict[str, ExperimentPrediction] = {}
    for exp, (b, row) in zip(dataset.experiments, dataset._locations, strict=True):
        arrays = _per_experiment_arrays(exp, names)
        values = buckets[b][row]
        grid = grids[b]
        if grid is None:
            # Under make_dataset(max_buckets=) the row holds added times too;
            # the experiment reports only its own.
            bucket_ts = np.asarray(dataset.bucket_payloads[b].ts[row])
            if bucket_ts.shape[0] != arrays.ts.shape[0]:
                values = values[np.searchsorted(bucket_ts, np.asarray(arrays.ts, bucket_ts.dtype))]
            observed = jnp.where(arrays.mask, arrays.observed, jnp.nan)
            experiments[exp.exp_id] = ExperimentPrediction(
                ts=arrays.ts,
                outputs=dict(zip(names, values.T, strict=True)),
                observed=dict(zip(names, observed.T, strict=True)),
                mask=dict(zip(names, arrays.mask.T, strict=True)),
            )
        else:
            experiments[exp.exp_id] = ExperimentPrediction(
                ts=grid[row], outputs=dict(zip(names, values.T, strict=True))
            )
    return PredictionResult(experiments, tuple(buckets))


def predict_ensemble(
    models: Sequence[HybridModel | FitResult],
    data: Data,
    *,
    ts: ArrayLike | None = None,
    n_points: int | None = None,
    quantiles: Sequence[float] = (0.025, 0.975),
) -> PredictionResult:
    """Predict with every model and return the member mean, with the spread.

    ``outputs`` (and ``buckets``) hold the mean across members, so
    ``compute_metrics`` scores the ensemble mean. Each experiment's ``std``
    holds the per-channel standard deviation across members (population,
    ``ddof=0``; zero for one member), and ``members`` holds each model's
    own :class:`~jaxhybridmodels.PredictionResult`.

    ``quantiles`` adds those levels across members to each experiment, by
    default a central 95% band; ``()`` leaves them out. They interpolate
    linearly between members, so with few members a 95% band sits near the
    smallest and largest. A posterior predictive is usually skewed, so an
    interval from ``std`` alone can leave a bound's range.

    A member that predicts a non-finite value anywhere is left out of the
    statistics, with a warning, and still appears in ``members``.

    ``models`` holds ``HybridModel`` or ``FitResult`` objects, so the list
    ``fit_ensemble`` returns can be passed as is. They must share
    ``output_channel_names``. ``data``, ``ts`` and ``n_points`` are as for
    :meth:`HybridModel.predict`.
    """
    levels = _check_quantiles(quantiles, "predict_ensemble")
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
    # A member whose solve fails anywhere would turn the mean, spread and band
    # into NaN wherever it fails, so it is left out of the statistics.
    finite = [
        all(bool(jnp.all(jnp.isfinite(out))) for p in m.values() for out in p.outputs.values())
        for m in members
    ]
    if not all(finite):
        dropped = [i for i, ok in enumerate(finite) if not ok]
        warnings.warn(
            f"predict_ensemble: members {dropped} predict non-finite values on this grid and "
            "are left out of the mean, std and quantiles"
            if any(finite)
            else "predict_ensemble: every member predicts non-finite values on this grid",
            RuntimeWarning,
            stacklevel=2,
        )
    # With no finite member there is nothing to leave out, and the NaN result
    # says so.
    kept = [m for m, ok in zip(members, finite, strict=True) if ok] or list(members)
    experiments: dict[str, ExperimentPrediction] = {}
    for exp_id, first in members[0].items():
        stacked = {name: jnp.stack([m[exp_id].outputs[name] for m in kept]) for name in names}
        experiments[exp_id] = ExperimentPrediction(
            ts=first.ts,
            outputs={name: jnp.mean(values, axis=0) for name, values in stacked.items()},
            observed=first.observed,
            mask=first.mask,
            std={name: jnp.std(values, axis=0) for name, values in stacked.items()},
            quantiles=_member_quantiles(stacked, levels),
        )
    buckets = tuple(
        jnp.mean(jnp.stack([m.buckets[b] for m in kept]), axis=0)
        for b in range(len(dataset.bucket_payloads))
    )
    return PredictionResult(experiments, buckets, members)


def _check_quantiles(quantiles: Sequence[float], caller: str) -> tuple[float, ...]:
    """``quantiles`` as floats, each strictly inside (0, 1)."""
    levels = tuple(float(q) for q in quantiles)
    if any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError(f"{caller}: quantiles must lie strictly in (0, 1); got {levels}")
    return levels


def _member_quantiles(
    stacked: Mapping[str, Array], levels: Sequence[float]
) -> dict[str, dict[float, Array]]:
    """``{channel: {q: [T] values}}`` from ``{channel: [members, T] values}``."""
    if not levels:
        return {}
    # One call per channel: each sorts its values, whatever the number of levels.
    at_levels = {
        name: jnp.quantile(values, jnp.asarray(levels), axis=0) for name, values in stacked.items()
    }
    return {name: dict(zip(levels, values, strict=True)) for name, values in at_levels.items()}


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


def _warn_outside_box(predictor: Any, arrays: Mapping[str, Array]) -> None:
    """Warn when a ``BoundedPredictor`` is asked about inputs outside its input box.

    The guarded inverse continues past the box, so the network still returns a
    value there, from latents it never saw in training. Concrete inputs only:
    under a trace the check is skipped.
    """
    keys = getattr(predictor, "input_keys", None)
    scaler = getattr(predictor, "in_scaler", None)
    if not keys or scaler is None or len(scaler.bounds) != len(keys):
        return
    outside = []
    for key, (low, high) in zip(keys, scaler.bounds, strict=True):
        if key not in arrays:
            continue
        try:
            values = np.asarray(arrays[key])
        except jax.errors.TracerArrayConversionError:
            return
        if np.any(values < low) or np.any(values > high):
            outside.append(f"{key} outside ({low:g}, {high:g})")
    if outside:
        warnings.warn(
            f"evaluate_predictor: {', '.join(outside)}. The network extrapolates there "
            "from a box it was not trained beyond, so the value is not reliable.",
            RuntimeWarning,
            stacklevel=3,
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
    if not isinstance(inputs, Mapping):
        # A predictor built without input_keys still has names ("x1", ...),
        # so name them rather than guess which axis of an array is the input.
        names = getattr(predictor, "input_keys", None)
        hint = f" This predictor's inputs are {list(names)}." if names else ""
        raise TypeError(
            "evaluate_predictor takes a mapping of input name to array, got an "
            f"object of type {type(inputs).__name__}.{hint} For rows of an array, use "
            "jax.vmap(predictor)(x)."
        )
    # Only the inputs the predictor reads are broadcast: an extra key, which a
    # BoundedPredictor allows, must not fail on a shape it never uses.
    used = getattr(predictor, "input_keys", None)
    arrays = {
        key: jnp.asarray(value, dtype=jnp.result_type(jnp.asarray(value).dtype, float))
        for key, value in inputs.items()
        if used is None or key in used
    }
    _warn_outside_box(predictor, arrays)
    shape = jnp.broadcast_shapes(*(a.shape for a in arrays.values())) if arrays else ()
    if not shape:
        return jnp.asarray(predictor(arrays))
    flat = {key: jnp.broadcast_to(a, shape).reshape(-1) for key, a in arrays.items()}
    out = jax.vmap(predictor)(flat)
    return out.reshape(*shape, *out.shape[1:])
