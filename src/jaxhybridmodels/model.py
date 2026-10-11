"""``HybridModel``: the predictors, simulator, observation map and solver held together.

The functional layer (``train_with_optax``, ``predict_dataset`` and so on)
takes these four pieces as separate arguments on every call. This module
binds them once and returns results keyed by experiment ID and channel
name instead of bucket arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaxhybridmodels.data import (
    BucketPayload,
    Dataset,
    Experiment,
    _per_experiment_arrays,
    make_dataset,
)
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.prediction import predict_dataset
from jaxhybridmodels.solver import SolverConfig, simulation_state
from jaxhybridmodels.trainable import (
    FreezeSpec,
    count_trainable_params,
    freeze_mask,
    trainable_mask,
)
from jaxhybridmodels.training.evosax import EvosaxTrainingConfig, _train_with_evosax
from jaxhybridmodels.training.optax import OptaxTrainingConfig, _train_with_optax

StopReason = Literal["budget_exhausted", "patience"]

Data = Dataset | Sequence[Experiment]
"""What ``fit``, ``predict`` and ``validation=`` accept: a ``Dataset`` or the experiments."""


@dataclass(frozen=True)
class ExperimentPrediction:
    """One experiment's times, named outputs, and optional observed values.

    ``outputs`` maps each channel name to its ``[T]`` prediction.
    ``observed`` holds the measurements on the same grid, NaN where a cell
    was not measured, and ``observation_mask`` says which cells were.
    Both are empty for :meth:`HybridModel.simulate`, which instead fills
    ``state`` with the full ``[T, S]`` simulator state.
    """

    times: Array
    outputs: dict[str, Array]
    observed: dict[str, Array] = field(default_factory=dict)
    observation_mask: dict[str, Array] = field(default_factory=dict)
    state: Array | None = None


@dataclass(frozen=True)
class PredictionResult(Mapping[str, ExperimentPrediction]):
    """Predictions keyed by experiment ID, in the order the experiments were given.

    Pass it straight to ``compute_metrics`` or ``print_metrics``.
    ``buckets`` keeps the raw ``[N, T, D]`` arrays from ``predict_dataset``
    for custom JAX analysis. :meth:`from_buckets` builds one from bucket
    arrays made by the functional layer (``predict_dataset``,
    ``ensemble_predictions``), so those can be scored too.
    """

    experiments: dict[str, ExperimentPrediction]
    buckets: tuple[Array, ...] = ()

    @classmethod
    def from_buckets(cls, buckets: Sequence[Array], dataset: Dataset) -> PredictionResult:
        """Key per-bucket ``[N, T, D]`` predictions by experiment ID.

        ``dataset`` must be the one the predictions were made on, built by
        ``make_dataset`` so it keeps its source experiments::

            mean = hm.ensemble_predictions(members, dataset, ...)
            hm.print_metrics(hm.PredictionResult.from_buckets(mean, dataset))
        """
        if not dataset.experiments:
            raise ValueError(
                "PredictionResult.from_buckets: the dataset holds no source experiments; "
                "build it with make_dataset"
            )
        buckets = tuple(buckets)
        if len(buckets) != len(dataset.bucket_payloads):
            raise ValueError(
                f"PredictionResult.from_buckets: got {len(buckets)} bucket arrays for a "
                f"dataset with {len(dataset.bucket_payloads)} buckets"
            )
        # make_dataset groups experiments by union-grid length, keeping input
        # order inside each group. Walking the experiments in input order and
        # counting rows per length recovers each one's bucket row.
        bucket_of_length = {bp.ts.shape[1]: i for i, bp in enumerate(dataset.bucket_payloads)}
        next_row: dict[int, int] = {}
        names = dataset.output_channel_names
        experiments: dict[str, ExperimentPrediction] = {}
        for exp in dataset.experiments:
            arrays = _per_experiment_arrays(exp, names)
            length = arrays.ts.shape[0]
            row = next_row.get(length, 0)
            next_row[length] = row + 1
            values = buckets[bucket_of_length[length]][row]
            observed = jnp.where(arrays.mask, arrays.y_observed, jnp.nan)
            experiments[exp.exp_id] = ExperimentPrediction(
                times=arrays.ts,
                outputs=dict(zip(names, values.T, strict=True)),
                observed=dict(zip(names, observed.T, strict=True)),
                observation_mask=dict(zip(names, arrays.mask.T, strict=True)),
            )
        return cls(experiments, buckets)

    def __getitem__(self, exp_id: str) -> ExperimentPrediction:
        return self.experiments[exp_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self.experiments)

    def __len__(self) -> int:
        return len(self.experiments)

    def to_dataframe(self) -> Any:
        """Long table: one row per experiment, time and channel.

        Columns are ``experiment_id``, ``time``, ``channel``,
        ``prediction``, ``observed`` and ``is_observed``. Needs pandas,
        which the ``examples`` extra installs.
        """
        import pandas as pd

        columns = ["experiment_id", "time", "channel", "prediction", "observed", "is_observed"]
        frames = [
            pd.DataFrame(
                {
                    "experiment_id": exp_id,
                    "time": np.asarray(experiment.times),
                    "channel": channel,
                    "prediction": np.asarray(values),
                    "observed": np.asarray(
                        experiment.observed.get(channel, np.full(values.shape, np.nan))
                    ),
                    "is_observed": np.asarray(
                        experiment.observation_mask.get(channel, np.zeros(values.shape, bool))
                    ),
                },
                columns=columns,
            )
            for exp_id, experiment in self.items()
            for channel, values in experiment.outputs.items()
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


@dataclass(frozen=True)
class FitHistory:
    """Per-step records from a fit.

    ``loss`` has one entry per step (Optax) or per generation (Evosax).

    Optax measures it at the parameters before that step's update: the data
    term plus any trajectory penalty. ``penalty`` is the weighted bound
    penalty at the same step, so ``loss + penalty`` is what the optimiser
    minimised, and ``phase`` is the step's phase index.

    Evosax records the best fitness found so far, so its ``loss`` is
    monotone non-increasing; with a bound penalty on, that fitness includes
    it. ``penalty`` and ``phase`` stay empty.

    ``validation_loss`` is the full-horizon data loss on the validation set,
    present when one was given: per step for Optax, and for each
    generation's best individual for Evosax.
    """

    loss: tuple[float, ...] = ()
    penalty: tuple[float, ...] = ()
    phase: tuple[int, ...] = ()
    validation_loss: tuple[float, ...] = ()


@dataclass(frozen=True)
class FitResult:
    """The fitted model, its history, and how the run ended.

    ``best_step`` indexes the history at the returned parameters. It is
    None when ``restore_best=False`` returned the final update instead.
    ``stop_reason`` is ``"patience"`` when early stopping cut the final
    phase short, else ``"budget_exhausted"``.
    """

    model: HybridModel
    history: FitHistory
    best_step: int | None
    stop_reason: StopReason


@dataclass(frozen=True)
class _VectorFieldSimulation:
    """The ``simulate_fn`` that :meth:`HybridModel.from_vector_field` builds.

    Frozen, so two models built from the same vector field compare equal
    here and share compiled kernels. ``predictors`` and ``covariates`` go
    through the solve's ``args`` rather than a closure, which keeps
    ``BacksolveAdjoint`` able to differentiate with respect to them.
    """

    vector_field: Callable[..., Any]

    def _term(self, t: Any, y: Any, args: tuple[Any, Any]) -> Any:
        predictors, covariates = args
        return self.vector_field(t, y, predictors, covariates)

    def __call__(
        self, predictors: Any, ts: Array, covariates: Any, y0: Array, solver: SolverConfig
    ) -> diffrax.Solution:
        return solver.diffeqsolve(
            diffrax.ODETerm(self._term), ts, y0, args=(predictors, covariates)
        )


class HybridModel(eqx.Module):
    """The four pieces of a hybrid model, bound together.

    An ``eqx.Module``: ``predictors`` is its only array subtree, and the
    other fields are static. Equinox tools work on it directly, for example
    ``eqx.tree_at(lambda m: m.predictors["rate"], model, new_rate)`` to swap
    one predictor, or ``eqx.filter_vmap`` over a function that builds models
    from different keys.

    Parameters
    ----------
    predictors
        Any predictor pytree, usually a dict of ``Parameter``, ``Constant``
        and ``BoundedPredictor`` leaves.
    simulate_fn
        Pure ``(predictors, ts, covariates, y0, solver)``, returning the
        ``[T, S]`` state at ``ts`` or the ``diffrax.Solution`` that holds it.
        :meth:`from_vector_field` writes one from a vector field.
    state_to_output
        Pure ``[T, S] -> [T, D]``.
    output_channel_names
        Names of the ``D`` output columns, in order. Must match the
        ``output_channel_names`` of every dataset passed in.
    solver
        Integration settings, default ``SolverConfig()``.

    Fitting returns a new model; the original is unchanged.
    """

    predictors: Any
    simulate_fn: Callable[..., Any] = eqx.field(static=True)
    state_to_output: Callable[[Array], Array] = eqx.field(static=True)
    output_channel_names: tuple[str, ...] = eqx.field(static=True)
    solver: SolverConfig = eqx.field(static=True)

    def __init__(
        self,
        *,
        predictors: Any,
        simulate_fn: Callable[..., Any],
        state_to_output: Callable[[Array], Array],
        output_channel_names: Sequence[str],
        solver: SolverConfig | None = None,
    ) -> None:
        if isinstance(output_channel_names, str):
            raise TypeError(
                "output_channel_names must be a sequence of names, e.g. ('conc',), not a string"
            )
        names = tuple(output_channel_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("output_channel_names must be nonempty and unique")
        if not callable(simulate_fn) or not callable(state_to_output):
            raise TypeError("simulate_fn and state_to_output must be callables")
        self.predictors = predictors
        self.simulate_fn = simulate_fn
        self.state_to_output = state_to_output
        self.output_channel_names = names
        self.solver = SolverConfig() if solver is None else solver

    @classmethod
    def from_vector_field(
        cls,
        *,
        vector_field: Callable[..., Any],
        predictors: Any,
        state_to_output: Callable[[Array], Array],
        output_channel_names: Sequence[str],
        solver: SolverConfig | None = None,
    ) -> HybridModel:
        """Build a model whose ``simulate_fn`` is one ODE solve of ``vector_field``.

        ``vector_field(t, y, predictors, covariates) -> dy/dt`` is the whole
        of the physics. The solve uses :meth:`SolverConfig.diffeqsolve` and
        saves at the requested times. Write ``simulate_fn`` yourself for
        anything else: events, a per-experiment ``dt0``, several solves.
        """
        return cls(
            predictors=predictors,
            simulate_fn=_VectorFieldSimulation(vector_field),
            state_to_output=state_to_output,
            output_channel_names=output_channel_names,
            solver=solver,
        )

    def _with_predictors(self, predictors: Any) -> HybridModel:
        return eqx.tree_at(lambda m: m.predictors, self, predictors)

    def _dataset(self, data: Data, caller: str) -> Dataset:
        """``data`` as a ``Dataset`` with this model's channels.

        A sequence of experiments goes through ``make_dataset`` with the
        model's ``output_channel_names``.
        """
        if isinstance(data, Experiment):
            raise TypeError(
                f"{caller}: pass a Dataset or a list of experiments, not one Experiment"
            )
        if not isinstance(data, Dataset):
            data = make_dataset(list(data), output_channel_names=self.output_channel_names)
        if data.output_channel_names != self.output_channel_names:
            raise ValueError(
                f"{caller}: dataset channels {data.output_channel_names} do not match the "
                f"model's {self.output_channel_names}"
            )
        if not data.bucket_payloads:
            raise ValueError(f"{caller}: dataset has no experiments")
        return data

    def _check_shapes(self, bp: BucketPayload, caller: str) -> None:
        """Check ``simulate_fn`` and ``state_to_output`` shapes on one bucket row.

        Traced abstractly with ``eval_shape``, so nothing is integrated and
        nothing is compiled. Raises naming the function at fault, its
        actual shape, and the expected one.
        """
        ts = bp.ts[0]
        covariates = {name: value[0] for name, value in bp.covariates.items()}
        y0 = bp.y0[0]

        def state_of(predictors: Any) -> Array:
            return simulation_state(self.simulate_fn(predictors, ts, covariates, y0, self.solver))

        state = eqx.filter_eval_shape(state_of, self.predictors)
        self._check_state_shape(state.shape, ts.shape[0], caller)
        output = jax.eval_shape(self.state_to_output, state)
        self._check_output_shape(output.shape, state.shape, caller)

    def _check_state_shape(self, shape: tuple[int, ...], n_times: int, caller: str) -> None:
        if len(shape) != 2 or shape[0] != n_times:
            raise ValueError(
                f"{caller}: simulate_fn returned state of shape {shape}; expected [T, S] with "
                f"T = {n_times}, one row per requested time"
            )

    def _check_output_shape(
        self, shape: tuple[int, ...], state_shape: tuple[int, ...], caller: str
    ) -> None:
        expected = (state_shape[0], len(self.output_channel_names))
        if tuple(shape) != expected:
            raise ValueError(
                f"{caller}: state_to_output returned shape {tuple(shape)} for a state of shape "
                f"{tuple(state_shape)}; expected [T, D] = {expected}, one column per name in "
                f"output_channel_names {self.output_channel_names}"
            )

    def fit(
        self,
        data: Data,
        config: OptaxTrainingConfig | EvosaxTrainingConfig,
        *,
        key: Array,
        objective: Objective | None = None,
        freeze: Sequence[FreezeSpec] | FreezeSpec | None = None,
        trainable: Any = None,
        ui: Any = None,
        validation: Data | None = None,
    ) -> FitResult:
        """Train the predictors and return a new model with the result.

        ``config`` says how to search: Optax for gradient training, Evosax
        for an evolutionary search. ``objective`` says what is minimised;
        ``None`` means ``Objective()``, plain MSE with no penalty.

        ``freeze`` holds fixed, on top of the default
        :func:`~jaxhybridmodels.trainable.trainable_mask`, whatever its
        entries match: dotted paths (``"rate"``, ``"rate.inner"``),
        predictor classes (``hm.MLPPredictor``), or predicates
        ``(module) -> bool``. ``trainable`` is an explicit mask for full
        control; pass one or the other.

        With ``validation``, the validation data loss replaces the training
        loss for choosing the returned step (Optax) or generation (Evosax),
        and for ``patience``. It costs one extra forward pass over the
        validation set per step or generation.

        ``data`` and ``validation`` may be a ``Dataset`` or a sequence of
        experiments. ``simulate_fn`` and ``state_to_output`` shapes are
        checked once on the first experiment before anything compiles.
        """
        dataset = self._dataset(data, "fit")
        validation_set = None if validation is None else self._dataset(validation, "validation")
        if freeze is not None and trainable is not None:
            raise ValueError(
                "Pass freeze= or trainable=, not both; build the mask with "
                "freeze_mask(predictors, freeze, mask=...) to combine them"
            )
        if freeze is not None:
            mask = freeze_mask(self.predictors, freeze)
        elif trainable is not None:
            mask = trainable
        else:
            mask = trainable_mask(self.predictors)
        if count_trainable_params(self.predictors, mask) == 0:
            raise ValueError("Model has no trainable parameters")
        self._check_shapes(dataset.bucket_payloads[0], "fit")
        common: dict[str, Any] = dict(
            objective=objective,
            simulate_fn=self.simulate_fn,
            state_to_output=self.state_to_output,
            solver=self.solver,
            trainable=mask,
            key=key,
            ui=ui,
            validation=validation_set,
        )

        if isinstance(config, EvosaxTrainingConfig):
            evo = _train_with_evosax(self.predictors, dataset, config, **common)
            return FitResult(
                model=self._with_predictors(evo.predictors),
                history=FitHistory(loss=tuple(evo.loss), validation_loss=tuple(evo.validation)),
                best_step=evo.best_step,
                stop_reason="budget_exhausted",
            )
        if not isinstance(config, OptaxTrainingConfig):
            raise TypeError(
                "config must be an OptaxTrainingConfig or an EvosaxTrainingConfig; "
                f"got {type(config).__name__}"
            )
        run = _train_with_optax(self.predictors, dataset, config, **common)
        return FitResult(
            model=self._with_predictors(run.predictors),
            history=FitHistory(
                loss=tuple(run.loss),
                penalty=tuple(run.penalty),
                phase=tuple(run.phase),
                validation_loss=tuple(run.selection),
            ),
            best_step=run.best_step,
            stop_reason="patience" if run.stopped_early else "budget_exhausted",
        )

    def predict(self, data: Data) -> PredictionResult:
        """Predict every experiment on its measured time grid, keyed by experiment ID.

        ``data`` may be a ``Dataset`` from ``make_dataset`` or a sequence of
        experiments.
        """
        dataset = self._dataset(data, "predict")
        if not dataset.experiments:
            raise ValueError(
                "predict: the dataset holds no source experiments; build it with make_dataset"
            )
        buckets = predict_dataset(
            self.predictors,
            dataset,
            simulate_fn=self.simulate_fn,
            state_to_output=self.state_to_output,
            solver=self.solver,
        )
        for bp, values in zip(dataset.bucket_payloads, buckets, strict=True):
            if values.shape != bp.y_observed.shape:
                # Re-trace one row abstractly to say which function is at
                # fault; the batched shape alone cannot tell.
                self._check_shapes(bp, "predict")
                raise ValueError(
                    f"predict: bucket predictions have shape {values.shape}; expected "
                    f"[N, T, D] = {bp.y_observed.shape}"
                )
        return PredictionResult.from_buckets(buckets, dataset)

    def simulate(
        self,
        *,
        experiment: Experiment | None = None,
        ts: Any = None,
        y0: Any = None,
        covariates: Mapping[str, Any] | None = None,
        t0: float | None = None,
    ) -> ExperimentPrediction:
        """Simulate one set of conditions, from an experiment or given directly.

        With ``experiment``, its ``y0``, ``t0`` and covariates are used, and
        ``ts`` defaults to its measured times (``t0`` included when set).
        ``y0`` is placed at the experiment's ``t0``, or at its first
        observation time when ``t0`` is unset, whatever ``ts`` is passed; a
        ``ts`` starting before that time is rejected.
        Otherwise pass ``ts`` and ``y0``, and optionally ``covariates`` and
        ``t0``.

        ``y0`` is the state at ``t0``, which defaults to ``ts[0]``. ``ts``
        must be finite, strictly increasing, and not before ``t0``. When
        ``t0`` precedes ``ts[0]`` it is integrated from but left out of the
        returned times and outputs.
        """
        if experiment is not None:
            if y0 is not None or covariates is not None or t0 is not None:
                raise ValueError(
                    "simulate: experiment= supplies y0, covariates and t0; do not pass them too"
                )
            measured = _per_experiment_arrays(experiment, tuple(experiment.channels)).ts
            if ts is None:
                ts = measured
            # y0 belongs to the experiment's own start: t0 when set, else its
            # first observation time. Pin it so a caller's ts cannot move it.
            start_time = experiment.t0 if experiment.t0 is not None else float(measured[0])
            y0, covariates, t0 = experiment.y0, experiment.covariates, start_time
        if ts is None or y0 is None:
            raise ValueError("simulate: pass experiment=, or both ts= and y0=")

        times = jnp.asarray(ts, dtype=jnp.result_type(jnp.asarray(ts).dtype, float))
        values = np.asarray(times)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError(
                f"simulate: ts must be a nonempty finite rank-1 array; got {values.shape}"
            )
        if np.any(np.diff(values) <= 0):
            raise ValueError("simulate: ts must be strictly increasing")
        start = float(values[0]) if t0 is None else float(t0)
        if not math.isfinite(start) or start > float(values[0]):
            raise ValueError(
                "simulate: t0 must be finite and at or before the first requested time"
            )
        initial = jnp.asarray(y0, dtype=jnp.result_type(jnp.asarray(y0).dtype, float))
        if initial.ndim != 1 or not bool(jnp.all(jnp.isfinite(initial))):
            raise ValueError(
                f"simulate: y0 must be a finite rank-1 array; got shape {initial.shape}"
            )
        offset = int(start < float(values[0]))
        grid = jnp.concatenate((jnp.asarray([start], times.dtype), times)) if offset else times
        cov = {name: jnp.asarray(value) for name, value in (covariates or {}).items()}
        state = simulation_state(self.simulate_fn(self.predictors, grid, cov, initial, self.solver))
        self._check_state_shape(state.shape, grid.shape[0], "simulate")
        state = state[offset:]
        output = jnp.asarray(self.state_to_output(state))
        self._check_output_shape(output.shape, state.shape, "simulate")
        return ExperimentPrediction(
            times=times,
            outputs=dict(zip(self.output_channel_names, output.T, strict=True)),
            state=state,
        )
