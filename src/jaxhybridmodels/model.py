"""``HybridModel``: the predictors, simulator, observation map and solver held together.

The functional layer (``train_with_optax``, ``predict_dataset`` and so on)
takes these four pieces as separate arguments on every call. This module
binds them once and returns results keyed by experiment ID and channel
name instead of bucket arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaxhybridmodels.data import Dataset, _per_experiment_arrays
from jaxhybridmodels.losses import resolve_loss_fn
from jaxhybridmodels.prediction import predict_dataset
from jaxhybridmodels.solver import SolverConfig
from jaxhybridmodels.trainable import count_trainable_params, trainable_mask
from jaxhybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from jaxhybridmodels.training.kernels import build_score_bucket
from jaxhybridmodels.training.optax import OptaxTrainingConfig, _train_with_optax

StopReason = Literal["budget_exhausted", "patience"]


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

    ``buckets`` keeps the raw ``[N, T, D]`` arrays from ``predict_dataset``
    for ``compute_metrics`` or custom JAX analysis. Experiments without an
    ID are named ``experiment_<index>``.
    """

    experiments: dict[str, ExperimentPrediction]
    buckets: tuple[Array, ...] = ()

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

    Optax fills ``loss``, ``penalty`` and ``phase``, one entry per step,
    measured at the parameters before that step's update. ``loss`` is the
    data term plus any trajectory penalty; ``penalty`` is the weighted
    bound penalty, so ``loss + penalty`` is the objective the optimiser
    minimised. ``validation_loss`` is the full-horizon data loss on the
    validation set, present when one was given.

    Evosax fills only ``best_fitness``, the best-so-far fitness per
    generation. It is monotone and not comparable to ``loss``.
    """

    loss: tuple[float, ...] = ()
    penalty: tuple[float, ...] = ()
    phase: tuple[int, ...] = ()
    validation_loss: tuple[float, ...] = ()
    best_fitness: tuple[float, ...] = ()


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
class HybridModel:
    """The four pieces of a hybrid model, bound together.

    Parameters
    ----------
    predictors
        Any predictor pytree, usually a dict of ``Parameter``, ``Constant``
        and ``BoundedPredictor`` leaves.
    simulate_fn
        Pure ``(predictors, ts, covariates, y0, solver) -> [T, S]``.
    state_to_output
        Pure ``[T, S] -> [T, D]``.
    output_channel_names
        Names of the ``D`` output columns, in order. Must match the
        ``output_channel_names`` of every dataset passed in.
    solver
        Integration settings.

    Fitting returns a new model; the original is unchanged.
    """

    predictors: Any
    simulate_fn: Callable[..., Array]
    state_to_output: Callable[[Array], Array]
    output_channel_names: tuple[str, ...]
    solver: SolverConfig

    def __post_init__(self) -> None:
        names = tuple(self.output_channel_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("output_channel_names must be nonempty and unique")
        object.__setattr__(self, "output_channel_names", names)

    def _check_dataset(self, dataset: Dataset) -> None:
        if dataset.output_channel_names != self.output_channel_names:
            raise ValueError(
                f"Dataset channels {dataset.output_channel_names} do not match the model's "
                f"{self.output_channel_names}"
            )
        if not dataset._experiments:
            raise ValueError("Dataset must contain at least one experiment from make_dataset")

    def fit(
        self,
        dataset: Dataset,
        config: OptaxTrainingConfig | EvosaxTrainingConfig,
        *,
        key: Array,
        trainable: Any = None,
        ui: Any = None,
        validation: Dataset | None = None,
    ) -> FitResult:
        """Train the predictors and return a new model with the result.

        ``config`` picks the optimiser: Optax for gradient training, Evosax
        for an evolutionary search. ``trainable`` defaults to
        :func:`~jaxhybridmodels.trainable.trainable_mask`.

        With ``validation`` (Optax only), the validation data loss replaces
        the training loss for choosing the returned step and for
        ``patience``. It costs one extra forward pass over the validation
        set per step.
        """
        self._check_dataset(dataset)
        mask = trainable_mask(self.predictors) if trainable is None else trainable
        if count_trainable_params(self.predictors, mask) == 0:
            raise ValueError("Model has no trainable parameters")
        if validation is not None:
            self._check_dataset(validation)
        common: dict[str, Any] = dict(
            simulate_fn=self.simulate_fn,
            state_to_output=self.state_to_output,
            solver=self.solver,
            key=key,
            trainable=mask,
            ui=ui,
        )

        if isinstance(config, EvosaxTrainingConfig):
            if validation is not None:
                raise ValueError("validation= is supported for Optax fits only")
            fitness, predictors = train_with_evosax(self.predictors, dataset, config, **common)
            return FitResult(
                model=replace(self, predictors=predictors),
                history=FitHistory(best_fitness=tuple(fitness)),
                best_step=int(np.argmin(fitness)) if fitness else None,
                stop_reason="budget_exhausted",
            )

        select_score: Callable[[Any], float] | None = None
        if validation is not None:
            score = build_score_bucket(
                simulate_fn=self.simulate_fn,
                state_to_output=self.state_to_output,
                solver=self.solver,
                loss_fn=resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights),
            )
            full_horizon = jnp.asarray(1.0)
            buckets = validation.bucket_payloads

            def validation_score(predictors: Any) -> float:
                total = sum(float(score(predictors, bp, full_horizon)) for bp in buckets)
                value = total / len(buckets)
                if not math.isfinite(value):
                    raise FloatingPointError("Validation loss is not finite")
                return value

            select_score = validation_score

        run = _train_with_optax(
            self.predictors, dataset, config, select_score=select_score, **common
        )
        return FitResult(
            model=replace(self, predictors=run.predictors),
            history=FitHistory(
                loss=tuple(run.loss),
                penalty=tuple(run.penalty),
                phase=tuple(run.phase),
                validation_loss=tuple(run.selection),
            ),
            best_step=run.best_step,
            stop_reason="patience" if run.stopped_early else "budget_exhausted",
        )

    def predict(self, dataset: Dataset) -> PredictionResult:
        """Predict every experiment on its measured time grid, keyed by experiment ID."""
        self._check_dataset(dataset)
        buckets = predict_dataset(
            self.predictors,
            dataset,
            simulate_fn=self.simulate_fn,
            state_to_output=self.state_to_output,
            solver=self.solver,
        )
        # make_dataset groups experiments by union-grid length, keeping input
        # order inside each group. Walking the experiments in input order and
        # counting rows per length recovers each one's bucket row.
        bucket_of_length = {bp.ts.shape[1]: i for i, bp in enumerate(dataset.bucket_payloads)}
        next_row: dict[int, int] = {}
        names = self.output_channel_names
        experiments: dict[str, ExperimentPrediction] = {}
        for i, exp in enumerate(dataset._experiments):
            exp_id = exp.exp_id or f"experiment_{i}"
            if exp_id in experiments:
                raise ValueError(f"Duplicate experiment ID {exp_id!r}; assign unique exp_id values")
            arrays = _per_experiment_arrays(exp, names)
            length = arrays.ts.shape[0]
            row = next_row.get(length, 0)
            next_row[length] = row + 1
            values = buckets[bucket_of_length[length]][row]
            if values.shape != arrays.y_observed.shape:
                raise ValueError("state_to_output must return [T, D] matching output_channel_names")
            observed = jnp.where(arrays.mask, arrays.y_observed, jnp.nan)
            experiments[exp_id] = ExperimentPrediction(
                times=arrays.ts,
                outputs=dict(zip(names, values.T, strict=True)),
                observed=dict(zip(names, observed.T, strict=True)),
                observation_mask=dict(zip(names, arrays.mask.T, strict=True)),
            )
        return PredictionResult(experiments, buckets)

    def simulate(
        self,
        *,
        ts: Any,
        y0: Any,
        covariates: Mapping[str, Any] | None = None,
        t0: float | None = None,
    ) -> ExperimentPrediction:
        """Simulate new conditions without a dataset.

        ``y0`` is the state at ``t0``, which defaults to ``ts[0]``. ``ts``
        must be finite, strictly increasing, and not before ``t0``. When
        ``t0`` precedes ``ts[0]`` it is integrated from but left out of
        the returned times and outputs.
        """
        times = jnp.asarray(ts, dtype=jnp.result_type(jnp.asarray(ts).dtype, float))
        values = np.asarray(times)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("ts must be a nonempty finite rank-1 array")
        if np.any(np.diff(values) <= 0):
            raise ValueError("ts must be strictly increasing")
        start = float(values[0]) if t0 is None else float(t0)
        if not math.isfinite(start) or start > float(values[0]):
            raise ValueError("t0 must be finite and at or before the first requested time")
        initial = jnp.asarray(y0, dtype=jnp.result_type(jnp.asarray(y0).dtype, float))
        if initial.ndim != 1 or not bool(jnp.all(jnp.isfinite(initial))):
            raise ValueError("y0 must be a finite rank-1 array")
        offset = int(start < float(values[0]))
        grid = jnp.concatenate((jnp.asarray([start], times.dtype), times)) if offset else times
        cov = {name: jnp.asarray(value) for name, value in (covariates or {}).items()}
        state = jnp.asarray(self.simulate_fn(self.predictors, grid, cov, initial, self.solver))
        if state.ndim != 2 or state.shape[0] != grid.shape[0]:
            raise ValueError("simulate_fn must return [T, S] for the requested time grid")
        state = state[offset:]
        output = jnp.asarray(self.state_to_output(state))
        if output.shape != (times.shape[0], len(self.output_channel_names)):
            raise ValueError("state_to_output must return [T, D] matching output_channel_names")
        return ExperimentPrediction(
            times=times,
            outputs=dict(zip(self.output_channel_names, output.T, strict=True)),
            state=state,
        )
