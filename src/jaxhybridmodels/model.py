"""Model methods and results organized around experiments and physical quantities.

The functional training and bucket APIs remain available. This module holds
those pieces together for fitting, inspecting, and simulating a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

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


@dataclass(frozen=True)
class ExperimentPrediction:
    """One experiment's times, named outputs, and optional observed values.

    Unobserved entries in ``observed`` are NaN. ``observation_mask`` states
    which cells are real measurements. ``state`` is present for simulations
    made directly with :meth:`HybridModel.simulate`.
    """

    times: Array
    outputs: dict[str, Array]
    observed: dict[str, Array] = field(default_factory=dict)
    observation_mask: dict[str, Array] = field(default_factory=dict)
    state: Array | None = None

    def __getitem__(self, channel: str) -> Array:
        return self.outputs[channel]


@dataclass(frozen=True)
class PredictionResult(Mapping[str, ExperimentPrediction]):
    """Predictions keyed by experiment ID, retaining input experiment order.

    ``buckets`` exposes the original arrays for use with ``compute_metrics``
    or custom JAX analysis. ``to_dataframe`` requires optional pandas.
    Missing experiment IDs receive ``experiment_<index>`` names; collisions
    raise rather than overwrite experiments.
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
        """Export a long table with experiment, time, channel, and prediction.

        Columns also include ``observed`` and ``is_observed``. Install
        pandas separately, or use the package's ``examples`` extra.
        """
        import pandas as pd  # ty: ignore[unresolved-import]

        frames = []
        for exp_id, experiment in self.items():
            for channel, values in experiment.outputs.items():
                frames.append(
                    pd.DataFrame(
                        {
                            "experiment_id": exp_id,
                            "time": np.asarray(experiment.times),
                            "channel": channel,
                            "prediction": np.asarray(values),
                            "observed": np.asarray(
                                experiment.observed.get(channel, jnp.full_like(values, jnp.nan))
                            ),
                            "is_observed": np.asarray(
                                experiment.observation_mask.get(
                                    channel, jnp.zeros(values.shape, dtype=bool)
                                )
                            ),
                        }
                    )
                )
        if not frames:
            return pd.DataFrame(
                columns=[
                    "experiment_id",
                    "time",
                    "channel",
                    "prediction",
                    "observed",
                    "is_observed",
                ]
            )
        return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class FitHistory:
    """Optax pre-update losses, or Evosax best-so-far fitness.

    ``data_loss`` excludes all penalties. ``tracked_loss`` includes the
    trajectory penalty, while ``objective`` includes both trajectory and
    fixed-point penalties. Validation uses full-horizon data loss. Evosax
    fills only ``best_fitness``: its generation history has different
    semantics from Optax steps.
    """

    data_loss: tuple[float, ...] = ()
    tracked_loss: tuple[float, ...] = ()
    objective: tuple[float, ...] = ()
    validation_loss: tuple[float, ...] = ()
    phase: tuple[int, ...] = ()
    best_fitness: tuple[float, ...] = ()


@dataclass(frozen=True)
class FitResult:
    """A fitted model, loss histories, and the reason fitting stopped.

    ``best_step`` is zero-based, referring to Optax's pre-update parameters
    or the first Evosax generation reporting the best fitness (which may
    include the input seed). It is None
    when Optax returns the final update instead of a selected checkpoint.
    """

    model: HybridModel
    history: FitHistory
    best_step: int | None
    stop_reason: str
    selected_by: str


@dataclass(frozen=True)
class HybridModel:
    """Hold the predictors, simulator, observation map, and solver together.

    Parameters
    ----------
    predictors
        Any predictor PyTree, including named dictionaries.
    simulate_fn
        Pure ``(predictors, ts, covariates, y0, solver) -> [T, S]`` callback.
    state_to_output
        Pure ``[T, S] -> [T, D]`` callback.
    output_channels
        Ordered names matching the columns returned by state_to_output.
    solver
        Integration settings.

    Notes
    -----
    Fitting returns a new model; the original remains unchanged. The
    functional APIs accept these same pieces separately.
    """

    predictors: Any
    simulate_fn: Callable[..., Array]
    state_to_output: Callable[[Array], Array]
    output_channels: tuple[str, ...]
    solver: SolverConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_channels", tuple(self.output_channels))
        if not self.output_channels or len(set(self.output_channels)) != len(self.output_channels):
            raise ValueError("output_channels must be nonempty and unique")

    def _check_dataset(self, dataset: Dataset) -> None:
        if dataset.output_channel_names != self.output_channels:
            raise ValueError("Dataset output channels must match model output_channels in order")
        if not dataset.bucket_payloads:
            raise ValueError("Dataset must contain at least one experiment")

    def fit(
        self,
        dataset: Dataset,
        config: OptaxTrainingConfig | EvosaxTrainingConfig | None = None,
        *,
        key: Array,
        steps: int | None = None,
        lr: float | None = None,
        trainable: Any = None,
        ui: Any = None,
        validation: Dataset | None = None,
        select_best: str | None = None,
    ) -> FitResult:
        """Fit with Optax by default, or with an Evosax config.

        Supply a config, or use ``steps=`` and ``lr=`` for a simple Optax
        run. ``select_best`` is ``training_loss`` or ``validation_loss``;
        it defaults to validation when supplied, otherwise training.
        Validation scoring is an additional forward pass at every step.
        Validation selection and patience are supported for Optax only.
        """
        return fit(
            self,
            dataset,
            config,
            key=key,
            steps=steps,
            lr=lr,
            trainable=trainable,
            ui=ui,
            validation=validation,
            select_best=select_best,
        )

    def predict(self, dataset: Dataset) -> PredictionResult:
        """Predict measured union grids, with named experiment/channel access."""
        return predict(self, dataset)

    def simulate(
        self,
        *,
        ts: Any,
        y0: Any,
        covariates: Mapping[str, Any] | None = None,
        t0: float | None = None,
    ) -> ExperimentPrediction:
        """Simulate new conditions without an observation dataset.

        ``y0`` is the state at ``t0`` (default: the first requested time).
        Times must be finite, strictly increasing, and at or after t0.
        A leading integration point is added when needed and omitted from
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
        grid = (
            jnp.concatenate((jnp.asarray([start], dtype=times.dtype), times)) if offset else times
        )
        state = jnp.asarray(
            self.simulate_fn(
                self.predictors,
                grid,
                {name: jnp.asarray(value) for name, value in (covariates or {}).items()},
                initial,
                self.solver,
            )
        )
        if state.ndim != 2 or state.shape[0] != grid.shape[0]:
            raise ValueError("simulate_fn must return [T, S] for the requested time grid")
        state = state[offset:]
        output = jnp.asarray(self.state_to_output(state))
        if output.shape != (times.shape[0], len(self.output_channels)):
            raise ValueError("state_to_output must return [T, D] matching output_channels")
        return ExperimentPrediction(
            times, dict(zip(self.output_channels, output.T, strict=True)), state=state
        )


def predict(model: HybridModel, dataset: Dataset) -> PredictionResult:
    """Functional equivalent of ``model.predict(dataset)``."""
    model._check_dataset(dataset)
    if not dataset._experiments:
        raise ValueError("Named predictions require a dataset built with make_dataset")
    buckets = predict_dataset(
        model.predictors,
        dataset,
        simulate_fn=model.simulate_fn,
        state_to_output=model.state_to_output,
        solver=model.solver,
    )
    # Bucket grouping preserves input order within each length, while the
    # public result restores the original experiment order across lengths.
    rows: dict[int, int] = {}
    bucket_indices = {bp.ts.shape[1]: i for i, bp in enumerate(dataset.bucket_payloads)}
    experiments = {}
    for i, exp in enumerate(dataset._experiments):
        exp_id = exp.exp_id or f"experiment_{i}"
        if exp_id in experiments:
            raise ValueError(f"Duplicate experiment ID {exp_id!r}; assign unique exp_id values")
        arrays = _per_experiment_arrays(exp, model.output_channels)
        length = arrays.ts.shape[0]
        row = rows.get(length, 0)
        bucket = buckets[bucket_indices[length]]
        values = bucket[row]
        if values.shape != arrays.y_observed.shape:
            raise ValueError("state_to_output must return [T, D] matching output_channels")
        rows[length] = row + 1
        experiments[exp_id] = ExperimentPrediction(
            times=arrays.ts,
            outputs={name: values[:, d] for d, name in enumerate(model.output_channels)},
            observed={
                name: jnp.where(arrays.mask[:, d], arrays.y_observed[:, d], jnp.nan)
                for d, name in enumerate(model.output_channels)
            },
            observation_mask={
                name: arrays.mask[:, d] for d, name in enumerate(model.output_channels)
            },
        )
    return PredictionResult(experiments, buckets)


def fit(
    model: HybridModel,
    dataset: Dataset,
    config: OptaxTrainingConfig | EvosaxTrainingConfig | None = None,
    *,
    key: Array,
    steps: int | None = None,
    lr: float | None = None,
    trainable: Any = None,
    ui: Any = None,
    validation: Dataset | None = None,
    select_best: str | None = None,
) -> FitResult:
    """Functional equivalent of ``model.fit``; return a new fitted model."""
    model._check_dataset(dataset)
    mask = trainable_mask(model.predictors) if trainable is None else trainable
    if count_trainable_params(model.predictors, mask) == 0:
        raise ValueError("Model has no trainable parameters; ConstantPredictor values stay fixed")
    if config is not None and (steps is not None or lr is not None):
        raise ValueError("Supply config= or steps=/lr=, not both")
    if config is None:
        config = OptaxTrainingConfig(
            steps=1000 if steps is None else steps, lr=1e-3 if lr is None else lr
        )
    selection = select_best or ("validation_loss" if validation is not None else "training_loss")
    if selection not in ("training_loss", "validation_loss"):
        raise ValueError("select_best must be training_loss or validation_loss")
    if selection == "validation_loss" and validation is None:
        raise ValueError("validation_loss selection requires validation data")
    if validation is not None:
        model._check_dataset(validation)
    kwargs: dict[str, Any] = dict(
        simulate_fn=model.simulate_fn,
        state_to_output=model.state_to_output,
        solver=model.solver,
        key=key,
        trainable=trainable,
        ui=ui,
    )
    if isinstance(config, EvosaxTrainingConfig):
        if validation is not None:
            raise ValueError("Per-generation validation selection is supported for Optax only")
        history, predictors = train_with_evosax(model.predictors, dataset, config, **kwargs)
        return FitResult(
            replace(model, predictors=predictors),
            FitHistory(best_fitness=tuple(history)),
            int(np.argmin(history)) if history else None,
            "budget_exhausted",
            "best_fitness",
        )
    if not isinstance(config, OptaxTrainingConfig):
        raise TypeError("config must be OptaxTrainingConfig or EvosaxTrainingConfig")

    loss_fn = resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights)
    score = build_score_bucket(
        simulate_fn=model.simulate_fn,
        state_to_output=model.state_to_output,
        solver=model.solver,
        loss_fn=loss_fn,
    )
    data_losses: list[float] = []
    tracked: list[float] = []
    objectives: list[float] = []
    validation_losses: list[float] = []
    phases: list[int] = []
    selected_predictors = model.predictors
    best_step = None
    best_loss = math.inf
    stale = 0
    stopped_for_validation = False

    def score_dataset(predictors: Any, data: Dataset, fraction: float) -> float:
        return sum(
            float(score(predictors, bp, jnp.asarray(fraction))) for bp in data.bucket_payloads
        ) / len(data.bucket_payloads)

    def record(predictors: Any, phase: int, step: int, loss: float, penalty: float) -> bool:
        nonlocal selected_predictors, best_step, best_loss, stale, stopped_for_validation
        # The legacy trainer already computed this term unless a trajectory
        # regularizer is present. Only that case needs another training solve.
        data_loss = (
            score_dataset(predictors, dataset, config.length_schedule[phase])
            if config.trajectory_penalty_fn is not None
            else loss
        )
        data_losses.append(data_loss)
        tracked.append(loss)
        objectives.append(loss + config.penalty_weight_for_phase(phase) * penalty)
        phases.append(phase)
        if not math.isfinite(objectives[-1]):
            raise FloatingPointError("Training produced a non-finite objective")
        if validation is not None:
            validation_losses.append(score_dataset(predictors, validation, 1.0))
        selected_loss = validation_losses[-1] if selection == "validation_loss" else loss
        if not math.isfinite(selected_loss):
            raise FloatingPointError("Model selection produced a non-finite loss")
        # Match the legacy horizon reset for training selection. Full-horizon
        # validation scores remain comparable across training phases.
        if step == 0:
            stale = 0
            if (
                selection == "training_loss"
                and phase > 0
                and config.length_schedule[phase] != config.length_schedule[phase - 1]
            ):
                best_loss = math.inf
        if selected_loss < best_loss:
            best_loss = selected_loss
            best_step = len(tracked) - 1
            selected_predictors = predictors
            stale = 0
        else:
            stale += 1
        if selection == "validation_loss" and config.patience > 0 and stale >= config.patience:
            stopped_for_validation = True
            return True
        return False

    run_config = replace(config, patience=0) if selection == "validation_loss" else config
    _, trained = _train_with_optax(
        model.predictors, dataset, run_config, _step_callback=record, **kwargs
    )
    restore = selection == "validation_loss" or config.restore_best
    predictors = selected_predictors if restore else trained
    reason = (
        "validation_patience"
        if stopped_for_validation
        else "training_patience"
        if len(tracked) < sum(config.steps)
        else "budget_exhausted"
    )
    return FitResult(
        replace(model, predictors=predictors),
        FitHistory(
            tuple(data_losses),
            tuple(tracked),
            tuple(objectives),
            tuple(validation_losses),
            tuple(phases),
        ),
        best_step if restore else None,
        reason,
        selection if restore else "final_update",
    )
