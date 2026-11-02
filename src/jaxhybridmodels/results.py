"""The keyed result types: what ``fit``, ``predict`` and ``load`` hand back.

Kept apart from :mod:`jaxhybridmodels.model` so the trainers in
:mod:`jaxhybridmodels.training` and the prediction helpers in
:mod:`jaxhybridmodels.prediction` can build them without importing the
model, which imports both of them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from jaxtyping import Array

if TYPE_CHECKING:
    from jaxhybridmodels.model import HybridModel
    from jaxhybridmodels.objective import Objective
    from jaxhybridmodels.training.evosax import EvosaxTrainingConfig
    from jaxhybridmodels.training.optax import OptaxTrainingConfig

StopReason = Literal["budget_exhausted", "patience", "diverged"]


@dataclass(frozen=True)
class ExperimentPrediction:
    """One experiment's times, named outputs, and optional observed values.

    ``outputs`` maps each channel name to its ``[T]`` prediction.
    ``observed`` holds the measurements on the same grid, NaN where a cell
    was not measured, and ``observation_mask`` says which cells were. Both
    are empty when the times are not the measured grid: for
    :meth:`HybridModel.simulate`, and for ``predict`` with ``ts=`` or
    ``n_points=``. ``simulate`` fills ``state`` with the full ``[T, S]``
    simulator state. ``std`` is the per-channel spread across members, set
    only by :func:`~jaxhybridmodels.predict_ensemble`.
    """

    times: Array
    outputs: dict[str, Array]
    observed: dict[str, Array] = field(default_factory=dict)
    observation_mask: dict[str, Array] = field(default_factory=dict)
    state: Array | None = None
    std: dict[str, Array] = field(default_factory=dict)
    quantiles: dict[str, dict[float, Array]] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult(Mapping[str, ExperimentPrediction]):
    """Predictions keyed by experiment ID, in the order the experiments were given.

    Pass it straight to ``compute_metrics`` or ``print_metrics``.
    ``buckets`` keeps the raw ``[N, T, D]`` arrays, one per bucket of the
    dataset predicted on, for custom JAX analysis. For an ensemble,
    ``outputs`` and ``buckets`` hold the member mean and ``members`` holds
    each member's own result, in the order the models were given.
    """

    experiments: dict[str, ExperimentPrediction]
    buckets: tuple[Array, ...] = ()
    members: tuple[PredictionResult, ...] = ()

    def __getitem__(self, exp_id: str) -> ExperimentPrediction:
        return self.experiments[exp_id]

    def __getattr__(self, name: str) -> Any:
        # Runs only for a missing attribute. Reading ``result.quantiles`` is an
        # easy guess, and the plain error would not say where they are.
        message = f"{type(self).__name__!r} object has no attribute {name!r}"
        if name in {f.name for f in fields(ExperimentPrediction)}:
            message += f"; it is keyed by experiment, so read result[exp_id].{name}"
        raise AttributeError(message)

    def __iter__(self) -> Iterator[str]:
        return iter(self.experiments)

    def __len__(self) -> int:
        return len(self.experiments)

    def to_dataframe(self) -> Any:
        """Long table: one row per experiment, time and channel.

        Columns are ``experiment_id``, ``time``, ``channel``,
        ``prediction``, ``observed`` and ``is_observed``, plus ``std`` for
        an ensemble and ``q0.05``-style columns for requested quantiles. Needs
        pandas: ``pip install jax-hybridmodels[pandas]``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe needs pandas; install it with: pip install jax-hybridmodels[pandas]"
            ) from exc

        with_std = any(experiment.std for experiment in self.values())
        columns = ["experiment_id", "time", "channel", "prediction", "observed", "is_observed"]
        if with_std:
            columns.append("std")
        levels = sorted(
            {
                q
                for experiment in self.values()
                for by_q in experiment.quantiles.values()
                for q in by_q
            }
        )
        columns.extend(f"q{q:g}" for q in levels)
        frames = []
        for exp_id, experiment in self.items():
            for channel, values in experiment.outputs.items():
                missing = np.full(values.shape, np.nan)
                table = {
                    "experiment_id": exp_id,
                    "time": np.asarray(experiment.times),
                    "channel": channel,
                    "prediction": np.asarray(values),
                    "observed": np.asarray(experiment.observed.get(channel, missing)),
                    "is_observed": np.asarray(
                        experiment.observation_mask.get(channel, np.zeros(values.shape, bool))
                    ),
                }
                if with_std:
                    table["std"] = np.asarray(experiment.std.get(channel, missing))
                by_q = experiment.quantiles.get(channel, {})
                for q in levels:
                    table[f"q{q:g}"] = np.asarray(by_q.get(q, missing))
                frames.append(pd.DataFrame(table, columns=columns))
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

    ``failed_candidates`` is Evosax only: how many candidates in each
    generation had a non-finite fitness, usually a failed solve. They rank
    last and the search goes on.
    """

    loss: tuple[float, ...] = ()
    penalty: tuple[float, ...] = ()
    phase: tuple[int, ...] = ()
    validation_loss: tuple[float, ...] = ()
    failed_candidates: tuple[int, ...] = ()


@dataclass(frozen=True, repr=False)
class FitResult:
    """The fitted model, its history, and how the run ended.

    ``best_step`` indexes the history at the returned parameters. It is
    None when ``restore_best=False`` returned the final update instead.
    ``stop_reason`` is ``"patience"`` when early stopping cut the final
    phase short, ``"diverged"`` when a solver error or a non-finite loss ended
    the run (it warned, and the model holds the best parameters so far), else
    ``"budget_exhausted"``.
    """

    model: HybridModel
    history: FitHistory
    best_step: int | None
    stop_reason: StopReason

    @property
    def final_loss(self) -> float:
        """The training loss recorded at the returned parameters.

        ``history.loss[best_step]``, or the last entry when ``best_step`` is
        None. ``fit_ensemble`` ranks its members by it.
        """
        if not self.history.loss:
            return float("nan")
        step = -1 if self.best_step is None else self.best_step
        return self.history.loss[step]

    def summary(self, data: Any = None) -> str:
        """A readable report on the run; ``print(result.summary())``.

        Says how the run ended, the loss by phase, each ``Parameter`` with any
        that sit near a bound, and how much of each network's input box is
        pinned against an output bound. With ``data``, a list of experiments or
        a ``Dataset``, it also predicts on them and adds the metrics and the
        worst experiment per channel.

        A flagged parameter is in the range where the bound penalty would
        charge it: the outer 5% of its box for the default sigmoid. The network
        figure covers the whole declared box, including inputs no experiment
        reached, so a large share can be harmless if the data never goes there.
        """
        from jaxhybridmodels._fit_summary import format_fit_summary

        return format_fit_summary(self, data=data)

    def __repr__(self) -> str:
        # The default would print the whole model and every history entry.
        return (
            f"FitResult(stop_reason={self.stop_reason!r}, best_step={self.best_step}, "
            f"final_loss={self.final_loss:.6g}, steps={len(self.history.loss)})"
        )


@dataclass(frozen=True)
class SavedRun:
    """A run read back by :func:`~jaxhybridmodels.load`.

    ``model`` is the ``like=`` model with the saved predictors; it keeps
    ``like``'s solver.
    ``history``, ``best_step`` and ``stop_reason`` are the saved
    ``FitResult``'s; a saved bare model leaves them as an empty
    ``FitHistory`` and None.

    ``config`` and ``objective`` come back as typed objects. A section that
    names a callable (a custom loss, a ``penalty_fn``, an optimizer factory,
    an lr schedule) cannot be rebuilt, so it comes back as the raw dict
    with each callable as ``{"__callable__": "module.qualname"}``; rebind
    those fields yourself. ``extras`` is the dict passed to ``save``.
    """

    model: HybridModel
    history: FitHistory
    best_step: int | None
    stop_reason: StopReason | None
    config: OptaxTrainingConfig | EvosaxTrainingConfig | dict[str, Any] | None
    objective: Objective | dict[str, Any] | None
    extras: dict[str, Any]

    def summary(self, data: Any = None) -> str:
        """The report :meth:`FitResult.summary` gives, for a run read back from disk."""
        from jaxhybridmodels._fit_summary import format_fit_summary

        return format_fit_summary(self, data=data)
