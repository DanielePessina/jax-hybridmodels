"""``HybridModel``: the predictors, simulator, observation map and solver held together.

The model binds the four pieces once. ``fit``, ``fit_ensemble``,
``predict`` and ``simulate`` take data keyed by experiment and return
results keyed by experiment ID and channel name (:mod:`jaxhybridmodels.results`).
``fit`` dispatches to :func:`~jaxhybridmodels.training.train_with_optax`
or :func:`~jaxhybridmodels.training.train_with_evosax`, which take the
model and the same keywords.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaxhybridmodels.data import (
    BucketPayload,
    Data,
    Dataset,
    Experiment,
    _per_experiment_arrays,
    make_dataset,
)
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.prediction import _check_grid_args, _predict
from jaxhybridmodels.results import ExperimentPrediction, FitResult, PredictionResult
from jaxhybridmodels.solver import SolverConfig, simulation_state
from jaxhybridmodels.trainable import (
    FreezeSpec,
    count_trainable_params,
    freeze_mask,
    trainable_mask,
)
from jaxhybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from jaxhybridmodels.training.optax import OptaxTrainingConfig, _fit_ensemble, train_with_optax


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

    def _prepare_fit(
        self,
        data: Data,
        *,
        validation: Data | None,
        freeze: Sequence[FreezeSpec] | FreezeSpec | None,
        trainable: Any,
        caller: str,
    ) -> tuple[Dataset, Dataset | None, Any]:
        """Everything the trainers check before compiling: data, mask and shapes.

        Returns the training ``Dataset``, the validation one (or None) and
        the trainable mask built from ``freeze`` or ``trainable``.
        """
        dataset = self._dataset(data, caller)
        validation_set = None if validation is None else self._dataset(validation, "validation")
        if freeze is not None and trainable is not None:
            raise ValueError(
                "Pass freeze= or trainable=, not both; build the mask with "
                "jaxhybridmodels.trainable.freeze_mask(predictors, freeze, mask=...) "
                "to combine them"
            )
        if freeze is not None:
            mask = freeze_mask(self.predictors, freeze)
        elif trainable is not None:
            mask = trainable
        else:
            mask = trainable_mask(self.predictors)
        if count_trainable_params(self.predictors, mask=mask) == 0:
            raise ValueError("Model has no trainable parameters")
        self._check_shapes(dataset.bucket_payloads[0], caller)
        return dataset, validation_set, mask

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

        Same as ``train_with_optax(model, ...)`` or
        ``train_with_evosax(model, ...)`` in :mod:`jaxhybridmodels.training`,
        picked by the type of ``config``.
        """
        kwargs: dict[str, Any] = dict(
            key=key,
            objective=objective,
            freeze=freeze,
            trainable=trainable,
            ui=ui,
            validation=validation,
        )
        if isinstance(config, EvosaxTrainingConfig):
            return train_with_evosax(self, data, config, **kwargs)
        if isinstance(config, OptaxTrainingConfig):
            return train_with_optax(self, data, config, **kwargs)
        raise TypeError(
            "config must be an OptaxTrainingConfig or an EvosaxTrainingConfig; "
            f"got {type(config).__name__}"
        )

    def fit_ensemble(
        self,
        data: Data,
        config: OptaxTrainingConfig,
        *,
        key: Array,
        n_members: int,
        n_seeds: int | None = None,
        bootstrap: bool = False,
        objective: Objective | None = None,
        freeze: Sequence[FreezeSpec] | FreezeSpec | None = None,
        trainable: Any = None,
        ui: Any = None,
    ) -> list[FitResult]:
        """Train an ensemble of up to ``n_members`` models, best first.

        Every member starts from a fresh initialisation of the predictors
        drawn from ``key``. A candidate seed is trained briefly
        (``config.tournament_steps`` steps at ``config.tournament_lr``) and
        scored, as in the tournament, then the chosen seeds get the full
        phase schedule.

        Seed ensemble (``bootstrap=False``)
            Score ``n_seeds`` seeds on all of ``data`` and fully train the
            best ``n_members``. ``n_seeds`` defaults to ``n_members``, which
            trains every seed.
        Bootstrap ensemble (``bootstrap=True``)
            Draw ``n_members`` resamples of the experiments with replacement
            (:func:`~jaxhybridmodels.make_bootstrap_dataset`) and train one
            member on each: the best of ``n_seeds`` seeds scored on that
            resample. ``n_seeds`` defaults to 1, one fresh seed per resample.

        ``config.tournament_attempts`` is not used: ``n_seeds`` takes its
        place. ``objective``, ``freeze``, ``trainable`` and ``ui`` are as
        for :meth:`fit`; each member is its own UI run. The result is sorted
        by each member's final loss, the loss at its returned parameters
        when ``config.restore_best`` is set. A seed whose warm-up fails is
        dropped, so fewer than ``n_members`` may come back; if every seed
        fails it raises ``RuntimeError``. Optax only. The list goes straight
        into :func:`~jaxhybridmodels.predict_ensemble`.
        """
        if not isinstance(config, OptaxTrainingConfig):
            raise TypeError(
                "fit_ensemble trains with Optax only; pass an OptaxTrainingConfig, "
                f"got {type(config).__name__}"
            )
        return _fit_ensemble(
            self,
            data,
            config,
            key=key,
            n_members=n_members,
            n_seeds=n_seeds,
            bootstrap=bootstrap,
            objective=objective,
            freeze=freeze,
            trainable=trainable,
            ui=ui,
        )

    def predict(
        self,
        data: Data,
        *,
        ts: Any = None,
        n_points: int | None = None,
    ) -> PredictionResult:
        """Predict every experiment, keyed by experiment ID.

        By default each experiment is predicted on its measured time grid
        (``t0`` included when set), with its observations alongside, ready
        for ``compute_metrics``. ``n_points`` instead gives each experiment
        an even grid of that many points over its own time span, from its
        ``t0`` (or first observation) to its last observation, for smooth
        plots. ``ts`` gives one explicit grid for every experiment; it must
        not start before any experiment's start. Off the measured grid,
        ``observed`` and ``observation_mask`` are empty. Each experiment
        still integrates from its own start, so ``y0`` stays in place.

        ``data`` may be a ``Dataset`` from ``make_dataset`` or a sequence of
        experiments.
        """
        dataset = self._dataset(data, "predict")
        grid = _check_grid_args(ts, n_points, "predict")
        return _predict(self, dataset, ts=grid, n_points=n_points, caller="predict")

    def simulate(
        self,
        experiment: Experiment | None = None,
        *,
        ts: Any = None,
        n_points: int | None = None,
        y0: Any = None,
        covariates: Mapping[str, Any] | None = None,
        t0: float | None = None,
    ) -> ExperimentPrediction:
        """Simulate one set of conditions, from an experiment or given directly.

        With ``experiment``, its ``y0``, ``t0`` and covariates are used, and
        ``ts`` defaults to its measured times (``t0`` included when set).
        ``n_points`` instead gives an even grid of that many points from the
        experiment's start to its last observation, as in :meth:`predict`.
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
        _check_grid_args(ts, n_points, "simulate")
        if n_points is not None and experiment is None:
            raise ValueError(
                "simulate: n_points= needs experiment= for its time span; pass ts= instead"
            )
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
            if n_points is not None:
                ts = jnp.linspace(start_time, measured[-1], n_points, dtype=measured.dtype)
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
