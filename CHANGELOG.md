# Changelog

## Unreleased

### Added

- `HybridModel` binds predictors, simulator, observation map, channel names
  and solver. It is an `eqx.Module` built with keyword arguments only, so
  `eqx.tree_at` swaps a predictor and `eqx.filter_vmap` builds a batch of
  models. `fit` returns a `FitResult` with the new model, a `FitHistory`
  (`loss`, `penalty`, `phase`, `validation_loss`), `best_step` and
  `stop_reason`. `predict` returns predictions keyed by experiment ID and
  channel name, with observed values, masks and an optional pandas export.
  `simulate` runs new conditions without a dataset.
- `fit`, `predict` and `validation=` take a list of experiments as well as
  a `Dataset`. `Dataset.experiments` exposes the source experiments.
- `hm.Objective` holds what a fit minimises: `loss`, `channel_weights`
  (a mapping by channel name, or one weight per channel), the bound penalty
  (`penalty_weight`, `penalty_points`, `penalty_fn`) and the trajectory
  penalty. `fit`, `train_with_optax`, `train_with_evosax` and the ensemble
  helpers take it as `objective=`. Channel names and penalty paths are
  checked before anything compiles.
- `validation=` on `HybridModel.fit`, for Optax and Evosax: the validation
  loss picks the returned step or generation and drives `patience`.
- `TrainingPhase.lr` accepts an optax schedule, counted from the start of
  the phase. Chain several with `optax.join_schedules`.
- `fit(freeze=...)` freezes by dotted path, module class or predicate on
  top of the default mask. `freeze_mask` builds the same mask by hand.
- `simulate(experiment=...)` reruns an experiment's `y0`, `t0` and
  covariates. `ts` defaults to the experiment's time axis: its measured
  times, plus `t0` when set. `y0` stays at the experiment's `t0`, or its
  first measured time when `t0` is unset, whatever `ts` is passed.
- `HybridModel.from_vector_field` builds `simulate_fn` from a vector field
  `(t, y, predictors, covariates)`.
- `simulate_fn` may return the `diffrax.Solution` instead of its `ys`.
- `SolverConfig()` works with no arguments: `Tsit5`, `rtol=1e-5`,
  `atol=1e-7`, `max_steps=4096`, `dt0=None`.
- `experiment_from_array` builds an experiment from a shared time grid and
  a `[T, D]` array, and `experiments_from_dataframe` builds one per run from
  a wide pandas table. `NaN` marks an unmeasured cell.
- `evaluate_predictor(predictor, {name: array})` evaluates over a grid,
  broadcasting the inputs.
- `PredictionResult.from_buckets(buckets, dataset)` keys bucket arrays from
  `predict_dataset` or `ensemble_predictions` by experiment, so
  `compute_metrics` and `print_metrics` can score them.
- `freeze_paths` accepts a subtree prefix (`"rate"` freezes everything
  under `mask["rate"]`) and a bare string as well as a tuple.
- `jaxhybridmodels.penalties.bounded_predictor_paths` lists the path of
  every `BoundedPredictor`, the keys `Objective.penalty_points` takes.
- `TrainingPhase.lr` accepts any callable. One JAX cannot trace, such as a
  Python `if` on the step, is evaluated step by step on the host.
- `save_run(objective=...)` stores the objective; `load_run` returns it and
  rebuilds it with `objective_cls=hm.Objective`.
- `fit` checks the shapes returned by `simulate_fn` and `state_to_output`
  on the first experiment before compiling. Shape errors in `fit`,
  `predict` and `simulate` name the function, its shape and the expected
  one.
- `Parameter(value, bounds=...)` for trainable physical scalars and arrays,
  and `Constant(value)` for fixed ones, both read through `.value`.
- Open bounds: `None` or an infinite endpoint gives a one-sided softplus or
  an identity transform. Open network inputs pass through unscaled.
- `TrainingPhase` and `OptaxTrainingConfig(phases=...)` for schedules;
  `OptaxTrainingConfig(steps=..., lr=...)` for a single phase.
- `make_experiment(t0=..., y0=...)`: an initial time before the first
  observation, and a direct initial state as an alternative to `y0_fn`.
  `covariates` is optional.
- CI lint job and pre-commit hooks running `ruff check`, `ruff format` and
  `ty check` over sources, tests, examples and scripts.
- A rewritten test suite built on one small shared decay model. It is
  mostly end to end and runs in about two minutes instead of twenty.

### Changed (breaking against 0.2.0b1)

- `OptaxTrainingConfig` stores `phases`, a tuple of `TrainingPhase`. The
  parallel `steps`/`lr`/`optimizer`/`reset_optimiser_state`/
  `length_schedule`/`penalty_weight` tuples are gone. `length_schedule` is
  now `TrainingPhase.horizon` and `reset_optimiser_state` is
  `TrainingPhase.reset_optimizer`. `Objective.penalty_weight` sets the
  bound-penalty weight; `TrainingPhase.penalty_weight` overrides it for one
  phase.
- The top-level namespace holds the modelling workflow only. Training
  kernels, penalty helpers, losses, transforms, registries, UI classes,
  `fold`, `annealing_schedule` and `predict_bucket` are imported from their
  submodules (for example `jaxhybridmodels.penalties.box_grid`).
- `trainable_mask` freezes `BoundScaler` subtrees by default; pass
  `train_scalers=True` to train scaler temperatures. `frozen_default_mask`
  is removed.
- `BoundedPredictor` accepts any mapping of array-like values, or an
  array-like vector, and can build its scalers from `input_bounds` /
  `output_bounds`.
- Profiles and `annealing_schedule` accept array-like arguments and return
  JAX arrays.
- Dense prediction keeps each experiment's initial time when the requested
  grid starts later.
- **Breaking.** `loss`, `channel_idx`, `channel_weights`, `penalty_weight`,
  `penalty_points`, `penalty_fn`, `trajectory_penalty_fn` and
  `trajectory_penalty_weight` are gone from `OptaxTrainingConfig` and
  `EvosaxTrainingConfig`; they live on `Objective`. `channel_idx` is
  replaced by `channel_weights` keyed by channel name.
  `TrainingPhase.penalty_weight` defaults to `None`, meaning the
  objective's weight, and overrides it when set.
- **Breaking.** `penalty_points` is a mapping from predictor path to a
  `[G, n_inputs]` array, not a tuple in traversal order. An unknown path
  raises.
- **Breaking.** `EvosaxTrainingConfig(steps=...)` replaces
  `num_generations`. An Evosax fit fills `FitHistory.loss` with the
  best-so-far loss per generation; `FitHistory.best_fitness` is removed.
  `EvosaxUI.on_run_start` takes `total_steps` and `on_run_end` takes
  `final_loss`, as in `TrainingUI`.
- **Breaking.** `HybridModel` takes keyword arguments only and is an
  `eqx.Module`, so `dataclasses.replace` no longer works on it.
- **Breaking.** `ChannelObs` takes keyword arguments only.
- **Breaking.** A `BoundedPredictor` with one output bound returns a
  scalar instead of shape `(1,)`. A one-component `BoundScaler` keeps the
  shape of its input.
- **Breaking.** One bounds format everywhere: `Parameter(bounds=)`,
  `BoundedPredictor(input_bounds=, output_bounds=)` and `BoundScaler` take
  one `(low, high)` pair or a sequence of pairs. `BoundScaler((270.0,
  350.0))` works.
- **Breaking.** `evaluate_predictor(predictor, inputs)` returns an array
  and vectorises over array inputs, instead of returning a float.
- **Breaking.** `compute_metrics(predictions)` takes the `PredictionResult`
  from `predict`, not bucket arrays and a dataset. `print_metrics` accepts
  a `PredictionResult` too.
- **Breaking.** `freeze_modules_of_type(mask, cls)` and
  `freeze_where(mask, fn)` no longer take the predictors.
- **Breaking.** `make_dataset` raises on duplicate `exp_id` and names empty
  IDs `experiment_<i>`, adding a `_<n>` suffix when an explicit ID already
  uses that name.
- **Breaking.** `make_bootstrap_dataset` renames an experiment's second and
  later draws `<exp_id>#2`, `<exp_id>#3` and so on, skipping names already
  taken, so predictions on a resample carry these IDs.
- **Breaking.** `OptaxTrainingConfig` and `EvosaxTrainingConfig` take
  keyword arguments only. `TrainingPhase` takes `steps` by position and the
  rest by keyword.
- **Breaking.** `EvosaxUI.on_generation_end` takes `best_loss` and
  `mean_loss` instead of `best_fitness` and `mean_fitness`. The Rich
  Evosax dashboard labels its columns and its summary line in loss terms
  ("final loss", not "final fitness").
- **Breaking.** `save_run` writes a callable `loss` as
  `{"__callable__": "module.qualname"}`, like every other callable field,
  instead of a bare string, and `load_run(objective_cls=...)` raises on a
  section that holds one. `penalty_points` grids are saved with their
  shape, so an empty grid reloads as `[0, n_inputs]`.
- `experiments_from_dataframe` raises on a missing experiment ID instead of
  dropping those rows.
- `train_with_optax`, `train_with_evosax`, the ensemble helpers,
  `predict_dataset`, `predict_dense` and `ensemble_predictions` raise a
  `TypeError` naming `make_dataset` when given something other than a
  `Dataset`. `compute_metrics` does the same for anything but a mapping of
  predictions. `Objective(penalty_points=...)` raises a `TypeError` for the
  old tuple form, and a `BoundedPredictor` missing an input names the
  inputs it needs.

Migration for code written against the earlier API:

| Old | New |
| --- | --- |
| `OptaxTrainingConfig(steps=500, loss="bal_mse", penalty_weight=1e-3)` | `OptaxTrainingConfig(steps=500)` and `fit(..., objective=hm.Objective(loss="bal_mse", penalty_weight=1e-3))` |
| `channel_idx=(0,), channel_weights=(1.0,)` | `Objective(channel_weights={"conc": 1.0})` |
| `penalty_points=(grid_a, grid_b)` | `Objective(penalty_points={"growth": grid_a, "nucleation": grid_b})` |
| `EvosaxTrainingConfig(num_generations=200)` | `EvosaxTrainingConfig(steps=200)` |
| `result.history.best_fitness` | `result.history.loss` |
| `HybridModel(predictors, simulate_fn, ...)` | `HybridModel(predictors=..., simulate_fn=..., ...)` |
| `dataclasses.replace(model, predictors=p)` | `eqx.tree_at(lambda m: m.predictors, model, p)` |
| `ChannelObs(ts, values)` | `ChannelObs(ts=ts, values=values)` |
| `rate(inputs)[0]` | `rate(inputs)` |
| `jax.vmap(lambda T: rate({"T": T})[0])(temps)` | `hm.evaluate_predictor(rate, {"T": temps})` |
| `input_bounds=((0.0, 1.0),)` | `input_bounds=(0.0, 1.0)` (both still work) |
| `compute_metrics(predict_dataset(...), dataset)` | `compute_metrics(model.predict(data))`, or `compute_metrics(PredictionResult.from_buckets(buckets, dataset))` |
| `print_metrics(compute_metrics(predictions))` | `print_metrics(predictions)` |
| `freeze_modules_of_type(mask, predictors, cls)` | `freeze_modules_of_type(mask, cls)`, or `fit(freeze=(cls,))` |
| `model.fit(make_dataset(exps, ...), ...)` | `model.fit(exps, ...)` |
| `model.simulate(ts=..., y0=exp.y0, covariates=exp.covariates, t0=exp.t0)` | `model.simulate(experiment=exp, ts=...)` |
| `SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=4096, dt0=None)` | `SolverConfig()` |

### Fixed

- A training run that raised after its progress display opened, for
  example on an unknown channel name or a compile error, left the Rich
  dashboard running. The objective is now checked before the display
  opens, and any later error closes it with `on_run_end(final_loss=nan)`.
- `annealing_schedule("warmup_cosine", ...)` reached `end_value`
  `warmup_epochs` steps early, and raised when `total_epochs` was twice
  `warmup_epochs`.
- The Rich dashboards end their final frame with a newline when output is
  piped, so the next print no longer lands on the panel border.

### Documentation

- Guide pages, example pages and example scripts rewritten for the new API.
  Examples train through `HybridModel.fit` unless they teach the lower
  layer, and quoted results come from fresh runs.
- Examples are plain scripts with one docs page each, all in the same
  layout. The notebook scripts, the batch reactor slide figures and the
  example SPEC files are gone; their content is on the pages. The two
  pendulum scripts are merged into one quickstart.
- New tubular reactor example: an advection-dispersion-reaction PDE solved
  by the method of lines, with a learned rate law and four sensors.
- New `batch_reactor/compare_baselines.py`: per-pH Arrhenius fits against
  the hybrid, scored on held-out runs. The RL example now scores against
  the true rate law and the noise-limited optimum, and its committed trunk
  artefact is regenerated from the current `train_hybrid.py`.

## 0.2.0b1 — 2026-09-18

First public beta of the current v1 architecture.

- Publish the `jax-hybridmodels` distribution; Python imports are
  `jaxhybridmodels`.
- Support bucketed-irregular experiments, user-written ODE dynamics, bounded
  predictors, Optax training, and Evosax training.
- Validate the public KAN import on the supported JAX 0.10.x / Flax 0.12.8
  compatibility window.
- Include trajectory-aware penalties, profiles, schedules, ensembles,
  serialisation, and public training kernels.
- Support Python 3.11–3.14 in the release CI matrix.

Citation and Zenodo metadata are intentionally deferred until the project is
ready for publication.
