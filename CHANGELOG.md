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
  penalty. `fit`, `fit_ensemble` and the trainers in
  `jaxhybridmodels.training` take it as `objective=`. Channel names and penalty paths are
  checked before anything compiles.
- `validation=` on `HybridModel.fit`, for Optax and Evosax: the validation
  loss picks the returned step or generation and drives `patience`.
- `TrainingPhase.lr` accepts an optax schedule, counted from the start of
  the phase. Chain several with `optax.join_schedules`.
- `fit(freeze=...)` freezes by dotted path, module class or predicate on
  top of the default mask. `jaxhybridmodels.trainable.freeze_mask(predictors,
  freeze, *, mask=None)` builds the same mask by hand and chains onto a base
  mask through `mask=`.
- `simulate(experiment)` reruns an experiment's `y0`, `t0` and
  covariates. `ts` defaults to the experiment's time axis: its measured
  times, plus `t0` when set. `y0` stays at the experiment's `t0`, or its
  first measured time when `t0` is unset, whatever `ts` is passed.
  `n_points` gives an even grid over the experiment's span, as in
  `predict`.
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
- `HybridModel.fit_ensemble(data, config, *, key, n_members, n_seeds=None,
  bootstrap=False, ...)` trains an ensemble and returns its `FitResult`s,
  best first. A seed ensemble scores `n_seeds` initialisations (default
  `n_members`) and fully trains the best `n_members`. A bootstrap ensemble
  trains one member per resample, the best of `n_seeds` seeds (default 1).
  Optax only. It raises `RuntimeError` when every seed fails.
- `HybridModel.predict(data, *, ts=None, n_points=None)`: `n_points` gives
  each experiment an even grid over its own span, `ts` one grid for all.
  Off the measured grid, `observed` and `observation_mask` are empty.
- `hm.predict_ensemble(models, data, *, ts=None, n_points=None)` returns a
  `PredictionResult` of the member mean. `models` holds `HybridModel`s or
  the `FitResult`s from `fit_ensemble`. `ExperimentPrediction.std` holds
  the per-channel spread (`ddof=0`), `PredictionResult.members` each
  model's own result, and `to_dataframe()` gains a `std` column.
  `compute_metrics` scores the mean.
- `FitResult.final_loss`: the training loss at the returned parameters.
- `hm.save(path, run, *, config=None, objective=None, extras=None)` writes
  a `HybridModel` or a `FitResult` (with its `history`, `best_step` and
  `stop_reason`). `hm.load(path, *, like)` returns a `SavedRun` whose
  `model` is `like` with the saved predictors. `like.solver` is kept,
  since the saved solver record holds class names but not constructor
  arguments; `load` warns when the two differ. A solver outside the
  registries saves as its `repr` instead of failing. `config` and
  `objective` come back typed; a section naming a callable comes back as
  its raw dict.
- Result types (`FitResult`, `FitHistory`, `PredictionResult`,
  `ExperimentPrediction`, `SavedRun`) live in `jaxhybridmodels.results`
  and are re-exported at the top level.
- `jaxhybridmodels.penalties.bounded_predictor_paths` lists the path of
  every `BoundedPredictor`, the keys `Objective.penalty_points` takes.
- `TrainingPhase.lr` accepts any callable. One JAX cannot trace, such as a
  Python `if` on the step, is evaluated step by step on the host.
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
  `fold` and `annealing_schedule` are imported from their submodules (for
  example `jaxhybridmodels.penalties.box_grid`). So are the trainers
  `train_with_optax` and `train_with_evosax` (`jaxhybridmodels.training`),
  `freeze_mask` (`jaxhybridmodels.trainable`) and `save_predictors` /
  `load_predictors` (`jaxhybridmodels.serialise`).
- `trainable_mask` freezes `BoundScaler` subtrees by default; pass
  `train_scalers=True` to train scaler temperatures. `frozen_default_mask`
  is removed.
- `BoundedPredictor` accepts any mapping of array-like values, or an
  array-like vector, and can build its scalers from `input_bounds` /
  `output_bounds`.
- Profiles and `annealing_schedule` accept array-like arguments and return
  JAX arrays.
- `predict(ts=...)` keeps each experiment's initial time when the requested
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
- **Breaking.** `freeze_paths`, `freeze_where` and `freeze_modules_of_type`
  are removed. `jaxhybridmodels.trainable.freeze_mask(predictors, freeze, *,
  mask=None)` takes paths, classes and predicates; `fit(freeze=...)` is the
  usual way.
- **Breaking.** `trainable_mask(predictors, *, predicate=...,
  train_scalers=...)` takes `predicate` by keyword, and
  `count_trainable_params(predictors, *, mask=None)` takes `mask` by
  keyword, defaulting to `trainable_mask(predictors)`.
- **Breaking.** `train_with_optax` and `train_with_evosax` take
  `(model, data, config, *, key, objective=None, freeze=None,
  trainable=None, ui=None, validation=None)`, the arguments of `fit`, and
  return a `FitResult`. `data` and `validation` accept a `Dataset` or a
  list of experiments. `fit` dispatches to them.
- **Breaking.** `train_seed_ensemble` and `train_bootstrap_ensemble` are
  removed; use `fit_ensemble`. A bootstrap ensemble with several seeds now
  keeps one member per resample instead of every seed, and an ensemble
  whose seeds all fail raises instead of returning the untrained input.
- **Breaking.** `predict_dataset`, `predict_dense`, `ensemble_predictions`,
  `predict_bucket` and `PredictionResult.from_buckets` are removed; use
  `predict` and `predict_ensemble`.
- **Breaking.** `save_run` / `load_run` are replaced by `save` / `load`.
  `metadata.json` stores the config as `config: {kind, fields}` instead of
  `optax_config` / `evosax_config` and `loss_history_kind`, and records
  `output_channel_names`, which `load` checks against `like`. Runs written
  by `save_run` do not load. `load_predictors(path, *, like)` takes the
  template by keyword.
- **Breaking.** `training.kernels.build_apply_update(*, optimizer,
  trainable)` is keyword-only, like the other builders.
  `build_dataset_score`'s `channel_names=` is now `output_channel_names=`,
  the name the rest of the API uses.
- **Breaking.** `BoundScaler(bounds, *, ...)`: every option after `bounds`
  is keyword-only. `step_profile(*, before, after, jump_at)` and
  `ramp_profile(*, t0, t1, v0, v1)` are keyword-only.
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
- **Breaking.** `save` writes a callable `loss` as
  `{"__callable__": "module.qualname"}`, like every other callable field,
  instead of a bare string. `penalty_points` grids are saved with their
  shape, so an empty grid reloads as `[0, n_inputs]`.
- `experiments_from_dataframe` raises on a missing experiment ID instead of
  dropping those rows.
- `compute_metrics` raises a `TypeError` for anything but a mapping of
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
| `compute_metrics(predict_dataset(...), dataset)` | `compute_metrics(model.predict(data))` |
| `predict_dense(predictors, dataset, ..., ts_grid=ts)` | `model.predict(data, ts=ts)`, or `model.predict(data, n_points=200)` for a per-experiment grid |
| `train_with_optax(predictors, dataset, config, simulate_fn=..., ...)` returning `(history, trained)` | `model.fit(data, config, key=key)` returning a `FitResult` |
| `train_seed_ensemble(predictors, dataset, config, ..., n_seeds=16, k_best=4)` | `model.fit_ensemble(data, config, key=key, n_members=4, n_seeds=16)` |
| `train_bootstrap_ensemble(..., n_bootstraps=10)` | `model.fit_ensemble(data, config, key=key, n_members=10, bootstrap=True)` |
| `ensemble_predictions([p for _, p in ranked], dataset, ...)` | `hm.predict_ensemble(results, data)` |
| `freeze_paths(mask, paths)`, `freeze_where(mask, fn)`, `freeze_modules_of_type(mask, cls)` | `fit(freeze=...)`, or `freeze_mask(predictors, freeze, mask=mask)` from `jaxhybridmodels.trainable` |
| `count_trainable_params(predictors, mask)` | `count_trainable_params(predictors, mask=mask)` |
| `save_run(path, predictors=..., solver=..., optax_config=...)` | `hm.save(path, result, config=config, objective=objective)` |
| `load_run(path, predictors_template=..., optax_cls=...)` | `run = hm.load(path, like=build_model())`; `run.model`, `run.history`, `run.config` |
| `hm.save_predictors` / `hm.load_predictors(path, template)` | `hm.save(path, model)` / `hm.load(path, like=model).model`, or `jaxhybridmodels.serialise.load_predictors(path, like=template)` |
| `build_apply_update(optimizer, mask)` | `build_apply_update(optimizer=optimizer, trainable=mask)` |
| `step_profile(a, b, t)` | `step_profile(before=a, after=b, jump_at=t)` |
| `print_metrics(compute_metrics(predictions))` | `print_metrics(predictions)` |
| `model.fit(make_dataset(exps, ...), ...)` | `model.fit(exps, ...)` |
| `model.simulate(ts=..., y0=exp.y0, covariates=exp.covariates, t0=exp.t0)` | `model.simulate(exp, ts=...)` |
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
- The tournament re-initialised frozen leaves along with the trainable
  ones, so a subtree frozen with `freeze=` or `trainable=` (and the
  default-frozen `BoundScaler`s) was held fixed at random values. Frozen
  leaves now keep their original values in `fit` and `fit_ensemble`.
- `fit_ensemble` ranks a member whose final loss is NaN last instead of
  scrambling the order of the whole list.

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
