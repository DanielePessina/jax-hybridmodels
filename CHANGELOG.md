# Changelog

## 0.3.0 — 2026-11-01

### Added

- `experiments_from_long_dataframe` loads a table with one row per measurement,
  keeping variance and covariates.
- `HybridModel.named_values()` returns every `Parameter` and `Constant` in physical
  units by dotted path, and `HybridModel.with_predictors` puts a custom loop's
  predictors back.
- `summary()` reports the loss change over the last 10% of steps, which shows
  whether a fit that ran out of budget had settled.
- With `verbose=True` and no terminal, training prints plain progress lines through
  `PlainTrainingUI` or `PlainEvosaxUI`, in place of the Rich dashboard.
- A failed fit names the experiments that failed, by re-solving with `throw` off, and
  says when more `max_steps` will not help.
- `BoundedPredictor(input_warp=, output_warp=)` warps a scaler built from bounds.
- `reinitialize` and `reinitialize_pytree` are top-level exports.
- `InferenceProblem.physical` takes leading sample axes.
- `FitResult.summary()` and `SavedRun.summary()` report how the run ended, the
  loss by phase, each `Parameter` with any near a bound, and how much of each
  network's input box is pinned against an output bound. Pass data to add the
  metrics and the worst experiment per channel.
- `compute_metrics(predictions, by="experiment")` scores each experiment alone,
  and `print_metrics` lists those worst first.
- `jaxhybridmodels.inference.make_problem` builds a log-density for a model and
  its data, for any sampler such as BlackJAX. It sums over buckets, adds the
  change-of-variables term for free `Parameter`s, and turns a failed solve into
  a rejected proposal. `estimate=` picks what is sampled, and what stays fixed is
  listed in `InferenceProblem.conditional_on`.
- `InferenceProblem.preconditioned` whitens the position with the Gauss-Newton
  curvature at the fit, and `make_problem(gradient=)` picks forward or reverse
  mode, which together take most of the cost out of a NUTS run.
- `InferenceProblem.predict` turns chain draws into a posterior predictive
  `PredictionResult`, with or without observation noise, and
  `InferenceProblem.to_arviz` exports them for diagnostics. The `bayes` extra
  installs BlackJAX and `arviz-base`.
- `predict_ensemble(quantiles=)` adds quantiles across members, also in
  `to_dataframe`.
- Multiple shooting: `TrainingPhase(segments=K)` splits each trajectory into `K`
  segments that start from trainable states, with a gap term weighted by
  `continuity_weight`. The schedule must end in single shooting.
  `FitHistory.continuity` records the gap term, and the summary's phase table
  shows it. See the multiple shooting example.
- `FitHistory.tournament_scores` keeps every tournament attempt's score, and the
  summary and the training display report the spread.
- `InferenceProblem.predict` drops a draw whose solve fails on the requested grid,
  with a warning.
- `SolverConfig(throw=False)` returns non-finite states from a failed solve
  instead of raising.
- `make_dataset(max_buckets=)` caps the number of compiled bucket shapes by
  padding shorter experiments with unobserved rows. Off by default, because it
  reweights the loss.
- `FitResult.stop_reason` gains `"diverged"`. A solver error or a non-finite
  loss after the first step no longer loses the run: `fit` warns and returns
  the best parameters so far. Failing on the first step still raises, and now
  includes a non-finite loss, which used to run the whole budget on NaN.
- `fit_ensemble(validation=)`, with the meaning it has in `fit`. Members are
  sorted by the validation loss.
- A `pandas` extra, `pip install jax-hybridmodels[pandas]`, for
  `PredictionResult.to_dataframe`, which names the extra when pandas is missing.
- An exception raised inside `simulate_fn` or `state_to_output` carries a note
  listing the predictors, covariates and `y0` shape it was given.
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
  Off the measured grid, `observed` and `mask` are empty.
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
- Frequentist uncertainty on `hm.inference.make_problem`, with no sampler:
  `optimize` finds the maximum likelihood or posterior mode with Optimistix's
  BFGS, `fisher` gives the Fisher information with physical standard errors,
  Wald intervals, correlations, the condition number and the directions the
  data cannot separate, and `laplace` gives a Gaussian posterior whose draws
  `physical`, `predict` and `to_arviz` read. `make_problem(log_prior=None)`
  builds a likelihood-only problem for this. `optimistix` is now a declared
  dependency; diffrax already required it.
- `InferenceProblem.profile` and `profiles`: profile likelihood intervals.
  Steps adapt as in pyPESTO, every other value is re-optimised with BFGS at
  each point, and a side the data leave open is reported as open. A profile
  starting from a value against its bound still walks away from it.
  `ProfileResult.at_bound` names the other values that were pinned at a bound
  along the walk, since the interval then holds only for those bounds.
- `FisherResult.sample` draws from the estimate's asymptotic sampling
  distribution, as `LaplaceResult.sample` draws from the posterior's Gaussian.
- `derived(fn)` on `FisherResult` and `LaplaceResult` gives the delta-method
  estimate and standard error of any function of the values, as a
  `DerivedQuantity` with one- or two-sided Wald intervals.
- `OptimizeResult.gradient` and `gradient_norm` show how flat the optimum is.
- Profiles list the times where they jump or bend in `jump_ts`, and
  `SolverConfig.diffeqsolve(jump_ts=)` ends a step on each. Without them an
  adaptive solver can step over a pulse shorter than a step.
- Fisher and Laplace report `nan` for the standard error and Wald interval of
  a value against its bound, mark it "at bound" in the summary, and warn.
- `examples/shelf_life/shelf_life.py` and its page: a label claim from an
  accelerated stability study, set from the profile likelihood's one-sided 95%
  lower bound on the extrapolated rate.
- `examples/identifiability/arrhenius.py`: how correlated Arrhenius constants
  are, and the reference temperature that decorrelates them.
- Evosax ranks a candidate whose solve fails last instead of ending the run,
  and `FitHistory.failed_candidates` counts them per generation. `summary()`
  gives the total. Only a generation where every candidate fails ends the run.
- A rewritten test suite built on one small shared decay model. It is
  mostly end to end and runs in about two minutes instead of twenty.

### Changed (breaking against 0.2.0b1)

- The training loss pools buckets by what each one averages over, so `mse`
  is the mean over every measurement and `bal_mse` the mean over every
  experiment, however the experiments fall into buckets. Each bucket used to
  count the same, which reweighted experiments by their grid length, and a
  bootstrap duplicate added no weight. `mle` is the summed likelihood. A
  custom loss declares how it pools with a `pooling` attribute. Losses in a
  multi-length fit change value. `build_bucket_step` and `build_score_bucket`
  take a `share`, and `build_dataset_score` a `pooling`. The trajectory
  penalty and the multiple-shooting gap term are means over every
  experiment, probes included.
- `TrainingPhase.horizon` cuts each experiment at that fraction of its own
  span, from `t0` (or its first measurement) to its last measurement, and
  always scores its first measurement. It used to count rows of the bucket's
  shared time grid, so a short horizon after `t0` could score nothing.
- The Fisher condition number and the not-identified test read the
  information scaled to a unit diagonal, so a change of bound or warp no
  longer moves them. A value the data barely see on its own, whose latent
  standard error passes 1e3 times its latent's size, is also listed as not
  identified, with an infinite standard error.
- A multiple-shooting phase uses the number of segments asked for, up to
  `T - 1`, with lengths that differ by at most one step.
- `annealing_schedule("exponential", ...)` needs `end_value > 0`. With 0 it
  used to be a straight line.
- `fit_ensemble` starts every member from a fresh draw. It used to keep the
  model as given for one member, and for every bootstrap member when
  `n_seeds=1`, which left the initialisation out of the spread. With one seed
  it no longer trains an unrecorded tournament.
- `FitResult.final_loss` is measured at the returned parameters: after the
  last update with `restore_best=False`, and at the chosen candidate for an
  Evosax fit with validation. It is kept in `FitResult.measured_loss`, and
  `fit_ensemble` ranks members on it.
- A phase that changes `lr` on the same raw `optax.GradientTransformation`
  raises even with `reset_optimizer=True`, since the instance keeps the rate
  it was built with. A reset now writes the phase's `lr` into an instance
  built with `optax.inject_hyperparams`.
- Measurement times are always stored as floats, so a float prediction grid
  on integer times is no longer floored.
- A log warp is allowed on one-sided bounds and acts on the distance from the
  bound: `Parameter(5000.0, bounds=(0.0, None), warp="log")` is
  `exp(z)`, so each Adam step is a relative change. With a softplus a value of
  5000 moved by about the learning rate per step.
- The tournament's first attempt trains from the predictors as given, so the
  starting guess is always a candidate; the others redraw as before.
- In a multiple-shooting phase with `horizon < 1`, continuity gaps past the
  horizon are left out of the gap term, as the data there are.
- `train_scalers=True` no longer trains a `Parameter`'s own temperature, which
  only duplicated its latent.
- `verbose` moved from `OptaxTrainingConfig` and `EvosaxTrainingConfig` to `fit`,
  `fit_ensemble`, `train_with_optax` and `train_with_evosax`. A run saved with the old
  field still loads.
- `length_mask_fraction` is now `horizon` on the kernels and `penalties` helpers.
  `build_bucket_step` and `build_score_bucket` return functions that default it to
  1.0 and do not recompile for a new float.
- `exp_id` is optional on `experiment_from_array`, and the three builders share an
  argument order.

- A channel with no observations now has NaN error statistics instead of 0.
- **Breaking.** `fit` and `fit_ensemble` raise `TypeError` unless `key` is a JAX
  PRNG key. `key=None` and `key=0` used to pass when no random numbers were
  drawn.
- **Breaking.** `tournament_attempts > 1` with `tournament_steps=0` raises
  instead of silently running no tournament. `tournament_steps > 0` with
  `tournament_attempts=1` warns in `fit`.
- **Breaking.** The tournament trains its attempts with the first phase's
  learning rate and horizon. `tournament_lr` defaults to None, meaning that
  rate, instead of 1e-4, at which 30 steps barely moved an attempt and the
  tournament ranked random initialisations. A tournament re-draws a
  `Parameter` only where it has two finite bounds; one without keeps its
  starting value instead of a draw near 1.
- `repr(FitResult)` is one line instead of the whole model and history.
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
  is keyword-only. `step_profile(*, at, before, after)` and
  `ramp_profile(*, start, end, before, after)` are keyword-only, and name
  the two levels alike.
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
| `step_profile(a, b, t)` | `step_profile(at=t, before=a, after=b)` |
| `ramp_profile(t0, t1, v0, v1)` | `ramp_profile(start=t0, end=t1, before=v0, after=v1)` |
| `print_metrics(compute_metrics(predictions))` | `print_metrics(predictions)` |
| `model.fit(make_dataset(exps, ...), ...)` | `model.fit(exps, ...)` |
| `model.simulate(ts=..., y0=exp.y0, covariates=exp.covariates, t0=exp.t0)` | `model.simulate(exp, ts=...)` |
| `SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=4096, dt0=None)` | `SolverConfig()` |
| `bp.y_observed`, `bp.yvar` in a custom loss | `bp.observed`, `bp.variance`, as on `ChannelObs` |
| `MLPPredictor(..., activation_name="tanh")` | `MLPPredictor(..., activation="tanh")` |
| `predictor.initialized_with_key(key)`, `reinitialize_with_key(p, key)`, `reinitialize_pytree_with_key(p, key)` | `predictor.reinitialized(key=key)`, `reinitialize(p, key=key)`, `reinitialize_pytree(p, key=key)` |
| `split_dataset(data, train=0.8, val=0.1, test=0.1, key=key)` | `split_dataset(data, train=0.8, validation=0.1, test=0.1, key=key)` |
| `EvosaxTrainingConfig(sigma_init=0.3)` | `EvosaxTrainingConfig(init_sigma=0.3)` |

### Fixed

- The inverse squash on a boxed input was exact only to first order, so a
  Hessian through a `BoundedPredictor` input that depends on the fitted
  values, as `fisher(curvature="exact")` and `laplace()` take, came out
  wrong by up to about 10%. It is now exact to every order inside the box.
- A profile of a value pinned at a one-sided bound jumped to a huge latent on
  its first step and reported a wrong or open far side.
- Evosax with `validation=` could return a search candidate that validated
  worse than the starting model. The start is now a candidate.
- With `restore_best=False` and validation, the summary's validation line is
  measured at the returned parameters (`FitResult.measured_validation`).
- A reset no longer overrides the rate of an `optax.inject_hyperparams`
  instance when the phase leaves `lr` at its default.
- `SolverConfig(adjoint=ForwardMode())` in training raises a clear error
  instead of a while-loop error from JAX.
- A segmented phase that changed only `continuity_weight` kept the previous
  phase's best loss, so patience could end it while it still improved.
- `simulate(experiment)` with x64 off rejected some `t0` values that float32
  rounds, and put `y0` at a channel the model does not output.
- `split_dataset(train=0.29)` on 100 experiments gave 28 for training.
- `experiments_from_dataframe` reads each run's rows in time order, so a
  `y0_fn` reading `values[0]` gets the first measurement, and raises on a
  covariate that is missing for a run.
- `load` names a precision mismatch: a run saved with x64 off refuses to load
  with it on, and the reverse warns that values are rounded.
- `evaluate_predictor` ignores inputs the predictor does not read.
- `InferenceProblem.conditional_on` names a fixed `Parameter` as `physical`
  does, without `.latent`.
- `KANPredictor` weights follow the x64 flag, and `grid_range` sets the
  interval its spline grid covers. It defaults to (-3, 3), which covers the
  inner 90% of a bounded input; jaxkan's (-1, 1) covered the middle 46%.
- Fisher and Laplace flagged a value with one finite bound as "at bound" when
  it sat far from that bound, and a profile of a value with only an upper
  bound returned its ends in reverse.
- The likelihood floored every variance at 1e-12, so data with a standard
  deviation below 1e-6 in their own units got intervals too wide. The floor
  is now the smallest positive float.
- A channel whose float64 times collide in float32, next to a float32
  channel, lost a measurement silently. It now raises.
- `predict_ensemble` leaves a member whose prediction is non-finite out of
  the mean, spread and quantiles, with a warning.
- `predict(observation_noise=True)` on a problem with a custom likelihood
  refuses, since it would draw Gaussian noise the likelihood does not define.
- The bound penalty's mean ran over uncharged one-sided outputs too, which
  diluted its weight.
- A log-warped input scaler with mixed boxed and one-sided inputs returned
  NaN for a non-positive input instead of raising.
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
- `evaluate_predictor` given an array, not a mapping, raised an unrelated
  `AttributeError`. It now raises a `TypeError` naming the predictor's inputs.
- Tournament restarts and `fit_ensemble` overwrote every leaf with N(0, 1)
  when the predictors sat in a user's `eqx.Module` container instead of a
  dict, so a `Parameter` lost its value and a scaler its temperature. A
  plain container is now walked into like a dict.
- The bound penalty charged an output with one finite bound whenever it came
  within about 0.05 of that bound, pushing small rates up whatever the data
  said. Only outputs with two finite bounds are charged now: near its edge, a
  one-sided output is log-like and does not saturate.
- `Parameter` accepted a value that rounds onto its bound in its own dtype,
  such as float32 0.1 against a bound of 0.1, leaving an infinite latent. It
  now compares in the value's dtype and refuses a non-finite latent.
- An array `Parameter` with one shared bound compiled in time linear in its
  size (17 s at 1000 entries); it is now one elementwise transform.
- A log-warped value could round one ulp past its bound; it is clipped.
- A learning-rate schedule written with NumPy, or as a dict lookup, failed at
  the phase start; it is evaluated on the host like other plain Python ones.
- An lr change on a raw `optax.GradientTransformation` without a reset is
  refused when the config is built, not after the earlier phases trained.
- `summary()` raised when every tournament attempt failed. A raw optimiser
  no longer warns about `tournament_lr` when it was not set.
- The API reference lists the methods and properties a class inherits from a
  private base, such as `FisherResult.standard_errors`, and leaves private
  fields out of signatures.

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
