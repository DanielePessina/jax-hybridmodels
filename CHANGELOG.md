# Changelog

## Unreleased

### Added

- `HybridModel` binds predictors, simulator, observation map, channel names
  and solver. `fit` returns a `FitResult` with the new model, a `FitHistory`
  (`loss`, `penalty`, `phase`, `validation_loss`, `best_fitness`),
  `best_step` and `stop_reason`. `predict` returns predictions keyed by
  experiment ID and channel name, with observed values, masks and an
  optional pandas export. `simulate` runs new conditions without a dataset.
- `validation=` on `HybridModel.fit` (Optax): the validation loss picks the
  returned step and drives `patience`.
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

### Changed (breaking against 0.2.0b1)

- `OptaxTrainingConfig` stores `phases`, a tuple of `TrainingPhase`. The
  parallel `steps`/`lr`/`optimizer`/`reset_optimiser_state`/
  `length_schedule`/`penalty_weight` tuples are gone. `length_schedule` is
  now `TrainingPhase.horizon` and `reset_optimiser_state` is
  `TrainingPhase.reset_optimizer`. Penalty weights are set per phase.
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
