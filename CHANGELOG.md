# Changelog

## Unreleased

- Add `HybridModel` fitting, named predictions, direct simulation, and typed
  fit/history results, with optional Optax validation selection.
- Add fixed `ConstantPredictor.value` and trainable
  `BoundedPredictor(value=...).value` without dummy inputs.
- Support missing/infinite bounds, one-sided softplus output transforms,
  and unbounded identity transforms.
- Accept scalar Optax configurations and `TrainingPhase` objects alongside
  existing tuples. Freeze constants and scaler temperatures by default.
- Add explicit experiment initial times and direct initial states. Preserve
  the original integration start when predicting on a later dense grid.


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
