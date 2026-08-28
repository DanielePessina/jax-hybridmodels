# No `Model` wrapper class

A "model" is the loose triple-plus carried as separate top-level objects, not bundled in a class: `predictors` (trainable pytree, serialised), `simulate_fn` (code, re-imported), `state_to_output` (code, passed to prediction/training per ADR-0008), and `SolverConfig` (JSON). The source package's hierarchy of `BoundedRegressor → RateComponent → RateRegressorPair + simulate_ode` is collapsed because adding a wrapper class buys nothing. It conflates "trainable thing" with "physics evaluation" and forces every consumer to know about both.

## Why this is non-obvious

A future reader will look at the public API and wonder why there's no `Model` class to instantiate. Bundling them creates serialisation friction (closures inside the wrapper) and steals the user's ability to swap the simulate function independently of weights. The triple is honest about what's code vs. state.

## Considered alternatives

- `HybridModel(predictor, simulate_fn, solver_config)` dataclass. Rejected because the wrapper adds no behaviour, only ceremony, and complicates serialisation.
- `Model.predict(batch) -> predictions` flat protocol. Rejected because it forces ODE-style and closed-form models into one shape that fits neither well.
