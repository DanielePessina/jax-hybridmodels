# No `Model` wrapper class

A "model" is the loose triple `(predictor, simulate_fn, solver_config)` carried as separate top-level objects, not bundled in a class. The trainable `Predictor` is the only thing serialised; `simulate_fn` is code (re-imported); `SolverConfig` is JSON; `state_to_output` lives on `Dataset`. The source package's hierarchy of `BoundedRegressor → RateComponent → RateRegressorPair + simulate_ode` is collapsed because adding a wrapper class buys nothing — it conflates "trainable thing" with "physics evaluation" and forces every consumer to know about both.

## Why this is non-obvious

A future reader will look at the public API and wonder why there's no `Model` class to instantiate. The answer is that bundling them creates serialisation friction (closures inside the wrapper) and steals the user's ability to swap the simulate function independently of weights. The triple is honest about what's code vs. state.

## Considered alternatives

- `HybridModel(predictor, simulate_fn, solver_config)` dataclass — rejected because the wrapper adds no behaviour, only ceremony, and complicates serialisation.
- `Model.predict(batch) -> predictions` flat protocol — rejected because it forces ODE-style and closed-form models into one shape that fits neither well.
