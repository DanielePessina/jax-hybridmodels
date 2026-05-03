# `simulate_fn` is a user-supplied callable with a mandatory signature

Instead of subclassing a framework-provided `ODEModel` or implementing a `predict(batch)` protocol, users write a pure function with a fixed signature:

```python
def simulate_fn(predictor, ts, covariates: dict[str, Array], y0, solver: SolverConfig) -> Array["T S"]: ...
```

The framework passes this function into `train_with_optax` / `train_with_evosax` / `predict_*`, vmaps it across each bucket, jits the resulting closure, and routes gradients through `predictor`. The user owns physics inside `simulate_fn` (the diffrax call, the vector field, any state-to-state lifting); the framework owns shape, jit, and grad mechanics.

## Why this is non-obvious

A future reader may expect a `class HybridODEModel(eqx.Module)` for users to subclass — that's the conventional shape. The function-first pattern was chosen because (1) physics is genuinely user-specific and varies wildly across applications (crystallisation moments vs. pendulum dynamics vs. closed-form maps), (2) a class hierarchy would either be too narrow (force ODE structure) or too wide (degenerate to "Module with `__call__`"), and (3) closing the simulate function over the user's own helpers is more natural than wrapping them in a class. The signature is the contract; the body is theirs.

## Considered alternatives

- `class ODEModel(eqx.Module)` with subclassed `vector_field` — rejected for inheritance ceremony and forcing ODE shape on non-ODE models.
- Passing `simulate_fn` as a static field on a `HybridModel` wrapper — rejected with the no-`Model`-class decision (ADR-0001).
- A `Predictor.simulate(ts, covariates, y0, solver)` method — rejected because it mixes the trainable component with physics evaluation and breaks composition (NeuralNPolynomial as a vector field component would be impossible).
