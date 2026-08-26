# `simulate_fn` is a user-supplied callable with a mandatory signature

Instead of subclassing a framework-provided `ODEModel` or implementing a `predict(batch)` protocol, users write a pure function with a fixed signature:

```python
def simulate_fn(
    predictors,                           # PyTree[eqx.Module], convention: tuple of BoundedPredictor leaves
    ts: Float[Array, "T"],
    covariates: dict[str, Array],          # constant-in-time per-experiment scalars
    y0: Float[Array, "S"],
    solver: SolverConfig,
) -> Float[Array, "T S"]: ...
```

The framework passes this function into `train_with_optax` / `train_with_evosax` / `predict_*`, vmaps it across each bucket, jits the resulting closure, and routes gradients through `predictors`. The user owns physics inside `simulate_fn` (the diffrax call, the vector field, any state-to-state lifting, any algebraic combinators like supersaturation polynomials, exp/log transforms, masks). The framework owns shape, jit, and grad mechanics.

The first argument is a pytree of `eqx.Module` leaves, not a single `Predictor`. This is runtime-permissive. Tuple, list, dict, NamedTuple, or a single Module all type-check. The canonical convention shown in CONTEXT.md and `examples/crystallisation/train_kinetic.py` is a tuple, single-predictor case = `(BP,)`. See [ADR-0006](./0006-predictors-as-pytree.md) for the contract-vs-convention split.

Inside the function, the user typically unpacks the tuple (`growth, nucleation = predictors`) and constructs per-call predictor input dicts that mix `covariates` with state-derived and exogenous time-dependent values (e.g. `inputs = {**covariates, "supersaturation": y[CONC_IDX] / covariates["c_sat"]}`). This collapses "covariate input", "state-derived input", and "exogenous time-dependent input" into one mechanism. The predictor sees a `dict[str, Array]` and is unaware of provenance.

## Why this is non-obvious

A future reader may expect a `class HybridODEModel(eqx.Module)` for users to subclass, the conventional shape. The function-first pattern was chosen because (1) physics is genuinely user-specific and varies wildly across applications (crystallisation moments vs. pendulum dynamics vs. closed-form maps), (2) a class hierarchy would either be too narrow (force ODE structure) or too wide (degenerate to "Module with `__call__`"), and (3) closing the simulate function over the user's own helpers is more natural than wrapping them in a class. The signature is the contract; the body is theirs.

A reader may also expect `predictors` to be a single `Predictor` (or `BoundedPredictor`), the simplest case. The pytree contract was chosen because (1) multi-rate models (nucleation + growth, multiple reactions, etc.) are common and the source package's `RatePair` wrapper bought nothing, since composition by `tuple` unpacking in the vector field is just as clear and zero framework code, (2) trainability mask, tournament re-init, and serialisation all already operate on pytrees natively (`eqx.partition`, `eqx.tree_serialise_leaves`, `jax.tree_util.tree_map`), so widening the contract is free, (3) a single Module is a one-leaf pytree, so the trivial case still type-checks without special handling.

## Considered alternatives

- `class ODEModel(eqx.Module)` with subclassed `vector_field`. Rejected for inheritance ceremony and forcing ODE shape on non-ODE models.
- Passing `simulate_fn` as a static field on a `HybridModel` wrapper. Rejected with the no-`Model`-class decision (ADR-0001).
- A `Predictor.simulate(ts, covariates, y0, solver)` method. Rejected because it mixes the trainable component with physics evaluation and breaks composition (NeuralNPolynomial as a vector field component would be impossible).
- First arg = single `Predictor`, with a separate `RatePair` wrapper for multi-rate cases. Rejected because `RatePair` was a domain-flavoured composition wrapper that doesn't earn its keep when the user already owns the vector field. Two predictors as a tuple, unpacked by the user, is one fewer class with the same expressive power.
- First arg = `dict[str, Predictor]` (named, fixed structure). Rejected for being one of several valid shapes. Locked as the *runtime contract* it would forbid users who prefer a NamedTuple or a single Module. Locked as *convention* it feels arbitrary; tuple is simpler and shorter for the typical 2-predictor case.
