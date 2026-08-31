# Troubleshooting

Start with the shape and domain checks below. Most failures occur at the
boundary between user code and a JAX transformation.

## The solver fails during training

Check these conditions:

1. `ts` is finite, strictly increasing, and has at least two entries.
2. `y0` has shape `[S]`.
3. `simulate_fn` returns `[T, S]`.
4. `state_to_output` returns `[T, D]`.
5. `solver.max_steps` is large enough for the problem.
6. State-derived inputs satisfy the domain of their input warp.

If the problem is stiff, try an implicit solver or tighter state-specific
absolute tolerances. See [Recommendations](/guide/recommendations).

## `TracerBoolConversionError`

JAX is tracing an array value. Do not use that value in a Python conditional:

```python
# Do not do this inside simulate_fn or the vector field.
if temperature > 30.0:
    ...
```

Use `jnp.where`, `jax.lax.cond`, or a profile factory instead.

## `NonConcreteBooleanIndexError`

JAX cannot create a dynamically sized result from a traced boolean mask. Use
masked arithmetic with `jnp.where` and a fixed shape instead of boolean
selection inside a jitted function.

## A predictor receives the wrong shape

For `BoundedPredictor`, check:

- the number of input bounds;
- the order of `input_keys`;
- the inner predictor's `in_size` and `out_size`;
- the output scaler's number of bounds.

For a vector covariate, index its components in the vector field or pass a
rank-1 array to the predictor.

## Training produces a non-finite loss

Check the first invalid quantity in this order:

1. `y0` and covariates;
2. predictor inputs;
3. predictor outputs before the ODE update;
4. the full ODE state;
5. the projected observations;
6. the loss variance values.

Use `verbose=False` and a small dataset while isolating the first failing
bucket. The tournament retries only Diffrax runtime failures and non-finite
scores; shape and user-code errors are reported directly.

## The run recompiles more than expected

Compilation is per bucket shape. Keep the Python bucket loop outside your own
JIT boundary. Avoid changing static predictor fields, solver settings, or
callback identities during a run.

## A saved run does not load

`load_run` requires a template with the same predictor container, module types,
and static fields. Rebuild the template with the original bounds, warps,
activation names, and dimensions. Training configs containing executable
callables are returned as metadata markers and must be rebound explicitly.

