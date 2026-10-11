# Troubleshooting

## The solver fails or hits `max_steps`

Check that:

- `simulate_fn` returns `[T, S]` (or a `diffrax.Solution` holding it) and
  `state_to_output` returns `[T, D]`, with `D` matching
  `output_channel_names`. `fit`, `predict` and `simulate` check this and
  name the function at fault with its actual and expected shape;
- `y0` has shape `[S]`;
- every state-derived predictor input stays inside the domain of its warp
  (a `log10` input must stay positive);
- the initial output of each predictor is plausible
  ([Centre the box](/guide/recommendations#centre-the-box-on-your-guess)).

If only some seeds fail, turn on the
[tournament](/guide/training#the-tournament) or zero the readout with
`with_zero_final_head()`. If every seed fails, the problem may be stiff:
try `diffrax.Kvaerno3()`, per-state `atol`, or x64. See
[Recommendations](/guide/recommendations#solver).

## `TracerBoolConversionError`

A traced array reached a Python `if`:

```python
if temperature > 30.0:   # fails inside simulate_fn or the vector field
    ...
```

Use `jnp.where`, `jax.lax.cond`, or a [profile](/guide/profiles-and-schedules).

## `NonConcreteBooleanIndexError`

Boolean indexing such as `x[mask]` has a data-dependent shape, which JAX
cannot trace. Keep the shape fixed and mask with `jnp.where`.

## A predictor gets the wrong shape

For a `BoundedPredictor`, the number of `input_keys`, the input bounds and
the inner `in_size` must agree, as must the output bounds and `out_size`.
With one output bound the output is a scalar; with several it is rank-1.
Code written for the old rank-1 output must drop its `[0]`. A vector
covariate must be indexed into named entries, or passed whole as a rank-1
array.

## The loss is `nan` or `inf`

Find the first non-finite value, checking in this order: `y0` and
covariates, predictor inputs, predictor outputs, the ODE state, the
outputs of `state_to_output`, and the observation variances (`"mle"`
losses divide by them). Debug with `verbose=False` on a dataset of one or
two experiments. A guarded division is the usual culprit; see
[Traps](/guide/recommendations#guard-the-divisor-not-the-result).

## Training recompiles more than expected

Each bucket shape compiles once. Further compiles mean something static
changed: a static predictor field, the solver config, or the identity of
`simulate_fn` or `state_to_output`. Define those functions once at module
level; a closure rebuilt per call is a new function to JAX.

## A saved run does not load

The template passed to `load_run` or `load_predictors` must be built with
the same container, predictor classes, sizes, bounds, warps and activation
names as the saved model. `load_run` names the part that differs. See
[Saving and loading](/guide/serialization).
