# Model interface

`hybridmodels` does not require a model wrapper class. You keep the model
pieces in your own code and pass them to prediction or training functions.

## The model pieces

| Piece | Type | Role |
| --- | --- | --- |
| `predictors` | Equinox PyTree | Trainable functions used by the ODE. |
| `simulate_fn` | Callable | Integrates one experiment. |
| `state_to_output` | Callable | Maps the full state to observed channels. |
| `solver` | `SolverConfig` | Diffrax solver settings. |

The dataset is separate from the model. It contains observations, masks,
initial states, and covariates. It does not contain `state_to_output`.

## `simulate_fn`

Write one function with this signature:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    ...
```

The arguments are:

- `predictors`: one predictor or any PyTree of predictors;
- `ts`: the observation times for one experiment, with shape `[T]`;
- `covariates`: that experiment's time-constant scalar or vector values;
- `y0`: the full initial state, with shape `[S]`;
- `solver`: the `SolverConfig` supplied to the training or prediction call.

Return the full state trajectory with shape `[T, S]`. Construct the Diffrax
term and call the solver inside the function. The library applies `vmap`,
JIT compilation, and differentiation around this single-experiment function.

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    rate = predictors[0]({"temperature": covariates["temperature"]})

    def vector_field(t, y, args):
        del t, args
        return rate * y

    return solver.diffeqsolve(
        diffrax.ODETerm(vector_field), ts, y0
    ).ys
```

The function must be pure and compatible with JAX transformations. Do not
branch on traced array values with Python `if` statements.

## `state_to_output`

The simulator returns the full state. `state_to_output` selects the observed
channels:

```python
def state_to_output(state):
    return state[..., :2]
```

It receives either one trajectory `[T, S]` or, when vmapped by the framework,
the corresponding leading batch dimension. Keep this function separate from
the dataset so the same data can be used with different observation maps.

## Predictor containers

The conventional container is a tuple:

```python
predictors = (growth_predictor, nucleation_predictor)
```

Dictionaries, lists, `NamedTuple`s, and a single predictor also work. The
framework walks the PyTree leaves; it does not inspect the container type.

## Covariates

Covariates are constant during one experiment. A value may be a scalar or a
rank-1 vector:

```python
covariates = {
    "temperature_C": 25.0,
    "feed_composition": jnp.array([0.2, 0.5, 0.3]),
}
```

For a dataset, every experiment must use the same shape for a given key. The
vector is passed to `simulate_fn` unchanged. Use its components in the vector
field or pass the vector to an array-based custom predictor.

