# Model interface

`HybridModel` holds the model pieces together. Its methods call the same
functional training and prediction machinery used by custom loops.

```python
model = hm.HybridModel(
    predictors=predictors,
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    output_channels=("concentration",),
    solver=solver,
)
result = model.fit(dataset, steps=1000, lr=1e-3, key=key)
fitted = result.model
predictions = fitted.predict(dataset)
experiment = predictions["experiment_A"]
concentration = experiment.outputs["concentration"]
```

`experiment.times`, `experiment.observed`, and `experiment.observation_mask`
keep predictions aligned with measurements. Unobserved values are NaN.
The result preserves input experiment order across different buckets. Empty
IDs receive `experiment_<index>` names; duplicate names raise an error.
Use `predictions.buckets` with existing metrics or JAX code, and
`predictions.to_dataframe()` for a long table (requires optional pandas).

Simulate new conditions without constructing observations:

```python
trajectory = fitted.simulate(
    ts=jnp.linspace(1.0, 10.0, 100), t0=0.0,
    y0=jnp.array([1.0]), covariates={"temperature": 300.0},
)
concentration = trajectory["concentration"]
full_state = trajectory.state
```

The initial state belongs to `t0`, which defaults to the first requested time.
The simulator must integrate from the first timestamp it receives; the
method prepends t0 when needed and removes that point from its result.

`hm.fit(model, ...)` and `hm.predict(model, dataset)` are functional
equivalents. Existing `train_with_*` and `predict_dataset` calls remain
available without constructing a model wrapper.

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

The framework vmaps this function over a bucket, so your callback receives one
trajectory at a time with shape `[T, S]`. Return `[T, D]`; the leading bucket
dimension is added by the framework. Keep this function separate from the
dataset so the same data can be used with different observation maps.

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

## Time-varying inputs

Covariates stay constant during an experiment. For a quantity that changes
continuously or in a step, create a pure-JAX profile and evaluate it inside
the vector field. The profile parameters still travel as ordinary covariates.

```python
temperature = hm.ramp_profile(
    t0=covariates["heat_start"],
    t1=covariates["heat_end"],
    v0=covariates["temperature_initial"],
    v1=covariates["temperature_final"],
)

def vector_field(t, y, args):
    inputs = {"temperature": temperature(t)}
    rate = predictors[0](inputs)
    return physics_rhs(t, y, rate)
```

See [Profiles and schedules](/guide/profiles-and-schedules) for the built-in
profile factories and the custom-loop schedule helper.
