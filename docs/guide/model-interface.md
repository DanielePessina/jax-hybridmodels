# Model interface

`HybridModel` binds the four model pieces. Its methods call the same
functions a custom workflow would call directly.

```python
model = hm.HybridModel(
    predictors=predictors,
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    output_channel_names=("concentration",),
    solver=solver,
)
result = model.fit(dataset, hm.OptaxTrainingConfig(steps=1000, lr=1e-3), key=key)
fitted = result.model
predictions = fitted.predict(dataset)
experiment = predictions["experiment_A"]
concentration = experiment.outputs["concentration"]
```

`fit` returns a `FitResult`: the fitted `model`, its `history`, the
`best_step` that `restore_best` returned, and a `stop_reason` of
`"budget_exhausted"` or `"patience"`. The original model is unchanged.

`predict` returns a `PredictionResult` keyed by experiment ID, in dataset
order. Each entry has `times`, `outputs`, `observed` and `observation_mask`,
all aligned on the experiment's own time axis; `observed` is NaN where a
channel was not measured. Experiments without an ID are named
`experiment_<index>`, and duplicate IDs raise. `predictions.buckets` holds
the raw `[N, T, D]` arrays for metrics code, and `predictions.to_dataframe()`
returns a long table (needs pandas).

Simulate new conditions without constructing observations:

```python
trajectory = fitted.simulate(
    ts=jnp.linspace(1.0, 10.0, 100), t0=0.0,
    y0=jnp.array([1.0]), covariates={"temperature": 300.0},
)
concentration = trajectory.outputs["concentration"]
full_state = trajectory.state
```

`y0` is the state at `t0`, which defaults to the first requested time. When
`t0` is earlier, `simulate` integrates from it and drops that point from the
result. `trajectory.state` is the full `[T, S]` state.

`output_channel_names` must match the dataset's, in order.
`train_with_optax`, `train_with_evosax` and `predict_dataset` take the pieces
below as separate arguments instead of a model.

## The model pieces

| Piece | Type | Role |
| --- | --- | --- |
| `predictors` | Equinox PyTree | Values and networks the ODE reads. |
| `simulate_fn` | Callable | Integrates one experiment. |
| `state_to_output` | Callable | Maps the full state to observed channels. |
| `solver` | `SolverConfig` | Diffrax solver settings. |

The dataset holds observations, masks, initial states and covariates, and
nothing model-shaped.

## `simulate_fn`

Write one function with this signature:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    ...
```

| Argument | Meaning |
| --- | --- |
| `predictors` | The predictors PyTree. |
| `ts` | This experiment's time axis, shape `[T]`. `ts[0]` is where `y0` applies. |
| `covariates` | This experiment's constant conditions, a dict of scalars or vectors. |
| `y0` | The full initial state, shape `[S]`. |
| `solver` | The model's `SolverConfig`. |

Return the full state at every `ts`, shape `[T, S]`. The library vmaps,
compiles and differentiates this single-experiment function.

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    rate = predictors["rate"]({"temperature": covariates["temperature"]})

    def vector_field(t, y, args):
        return -rate * y

    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys
```

The function must be pure and traceable: no Python `if` on array values.

## `state_to_output`

The simulator returns the full state. `state_to_output` selects the observed
channels:

```python
def state_to_output(state):
    return state[..., :2]
```

It receives one trajectory, `[T, S]`, and returns `[T, D]` with columns in
`output_channel_names` order. The bucket dimension is added by vmap.

## Predictor containers

A dict keyed by role reads best inside `simulate_fn`:

```python
predictors = {
    "growth": growth_network,
    "k_nucleation": hm.Parameter(1e-3, bounds=(1e-6, 1.0), warp="log10"),
    "density": hm.Constant(2.66),
}
```

Tuples, lists, `NamedTuple`s and a single module also work. The library walks
the PyTree leaves and never inspects the container type.

## Covariates

Covariates are constant during one experiment, scalars or rank-1 vectors,
and passed to `simulate_fn` unchanged. Every experiment in a dataset must use
the same keys and shapes. See [Data and buckets](/guide/data).

## Time-varying inputs

For a quantity that changes during an experiment, store its profile
parameters as covariates and evaluate a profile inside the vector field.

```python
temperature = hm.ramp_profile(
    t0=covariates["heat_start"],
    t1=covariates["heat_end"],
    v0=covariates["temperature_initial"],
    v1=covariates["temperature_final"],
)

def vector_field(t, y, args):
    inputs = {"temperature": temperature(t)}
    rate = predictors["rate"](inputs)
    return physics_rhs(t, y, rate)
```

See [Profiles and schedules](/guide/profiles-and-schedules) for the built-in
profiles.
