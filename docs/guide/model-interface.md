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
result = model.fit(experiments, hm.OptaxTrainingConfig(steps=1000, lr=1e-3), key=key)
fitted = result.model
predictions = fitted.predict(experiments)
experiment = predictions["experiment_A"]
concentration = experiment.outputs["concentration"]
```

`fit`, `predict` and `fit(validation=...)` take a list of experiments or a
`Dataset`; a list is passed through `make_dataset` with the model's
`output_channel_names`. `solver` defaults to `SolverConfig()`: `Tsit5`,
`rtol=1e-5`, `atol=1e-7`, `max_steps=4096`, and Diffrax's choice of first
step.

`fit` returns a `FitResult`: the fitted `model`, its `history`, the
`best_step` that `restore_best` returned, and a `stop_reason` of
`"budget_exhausted"` or `"patience"`. The original model is unchanged.

`predict` returns a `PredictionResult` keyed by experiment ID, in dataset
order. Each entry has `times`, `outputs`, `observed` and `observation_mask`,
all aligned on the experiment's own time axis; `observed` is NaN where a
channel was not measured. Experiments without an ID are named
`experiment_<index>`, and duplicate IDs raise. `predictions.to_dataframe()`
returns a long table (needs pandas), and `predictions.buckets` keeps the raw
`[N, T, D]` arrays for custom JAX code.

`hm.compute_metrics(predictions)` returns per-channel statistics over the
measured cells, and `hm.print_metrics(predictions)` prints them as a table.

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

To rerun one experiment's conditions, pass it whole. Its `y0`, `t0` and
covariates are used, and `ts` defaults to its time axis: the measured times,
plus `t0` when set. `y0` stays at the experiment's `t0`, or at its first
measured time when `t0` is unset, so a `ts` that starts before that is
rejected:

```python
dense = fitted.simulate(experiment=experiments[0], ts=jnp.linspace(0.0, 10.0, 200))
```

`output_channel_names` must match the dataset's, in order.
`train_with_optax`, `train_with_evosax` and `predict_dataset` take the pieces
below as separate arguments instead of a model.

## The model is an Equinox module

`HybridModel` is an `eqx.Module` built with keyword arguments only.
`predictors` is its only array subtree; the other fields are static. Swap
part of it with `eqx.tree_at`:

```python
refit = eqx.tree_at(lambda m: m.predictors["rate"], model, hm.Parameter(0.5, bounds=(0.0, None)))
```

Build a batch of models under `eqx.filter_vmap`, one member per key:

```python
def build(key):
    return hm.HybridModel(predictors=make_predictors(key), simulate_fn=simulate_fn,
                          state_to_output=state_to_output, output_channel_names=("conc",))

members = eqx.filter_vmap(build)(jr.split(key, 8))   # array leaves gain a leading axis of 8
curves = eqx.filter_vmap(
    lambda m: m.simulate(ts=ts, y0=y0, covariates=conditions).outputs["conc"]
)(members)                                           # [8, T]
```

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

Return the full state at every `ts`, shape `[T, S]`, or the
`diffrax.Solution` whose `ys` holds it. The library vmaps, compiles and
differentiates this single-experiment function.

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    rate = predictors["rate"]({"temperature": covariates["temperature"]})

    def vector_field(t, y, args):
        return -rate * y

    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0)
```

The function must be pure and traceable: no Python `if` on array values.
`fit` checks the shapes of `simulate_fn` and `state_to_output` on the first
experiment before compiling, and a mismatch names the function, its shape
and the expected one.

When the simulation is one ODE solve, as here, write only the vector field
and let `from_vector_field` build `simulate_fn`:

```python
def vector_field(t, y, predictors, covariates):
    rate = predictors["rate"]({"temperature": covariates["temperature"]})
    return -rate * y


model = hm.HybridModel.from_vector_field(
    vector_field=vector_field,
    predictors=predictors,
    state_to_output=state_to_output,
    output_channel_names=("concentration",),
)
```

It solves with `solver.diffeqsolve` and saves at `ts`. Write `simulate_fn`
yourself for anything else: events, several solves, a per-experiment `dt0`,
or work outside the solve.

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
