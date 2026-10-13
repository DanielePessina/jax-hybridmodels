# Data and buckets

Each measured channel has its own timestamps, so experiments may be sparse
and irregular. Regular data is the case where every channel shares one grid.

## Build an experiment

Use `ChannelObs` for one observed channel and `make_experiment` for one run:

```python
experiment = hm.make_experiment(
    covariates={
        "temperature_C": 25.0,
        "feed_composition": jnp.array([0.2, 0.5, 0.3]),
    },
    channels={
        "concentration": hm.ChannelObs(
            ts=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([1.0, 0.8, 0.6]),
            variance=0.01,
        ),
        "size": hm.ChannelObs(
            ts=jnp.array([0.5, 2.0]),
            values=jnp.array([4.0, 5.0]),
        ),
    },
    y0_fn=lambda covariates, channels: jnp.array([1.0, 0.0]),
    exp_id="run-01",
)
```

`ChannelObs` is keyword-only: `ts`, `values`, and an optional `variance`
(default 1.0, a scalar or one value per observation). `y0_fn(covariates,
channels)` returns the full initial state, which may include unobserved
components. Pass `y0=` instead when the state is known directly; give one or
the other.

`exp_id` names the experiment in predictions and must be unique within a
dataset. An experiment left without one is named `experiment_<i>` by its
position.

## From an array

When every channel shares one time grid, `experiment_from_array` takes a
`[T, D]` array with columns in `channel_names` order. A `NaN` cell means not
measured, so channels can still have different sample times:

```python
experiment = hm.experiment_from_array(
    exp_id="run-02",
    ts=jnp.array([0.0, 1.0, 2.0]),
    values=jnp.array([[1.0, jnp.nan], [0.8, 4.2], [0.6, 5.0]]),
    channel_names=("concentration", "size"),
    variance=(0.01, 1.0),            # one per channel; a scalar or [T, D] also work
    covariates={"temperature_C": 25.0, "feed_composition": jnp.array([0.2, 0.5, 0.3])},
    y0=[1.0, 0.0],
)
```

`covariates`, `t0`, `y0` and `y0_fn` mean the same as for `make_experiment`.

## From a table

`experiments_from_dataframe` builds one experiment per run from a wide pandas
table: one row per run and sample time, one column per channel, `NaN` where a
channel was not measured.

```python
experiments = hm.experiments_from_dataframe(
    df,
    time="time_h",
    experiment="run",
    channels={"concentration": "conc_g_L", "size": "d43_um"},   # {channel: column}
    covariates=["temperature_C"],                               # constant within a run
    variance={"concentration": "conc_var", "size": 1.0},        # column or number
    y0_fn=lambda covariates, channels: jnp.array([channels["concentration"].values[0], 0.0]),
)
```

A list names each channel or covariate after its column; a mapping renames.
`t0` may be one number or a column, and `y0` one shared state or a mapping
`{exp_id: state}`. Experiments come back in order of first appearance, with
the run column's values as `exp_id`. A long table (one row per measurement)
becomes wide with
`df.pivot_table(index=["run", "time_h"], columns="channel", values="value").reset_index()`.
pandas is imported only by your code.

## Union timestamps and masks

`HybridModel.fit` and `predict` accept a list of experiments and call
`make_dataset` for you. Call it yourself to split the data, inspect the
buckets, or reuse one dataset across many calls.

`make_dataset` merges each experiment's channel timestamps into one union
axis, places every channel on it, and builds a mask that is `True` only where
a value was measured.

```python
dataset = hm.make_dataset(
    experiments,
    output_channel_names=("concentration", "size"),
)
```

Nothing is interpolated. A probe channel gives `ts` with empty `values`: its
timestamps join the union axis and its mask is all `False`, so the experiment
is simulated but never scored.

## Buckets

A bucket holds the experiments whose union axes have the same length. Their
timestamp values and masks may differ. `dataset.bucket_payloads` holds one
`BucketPayload` per bucket, and `hm.describe_buckets(dataset)` summarises
them.

| Quantity | Shape |
| --- | --- |
| `bp.ts` | `[N, T]` |
| `bp.y_observed` | `[N, T, D]` |
| `bp.yvar` | `[N, T, D]` |
| `bp.mask` | `[N, T, D]` |
| scalar covariate | `[N]` |
| vector covariate | `[N, K]` |
| `bp.y0` | `[N, S]` |

Training loops over buckets in Python, and each bucket shape compiles once.
Fewer distinct lengths means less compile time.

## Validation

`make_experiment` and `make_dataset` reject inputs that would otherwise fail
inside a compiled solve:

- channel timestamps must be finite and must not repeat;
- observed `ts` and `values` must have matching lengths;
- variances must be finite and strictly positive;
- covariates must be scalars or rank-1 vectors;
- all experiments must use the same covariate keys and shapes;
- each experiment must have at least one timestamp;
- experiment IDs must be unique.

## Split and resample

Split at the experiment level. Each result is re-bucketed independently:

```python
train, validation, test = hm.split_dataset(
    dataset,
    train=0.8,
    val=0.1,
    test=0.1,
    key=jr.key(0),
)
```

`make_bootstrap_dataset` samples experiments with replacement.

## Initial time and initial state

Set `t0=` when the initial state is defined before the first measurement:

```python
experiment = hm.make_experiment(
    exp_id="run_A", y0=[1.0], t0=0.0,
    channels={"concentration": hm.ChannelObs(ts=[1.0, 2.0], values=[0.8, 0.6])},
)
```

The dataset inserts `t0` as an unobserved timestamp, so the simulator
receives it as `ts[0]` and losses ignore it. A `t0` after the first
observation raises. Without `t0`, `y0` belongs to the first union timestamp.
The inserted point counts towards the length that a phase `horizon` cuts.
