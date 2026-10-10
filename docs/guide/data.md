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

`y0_fn(covariates, channels)` returns the full initial state, which may
include unobserved components. Pass `y0=` instead when the state is known
directly; give one or the other.

## Union timestamps and masks

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
- each experiment must have at least one timestamp.

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
