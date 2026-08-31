# Data and buckets

The data interface accepts experiments with sparse, irregular observations.
Each measured channel has its own timestamps. Regular data is the special case
where all channels share the same timestamps.

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

`y0_fn` returns the complete initial state. The state may contain components
that are not observed.

## Union timestamps and masks

`make_dataset` computes the union of all channel timestamps for each
experiment. It scatters each channel onto that axis and creates a boolean mask.
The mask is `True` only where a value was measured.

```python
dataset = hm.make_dataset(
    experiments,
    output_channel_names=("concentration", "size"),
)
```

No interpolation or padding is performed. A channel with no values may still
provide timestamps for a probe experiment; its mask is all `False`.

## Buckets

A bucket contains experiments with the same union-axis length. The timestamp
values and masks may still differ within a bucket.

| Quantity | Shape |
| --- | --- |
| `bp.ts` | `[N, T]` |
| `bp.y_observed` | `[N, T, D]` |
| `bp.yvar` | `[N, T, D]` |
| `bp.mask` | `[N, T, D]` |
| scalar covariate | `[N]` |
| vector covariate | `[N, K]` |
| `bp.y0` | `[N, S]` |

The training and prediction functions dispatch over buckets in Python. Each
bucket shape gets its own compiled kernel.

## Validation

The data boundary checks the conditions that would otherwise fail inside a
compiled solve:

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
    key=jr.PRNGKey(0),
)
```

Use `make_bootstrap_dataset` for sampling experiments with replacement.

## Next steps

- [Model interface](/guide/model-interface)
- [Training](/guide/training)
- [Losses API](/api/losses)
