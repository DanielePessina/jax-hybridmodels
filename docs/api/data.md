# Data: Experiments, Channels, Datasets

The data layer turns irregular, sparse experiment records into a JAX-traceable [`Dataset`](#dataset). Every observation channel can have its own timestamps; missing values are represented by a boolean mask, never by NaN sentinels. Experiments with the same number of union timestamps are stacked into one [`BucketPayload`](#bucketpayload), so each bucket compiles once and reuses its trace.

**Typical flow:** build `ChannelObs` per channel → wrap in `Experiment` via [`make_experiment`](#make_experiment) → batch via [`make_dataset`](#make_dataset) → optionally partition with [`split_dataset`](#split_dataset).

## Quick links

- [`ChannelObs`](#channelobs)
- [`Experiment`](#experiment)
- [`make_experiment`](#make_experiment)
- [`BucketPayload`](#bucketpayload)
- [`Dataset`](#dataset)
- [`make_dataset`](#make_dataset)
- [`split_dataset`](#split_dataset)

---

<a id="channelobs"></a>

### `ChannelObs`

<small>`from hybridmodels.data import ChannelObs` &nbsp;·&nbsp; also re-exported as `hybridmodels.ChannelObs`</small>

```python
ChannelObs(ts: 'Any', values: 'Any', variance: 'Any' = 1.0) -> 'None'
```

What one measured quantity of one experiment was observed to be, and when.

A *channel* is one observable quantity, for example concentration or
mean crystal size. Each carries its own ``Tc`` observation times, so
channels can be sampled at completely different rates.
``make_dataset`` later merges them onto a shared time axis.

All three arrays share the leading dimension ``Tc``. ``ts`` may be
unsorted, since ``_per_experiment_arrays`` sorts when it builds the
union axis, but must not repeat a time within one channel.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `ts` | `Float[Array, "Tc"]` | Observation times for this channel (same time units the user's ``simulate_fn`` consumes). |
| `values` | `Float[Array, "Tc"]` | Observed channel values aligned with ``ts``. |
| `variance` | `Float[Array, "Tc"]` | Per-observation variance used by ``masked_mle`` / ``bal_mle``. A scalar passed to the constructor is broadcast to ``values.shape`` so downstream code can assume rank-1. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L57)</small>

---

<a id="experiment"></a>

### `Experiment`

<small>`from hybridmodels.data import Experiment` &nbsp;·&nbsp; also re-exported as `hybridmodels.Experiment`</small>

```python
Experiment(
    covariates: 'dict[str, Array]',
    y0: "Float[Array, ' S']",
    channels: 'dict[str, ChannelObs]',
    exp_id: 'str',
) -> None
```

One run of the physical system: its conditions, its starting state, its measurements.

Build these with ``make_experiment``. They are kept on
``Dataset._experiments`` so ``split_dataset`` can re-bucket subsets
after a permutation.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `covariates` | `dict[str, Array]` | Named scalar conditions of the run that do not change with time, such as ``temperature_C`` or ``loading``. Every experiment passed to one ``make_dataset`` call must define the same keys. |
| `y0` | `Float[Array, "S"]` | Full model state at ``t=0``, of length ``S``. Built by the user's ``y0_fn`` hook when the experiment is constructed. The state may contain components that are never observed, so ``S`` need not equal the channel count. The framework never inspects ``S``. |
| `channels` | `dict[str, ChannelObs]` | Sparse observations, one entry per measured quantity. Must contain every name listed in ``make_dataset(..., output_channel_names=...)``. |
| `exp_id` | `str` | Identifier carried through for diagnostics. A static field, so it is not a JAX array leaf and never reaches a compiled kernel as data. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L108)</small>

---

<a id="make_experiment"></a>

### `make_experiment()`

<small>`from hybridmodels.data import make_experiment` &nbsp;·&nbsp; also re-exported as `hybridmodels.make_experiment`</small>

```python
make_experiment(
    covariates: 'dict[str, float]',
    channels: 'dict[str, ChannelObs]',
    y0_fn: 'Callable[[dict[str, Array], dict[str, ChannelObs]], Array]',
    exp_id: 'str' = '',
) -> Experiment
```

Build one ``Experiment`` from raw covariates, channels, and a state-init hook.

``y0_fn`` builds the model's full starting state from the covariates
(already JAX arrays) and the channels, returning ``Float[Array, "S"]``.
Where the observed channels are the whole state, a typical hook is
``lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])``.
Unobserved state components are constructed there too, a population
moment initialised to zero being the common case.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `covariates` |  | Scalar run conditions, constant in time. Values are converted to 0-d ``jnp`` arrays. |
| `channels` |  | Sparse observations keyed by channel name. |
| `y0_fn` |  | Hook ``(covariates, channels) -> [S]`` building the full initial state, where ``S`` is the state dimension the user's ``simulate_fn`` integrates. |
| `exp_id` |  | Optional human-readable id copied to ``Experiment.exp_id``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L228)</small>

---

<a id="bucketpayload"></a>

### `BucketPayload`

<small>`from hybridmodels.data import BucketPayload` &nbsp;·&nbsp; also re-exported as `hybridmodels.BucketPayload`</small>

```python
BucketPayload(
    ts: ForwardRef("Float[Array, 'N T']"),
    y_observed: ForwardRef("Float[Array, 'N T D']"),
    yvar: ForwardRef("Float[Array, 'N T D']"),
    mask: ForwardRef("Bool[Array, 'N T D']"),
    covariates: ForwardRef("dict[str, Float[Array, ' N']]"),
    y0: ForwardRef("Float[Array, 'N S']"),
    n_obs: ForwardRef("Int[Array, '']"),
)
```

One bucket of experiments, stacked into rectangular arrays for JAX.

A bucket holds ``N`` experiments that share the same union-timestamp
length ``T``. The bucketing rule fixes only that length. Two
experiments in the same bucket can still have different observation
times and different masks.

A ``NamedTuple`` rather than an ``eqx.Module`` because every field is a
stacked JAX array with no methods to hang on it, and a ``NamedTuple`` is
the lightest pytree container JAX already recognises.

**Fields**

ts : Float[Array, "N T"]
    Per-experiment union-timestamp axis, sorted ascending row-wise.
y_observed : Float[Array, "N T D"]
    Channel observations scattered onto ``ts``. Cells where the channel
    was not observed at that timestamp hold ``0.0``; consumers must read
    ``mask`` to know which entries are real.
yvar : Float[Array, "N T D"]
    Per-observation variance (used by MLE losses). Defaults to ``1.0``
    at unobserved cells so masked positions never divide by zero.
mask : Bool[Array, "N T D"]
    ``True`` where the corresponding ``y_observed`` cell came from a
    real ``ChannelObs`` entry, ``False`` where the union axis carries a
    time at which that channel was not measured. Every loss reads this
    to know which cells count.
covariates : dict[str, Float[Array, "N"]]
    Per-key covariate stacked across the bucket. Same keys as on
    ``Experiment.covariates``, with an ``N`` axis added.
y0 : Float[Array, "N S"]
    Per-experiment full initial state, stacked.
n_obs : Int[Array, ""]
    Total observed-cell count for the bucket (``mask.sum()``).

    No shipped loss reads it, and none should: it counts across *all*
    channels, while every loss reduces over a selected subset and needs
    its own denominator. Kept because examples and smoke scripts assert
    dataset shape with it (R-D4).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L140)</small>

---

<a id="dataset"></a>

### `Dataset`

<small>`from hybridmodels.data import Dataset` &nbsp;·&nbsp; also re-exported as `hybridmodels.Dataset`</small>

```python
Dataset(
    bucket_payloads: 'tuple[BucketPayload, ...]',
    state_to_output: 'Callable[..., Array]',
    output_channel_names: 'tuple[str, ...]',
    covariate_names: 'tuple[str, ...]',
    _experiments: 'tuple[Experiment, ...]' = (),
) -> None
```

All buckets of a dataset, plus the hook that maps model state to observed channels.

``bucket_payloads`` is the dispatch list, one compiled kernel per bucket
shape. ``state_to_output`` rides along so the loss pipeline can apply it
without the user passing it to every call.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `bucket_payloads` | `tuple[BucketPayload, ...]` | One ``BucketPayload`` per distinct union-axis length, ordered ascending by ``T``. |
| `state_to_output` | `Callable[[Array], Array]` | Pure mapping ``[T, S] -> [T, D]``. It picks out (or derives) the observed channels from the full simulator state, since the state usually carries components no instrument measures. A static field, so it is code rather than data. The framework never serialises it; the user re-imports it on load. |
| `output_channel_names` | `tuple[str, ...]` | Channel order along the trailing ``D`` axis of every payload. ``make_dataset`` scatters values in this same order. |
| `covariate_names` | `tuple[str, ...]` | Covariate keys, sorted. Matches each ``Experiment.covariates`` key set. Sorting makes dict iteration deterministic. |
| `_experiments` | `tuple[Experiment, ...]` | Source experiments, kept so ``split_dataset`` can re-bucket each split. Empty when a ``Dataset`` is built by hand from raw payloads, and ``split_dataset`` then raises. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L191)</small>

---

<a id="make_dataset"></a>

### `make_dataset()`

<small>`from hybridmodels.data import make_dataset` &nbsp;·&nbsp; also re-exported as `hybridmodels.make_dataset`</small>

```python
make_dataset(
    experiments: 'Sequence[Experiment]',
    state_to_output: 'Callable[..., Array]',
    output_channel_names: 'tuple[str, ...] | list[str]',
) -> Dataset
```

Bucket and stack ``experiments`` into a JAX-traceable ``Dataset``.

Three steps run in order.

1. Validation. All experiments must agree on the set of covariate keys,
   and each must define every requested output channel. A mismatch
   raises at once, naming the offending ``exp_id``.
2. Per-experiment scattering. ``_per_experiment_arrays`` builds each
   experiment's union timestamp axis and its ``[T, D]`` observation,
   variance, and mask tensors.
3. Bucketing. Experiments are grouped by their union-axis length ``T``,
   and each group is stacked along a new leading ``N`` axis into one
   ``BucketPayload``. Buckets come out in ascending ``T`` order.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `experiments` |  | Non-empty sequence of ``Experiment`` objects, usually built with ``make_experiment``. |
| `state_to_output` |  | Pure mapping ``[T, S] -> [T, D]`` from full simulator state to the observed channels. Stored as a static field on the ``Dataset``. |
| `output_channel_names` |  | Channel order for the trailing ``D`` axis. Coerced to a tuple before being stored statically on the ``Dataset``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Dataset` |  | ``bucket_payloads`` ordered ascending by ``T``, with ``_experiments`` kept so ``split_dataset`` can re-bucket subsets. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L415)</small>

---

<a id="split_dataset"></a>

### `split_dataset()`

<small>`from hybridmodels.data import split_dataset` &nbsp;·&nbsp; also re-exported as `hybridmodels.split_dataset`</small>

```python
split_dataset(
    dataset: 'Dataset',
    train: 'float' = 0.8,
    val: 'float' = 0.1,
    test: 'float' = 0.1,
    key: 'Array',
) -> tuple[Dataset, Dataset, Dataset]
```

Permute experiments and re-bucket each split independently.

Splitting happens at the ``Experiment`` level and each split is bucketed
from scratch, so its bucket structure suits its own contents rather than
the original bucket boundaries.

Counts use ``floor(train*n)`` and ``floor(val*n)``, with test taking the
remainder so the sizes sum to ``n``. An empty split comes back with no
payloads and no ``_experiments``, so it cannot be split again.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `dataset` |  | Source dataset. Must carry ``_experiments``, or this raises. |
| `train, val, test` |  | Fractions in ``[0, 1]`` summing to ``1.0`` (within ``np.isclose``). |
| `key` |  | Required ``jr.PRNGKey`` for the permutation, never defaulted, so reproducibility does not rest on a hidden global. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Dataset, Dataset, Dataset]` |  | ``(train_dataset, val_dataset, test_dataset)``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L544)</small>
