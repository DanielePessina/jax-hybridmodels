# Data: Experiments, Channels, Datasets

Build [`Experiment`](#experiment) records from sparse observations, then pass them to [`make_dataset`](#make_dataset). The data layer creates masks and groups experiments by union-axis length. Scalar and rank-1 vector covariates are stacked along the bucket's leading axis.

**Typical flow:** build `ChannelObs` → call [`make_experiment`](#make_experiment) → call [`make_dataset`](#make_dataset) → optionally call [`split_dataset`](#split_dataset).

## Quick links

- [`ChannelObs`](#channelobs)
- [`Experiment`](#experiment)
- [`make_experiment`](#make_experiment)
- [`experiment_from_array`](#experiment_from_array)
- [`experiments_from_dataframe`](#experiments_from_dataframe)
- [`BucketPayload`](#bucketpayload)
- [`Dataset`](#dataset)
- [`make_dataset`](#make_dataset)
- [`make_bootstrap_dataset`](#make_bootstrap_dataset)
- [`split_dataset`](#split_dataset)
- [`describe_buckets`](#describe_buckets)

---

<a id="channelobs"></a>

### `ChannelObs`

<small>`from jaxhybridmodels.data import ChannelObs` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.ChannelObs`</small>

```python
ChannelObs(*, ts: 'Any', values: 'Any', variance: 'Any' = 1.0) -> 'None'
```

What one measured quantity of one experiment was observed to be, and when.

A *channel* is one observable quantity, for example concentration or
mean crystal size. Each carries its own ``Tc`` observation times, so
channels can be sampled at completely different rates.
``make_dataset`` later merges them onto a shared time axis.

For observed channels, all three arrays share the leading dimension
``Tc``. A probe channel may instead provide nonempty ``ts`` with empty
``values``; its timestamps define the integration grid while contributing
no observations. ``ts`` may be unsorted, since
``_per_experiment_arrays`` sorts when it builds the union axis, but must
not repeat a time within one channel.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `ts` | `Float[Array, "Tc"]` | Observation times for this channel (same time units the user's ``simulate_fn`` consumes). |
| `values` | `Float[Array, "Tc"]` | Observed channel values aligned with ``ts``. |
| `variance` | `Float[Array, "Tc"]` | Per-observation variance used by ``masked_mle`` / ``bal_mle``. A scalar passed to the constructor is broadcast to ``values.shape`` so downstream code can assume rank-1. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L58)</small>

---

<a id="experiment"></a>

### `Experiment`

<small>`from jaxhybridmodels.data import Experiment` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.Experiment`</small>

```python
Experiment(
    covariates: 'dict[str, Array]',
    y0: "Float[Array, ' S']",
    channels: 'dict[str, ChannelObs]',
    exp_id: 'str',
    t0: 'float | None' = None,
) -> None
```

One run of the physical system: its conditions, its starting state, its measurements.

Build these with ``make_experiment``, ``experiment_from_array`` or
``experiments_from_dataframe``. A dataset keeps them, readable through
``Dataset.experiments``, so ``split_dataset`` can re-bucket subsets
after a permutation.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `covariates` | `dict[str, Array]` | Named scalar or rank-1 vector conditions of the run that do not change with time, such as ``temperature_C`` or a feed composition. Every experiment passed to one ``make_dataset`` call must define the same keys and shapes. |
| `y0` | `Float[Array, "S"]` | Full model state at ``t0`` (or the first observation time), of length ``S``. Built by the user's ``y0_fn`` hook when the experiment is constructed. The state may contain components that are never observed, so ``S`` need not equal the channel count. The framework never inspects ``S``. |
| `channels` | `dict[str, ChannelObs]` | Sparse observations, one entry per measured quantity. Must contain every name listed in ``make_dataset(..., output_channel_names=...)``. |
| `exp_id` | `str` | Identifier, unique within a dataset; ``make_dataset`` names an empty one ``experiment_<i>``. A static field, so it is not a JAX array leaf and never reaches a compiled kernel as data. |
| `t0` | `float | None` | Initial state time, added to the union axis as an unobserved row. None means ``y0`` is the state at the first observation. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L140)</small>

---

<a id="make_experiment"></a>

### `make_experiment()`

<small>`from jaxhybridmodels.data import make_experiment` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.make_experiment`</small>

```python
make_experiment(
    channels: 'dict[str, ChannelObs]',
    covariates: 'dict[str, float | Array] | None' = None,
    y0_fn: 'Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None' = None,
    exp_id: 'str' = '',
    t0: 'float | None' = None,
    y0: 'Any' = None,
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
| `channels` |  | Sparse observations keyed by channel name. |
| `covariates` |  | Scalar or rank-1 vector run conditions, constant in time. Values are converted to JAX arrays; a given key must have one consistent shape across a dataset. Omit when the model takes no run conditions. |
| `y0_fn` |  | Hook ``(covariates, channels) -> [S]`` building the full initial state, where ``S`` is the state dimension the user's ``simulate_fn`` integrates. |
| `exp_id` |  | Optional human-readable id copied to ``Experiment.exp_id``. |
| `t0` |  | Initial state time, at or before the first observation. Added as an unobserved integration point. None preserves the first-observation convention. |
| `y0` |  | Direct initial state, as an alternative to y0_fn. Supply exactly one. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L281)</small>

---

<a id="experiment_from_array"></a>

### `experiment_from_array()`

<small>`from jaxhybridmodels.data import experiment_from_array` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.experiment_from_array`</small>

```python
experiment_from_array(
    ts: 'Any',
    values: 'Any',
    channel_names: 'Sequence[str]',
    exp_id: 'str',
    variance: 'Any' = 1.0,
    covariates: 'dict[str, float | Array] | None' = None,
    t0: 'float | None' = None,
    y0: 'Any' = None,
    y0_fn: 'Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None' = None,
) -> Experiment
```

Build one ``Experiment`` from a shared time grid and a ``[T, D]`` table of values.

For data that arrives as a matrix, one row per sample time and one
column per channel. A ``NaN`` cell means "not measured at this time",
so each channel keeps only its own finite rows and channels may still be
sampled at different times.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `ts` |  | Sample times ``[T]``, no repeats. |
| `values` |  | Observations ``[T, D]``, columns in ``channel_names`` order. A rank-1 ``[T]`` array is accepted when there is one channel. |
| `channel_names` |  | The ``D`` channel names. |
| `exp_id` |  | Identifier, required so results stay traceable to the source run. |
| `variance` |  | A scalar for every cell, a length-``D`` sequence with one value per channel, or a ``[T, D]`` array. Read only at observed cells. |
| `covariates, t0, y0, y0_fn` |  | As for [`make_experiment`](/api/data#make_experiment). Supply exactly one of ``y0`` and ``y0_fn``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L341)</small>

---

<a id="experiments_from_dataframe"></a>

### `experiments_from_dataframe()`

<small>`from jaxhybridmodels.data import experiments_from_dataframe` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.experiments_from_dataframe`</small>

```python
experiments_from_dataframe(
    df: 'Any',
    time: 'str',
    experiment: 'str',
    channels: 'Sequence[str] | Mapping[str, str]',
    covariates: 'Sequence[str] | Mapping[str, str]' = (),
    variance: 'float | Mapping[str, str | float] | None' = None,
    t0: 'float | str | None' = None,
    y0: 'Any' = None,
    y0_fn: 'Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None' = None,
) -> list[Experiment]
```

Build one ``Experiment`` per run from a wide pandas table.

The table has one row per (experiment, sample time) and one column per
measured channel, which is how lab spreadsheets usually look. ``NaN``
marks a channel that was not measured at that row, so channels may be
sampled at different times. A long table (one row per measurement, with
``channel`` and ``value`` columns) becomes wide with
``df.pivot_table(index=["run", "time"], columns="channel", values="value").reset_index()``.

Experiments come back in order of first appearance, each built by
[`experiment_from_array`](/api/data#experiment_from_array).

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `df` |  | A ``pandas.DataFrame``. pandas itself is never imported here. |
| `time` |  | Column of sample times. |
| `experiment` |  | Column identifying the run. Its values, as strings, become ``exp_id``. A missing value raises ``ValueError``. |
| `channels` |  | Channel columns. A list names each channel after its column; a mapping ``{channel: column}`` renames. |
| `covariates` |  | Covariate columns, in the same two forms. Each must be constant within a run. |
| `variance` |  | None for unit variance, one number for every cell, or a mapping ``{channel: column or number}``. Channels missing from the mapping get unit variance. |
| `t0` |  | None, one initial time for every run, or a column holding each run's initial time (constant within the run). |
| `y0, y0_fn` |  | Exactly one. ``y0`` is either one state shared by every run or a mapping ``{exp_id: state}``. ``y0_fn`` is the ``(covariates, channels) -> [S]`` hook of [`make_experiment`](/api/data#make_experiment). |

**Examples**

::

    experiments = hm.experiments_from_dataframe(
        df.replace(-1.0, np.nan),
        time="Time",
        experiment="Exp_ID",
        channels={"conc": "Concentration", "d43": "PS"},
        covariates={"temperature": "Temperature", "loading": "Loading"},
        variance={"conc": "Concentration_var", "d43": "PS_var"},
        y0_fn=lambda cov, ch: jnp.array([ch["conc"].values[0], 0.0, 0.0]),
    )

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L427)</small>

---

<a id="bucketpayload"></a>

### `BucketPayload`

<small>`from jaxhybridmodels.data import BucketPayload` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.BucketPayload`</small>

```python
BucketPayload(
    ts: ForwardRef("Float[Array, 'N T']"),
    y_observed: ForwardRef("Float[Array, 'N T D']"),
    yvar: ForwardRef("Float[Array, 'N T D']"),
    mask: ForwardRef("Bool[Array, 'N T D']"),
    covariates: ForwardRef('dict[str, Array]'),
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
covariates : dict[str, Array]
    Per-key covariate stacked across the bucket. Same keys as on
    ``Experiment.covariates``, with an ``N`` axis added; scalar values
    have shape ``[N]`` and vectors have shape ``[N, K]``.
y0 : Float[Array, "N S"]
    Per-experiment full initial state, stacked.
n_obs : Int[Array, ""]
    Total observed-cell count for the bucket (``mask.sum()``).

    No shipped loss reads it, and none should: it counts across *all*
    channels, while every loss reduces over a selected subset and needs
    its own denominator. Kept because examples and smoke scripts assert
    dataset shape with it (R-D4).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L180)</small>

---

<a id="dataset"></a>

### `Dataset`

<small>`from jaxhybridmodels.data import Dataset` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.Dataset`</small>

```python
Dataset(
    bucket_payloads: 'tuple[BucketPayload, ...]',
    output_channel_names: 'tuple[str, ...]',
    covariate_names: 'tuple[str, ...]',
    _experiments: 'tuple[Experiment, ...]' = (),
) -> None
```

All buckets of a dataset, as pure data.

``bucket_payloads`` is the dispatch list, one compiled kernel per bucket
shape. The ``Dataset`` carries no model-shaped callables: it never sees
full simulator states, and ``state_to_output`` belongs to the model, so
prediction and training take it as a separate argument.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `bucket_payloads` | `tuple[BucketPayload, ...]` | One ``BucketPayload`` per distinct union-axis length, ordered ascending by ``T``. |
| `output_channel_names` | `tuple[str, ...]` | Channel order along the trailing ``D`` axis of every payload. ``make_dataset`` scatters values in this same order. |
| `covariate_names` | `tuple[str, ...]` | Covariate keys, sorted. Matches each ``Experiment.covariates`` key set. Sorting makes dict iteration deterministic. |
| `_experiments` | `tuple[Experiment, ...]` | Source experiments in input order, read through the `experiments` property. Empty when a ``Dataset`` is built by hand from raw payloads, and ``split_dataset`` then raises. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L232)</small>

---

<a id="make_dataset"></a>

### `make_dataset()`

<small>`from jaxhybridmodels.data import make_dataset` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.make_dataset`</small>

```python
make_dataset(
    experiments: 'Sequence[Experiment]',
    output_channel_names: 'tuple[str, ...] | list[str]',
) -> Dataset
```

Bucket and stack ``experiments`` into a JAX-traceable ``Dataset``.

Three steps run in order.

1. Validation. Experiment IDs must be unique; an empty one is named
   ``experiment_<i>`` after its position. All experiments must agree on
   the set of covariate keys, and each must define every requested
   output channel. A mismatch raises at once, naming the offending
   ``exp_id``.
2. Per-experiment scattering. ``_per_experiment_arrays`` builds each
   experiment's union timestamp axis and its ``[T, D]`` observation,
   variance, and mask tensors.
3. Bucketing. Experiments are grouped by their union-axis length ``T``,
   and each group is stacked along a new leading ``N`` axis into one
   ``BucketPayload``. Buckets come out in ascending ``T`` order.

The ``Dataset`` is pure data: ``state_to_output``, being a property of
the model, is passed to prediction and training separately.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `experiments` |  | Non-empty sequence of ``Experiment`` objects, usually built with ``make_experiment``. |
| `output_channel_names` |  | Channel order for the trailing ``D`` axis. Coerced to a tuple before being stored statically on the ``Dataset``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Dataset` |  | ``bucket_payloads`` ordered ascending by ``T``, with the (named) experiments kept on ``Dataset.experiments``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L817)</small>

---

<a id="make_bootstrap_dataset"></a>

### `make_bootstrap_dataset()`

<small>`from jaxhybridmodels.data import make_bootstrap_dataset` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.make_bootstrap_dataset`</small>

```python
make_bootstrap_dataset(
    dataset: 'Dataset',
    key: 'Array',
    n_experiments: 'int | None' = None,
) -> Dataset
```

Bootstrap resample the dataset's experiments, re-bucketing the result.

Draws ``n_experiments`` experiments **with replacement** from the
source (default: as many as the source holds), then re-buckets via
[`make_dataset`](/api/data#make_dataset). Irregular per-channel timestamps are handled
automatically: re-bucketing regroups by union-axis length, and a
duplicated experiment simply contributes more ``N`` rows to its bucket.
Repeats are renamed ``<exp_id>#2``, ``<exp_id>#3`` and so on, since IDs
within a dataset must be unique.

This is the data half of a bagging ensemble: each call yields one
resampled dataset, and training on several of them produces an
ensemble whose members saw different resamples.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `dataset` |  | Source dataset. Must carry its experiments (built by ``make_dataset``), or this raises. |
| `key` |  | Required ``jr.PRNGKey`` for the resample, never defaulted. |
| `n_experiments` |  | Number of experiments to draw. Defaults to the source size. Must be at least 1. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Dataset` |  | A new dataset of ``n_experiments`` experiments (some duplicated), re-bucketed from scratch. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L966)</small>

---

<a id="split_dataset"></a>

### `split_dataset()`

<small>`from jaxhybridmodels.data import split_dataset` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.split_dataset`</small>

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
payloads and no experiments, so it cannot be split again.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `dataset` |  | Source dataset. Must carry its experiments, or this raises. |
| `train, val, test` |  | Fractions in ``[0, 1]`` summing to ``1.0`` (within ``np.isclose``). |
| `key` |  | Required ``jr.PRNGKey`` for the permutation, never defaulted, so reproducibility does not rest on a hidden global. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Dataset, Dataset, Dataset]` |  | ``(train_dataset, val_dataset, test_dataset)``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L1037)</small>

---

<a id="describe_buckets"></a>

### `describe_buckets()`

<small>`from jaxhybridmodels.data import describe_buckets` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.describe_buckets`</small>

```python
describe_buckets(dataset: 'Dataset') -> 'str'
```

One line per bucket: experiments, length, and how full the mask is.

A quick human-readable census of the bucketed-irregular structure, for
debugging and example output. Each line reports the bucket's ``N``
(experiments), ``T`` (union timestamp axis), ``D`` (channels), and the
fraction of ``[N, T, D]`` cells the mask marks as real observations.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/data.py#L799)</small>
