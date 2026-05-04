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

Per-channel sparse observation triple ``(ts, values, variance)``.

One ``ChannelObs`` describes a single observable channel for a single
experiment. Channels are sparse: each channel carries its own ``Tc``
timestamps independent of other channels, and the framework computes
the union timestamp axis and resulting mask at ``make_dataset`` time.

**Shape contract**

All three arrays share the same leading dimension ``Tc``. ``ts`` does not
need to be sorted — ``_per_experiment_arrays`` re-sorts when building the
union axis — but it should not contain duplicates within a single channel.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `ts` | `Float[Array, "Tc"]` | Observation times for this channel (same time units the user's ``simulate_fn`` consumes). |
| `values` | `Float[Array, "Tc"]` | Observed channel values aligned with ``ts``. |
| `variance` | `Float[Array, "Tc"]` | Per-observation variance used by ``masked_mle`` / ``bal_mle``. A scalar passed to the constructor is broadcast to ``values.shape`` so downstream code can assume rank-1. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L51)</small>

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

One experiment: covariates, full initial state, and per-channel observations.

Built by ``make_experiment``; stored on ``Dataset._experiments`` so
``split_dataset`` can re-bucket subsets after a permutation.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `covariates` | `dict[str, Array]` | Named scalar covariates, **constant in time**. Keys must agree across all experiments handed to a single ``make_dataset`` call. |
| `y0` | `Float[Array, "S"]` | Full model state at ``t=0``, constructed via the user's ``y0_fn`` hook at experiment-build time. Shape is whatever the user's ``simulate_fn`` consumes; the framework never inspects ``S``. |
| `channels` | `dict[str, ChannelObs]` | Per-channel sparse observations. Must contain every name listed in ``make_dataset(..., output_channel_names=...)``. |
| `exp_id` | `str` | Identifier carried through for diagnostics (static field; not a leaf). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L105)</small>

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

The ``y0_fn`` hook receives the (already-jnp) covariates dict and the
channels dict and returns the full state at ``t=0`` as
``Float[Array, "S"]``. For systems where the observed channels *are*
the state, a typical hook is
``lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])``.
For systems with hidden latent components, the hook constructs them
from covariates and/or initial channel values — for example, a
rate-of-change latent that is initialised to zero, or a temperature
derived from a covariate.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `covariates` |  | Scalar covariates; values are converted to 0-d ``jnp`` arrays. |
| `channels` |  | Sparse observations keyed by channel name. |
| `y0_fn` |  | Hook ``(covariates, channels) -> [S]`` building the full initial state. |
| `exp_id` |  | Optional human-readable id propagated to ``Experiment.exp_id``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L218)</small>

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

Bucket-shaped, JAX-traceable payload produced by ``make_dataset``.

A bucket holds ``N`` experiments that share the same union-timestamp
length ``T``. Within a bucket, individual experiments may still
have **different ts values and different masks** — the bucketing rule
only fixes ``len(union_ts)``, not the values themselves.

This is a ``NamedTuple`` rather than an ``eqx.Module`` because every
field is a stacked JAX array and there is no module-level method
surface; the whole struct is consumed positionally by jitted training
and prediction kernels, and a NamedTuple is the lightest container
JAX recognises as a registered pytree out of the box.

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
    ``True`` iff the corresponding ``y_observed`` cell came from a real
    ``ChannelObs`` entry; ``False`` for union-axis padding.
covariates : dict[str, Float[Array, "N"]]
    Per-key covariate stacked across the bucket. Same keys as on
    ``Experiment.covariates``, hoisted by an ``N`` axis.
y0 : Float[Array, "N S"]
    Per-experiment full initial state, stacked.
n_obs : Int[Array, ""]
    Total observed-cell count for the bucket (``mask.sum()``). Used by
    weighted reductions; not used by ``masked_*`` (which compute their
    own denominators).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L133)</small>

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

Container of bucketed experiments plus the static ``state_to_output`` hook.

A ``Dataset`` is the artifact training and prediction loops iterate
over. The ``bucket_payloads`` tuple is the dispatch list (one
compiled trace per bucket shape); ``state_to_output`` is held here
so the loss pipeline can apply it without the user threading it
through every call site.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `bucket_payloads` | `tuple[BucketPayload, ...]` | One ``BucketPayload`` per distinct ``len(union_ts)`` value, ordered ascending by ``T``. |
| `state_to_output` | `Callable[[Array], Array]` | Pure mapping ``[T, S] -> [T, D]`` projecting full simulator state onto observed channels. Static — never serialised by the framework; the user re-imports it on load. |
| `output_channel_names` | `tuple[str, ...]` | Channel order along the trailing ``D`` axis of every payload. The same order is honoured by ``make_dataset`` when scattering values. |
| `covariate_names` | `tuple[str, ...]` | Sorted covariate keys (matches each ``Experiment.covariates`` key set; sorted for deterministic dict iteration). |
| `_experiments` | `tuple[Experiment, ...]` | Source experiments, retained so ``split_dataset`` can re-bucket per-split subsets. Empty when a ``Dataset`` is constructed manually from raw payloads (in which case ``split_dataset`` will raise). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L181)</small>

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

Three things happen here, in order:

1. **Validation.** All experiments must agree on the set of covariate
   keys, and each must define every requested output channel. Mismatches
   raise immediately with the offending ``exp_id``.
2. **Per-experiment scattering.** For each experiment, ``_per_experiment_arrays``
   builds its union timestamp axis and the ``[T, D]`` observation/mask
   tensors.
3. **Bucketing.** Experiments are grouped by
   ``T = len(union_ts)``, and each group is stacked along a new
   leading ``N`` axis to produce one ``BucketPayload``. Buckets are
   emitted in ascending ``T`` order.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `experiments` |  | Non-empty sequence of ``Experiment`` objects (typically built via ``make_experiment``). |
| `state_to_output` |  | Pure mapping ``[T, S] -> [T, D]`` projecting full state onto the observed channels. Stored static on the resulting ``Dataset``. |
| `output_channel_names` |  | Channel order for the trailing ``D`` axis. Coerced to a tuple before being captured statically on the ``Dataset``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Dataset` |  | ``bucket_payloads`` ordered ascending by ``T``; ``_experiments`` retained so ``split_dataset`` can re-bucket subsets. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L329)</small>

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

Splits are computed at the ``Experiment`` level — *not* by carving up
bucket payloads — so each split is re-bucketed from scratch. (Carving
payloads would couple the split sizes to the original bucket
boundaries; rebucketing keeps each split's bucket structure
appropriate for its own contents.) Counts use ``floor(train*n)`` and
``floor(val*n)``; the test split takes the remainder so the three
sizes sum to ``n`` even with rounding. An empty split is returned as
a ``Dataset`` with no payloads (and no ``_experiments``, so it cannot
be split again).

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `dataset` |  | Source dataset; must carry ``_experiments`` (raises otherwise). |
| `train, val, test` |  | Fractions in ``[0, 1]`` summing to ``1.0`` (within ``np.isclose``). |
| `key` |  | Required ``jr.PRNGKey`` for the permutation. There is no silent default — the framework refuses to permute under an implicit key so reproducibility never relies on a hidden global. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Dataset, Dataset, Dataset]` |  | ``(train_dataset, val_dataset, test_dataset)``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/data.py#L427)</small>
