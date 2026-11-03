# Losses: Masked & Balanced Objectives

Loss functions consume a model's predicted output `[N, T, D]` and the bucket payload, return a scalar, and respect the bucket mask. Two families: `masked_*` weights every observation equally; `bal_*` normalises per-experiment so duplication of one experiment can't dominate the gradient.

Pass them by name via training-config `loss="mse"` (resolved through [`LOSS_REGISTRY`](#loss_registry)) or as a callable for custom losses.

## Quick links

- [`masked_mse`](#masked_mse)
- [`masked_mle`](#masked_mle)
- [`bal_mse`](#bal_mse)
- [`bal_mle`](#bal_mle)
- [`LOSS_REGISTRY`](#loss_registry)
- [`resolve_loss_fn`](#resolve_loss_fn)
- [`loss_pooling`](#loss_pooling)
- [`Pooling`](#pooling)
- [`POOLING_KINDS`](#pooling_kinds)

---

<a id="masked_mse"></a>

### `masked_mse()`

<small>`from jaxhybridmodels.losses import masked_mse`</small>

```python
masked_mse(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    *,
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Mean squared error reduced over a single global denominator.

Computes ``sum(mask * weights * (pred - observed)**2) / max(mask.sum(), 1)``
over the selected channels. A long-trajectory experiment adds terms to
the numerator and the denominator in proportion, and gets no explicit
per-experiment weighting. Use ``bal_mse`` when you want that weighting.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `pred_obs` | `Float[Array, "N T D"]` | Predicted output channels, one ``[T, D]`` block per experiment in the bucket, from ``state_to_output(simulate_fn(...))``. |
| `bp` | `BucketPayload` | Bucket data. Reads ``mask`` and ``observed``. |
| `channel_idx` |  | Trailing-axis indices to keep. ``None`` keeps all ``D`` channels. |
| `channel_weights` |  | Per-channel multipliers; length must match ``channel_idx`` (or ``D`` when ``channel_idx`` is ``None``). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L144)</small>

---

<a id="masked_mle"></a>

### `masked_mle()`

<small>`from jaxhybridmodels.losses import masked_mle`</small>

```python
masked_mle(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    *,
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Total Gaussian negative log-likelihood across the bucket.

Sums the pointwise Gaussian negative log-likelihood (variance from
``bp.variance``) over the time axis, giving one NLL per experiment and
channel, shape ``[N, D]``. Multiplies by the per-channel ``weights`` and
sums those.

Nothing is averaged. The result is a sum of log-likelihoods and grows
linearly with the number of observations. Use ``bal_mle`` for the
per-experiment averaged version.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L179)</small>

---

<a id="bal_mse"></a>

### `bal_mse()`

<small>`from jaxhybridmodels.losses import bal_mse`</small>

```python
bal_mse(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    *,
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

MSE averaged over time within each experiment, then averaged over experiments.

Reduction order is ``[N, T, D] -> [N, D]`` (per-channel time average),
then ``[N]`` (channel-weighted sum), then a scalar (mean across ``N``).
Every experiment therefore counts the same, however many observations it
contributed, so a long trajectory cannot dominate the gradient.

Per-experiment, per-channel denominators are clamped to ``1`` with
``maximum(count, 1)``, so an experiment with zero observations on a
channel does not divide by zero. Experiments with no observations in any
selected channel (for example, a trajectory-penalty probe) are excluded
from the outer mean rather than diluting measured experiments.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L205)</small>

---

<a id="bal_mle"></a>

### `bal_mle()`

<small>`from jaxhybridmodels.losses import bal_mle`</small>

```python
bal_mle(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    *,
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Per-experiment-balanced Gaussian NLL.

Averages over time within each observed experiment, then over those
experiments. Same reduction as ``bal_mse``, with
``_gaussian_nll_terms`` (which reads ``bp.variance``) in place of the
pointwise squared error. Experiments with no observations in any selected
channel are excluded from the outer mean.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L238)</small>

---

<a id="loss_registry"></a>

### `LOSS_REGISTRY`

<small>`from jaxhybridmodels.losses import LOSS_REGISTRY`</small>

```python
LOSS_REGISTRY = {
  'bal_mle': function
  'bal_mse': function
  'mle': function
  'mse': function
}
```

dict() -> new empty dictionary

dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

---

<a id="resolve_loss_fn"></a>

### `resolve_loss_fn()`

<small>`from jaxhybridmodels.losses import resolve_loss_fn`</small>

```python
resolve_loss_fn(
    loss: 'Callable[..., Array] | str',
    channel_idx: 'tuple[int, ...] | None',
    channel_weights: 'tuple[float, ...] | None',
) -> Callable[[Array, BucketPayload], Array]
```

Turn an ``Objective``'s ``loss`` field into a ``(pred_obs, bp) -> scalar``.

``loss`` is either a ``LOSS_REGISTRY`` key, matched case- and
whitespace-insensitively, or a callable already in the right shape.

With neither ``channel_idx`` nor ``channel_weights`` the resolved
function is returned as-is rather than wrapped. That identity matters: a
user loss written to the bare ``(pred_obs, bp)`` signature would raise
``TypeError`` on the unexpected keywords if it were wrapped
unconditionally.

How channel selection composes depends on the loss:

- A registry name, or a callable whose signature accepts
  ``channel_idx``/``channel_weights``, is called with them as keyword
  arguments (the built-ins reduce with the weights inside their own
  per-channel sum).
- A plain ``(pred_obs, bp)`` callable is *projected* instead: the
  selected channels are sliced out of ``pred_obs`` and ``bp`` before the
  call, so any ``(pred_obs, bp)`` loss composes with ``channel_idx``.
  ``channel_weights`` cannot be projected this way, because per-channel
  weighting has to happen inside a loss's own reduction. A plain
  callable combined with ``channel_weights`` raises rather than
  silently ignoring the weights.

Lives here rather than in the training modules because it is registry
lookup and channel binding, not training logic, and both loops need it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L371)</small>

---

<a id="loss_pooling"></a>

### `loss_pooling()`

<small>`from jaxhybridmodels.losses import loss_pooling`</small>

```python
loss_pooling(loss: 'Callable[..., Array] | str') -> 'str'
```

How ``loss`` combines across buckets: one of :data:`POOLING_KINDS`.

Each bucket is scored on its own, so the trainers need to know what a
bucket's value is an average over to put the buckets back together:

- ``"observations"``: a mean over measured cells (``mse``). Buckets are
  weighted by their measured-cell count, which gives the mean over every
  measurement in the dataset.
- ``"experiments"``: a mean over experiments (``bal_mse``, ``bal_mle``).
  Buckets are weighted by how many measured experiments they hold, so
  every experiment counts the same across the dataset.
- ``"sum"``: already a sum (``mle``). Bucket values are added.

A ``functools.partial`` of a built-in pools as the built-in. Any other
custom loss declares its kind with a ``pooling`` attribute, for example
``my_loss.pooling = "sum"``, and defaults to ``"observations"``: a
wrapper around ``bal_mse`` must set ``"experiments"`` itself.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L286)</small>

---

<a id="pooling"></a>

### `Pooling`

<small>`from jaxhybridmodels.losses import Pooling`</small>

```python
Pooling(
    kind: ForwardRef('str'),
    channel_idx: ForwardRef('tuple[int, ...] | None') = None,
)
```

How to weight each bucket's loss so that the buckets add up to one loss.

Build it with [`Objective.pooling`](/api/training#objectivepooling). ``channel_idx``
is the objective's channel selection, so only the channels the loss
reads are counted.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L319)</small>

<a id="poolingshares"></a>

#### `Pooling.shares()`

```python
shares(
    self,
    buckets: 'Sequence[BucketPayload]',
    horizon: 'ArrayLike' = 1.0,
) -> tuple[Array, ...]
```

Each bucket's ``(data, penalty)`` multipliers for the dataset loss.

The data multiplier makes ``sum(share * bucket_loss)`` the loss over
the whole dataset: the shares of a mean-type loss add up to 1, and
with ``"sum"`` each is 1. A dataset with nothing measured inside
``horizon`` gets equal data shares, so the loss stays defined.

The penalty multiplier weights each bucket's trajectory penalty, and a
segmented phase's gap term, by its share of the experiments. Both are
means over a bucket's experiments, so this makes them means over the
dataset's, whatever the loss, and a bucket of unmeasured probe
experiments still counts. Computed on the host, once per dataset and
horizon, not per step.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L339)</small>

<a id="poolingweight"></a>

#### `Pooling.weight()`

```python
weight(self, bp: 'BucketPayload') -> 'Array'
```

What one bucket's loss is an average over, after any horizon mask.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L330)</small>

---

<a id="pooling_kinds"></a>

### `POOLING_KINDS`

<small>`from jaxhybridmodels.losses import POOLING_KINDS`</small>

```python
POOLING_KINDS = ('observations', 'experiments', 'sum')
```

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.
