# Losses: Masked & Balanced Objectives

Loss functions consume a model's predicted output `[N, T, D]` and the bucket payload, return a scalar, and respect the bucket mask. Two families: `masked_*` weights every observation equally; `bal_*` normalises per-experiment so duplication of one experiment can't dominate the gradient.

Pass them by name via training-config `loss="mse"` (resolved through [`LOSS_REGISTRY`](#loss_registry)) or as a callable for custom losses.

## Quick links

- [`masked_mse`](#masked_mse)
- [`masked_mle`](#masked_mle)
- [`bal_mse`](#bal_mse)
- [`bal_mle`](#bal_mle)
- [`LOSS_REGISTRY`](#loss_registry)

---

<a id="masked_mse"></a>

### `masked_mse()`

<small>`from hybridmodels.losses import masked_mse` &nbsp;·&nbsp; also re-exported as `hybridmodels.masked_mse`</small>

```python
masked_mse(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Mean squared error reduced over a single global denominator.

Computes ``sum(mask * weights * (pred - y_observed)**2) / max(mask.sum(), 1)``
over the selected channels. A long-trajectory experiment adds terms to
the numerator and the denominator in proportion, and gets no explicit
per-experiment weighting. Use ``bal_mse`` when you want that weighting.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `pred_obs` | `Float[Array, "N T D"]` | Predicted output channels, one ``[T, D]`` block per experiment in the bucket, from ``state_to_output(simulate_fn(...))``. |
| `bp` | `BucketPayload` | Bucket data. Reads ``mask`` and ``y_observed``. |
| `channel_idx` |  | Trailing-axis indices to keep. ``None`` keeps all ``D`` channels. |
| `channel_weights` |  | Per-channel multipliers; length must match ``channel_idx`` (or ``D`` when ``channel_idx`` is ``None``). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L99)</small>

---

<a id="masked_mle"></a>

### `masked_mle()`

<small>`from hybridmodels.losses import masked_mle` &nbsp;·&nbsp; also re-exported as `hybridmodels.masked_mle`</small>

```python
masked_mle(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Total Gaussian negative log-likelihood across the bucket.

Sums the pointwise Gaussian negative log-likelihood (variance from
``bp.yvar``) over the time axis, giving one NLL per experiment and
channel, shape ``[N, D]``. Multiplies by the per-channel ``weights`` and
sums those.

Nothing is averaged. The result is a sum of log-likelihoods and grows
linearly with the number of observations. Use ``bal_mle`` for the
per-experiment averaged version.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L139)</small>

---

<a id="bal_mse"></a>

### `bal_mse()`

<small>`from hybridmodels.losses import bal_mse` &nbsp;·&nbsp; also re-exported as `hybridmodels.bal_mse`</small>

```python
bal_mse(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
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
channel does not divide by zero. Mask gating has already zeroed the
matching numerator, so that contribution is exactly ``0.0``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L165)</small>

---

<a id="bal_mle"></a>

### `bal_mle()`

<small>`from hybridmodels.losses import bal_mle` &nbsp;·&nbsp; also re-exported as `hybridmodels.bal_mle`</small>

```python
bal_mle(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Per-experiment-balanced Gaussian NLL.

Averages over time within each experiment, then over experiments. Same
reduction as ``bal_mse``, with ``_gaussian_nll_terms`` (which reads
``bp.yvar``) in place of the pointwise squared error. Returns the bucket
mean of the per-experiment, channel-weighted, time-averaged NLLs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L199)</small>

---

<a id="loss_registry"></a>

### `LOSS_REGISTRY`

<small>`from hybridmodels.losses import LOSS_REGISTRY` &nbsp;·&nbsp; also re-exported as `hybridmodels.LOSS_REGISTRY`</small>

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
