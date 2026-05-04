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
over the selected channels. Long-trajectory experiments contribute more
terms to the numerator and the denominator proportionally — they do not
receive an explicit per-experiment weighting (see ``bal_mse`` for that).

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `pred_obs` | `Float[Array, "N T D"]` | Predicted output (``state_to_output(simulate_fn(...))`` per experiment). |
| `bp` | `BucketPayload` | Bucket data; ``mask`` and ``y_observed`` are read. |
| `channel_idx` |  | Trailing-axis indices to keep. ``None`` keeps all ``D`` channels. |
| `channel_weights` |  | Per-channel multipliers; length must match ``channel_idx`` (or ``D`` when ``channel_idx`` is ``None``). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L94)</small>

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

Sums the pointwise Gaussian NLL (variance from ``bp.yvar``) over the
time axis to produce per-(experiment, channel) NLLs ``[N, D]``, then
multiplies by per-channel ``weights`` and sums. **No averaging** is
performed — this is a sum-of-likelihoods, scaling linearly with the
number of observations. Use ``bal_mle`` for the per-experiment averaged
counterpart.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L128)</small>

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

Per-experiment-balanced MSE: average over time per experiment, then mean over experiments.

Reduction order is ``[N, T, D] -> [N, D] (per-channel time-average) ->
[N] (channel-weighted sum) -> scalar (mean across N)``. This balances
experiments regardless of how many observations each contributed,
preventing long trajectories from dominating the gradient signal.

Per-experiment, per-channel denominators are clamped to ``1`` (via
``maximum(count, 1)``) so an experiment with zero observations on a
channel does not divide by zero; the corresponding numerator is also
zero in that case (mask gating), so the contribution is exactly ``0.0``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L152)</small>

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

Time-averages per experiment, then takes the mean over experiments.
Same reduction skeleton as ``bal_mse`` but with ``_gaussian_nll_terms``
(using ``bp.yvar``) replacing the pointwise squared error. Output is the
bucket mean of per-experiment, channel-weighted, time-averaged NLLs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/losses.py#L181)</small>

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
