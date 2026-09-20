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

---

<a id="masked_mse"></a>

### `masked_mse()`

<small>`from jaxhybridmodels.losses import masked_mse` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.masked_mse`</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L131)</small>

---

<a id="masked_mle"></a>

### `masked_mle()`

<small>`from jaxhybridmodels.losses import masked_mle` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.masked_mle`</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L166)</small>

---

<a id="bal_mse"></a>

### `bal_mse()`

<small>`from jaxhybridmodels.losses import bal_mse` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.bal_mse`</small>

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
channel does not divide by zero. Experiments with no observations in any
selected channel (for example, a trajectory-penalty probe) are excluded
from the outer mean rather than diluting measured experiments.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L192)</small>

---

<a id="bal_mle"></a>

### `bal_mle()`

<small>`from jaxhybridmodels.losses import bal_mle` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.bal_mle`</small>

```python
bal_mle(
    pred_obs: "Float[Array, 'N T D']",
    bp: 'BucketPayload',
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
) -> Array
```

Per-experiment-balanced Gaussian NLL.

Averages over time within each observed experiment, then over those
experiments. Same reduction as ``bal_mse``, with
``_gaussian_nll_terms`` (which reads ``bp.yvar``) in place of the
pointwise squared error. Experiments with no observations in any selected
channel are excluded from the outer mean.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L225)</small>

---

<a id="loss_registry"></a>

### `LOSS_REGISTRY`

<small>`from jaxhybridmodels.losses import LOSS_REGISTRY` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.LOSS_REGISTRY`</small>

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

<small>`from jaxhybridmodels.losses import resolve_loss_fn` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.resolve_loss_fn`</small>

```python
resolve_loss_fn(
    loss: 'Callable[..., Array] | str',
    channel_idx: 'tuple[int, ...] | None',
    channel_weights: 'tuple[float, ...] | None',
) -> Callable[[Array, BucketPayload], Array]
```

Turn a training config's ``loss`` field into a ``(pred_obs, bp) -> scalar``.

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
  ``channel_weights`` cannot be projected this way — per-channel
  weighting must happen inside a loss's own reduction — so a plain
  callable combined with ``channel_weights`` raises rather than
  silently ignoring the weights.

Lives here rather than in the training modules because it is registry
lookup and channel binding, not training logic, and both loops need it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/losses.py#L263)</small>
