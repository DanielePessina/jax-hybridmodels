"""How far a simulated trajectory is from the measurements.

Each loss is a pure function ``loss(pred_obs, bp) -> scalar``, where
``pred_obs`` has shape ``[N, T, D]`` (``N`` experiments in the bucket,
``T`` timestamps on the bucket's union axis, ``D`` output channels) and
``bp`` is the matching ``BucketPayload``. The framework composes the loss
with the simulator and ``state_to_output`` and compiles the whole
pipeline. The losses here are left uncompiled, so they can be written and
tested by calling them on plain arrays.

Only cells the bucket's ``mask`` marks as real measurements count. The
union timestamp axis puts a row at every time any channel was measured,
so most cells of a sparse dataset hold filler.

NaN safety. Contributions at masked-out positions are multiplied by zero,
never divided by the mask, so ``pred_obs`` may carry NaNs in masked-out
cells without poisoning the result. Per-experiment denominators are
clamped at ``1``, so an experiment with zero observations on a channel
does not divide by zero.

Variants in this module
-----------------------
- ``mse`` is the mean over every measured cell, and ``mle`` the summed
  Gaussian negative log-likelihood.
- ``per_experiment_*`` normalises per experiment. Each experiment contributes its own
  per-channel average across time, then the experiments are averaged. Use
  it when experiments differ widely in observation count and you do not
  want long trajectories to dominate the gradient. ``per_experiment_*`` adds the
  channels where ``mse`` averages them, so it is about ``D`` times larger
  for ``D`` channels.

Each function here reduces one bucket. The trainers put the buckets back
together with :class:`Pooling`, weighting each bucket by what its value is
an average over (see :func:`loss_pooling`). So ``mse`` over a dataset is
the mean over all its measurements, and ``per_experiment_mse`` the mean over all its
experiments, however the experiments were split into buckets.

All four accept ``channel_idx`` (which trailing-``D`` indices to keep) and
``channel_weights`` (per-channel multipliers, length matching
``channel_idx`` or ``D``). Training code does not pass these by hand:
:class:`~jaxhybridmodels.Objective` takes weights keyed by channel name and
resolves them to this pair.
"""

# ruff: noqa: F722

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from jaxhybridmodels.data import BucketPayload, horizon_rows


def _validate_channel_idx(channel_idx: tuple[int, ...], total_channels: int) -> None:
    """Reject channel selections that JAX advanced indexing would clamp."""
    invalid = [idx for idx in channel_idx if idx < 0 or idx >= total_channels]
    if invalid:
        raise ValueError(
            f"channel_idx contains out-of-range entries {invalid}; "
            f"pred_obs has {total_channels} channels"
        )


def _resolve_channels(
    pred_obs: Array,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> tuple[Array, Array]:
    if channel_idx is None:
        n_channels = int(pred_obs.shape[-1])
        indices = jnp.arange(n_channels)
    else:
        total_channels = int(pred_obs.shape[-1])
        _validate_channel_idx(channel_idx, total_channels)
        n_channels = len(channel_idx)
        indices = jnp.asarray(channel_idx)
    if channel_weights is None:
        weights = jnp.ones(n_channels)
    else:
        if len(channel_weights) != n_channels:
            raise ValueError("channel_weights length must match channel_idx length")
        weights = jnp.asarray(channel_weights)
    return indices, weights


def _select(
    pred_obs: Array,
    bp: BucketPayload,
    indices: Array,
) -> tuple[Array, Array, Array]:
    p = pred_obs[..., indices]
    y = bp.observed[..., indices]
    m = bp.mask[..., indices].astype(bool)
    return p, y, m


def _masked_safe(p: Array, y: Array, m: Array) -> tuple[Array, Array]:
    """Benign replacements for masked cells, before any arithmetic.

    Gating only the output is not enough: ``jnp.where`` still computes the
    dead branch's local derivative, so a NaN in a masked cell returns as
    ``0 * nan = nan``. Swap masked values for zeros up front, then gate.
    """
    return jnp.where(m, p, 0.0), jnp.where(m, y, 0.0)


def _masked_squared_error(p: Array, y: Array, m: Array) -> Array:
    """Per-cell squared error, zeroed at masked cells.

    Sanitises the inputs via :func:`_masked_safe` before squaring, so a
    NaN or inf at a masked cell cannot leak through the gate's dead branch.
    """
    p_safe, y_safe = _masked_safe(p, y, m)
    return jnp.where(m, (p_safe - y_safe) ** 2, 0.0)


def _gaussian_nll_terms(p: Array, y: Array, var: Array, m: Array) -> Array:
    """Pointwise Gaussian NLL ``0.5 * (log(2*pi*var) + (p-y)**2/var)`` with mask gating.

    Every input that could be NaN at a masked-out cell is replaced with a
    benign value before any arithmetic, and only then gated by the mask.
    ``var_safe`` swaps masked variance for ``1.0``. ``var_stable`` clips at
    the dtype's machine epsilon squared (about 1e-14 in float32, 5e-32 in
    float64), only to keep the gradient in the variance, ``r^2 / var^2``,
    finite. A fixed floor such as 1e-12 would change the likelihood of data
    measured in small units, where a variance of 1e-14 is real.

    Same ``[N, T, D]`` shape as the inputs, exactly zero where masked out.
    """
    var_safe = jnp.where(m, var, 1.0)
    var_stable = jnp.maximum(var_safe, jnp.finfo(jnp.result_type(var_safe)).eps ** 2)
    p_safe, y_safe = _masked_safe(p, y, m)
    log_term = jnp.where(m, 0.5 * jnp.log(2.0 * jnp.pi * var_stable), 0.0)
    sq_term = jnp.where(m, 0.5 * (p_safe - y_safe) ** 2 / var_stable, 0.0)
    return log_term + sq_term


def masked_mse(
    pred_obs: Float[Array, "N T D"],
    bp: BucketPayload,
    *,
    channel_idx: tuple[int, ...] | None = None,
    channel_weights: tuple[float, ...] | None = None,
) -> Array:
    """Mean squared error reduced over a single global denominator.

    Computes ``sum(mask * weights * (pred - observed)**2) / max(mask.sum(), 1)``
    over the selected channels. A long-trajectory experiment adds terms to
    the numerator and the denominator in proportion, and gets no explicit
    per-experiment weighting. Use ``per_experiment_mse`` when you want that weighting.

    Parameters
    ----------
    pred_obs : Float[Array, "N T D"]
        Predicted output channels, one ``[T, D]`` block per experiment in
        the bucket, from ``state_to_output(simulate_fn(...))``.
    bp : BucketPayload
        Bucket data. Reads ``mask`` and ``observed``.
    channel_idx
        Trailing-axis indices to keep. ``None`` keeps all ``D`` channels.
    channel_weights
        Per-channel multipliers; length must match ``channel_idx`` (or ``D``
        when ``channel_idx`` is ``None``).
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    se = _masked_squared_error(p, y, m)
    weighted = se * weights[None, None, :]
    denom = jnp.maximum(m.sum(), 1)
    return weighted.sum() / denom


def masked_mle(
    pred_obs: Float[Array, "N T D"],
    bp: BucketPayload,
    *,
    channel_idx: tuple[int, ...] | None = None,
    channel_weights: tuple[float, ...] | None = None,
) -> Array:
    """Total Gaussian negative log-likelihood across the bucket.

    Sums the pointwise Gaussian negative log-likelihood (variance from
    ``bp.variance``) over the time axis, giving one NLL per experiment and
    channel, shape ``[N, D]``. Multiplies by the per-channel ``weights`` and
    sums those.

    Nothing is averaged. The result is a sum of log-likelihoods and grows
    linearly with the number of observations. Use ``per_experiment_mle`` for the
    per-experiment averaged version.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    var = bp.variance[..., indices]
    nll = _gaussian_nll_terms(p, y, var, m)
    nll_per_exp_channel = nll.sum(axis=1)
    return (nll_per_exp_channel * weights[None, :]).sum()


def per_experiment_mse(
    pred_obs: Float[Array, "N T D"],
    bp: BucketPayload,
    *,
    channel_idx: tuple[int, ...] | None = None,
    channel_weights: tuple[float, ...] | None = None,
) -> Array:
    """MSE averaged over time within each experiment, then averaged over experiments.

    Reduction order is ``[N, T, D] -> [N, D]`` (per-channel time average),
    then ``[N]`` (channel-weighted sum), then a scalar (mean across ``N``).
    Every experiment therefore counts the same, however many observations it
    contributed, so a long trajectory cannot dominate the gradient.

    Per-experiment, per-channel denominators are clamped to ``1`` with
    ``maximum(count, 1)``, so an experiment with zero observations on a
    channel does not divide by zero. Experiments with no observations in any
    selected channel (for example, a trajectory-penalty probe) are excluded
    from the outer mean rather than diluting measured experiments.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    se = _masked_squared_error(p, y, m)
    sum_se = se.sum(axis=1)
    count = jnp.maximum(m.sum(axis=1), 1)
    avg = sum_se / count
    weighted = avg * weights[None, :]
    per_experiment = weighted.sum(axis=1)
    has_observations = m.any(axis=(1, 2))
    n_experiments = jnp.maximum(has_observations.sum(), 1)
    return jnp.where(has_observations, per_experiment, 0.0).sum() / n_experiments


def per_experiment_mle(
    pred_obs: Float[Array, "N T D"],
    bp: BucketPayload,
    *,
    channel_idx: tuple[int, ...] | None = None,
    channel_weights: tuple[float, ...] | None = None,
) -> Array:
    """Per-experiment-balanced Gaussian NLL.

    Averages over time within each observed experiment, then over those
    experiments. Same reduction as ``per_experiment_mse``, with
    ``_gaussian_nll_terms`` (which reads ``bp.variance``) in place of the
    pointwise squared error. Experiments with no observations in any selected
    channel are excluded from the outer mean.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    var = bp.variance[..., indices]
    nll = _gaussian_nll_terms(p, y, var, m)
    sum_nll = nll.sum(axis=1)
    count = jnp.maximum(m.sum(axis=1), 1)
    avg = sum_nll / count
    weighted = avg * weights[None, :]
    per_experiment = weighted.sum(axis=1)
    has_observations = m.any(axis=(1, 2))
    n_experiments = jnp.maximum(has_observations.sum(), 1)
    return jnp.where(has_observations, per_experiment, 0.0).sum() / n_experiments


LOSS_REGISTRY: dict[str, Callable[..., Array]] = {
    "mse": masked_mse,
    "mle": masked_mle,
    "per_experiment_mse": per_experiment_mse,
    "per_experiment_mle": per_experiment_mle,
}
"""Short name to loss function, so an ``Objective`` can name its loss as a string."""

POOLING_KINDS = ("observations", "experiments", "sum")
"""How a loss's per-bucket values combine into the loss over the whole dataset."""

_BUILTIN_POOLING: tuple[tuple[Callable[..., Array], str], ...] = (
    (masked_mse, "observations"),
    (masked_mle, "sum"),
    (per_experiment_mse, "experiments"),
    (per_experiment_mle, "experiments"),
)


def loss_pooling(loss: Callable[..., Array] | str) -> str:
    """How ``loss`` combines across buckets: one of :data:`POOLING_KINDS`.

    Each bucket is scored on its own, so the trainers need to know what a
    bucket's value is an average over to put the buckets back together:

    - ``"observations"``: a mean over measured cells (``mse``). Buckets are
      weighted by their measured-cell count, which gives the mean over every
      measurement in the dataset.
    - ``"experiments"``: a mean over experiments (``per_experiment_mse``, ``per_experiment_mle``).
      Buckets are weighted by how many measured experiments they hold, so
      every experiment counts the same across the dataset.
    - ``"sum"``: already a sum (``mle``). Bucket values are added.

    A ``functools.partial`` of a built-in pools as the built-in. Any other
    custom loss declares its kind with a ``pooling`` attribute, for example
    ``my_loss.pooling = "sum"``, and defaults to ``"observations"``: a
    wrapper around ``per_experiment_mse`` must set ``"experiments"`` itself.
    """
    if isinstance(loss, str):
        loss = LOSS_REGISTRY[loss.lower().strip()]
    # A partial of a built-in reduces the same way the built-in does.
    base: object = loss
    while isinstance(base, functools.partial):
        base = base.func
    kind = next((k for fn, k in _BUILTIN_POOLING if fn is base), None)
    if kind is None:
        kind = getattr(loss, "pooling", None) or getattr(base, "pooling", "observations")
    if kind not in POOLING_KINDS:
        raise ValueError(f"loss pooling must be one of {POOLING_KINDS}; got {kind!r}")
    return kind


class Pooling(NamedTuple):
    """How to weight each bucket's loss so that the buckets add up to one loss.

    Build it with :meth:`jaxhybridmodels.Objective.pooling`. ``channel_idx``
    is the objective's channel selection, so only the channels the loss
    reads are counted.
    """

    kind: str
    channel_idx: tuple[int, ...] | None = None

    def weight(self, bp: BucketPayload) -> Array:
        """What one bucket's loss is an average over, after any horizon mask."""
        mask = bp.mask if self.channel_idx is None else bp.mask[..., jnp.asarray(self.channel_idx)]
        if self.kind == "observations":
            return mask.sum().astype(float)
        if self.kind == "experiments":
            return mask.any(axis=(1, 2)).sum().astype(float)
        return jnp.asarray(1.0)

    def shares(
        self, buckets: Sequence[BucketPayload], horizon: ArrayLike = 1.0
    ) -> tuple[Array, ...]:
        """Each bucket's ``(data, penalty)`` multipliers for the dataset loss.

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
        """
        sizes = [float(bp.ts.shape[0]) for bp in buckets]
        penalty = [size / sum(sizes) for size in sizes]
        if self.kind == "sum":
            data = [1.0] * len(buckets)
        else:
            weights = []
            for bp in buckets:
                keep = horizon_rows(bp.ts, bp.mask.any(axis=-1), horizon)
                weights.append(float(self.weight(bp._replace(mask=bp.mask & keep[:, :, None]))))
            total = sum(weights)
            equal = [1.0 / len(buckets)] * len(buckets)
            data = [w / total for w in weights] if total > 0.0 else equal
        return tuple(jnp.asarray([d, q]) for d, q in zip(data, penalty, strict=True))


def resolve_loss_fn(
    loss: Callable[..., Array] | str,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> Callable[[Array, BucketPayload], Array]:
    """Turn an ``Objective``'s ``loss`` field into a ``(pred_obs, bp) -> scalar``.

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
    """
    if isinstance(loss, str):
        key = loss.lower().strip()
        if key not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss name {loss!r}; available: {sorted(LOSS_REGISTRY)}")
        base = LOSS_REGISTRY[key]
        if channel_idx is None and channel_weights is None:
            return base

        def named_loss(pred_obs: Array, bp: BucketPayload) -> Array:
            return base(pred_obs, bp, channel_idx=channel_idx, channel_weights=channel_weights)

        return named_loss

    if channel_idx is None and channel_weights is None:
        return loss

    if _accepts_channel_kwargs(loss):

        def kwarg_loss(pred_obs: Array, bp: BucketPayload) -> Array:
            return loss(pred_obs, bp, channel_idx=channel_idx, channel_weights=channel_weights)

        return kwarg_loss

    if channel_weights is not None:
        raise ValueError(
            "channel_weights cannot be applied to a custom loss with the plain "
            "(pred_obs, bp) signature: per-channel weighting must happen inside "
            "the loss's own reduction. Either pass a built-in loss name "
            f"({sorted(LOSS_REGISTRY)}), or accept channel_idx/channel_weights "
            "in your loss's signature."
        )

    if channel_idx is None:
        raise ValueError("channel_idx is required when projecting a custom loss")
    indices = jnp.asarray(channel_idx)

    def projected_loss(pred_obs: Array, bp: BucketPayload) -> Array:
        _validate_channel_idx(channel_idx, int(pred_obs.shape[-1]))
        p = pred_obs[..., indices]
        y = bp.observed[..., indices]
        var = bp.variance[..., indices]
        m = bp.mask[..., indices]
        return loss(p, bp._replace(observed=y, variance=var, mask=m))

    return projected_loss


def _accepts_channel_kwargs(loss: Callable[..., Array]) -> bool:
    """Does the callable accept ``channel_idx``/``channel_weights`` or ``**kwargs``?

    Guiding how channel selection composes with a user loss: a callable that
    can take the channel arguments receives them as keywords (so the built-in
    losses reduce with weights inside their own sum); a plain ``(pred_obs,
    bp)`` callable is projected instead. Signature inspection is cheap and
    runs once per config, never inside a trace.
    """
    try:
        sig = inspect.signature(loss)
    except (ValueError, TypeError):
        return False
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    names = set(sig.parameters)
    return "channel_idx" in names and "channel_weights" in names
