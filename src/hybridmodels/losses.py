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
- ``masked_*`` uses a single denominator across the whole bucket. This is
  the same as concatenating every observation into one flat vector and
  reducing that.
- ``bal_*`` normalises per experiment. Each experiment contributes its own
  per-channel average across time, then the bucket takes the mean over the
  ``N`` axis. Use it when experiments differ widely in observation count
  and you do not want long trajectories to dominate the gradient.

All four accept ``channel_idx`` (which trailing-``D`` indices to keep) and
``channel_weights`` (per-channel multipliers, length matching
``channel_idx`` or ``D``).
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from hybridmodels.data import BucketPayload


def _resolve_channels(
    pred_obs: Array,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> tuple[Array, Array]:
    if channel_idx is None:
        n_channels = int(pred_obs.shape[-1])
        indices = jnp.arange(n_channels)
    else:
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
    y = bp.y_observed[..., indices]
    m = bp.mask[..., indices].astype(bool)
    return p, y, m


def _gaussian_nll_terms(p: Array, y: Array, var: Array, m: Array) -> Array:
    """Pointwise Gaussian NLL ``0.5 * (log(2*pi*var) + (p-y)**2/var)`` with mask gating.

    Every input that could be NaN at a masked-out cell is replaced with a
    benign value before any arithmetic, and only then gated by the mask.
    ``var_safe`` swaps masked variance for ``1.0``; ``var_stable`` clips at
    ``1e-12`` so a real but tiny variance still gives finite ``log`` and
    division.

    Same ``[N, T, D]`` shape as the inputs, exactly zero where masked out.
    """
    var_safe = jnp.where(m, var, 1.0)
    var_stable = jnp.maximum(var_safe, 1e-12)
    p_safe = jnp.where(m, p, 0.0)
    y_safe = jnp.where(m, y, 0.0)
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

    Computes ``sum(mask * weights * (pred - y_observed)**2) / max(mask.sum(), 1)``
    over the selected channels. A long-trajectory experiment adds terms to
    the numerator and the denominator in proportion, and gets no explicit
    per-experiment weighting. Use ``bal_mse`` when you want that weighting.

    Parameters
    ----------
    pred_obs : Float[Array, "N T D"]
        Predicted output channels, one ``[T, D]`` block per experiment in
        the bucket, from ``state_to_output(simulate_fn(...))``.
    bp : BucketPayload
        Bucket data. Reads ``mask`` and ``y_observed``.
    channel_idx
        Trailing-axis indices to keep. ``None`` keeps all ``D`` channels.
    channel_weights
        Per-channel multipliers; length must match ``channel_idx`` (or ``D``
        when ``channel_idx`` is ``None``).
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    # Sanitise the inputs before squaring, then gate the output. Gating only
    # the output is not enough: jnp.where still computes the dead branch's
    # local derivative, so a NaN in a masked cell returns as 0 * nan = nan.
    p_safe = jnp.where(m, p, 0.0)
    y_safe = jnp.where(m, y, 0.0)
    se = jnp.where(m, (p_safe - y_safe) ** 2, 0.0)
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
    ``bp.yvar``) over the time axis, giving one NLL per experiment and
    channel, shape ``[N, D]``. Multiplies by the per-channel ``weights`` and
    sums those.

    Nothing is averaged. The result is a sum of log-likelihoods and grows
    linearly with the number of observations. Use ``bal_mle`` for the
    per-experiment averaged version.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    var = bp.yvar[..., indices]
    nll = _gaussian_nll_terms(p, y, var, m)
    nll_per_exp_channel = nll.sum(axis=1)
    return (nll_per_exp_channel * weights[None, :]).sum()


def bal_mse(
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
    channel does not divide by zero. Mask gating has already zeroed the
    matching numerator, so that contribution is exactly ``0.0``.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    # Sanitise the inputs before squaring, then gate the output. Gating only
    # the output is not enough: jnp.where still computes the dead branch's
    # local derivative, so a NaN in a masked cell returns as 0 * nan = nan.
    p_safe = jnp.where(m, p, 0.0)
    y_safe = jnp.where(m, y, 0.0)
    se = jnp.where(m, (p_safe - y_safe) ** 2, 0.0)
    sum_se = se.sum(axis=1)
    count = jnp.maximum(m.sum(axis=1), 1)
    avg = sum_se / count
    weighted = avg * weights[None, :]
    return weighted.sum(axis=1).mean()


def bal_mle(
    pred_obs: Float[Array, "N T D"],
    bp: BucketPayload,
    *,
    channel_idx: tuple[int, ...] | None = None,
    channel_weights: tuple[float, ...] | None = None,
) -> Array:
    """Per-experiment-balanced Gaussian NLL.

    Averages over time within each experiment, then over experiments. Same
    reduction as ``bal_mse``, with ``_gaussian_nll_terms`` (which reads
    ``bp.yvar``) in place of the pointwise squared error. Returns the bucket
    mean of the per-experiment, channel-weighted, time-averaged NLLs.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    var = bp.yvar[..., indices]
    nll = _gaussian_nll_terms(p, y, var, m)
    sum_nll = nll.sum(axis=1)
    count = jnp.maximum(m.sum(axis=1), 1)
    avg = sum_nll / count
    weighted = avg * weights[None, :]
    return weighted.sum(axis=1).mean()


LOSS_REGISTRY: dict[str, Callable[..., Array]] = {
    "mse": masked_mse,
    "mle": masked_mle,
    "bal_mse": bal_mse,
    "bal_mle": bal_mle,
}
"""Short name to loss function, so a training config can name its loss as a string."""


def _resolve_loss_fn(
    loss: Callable[..., Array] | str,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> Callable[[Array, BucketPayload], Array]:
    """Turn a training config's ``loss`` field into a ``(pred_obs, bp) -> scalar``.

    ``loss`` is either a ``LOSS_REGISTRY`` key, matched case- and
    whitespace-insensitively, or a callable already in the right shape.

    With neither ``channel_idx`` nor ``channel_weights`` the resolved
    function is returned as-is rather than wrapped. That identity matters: a
    user loss written to the bare ``(pred_obs, bp)`` signature would raise
    ``TypeError`` on the unexpected keywords if it were wrapped
    unconditionally.

    Lives here rather than in the training modules because it is registry
    lookup and channel binding, not training logic, and both loops need it.
    """
    if isinstance(loss, str):
        key = loss.lower().strip()
        if key not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss name {loss!r}; available: {sorted(LOSS_REGISTRY)}")
        base = LOSS_REGISTRY[key]
    else:
        base = loss
    if channel_idx is None and channel_weights is None:
        return base

    def loss_fn(pred_obs: Array, bp: BucketPayload) -> Array:
        return base(pred_obs, bp, channel_idx=channel_idx, channel_weights=channel_weights)

    return loss_fn
