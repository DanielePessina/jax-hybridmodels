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

import inspect
from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxhybridmodels.data import BucketPayload


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
    y = bp.y_observed[..., indices]
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
    ``var_safe`` swaps masked variance for ``1.0``; ``var_stable`` clips at
    ``1e-12`` so a real but tiny variance still gives finite ``log`` and
    division.

    Same ``[N, T, D]`` shape as the inputs, exactly zero where masked out.
    """
    var_safe = jnp.where(m, var, 1.0)
    var_stable = jnp.maximum(var_safe, 1e-12)
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


def bal_mle(
    pred_obs: Float[Array, "N T D"],
    bp: BucketPayload,
    *,
    channel_idx: tuple[int, ...] | None = None,
    channel_weights: tuple[float, ...] | None = None,
) -> Array:
    """Per-experiment-balanced Gaussian NLL.

    Averages over time within each observed experiment, then over those
    experiments. Same reduction as ``bal_mse``, with
    ``_gaussian_nll_terms`` (which reads ``bp.yvar``) in place of the
    pointwise squared error. Experiments with no observations in any selected
    channel are excluded from the outer mean.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    var = bp.yvar[..., indices]
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
    "bal_mse": bal_mse,
    "bal_mle": bal_mle,
}
"""Short name to loss function, so a training config can name its loss as a string."""


def resolve_loss_fn(
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
        y = bp.y_observed[..., indices]
        var = bp.yvar[..., indices]
        m = bp.mask[..., indices]
        return loss(p, bp._replace(y_observed=y, yvar=var, mask=m))

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
