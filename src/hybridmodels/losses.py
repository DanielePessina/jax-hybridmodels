"""Loss functions for hybrid mechanistic training (SPEC §5.4 / R-L1 / R-L2).

Each loss is a pure function ``loss(pred_obs, bp) -> scalar``. The framework
wraps it with simulate + state_to_output + jit; the losses themselves are
unjitted. NaN-safety: contributions at masked-out positions are multiplied
by zero (never divided by the mask), so ``pred_obs`` may carry NaNs in
masked-out cells without poisoning the result.

Variants in this module
-----------------------
- ``masked_*`` — single denominator across the whole bucket (one global
  reduction). Equivalent to "concatenate every observation into one flat
  vector and reduce."
- ``bal_*``     — per-experiment normalised: each experiment contributes
  its own per-channel average across time, then the bucket mean over the
  ``N`` axis is taken. Useful when experiments have wildly different
  observation counts and you do not want long-trajectory experiments to
  dominate the gradient.

All four accept ``channel_idx`` (which trailing-``D`` indices to keep) and
``channel_weights`` (multiplicative per-channel weights, length matching
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


def _gaussian_nll_terms(
    p: Array, y: Array, var: Array, m: Array
) -> Array:
    """Pointwise Gaussian NLL ``0.5 * (log(2*pi*var) + (p-y)**2/var)`` with mask gating.

    NaN-safety strategy: every input that could be NaN at masked-out cells is
    replaced with a benign value before any arithmetic, then the final term
    is multiplied by the mask. ``var_safe`` swaps masked variance for ``1.0``
    so subsequent ``log`` and division are finite even if the simulator filled
    the cell with NaN; ``var_stable`` clips at ``1e-12`` to keep ``log`` and
    division finite when a real but tiny variance was supplied. The result
    has the same ``[N, T, D]`` shape as the inputs and is zero at masked-out
    positions.
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
    over the selected channels. Long-trajectory experiments contribute more
    terms to the numerator and the denominator proportionally — they do not
    receive an explicit per-experiment weighting (see ``bal_mse`` for that).

    Parameters
    ----------
    pred_obs : Float[Array, "N T D"]
        Predicted output (``state_to_output(simulate_fn(...))`` per experiment).
    bp : BucketPayload
        Bucket data; ``mask`` and ``y_observed`` are read.
    channel_idx
        Trailing-axis indices to keep. ``None`` keeps all ``D`` channels.
    channel_weights
        Per-channel multipliers; length must match ``channel_idx`` (or ``D``
        when ``channel_idx`` is ``None``).
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    se = jnp.where(m, (p - y) ** 2, 0.0)
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

    Sums the pointwise Gaussian NLL (variance from ``bp.yvar``) over the
    time axis to produce per-(experiment, channel) NLLs ``[N, D]``, then
    multiplies by per-channel ``weights`` and sums. **No averaging** is
    performed — this is a sum-of-likelihoods, scaling linearly with the
    number of observations. Use ``bal_mle`` for the per-experiment averaged
    counterpart.
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
    """Per-experiment-balanced MSE: average over time per experiment, then mean over experiments.

    Reduction order is ``[N, T, D] -> [N, D] (per-channel time-average) ->
    [N] (channel-weighted sum) -> scalar (mean across N)``. This balances
    experiments regardless of how many observations each contributed,
    preventing long trajectories from dominating the gradient signal.

    Per-experiment, per-channel denominators are clamped to ``1`` (via
    ``maximum(count, 1)``) so an experiment with zero observations on a
    channel does not divide by zero; the corresponding numerator is also
    zero in that case (mask gating), so the contribution is exactly ``0.0``.
    """
    indices, weights = _resolve_channels(pred_obs, channel_idx, channel_weights)
    p, y, m = _select(pred_obs, bp, indices)
    se = jnp.where(m, (p - y) ** 2, 0.0)
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

    Time-averages per experiment, then takes the mean over experiments.
    Same reduction skeleton as ``bal_mse`` but with ``_gaussian_nll_terms``
    (using ``bp.yvar``) replacing the pointwise squared error. Output is the
    bucket mean of per-experiment, channel-weighted, time-averaged NLLs.
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
