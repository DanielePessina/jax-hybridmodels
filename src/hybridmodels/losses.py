"""Loss functions for hybrid mechanistic training (SPEC §5.4 / R-L1 / R-L2).

Each loss is a pure function ``loss(pred_obs, bp) -> scalar``. The framework
wraps it with simulate + state_to_output + jit; the losses themselves are
unjitted. NaN-safety: contributions at masked-out positions are multiplied
by zero (never divided by the mask), so ``pred_obs`` may carry NaNs in
masked-out cells without poisoning the result.
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
