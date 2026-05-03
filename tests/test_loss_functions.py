"""Loss-function behaviour tests (SPEC §5.4 / R-L1 / R-L2).

Pin the four built-in losses' contract:
- pure ``loss(pred_obs, bp) -> scalar`` (no jit, no simulation)
- masks contribute identically zero (multiply, not divide)
- ``channel_idx`` restricts and ``channel_weights`` re-weights
- ``bal_*`` normalise per experiment (duplication-invariant)
- ``masked_mle`` consumes ``bp.yvar`` with the Gaussian NLL relationship
- ``LOSS_REGISTRY`` exposes the four names
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from hybridmodels.data import BucketPayload
from hybridmodels.losses import (
    LOSS_REGISTRY,
    bal_mle,
    bal_mse,
    masked_mle,
    masked_mse,
)


def _make_bp(
    *,
    y_observed: Array,
    mask: Array,
    yvar: Array | None = None,
) -> BucketPayload:
    if yvar is None:
        yvar = jnp.ones_like(y_observed)
    n = y_observed.shape[0]
    t = y_observed.shape[1]
    return BucketPayload(
        ts=jnp.zeros((n, t)),
        y_observed=y_observed,
        yvar=yvar,
        mask=mask,
        covariates={},
        y0=jnp.zeros((n, 1)),
        n_obs=mask.sum().astype(jnp.int32),
    )


def _baseline_fixture() -> tuple[Array, Array, Array]:
    pred = jnp.asarray(
        [
            [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
            [[0.5, 1.0], [0.7, 1.2], [0.9, 1.4], [1.1, 1.6]],
            [[3.0, 4.0], [3.1, 4.1], [3.2, 4.2], [3.3, 4.3]],
        ]
    )
    y = jnp.asarray(
        [
            [[1.1, 1.9], [1.4, 2.6], [2.1, 2.9], [2.4, 3.6]],
            [[0.6, 1.1], [0.6, 1.3], [1.0, 1.3], [1.0, 1.7]],
            [[2.9, 4.1], [3.2, 4.0], [3.1, 4.3], [3.4, 4.2]],
        ]
    )
    mask = jnp.asarray(
        [
            [[True, True], [True, False], [False, True], [True, True]],
            [[True, True], [True, True], [False, False], [True, True]],
            [[True, False], [True, True], [True, True], [False, True]],
        ]
    )
    return pred, y, mask


@pytest.mark.parametrize("loss_fn", [masked_mse, masked_mle, bal_mse, bal_mle])
def test_loss_signature_returns_scalar(loss_fn) -> None:
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    out = loss_fn(pred, bp)
    assert out.shape == ()


def test_loss_registry_keys_and_values() -> None:
    assert LOSS_REGISTRY == {
        "mse": masked_mse,
        "mle": masked_mle,
        "bal_mse": bal_mse,
        "bal_mle": bal_mle,
    }


@pytest.mark.parametrize("loss_fn", [masked_mse, masked_mle, bal_mse, bal_mle])
def test_loss_ignores_masked_positions(loss_fn) -> None:
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    base = loss_fn(pred, bp)
    perturbed_pred = pred + jnp.where(mask, 0.0, 9_999.0)
    perturbed = loss_fn(perturbed_pred, bp)
    assert jnp.allclose(base, perturbed, atol=1e-5, rtol=0)


@pytest.mark.parametrize("loss_fn", [masked_mse, masked_mle, bal_mse, bal_mle])
def test_loss_is_nan_safe_under_mask(loss_fn) -> None:
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    pred_with_nan = jnp.where(mask, pred, jnp.nan)
    out = loss_fn(pred_with_nan, bp)
    assert jnp.isfinite(out)


def test_masked_mse_channel_idx_matches_single_channel_slice() -> None:
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    restricted = masked_mse(pred, bp, channel_idx=(0,))
    se0 = jnp.where(mask[..., 0], (pred[..., 0] - y[..., 0]) ** 2, 0.0)
    expected = se0.sum() / jnp.maximum(mask[..., 0].sum(), 1)
    assert jnp.allclose(restricted, expected, atol=1e-6, rtol=0)


def test_masked_mse_channel_weights_match_manual_formula() -> None:
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    weighted = masked_mse(pred, bp, channel_weights=(2.0, 1.0))
    se = jnp.where(mask, (pred - y) ** 2, 0.0)
    weights = jnp.asarray([2.0, 1.0])
    expected = (se * weights[None, None, :]).sum() / jnp.maximum(mask.sum(), 1)
    assert jnp.allclose(weighted, expected, atol=1e-6, rtol=0)


def test_bal_mse_channel_weights_are_linear() -> None:
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    weighted = bal_mse(pred, bp, channel_weights=(2.0, 1.0))
    only_0 = bal_mse(pred, bp, channel_idx=(0,))
    only_1 = bal_mse(pred, bp, channel_idx=(1,))
    assert jnp.allclose(weighted, 2.0 * only_0 + 1.0 * only_1, atol=1e-6, rtol=0)


@pytest.mark.parametrize("loss_fn", [bal_mse, bal_mle])
def test_bal_loss_unchanged_under_experiment_duplication(loss_fn) -> None:
    pred, y, mask = _baseline_fixture()
    yvar = 0.5 * jnp.ones_like(y)
    bp_single = _make_bp(y_observed=y, mask=mask, yvar=yvar)
    base = loss_fn(pred, bp_single)

    pred_dup = jnp.concatenate([pred, pred], axis=0)
    y_dup = jnp.concatenate([y, y], axis=0)
    mask_dup = jnp.concatenate([mask, mask], axis=0)
    yvar_dup = jnp.concatenate([yvar, yvar], axis=0)
    bp_dup = _make_bp(y_observed=y_dup, mask=mask_dup, yvar=yvar_dup)
    duplicated = loss_fn(pred_dup, bp_dup)
    assert jnp.allclose(base, duplicated, atol=1e-6, rtol=0)


def test_masked_mle_doubling_yvar_yields_predicted_delta() -> None:
    pred, y, mask = _baseline_fixture()
    var_low = 0.5 * jnp.ones_like(y)
    var_high = 1.0 * jnp.ones_like(y)
    bp_low = _make_bp(y_observed=y, mask=mask, yvar=var_low)
    bp_high = _make_bp(y_observed=y, mask=mask, yvar=var_high)
    loss_low = masked_mle(pred, bp_low)
    loss_high = masked_mle(pred, bp_high)

    se_total = jnp.where(mask, (pred - y) ** 2, 0.0).sum()
    n_unmasked = mask.sum().astype(jnp.float32)
    expected_delta = 0.5 * n_unmasked * jnp.log(2.0) - 0.5 * se_total
    assert jnp.allclose(loss_high - loss_low, expected_delta, atol=1e-5, rtol=0)


def test_bal_mle_uses_yvar() -> None:
    pred, y, mask = _baseline_fixture()
    var_a = 0.5 * jnp.ones_like(y)
    var_b = 2.0 * jnp.ones_like(y)
    bp_a = _make_bp(y_observed=y, mask=mask, yvar=var_a)
    bp_b = _make_bp(y_observed=y, mask=mask, yvar=var_b)
    assert not jnp.allclose(bal_mle(pred, bp_a), bal_mle(pred, bp_b), atol=1e-6, rtol=0)
