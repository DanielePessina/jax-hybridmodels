"""Loss-function behaviour tests.

Pin the four built-in losses' contract:

- pure ``loss(pred_obs, bp) -> scalar`` (no jit, no simulation)
- masks contribute identically zero (multiply, not divide)
- ``channel_idx`` restricts and ``channel_weights`` re-weights
- ``bal_*`` normalise per experiment (duplication-invariant)
- ``masked_mle`` consumes ``bp.yvar`` with the Gaussian NLL relationship
- ``LOSS_REGISTRY`` exposes the four names
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from hybridmodels.data import BucketPayload
from hybridmodels.losses import (
    LOSS_REGISTRY,
    _resolve_loss_fn,
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


def test_loss_registry_keys_are_the_documented_set() -> None:
    assert set(LOSS_REGISTRY) == {"mse", "mle", "bal_mse", "bal_mle"}


@pytest.mark.parametrize(
    ("name", "loss_fn"),
    [("mse", masked_mse), ("mle", masked_mle), ("bal_mse", bal_mse), ("bal_mle", bal_mle)],
)
def test_registry_entries_compute_what_their_name_says(name, loss_fn) -> None:
    # Asserting the dict equals a dict of the same objects only checks that
    # the literal was typed twice. Run both and compare the numbers, so a
    # mis-wired entry is caught even if the identity check would pass.
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    assert jnp.allclose(LOSS_REGISTRY[name](pred, bp), loss_fn(pred, bp))


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


def test_masked_mse_channel_idx_and_weights_agree_up_to_the_denominator() -> None:
    # Two independent API routes to the same numerator. channel_idx=(0,)
    # divides by the channel-0 mask count; channel_weights=(1, 0) keeps the
    # full-mask denominator. Relating them pins both without restating the
    # implementation's arithmetic.
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    via_idx = masked_mse(pred, bp, channel_idx=(0,))
    via_weights = masked_mse(pred, bp, channel_weights=(1.0, 0.0))
    n_channel_0 = jnp.maximum(mask[..., 0].sum(), 1)
    n_all = jnp.maximum(mask.sum(), 1)
    assert jnp.allclose(via_idx * n_channel_0, via_weights * n_all, atol=1e-6, rtol=0)


def test_masked_mse_is_homogeneous_in_channel_weights() -> None:
    # Scaling every weight scales the loss by the same factor.
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    base = masked_mse(pred, bp, channel_weights=(1.0, 1.0))
    scaled = masked_mse(pred, bp, channel_weights=(3.0, 3.0))
    assert jnp.allclose(scaled, 3.0 * base, atol=1e-6, rtol=0)


def test_masked_mse_channel_weights_are_additive() -> None:
    # The weighted loss decomposes over channels, so a two-channel weight
    # vector equals the weighted sum of its one-hot parts.
    pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    both = masked_mse(pred, bp, channel_weights=(2.0, 1.0))
    only_0 = masked_mse(pred, bp, channel_weights=(1.0, 0.0))
    only_1 = masked_mse(pred, bp, channel_weights=(0.0, 1.0))
    assert jnp.allclose(both, 2.0 * only_0 + 1.0 * only_1, atol=1e-6, rtol=0)


def test_masked_mse_is_zero_for_a_perfect_fit_and_positive_otherwise() -> None:
    _pred, y, mask = _baseline_fixture()
    bp = _make_bp(y_observed=y, mask=mask)
    assert float(masked_mse(y, bp)) == 0.0
    assert float(masked_mse(y + 1.0, bp)) > 0.0


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


class TestMaskedNanGradients:
    """The module docstring promises masked NaNs cannot poison the result.

    "Result" has to mean the gradient too. ``length_schedule`` masks the
    tail of every trajectory, so a solve that diverges only in the masked
    tail would otherwise report a finite, plausible loss while handing the
    optimiser NaN gradients.
    """

    def _payload(self):
        mask = jnp.array([[[True], [True], [False]]])
        return BucketPayload(
            ts=jnp.zeros((1, 3)),
            y_observed=jnp.zeros((1, 3, 1)),
            yvar=jnp.ones((1, 3, 1)),
            mask=mask,
            covariates={},
            y0=jnp.zeros((1, 1)),
            n_obs=jnp.array([2]),
        )

    @pytest.mark.parametrize("loss_fn", [masked_mse, bal_mse, masked_mle, bal_mle])
    def test_gradient_is_finite_when_masked_cells_hold_nan(self, loss_fn):
        bp = self._payload()
        pred = jnp.array([[[1.0], [1.0], [jnp.nan]]])
        grad = jax.grad(lambda p: loss_fn(p, bp))(pred)
        assert not bool(jnp.isnan(grad).any())

    @pytest.mark.parametrize("loss_fn", [masked_mse, bal_mse, masked_mle, bal_mle])
    def test_gradient_is_finite_when_masked_cells_hold_inf(self, loss_fn):
        bp = self._payload()
        pred = jnp.array([[[1.0], [1.0], [jnp.inf]]])
        grad = jax.grad(lambda p: loss_fn(p, bp))(pred)
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.parametrize("loss_fn", [masked_mse, bal_mse, masked_mle, bal_mle])
    def test_masked_cell_gets_exactly_zero_gradient(self, loss_fn):
        # Stronger than "not NaN": a masked observation must not influence
        # the fit at all.
        bp = self._payload()
        pred = jnp.array([[[1.0], [1.0], [7.0]]])
        grad = jax.grad(lambda p: loss_fn(p, bp))(pred)
        assert float(grad[0, 2, 0]) == 0.0


class TestResolveLossFn:
    """``_resolve_loss_fn`` turns a config's ``loss`` field into a callable.

    It had no direct test while living duplicated in the two training
    modules, and the string path was reachable from the suite only through a
    full training run. Both training loops now share this one copy, so its
    contract is worth pinning.
    """

    def _bp(self) -> BucketPayload:
        return _make_bp(
            y_observed=jnp.array([[[1.0, 2.0]]]),
            mask=jnp.array([[[True, True]]]),
        )

    @pytest.mark.parametrize("name", ["mse", "mle", "bal_mse", "bal_mle"])
    def test_registry_names_resolve_to_their_function(self, name):
        assert _resolve_loss_fn(name, None, None) is LOSS_REGISTRY[name]

    @pytest.mark.parametrize("spelling", ["MSE", "  mse", "Mse  "])
    def test_names_are_case_and_whitespace_insensitive(self, spelling):
        assert _resolve_loss_fn(spelling, None, None) is LOSS_REGISTRY["mse"]

    def test_unknown_name_raises_listing_what_is_available(self):
        with pytest.raises(ValueError, match="Unknown loss name"):
            _resolve_loss_fn("rmse", None, None)
        # The message has to name the alternatives; a bare rejection leaves
        # the user guessing at the registry contents.
        with pytest.raises(ValueError, match="bal_mle"):
            _resolve_loss_fn("rmse", None, None)

    def test_callable_passes_through_untouched_without_channel_args(self):
        # Identity, not a wrapper: a user loss that takes no channel kwargs
        # would raise TypeError if it were wrapped unconditionally.
        def custom(pred_obs, bp):
            return jnp.asarray(0.0)

        assert _resolve_loss_fn(custom, None, None) is custom

    def test_channel_args_are_bound_onto_a_registry_loss(self):
        bp = self._bp()
        pred = jnp.array([[[1.0, 99.0]]])
        # Keeping only channel 0 must hide the large error on channel 1.
        restricted = _resolve_loss_fn("mse", (0,), None)
        assert float(restricted(pred, bp)) == pytest.approx(0.0)
        assert float(masked_mse(pred, bp)) > 1.0

    def test_channel_weights_reach_the_underlying_loss(self):
        bp = self._bp()
        pred = jnp.array([[[3.0, 2.0]]])
        singly = _resolve_loss_fn("mse", None, (1.0, 1.0))
        doubly = _resolve_loss_fn("mse", None, (2.0, 2.0))
        assert float(doubly(pred, bp)) == pytest.approx(2.0 * float(singly(pred, bp)))

    def test_channel_args_are_bound_onto_a_custom_callable(self):
        seen = {}

        def custom(pred_obs, bp, *, channel_idx=None, channel_weights=None):
            seen["channel_idx"] = channel_idx
            seen["channel_weights"] = channel_weights
            return jnp.asarray(0.0)

        wrapped = _resolve_loss_fn(custom, (1,), (0.5,))
        assert wrapped is not custom
        wrapped(jnp.zeros((1, 1, 2)), self._bp())
        assert seen == {"channel_idx": (1,), "channel_weights": (0.5,)}
