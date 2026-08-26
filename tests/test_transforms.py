"""Tests for the transform and warp registries.

Behavioural claims only. A test that retypes a transform's own algebra
proves the formula was copied twice, not that it is right, so the checks
below are about round trips, orderings, and what the gradient does.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hybridmodels import BoundScaler
from hybridmodels.transforms import (
    BOUND_TRANSFORMS,
    WARPS,
    BoundTransform,
    Warp,
    register_bound_transform,
    register_warp,
)

TRANSFORM_NAMES = sorted(BOUND_TRANSFORMS)
WARP_NAMES = sorted(WARPS)


@pytest.fixture
def restore_registries():
    t, w = dict(BOUND_TRANSFORMS), dict(WARPS)
    yield
    BOUND_TRANSFORMS.clear()
    BOUND_TRANSFORMS.update(t)
    WARPS.clear()
    WARPS.update(w)


class TestTransformRoundTrip:
    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_round_trip_is_the_identity_inside_the_box(self, name):
        s = BoundScaler(bounds=((0.0, 10.0), (-2.0, 2.0)), transform=name)
        x = jnp.array([3.5, 0.75])
        assert jnp.allclose(s.from_latent(s.to_latent(x)), x, rtol=1e-4)

    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_output_never_leaves_the_box(self, name):
        s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
        out = s.from_latent(jnp.array([-1e6, -20.0, 0.0, 20.0, 1e6]))
        assert float(out.min()) >= 0.0 and float(out.max()) <= 10.0

    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_zero_latent_is_the_box_midpoint(self, name):
        s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
        assert jnp.allclose(s.from_latent(jnp.array([0.0])), 5.0, atol=1e-5)

    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_to_latent_stays_differentiable_outside_the_box(self, name):
        s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
        g = jax.grad(lambda x: s.to_latent(x).sum())(jnp.array([13.0]))
        assert jnp.all(jnp.isfinite(g)) and float(g[0]) > 0.0

    def test_unknown_transform_names_the_registered_set(self):
        with pytest.raises(ValueError, match="algebraic"):
            BoundScaler(bounds=((0.0, 1.0),), transform="nope")

    def test_tanh_is_absent_because_it_is_sigmoid_at_half_temperature(self):
        # (1 + tanh z) / 2 == sigmoid(2z) identically, so a "tanh" entry
        # would be the sigmoid entry with a factor-2 temperature convention.
        assert "tanh" not in BOUND_TRANSFORMS
        half = BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid", temperature=0.5)
        z = jnp.array([0.7])
        tanh_equivalent = 0.0 + 10.0 * 0.5 * (1.0 + jnp.tanh(z))
        assert jnp.allclose(half.from_latent(z), tanh_equivalent, rtol=1e-5)


class TestGradientDecay:
    """The reason to offer a choice of squash at all."""

    def test_sigmoid_is_exactly_dead_where_the_polynomial_ones_are_not(self):
        z = jnp.array([25.0])
        grads = {}
        for name in ("sigmoid", "algebraic", "softsign"):
            s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
            grads[name] = float(jax.grad(lambda v, _s=s: _s.from_latent(v).sum())(z)[0])
        assert grads["sigmoid"] == 0.0
        assert grads["algebraic"] > 0.0
        assert grads["softsign"] > 0.0

    def test_tail_ordering_is_softsign_then_algebraic_then_sigmoid(self):
        z = jnp.array([12.0])
        g = {}
        for name in ("sigmoid", "algebraic", "softsign"):
            s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
            g[name] = float(jax.grad(lambda v, _s=s: _s.from_latent(v).sum())(z)[0])
        assert g["sigmoid"] < g["algebraic"] < g["softsign"]


class TestKnee:
    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_default_knee_marks_the_outer_five_percent_of_the_box(self, name):
        # The knee has to be transform-derived: it encodes a physical
        # criterion, and sigmoid's 2.944 means 12.5% from the bound on
        # softsign, not 5%.
        s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
        at_knee = float(s.from_latent(jnp.array([s.z_knee]))[0])
        assert at_knee == pytest.approx(9.5, abs=0.02)

    def test_explicit_knee_overrides_the_transform_default(self):
        s = BoundScaler(bounds=((0.0, 10.0),), transform="softsign", z_knee=1.0)
        assert s.z_knee == 1.0

    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_saturation_is_zero_at_the_knee_and_positive_beyond(self, name):
        s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
        assert float(s.saturation(jnp.array([s.z_knee]))) == 0.0
        assert float(s.saturation(jnp.array([s.z_knee * 2.0 + 1.0]))) > 0.0


class TestWarps:
    def test_log10_puts_the_midpoint_in_the_middle_of_the_decades(self):
        # The point of the feature. A linear box over (1e-6, 1e2) has a
        # midpoint of 50, so the whole low end is unreachable in practice.
        linear = BoundScaler(bounds=((1e-6, 1e2),), transform="sigmoid")
        log10 = BoundScaler(bounds=((1e-6, 1e2),), transform="sigmoid", warp="log10")
        z = jnp.array([0.0])
        assert float(linear.from_latent(z)[0]) == pytest.approx(50.0, rel=1e-3)
        assert float(log10.from_latent(z)[0]) == pytest.approx(1e-2, rel=1e-3)

    @pytest.mark.parametrize("warp", WARP_NAMES)
    def test_round_trip_holds_under_every_warp(self, warp):
        bounds = ((1e-6, 1e2),) if warp != "linear" else ((-5.0, 5.0),)
        s = BoundScaler(bounds=bounds, transform="sigmoid", warp=warp)
        x = jnp.array([1e-3]) if warp != "linear" else jnp.array([1.25])
        assert jnp.allclose(s.from_latent(s.to_latent(x)), x, rtol=1e-4)

    def test_log_and_log10_reach_the_same_physical_values(self):
        # They differ in latent scale, not in what is representable.
        a = BoundScaler(bounds=((1e-6, 1e2),), transform="sigmoid", warp="log")
        b = BoundScaler(bounds=((1e-6, 1e2),), transform="sigmoid", warp="log10")
        x = jnp.array([2e-4])
        assert jnp.allclose(a.from_latent(a.to_latent(x)), b.from_latent(b.to_latent(x)), rtol=1e-4)

    def test_log_warp_resolves_the_low_decades_a_linear_warp_cannot(self):
        # Two physical values one decade apart near the bottom of the box
        # must be distinguishable in latent space.
        bounds = ((1e-6, 1e2),)
        lin = BoundScaler(bounds=bounds, transform="sigmoid")
        log = BoundScaler(bounds=bounds, transform="sigmoid", warp="log10")
        pair = jnp.array([1e-5, 1e-4])
        lin_gap = abs(float(lin.to_latent(pair[:1])[0]) - float(lin.to_latent(pair[1:])[0]))
        log_gap = abs(float(log.to_latent(pair[:1])[0]) - float(log.to_latent(pair[1:])[0]))
        assert log_gap > 10.0 * lin_gap

    def test_output_stays_inside_the_box_under_a_log_warp(self):
        s = BoundScaler(bounds=((1e-6, 1e2),), transform="sigmoid", warp="log10")
        out = s.from_latent(jnp.array([-1e4, 0.0, 1e4]))
        # rtol, not an exact compare: 10 ** -6 in float32 lands a few ulp
        # below 1e-6, which is rounding, not an escaped bound.
        assert float(out.min()) == pytest.approx(1e-6, rel=1e-5)
        assert float(out.max()) == pytest.approx(1e2, rel=1e-5)

    def test_nonpositive_bounds_under_a_log_warp_raise(self):
        with pytest.raises(ValueError, match="undefined at or below zero"):
            BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid", warp="log10")

    def test_unknown_warp_names_the_registered_set(self):
        with pytest.raises(ValueError, match="linear"):
            BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid", warp="nope")


class TestBoundsValidation:
    def test_infinite_bound_raises(self):
        # Used to construct silently, then map every finite input to one
        # latent, return inf from from_latent, and nan from input_violation.
        with pytest.raises(ValueError, match="finite"):
            BoundScaler(bounds=((0.0, float("inf")),), transform="sigmoid")

    def test_nan_bound_raises(self):
        with pytest.raises(ValueError, match="finite"):
            BoundScaler(bounds=((float("nan"), 1.0),), transform="sigmoid")

    def test_inverted_bound_raises(self):
        with pytest.raises(ValueError, match="low < high"):
            BoundScaler(bounds=((10.0, 0.0),), transform="sigmoid")

    def test_degenerate_bound_raises(self):
        with pytest.raises(ValueError, match="low < high"):
            BoundScaler(bounds=((5.0, 5.0),), transform="sigmoid")


class TestExtensibility:
    """Users register their own without touching the package."""

    def test_a_custom_transform_works_end_to_end(self, restore_registries):
        # arctan, which the shipped set omits as dominated by softsign.
        register_bound_transform(
            "arctan",
            BoundTransform(
                forward=lambda z: 0.5 + jnp.arctan(z) / jnp.pi,
                inverse=lambda s: jnp.tan(jnp.pi * (s - 0.5)),
                inverse_slope=lambda s: jnp.pi / jnp.cos(jnp.pi * (s - 0.5)) ** 2,
                knee=6.313752,
            ),
        )
        s = BoundScaler(bounds=((0.0, 10.0),), transform="arctan")
        x = jnp.array([2.5])
        assert jnp.allclose(s.from_latent(s.to_latent(x)), x, rtol=1e-4)
        assert float(s.from_latent(jnp.array([s.z_knee]))[0]) == pytest.approx(9.5, abs=0.02)

    def test_a_custom_warp_works_end_to_end(self, restore_registries):
        # A square-root warp, for a quantity whose resolution should grow
        # toward the low end but which may legitimately reach zero.
        register_warp(
            "sqrt",
            Warp(forward=jnp.sqrt, inverse=lambda w: jnp.asarray(w) ** 2, requires_positive=False),
        )
        s = BoundScaler(bounds=((0.0, 100.0),), transform="sigmoid", warp="sqrt")
        assert float(s.from_latent(jnp.array([0.0]))[0]) == pytest.approx(25.0, rel=1e-3)
        x = jnp.array([9.0])
        assert jnp.allclose(s.from_latent(s.to_latent(x)), x, rtol=1e-4)


class TestJaxSafety:
    @pytest.mark.parametrize("name", TRANSFORM_NAMES)
    def test_transforms_are_jit_and_vmap_safe(self, name):
        s = BoundScaler(bounds=((0.0, 10.0),), transform=name)
        out = eqx.filter_jit(jax.vmap(s.from_latent))(jnp.array([[-3.0], [0.0], [3.0]]))
        assert out.shape == (3, 1) and jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("warp", WARP_NAMES)
    def test_warps_are_jit_safe(self, warp):
        bounds = ((1e-6, 1e2),) if warp != "linear" else ((0.0, 10.0),)
        s = BoundScaler(bounds=bounds, transform="sigmoid", warp=warp)
        assert jnp.isfinite(eqx.filter_jit(s.from_latent)(jnp.array([0.5]))).all()
