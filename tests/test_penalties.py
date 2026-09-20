"""Tests for the gradient-safe bound primitives in ``jaxhybridmodels.penalties``.

These pin the properties the rest of the framework relies on, and they
are written as *behavioural* claims (what the gradient does at a bound)
rather than as restatements of the formulas — a test that retypes the
implementation's own algebra proves only that the algebra was copied
twice.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxhybridmodels.penalties import box_violation, clip_ste, soft_logit, softclip


class TestSoftLogit:
    """``soft_logit`` is the guard that replaced ``logit(jnp.clip(...))``."""

    def test_matches_logit_exactly_inside_the_band(self):
        s = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99])
        assert jnp.allclose(soft_logit(s, 1e-3), jax.scipy.special.logit(s), atol=1e-6)

    def test_derivative_matches_logit_inside_the_band(self):
        # The stop_gradient on the clamp is load-bearing; if it were
        # dropped the interior derivative would pick up a spurious term.
        for x in (0.05, 0.3, 0.8):
            got = jax.grad(lambda v: soft_logit(v, 1e-3))(jnp.asarray(x))
            want = 1.0 / (x * (1.0 - x))
            assert jnp.allclose(got, want, rtol=1e-4)

    def test_finite_and_nonzero_gradient_outside_the_band(self):
        for x in (-5.0, -0.01, 1.01, 7.0):
            v = soft_logit(jnp.asarray(x), 1e-3)
            g = jax.grad(lambda u: soft_logit(u, 1e-3))(jnp.asarray(x))
            assert jnp.isfinite(v)
            assert jnp.isfinite(g) and float(g) > 0.0

    def test_hard_clip_baseline_really_does_kill_the_gradient(self):
        # Guards the premise of the whole module: if this ever stops being
        # true, `soft_logit` has no reason to exist.
        hard = lambda x: jax.scipy.special.logit(jnp.clip(x, 1e-3, 1 - 1e-3))  # noqa: E731
        assert float(jax.grad(hard)(jnp.asarray(2.0))) == 0.0

    def test_continuation_slope_is_set_by_eps(self):
        slope_coarse = float(jax.grad(lambda v: soft_logit(v, 1e-2))(jnp.asarray(3.0)))
        slope_fine = float(jax.grad(lambda v: soft_logit(v, 1e-4))(jnp.asarray(3.0)))
        assert slope_fine > slope_coarse

    def test_is_jit_and_vmap_safe(self):
        xs = jnp.array([-1.0, 0.5, 2.0])
        out = eqx.filter_jit(jax.vmap(lambda v: soft_logit(v, 1e-3)))(xs)
        assert out.shape == (3,) and jnp.all(jnp.isfinite(out))


class TestBoxViolation:
    def test_zero_value_and_zero_gradient_inside(self):
        lows, highs = jnp.array([0.0]), jnp.array([10.0])
        x = jnp.array([4.0])
        assert float(box_violation(x, lows, highs)) == 0.0
        # Zero *gradient* matters as much as zero value: the penalty must
        # not perturb the feasible interior it is meant to leave alone.
        g = jax.grad(lambda v: box_violation(v, lows, highs))(x)
        assert float(g[0]) == 0.0

    def test_gradient_points_back_into_the_box(self):
        lows, highs = jnp.array([0.0]), jnp.array([10.0])
        g_above = jax.grad(lambda v: box_violation(v, lows, highs))(jnp.array([13.0]))
        g_below = jax.grad(lambda v: box_violation(v, lows, highs))(jnp.array([-3.0]))
        assert float(g_above[0]) > 0.0  # pushes down
        assert float(g_below[0]) < 0.0  # pushes up

    def test_push_back_grows_without_bound(self):
        # The property a reparameterised bound cannot offer.
        lows, highs = jnp.array([0.0]), jnp.array([1.0])
        grads = [
            float(jax.grad(lambda v: box_violation(v, lows, highs))(jnp.array([x]))[0])
            for x in (2.0, 20.0, 200.0)
        ]
        assert grads[0] < grads[1] < grads[2]

    def test_is_nan_free_without_double_where_guarding(self):
        # max(v, 0) ** 2 is pole-free, which is why the module standardises
        # on the squared hinge rather than abs or sqrt.
        lows, highs = jnp.array([0.0]), jnp.array([1.0])
        for x in (-1.0, 0.0, 1.0, 2.0):
            g = jax.grad(lambda v: box_violation(v, lows, highs))(jnp.array([x]))
            assert jnp.all(jnp.isfinite(g))

    def test_width_normalisation_makes_channels_comparable(self):
        narrow = box_violation(jnp.array([2.0]), jnp.array([0.0]), jnp.array([1.0]))
        wide = box_violation(jnp.array([2000.0]), jnp.array([0.0]), jnp.array([1000.0]))
        assert jnp.allclose(narrow, wide)

    def test_multiple_components_accumulate(self):
        lows, highs = jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])
        one = box_violation(jnp.array([2.0, 0.5]), lows, highs)
        two = box_violation(jnp.array([2.0, 2.0]), lows, highs)
        assert float(two) > float(one)


class TestClipSte:
    def test_forward_is_a_hard_clip(self):
        assert float(clip_ste(jnp.asarray(3.0), 0.0, 1.0)) == 1.0
        assert float(clip_ste(jnp.asarray(-2.0), 0.0, 1.0)) == 0.0
        assert jnp.allclose(clip_ste(jnp.asarray(0.4), 0.0, 1.0), 0.4)

    def test_backward_is_the_identity(self):
        for x in (-2.0, 0.4, 3.0):
            g = jax.grad(lambda v: clip_ste(v, 0.0, 1.0))(jnp.asarray(x))
            assert float(g) == 1.0


class TestSoftclip:
    def test_stays_within_the_box(self):
        xs = jnp.linspace(-10.0, 10.0, 41)
        out = softclip(xs, 0.0, 1.0, beta=20.0)
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0

    def test_gradient_is_nonzero_just_outside(self):
        g = jax.grad(lambda v: softclip(v, 0.0, 1.0, 20.0))(jnp.asarray(1.05))
        assert 0.0 < float(g) < 1.0

    def test_interior_error_is_order_one_over_beta(self):
        # Documents *why* softclip is not used for the logit guard: its
        # interior distortion scales as 1/beta, which on a [0, 1]-normalised
        # coordinate is a visible fraction of the whole box. Measured near
        # an edge, not at the midpoint -- the two shoulders cancel exactly
        # at the centre, so the centre would report zero error for any beta.
        near_edge = jnp.asarray(0.05)
        err_coarse = abs(float(softclip(near_edge, 0.0, 1.0, 20.0)) - 0.05)
        err_fine = abs(float(softclip(near_edge, 0.0, 1.0, 200.0)) - 0.05)
        assert err_coarse > err_fine
        # At beta=20 the distortion is percent-level on a unit box, which
        # is what rules it out as the logit guard.
        assert err_coarse > 1e-3
