"""Tests for the ready-made time profiles (exogenous, time-varying inputs).

``hybridmodels.profiles`` provides factory callables ``t -> Array`` for
quantities that change over time inside the user's vector field — a
reactor temperature ramp, a feed step. The profile *parameters* travel as
ordinary per-experiment covariates; the factory is evaluated at the
solver's continuous ``t``.

Each test pins one property of the contract:

- pure-JAX safety: every profile composes with ``jit``, ``vmap`` and
  ``grad`` (the vector field calls it at arbitrary solver times);
- edge semantics: flat edges, exact values at knots, step direction;
- validation: ``ramp_profile`` refuses a degenerate ``t1 <= t0``;
  ``piecewise_linear_profile`` refuses non-increasing knots;
- the library usage pattern: per-experiment parameters read from
  ``covariates`` inside a vmapped vector field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from hybridmodels.profiles import (
    constant_profile,
    piecewise_linear_profile,
    ramp_profile,
    step_profile,
)


def test_constant_profile_returns_value_everywhere():
    p = constant_profile(3.5)
    ts = jnp.linspace(-2.0, 9.0, 7)
    assert jnp.allclose(p(ts), jnp.full(ts.shape, 3.5))


def test_step_profile_before_at_and_after_jump():
    p = step_profile(before=1.0, after=4.0, jump_at=2.0)
    assert float(p(jnp.asarray(1.9))) == pytest.approx(1.0)
    assert float(p(jnp.asarray(2.0))) == pytest.approx(4.0)  # t >= jump_at
    assert float(p(jnp.asarray(5.0))) == pytest.approx(4.0)


def test_ramp_profile_has_two_flat_edges_and_a_linear_middle():
    p = ramp_profile(t0=1.0, t1=3.0, v0=10.0, v1=30.0)
    # Flat before t0 and after t1, exactly the edge values.
    assert float(p(jnp.asarray(0.0))) == pytest.approx(10.0)
    assert float(p(jnp.asarray(1.0))) == pytest.approx(10.0)
    assert float(p(jnp.asarray(3.0))) == pytest.approx(30.0)
    assert float(p(jnp.asarray(7.0))) == pytest.approx(30.0)
    # Linear in between.
    assert float(p(jnp.asarray(2.0))) == pytest.approx(20.0)


def test_ramp_profile_downward():
    p = ramp_profile(t0=1.0, t1=3.0, v0=30.0, v1=10.0)
    assert float(p(jnp.asarray(0.0))) == pytest.approx(30.0)
    assert float(p(jnp.asarray(2.0))) == pytest.approx(20.0)
    assert float(p(jnp.asarray(5.0))) == pytest.approx(10.0)


def test_ramp_profile_rejects_degenerate_times():
    with pytest.raises(ValueError, match="t1"):
        ramp_profile(t0=3.0, t1=3.0, v0=0.0, v1=1.0)


def test_piecewise_linear_profile_interpolates_and_extends():
    p = piecewise_linear_profile(((0.0, 0.0), (1.0, 2.0), (3.0, 6.0)))
    assert float(p(jnp.asarray(-1.0))) == pytest.approx(0.0)  # first value extends
    assert float(p(jnp.asarray(0.5))) == pytest.approx(1.0)
    assert float(p(jnp.asarray(2.0))) == pytest.approx(4.0)
    assert float(p(jnp.asarray(4.0))) == pytest.approx(6.0)  # last value extends


def test_piecewise_linear_profile_validates_knots():
    with pytest.raises(ValueError, match="at least two"):
        piecewise_linear_profile(((0.0, 1.0),))
    with pytest.raises(ValueError, match="strictly increasing"):
        piecewise_linear_profile(((0.0, 1.0), (1.0, 2.0), (0.5, 3.0)))


def test_piecewise_linear_profile_skips_validation_for_traced_knots():
    """Mirrors ramp_profile: traced knot times (a factory built inside
    jit/vmap) skip the host-side increasing check instead of raising."""

    def build(t_offset):
        return piecewise_linear_profile(
            ((t_offset, 0.0), (1.0 + t_offset, 1.0), (2.0 + t_offset, 2.0))
        )

    jitted = jax.jit(lambda t: build(t)(jnp.asarray(0.0)))
    assert jnp.isfinite(jitted(jnp.asarray(0.5)))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: constant_profile(1.5),
        lambda: step_profile(0.0, 1.0, jump_at=2.0),
        lambda: ramp_profile(t0=1.0, t1=3.0, v0=10.0, v1=30.0),
        lambda: piecewise_linear_profile(((0.0, 0.0), (1.0, 2.0), (3.0, 6.0))),
    ],
)
def test_profiles_are_jit_and_vmap_safe(factory):
    """The solver calls the profile at arbitrary traced ``t``: it must survive
    ``jit`` and ``vmap`` and stay differentiable where it is smooth."""
    p = factory()
    ts = jnp.linspace(0.0, 5.0, 11)

    def stepwise(t):
        return jnp.sum(p(t) ** 2)

    jitted = jax.jit(stepwise)
    assert jnp.allclose(jitted(ts), stepwise(ts))
    v = jax.vmap(lambda t: p(t))(ts)
    assert v.shape == ts.shape
    grad = jax.grad(lambda t: jnp.sum(p(jnp.stack([t]))))(jnp.asarray(2.0))
    assert jnp.isfinite(grad)


def test_parameters_ride_as_covariates_in_a_vmapped_field():
    """The library usage pattern: per-experiment ramp parameters read from
    covariates inside a vector field that ``simulate_bucket``-style vmap
    walks. Each experiment ramps on its own schedule."""
    cov_lo = jnp.asarray([10.0, 20.0])
    cov_hi = jnp.asarray([30.0, 40.0])
    cov_t0 = jnp.asarray([0.0, 1.0])
    cov_t1 = jnp.asarray([4.0, 5.0])

    def per_experiment(t, lo, hi, t0, t1):
        return ramp_profile(t0=t0, t1=t1, v0=lo, v1=hi)(t)

    ts = jnp.linspace(0.0, 6.0, 7)
    field = jax.vmap(
        lambda t: jnp.stack(
            [per_experiment(t, lo, hi, t0, t1) for lo, hi, t0, t1 in
             zip(cov_lo, cov_hi, cov_t0, cov_t1, strict=True)]
        )
    )(ts)
    assert field.shape == (7, 2)
    # Experiment 0 ramps 10 -> 30 over [0, 4]; at t=2 it is at 20.
    assert float(field[2, 0]) == pytest.approx(20.0)
    # Experiment 1 ramps 20 -> 40 over [1, 5]; at t=0 it is still flat at 20.
    assert float(field[0, 1]) == pytest.approx(20.0)


def test_profiles_evaluate_at_scalar_float_times():
    """diffrax calls the vector field with a traced scalar ``t``, but the
    same callable must also work host-side with a Python float."""
    p = ramp_profile(t0=1.0, t1=3.0, v0=10.0, v1=30.0)
    assert float(p(2.0)) == pytest.approx(20.0)
    s = step_profile(0.0, 1.0, jump_at=2.0)
    assert float(s(1.9)) == pytest.approx(0.0)
    c = constant_profile(4.0)
    assert float(c(123.0)) == pytest.approx(4.0)


def test_output_is_float32_array_not_python_float():
    """The closure returns a JAX array (not a Python float), so it flows
    through the traced vector field without conversion."""
    p = constant_profile(1.0)
    out = p(jnp.asarray(0.5))
    assert isinstance(out, Array)
    assert out.dtype == jnp.float32