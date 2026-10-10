"""Bounds: ``BoundScaler``, ``Parameter``, and the transform and warp registries.

The scaler is what keeps every physical value inside its box while the
optimiser works on an unbounded latent. One parametrized sweep covers every
transform against every kind of box; the targeted tests after it pin the
properties a sweep cannot state (gradient decay, the knee, the runtime
guard on positive-only warps). Claims are behavioural: round trips,
orderings and gradient signs, never a retyped formula.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import jaxhybridmodels as hm
from jaxhybridmodels.transforms import (
    BOUND_TRANSFORMS,
    WARPS,
    BoundTransform,
    Warp,
    register_bound_transform,
    register_warp,
)

TRANSFORMS = sorted(BOUND_TRANSFORMS)
INF = math.inf

# (id, bounds pair, warp, physical values inside, physical values outside)
BOXES = [
    ("finite", (0.0, 10.0), "linear", [0.5, 3.5, 9.9], [-3.0, 13.0]),
    ("log", (1e-6, 1e2), "log", [1e-5, 1e-2, 50.0], [200.0]),
    ("log10", (1e-6, 1e2), "log10", [1e-5, 1e-2, 50.0], [200.0]),
    ("lower", (0.0, INF), "linear", [1e-3, 0.5, 50.0], [-2.0]),
    ("upper", (-INF, 1.0), "linear", [-50.0, 0.0, 0.99], [3.0]),
    ("open", (-INF, INF), "linear", [-50.0, 0.0, 50.0], []),
]


@pytest.fixture
def restore_registries():
    transforms, warps = dict(BOUND_TRANSFORMS), dict(WARPS)
    yield
    BOUND_TRANSFORMS.clear()
    BOUND_TRANSFORMS.update(transforms)
    WARPS.clear()
    WARPS.update(warps)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("box", BOXES, ids=[b[0] for b in BOXES])
def test_scaler_properties(transform, box):
    _, (low, high), warp, inside, outside = box
    scaler = hm.BoundScaler(((low, high),), transform=transform, warp=warp)

    # Round trip is the identity inside the box.
    x = jnp.asarray(inside)
    np.testing.assert_allclose(scaler.from_latent(scaler.to_latent(x)), x, rtol=1e-4)

    # Any finite latent lands inside the box.
    out = scaler.from_latent(jnp.array([-1e6, -20.0, 0.0, 20.0, 1e6]))
    assert jnp.all(jnp.isfinite(out))
    assert float(out.min()) >= low * (1 - 1e-5) and float(out.max()) <= high * (1 + 1e-5)

    # Outside the box the inverse keeps a finite, non-zero gradient, so a
    # state-derived input that strays does not cut the adjoint.
    if outside:
        grads = jax.vmap(jax.grad(lambda v: scaler.to_latent(v)[0]))(jnp.asarray(outside))
        assert jnp.all(jnp.isfinite(grads)) and jnp.all(grads != 0.0)

    # Zero latent is the midpoint of the warped box.
    if math.isfinite(low) and math.isfinite(high):
        lo, hi = WARPS[warp].forward(low), WARPS[warp].forward(high)
        midpoint = float(WARPS[warp].inverse((lo + hi) / 2))
        np.testing.assert_allclose(scaler.from_latent(jnp.array([0.0])), [midpoint], rtol=1e-4)


def test_one_component_scaler_acts_elementwise_on_any_shape():
    scaler = hm.BoundScaler(((0.0, 10.0),))
    z = jnp.linspace(-3.0, 3.0, 6).reshape(2, 3)
    assert scaler.from_latent(z).shape == (2, 3)
    assert scaler.to_latent(jnp.array([-1e4, 5.0, 1e4])).shape == (3,)
    assert jnp.all(jnp.isfinite(scaler.to_latent(jnp.array([-1e4, 1e4]))))
    # A scalar query comes back rank 1 for every bound kind.
    for pair in ((0.0, None), (None, 1.0), (None, None)):
        open_scaler = hm.BoundScaler((pair,))
        np.testing.assert_allclose(
            open_scaler.from_latent(open_scaler.to_latent(jnp.asarray(0.2))), [0.2]
        )


def test_vector_scaler_maps_each_component_against_its_own_box():
    scaler = hm.BoundScaler(((0.0, 10.0), (-1.0, 1.0), (0.0, None)))
    x = jnp.array([3.5, 0.25, 7.0])
    np.testing.assert_allclose(scaler.from_latent(scaler.to_latent(x)), x, rtol=1e-5)
    out = scaler.from_latent(jnp.array([100.0, -100.0, -100.0]))
    assert 0.0 <= out[0] <= 10.0 and -1.0 <= out[1] <= 1.0 and out[2] >= 0.0
    # Network inputs: finite boxes go to latent, open ones pass through.
    np.testing.assert_allclose(scaler.scale_inputs(x)[2], 7.0)
    np.testing.assert_allclose(scaler.scale_inputs(x)[:2], scaler.to_latent(x)[:2])


def test_temperature_rescales_the_latent_but_not_the_physical_value():
    hot = hm.BoundScaler(((0.0, 10.0),), temperature=2.0)
    cold = hm.BoundScaler(((0.0, 10.0),))
    x = jnp.array([7.0])
    assert not jnp.allclose(hot.to_latent(x), cold.to_latent(x))
    np.testing.assert_allclose(hot.from_latent(hot.to_latent(x)), x, rtol=1e-5)
    # (1 + tanh z) / 2 is sigmoid(2z), which is why there is no "tanh" entry.
    assert "tanh" not in BOUND_TRANSFORMS
    half = hm.BoundScaler(((0.0, 10.0),), temperature=0.5)
    np.testing.assert_allclose(half.from_latent(jnp.array([0.7])), 5.0 * (1 + jnp.tanh(0.7)))


def test_polynomial_tails_keep_gradient_where_sigmoid_dies():
    def tail_gradient(name, z):
        scaler = hm.BoundScaler(((0.0, 10.0),), transform=name)
        return float(jax.grad(lambda v: scaler.from_latent(v)[0])(jnp.array([z]))[0])

    assert tail_gradient("sigmoid", 25.0) == 0.0
    assert tail_gradient("algebraic", 25.0) > 0.0 and tail_gradient("softsign", 25.0) > 0.0
    g = {name: tail_gradient(name, 12.0) for name in ("sigmoid", "algebraic", "softsign")}
    assert g["sigmoid"] < g["algebraic"] < g["softsign"]


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_the_knee_marks_the_outer_five_percent(transform):
    scaler = hm.BoundScaler(((0.0, 10.0),), transform=transform)
    knee = scaler.z_knee
    assert float(scaler.from_latent(jnp.array([knee]))[0]) == pytest.approx(9.5, abs=0.02)
    assert float(scaler.saturation(jnp.array([knee]))) == 0.0
    assert float(scaler.saturation(jnp.array([2 * knee + 1]))) > 0.0
    assert hm.BoundScaler(((0.0, 10.0),), transform=transform, z_knee=1.0).z_knee == 1.0


def test_saturation_gradient_survives_where_the_physical_gradient_underflows():
    scaler = hm.BoundScaler(((0.0, 10.0),))
    z = jnp.array([40.0])
    assert float(jax.grad(lambda v: scaler.saturation(v))(z)[0]) > 0.0
    assert float(jax.grad(lambda v: scaler.from_latent(v)[0])(z)[0]) == 0.0
    assert float(scaler.saturation(jnp.array([0.0, 1.0, -2.0]))) == 0.0


def test_input_violation_is_a_width_normalised_hinge():
    scaler = hm.BoundScaler(((0.0, 10.0), (-1.0, 1.0)))
    assert float(scaler.input_violation(jnp.array([5.0, 0.0]))) == 0.0
    near = scaler.input_violation(jnp.array([11.0, 0.0]))
    far = scaler.input_violation(jnp.array([20.0, 0.0]))
    assert 0.0 < float(near) < float(far)
    assert float(jax.grad(lambda x: scaler.input_violation(x))(jnp.array([13.0, 0.0]))[0]) > 0.0
    narrow = hm.BoundScaler(((0.0, 1.0),)).input_violation(jnp.array([1.5]))
    wide = hm.BoundScaler(((0.0, 1000.0),)).input_violation(jnp.array([1500.0]))
    np.testing.assert_allclose(narrow, wide)
    # A one-sided box has no width, so the overshoot is not normalised.
    assert float(hm.BoundScaler(((0.0, None),)).input_violation(jnp.array([-2.0]))) == 4.0


def test_positive_warp_rejects_nonpositive_runtime_inputs():
    scaler = hm.BoundScaler(((1e-3, 1.0),), warp="log10")
    with pytest.raises(ValueError, match="positive"):
        scaler.to_latent(jnp.array([0.0]))
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive"):
        eqx.filter_jit(scaler.to_latent)(jnp.array([0.0]))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(bounds=((0.0, 1.0),), transform="tanh"), "algebraic"),
        (dict(bounds=((0.0, 1.0),), warp="nope"), "linear"),
        (dict(bounds=((math.nan, 1.0),)), "NaN"),
        (dict(bounds=((10.0, 0.0),)), "low < high"),
        (dict(bounds=((5.0, 5.0),)), "low < high"),
        (dict(bounds=((0.0, 10.0),), warp="log10"), "undefined at or below zero"),
        (dict(bounds=((1.0, None),), warp="log10"), "Open bounds"),
        (dict(bounds=((0.0, 1.0),), temperature=0.0), "temperature"),
        (dict(bounds=((0.0, 1.0),), temperature=math.nan), "temperature"),
        (dict(bounds=((0.0, 1.0),), temperature=jnp.array([1.0, 2.0])), "temperature"),
        (dict(bounds=((0.0, 1.0),), logit_eps=0.5), "logit_eps"),
        (dict(bounds=((0.0, 1.0),), logit_eps=0.0), "logit_eps"),
        (dict(bounds=((0.0, 1.0),), z_knee=-1.0), "z_knee"),
    ],
)
def test_scaler_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hm.BoundScaler(**kwargs)


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bounds,value",
    [
        (None, -2.0),
        ((0.0, None), 0.1),
        ((None, 1.0), 0.9),
        ((0.0, 1.0), 1e-4),
        ((-INF, INF), 2.0),
        ((1e-6, 1e2), 1e-3),
    ],
)
def test_parameter_is_declared_in_physical_units_and_differentiable(bounds, value):
    warp = "log10" if bounds == (1e-6, 1e2) else "linear"
    parameter = hm.Parameter(value, bounds=bounds, warp=warp)
    np.testing.assert_allclose(parameter.value, value, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(eqx.filter_jit(lambda p: p.value)(parameter), value, rtol=1e-5)
    # Only the latent trains; the scaler is geometry, not state.
    mask = hm.trainable_mask(parameter)
    assert mask.latent is True and mask.scaler.temperature is False
    gradient = eqx.filter_grad(lambda p: p.value)(parameter)
    assert jnp.isfinite(gradient.latent).all() and float(jnp.abs(gradient.latent).sum()) > 0


def test_array_parameter_takes_one_bound_per_element():
    parameter = hm.Parameter(
        [[0.2, 0.4], [-2.0, 0.8]], bounds=[(0.0, None), (None, 1.0), (None, None), (0.0, 1.0)]
    )
    assert parameter.value.shape == (2, 2)
    np.testing.assert_allclose(parameter.value, [[0.2, 0.4], [-2.0, 0.8]], rtol=1e-5)
    assert jnp.isfinite(eqx.filter_grad(lambda p: p.value.sum())(parameter).latent).all()
    unbounded = hm.Parameter([-100.0, 100.0])
    assert float(unbounded.scaler.saturation(unbounded.latent)) == 0.0


def test_parameter_restart_draws_a_new_latent_inside_the_same_box():
    rate = hm.Parameter(0.3, bounds=(0.0, 1.0))
    fresh = rate.initialized_with_key(jr.key(0))
    assert float(fresh.latent[0]) != float(rate.latent[0])
    assert fresh.scaler.bounds == rate.scaler.bounds
    assert 0.0 < float(fresh.value) < 1.0


@pytest.mark.parametrize(
    "value,bounds,match",
    [
        (math.nan, None, "finite"),
        ([], None, "nonempty"),
        (0.0, (0.0, 1.0), "strictly inside"),
        (2.0, (None, 1.0), "strictly inside"),
        ([0.1, 0.2], [(0.0, 1.0)] * 3, "Expected 2 bound pairs"),
    ],
)
def test_parameter_validation(value, bounds, match):
    with pytest.raises(ValueError, match=match):
        hm.Parameter(value, bounds=bounds)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def test_a_registered_transform_and_warp_work_end_to_end(restore_registries):
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
    # Square root: more resolution toward the low end, and zero is allowed.
    register_warp(
        "sqrt",
        Warp(forward=jnp.sqrt, inverse=lambda w: jnp.asarray(w) ** 2, requires_positive=False),
    )
    parameter = hm.Parameter(9.0, bounds=(0.0, 100.0), transform="arctan", warp="sqrt")
    np.testing.assert_allclose(parameter.value, 9.0, rtol=1e-4)
    scaler = parameter.scaler
    assert float(scaler.from_latent(jnp.array([0.0]))[0]) == pytest.approx(25.0, rel=1e-3)
    knee_value = float(scaler.from_latent(jnp.array([scaler.z_knee]))[0])
    assert knee_value == pytest.approx(9.5**2, rel=1e-2)
