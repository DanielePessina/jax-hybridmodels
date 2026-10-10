"""Predictor families and the ``BoundedPredictor`` wrapper around them.

The family tests are parametrized over MLP, KAN and NeuralNPolynomial, so a
contract (shape, jit, restart behaviour, use as a bounded inner) is stated
once and checked for every family. KAN builds its scaffold once per
architecture and that first build is slow, so every KAN here shares one
architecture.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.predictors import (
    BoundedPredictor,
    KANPredictor,
    MLPPredictor,
    NeuralNPolynomial,
    reinitialize_pytree_with_key,
    reinitialize_with_key,
)
from jaxhybridmodels.predictors.kan import _scaffold_parts

IN, OUT = 2, 1


def mlp(key: Array) -> MLPPredictor:
    return MLPPredictor(in_size=IN, out_size=OUT, width_size=4, depth=2, key=key)


def kan(key: Array) -> KANPredictor:
    return KANPredictor(in_size=IN, out_size=OUT, hidden_widths=(3,), grid_size=3, key=key)


def npoly(key: Array) -> NeuralNPolynomial:
    exponents = (0.0, 1.0, 2.0)
    coeff_net = MLPPredictor(
        in_size=IN, out_size=OUT * len(exponents), width_size=4, depth=1, key=key
    )
    return NeuralNPolynomial(coeff_net=coeff_net, exponents=exponents, in_size=IN, out_size=OUT)


FAMILIES: dict[str, Callable[[Array], hm.Predictor]] = {"mlp": mlp, "kan": kan, "npoly": npoly}
family = pytest.mark.parametrize("build", FAMILIES.values(), ids=FAMILIES.keys())


def arrays(module: eqx.Module) -> list[Array]:
    return jtu.tree_leaves(eqx.filter(module, eqx.is_array))


def static_part(module: eqx.Module) -> eqx.Module:
    return eqx.filter(module, eqx.is_array, inverse=True)


@family
def test_forward_shape_jit_and_float_leaves(build):
    predictor = build(jr.key(0))
    x = jnp.array([0.3, -0.2])
    out = predictor(x)
    assert out.shape == (OUT,) and jnp.all(jnp.isfinite(out))
    np.testing.assert_allclose(eqx.filter_jit(lambda p, v: p(v))(predictor, x), out, rtol=1e-5)
    # No integer or key leaves in the trainable tree.
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in arrays(predictor))
    assert not jnp.allclose(build(jr.key(1))(x), out)


@family
def test_save_and_load_are_bit_exact(build, tmp_path):
    """R-A5: every family round-trips through save_predictors into a fresh template."""
    predictor = build(jr.key(0))
    path = tmp_path / "predictor.eqx"
    hm.save_predictors(path, predictor)
    loaded = hm.load_predictors(path, build(jr.key(1)))
    for a, b in zip(arrays(predictor), arrays(loaded), strict=True):
        np.testing.assert_array_equal(a, b)
    x = jnp.array([0.3, -0.2])
    np.testing.assert_array_equal(loaded(x), predictor(x))


@family
def test_restart_keeps_the_architecture_and_redraws_the_weights(build):
    predictor = build(jr.key(0))
    fresh = predictor.initialized_with_key(jr.key(11))
    # Same architecture: same type and leaf shapes. (KAN derives a new
    # static seed from the key, so the static fields need not match.)
    assert type(fresh) is type(predictor)
    assert [a.shape for a in arrays(fresh)] == [a.shape for a in arrays(predictor)]
    assert any(
        not jnp.array_equal(a, b) for a, b in zip(arrays(predictor), arrays(fresh), strict=True)
    )
    again = predictor.initialized_with_key(jr.key(11))
    assert all(jnp.array_equal(a, b) for a, b in zip(arrays(fresh), arrays(again), strict=True))


@family
def test_drops_in_as_a_bounded_inner(build):
    bounded = BoundedPredictor(
        inner=build(jr.key(0)),
        input_keys=("a", "b"),
        input_bounds=((0.0, 1.0), (-1.0, 1.0)),
        output_bounds=(0.0, 10.0),
    )
    out = bounded({"a": 0.3, "b": 0.5})
    assert out.shape == (1,) and 0.0 <= float(out[0]) <= 10.0
    grads = eqx.filter_jit(eqx.filter_grad(lambda p: p({"a": 0.3, "b": 0.5})[0]))(bounded)
    assert all(jnp.all(jnp.isfinite(g)) for g in arrays(grads.inner))


@pytest.mark.parametrize("build", [mlp, kan], ids=["mlp", "kan"])
def test_zero_final_head_zeroes_only_the_readout(build):
    predictor = build(jr.key(0))
    zeroed = predictor.with_zero_final_head()
    for x in (jnp.zeros(IN), jnp.array([0.5, -1.7])):
        assert jnp.array_equal(zeroed(x), jnp.zeros(OUT))
    assert eqx.tree_equal(static_part(zeroed), static_part(predictor))
    before, after = arrays(predictor), arrays(zeroed)
    changed = [not jnp.array_equal(a, b) for a, b in zip(before, after, strict=True)]
    # Some leaves were zeroed, the rest (the hidden layers) are untouched,
    # and the input module was not mutated.
    assert any(changed) and not all(changed)
    assert all(
        jnp.array_equal(a, jnp.zeros_like(a)) for a, c in zip(after, changed, strict=True) if c
    )
    assert not any(
        jnp.array_equal(a, jnp.zeros_like(a)) for a, c in zip(before, changed, strict=True) if c
    )


def test_mlp_activation_and_depth_zero():
    tanh = mlp(jr.key(0))
    relu = MLPPredictor(
        in_size=IN, out_size=OUT, width_size=4, depth=2, activation_name="relu", key=jr.key(0)
    )
    x = jnp.array([1.0, -1.0])
    assert not jnp.allclose(tanh(x), relu(x))
    linear = MLPPredictor(in_size=IN, out_size=OUT, width_size=4, depth=0, key=jr.key(0))
    assert jnp.array_equal(linear.with_zero_final_head()(x), jnp.zeros(OUT))


class FixedCoefficients(hm.Predictor):
    """Returns the same coefficients for any input, isolating the polynomial."""

    coefficients: Array

    def __call__(self, x: Array) -> Array:
        return self.coefficients


def test_npoly_evaluates_its_polynomial():
    exponents = (0.0, 1.0, 2.0)
    coefficients = jnp.arange(6.0)
    poly = NeuralNPolynomial(
        coeff_net=FixedCoefficients(coefficients), exponents=exponents, in_size=3, out_size=2
    )
    x = jnp.array([0.5, 1.5, 2.0])
    expected = coefficients.reshape(2, 3) @ (jnp.sum(x) ** jnp.asarray(exponents))
    np.testing.assert_allclose(poly(x), expected, rtol=1e-6)
    # 0 ** 0 at a zero basis must stay finite.
    assert jnp.all(jnp.isfinite(poly(jnp.zeros(3))))


@pytest.mark.parametrize(
    "build,match",
    [
        (
            lambda: MLPPredictor(
                in_size=1, out_size=1, width_size=2, depth=1, activation_name="nope", key=jr.key(0)
            ),
            "not supported",
        ),
        (
            lambda: KANPredictor(
                in_size=IN,
                out_size=OUT,
                hidden_widths=(3,),
                grid_size=3,
                basis="nope",
                key=jr.key(0),
            ),
            "not supported",
        ),
        (
            lambda: NeuralNPolynomial(
                coeff_net=MLPPredictor(in_size=3, out_size=5, width_size=2, depth=1, key=jr.key(0)),
                exponents=(0.0, 1.0, 2.0),
                in_size=3,
                out_size=2,
            ),
            "out_size",
        ),
        (
            lambda: NeuralNPolynomial(
                coeff_net=mlp(jr.key(0)), exponents=(-1.0,), in_size=IN, out_size=OUT
            ),
            "exponents",
        ),
        (
            lambda: NeuralNPolynomial(
                coeff_net=mlp(jr.key(0)), exponents=(0.5,), in_size=IN, out_size=OUT
            ),
            "exponents",
        ),
        (lambda: BoundedPredictor(inner=mlp(jr.key(0)), input_keys=("a",)), "input dimension"),
        (
            lambda: BoundedPredictor(inner=mlp(jr.key(0)), output_bounds=((0, 1), (0, 1))),
            "output dimension",
        ),
        (
            lambda: BoundedPredictor(
                inner=mlp(jr.key(0)), input_keys=("a",), input_bounds=((0, 1), (0, 1))
            ),
            "input_keys length",
        ),
        (
            lambda: BoundedPredictor(inner=mlp(jr.key(0)), input_bounds=(), output_bounds=(0, 1)),
            "at least one input",
        ),
        (
            lambda: BoundedPredictor(
                inner=mlp(jr.key(0)),
                in_scaler=hm.BoundScaler(((0, 1), (0, 1))),
                input_bounds=(0, 1),
            ),
            "not both",
        ),
    ],
)
def test_construction_errors(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# ---------------------------------------------------------------------------
# BoundedPredictor inputs
# ---------------------------------------------------------------------------


class Identity(hm.Predictor):
    """Latent in, latent out, so the bounded output equals the input."""

    def __call__(self, x: Array) -> Array:
        return x


def identity_bounded(keys: tuple[str, ...] | None = ("a", "b")) -> BoundedPredictor:
    return BoundedPredictor(
        inner=Identity(),
        input_keys=keys,
        input_bounds=((0.0, 1.0),) * 2,
        output_bounds=((0.0, 1.0),) * 2,
    )


def test_bounded_predictor_reads_named_or_positional_inputs():
    bp = identity_bounded()
    mapping = {"a": 0.4, "b": 0.7, "unused": 99.0}
    np.testing.assert_allclose(bp(mapping), [0.4, 0.7], atol=1e-5)
    np.testing.assert_allclose(bp(jnp.array([0.4, 0.7])), [0.4, 0.7], atol=1e-5)
    np.testing.assert_allclose(identity_bounded(("b", "a"))(mapping), [0.7, 0.4], atol=1e-5)
    jitted = eqx.filter_jit(lambda p, v: p(v))
    np.testing.assert_allclose(jitted(bp, {"a": 0.4, "b": 0.7}), [0.4, 0.7], atol=1e-5)
    np.testing.assert_allclose(jitted(bp, jnp.array([0.4, 0.7])), [0.4, 0.7], atol=1e-5)
    assert identity_bounded(None).input_keys == ("x1", "x2")


@pytest.mark.parametrize(
    "inputs,error",
    [
        ({"a": 0.4}, KeyError),
        (jnp.array([0.4, 0.7, 0.1]), eqx.EquinoxTracetimeError),
        (jnp.array([[0.4, 0.7]]), eqx.EquinoxTracetimeError),
    ],
    ids=["missing-key", "wrong-length", "wrong-rank"],
)
def test_bounded_predictor_rejects_malformed_inputs(inputs, error):
    with pytest.raises(error):
        identity_bounded()(inputs)


def test_inner_output_size_is_checked_at_call_when_unknown():
    bp = BoundedPredictor(inner=Identity(), input_keys=("x",), output_bounds=((0, 1), (0, 1)))
    with pytest.raises(eqx.EquinoxTracetimeError):
        bp({"x": 0.5})


def test_open_input_bounds_pass_inputs_through_unscaled():
    bp = BoundedPredictor(
        inner=Identity(),
        input_keys=("a", "b"),
        input_bounds=((0.0, None), (None, None)),
        output_bounds=((None, None),) * 2,
    )
    np.testing.assert_allclose(bp({"a": -3.0, "b": 4.0}), [-3.0, 4.0])


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------


class Linear(hm.Predictor):
    """No ``initialized_with_key``, so restarts take the generic leaf branch."""

    weights: Array
    counter: Array

    def __call__(self, x: Array) -> Array:
        return x @ self.weights


def test_generic_restart_redraws_floats_and_keeps_integers():
    p = Linear(weights=jnp.ones((3, 2)), counter=jnp.asarray(5, dtype=jnp.int32))
    a = reinitialize_with_key(p, jr.key(0))
    assert isinstance(a, Linear)
    assert not jnp.allclose(a.weights, p.weights)
    assert jnp.array_equal(a.counter, p.counter) and a.counter.dtype == jnp.int32
    assert jnp.array_equal(reinitialize_with_key(p, jr.key(0)).weights, a.weights)


def test_pytree_restart_keeps_scalers_and_constants():
    def bounded(key):
        return BoundedPredictor(
            inner=mlp(key),
            input_keys=("a", "b"),
            input_bounds=((0, 10), (0, 5)),
            output_bounds=(-2, 2),
        )

    predictors = {
        "twins": (bounded(jr.key(0)), bounded(jr.key(0))),
        "fixed": hm.Constant((1.0, [2.0, 3.0])),
    }
    fresh = reinitialize_pytree_with_key(predictors, jr.key(2))
    a, b = fresh["twins"]
    # Identical siblings get independent subkeys.
    assert not jnp.allclose(a.inner.mlp.layers[0].weight, b.inner.mlp.layers[0].weight)
    # Scalers hold geometry, not state. A temperature redrawn from N(0, 1)
    # can go negative and invert the squash.
    for leaf in (a, b):
        assert float(leaf.in_scaler.temperature) == 1.0
        assert float(leaf.out_scaler.temperature) == 1.0
    # The restart uses the MLP's own init, not leaf-level N(0, 1).
    assert float(jnp.std(a.inner.mlp.layers[0].weight)) < 0.7
    assert eqx.tree_equal(fresh["fixed"], predictors["fixed"])


# ---------------------------------------------------------------------------
# KAN scaffold cache
# ---------------------------------------------------------------------------


def test_kan_scaffold_warmed_inside_jit_survives_a_second_trace():
    """The cached scaffold must hold concrete arrays, not tracers.

    ``_scaffold_parts`` memoises the jaxkan scaffold per architecture. The
    first call usually happens inside ``jit``; built carelessly, the cache
    then stores that trace's tracers and the next trace dies with
    ``UnexpectedTracerError``.
    """
    _scaffold_parts.cache_clear()
    predictor = kan(jr.key(0))
    x = jnp.array([0.3, -0.2])
    # Two distinct jitted callables, so two traces.
    first = eqx.filter_jit(lambda p, v: p(v))(predictor, x)
    second = eqx.filter_jit(lambda p, v: 2.0 * p(v))(predictor, x)
    np.testing.assert_allclose(second, 2.0 * first, rtol=1e-6)
    _, rest = _scaffold_parts(IN, OUT, (3,), 3, predictor.basis, predictor.seed)
    leaves = [leaf for leaf in jtu.tree_leaves(rest) if hasattr(leaf, "shape")]
    assert leaves and not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
