# ruff: noqa: F722

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from hybridmodels.predictors import MLPPredictor


def _mlp(
    *,
    in_size: int = 3,
    out_size: int = 2,
    width_size: int = 8,
    depth: int = 2,
    activation_name: str = "tanh",
    seed: int = 0,
) -> MLPPredictor:
    return MLPPredictor(
        in_size=in_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
        activation_name=activation_name,
        key=jr.PRNGKey(seed),
    )


def _array_leaves(predictor: eqx.Module) -> list:
    arrays, _ = eqx.partition(predictor, eqx.is_array)
    return jtu.tree_leaves(arrays)


class TestForwardShape:
    def test_shape_matches_out_size(self):
        predictor = _mlp(in_size=3, out_size=2)
        out = predictor(jnp.zeros((3,)))
        assert out.shape == (2,)

    def test_scalar_in_out(self):
        predictor = _mlp(in_size=1, out_size=1)
        out = predictor(jnp.zeros((1,)))
        assert out.shape == (1,)

    def test_jits(self):
        predictor = _mlp(in_size=4, out_size=3)
        out = eqx.filter_jit(lambda p, x: p(x))(predictor, jnp.ones((4,)))
        assert out.shape == (3,)


class TestStaticFields:
    def test_hyperparams_are_static(self):
        predictor = _mlp(in_size=3, out_size=2, width_size=7, depth=2)
        assert predictor.in_size == 3
        assert predictor.out_size == 2
        assert predictor.width_size == 7
        assert predictor.depth == 2
        assert predictor.activation_name == "tanh"

        for leaf in _array_leaves(predictor):
            assert jnp.issubdtype(leaf.dtype, jnp.floating)

    def test_static_fields_not_in_dynamic_leaves(self):
        predictor = _mlp()
        leaves = [
            leaf for leaf in jtu.tree_leaves(predictor) if not eqx.is_array(leaf)
        ]
        for hyperparam in (
            predictor.in_size,
            predictor.out_size,
            predictor.width_size,
            predictor.depth,
            predictor.activation_name,
        ):
            assert hyperparam not in leaves


class TestKeyDeterminism:
    def test_same_key_gives_identical_weights(self):
        a = _mlp(seed=0)
        b = _mlp(seed=0)
        leaves_a = _array_leaves(a)
        leaves_b = _array_leaves(b)
        assert len(leaves_a) == len(leaves_b)
        for la, lb in zip(leaves_a, leaves_b, strict=True):
            assert jnp.array_equal(la, lb)

    def test_different_keys_diverge(self):
        a = _mlp(seed=0)
        b = _mlp(seed=1)
        leaves_a = _array_leaves(a)
        leaves_b = _array_leaves(b)
        assert any(
            not jnp.array_equal(la, lb)
            for la, lb in zip(leaves_a, leaves_b, strict=True)
        )

        x = jnp.array([0.5, -0.2, 0.1])
        out_a = a(x)
        out_b = b(x)
        assert not jnp.allclose(out_a, out_b)


class TestInitializedWithKey:
    def test_preserves_static_fields(self):
        predictor = _mlp(in_size=3, out_size=2, width_size=7, depth=2)
        fresh = predictor.initialized_with_key(jr.PRNGKey(11))
        assert fresh.in_size == predictor.in_size
        assert fresh.out_size == predictor.out_size
        assert fresh.width_size == predictor.width_size
        assert fresh.depth == predictor.depth
        assert fresh.activation_name == predictor.activation_name

    def test_changes_inexact_leaves(self):
        predictor = _mlp(seed=0)
        fresh = predictor.initialized_with_key(jr.PRNGKey(11))
        leaves_old = _array_leaves(predictor)
        leaves_new = _array_leaves(fresh)
        assert len(leaves_old) == len(leaves_new)
        assert any(
            not jnp.array_equal(lo, ln)
            for lo, ln in zip(leaves_old, leaves_new, strict=True)
        )

    def test_deterministic_for_same_key(self):
        predictor = _mlp()
        a = predictor.initialized_with_key(jr.PRNGKey(7))
        b = predictor.initialized_with_key(jr.PRNGKey(7))
        for la, lb in zip(_array_leaves(a), _array_leaves(b), strict=True):
            assert jnp.array_equal(la, lb)


class TestActivation:
    def test_supports_relu(self):
        predictor = _mlp(activation_name="relu")
        assert predictor.activation_name == "relu"
        out = predictor(jnp.zeros((3,)))
        assert out.shape == (2,)

    def test_unsupported_activation_raises(self):
        with pytest.raises(ValueError):
            _mlp(activation_name="not_a_real_activation")

    def test_activation_changes_output(self):
        a = _mlp(activation_name="tanh", seed=0)
        b = _mlp(activation_name="relu", seed=0)
        x = jnp.array([1.0, -1.0, 0.5])
        out_a = a(x)
        out_b = b(x)
        assert not jnp.allclose(out_a, out_b)


def test_top_level_export():
    import hybridmodels

    assert hybridmodels.MLPPredictor is MLPPredictor
