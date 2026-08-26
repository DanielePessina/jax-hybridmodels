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
        leaves = [leaf for leaf in jtu.tree_leaves(predictor) if not eqx.is_array(leaf)]
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
        assert any(not jnp.array_equal(la, lb) for la, lb in zip(leaves_a, leaves_b, strict=True))

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
            not jnp.array_equal(lo, ln) for lo, ln in zip(leaves_old, leaves_new, strict=True)
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


class TestWithZeroFinalHead:
    """Contract for ``with_zero_final_head``: zero readout, identical hidden layers."""

    def test_output_is_zero_for_arbitrary_input(self):
        predictor = _mlp(in_size=3, out_size=2)
        zeroed = predictor.with_zero_final_head()
        # Several inputs spanning the trained domain — output must be exactly zero.
        for x in (jnp.zeros((3,)), jnp.ones((3,)), jnp.array([0.5, -1.7, 2.3])):
            assert jnp.allclose(zeroed(x), jnp.zeros((2,)), atol=0.0)

    def test_final_layer_weight_and_bias_are_zero(self):
        predictor = _mlp(in_size=3, out_size=2, width_size=8, depth=2)
        zeroed = predictor.with_zero_final_head()
        final = zeroed.mlp.layers[-1]
        # eqx.nn.Linear.bias is Optional (use_bias=False leaves it None);
        # this predictor always builds with a bias, and asserting that
        # states the invariant instead of hiding it from the type checker.
        assert final.bias is not None
        assert jnp.array_equal(final.weight, jnp.zeros_like(final.weight))
        assert jnp.array_equal(final.bias, jnp.zeros_like(final.bias))

    def test_hidden_layers_unchanged(self):
        predictor = _mlp(in_size=3, out_size=2, depth=2)
        zeroed = predictor.with_zero_final_head()
        # Hidden layers (everything except the last) keep their LeCun-uniform init.
        for orig_layer, zeroed_layer in zip(
            predictor.mlp.layers[:-1], zeroed.mlp.layers[:-1], strict=True
        ):
            assert jnp.array_equal(orig_layer.weight, zeroed_layer.weight)
            assert jnp.array_equal(orig_layer.bias, zeroed_layer.bias)

    def test_returns_new_instance_input_unchanged(self):
        predictor = _mlp()
        zeroed = predictor.with_zero_final_head()
        assert zeroed is not predictor
        # Original final layer is *not* zero — confirms we didn't mutate.
        original_final = predictor.mlp.layers[-1]
        assert not jnp.array_equal(original_final.weight, jnp.zeros_like(original_final.weight))

    def test_preserves_static_fields(self):
        predictor = _mlp(in_size=4, out_size=3, width_size=7, depth=3, activation_name="relu")
        zeroed = predictor.with_zero_final_head()
        assert zeroed.in_size == 4
        assert zeroed.out_size == 3
        assert zeroed.width_size == 7
        assert zeroed.depth == 3
        assert zeroed.activation_name == "relu"

    def test_works_for_depth_zero(self):
        # depth=0 means a single linear layer (no hidden), so "final" == only layer.
        predictor = _mlp(in_size=3, out_size=2, depth=0)
        zeroed = predictor.with_zero_final_head()
        assert jnp.allclose(zeroed(jnp.array([1.0, 2.0, 3.0])), jnp.zeros((2,)), atol=0.0)


def test_top_level_export():
    import hybridmodels

    assert hybridmodels.MLPPredictor is MLPPredictor
