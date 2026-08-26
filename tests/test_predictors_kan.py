# ruff: noqa: F722

from __future__ import annotations

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from hybridmodels.predictors import (
    BoundedPredictor,
    BoundScaler,
    KANPredictor,
)
from hybridmodels.predictors.kan import _scaffold_parts


def _kan(
    *,
    in_size: int = 2,
    out_size: int = 1,
    hidden_widths: tuple[int, ...] = (8,),
    grid_size: int = 5,
    basis: str = "spline",
    seed: int = 0,
) -> KANPredictor:
    return KANPredictor(
        in_size=in_size,
        out_size=out_size,
        hidden_widths=hidden_widths,
        grid_size=grid_size,
        basis=basis,
        key=jr.PRNGKey(seed),
    )


def _array_leaves(predictor: eqx.Module) -> list:
    arrays, _ = eqx.partition(predictor, eqx.is_array)
    return jtu.tree_leaves(arrays)


class TestForwardShape:
    def test_shape_matches_out_size(self):
        predictor = _kan(in_size=3, out_size=2)
        out = predictor(jnp.zeros((3,)))
        assert out.shape == (2,)

    def test_scalar_in_out(self):
        predictor = _kan(in_size=1, out_size=1)
        out = predictor(jnp.zeros((1,)))
        assert out.shape == (1,)

    def test_jits(self):
        predictor = _kan(in_size=4, out_size=3)
        out = eqx.filter_jit(lambda p, x: p(x))(predictor, jnp.ones((4,)))
        assert out.shape == (3,)
        assert jnp.all(jnp.isfinite(out))


class TestStaticFields:
    def test_hyperparams_are_static(self):
        predictor = _kan(in_size=3, out_size=2, hidden_widths=(7, 5), grid_size=4)
        assert predictor.in_size == 3
        assert predictor.out_size == 2
        assert predictor.hidden_widths == (7, 5)
        assert predictor.grid_size == 4
        assert predictor.basis == "spline"

        for leaf in _array_leaves(predictor):
            assert jnp.issubdtype(leaf.dtype, jnp.floating), (
                f"Non-float dynamic leaf with dtype {leaf.dtype}; "
                "rng/state leaves must not appear in the trainable tree."
            )


class TestKeyDeterminism:
    def test_same_key_gives_identical_weights(self):
        a = _kan(seed=0)
        b = _kan(seed=0)
        leaves_a = _array_leaves(a)
        leaves_b = _array_leaves(b)
        assert len(leaves_a) == len(leaves_b)
        for la, lb in zip(leaves_a, leaves_b, strict=True):
            assert jnp.array_equal(la, lb)

    def test_different_keys_diverge(self):
        a = _kan(seed=0)
        b = _kan(seed=1)
        x = jnp.array([0.5, -0.2])
        out_a = a(x)
        out_b = b(x)
        assert not jnp.allclose(out_a, out_b)


class TestInitializedWithKey:
    def test_preserves_static_fields(self):
        predictor = _kan(in_size=3, out_size=2, hidden_widths=(6,), grid_size=4)
        fresh = predictor.initialized_with_key(jr.PRNGKey(11))
        assert fresh.in_size == predictor.in_size
        assert fresh.out_size == predictor.out_size
        assert fresh.hidden_widths == predictor.hidden_widths
        assert fresh.grid_size == predictor.grid_size
        assert fresh.basis == predictor.basis

    def test_changes_inexact_leaves(self):
        predictor = _kan(seed=0)
        fresh = predictor.initialized_with_key(jr.PRNGKey(11))
        leaves_old = _array_leaves(predictor)
        leaves_new = _array_leaves(fresh)
        assert len(leaves_old) == len(leaves_new)
        assert any(
            not jnp.array_equal(lo, ln) for lo, ln in zip(leaves_old, leaves_new, strict=True)
        )

    def test_deterministic_for_same_key(self):
        predictor = _kan()
        a = predictor.initialized_with_key(jr.PRNGKey(7))
        b = predictor.initialized_with_key(jr.PRNGKey(7))
        for la, lb in zip(_array_leaves(a), _array_leaves(b), strict=True):
            assert jnp.array_equal(la, lb)


class TestBoundedPredictorComposition:
    def test_kan_drops_in_as_bounded_inner(self):
        kan = _kan(in_size=2, out_size=1, hidden_widths=(6,))
        bounded = BoundedPredictor(
            input_keys=("a", "b"),
            in_scaler=BoundScaler(
                bounds=((0.0, 1.0), (-1.0, 1.0)),
                transform="sigmoid",
            ),
            inner=kan,
            out_scaler=BoundScaler(
                bounds=((0.0, 10.0),),
                transform="sigmoid",
            ),
        )
        covariates = {"a": jnp.array(0.3), "b": jnp.array(0.5)}
        out = bounded(covariates)
        assert out.shape == (1,)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(out >= 0.0) and jnp.all(out <= 10.0)


class TestBasisValidation:
    def test_unsupported_basis_raises(self):
        with pytest.raises(ValueError):
            _kan(basis="not_a_real_basis")


class TestWithZeroFinalHead:
    """Contract for ``with_zero_final_head``: zero readout layer, untouched hidden.

    Mirrors :class:`tests.test_predictors_mlp.TestWithZeroFinalHead`. The KAN
    "final head" is the readout layer at index ``len(hidden_widths)`` — its
    four trainable arrays (``c_basis``, ``c_spl``, ``c_res``, ``bias``) are
    all zeroed so the layer produces ``zeros(out_size)`` for any input.
    """

    def test_output_is_zero_for_arbitrary_input(self):
        predictor = _kan(in_size=3, out_size=2, hidden_widths=(6,))
        zeroed = predictor.with_zero_final_head()
        for x in (jnp.zeros((3,)), jnp.ones((3,)), jnp.array([0.5, -1.7, 2.3])):
            out = zeroed(x)
            assert jnp.allclose(out, jnp.zeros((2,)), atol=0.0)

    def test_final_layer_params_are_zero(self):
        predictor = _kan(in_size=3, out_size=2, hidden_widths=(6,))
        zeroed = predictor.with_zero_final_head()
        last_idx = len(predictor.hidden_widths)
        last_layer = zeroed.params["layers"][last_idx]
        # Walk every leaf of the final layer; each must be zero.
        for leaf in jtu.tree_leaves(last_layer):
            assert jnp.array_equal(leaf, jnp.zeros_like(leaf))

    def test_hidden_layers_unchanged(self):
        predictor = _kan(in_size=3, out_size=2, hidden_widths=(6, 4))
        zeroed = predictor.with_zero_final_head()
        last_idx = len(predictor.hidden_widths)
        for i in range(last_idx):
            orig_leaves = jtu.tree_leaves(predictor.params["layers"][i])
            new_leaves = jtu.tree_leaves(zeroed.params["layers"][i])
            for orig, new in zip(orig_leaves, new_leaves, strict=True):
                assert jnp.array_equal(orig, new)

    def test_returns_new_instance_input_unchanged(self):
        predictor = _kan(in_size=3, out_size=2, hidden_widths=(6,))
        zeroed = predictor.with_zero_final_head()
        assert zeroed is not predictor
        # Original final layer is not zero — confirms no mutation.
        last_idx = len(predictor.hidden_widths)
        original_final_leaves = jtu.tree_leaves(predictor.params["layers"][last_idx])
        assert any(
            not jnp.array_equal(leaf, jnp.zeros_like(leaf)) for leaf in original_final_leaves
        )

    def test_preserves_static_fields(self):
        predictor = _kan(in_size=4, out_size=3, hidden_widths=(7, 5), grid_size=6)
        zeroed = predictor.with_zero_final_head()
        assert zeroed.in_size == 4
        assert zeroed.out_size == 3
        assert zeroed.hidden_widths == (7, 5)
        assert zeroed.grid_size == 6
        assert zeroed.basis == predictor.basis
        assert zeroed.seed == predictor.seed

    def test_works_with_no_hidden_layers(self):
        # hidden_widths=() means the KAN is a single layer (in_size -> out_size),
        # so the readout layer is at index 0. Same zero-output guarantee.
        predictor = _kan(in_size=3, out_size=2, hidden_widths=())
        zeroed = predictor.with_zero_final_head()
        assert jnp.allclose(zeroed(jnp.array([1.0, 2.0, 3.0])), jnp.zeros((2,)), atol=0.0)


def test_top_level_export():
    import hybridmodels

    assert hybridmodels.KANPredictor is KANPredictor


class TestScaffoldCacheTracing:
    """The cached scaffold has to hold concrete arrays, not tracers.

    ``_scaffold_parts`` memoises ``(graphdef, rest_states)`` on the static
    architecture so a KAN inside a vector field does not rebuild jaxkan on
    every retrace. The first call is normally made *under* ``jit``,
    because the first thing anyone does with a KAN is evaluate it inside a
    compiled function. Build the scaffold plainly and jaxkan's grid
    construction stages into that trace, the cache stores its tracers, and
    the next trace merges them and dies with ``UnexpectedTracerError``.
    Nothing catches it until a KAN is used twice, which is why it survived
    the change that introduced the cache.
    """

    def test_a_cold_cache_warmed_inside_jit_survives_a_second_trace(self):
        _scaffold_parts.cache_clear()
        predictor = _kan(in_size=2, out_size=2, hidden_widths=(4,), grid_size=4)
        x = jnp.array([0.3, -0.2])

        # Two distinct jitted callables, so two traces. The first warms the
        # cache from inside a trace; the second is the one that used to die.
        first = eqx.filter_jit(lambda p, v: p(v))(predictor, x)
        second = eqx.filter_jit(lambda p, v: 2.0 * p(v))(predictor, x)
        assert jnp.allclose(second, 2.0 * first)

    def test_cached_scaffold_leaves_are_concrete_after_a_jit_warm(self):
        _scaffold_parts.cache_clear()
        predictor = _kan(in_size=2, out_size=1, hidden_widths=(3,), grid_size=4)
        eqx.filter_jit(lambda p, v: p(v))(predictor, jnp.array([0.1, 0.2]))

        _graphdef, rest_states = _scaffold_parts(
            predictor.in_size,
            predictor.out_size,
            predictor.hidden_widths,
            predictor.grid_size,
            predictor.basis,
            predictor.seed,
        )
        leaves = [leaf for leaf in jtu.tree_leaves(rest_states) if hasattr(leaf, "shape")]
        assert leaves, "expected the scaffold to carry at least one array leaf"
        for leaf in leaves:
            assert isinstance(leaf, jax.Array)
            assert not isinstance(leaf, jax.core.Tracer)
