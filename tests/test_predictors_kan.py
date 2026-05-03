# ruff: noqa: F722

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from hybridmodels.predictors import (
    BoundedPredictor,
    BoundScaler,
    CovariateSelector,
    KANPredictor,
)


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
            not jnp.array_equal(lo, ln)
            for lo, ln in zip(leaves_old, leaves_new, strict=True)
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
            selector=CovariateSelector(keys=("a", "b")),
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


def test_top_level_export():
    import hybridmodels

    assert hybridmodels.KANPredictor is KANPredictor
