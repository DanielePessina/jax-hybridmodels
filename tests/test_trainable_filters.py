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
    MLPPredictor,
)
from hybridmodels.trainable import (
    default_trainable,
    freeze_modules_of_type,
    freeze_paths,
    freeze_where,
    trainable_mask,
)


def _bounded() -> BoundedPredictor:
    return BoundedPredictor(
        input_keys=("a", "b"),
        in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
        inner=MLPPredictor(in_size=2, out_size=2, width_size=4, depth=2, key=jr.PRNGKey(0)),
        out_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
    )


def _all_true(mask) -> bool:
    return all(bool(leaf) for leaf in jtu.tree_leaves(mask))


def _all_false(mask) -> bool:
    return all(not bool(leaf) for leaf in jtu.tree_leaves(mask))


def _count_true(mask) -> int:
    return sum(1 for leaf in jtu.tree_leaves(mask) if bool(leaf))


def _leaves_equal(a, b) -> bool:
    leaves_a = jtu.tree_leaves(a)
    leaves_b = jtu.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(bool(x) is bool(y) for x, y in zip(leaves_a, leaves_b, strict=True))


def _path_to_dotted_first_leaf(mask) -> str:
    """First realised dotted path in a mask, for building a near-miss typo."""
    from hybridmodels.trainable import _path_to_dotted

    return _path_to_dotted(jtu.tree_flatten_with_path(mask)[0][0][0])


class TestDefaultTrainable:
    def test_inexact_array_returns_true(self):
        assert default_trainable(jnp.asarray(1.5)) is True
        assert default_trainable(jnp.array([1.0, 2.0])) is True

    def test_int_array_returns_false(self):
        assert default_trainable(jnp.asarray(3, dtype=jnp.int32)) is False

    def test_bool_array_returns_false(self):
        assert default_trainable(jnp.asarray(True)) is False

    def test_python_int_returns_false(self):
        assert default_trainable(3) is False

    def test_string_returns_false(self):
        assert default_trainable("hello") is False

    def test_none_returns_false(self):
        assert default_trainable(None) is False


class TestTrainableMask:
    def test_structure_matches_predictor(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        assert jtu.tree_structure(mask) == jtu.tree_structure(bp)

    def test_inexact_arrays_are_true_other_leaves_false(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        leaves_bp = jtu.tree_leaves(bp)
        leaves_mask = jtu.tree_leaves(mask)
        assert len(leaves_bp) == len(leaves_mask)
        assert len(leaves_bp) > 0
        for lp, lm in zip(leaves_bp, leaves_mask, strict=True):
            assert bool(lm) is bool(eqx.is_inexact_array(lp))

    def test_custom_predicate_all_false(self):
        bp = _bounded()
        mask = trainable_mask(bp, predicate=lambda _: False)
        assert _all_false(mask)

    def test_custom_predicate_all_true(self):
        bp = _bounded()
        mask = trainable_mask(bp, predicate=lambda _: True)
        assert _all_true(mask)


class TestFreezeModulesOfType:
    def test_zeros_boundscaler_temperature_leaves(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_modules_of_type(mask, bp, BoundScaler)
        assert bool(new_mask.in_scaler.temperature) is False
        assert bool(new_mask.out_scaler.temperature) is False

    def test_leaves_other_subtrees_unchanged(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_modules_of_type(mask, bp, BoundScaler)
        assert _leaves_equal(mask.inner, new_mask.inner)

    def test_returns_new_mask_input_unmutated(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_modules_of_type(mask, bp, BoundScaler)
        assert new_mask is not mask
        assert bool(mask.in_scaler.temperature) is True
        assert bool(mask.out_scaler.temperature) is True

    def test_no_match_returns_equivalent_mask(self):
        bp = _bounded()
        mask = trainable_mask(bp)

        class _Unrelated(eqx.Module):
            pass

        new_mask = freeze_modules_of_type(mask, bp, _Unrelated)
        for o, n in zip(jtu.tree_leaves(mask), jtu.tree_leaves(new_mask), strict=True):
            assert bool(o) is bool(n)

    def test_composable_union_of_two_types(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        m1 = freeze_modules_of_type(mask, bp, BoundScaler)
        m2 = freeze_modules_of_type(m1, bp, MLPPredictor)
        assert bool(m2.in_scaler.temperature) is False
        assert bool(m2.out_scaler.temperature) is False
        assert _all_false(m2.inner)
        # Original mask still untouched.
        assert _leaves_equal(mask, trainable_mask(bp))
        # Intermediate m1 has MLP leaves still as in default.
        assert _leaves_equal(m1.inner, mask.inner)


class TestFreezeWhere:
    def test_isinstance_predicate_zeros_matching_subtrees(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_where(mask, bp, lambda m: isinstance(m, BoundScaler))
        assert bool(new_mask.in_scaler.temperature) is False
        assert bool(new_mask.out_scaler.temperature) is False
        assert _leaves_equal(mask.inner, new_mask.inner)

    def test_predicate_never_true_is_no_op(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_where(mask, bp, lambda _: False)
        for o, n in zip(jtu.tree_leaves(mask), jtu.tree_leaves(new_mask), strict=True):
            assert bool(o) is bool(n)

    def test_returns_new_mask_input_unmutated(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_where(mask, bp, lambda m: isinstance(m, BoundScaler))
        assert new_mask is not mask
        assert bool(mask.in_scaler.temperature) is True

    def test_freezing_module_with_no_leaves_is_safe(self):
        # No module in the natural predictor tree is now "leafless" — this
        # test pins the leafless-module branch in freeze_where via a synthetic
        # Predictor subclass with no inexact-array leaves. Freezing a node
        # that contributes no leaves to the mask must leave the mask untouched.
        from hybridmodels.predictors import Predictor

        class _LeaflessPredictor(Predictor):
            pass

            def __call__(self, x):
                return x

        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_where(mask, bp, lambda m: isinstance(m, _LeaflessPredictor))
        for o, n in zip(jtu.tree_leaves(mask), jtu.tree_leaves(new_mask), strict=True):
            assert bool(o) is bool(n)


class TestFreezePaths:
    def test_zeros_named_leaf(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_paths(mask, ("in_scaler.temperature",))
        assert bool(new_mask.in_scaler.temperature) is False
        assert bool(new_mask.out_scaler.temperature) is True

    def test_zeros_multiple_named_leaves(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_paths(mask, ("in_scaler.temperature", "out_scaler.temperature"))
        assert bool(new_mask.in_scaler.temperature) is False
        assert bool(new_mask.out_scaler.temperature) is False

    def test_unknown_path_raises(self):
        # A silent no-op meant a typo left a leaf the caller believed was
        # frozen training normally: a wrong experiment, not a wrong program,
        # and nothing to notice at the time.
        bp = _bounded()
        mask = trainable_mask(bp)
        with pytest.raises(ValueError, match="no leaf matches"):
            freeze_paths(mask, ("does.not.exist",))

    def test_unknown_path_error_suggests_close_matches(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        real = _path_to_dotted_first_leaf(mask)
        typo = real[:-1] + "X"
        with pytest.raises(ValueError, match="Closest matches"):
            freeze_paths(mask, (typo,))

    def test_a_valid_path_alongside_an_invalid_one_still_raises(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        real = _path_to_dotted_first_leaf(mask)
        with pytest.raises(ValueError, match="no leaf matches"):
            freeze_paths(mask, (real, "nope.nope"))

    def test_empty_paths_is_no_op(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_paths(mask, ())
        for o, n in zip(jtu.tree_leaves(mask), jtu.tree_leaves(new_mask), strict=True):
            assert bool(o) is bool(n)

    def test_returns_new_mask_input_unmutated(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        new_mask = freeze_paths(mask, ("in_scaler.temperature",))
        assert new_mask is not mask
        assert bool(mask.in_scaler.temperature) is True

    def test_freezes_only_one_leaf_total(self):
        bp = _bounded()
        mask = trainable_mask(bp)
        before = _count_true(mask)
        new_mask = freeze_paths(mask, ("in_scaler.temperature",))
        after = _count_true(new_mask)
        assert after == before - 1
