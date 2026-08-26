"""Tests for the top-level collocation penalty over a ``predictors`` pytree.

The property that earns this design its place is *pytree invariance*: the
penalty must find every ``BoundedPredictor`` leaf regardless of how the
user chose to nest them, because the framework's whole contract is that
``predictors`` is any pytree (ADR-0006). These tests pin that, plus the
gradient behaviour that makes the penalty worth adding at all.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hybridmodels import BoundedPredictor, BoundScaler, MLPPredictor
from hybridmodels.penalties import bound_penalty, collocation_grids


def _bp(*, key, in_bounds=((0.0, 10.0), (0.0, 5.0)), out_bounds=((-2.0, 2.0),), scale=1.0):
    inner = MLPPredictor(
        in_size=len(in_bounds),
        out_size=len(out_bounds),
        width_size=8,
        depth=1,
        activation_name="tanh",
        key=key,
    )
    if scale != 1.0:
        # Blow up the final layer to force the inner net deep into the
        # output squash's saturated region.
        inner = eqx.tree_at(
            lambda m: m.mlp.layers[-1].weight,
            inner,
            inner.mlp.layers[-1].weight * scale,
        )
    return BoundedPredictor(
        input_keys=tuple(f"x{i}" for i in range(len(in_bounds))),
        in_scaler=BoundScaler(bounds=in_bounds, transform="sigmoid"),
        inner=inner,
        out_scaler=BoundScaler(bounds=out_bounds, transform="sigmoid"),
    )


class TestCollocationGrids:
    def test_one_grid_per_bounded_predictor_leaf(self):
        preds = (_bp(key=jr.PRNGKey(0)), _bp(key=jr.PRNGKey(1)))
        grids = collocation_grids(preds, n_per_dim=3)
        assert len(grids) == 2

    def test_grid_spans_the_declared_box(self):
        preds = (_bp(key=jr.PRNGKey(0), in_bounds=((0.0, 10.0),)),)
        (grid,) = collocation_grids(preds, n_per_dim=5)
        assert grid.shape == (5, 1)
        assert float(grid.min()) == 0.0
        assert float(grid.max()) == 10.0

    def test_grid_is_a_tensor_product_over_dimensions(self):
        preds = (_bp(key=jr.PRNGKey(0), in_bounds=((0.0, 1.0), (0.0, 1.0))),)
        (grid,) = collocation_grids(preds, n_per_dim=4)
        assert grid.shape == (16, 2)

    def test_grids_are_deterministic(self):
        # No key threading, no per-step resampling: restore_best compares
        # raw loss values, and a stochastic penalty would make that
        # comparison noisy.
        preds = (_bp(key=jr.PRNGKey(0)),)
        a = collocation_grids(preds, n_per_dim=3)
        b = collocation_grids(preds, n_per_dim=3)
        assert jnp.array_equal(a[0], b[0])


class TestBoundPenaltyPytreeInvariance:
    """The penalty must not care how the user shaped their pytree."""

    def _shapes(self):
        bp0, bp1 = _bp(key=jr.PRNGKey(0)), _bp(key=jr.PRNGKey(1))

        class Pair(NamedTuple):
            growth: BoundedPredictor
            nucleation: BoundedPredictor

        return {
            "tuple": (bp0, bp1),
            "list": [bp0, bp1],
            "dict": {"growth": bp0, "nucleation": bp1},
            "namedtuple": Pair(growth=bp0, nucleation=bp1),
            "nested": ((bp0,), {"inner": bp1}),
        }

    def test_every_container_shape_gives_the_same_penalty(self):
        values = {}
        for name, preds in self._shapes().items():
            grids = collocation_grids(preds, n_per_dim=3)
            values[name] = float(bound_penalty(preds, grids))
        assert len({round(v, 6) for v in values.values()}) == 1, values

    def test_single_bare_module_is_a_valid_pytree(self):
        bp = _bp(key=jr.PRNGKey(0))
        grids = collocation_grids(bp, n_per_dim=3)
        assert float(bound_penalty(bp, grids)) >= 0.0

    def test_predictors_without_bounded_leaves_penalise_zero(self):
        # A user's parametric trunk need not be a BoundedPredictor at all.
        inner = MLPPredictor(
            in_size=1, out_size=1, width_size=4, depth=1, activation_name="tanh", key=jr.PRNGKey(0)
        )
        grids = collocation_grids(inner, n_per_dim=3)
        assert grids == ()
        assert float(bound_penalty(inner, grids)) == 0.0


class TestBoundPenaltyBehaviour:
    def test_saturated_predictor_scores_far_above_a_fresh_one(self):
        fresh = (_bp(key=jr.PRNGKey(0)),)
        saturated = (_bp(key=jr.PRNGKey(0), scale=50.0),)
        g_fresh = collocation_grids(fresh, n_per_dim=5)
        g_sat = collocation_grids(saturated, n_per_dim=5)
        assert float(bound_penalty(saturated, g_sat)) > 100.0 * float(bound_penalty(fresh, g_fresh))

    def test_gradient_reaches_the_inner_weights(self):
        preds = (_bp(key=jr.PRNGKey(0), scale=50.0),)
        grids = collocation_grids(preds, n_per_dim=5)
        grads = eqx.filter_grad(lambda p: bound_penalty(p, grids))(preds)
        w = grads[0].inner.mlp.layers[-1].weight
        assert jnp.all(jnp.isfinite(w))
        assert float(jnp.linalg.norm(w)) > 1.0

    def test_penalty_is_nonnegative_and_finite(self):
        preds = (_bp(key=jr.PRNGKey(3)),)
        grids = collocation_grids(preds, n_per_dim=4)
        v = bound_penalty(preds, grids)
        assert float(v) >= 0.0 and jnp.isfinite(v)

    def test_is_jit_safe(self):
        preds = (_bp(key=jr.PRNGKey(0)),)
        grids = collocation_grids(preds, n_per_dim=3)
        fn = eqx.filter_jit(lambda p: bound_penalty(p, grids))
        assert jnp.allclose(fn(preds), bound_penalty(preds, grids))

    def test_frozen_scaler_temperature_does_not_absorb_the_gradient(self):
        # The recommended convention freezes BoundScaler leaves, so the
        # penalty's push must land on `inner`, not be soaked up by the
        # scaler's temperature.
        preds = (_bp(key=jr.PRNGKey(0), scale=50.0),)
        grids = collocation_grids(preds, n_per_dim=4)
        grads = eqx.filter_grad(lambda p: bound_penalty(p, grids))(preds)
        inner_norm = float(
            jnp.linalg.norm(
                jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(grads[0].inner)])
            )
        )
        assert inner_norm > 0.0
