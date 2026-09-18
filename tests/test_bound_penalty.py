"""Tests for the data-point bound penalty and its point sources.

The default penalty charges output-squash saturation at the *measured
points*: the actual input vectors the loss sees at observed cells,
gathered from the dataset, optionally extended with user-supplied
penalty-only points (no measurements needed). ``box_grid`` is the
collocation-as-extension helper: a deterministic sweep of the input box,
uniform in *warped* coordinates, for when the user wants coverage beyond
the data (a deployment region, a future operating point, ...).

The property that earns this design its place is *pytree invariance*: the
penalty must find every ``BoundedPredictor`` leaf regardless of how the
user chose to nest them, because the framework's whole contract is that
``predictors`` is any pytree. These tests pin that, plus the
gradient behaviour that makes the penalty worth adding at all.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from jaxhybridmodels import BoundedPredictor, BoundScaler, MLPPredictor
from jaxhybridmodels.data import ChannelObs, make_dataset, make_experiment
from jaxhybridmodels.penalties import (
    _bounded_leaves,
    bound_penalty,
    box_grid,
    data_penalty_points,
    length_mask_keep,
)
from jaxhybridmodels.training.kernels import apply_length_mask


def _bp(
    *,
    key,
    in_bounds=((0.0, 10.0), (0.0, 5.0)),
    out_bounds=((-2.0, 2.0),),
    scale=1.0,
    input_keys=("x0", "x1"),
):
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
        input_keys=input_keys,
        in_scaler=BoundScaler(bounds=in_bounds, transform="sigmoid"),
        inner=inner,
        out_scaler=BoundScaler(bounds=out_bounds, transform="sigmoid"),
    )


def _observed_dataset(*, n_experiments=3, t=5, with_probe=False):
    """One bucket whose cells are all observed, plus an optional probe.

    In the v1 data model the union axis of an experiment is the union of
    its observed timestamps, so a non-probe experiment's cells are all
    observed; the probe (empty values, full grid) contributes no measured
    cells. Covariates ``("x0", "x1")`` match the default ``_bp`` input
    keys, so a leaf built with the defaults resolves against this dataset.
    ``x0`` is ``i + 1`` and ``x1`` is ``10 * (i + 1)`` for experiment
    ``i``, so the gathered vectors are easy to assert on.
    """
    ts = jnp.linspace(0.0, 1.0, t)
    experiments = []
    for i in range(n_experiments):
        base = float(i) + 1.0
        experiments.append(
            make_experiment(
                covariates={"x0": base, "x1": 10.0 * base},
                channels={"y": ChannelObs(ts=ts, values=jnp.arange(ts.shape[0]) + base)},
                y0_fn=lambda _c, _ch: jnp.asarray([1.0, 1.0]),
                exp_id=f"exp_{i}",
            )
        )
    if with_probe:
        # R-P8 probe condition: no measurements, so it must contribute no
        # measured points (its ts still defines a wider grid).
        experiments.append(
            make_experiment(
                covariates={"x0": 99.0, "x1": 990.0},
                channels={"y": ChannelObs(ts=ts, values=jnp.array([], dtype=jnp.float32))},
                y0_fn=lambda _c, _ch: jnp.asarray([1.0, 1.0]),
                exp_id="probe",
            )
        )
    return make_dataset(experiments, output_channel_names=("y",))


class TestBoxGrid:
    """The collocation-as-extension helper: a box sweep, uniform in warped space."""

    def test_at_least_two_points_per_dimension_are_required(self):
        scaler = BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid")
        with pytest.raises(ValueError, match="n_per_dim"):
            box_grid(scaler, n_per_dim=1)

    def test_grid_is_a_tensor_product_over_dimensions(self):
        scaler = BoundScaler(bounds=((0.0, 1.0), (0.0, 2.0)), transform="sigmoid")
        grid = box_grid(scaler, n_per_dim=4)
        assert grid.shape == (16, 2)

    def test_linear_warp_reproduces_the_physical_linspace_sweep(self):
        # The old collocation grid sampled physical bounds uniformly; with
        # the linear warp the new grid must be bit-identical to that.
        scaler = BoundScaler(bounds=((0.0, 10.0), (0.0, 5.0)), transform="sigmoid")
        grid = box_grid(scaler, n_per_dim=5)
        axes = [jnp.linspace(0.0, 10.0, 5), jnp.linspace(0.0, 5.0, 5)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        expected = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)
        assert jnp.array_equal(grid, expected)

    def test_log_warp_is_uniform_in_log_space(self):
        # Under warp="log10" the sweep must cover decades evenly, so the
        # low end of the box is not starved of penalty points.
        scaler = BoundScaler(bounds=((1e-6, 1e2),), transform="sigmoid", warp="log10")
        grid = box_grid(scaler, n_per_dim=5)
        assert jnp.allclose(jnp.log10(grid[:, 0]), jnp.linspace(-6.0, 2.0, 5))
        assert float(grid[0, 0]) == pytest.approx(1e-6)
        assert float(grid[-1, 0]) == pytest.approx(1e2)

    def test_grids_are_deterministic(self):
        # No key threading, no per-step resampling: restore_best compares
        # raw loss values, and a stochastic penalty would make that
        # comparison noisy.
        scaler = BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid")
        assert jnp.array_equal(box_grid(scaler, 3), box_grid(scaler, 3))


class TestDataPenaltyPoints:
    """The gather: one point per observed cell, resolved via ``input_keys``."""

    def test_one_source_per_bounded_predictor_leaf(self):
        preds = (_bp(key=jr.PRNGKey(0)), _bp(key=jr.PRNGKey(1)))
        sources = data_penalty_points(preds, _observed_dataset())
        assert len(sources) == 2
        assert all(s is not None for s in sources)

    def test_points_sit_at_the_observed_cells(self):
        ds = _observed_dataset(n_experiments=3, t=5)
        (source,) = data_penalty_points((_bp(key=jr.PRNGKey(0)),), ds)
        # 3 experiments x 5 cells each.
        assert source.points.shape == (15, 2)
        assert source.cell_T.shape == (15,)
        # Experiment i contributes (x0, x1) = (i+1, 10*(i+1)) at every cell.
        for i in range(3):
            block = source.points[i * 5 : (i + 1) * 5]
            assert jnp.allclose(block, jnp.asarray([i + 1.0, 10.0 * (i + 1.0)]))
            assert jnp.array_equal(source.cell_ts[i * 5 : (i + 1) * 5], jnp.arange(5))
            assert jnp.array_equal(source.cell_T[i * 5 : (i + 1) * 5], jnp.asarray([5, 5, 5, 5, 5]))

    def test_vector_columns_follow_input_keys_order(self):
        ds = _observed_dataset(n_experiments=1)
        leaf = _bp(key=jr.PRNGKey(0), input_keys=("x1", "x0"))
        (source,) = data_penalty_points((leaf,), ds)
        # Columns are (x1, x0) = (10, 1), not (1, 10).
        assert jnp.allclose(source.points[0], jnp.asarray([10.0, 1.0]))

    def test_probe_experiments_contribute_no_points(self):
        ds = _observed_dataset(with_probe=True)
        (source,) = data_penalty_points((_bp(key=jr.PRNGKey(0)),), ds)
        assert source.points.shape[0] == 15  # 3 experiments x 5; the probe is all-False
        assert not jnp.any(source.points == 99.0)

    def test_unresolvable_input_key_yields_no_source(self):
        ds = _observed_dataset()
        leaf = _bp(key=jr.PRNGKey(0), input_keys=("x0", "missing"))
        sources = data_penalty_points((leaf,), ds)
        assert sources == (None,)

    def test_predictors_without_bounded_leaves_gather_nothing(self):
        inner = MLPPredictor(
            in_size=1, out_size=1, width_size=4, depth=1, activation_name="tanh", key=jr.PRNGKey(0)
        )
        assert data_penalty_points(inner, _observed_dataset()) == ()


class TestLengthMaskKeep:
    """The penalty follows the same prefix mask the loss uses (R-T curriculum)."""

    def test_full_fraction_keeps_every_observed_cell(self):
        ds = _observed_dataset()
        (source,) = data_penalty_points((_bp(key=jr.PRNGKey(0)),), ds)
        keep = length_mask_keep(source, jnp.asarray(1.0))
        assert jnp.all(keep)

    def test_half_fraction_matches_apply_length_mask(self):
        ds = _observed_dataset(t=5)
        (source,) = data_penalty_points((_bp(key=jr.PRNGKey(0)),), ds)
        (bp,) = ds.bucket_payloads
        masked = apply_length_mask(bp, jnp.asarray(0.5))
        expected = masked.mask.any(axis=-1).reshape(-1)  # same cell order
        keep = length_mask_keep(source, jnp.asarray(0.5))
        assert jnp.array_equal(keep, expected)

    def test_tiny_fraction_keeps_at_least_the_first_timestep(self):
        # apply_length_mask clamps the cutoff at 1 so a phase never scores
        # nothing; the penalty must not disagree with it.
        ds = _observed_dataset(t=5)
        (source,) = data_penalty_points((_bp(key=jr.PRNGKey(0)),), ds)
        keep = length_mask_keep(source, jnp.asarray(0.05))
        assert jnp.all(source.cell_ts[keep] == 0)


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
            points = tuple(box_grid(leaf.in_scaler, n_per_dim=3) for leaf in _bounded_leaves(preds))
            values[name] = float(bound_penalty(preds, points))
        assert len({round(v, 6) for v in values.values()}) == 1, values

    def test_single_bare_module_is_a_valid_pytree(self):
        bp = _bp(key=jr.PRNGKey(0))
        points = (box_grid(bp.in_scaler, n_per_dim=3),)
        assert float(bound_penalty(bp, points)) >= 0.0

    def test_a_bounded_predictor_nested_inside_another_is_found(self):
        # The penalty claims nesting invariance. An is_leaf-stopped traversal
        # only delivers that for nesting in *containers*: it halts at the
        # outermost BoundedPredictor, so an inner one declares a box that is
        # never penalised.
        inner_bp = _bp(
            key=jr.PRNGKey(0),
            in_bounds=((0.0, 10.0),),
            out_bounds=((0.0, 1.0),),
            input_keys=("x0",),
        )
        outer = BoundedPredictor(
            input_keys=("x0",),
            in_scaler=BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid"),
            inner=inner_bp,
            out_scaler=BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid"),
        )
        points = (box_grid(outer.in_scaler, 3), box_grid(inner_bp.in_scaler, 3))
        assert float(bound_penalty((outer,), points)) >= 0.0
        with pytest.raises(ValueError, match="points"):
            bound_penalty((outer,), points[:1])

    def test_nested_predictor_contributes_to_the_penalty(self):
        saturated_inner = _bp(
            key=jr.PRNGKey(0),
            in_bounds=((0.0, 10.0),),
            out_bounds=((0.0, 1.0),),
            input_keys=("x0",),
            scale=50.0,
        )
        fresh_inner = _bp(
            key=jr.PRNGKey(0),
            in_bounds=((0.0, 10.0),),
            out_bounds=((0.0, 1.0),),
            input_keys=("x0",),
        )

        def wrap(inner):
            return BoundedPredictor(
                input_keys=("x0",),
                in_scaler=BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid"),
                inner=inner,
                out_scaler=BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid"),
            )

        hot, cold = (wrap(saturated_inner),), (wrap(fresh_inner),)

        def points_for(preds):
            return tuple(box_grid(leaf.in_scaler, 5) for leaf in _bounded_leaves(preds))

        assert float(bound_penalty(hot, points_for(hot))) > float(
            bound_penalty(cold, points_for(cold))
        )

    def test_predictors_without_bounded_leaves_penalise_zero(self):
        # A user's parametric trunk need not be a BoundedPredictor at all.
        inner = MLPPredictor(
            in_size=1, out_size=1, width_size=4, depth=1, activation_name="tanh", key=jr.PRNGKey(0)
        )
        assert float(bound_penalty(inner, ())) == 0.0


class TestBoundPenaltyBehaviour:
    def test_saturated_predictor_scores_far_above_a_fresh_one(self):
        fresh = (_bp(key=jr.PRNGKey(0)),)
        saturated = (_bp(key=jr.PRNGKey(0), scale=50.0),)
        p_fresh = (box_grid(fresh[0].in_scaler, 5),)
        p_sat = (box_grid(saturated[0].in_scaler, 5),)
        assert float(bound_penalty(saturated, p_sat)) > 100.0 * float(
            bound_penalty(fresh, p_fresh)
        )

    def test_gradient_reaches_the_inner_weights(self):
        preds = (_bp(key=jr.PRNGKey(0), scale=50.0),)
        points = (box_grid(preds[0].in_scaler, 5),)
        grads = eqx.filter_grad(lambda p: bound_penalty(p, points))(preds)
        w = grads[0].inner.mlp.layers[-1].weight
        assert jnp.all(jnp.isfinite(w))
        assert float(jnp.linalg.norm(w)) > 1.0

    def test_penalty_is_nonnegative_and_finite(self):
        preds = (_bp(key=jr.PRNGKey(3)),)
        points = (box_grid(preds[0].in_scaler, 4),)
        v = bound_penalty(preds, points)
        assert float(v) >= 0.0 and jnp.isfinite(v)

    def test_is_jit_safe(self):
        preds = (_bp(key=jr.PRNGKey(0)),)
        points = (box_grid(preds[0].in_scaler, 3),)
        fn = eqx.filter_jit(lambda p: bound_penalty(p, points))
        assert jnp.allclose(fn(preds), bound_penalty(preds, points))

    def test_frozen_scaler_temperature_does_not_absorb_the_gradient(self):
        # The recommended convention freezes BoundScaler leaves, so the
        # penalty's push must land on `inner`, not be soaked up by the
        # scaler's temperature.
        preds = (_bp(key=jr.PRNGKey(0), scale=50.0),)
        points = (box_grid(preds[0].in_scaler, 4),)
        grads = eqx.filter_grad(lambda p: bound_penalty(p, points))(preds)
        inner_norm = float(
            jnp.linalg.norm(
                jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(grads[0].inner)])
            )
        )
        assert inner_norm > 0.0

    def test_empty_points_contribute_zero(self):
        # A leaf the penalty cannot reach (unresolvable keys, no extras) must
        # not NaN the run: an empty per-leaf point array is a zero term.
        preds = (_bp(key=jr.PRNGKey(0)),)
        assert float(bound_penalty(preds, (jnp.zeros((0, 2)),))) == 0.0