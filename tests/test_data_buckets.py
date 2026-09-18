from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxhybridmodels.data import (
    BucketPayload,
    ChannelObs,
    Dataset,
    make_dataset,
    make_experiment,
)


def _zero_y0(_covariates, _channels):
    return jnp.zeros((2,))


def _make_simple_experiment(
    *,
    exp_id: str = "exp",
    a: float = 1.0,
    b: float = 2.0,
    c_ts=(0.0, 1.0, 2.0),
    c_vals=(10.0, 11.0, 12.0),
):
    return make_experiment(
        covariates={"a": a, "b": b},
        channels={"c": ChannelObs(ts=jnp.asarray(c_ts), values=jnp.asarray(c_vals))},
        y0_fn=_zero_y0,
        exp_id=exp_id,
    )


class TestChannelObs:
    def test_scalar_variance_broadcasts(self):
        ch = ChannelObs(
            ts=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([10.0, 20.0, 30.0]),
            variance=0.5,
        )
        assert ch.variance.shape == (3,)
        assert jnp.allclose(ch.variance, 0.5)

    def test_array_variance_preserved(self):
        var = jnp.array([0.1, 0.2])
        ch = ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0]), variance=var)
        assert jnp.allclose(ch.variance, var)

    def test_default_variance_is_one(self):
        ch = ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))
        assert jnp.allclose(ch.variance, 1.0)

    def test_sparse_ts_independent_per_channel(self):
        # Two channels with totally different ts/values lengths.
        ch_x = ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0]))
        ch_y = ChannelObs(ts=jnp.array([0.5, 1.5, 2.5]), values=jnp.array([10.0, 20.0, 30.0]))
        assert ch_x.ts.shape == (2,)
        assert ch_y.ts.shape == (3,)

    def test_duplicate_timestamps_raise(self):
        with pytest.raises(ValueError, match="duplicate|repeat"):
            ChannelObs(ts=jnp.array([0.0, 0.0]), values=jnp.array([1.0, 2.0]))

    def test_nonpositive_variance_raises(self):
        with pytest.raises(ValueError, match="variance"):
            ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]), variance=0.0)

    @pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
    def test_nonfinite_values_raise(self, value):
        with pytest.raises(ValueError, match="finite|value"):
            ChannelObs(ts=jnp.array([0.0]), values=jnp.array([value]))


class TestMakeExperiment:
    def test_invokes_y0_fn_with_covariates_and_channels(self):
        captured = {}

        def y0_fn(cov, chans):
            captured["cov_keys"] = tuple(sorted(cov.keys()))
            captured["channel_keys"] = tuple(sorted(chans.keys()))
            return jnp.array([42.0, 43.0])

        channels = {"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))}
        exp = make_experiment(
            covariates={"a": 1.0, "b": 2.0},
            channels=channels,
            y0_fn=y0_fn,
            exp_id="e1",
        )
        assert captured["cov_keys"] == ("a", "b")
        assert captured["channel_keys"] == ("c",)
        assert jnp.allclose(exp.y0, jnp.array([42.0, 43.0]))

    def test_preserves_covariates_channels_exp_id(self):
        exp = _make_simple_experiment(exp_id="e1", a=3.0, b=5.0)
        assert exp.exp_id == "e1"
        assert set(exp.covariates.keys()) == {"a", "b"}
        assert float(exp.covariates["a"]) == pytest.approx(3.0)
        assert float(exp.covariates["b"]) == pytest.approx(5.0)
        assert "c" in exp.channels

    def test_y0_can_use_covariates(self):
        def y0_fn(cov, _channels):
            return jnp.array([cov["a"], cov["b"]])

        exp = make_experiment(
            covariates={"a": 7.0, "b": 11.0},
            channels={"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))},
            y0_fn=y0_fn,
            exp_id="e",
        )
        assert jnp.allclose(exp.y0, jnp.array([7.0, 11.0]))

    def test_covariates_must_be_scalar_or_rank_one(self):
        channels = {"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))}
        with pytest.raises(ValueError, match="scalar or rank-1"):
            make_experiment(
                covariates={"a": jnp.zeros((2, 2))},
                channels=channels,
                y0_fn=_zero_y0,
            )

    def test_vector_covariates_are_preserved(self):
        channels = {"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))}
        exp = make_experiment(
            covariates={"features": jnp.array([1.0, 2.0, 3.0])},
            channels=channels,
            y0_fn=_zero_y0,
        )
        assert exp.covariates["features"].shape == (3,)

    def test_y0_must_be_rank_one(self):
        channels = {"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))}
        with pytest.raises(ValueError, match="y0"):
            make_experiment(
                covariates={"a": 1.0},
                channels=channels,
                y0_fn=lambda _cov, _channels: jnp.zeros((1, 1)),
            )


class TestMakeDatasetUnion:
    def test_equal_timestamps_with_mixed_float_dtypes_share_one_union_row(self):
        # The same decimal timestamp has different exact binary values in
        # float32 and float64. Dataset construction must canonicalize before
        # using timestamps as dictionary/set keys.
        with jax.enable_x64():
            exp = make_experiment(
                covariates={},
                channels={
                    "x": ChannelObs(
                        ts=jnp.asarray([0.1], dtype=jnp.float32),
                        values=jnp.array([1.0]),
                    ),
                    "y": ChannelObs(
                        ts=jnp.asarray([0.1], dtype=jnp.float64),
                        values=jnp.array([2.0]),
                    ),
                },
                y0_fn=_zero_y0,
                exp_id="mixed-dtype",
            )
            bp = make_dataset([exp], output_channel_names=("x", "y")).bucket_payloads[0]

        assert bp.ts.shape == (1, 1)
        assert bp.mask.tolist() == [[[True, True]]]

    def test_union_timestamps_sorted(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(ts=jnp.array([2.0, 0.0]), values=jnp.array([20.0, 10.0])),
                "y": ChannelObs(ts=jnp.array([1.0, 3.0]), values=jnp.array([100.0, 300.0])),
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds = make_dataset(
            [exp], output_channel_names=("x", "y")
        )
        assert len(ds.bucket_payloads) == 1
        bp = ds.bucket_payloads[0]
        assert bp.ts.shape == (1, 4)
        assert jnp.allclose(bp.ts[0], jnp.array([0.0, 1.0, 2.0, 3.0]))

    def test_vector_covariates_stack_across_experiments(self):
        def channels(value):
            return {"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([value]))}

        experiments = [
            make_experiment(
                covariates={"features": jnp.array([1.0, 2.0])},
                channels=channels(1.0),
                y0_fn=_zero_y0,
                exp_id="e0",
            ),
            make_experiment(
                covariates={"features": jnp.array([3.0, 4.0])},
                channels=channels(2.0),
                y0_fn=_zero_y0,
                exp_id="e1",
            ),
        ]
        bp = make_dataset(experiments, output_channel_names=("c",)).bucket_payloads[0]
        assert bp.covariates["features"].shape == (2, 2)
        assert jnp.array_equal(
            bp.covariates["features"], jnp.array([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_vector_covariate_shape_mismatch_raises(self):
        channels = {"c": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))}
        experiments = [
            make_experiment(
                covariates={"features": jnp.array([1.0, 2.0])},
                channels=channels,
                y0_fn=_zero_y0,
                exp_id="e0",
            ),
            make_experiment(
                covariates={"features": jnp.array([3.0, 4.0, 5.0])},
                channels=channels,
                y0_fn=_zero_y0,
                exp_id="e1",
            ),
        ]
        with pytest.raises(ValueError, match="shape"):
            make_dataset(experiments, output_channel_names=("c",))

    def test_y_observed_and_mask_at_correct_positions(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0, 2.0]), values=jnp.array([10.0, 20.0])),
                "y": ChannelObs(ts=jnp.array([1.0]), values=jnp.array([99.0])),
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds = make_dataset(
            [exp], output_channel_names=("x", "y")
        )
        bp = ds.bucket_payloads[0]
        assert jnp.allclose(bp.ts[0], jnp.array([0.0, 1.0, 2.0]))
        expected_mask = jnp.array([[True, False], [False, True], [True, False]])
        assert jnp.array_equal(bp.mask[0], expected_mask)
        assert float(bp.y_observed[0, 0, 0]) == 10.0
        assert float(bp.y_observed[0, 1, 1]) == 99.0
        assert float(bp.y_observed[0, 2, 0]) == 20.0

    def test_yvar_filled_from_per_channel_variance(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(
                    ts=jnp.array([0.0, 1.0]),
                    values=jnp.array([1.0, 2.0]),
                    variance=jnp.array([0.1, 0.2]),
                ),
                "y": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([10.0]), variance=0.5),
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds = make_dataset(
            [exp], output_channel_names=("x", "y")
        )
        bp = ds.bucket_payloads[0]
        assert float(bp.yvar[0, 0, 0]) == pytest.approx(0.1)
        assert float(bp.yvar[0, 1, 0]) == pytest.approx(0.2)
        assert float(bp.yvar[0, 0, 1]) == pytest.approx(0.5)


class TestBucketing:
    def test_groups_by_union_length(self):
        exp1 = make_experiment(
            covariates={"a": 1.0},
            channels={"x": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0]))},
            y0_fn=_zero_y0,
            exp_id="e1",
        )
        exp2 = make_experiment(
            covariates={"a": 2.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0, 1.0, 2.0]), values=jnp.array([1.0, 2.0, 3.0]))
            },
            y0_fn=_zero_y0,
            exp_id="e2",
        )
        ds = make_dataset(
            [exp1, exp2],
            output_channel_names=("x",),
        )
        assert len(ds.bucket_payloads) == 2
        sizes = sorted(bp.ts.shape[1] for bp in ds.bucket_payloads)
        assert sizes == [2, 3]

    def test_same_bucket_with_different_timestamps(self):
        exp1 = make_experiment(
            covariates={"a": 1.0},
            channels={"x": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0]))},
            y0_fn=_zero_y0,
            exp_id="e1",
        )
        exp2 = make_experiment(
            covariates={"a": 2.0},
            channels={"x": ChannelObs(ts=jnp.array([0.0, 5.0]), values=jnp.array([1.0, 5.0]))},
            y0_fn=_zero_y0,
            exp_id="e2",
        )
        ds = make_dataset(
            [exp1, exp2],
            output_channel_names=("x",),
        )
        assert len(ds.bucket_payloads) == 1
        bp = ds.bucket_payloads[0]
        assert bp.ts.shape == (2, 2)
        # Different ts arrays per experiment in the same bucket.
        assert not jnp.allclose(bp.ts[0], bp.ts[1])

    def test_same_bucket_with_different_masks(self):
        exp_full = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])),
                "y": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([10.0, 20.0])),
            },
            y0_fn=_zero_y0,
            exp_id="full",
        )
        exp_partial = make_experiment(
            covariates={"a": 2.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([5.0, 6.0])),
                "y": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([50.0])),
            },
            y0_fn=_zero_y0,
            exp_id="partial",
        )
        ds = make_dataset(
            [exp_full, exp_partial],
            output_channel_names=("x", "y"),
        )
        assert len(ds.bucket_payloads) == 1
        bp = ds.bucket_payloads[0]
        # Both experiments live in T=2 bucket; their masks differ on channel y.
        assert bp.mask.shape == (2, 2, 2)
        assert bool(bp.mask[0, 1, 1]) is True
        assert bool(bp.mask[1, 1, 1]) is False


class TestBucketPayloadShapes:
    def test_shapes_match_spec(self):
        n = 3

        def y0_full(_cov, _chan):
            return jnp.zeros((4,))

        exps = [
            make_experiment(
                covariates={"a": float(i), "b": float(2 * i)},
                channels={
                    "x": ChannelObs(
                        ts=jnp.array([0.0, 1.0]), values=jnp.array([float(i), float(i + 1)])
                    ),
                    "y": ChannelObs(
                        ts=jnp.array([0.0, 1.0]), values=jnp.array([float(2 * i), float(2 * i + 1)])
                    ),
                },
                y0_fn=y0_full,
                exp_id=f"e{i}",
            )
            for i in range(n)
        ]
        ds = make_dataset(
            exps, output_channel_names=("x", "y")
        )
        bp = ds.bucket_payloads[0]
        T, D, S = 2, 2, 4
        assert bp.ts.shape == (n, T)
        assert bp.y_observed.shape == (n, T, D)
        assert bp.yvar.shape == (n, T, D)
        assert bp.mask.shape == (n, T, D)
        assert bp.y0.shape == (n, S)
        assert set(bp.covariates.keys()) == {"a", "b"}
        assert bp.covariates["a"].shape == (n,)
        assert bp.covariates["b"].shape == (n,)
        assert bp.n_obs.shape == ()
        assert int(bp.n_obs) == int(bp.mask.sum())

    def test_bucket_payload_is_namedtuple(self):
        exp = _make_simple_experiment()
        ds = make_dataset(
            [exp], output_channel_names=("c",)
        )
        bp = ds.bucket_payloads[0]
        assert isinstance(bp, BucketPayload)
        assert hasattr(bp, "_fields")
        assert bp._fields == (
            "ts",
            "y_observed",
            "yvar",
            "mask",
            "covariates",
            "y0",
            "n_obs",
        )
        # Not promoted to a class with extra methods/state.
        assert isinstance(bp, tuple)


class TestOutputChannelOrdering:
    def test_order_controls_D_axis(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([10.0])),
                "y": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([20.0])),
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds_xy = make_dataset(
            [exp], output_channel_names=("x", "y")
        )
        ds_yx = make_dataset(
            [exp], output_channel_names=("y", "x")
        )
        assert float(ds_xy.bucket_payloads[0].y_observed[0, 0, 0]) == 10.0
        assert float(ds_yx.bucket_payloads[0].y_observed[0, 0, 0]) == 20.0
        assert ds_xy.output_channel_names == ("x", "y")
        assert ds_yx.output_channel_names == ("y", "x")


class TestDatasetMetadata:
    def test_names_stored_and_dataset_carries_no_state_to_output(self):
        exp = _make_simple_experiment()
        ds = make_dataset(
            [exp], output_channel_names=("c",)
        )
        assert isinstance(ds, Dataset)
        assert ds.output_channel_names == ("c",)
        # covariate_names follow declared keys (sorted, since order isn't user-controlled).
        assert set(ds.covariate_names) == {"a", "b"}
        # The Dataset is pure data: the projection is a model property and is
        # passed to prediction and training separately.
        assert not hasattr(ds, "state_to_output")


class TestIdempotence:
    def test_make_dataset_deterministic(self):
        exps = [
            _make_simple_experiment(exp_id="e1"),
            _make_simple_experiment(exp_id="e2", c_vals=(20.0, 21.0, 22.0)),
        ]
        ds1 = make_dataset(
            exps, output_channel_names=("c",)
        )
        ds2 = make_dataset(
            exps, output_channel_names=("c",)
        )
        assert len(ds1.bucket_payloads) == len(ds2.bucket_payloads)
        for bp1, bp2 in zip(ds1.bucket_payloads, ds2.bucket_payloads, strict=True):
            assert jnp.array_equal(bp1.ts, bp2.ts)
            assert jnp.array_equal(bp1.y_observed, bp2.y_observed)
            assert jnp.array_equal(bp1.yvar, bp2.yvar)
            assert jnp.array_equal(bp1.mask, bp2.mask)
            assert jnp.array_equal(bp1.y0, bp2.y0)
            assert int(bp1.n_obs) == int(bp2.n_obs)


class TestErrors:
    def test_missing_requested_channel_raises(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={"x": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))},
            y0_fn=_zero_y0,
            exp_id="e",
        )
        with pytest.raises(ValueError, match="(?i)channel"):
            make_dataset(
                [exp],
                output_channel_names=("x", "y"),
            )

    def test_inconsistent_covariate_keys_raises(self):
        e1 = make_experiment(
            covariates={"a": 1.0, "b": 2.0},
            channels={"x": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))},
            y0_fn=_zero_y0,
            exp_id="e1",
        )
        e2 = make_experiment(
            covariates={"a": 1.0},
            channels={"x": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([1.0]))},
            y0_fn=_zero_y0,
            exp_id="e2",
        )
        with pytest.raises(ValueError, match="(?i)covariate"):
            make_dataset(
                [e1, e2],
                output_channel_names=("x",),
            )

    def test_empty_experiments_raises(self):
        with pytest.raises(ValueError):
            make_dataset([], output_channel_names=("x",))


class TestNObsCount:
    def test_n_obs_equals_total_observed(self):
        # 2 experiments, mixed sparsity, same union length -> same bucket.
        exp1 = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])),
                "y": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([10.0, 20.0])),
            },
            y0_fn=_zero_y0,
            exp_id="e1",
        )
        exp2 = make_experiment(
            covariates={"a": 2.0},
            channels={
                "x": ChannelObs(ts=jnp.array([0.0, 1.0]), values=jnp.array([3.0, 4.0])),
                "y": ChannelObs(ts=jnp.array([0.0]), values=jnp.array([30.0])),
            },
            y0_fn=_zero_y0,
            exp_id="e2",
        )
        ds = make_dataset(
            [exp1, exp2],
            output_channel_names=("x", "y"),
        )
        assert len(ds.bucket_payloads) == 1
        bp = ds.bucket_payloads[0]
        # exp1: 2*2 = 4 observed; exp2: 2 (x) + 1 (y) = 3 observed; total = 7.
        assert int(bp.n_obs) == 7
        assert int(bp.n_obs) == int(np.asarray(bp.mask).sum())


class TestDtypePreservation:
    def test_float32_preserved(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(
                    ts=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
                    values=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
                    variance=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
                )
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds = make_dataset(
            [exp], output_channel_names=("x",)
        )
        bp = ds.bucket_payloads[0]
        assert bp.ts.dtype == jnp.float32
        assert bp.y_observed.dtype == jnp.float32
        assert bp.yvar.dtype == jnp.float32

    def test_float16_preserved(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(
                    ts=jnp.asarray([0.0, 1.0], dtype=jnp.float16),
                    values=jnp.asarray([1.0, 2.0], dtype=jnp.float16),
                    variance=jnp.asarray([0.1, 0.2], dtype=jnp.float16),
                )
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds = make_dataset(
            [exp], output_channel_names=("x",)
        )
        bp = ds.bucket_payloads[0]
        # No silent downcast to float32 — the float16 inputs should round-trip.
        assert bp.ts.dtype == jnp.float16
        assert bp.y_observed.dtype == jnp.float16
        assert bp.yvar.dtype == jnp.float16

    def test_value_and_variance_dtypes_independent(self):
        # Channel x: values float16, variance float32 -> y_observed=float16, yvar=float32.
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={
                "x": ChannelObs(
                    ts=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
                    values=jnp.asarray([1.0, 2.0], dtype=jnp.float16),
                    variance=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
                )
            },
            y0_fn=_zero_y0,
            exp_id="e",
        )
        ds = make_dataset(
            [exp], output_channel_names=("x",)
        )
        bp = ds.bucket_payloads[0]
        assert bp.y_observed.dtype == jnp.float16
        assert bp.yvar.dtype == jnp.float32


class TestNoWarnings:
    def test_make_dataset_emits_no_user_warnings(self):
        exp = _make_simple_experiment()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_dataset(
                [exp],
                output_channel_names=("c",),
            )
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == [], [str(w.message) for w in user_warnings]


class TestManualConstruction:
    def test_construct_with_only_public_spec_fields(self):
        # The three user-facing fields of ``Dataset`` are ``bucket_payloads``,
        # ``output_channel_names``, and ``covariate_names``. ``_experiments``
        # is internal — kept so ``split_dataset`` can re-bucket subsets — and
        # must not be required when the user constructs a Dataset manually.
        ds = Dataset(
            bucket_payloads=(),
            output_channel_names=("c",),
            covariate_names=("a",),
        )
        assert ds.bucket_payloads == ()
        assert ds._experiments == ()


class TestDatasetBoundaryValidation:
    def test_an_experiment_without_any_timestamps_raises(self):
        exp = make_experiment(
            covariates={"a": 1.0},
            channels={"c": ChannelObs(ts=jnp.array([]), values=jnp.array([]))},
            y0_fn=lambda _cov, _channels: jnp.zeros((1,)),
            exp_id="empty",
        )
        with pytest.raises(ValueError, match="timestamp|empty"):
            make_dataset([exp], output_channel_names=("c",))
