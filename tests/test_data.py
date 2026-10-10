"""Datasets as users build them, and the losses that score predictions against them.

Most tests start from the shared decay dataset (four experiments, two
buckets, an unobserved ``t0`` row) or a small two-channel dataset built
here. Loss tests score analytic decay curves, ``exp(-k t)``, against real
buckets, so they need no ODE solve.
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _models import EXPERIMENTS, TRUE_RATE

import jaxhybridmodels as hm
from jaxhybridmodels.data import BucketPayload, Dataset
from jaxhybridmodels.losses import (
    LOSS_REGISTRY,
    bal_mle,
    bal_mse,
    masked_mle,
    masked_mse,
    resolve_loss_fn,
)

ALL_LOSSES = [masked_mse, masked_mle, bal_mse, bal_mle]


def _obs(ts, values, **kwargs) -> hm.ChannelObs:
    return hm.ChannelObs(jnp.asarray(ts), jnp.asarray(values), **kwargs)


def _exp(exp_id: str = "e", covariates: Any = None, **channels) -> hm.Experiment:
    """One experiment with a single-state ``y0``; channels as ``name=(ts, values)``."""
    return hm.make_experiment(
        channels={name: _obs(*spec) for name, spec in channels.items()},
        covariates={"a": 1.0} if covariates is None else covariates,
        y0=[0.0],
        exp_id=exp_id,
    )


def _two_channel() -> Dataset:
    """Two experiments in one bucket: ``x`` dense, ``y`` sparse in the second."""
    full = _exp("full", x=([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]), y=([0.0, 1.0, 2.0], [5.0, 6.0, 7.0]))
    sparse = _exp("sparse", x=([0.0, 1.0, 2.0], [2.0, 1.0, 0.5]), y=([1.0], [4.0]))
    return hm.make_dataset([full, sparse], output_channel_names=("x", "y"))


def _ids(ds: Dataset) -> list[str]:
    return [e.exp_id for e in ds._experiments]


def _analytic(bp: BucketPayload, rate: float) -> jax.Array:
    """Decay from ``y0 = 1`` on the bucket's grid, shaped ``[N, T, 1]``."""
    return jnp.exp(-rate * bp.ts)[..., None]


# ---------------------------------------------------------------------------
# Building datasets
# ---------------------------------------------------------------------------


def test_decay_dataset_buckets_by_union_length(decay_data):
    # t0 = 0 joins every grid as an unobserved row, so the 2- and 3-point
    # experiments land in buckets of length 3 and 4.
    lengths = [bp.ts.shape[1] for bp in decay_data.bucket_payloads]
    assert lengths == [3, 4]
    assert decay_data.output_channel_names == ("conc",)
    assert decay_data.covariate_names == ("temp",)
    total = 0
    for bp in decay_data.bucket_payloads:
        n, t = bp.ts.shape
        assert isinstance(bp, BucketPayload)
        assert bp.y_observed.shape == bp.yvar.shape == bp.mask.shape == (n, t, 1)
        assert bp.y0.shape == (n, 1)
        assert bp.covariates["temp"].shape == (n,)
        assert not bp.mask[:, 0].any()  # the t0 row
        assert bool(bp.mask[:, 1:].all())
        assert int(bp.n_obs) == int(bp.mask.sum())
        total += int(bp.n_obs)
    assert total == sum(len(times) for _, _, times in EXPERIMENTS)


def test_sparse_channels_scatter_onto_a_sorted_union_grid():
    exp = hm.make_experiment(
        channels={
            "x": _obs([2.0, 0.0], [20.0, 10.0]),
            "y": _obs([1.0, 3.0], [100.0, 300.0], variance=jnp.array([0.1, 0.2])),
        },
        y0=[0.0],
    )
    bp = hm.make_dataset([exp], output_channel_names=("x", "y")).bucket_payloads[0]
    np.testing.assert_allclose(bp.ts[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        bp.mask[0], [[True, False], [False, True], [True, False], [False, True]]
    )
    observed = np.where(bp.mask[0], bp.y_observed[0], np.nan)
    np.testing.assert_allclose(
        observed, [[10.0, np.nan], [np.nan, 100.0], [20.0, np.nan], [np.nan, 300.0]]
    )
    # Default variance is one; an array variance follows its timestamps.
    assert float(bp.yvar[0, 0, 0]) == 1.0
    assert float(bp.yvar[0, 1, 1]) == pytest.approx(0.1)
    assert float(bp.yvar[0, 3, 1]) == pytest.approx(0.2)


def test_one_bucket_holds_different_grids_and_masks():
    early = _exp("early", x=([0.0, 1.0], [1.0, 2.0]), y=([], []))
    late = _exp("late", x=([0.0, 5.0], [1.0, 5.0]), y=([5.0], [9.0]))
    bp = hm.make_dataset([early, late], output_channel_names=("x", "y")).bucket_payloads[0]
    np.testing.assert_allclose(bp.ts, [[0.0, 1.0], [0.0, 5.0]])
    assert not bp.mask[0, :, 1].any()
    assert bool(bp.mask[1, 1, 1])
    assert int(bp.n_obs) == 5


def test_output_channel_order_sets_the_last_axis():
    exp = _exp(x=([0.0], [10.0]), y=([0.0], [20.0]))
    for names, first in [(("x", "y"), 10.0), (("y", "x"), 20.0)]:
        ds = hm.make_dataset([exp], output_channel_names=names)
        assert ds.output_channel_names == names
        assert float(ds.bucket_payloads[0].y_observed[0, 0, 0]) == first


def test_vector_covariates_stack_and_y0_fn_sees_covariates_and_channels():
    seen = {}

    def y0_fn(covariates, channels):
        seen["keys"] = (sorted(covariates), sorted(channels))
        return jnp.array([covariates["f"][0], channels["x"].values[0]])

    experiments = [
        hm.make_experiment(
            channels={"x": _obs([0.0], [float(i)])},
            covariates={"f": jnp.array([1.0 + i, 2.0])},
            y0_fn=y0_fn,
            exp_id=f"e{i}",
        )
        for i in range(2)
    ]
    assert seen["keys"] == (["f"], ["x"])
    bp = hm.make_dataset(experiments, output_channel_names=("x",)).bucket_payloads[0]
    np.testing.assert_array_equal(bp.covariates["f"], [[1.0, 2.0], [2.0, 2.0]])
    np.testing.assert_array_equal(bp.y0, [[1.0, 0.0], [2.0, 1.0]])


@pytest.mark.parametrize(
    "ts_dtype, values_dtype, variance_dtype",
    [
        (jnp.float32, jnp.float32, jnp.float32),
        (jnp.float16, jnp.float16, jnp.float16),
        (jnp.float32, jnp.float16, jnp.float32),
    ],
)
def test_dtypes_survive_bucketing(ts_dtype, values_dtype, variance_dtype):
    channel = hm.ChannelObs(
        ts=jnp.asarray([0.0, 1.0], dtype=ts_dtype),
        values=jnp.asarray([1.0, 2.0], dtype=values_dtype),
        variance=jnp.asarray([0.1, 0.2], dtype=variance_dtype),
    )
    exp = hm.make_experiment(channels={"x": channel}, y0=[0.0])
    bp = hm.make_dataset([exp], output_channel_names=("x",)).bucket_payloads[0]
    assert (bp.ts.dtype, bp.y_observed.dtype, bp.yvar.dtype) == (
        ts_dtype,
        values_dtype,
        variance_dtype,
    )


def test_equal_timestamps_in_mixed_float_dtypes_share_one_row():
    # 0.1 has different exact values in float32 and float64; the union grid
    # must canonicalise before using timestamps as set keys.
    with jax.enable_x64():
        exp = hm.make_experiment(
            channels={
                "x": _obs(jnp.asarray([0.1], dtype=jnp.float32), [1.0]),
                "y": _obs(jnp.asarray([0.1], dtype=jnp.float64), [2.0]),
            },
            y0=[0.0],
        )
        bp = hm.make_dataset([exp], output_channel_names=("x", "y")).bucket_payloads[0]
    assert bp.ts.shape == (1, 1)
    assert bp.mask.tolist() == [[[True, True]]]


def test_make_dataset_is_deterministic_and_silent():
    experiments = [_exp("e0", x=([0.0, 1.0], [1.0, 2.0])), _exp("e1", x=([0.0], [3.0]))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = hm.make_dataset(experiments, output_channel_names=("x",))
    second = hm.make_dataset(experiments, output_channel_names=("x",))
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)


def test_dataset_constructs_from_its_public_fields_alone():
    ds = Dataset(bucket_payloads=(), output_channel_names=("c",), covariate_names=("a",))
    assert ds._experiments == ()


@pytest.mark.parametrize(
    "build, match",
    [
        pytest.param(lambda: _obs([0.0, 0.0], [1.0, 2.0]), "repeat", id="duplicate-ts"),
        pytest.param(lambda: _obs([0.0], [1.0], variance=0.0), "variance", id="zero-variance"),
        pytest.param(lambda: _obs([0.0], [jnp.nan]), "finite", id="nan-value"),
        pytest.param(lambda: _obs([0.0], [jnp.inf]), "finite", id="inf-value"),
        pytest.param(lambda: _obs([jnp.inf], [1.0]), "finite", id="inf-ts"),
        pytest.param(lambda: _obs([0.0, 1.0], [1.0]), "same length", id="length-mismatch"),
        pytest.param(lambda: _obs([[0.0]], [[1.0]]), "rank-1", id="rank-2"),
        pytest.param(
            lambda: _obs([0.0, 1.0], [1.0, 2.0], variance=jnp.ones(3)),
            "variance",
            id="variance-length",
        ),
    ],
)
def test_channel_obs_rejects_bad_observations(build, match):
    with pytest.raises(ValueError, match=match):
        build()


_X = {"x": _obs([1.0], [1.0])}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"covariates": {"a": jnp.zeros((2, 2))}, "y0": [0.0]}, "rank-1", id="cov"),
        pytest.param({}, "exactly one", id="no-y0"),
        pytest.param({"y0": [0.0], "y0_fn": lambda c, ch: jnp.zeros(1)}, "exactly one", id="both"),
        pytest.param({"y0": [[0.0]]}, "y0", id="y0-rank-2"),
        pytest.param({"y0_fn": lambda c, ch: jnp.zeros((1, 1))}, "y0", id="y0_fn-rank-2"),
        pytest.param({"y0": [0.0], "t0": float("nan")}, "t0", id="t0-nan"),
    ],
)
def test_make_experiment_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hm.make_experiment(channels=_X, **kwargs)


@pytest.mark.parametrize(
    "experiments, names, match",
    [
        pytest.param([], ("x",), "at least one experiment", id="empty"),
        pytest.param([_exp(x=([0.0], [1.0]))], (), "output channel", id="no-channels"),
        pytest.param([_exp(x=([0.0], [1.0]))], ("x", "x"), "duplicates", id="dup-names"),
        pytest.param([_exp(x=([0.0], [1.0]))], ("x", "y"), "missing", id="missing-channel"),
        pytest.param(
            [_exp("a", x=([0.0], [1.0])), _exp("b", {"b": 1.0}, x=([0.0], [1.0]))],
            ("x",),
            "covariate keys",
            id="covariate-keys",
        ),
        pytest.param(
            [
                _exp("a", {"f": jnp.zeros(2)}, x=([0.0], [1.0])),
                _exp("b", {"f": jnp.zeros(3)}, x=([0.0], [1.0])),
            ],
            ("x",),
            "shape",
            id="covariate-shape",
        ),
        pytest.param([_exp(x=([], []))], ("x",), "no timestamps", id="no-timestamps"),
        pytest.param(
            [
                hm.make_experiment(
                    channels={"x": _obs([1.0], [1.0])}, y0=[0.0], t0=2.0, exp_id="late"
                )
            ],
            ("x",),
            "t0 must precede",
            id="t0-after-data",
        ),
    ],
)
def test_make_dataset_rejects_inconsistent_experiments(experiments, names, match):
    with pytest.raises(ValueError, match=match):
        hm.make_dataset(experiments, output_channel_names=names)


# ---------------------------------------------------------------------------
# Splitting and resampling
# ---------------------------------------------------------------------------


def _numbered(n: int) -> Dataset:
    """``n`` experiments with grids of 2, 3 and 4 points in rotation."""
    experiments = []
    for i in range(n):
        ts = jnp.linspace(0.0, 1.0, 2 + i % 3)
        experiments.append(_exp(f"e{i}", {"a": float(i)}, c=(ts, ts * i)))
    return hm.make_dataset(experiments, output_channel_names=("c",))


@pytest.mark.parametrize(
    "n, fractions, counts",
    [
        (10, {}, (8, 1, 1)),
        # Floor for train and val; the remainder goes to test.
        (13, {"train": 0.7, "val": 0.2, "test": 0.1}, (9, 2, 2)),
        (17, {}, (13, 1, 3)),
        # An explicit zero stays zero, even when flooring leaves a remainder.
        (10, {"train": 1.0, "val": 0.0, "test": 0.0}, (10, 0, 0)),
        (4, {"train": 0.7, "val": 0.3, "test": 0.0}, (3, 1, 0)),
    ],
)
def test_split_partitions_experiments(n, fractions, counts):
    source = _numbered(n)
    splits = hm.split_dataset(source, key=jr.key(0), **fractions)
    assert tuple(len(s._experiments) for s in splits) == counts
    ids = [set(_ids(s)) for s in splits]
    assert set().union(*ids) == {f"e{i}" for i in range(n)}
    assert sum(len(i) for i in ids) == n
    for split, count in zip(splits, counts, strict=True):
        assert split.output_channel_names == source.output_channel_names
        if count == 0:
            assert split.bucket_payloads == ()
            continue
        # Re-bucketed from the split's own experiments.
        assert sum(bp.ts.shape[0] for bp in split.bucket_payloads) == count
        lengths = [bp.ts.shape[1] for bp in split.bucket_payloads]
        assert len(lengths) == len(set(lengths))


def test_split_is_deterministic_in_the_key():
    source = _numbered(20)
    first = hm.split_dataset(source, key=jr.key(42))
    again = hm.split_dataset(source, key=jr.key(42))
    other = hm.split_dataset(source, key=jr.key(1))
    assert [_ids(s) for s in first] == [_ids(s) for s in again]
    assert _ids(first[0]) != _ids(other[0])


@pytest.mark.parametrize(
    "dataset, fractions, match",
    [
        pytest.param(_numbered(10), {"train": 0.5, "val": 0.2, "test": 0.2}, "sum", id="sum"),
        pytest.param(_numbered(10), {"train": 0.5, "val": -0.2, "test": 0.7}, r"\[0, 1\]"),
        pytest.param(_numbered(10), {"train": 1.2, "val": 0.0, "test": -0.2}, r"\[0, 1\]"),
        # A positive fraction that floors to zero experiments would silently
        # drop a split the caller asked for.
        pytest.param(_numbered(7), {}, "val fraction", id="n7-val-empty"),
        pytest.param(_numbered(3), {}, "val fraction", id="n3-val-empty"),
        pytest.param(
            Dataset(bucket_payloads=(), output_channel_names=("c",), covariate_names=()),
            {},
            "experiments",
            id="manual-dataset",
        ),
    ],
)
def test_split_rejects_bad_fractions(dataset, fractions, match):
    with pytest.raises(ValueError, match=match):
        hm.split_dataset(dataset, key=jr.key(0), **fractions)


def test_bootstrap_resamples_with_replacement_and_rebuckets(decay_data):
    default = hm.make_bootstrap_dataset(decay_data, key=jr.key(0))
    assert len(default._experiments) == len(decay_data._experiments)

    big = hm.make_bootstrap_dataset(decay_data, key=jr.key(0), n_experiments=12)
    ids = _ids(big)
    assert len(ids) == 12 and len(set(ids)) < 12  # duplicates, so with replacement
    assert set(ids) <= set(_ids(decay_data))
    # Duplicated experiments add rows to their bucket; the irregular grids
    # still sort into the same two lengths.
    assert sum(bp.ts.shape[0] for bp in big.bucket_payloads) == 12
    assert {bp.ts.shape[1] for bp in big.bucket_payloads} <= {3, 4}

    again = hm.make_bootstrap_dataset(decay_data, key=jr.key(0), n_experiments=12)
    assert _ids(again) == ids


@pytest.mark.parametrize(
    "dataset, n, match",
    [
        pytest.param(_numbered(3), 0, "n_experiments", id="zero"),
        pytest.param(
            Dataset(bucket_payloads=(), output_channel_names=("c",), covariate_names=()),
            None,
            "source experiments",
            id="manual-dataset",
        ),
    ],
)
def test_bootstrap_rejects_bad_inputs(dataset, n, match):
    with pytest.raises(ValueError, match=match):
        hm.make_bootstrap_dataset(dataset, key=jr.key(0), n_experiments=n)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def _two_channel_scored() -> tuple[jax.Array, BucketPayload]:
    bp = _two_channel().bucket_payloads[0]
    offsets = jnp.arange(bp.y_observed.size, dtype=jnp.float32).reshape(bp.y_observed.shape)
    return bp.y_observed + 0.1 * offsets, bp


@pytest.mark.parametrize("loss_fn", ALL_LOSSES)
def test_losses_score_the_decay_fit(loss_fn, decay_data):
    for bp in decay_data.bucket_payloads:
        wrong = loss_fn(_analytic(bp, 0.5), bp)
        right = loss_fn(_analytic(bp, TRUE_RATE), bp)
        assert wrong.shape == ()
        assert float(right) < float(wrong)
        if loss_fn in (masked_mse, bal_mse):
            assert float(right) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("loss_fn", ALL_LOSSES)
@pytest.mark.parametrize("fill", [9_999.0, jnp.nan, jnp.inf])
def test_masked_cells_reach_neither_value_nor_gradient(loss_fn, fill, decay_data):
    # The t0 row of every decay bucket is masked. Garbage there, the shape
    # a diverging solve or a length mask leaves behind, must not change the
    # loss or put NaN into the optimiser.
    bp = decay_data.bucket_payloads[0]
    pred = _analytic(bp, 0.5)
    poisoned = pred.at[:, 0].set(fill)
    np.testing.assert_allclose(loss_fn(poisoned, bp), loss_fn(pred, bp), rtol=1e-6)
    grad = jax.grad(lambda p: loss_fn(p, bp))(poisoned)
    assert bool(jnp.isfinite(grad).all())
    assert not grad[:, 0].any()


def test_registry_names_resolve_to_the_losses_they_name():
    pred, bp = _two_channel_scored()
    pairs = {"mse": masked_mse, "mle": masked_mle, "bal_mse": bal_mse, "bal_mle": bal_mle}
    assert set(LOSS_REGISTRY) == set(pairs)
    for name, fn in pairs.items():
        assert resolve_loss_fn(name, None, None) is LOSS_REGISTRY[name]
        np.testing.assert_allclose(LOSS_REGISTRY[name](pred, bp), fn(pred, bp))
    for spelling in ["MSE", "  mse", "Mse  "]:
        assert resolve_loss_fn(spelling, None, None) is masked_mse


def test_channel_idx_and_channel_weights_compose_linearly():
    pred, bp = _two_channel_scored()
    mask = bp.mask
    # Same numerator, different denominators: channel_idx divides by the
    # selected channel's count, channel_weights by the full mask count.
    via_idx = masked_mse(pred, bp, channel_idx=(0,))
    via_weights = masked_mse(pred, bp, channel_weights=(1.0, 0.0))
    np.testing.assert_allclose(via_idx * mask[..., 0].sum(), via_weights * mask.sum(), rtol=1e-6)
    # Weighted masked_mse is homogeneous and additive over channels.
    one = masked_mse(pred, bp, channel_weights=(1.0, 1.0))
    np.testing.assert_allclose(masked_mse(pred, bp, channel_weights=(3.0, 3.0)), 3 * one, rtol=1e-6)
    parts = [masked_mse(pred, bp, channel_weights=w) for w in [(1.0, 0.0), (0.0, 1.0)]]
    both = masked_mse(pred, bp, channel_weights=(2.0, 1.0))
    np.testing.assert_allclose(both, 2 * parts[0] + parts[1], rtol=1e-6)
    # bal_mse weights are linear in the per-channel losses.
    single = [bal_mse(pred, bp, channel_idx=(i,)) for i in (0, 1)]
    weighted = bal_mse(pred, bp, channel_weights=(2.0, 1.0))
    np.testing.assert_allclose(weighted, 2 * single[0] + single[1], rtol=1e-6)


@pytest.mark.parametrize("loss_fn", [bal_mse, bal_mle])
def test_balanced_losses_weight_experiments_not_observations(loss_fn, decay_data):
    base = decay_data
    doubled = hm.make_dataset(list(base._experiments) * 2, output_channel_names=("conc",))
    for bp, bp2 in zip(base.bucket_payloads, doubled.bucket_payloads, strict=True):
        np.testing.assert_allclose(
            loss_fn(_analytic(bp, 0.5), bp), loss_fn(_analytic(bp2, 0.5), bp2), rtol=1e-6
        )

        # An all-masked probe row adds no data and must not dilute the mean.
        def extend(a):
            return jnp.concatenate([a, a[:1]])

        probe = bp._replace(
            ts=extend(bp.ts),
            y_observed=extend(bp.y_observed),
            yvar=extend(bp.yvar),
            mask=extend(bp.mask).at[-1].set(False),
            covariates={k: extend(v) for k, v in bp.covariates.items()},
            y0=extend(bp.y0),
        )
        pred = jnp.concatenate([_analytic(bp, 0.5), jnp.full_like(_analytic(bp, 0.5)[:1], 99.0)])
        np.testing.assert_allclose(loss_fn(pred, probe), loss_fn(_analytic(bp, 0.5), bp), rtol=1e-6)


def test_likelihood_losses_read_the_observation_variance():
    pred, bp = _two_channel_scored()
    low = bp._replace(yvar=jnp.full_like(bp.yvar, 0.5))
    high = bp._replace(yvar=jnp.full_like(bp.yvar, 1.0))
    # Gaussian NLL: doubling the variance adds n log(2) / 2 and halves the
    # squared-error term.
    se = jnp.where(bp.mask, (pred - bp.y_observed) ** 2, 0.0).sum()
    expected = 0.5 * bp.mask.sum() * jnp.log(2.0) - 0.5 * se
    np.testing.assert_allclose(masked_mle(pred, high) - masked_mle(pred, low), expected, atol=1e-4)
    assert not jnp.allclose(bal_mle(pred, low), bal_mle(pred, high))


def test_resolve_loss_fn_binds_channel_arguments():
    pred, bp = _two_channel_scored()
    big = pred.at[..., 1].add(100.0)
    # Registry losses take the channel arguments directly.
    assert float(resolve_loss_fn("mse", (0,), None)(big, bp)) == pytest.approx(
        float(masked_mse(pred, bp, channel_idx=(0,)))
    )
    doubled = resolve_loss_fn("mse", None, (2.0, 2.0))(pred, bp)
    np.testing.assert_allclose(doubled, 2 * resolve_loss_fn("mse", None, (1.0, 1.0))(pred, bp))

    # A custom loss without channel arguments passes through untouched.
    def plain(pred_obs, bp):
        return jnp.sum(jnp.where(bp.mask, (pred_obs - bp.y_observed) ** 2, 0.0))

    assert resolve_loss_fn(plain, None, None) is plain
    # With channel_idx the channels are sliced out before the call.
    assert float(resolve_loss_fn(plain, (0,), None)(big, bp)) == pytest.approx(
        float(
            plain(
                pred[..., :1], bp._replace(y_observed=bp.y_observed[..., :1], mask=bp.mask[..., :1])
            )
        )
    )

    # A custom loss that declares the keywords receives them.
    seen = {}

    def keyword(pred_obs, bp, *, channel_idx=None, channel_weights=None):
        seen.update(channel_idx=channel_idx, channel_weights=channel_weights)
        return jnp.asarray(0.0)

    resolve_loss_fn(keyword, (1,), (0.5,))(pred, bp)
    assert seen == {"channel_idx": (1,), "channel_weights": (0.5,)}


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(lambda: resolve_loss_fn("rmse", None, None), "bal_mle", id="unknown-name"),
        # Weights must act inside the loss's own reduction, which an opaque
        # (pred_obs, bp) callable cannot do.
        pytest.param(
            lambda: resolve_loss_fn(lambda p, bp: p.sum(), None, (0.5, 0.5)),
            "channel_weights",
            id="weights-on-plain-callable",
        ),
        pytest.param(
            lambda: masked_mse(*_two_channel_scored(), channel_idx=(2,)),
            "channel_idx",
            id="idx-out-of-range",
        ),
        pytest.param(
            lambda: masked_mse(*_two_channel_scored(), channel_idx=(-1,)),
            "channel_idx",
            id="idx-negative",
        ),
        pytest.param(
            lambda: resolve_loss_fn(lambda p, bp: p.sum(), (9,), None)(*_two_channel_scored()),
            "channel_idx",
            id="idx-out-of-range-plain",
        ),
    ],
)
def test_loss_configuration_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()
