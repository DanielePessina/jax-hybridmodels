from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment, split_dataset


def _identity_state_to_output(state):
    return state


def _zero_y0(_cov, _chan):
    return jnp.zeros((1,))


def _make_exp(i: int, n_ts: int = 2):
    ts = jnp.linspace(0.0, 1.0, n_ts)
    return make_experiment(
        covariates={"a": float(i)},
        channels={"c": ChannelObs(ts=ts, values=ts * float(i))},
        y0_fn=_zero_y0,
        exp_id=f"e{i}",
    )


def _dataset(n: int, *, mixed_lens: bool = False):
    exps = []
    for i in range(n):
        n_ts = 2 + (i % 3) if mixed_lens else 2
        exps.append(_make_exp(i, n_ts=n_ts))
    return make_dataset(
        exps, state_to_output=_identity_state_to_output, output_channel_names=("c",)
    )


def _exp_ids(ds: Dataset) -> tuple[str, ...]:
    return tuple(e.exp_id for e in ds._experiments)


class TestSplitDataset:
    def test_key_required(self):
        ds = _dataset(10)
        with pytest.raises(TypeError):
            split_dataset(ds, train=0.8, val=0.1, test=0.1)  # type: ignore[call-arg]

    def test_fractions_must_sum_to_one(self):
        ds = _dataset(10)
        with pytest.raises(ValueError, match="(?i)sum|fraction"):
            split_dataset(ds, train=0.5, val=0.2, test=0.2, key=jr.key(0))

    def test_fractions_must_be_in_unit_interval(self):
        ds = _dataset(10)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            split_dataset(ds, train=0.5, val=-0.2, test=0.7, key=jr.key(0))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            split_dataset(ds, train=1.2, val=0.0, test=-0.2, key=jr.key(0))

    def test_returns_three_datasets(self):
        ds = _dataset(10)
        result = split_dataset(ds, key=jr.key(0))
        assert len(result) == 3
        for split in result:
            assert isinstance(split, Dataset)

    def test_partition_sizes_match_fractions(self):
        ds = _dataset(100)
        train, val, test = split_dataset(
            ds, train=0.8, val=0.1, test=0.1, key=jr.key(0)
        )
        n_train = len(train._experiments)
        n_val = len(val._experiments)
        n_test = len(test._experiments)
        assert n_train + n_val + n_test == 100
        assert n_train == 80
        assert n_val == 10
        assert n_test == 10

    def test_deterministic(self):
        ds = _dataset(10)
        key = jr.key(42)
        a = split_dataset(ds, key=key)
        b = split_dataset(ds, key=key)
        for ax, bx in zip(a, b, strict=True):
            assert _exp_ids(ax) == _exp_ids(bx)

    def test_different_keys_give_different_splits(self):
        ds = _dataset(20)
        a_train, _, _ = split_dataset(ds, key=jr.key(0))
        b_train, _, _ = split_dataset(ds, key=jr.key(1))
        assert _exp_ids(a_train) != _exp_ids(b_train)

    def test_non_overlapping_and_complete(self):
        ds = _dataset(20)
        train, val, test = split_dataset(ds, key=jr.key(0))
        ids_train = set(_exp_ids(train))
        ids_val = set(_exp_ids(val))
        ids_test = set(_exp_ids(test))
        assert ids_train.isdisjoint(ids_val)
        assert ids_train.isdisjoint(ids_test)
        assert ids_val.isdisjoint(ids_test)
        assert ids_train | ids_val | ids_test == {f"e{i}" for i in range(20)}

    def test_preserves_metadata(self):
        ds = _dataset(10)
        train, val, test = split_dataset(ds, key=jr.key(0))
        for split in (train, val, test):
            assert split.state_to_output is ds.state_to_output
            assert split.output_channel_names == ds.output_channel_names
            assert split.covariate_names == ds.covariate_names

    def test_re_buckets_each_split_independently(self):
        ds = _dataset(15, mixed_lens=True)
        # Source dataset has buckets of size 2, 3, 4.
        src_sizes = sorted({bp.ts.shape[1] for bp in ds.bucket_payloads})
        assert src_sizes == [2, 3, 4]
        train, val, test = split_dataset(ds, key=jr.key(0))
        # Each split has its own bucketing where bucket sizes are unique within the split.
        for split in (train, val, test):
            sizes = [bp.ts.shape[1] for bp in split.bucket_payloads]
            assert len(sizes) == len(set(sizes))
            # Each bucket inside a split must hold only experiments in that split.
            split_ids = set(_exp_ids(split))
            for bp in split.bucket_payloads:
                # n=N: covariates["a"] has shape (N,); confirm against ids count.
                bp_n = bp.ts.shape[0]
                assert bp_n <= len(split_ids)
                assert bp.covariates["a"].shape == (bp_n,)
                assert bp.y0.shape[0] == bp_n


class TestSplitFloorPolicy:
    """Non-divisible splits use floor for train and val; the remainder lands in test."""

    def test_n7_default_fractions(self):
        # n=7, 0.8/0.1/0.1 -> floor(5.6)=5, floor(0.7)=0, remainder=2.
        ds = _dataset(7)
        train, val, test = split_dataset(ds, key=jr.key(0))
        assert len(train._experiments) == 5
        assert len(val._experiments) == 0
        assert len(test._experiments) == 2

    def test_n10_default_fractions(self):
        # n=10, 0.8/0.1/0.1 -> 8/1/1 (already integer).
        ds = _dataset(10)
        train, val, test = split_dataset(ds, key=jr.key(0))
        assert len(train._experiments) == 8
        assert len(val._experiments) == 1
        assert len(test._experiments) == 1

    def test_n13_uneven_fractions_remainder_goes_to_test(self):
        # n=13, 0.7/0.2/0.1 -> floor(9.1)=9, floor(2.6)=2, remainder=2.
        ds = _dataset(13)
        train, val, test = split_dataset(
            ds, train=0.7, val=0.2, test=0.1, key=jr.key(0)
        )
        assert len(train._experiments) == 9
        assert len(val._experiments) == 2
        assert len(test._experiments) == 2

    def test_total_is_always_n(self):
        # Sweep a few non-divisible sizes; the partition must cover every experiment.
        for n in (3, 7, 11, 17, 29):
            ds = _dataset(n)
            train, val, test = split_dataset(ds, key=jr.key(n))
            assert (
                len(train._experiments)
                + len(val._experiments)
                + len(test._experiments)
                == n
            )


class TestSplitWithoutExperiments:
    def test_split_raises_on_empty_experiments(self):
        ds = Dataset(
            bucket_payloads=(),
            state_to_output=_identity_state_to_output,
            output_channel_names=("c",),
            covariate_names=("a",),
        )
        with pytest.raises(ValueError, match="(?i)experiments"):
            split_dataset(ds, key=jr.key(0))
