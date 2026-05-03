"""Data containers and dataset construction.

Bucketed-irregular only (ADR-0004): per-channel sparse observations are
unioned per experiment into a single timestamp axis at make_dataset time;
experiments sharing union length are stacked into one BucketPayload.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Float, Int


class ChannelObs(eqx.Module):
    ts: Float[Array, " Tc"]
    values: Float[Array, " Tc"]
    variance: Float[Array, " Tc"]

    def __init__(
        self,
        ts: Any,
        values: Any,
        variance: Any = 1.0,
    ) -> None:
        ts_arr = jnp.asarray(ts)
        values_arr = jnp.asarray(values)
        var_arr = jnp.asarray(variance)
        if var_arr.ndim == 0:
            var_arr = jnp.broadcast_to(var_arr, values_arr.shape)
        self.ts = ts_arr
        self.values = values_arr
        self.variance = var_arr


class Experiment(eqx.Module):
    covariates: dict[str, Array]
    y0: Float[Array, " S"]
    channels: dict[str, ChannelObs]
    exp_id: str = eqx.field(static=True)


class BucketPayload(NamedTuple):
    ts: Float[Array, "N T"]
    y_observed: Float[Array, "N T D"]
    yvar: Float[Array, "N T D"]
    mask: Bool[Array, "N T D"]
    covariates: dict[str, Float[Array, " N"]]
    y0: Float[Array, "N S"]
    n_obs: Int[Array, ""]


class Dataset(eqx.Module):
    bucket_payloads: tuple[BucketPayload, ...]
    state_to_output: Callable[..., Array] = eqx.field(static=True)
    output_channel_names: tuple[str, ...] = eqx.field(static=True)
    covariate_names: tuple[str, ...] = eqx.field(static=True)
    _experiments: tuple[Experiment, ...] = ()


def make_experiment(
    *,
    covariates: dict[str, float],
    channels: dict[str, ChannelObs],
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array],
    exp_id: str = "",
) -> Experiment:
    cov_arr: dict[str, Array] = {k: jnp.asarray(v) for k, v in covariates.items()}
    y0 = jnp.asarray(y0_fn(cov_arr, channels))
    return Experiment(covariates=cov_arr, y0=y0, channels=channels, exp_id=exp_id)


def _per_experiment_arrays(
    experiment: Experiment,
    output_channel_names: tuple[str, ...],
) -> tuple[Array, Array, Array, Array]:
    ordered_ts: list[float] = []
    seen: dict[float, int] = {}
    val_dtypes: list[np.dtype[Any]] = []
    var_dtypes: list[np.dtype[Any]] = []
    ts_dtypes: list[np.dtype[Any]] = []
    for ch_name in output_channel_names:
        ch = experiment.channels[ch_name]
        ch_ts_np = np.asarray(ch.ts)
        ch_values_np = np.asarray(ch.values)
        ch_variance_np = np.asarray(ch.variance)
        ts_dtypes.append(ch_ts_np.dtype)
        val_dtypes.append(ch_values_np.dtype)
        var_dtypes.append(ch_variance_np.dtype)
        for t in ch_ts_np.tolist():
            tf = float(t)
            if tf not in seen:
                seen[tf] = len(ordered_ts)
                ordered_ts.append(tf)

    sorted_ts = sorted(ordered_ts)
    ts_to_idx = {t: i for i, t in enumerate(sorted_ts)}

    T = len(sorted_ts)
    D = len(output_channel_names)
    val_dtype = np.result_type(*val_dtypes) if val_dtypes else np.dtype(np.float32)
    var_dtype = np.result_type(*var_dtypes) if var_dtypes else np.dtype(np.float32)
    ts_dtype = np.result_type(*ts_dtypes) if ts_dtypes else np.dtype(np.float32)

    y_observed = np.zeros((T, D), dtype=val_dtype)
    yvar = np.ones((T, D), dtype=var_dtype)
    mask = np.zeros((T, D), dtype=bool)

    for d, ch_name in enumerate(output_channel_names):
        ch = experiment.channels[ch_name]
        ts_np = np.asarray(ch.ts).tolist()
        vals_np = np.asarray(ch.values).tolist()
        var_np = np.asarray(ch.variance).tolist()
        for t, v, var in zip(ts_np, vals_np, var_np, strict=True):
            idx = ts_to_idx[float(t)]
            y_observed[idx, d] = v
            yvar[idx, d] = var
            mask[idx, d] = True

    return (
        jnp.asarray(np.asarray(sorted_ts, dtype=ts_dtype)),
        jnp.asarray(y_observed),
        jnp.asarray(yvar),
        jnp.asarray(mask),
    )


def make_dataset(
    experiments: Sequence[Experiment],
    *,
    state_to_output: Callable[..., Array],
    output_channel_names: tuple[str, ...] | list[str],
) -> Dataset:
    if not experiments:
        raise ValueError("make_dataset requires at least one experiment")
    output_channel_names = tuple(output_channel_names)

    cov_keys = tuple(sorted(experiments[0].covariates.keys()))
    for exp in experiments:
        exp_keys = tuple(sorted(exp.covariates.keys()))
        if exp_keys != cov_keys:
            raise ValueError(
                f"Inconsistent covariate keys: experiment {exp.exp_id!r} has "
                f"{list(exp_keys)}, expected {list(cov_keys)}"
            )
        missing = [c for c in output_channel_names if c not in exp.channels]
        if missing:
            raise ValueError(
                f"Experiment {exp.exp_id!r} missing requested output channels: {missing}"
            )

    by_len: dict[
        int, list[tuple[tuple[Array, Array, Array, Array], Experiment]]
    ] = defaultdict(list)
    for exp in experiments:
        arrays = _per_experiment_arrays(exp, output_channel_names)
        by_len[arrays[0].shape[0]].append((arrays, exp))

    bucket_payloads: list[BucketPayload] = []
    for T in sorted(by_len.keys()):
        items = by_len[T]
        ts_stack = jnp.stack([item[0][0] for item in items], axis=0)
        yo_stack = jnp.stack([item[0][1] for item in items], axis=0)
        yv_stack = jnp.stack([item[0][2] for item in items], axis=0)
        mk_stack = jnp.stack([item[0][3] for item in items], axis=0)
        y0_stack = jnp.stack([item[1].y0 for item in items], axis=0)
        cov_stack: dict[str, Array] = {
            k: jnp.stack([jnp.asarray(item[1].covariates[k]) for item in items], axis=0)
            for k in cov_keys
        }
        n_obs = mk_stack.sum().astype(jnp.int32)
        bucket_payloads.append(
            BucketPayload(
                ts=ts_stack,
                y_observed=yo_stack,
                yvar=yv_stack,
                mask=mk_stack,
                covariates=cov_stack,
                y0=y0_stack,
                n_obs=n_obs,
            )
        )

    return Dataset(
        bucket_payloads=tuple(bucket_payloads),
        state_to_output=state_to_output,
        output_channel_names=output_channel_names,
        covariate_names=cov_keys,
        _experiments=tuple(experiments),
    )


def split_dataset(
    dataset: Dataset,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    key: Array,
) -> tuple[Dataset, Dataset, Dataset]:
    if any(f < 0.0 or f > 1.0 for f in (train, val, test)):
        raise ValueError("train, val, and test fractions must each be in [0, 1]")
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"train + val + test fractions must sum to 1.0, got {total}")

    experiments = dataset._experiments
    if not experiments:
        raise ValueError(
            "split_dataset requires the source experiments; the provided Dataset has none "
            "(was it constructed manually without _experiments?)"
        )
    n = len(experiments)
    perm = np.asarray(jr.permutation(key, n)).tolist()
    n_train = int(np.floor(train * n))
    n_val = int(np.floor(val * n))
    n_test = n - n_train - n_val

    splits_idx = (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val : n_train + n_val + n_test],
    )

    out: list[Dataset] = []
    for idxs in splits_idx:
        if idxs:
            split_exps = [experiments[int(i)] for i in idxs]
            out.append(
                make_dataset(
                    split_exps,
                    state_to_output=dataset.state_to_output,
                    output_channel_names=dataset.output_channel_names,
                )
            )
        else:
            out.append(
                Dataset(
                    bucket_payloads=(),
                    state_to_output=dataset.state_to_output,
                    output_channel_names=dataset.output_channel_names,
                    covariate_names=dataset.covariate_names,
                )
            )
    return cast(tuple[Dataset, Dataset, Dataset], tuple(out))
