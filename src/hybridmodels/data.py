"""Data containers and dataset construction.

Bucketed-irregular only (ADR-0004): per-channel sparse observations are
unioned per experiment into a single timestamp axis at ``make_dataset`` time;
experiments sharing union length are stacked into one ``BucketPayload``.

Lifecycle and lifetime
----------------------
``ChannelObs`` and ``Experiment`` are **host-side input containers**: users
build them, then hand them to ``make_dataset``, which scatters them onto the
per-experiment union axis and stacks bucket-shaped JAX arrays. After that,
training and prediction operate exclusively on ``BucketPayload``; the
original ``Experiment`` objects are kept on ``Dataset._experiments`` only so
``split_dataset`` can re-bucket subsets.

Shape conventions (used throughout the package)
-----------------------------------------------
- ``Tc`` — per-channel observation count (varies across channels)
- ``T``  — per-experiment union timestamp count (``= len(union(ts_c)) over all channels c``)
- ``D``  — number of output channels (``len(output_channel_names)``)
- ``S``  — full state dimension (``y0`` size; defined by the user's ``simulate_fn``)
- ``N``  — number of experiments stacked in a single bucket (i.e. sharing ``T``)
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
    """Per-channel sparse observation triple ``(ts, values, variance)``.

    One ``ChannelObs`` describes a single observable channel for a single
    experiment. Channels are sparse: each channel carries its own ``Tc``
    timestamps independent of other channels, and the framework computes the
    union timestamp axis and resulting mask at ``make_dataset`` time (R-D2).

    Shape contract
    --------------
    All three arrays share the same leading dimension ``Tc``. ``ts`` does not
    need to be sorted — ``_per_experiment_arrays`` re-sorts when building the
    union axis — but it should not contain duplicates within a single channel.

    Attributes
    ----------
    ts : Float[Array, "Tc"]
        Observation times for this channel (same time units the user's
        ``simulate_fn`` consumes).
    values : Float[Array, "Tc"]
        Observed channel values aligned with ``ts``.
    variance : Float[Array, "Tc"]
        Per-observation variance used by ``masked_mle`` / ``bal_mle``. A
        scalar passed to the constructor is broadcast to ``values.shape`` so
        downstream code can assume rank-1.
    """

    ts: Float[Array, " Tc"]
    values: Float[Array, " Tc"]
    variance: Float[Array, " Tc"]

    def __init__(
        self,
        ts: Any,
        values: Any,
        variance: Any = 1.0,
    ) -> None:
        """Construct a ``ChannelObs``, eagerly broadcasting scalar variance.

        ``variance`` may be passed as a scalar (typical when the user has no
        per-observation uncertainty estimate) or as a ``Tc``-shaped array; in
        the scalar case it is broadcast to ``values.shape`` here so all three
        attributes are guaranteed rank-1 thereafter.
        """
        ts_arr = jnp.asarray(ts)
        values_arr = jnp.asarray(values)
        var_arr = jnp.asarray(variance)
        if var_arr.ndim == 0:
            var_arr = jnp.broadcast_to(var_arr, values_arr.shape)
        self.ts = ts_arr
        self.values = values_arr
        self.variance = var_arr


class Experiment(eqx.Module):
    """One experiment: covariates, full initial state, and per-channel observations.

    Built by ``make_experiment``; stored on ``Dataset._experiments`` so
    ``split_dataset`` can re-bucket subsets after a permutation.

    Attributes
    ----------
    covariates : dict[str, Array]
        Named scalar covariates, **constant in time** (R-D5). Keys must agree
        across all experiments handed to a single ``make_dataset`` call.
    y0 : Float[Array, "S"]
        Full model state at ``t=0``, constructed via the user's ``y0_fn`` hook
        at import time (R-D6). Shape is whatever the user's ``simulate_fn``
        consumes; the framework never inspects ``S``.
    channels : dict[str, ChannelObs]
        Per-channel sparse observations. Must contain every name listed in
        ``make_dataset(..., output_channel_names=...)``.
    exp_id : str
        Identifier carried through for diagnostics (static field; not a leaf).
    """

    covariates: dict[str, Array]
    y0: Float[Array, " S"]
    channels: dict[str, ChannelObs]
    exp_id: str = eqx.field(static=True)


class BucketPayload(NamedTuple):
    """Bucket-shaped, JAX-traceable payload produced by ``make_dataset``.

    A bucket holds ``N`` experiments that share the same union-timestamp
    length ``T`` (R-D3). Within a bucket, individual experiments may still
    have **different ts values and different masks** — the bucketing rule
    only fixes ``len(union_ts)``, not the values themselves.

    This is a ``NamedTuple`` (R-D4) rather than an ``eqx.Module`` because
    every field is a stacked JAX array and there is no module-level method
    surface; the whole struct is consumed positionally by jitted training
    and prediction kernels.

    Fields
    ------
    ts : Float[Array, "N T"]
        Per-experiment union-timestamp axis, sorted ascending row-wise.
    y_observed : Float[Array, "N T D"]
        Channel observations scattered onto ``ts``. Cells where the channel
        was not observed at that timestamp hold ``0.0``; consumers must read
        ``mask`` to know which entries are real.
    yvar : Float[Array, "N T D"]
        Per-observation variance (used by MLE losses). Defaults to ``1.0``
        at unobserved cells so masked positions never divide by zero.
    mask : Bool[Array, "N T D"]
        ``True`` iff the corresponding ``y_observed`` cell came from a real
        ``ChannelObs`` entry; ``False`` for union-axis padding.
    covariates : dict[str, Float[Array, "N"]]
        Per-key covariate stacked across the bucket. Same keys as on
        ``Experiment.covariates``, hoisted by an ``N`` axis.
    y0 : Float[Array, "N S"]
        Per-experiment full initial state, stacked.
    n_obs : Int[Array, ""]
        Total observed-cell count for the bucket (``mask.sum()``). Used by
        weighted reductions; not used by ``masked_*`` (which compute their
        own denominators).
    """

    ts: Float[Array, "N T"]
    y_observed: Float[Array, "N T D"]
    yvar: Float[Array, "N T D"]
    mask: Bool[Array, "N T D"]
    covariates: dict[str, Float[Array, " N"]]
    y0: Float[Array, "N S"]
    n_obs: Int[Array, ""]


class Dataset(eqx.Module):
    """Container of bucketed experiments plus the static ``state_to_output`` hook.

    A ``Dataset`` is the artifact training and prediction loops iterate over.
    The ``bucket_payloads`` tuple is the dispatch list (one trace per bucket
    shape, R-J3); ``state_to_output`` is held here so the loss pipeline can
    apply it without the user threading it through every call site.

    Attributes
    ----------
    bucket_payloads : tuple[BucketPayload, ...]
        One ``BucketPayload`` per distinct ``len(union_ts)`` value, ordered
        ascending by ``T``.
    state_to_output : Callable[[Array], Array]
        Pure mapping ``[T, S] -> [T, D]`` projecting full simulator state
        onto observed channels (R-D7). Static — never serialised by the
        framework; users re-import.
    output_channel_names : tuple[str, ...]
        Channel order along the trailing ``D`` axis of every payload. The
        same order is honoured by ``make_dataset`` when scattering values.
    covariate_names : tuple[str, ...]
        Sorted covariate keys (matches each ``Experiment.covariates`` key
        set; sorted for deterministic dict iteration).
    _experiments : tuple[Experiment, ...]
        Source experiments, retained so ``split_dataset`` can re-bucket
        per-split subsets. Empty when a ``Dataset`` is constructed manually
        from raw payloads (in which case ``split_dataset`` will raise).
    """

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
    """Build one ``Experiment`` from raw covariates, channels, and a state-init hook.

    The ``y0_fn`` hook receives the (already-jnp) covariates dict and the
    channels dict and returns the full state at ``t=0`` as ``Float[Array, "S"]``.
    For systems where the observed channels *are* the state, a typical hook is
    ``lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])``. For
    systems with hidden state (e.g. crystallisation moments), the hook
    constructs the latent components from covariates: see CONTEXT.md ``y0_fn``.

    Parameters
    ----------
    covariates
        Scalar covariates; values are converted to 0-d ``jnp`` arrays.
    channels
        Sparse observations keyed by channel name.
    y0_fn
        Hook ``(covariates, channels) -> [S]`` building the full initial state.
    exp_id
        Optional human-readable id propagated to ``Experiment.exp_id``.
    """
    cov_arr: dict[str, Array] = {k: jnp.asarray(v) for k, v in covariates.items()}
    y0 = jnp.asarray(y0_fn(cov_arr, channels))
    return Experiment(covariates=cov_arr, y0=y0, channels=channels, exp_id=exp_id)


def _per_experiment_arrays(
    experiment: Experiment,
    output_channel_names: tuple[str, ...],
) -> tuple[Array, Array, Array, Array]:
    """Build the per-experiment union-axis tensors ``(ts, y_observed, yvar, mask)``.

    Computes ``T = len(union(ts_c) for c in output_channel_names)`` for one
    experiment, allocates ``[T]`` and ``[T, D]`` host buffers, then scatters
    each channel's ``(values, variance)`` into the rows matching its ``ts``
    and lights the corresponding ``mask`` entries.

    Done host-side via ``numpy`` (rather than ``jnp``) because this is a
    one-shot build at import time, not a traced operation; falling back to
    plain Python loops keeps the timestamp set-membership cheap and avoids
    spuriously promoting these dtypes to JAX defaults.

    Returns
    -------
    ts : Float[Array, "T"]
        Sorted union timestamp axis.
    y_observed : Float[Array, "T D"]
        Channel values scattered onto ``ts``; ``0.0`` at unobserved cells.
    yvar : Float[Array, "T D"]
        Per-cell variance; ``1.0`` at unobserved cells (read only via mask).
    mask : Bool[Array, "T D"]
        ``True`` iff cell ``[t, d]`` came from a real ``ChannelObs`` entry.
    """
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
    """Bucket and stack ``experiments`` into a JAX-traceable ``Dataset``.

    Three things happen here, in order:

    1. **Validation.** All experiments must agree on the set of covariate
       keys, and each must define every requested output channel. Mismatches
       raise immediately with the offending ``exp_id``.
    2. **Per-experiment scattering.** For each experiment, ``_per_experiment_arrays``
       builds its union timestamp axis and the ``[T, D]`` observation/mask
       tensors.
    3. **Bucketing (R-D3).** Experiments are grouped by ``T = len(union_ts)``,
       and each group is stacked along a new leading ``N`` axis to produce
       one ``BucketPayload``. Buckets are emitted in ascending ``T`` order.

    Parameters
    ----------
    experiments
        Non-empty sequence of ``Experiment`` objects (typically built via
        ``make_experiment``).
    state_to_output
        Pure mapping ``[T, S] -> [T, D]`` projecting full state onto the
        observed channels. Stored static on the resulting ``Dataset``.
    output_channel_names
        Channel order for the trailing ``D`` axis. Coerced to a tuple before
        being captured statically on the ``Dataset``.

    Returns
    -------
    Dataset
        ``bucket_payloads`` ordered ascending by ``T``; ``_experiments``
        retained so ``split_dataset`` can re-bucket subsets.
    """
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
    """Permute experiments and re-bucket each split independently (R-D8).

    Splits are computed at the ``Experiment`` level — *not* by carving up
    bucket payloads — so each split is re-bucketed from scratch. Counts use
    ``floor(train*n)`` and ``floor(val*n)``; the test split takes the
    remainder so the three sizes sum to ``n`` even with rounding. An empty
    split is returned as a ``Dataset`` with no payloads (and no
    ``_experiments``, so it cannot be split again).

    Parameters
    ----------
    dataset
        Source dataset; must carry ``_experiments`` (raises otherwise).
    train, val, test
        Fractions in ``[0, 1]`` summing to ``1.0`` (within ``np.isclose``).
    key
        Required ``jr.PRNGKey`` for the permutation (R-R1: no silent default).

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        ``(train_dataset, val_dataset, test_dataset)``.
    """
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
