"""Data containers and dataset construction.

This module turns a list of experiments, each with sparsely and
irregularly sampled measurements, into rectangular arrays that JAX can
compile against.

The framework accepts one data layout, called bucketed irregular. Every
measured quantity of an experiment (a *channel*) carries its own set of
observation times, and no two channels need to agree. ``make_dataset``
takes the union of those time sets per experiment (the *union timestamp
axis*), writes each channel's values into the rows where it was actually
measured, and records a boolean *mask* marking the real measurements.
Experiments whose union axes have the same length are then stacked into
one ``BucketPayload``, a *bucket*.

Buckets exist because JAX compiles once per distinct input shape, so
batching equal-length experiments lets one kernel run the whole group and
keeps the compilation count equal to the number of distinct lengths, with
no padding. Other layouts (regular grids, single-channel, ragged) are not
modelled here; pre-process such data into this shape.

Lifecycle and lifetime
----------------------
``ChannelObs`` and ``Experiment`` are host-side input containers. After
``make_dataset``, training and prediction read only ``BucketPayload``. The
original experiments stay on the dataset, readable through
``Dataset.experiments``, so ``split_dataset`` can re-bucket subsets after a
permutation and ``HybridModel.predict`` can key results by ``exp_id``.

Shape conventions (used throughout the package)
-----------------------------------------------
- ``Tc``: observation count for one channel of one experiment. Varies
  across channels.
- ``T``: length of one experiment's union timestamp axis
  (``= len(union(ts_c) for c in output_channel_names)``).
- ``D``: number of output channels (``len(output_channel_names)``).
- ``S``: full state dimension, the size of ``y0``. Defined by the user's
  ``simulate_fn``, never inspected by the framework.
- ``N``: number of experiments stacked in a single bucket, that is, the
  number sharing a given ``T``.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Float, Int


class ChannelObs(eqx.Module):
    """What one measured quantity of one experiment was observed to be, and when.

    A *channel* is one observable quantity, for example concentration or
    mean crystal size. Each carries its own ``Tc`` observation times, so
    channels can be sampled at completely different rates.
    ``make_dataset`` later merges them onto a shared time axis.

    For observed channels, all three arrays share the leading dimension
    ``Tc``. A probe channel may instead provide nonempty ``ts`` with empty
    ``values``; its timestamps define the integration grid while contributing
    no observations. ``ts`` may be unsorted, since
    ``_per_experiment_arrays`` sorts when it builds the union axis, but must
    not repeat a time within one channel.

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
        *,
        ts: Any,
        values: Any,
        variance: Any = 1.0,
    ) -> None:
        """Construct a ``ChannelObs``, broadcasting a scalar variance up front.

        Keyword-only, so ``ts`` and ``values`` cannot be swapped by position.
        ``variance`` may be a scalar or a ``Tc``-shaped array. A scalar is
        broadcast to ``values.shape`` here, so downstream code can assume
        all three attributes are rank-1.
        """
        ts_arr = jnp.asarray(ts)
        values_arr = jnp.asarray(values)
        var_arr = jnp.asarray(variance)
        if ts_arr.ndim != 1 or values_arr.ndim != 1:
            raise ValueError(
                "ChannelObs ts and values must both be rank-1 arrays; got "
                f"ts.ndim={ts_arr.ndim}, values.ndim={values_arr.ndim}."
            )
        if values_arr.shape[0] != 0 and ts_arr.shape != values_arr.shape:
            raise ValueError(
                "ChannelObs ts and values must have the same length; got "
                f"{ts_arr.shape[0]} and {values_arr.shape[0]}"
            )
        if var_arr.ndim == 0:
            var_arr = jnp.broadcast_to(var_arr, values_arr.shape)
        elif var_arr.ndim != 1 or var_arr.shape != values_arr.shape:
            raise ValueError(
                "ChannelObs variance must be scalar or rank-1 with the same "
                f"length as values; got shape {var_arr.shape} for values shape {values_arr.shape}."
            )
        ts_np = np.asarray(ts_arr)
        values_np = np.asarray(values_arr)
        var_np = np.asarray(var_arr)
        if not np.all(np.isfinite(ts_np)):
            raise ValueError("ChannelObs timestamps must be finite")
        if not np.all(np.isfinite(values_np)):
            raise ValueError("ChannelObs values must be finite")
        if np.unique(ts_np).size != ts_np.size:
            raise ValueError("ChannelObs timestamps must not repeat")
        if np.any(var_np <= 0.0) or not np.all(np.isfinite(var_np)):
            raise ValueError("ChannelObs variance must be finite and strictly positive")
        self.ts = ts_arr
        self.values = values_arr
        self.variance = var_arr


class Experiment(eqx.Module):
    """One run of the physical system: its conditions, its starting state, its measurements.

    Build these with ``make_experiment``, ``experiment_from_array`` or
    ``experiments_from_dataframe``. A dataset keeps them, readable through
    ``Dataset.experiments``, so ``split_dataset`` can re-bucket subsets
    after a permutation.

    Attributes
    ----------
    covariates : dict[str, Array]
        Named scalar or rank-1 vector conditions of the run that do not
        change with time, such as ``temperature_C`` or a feed composition.
        Every experiment passed to one ``make_dataset`` call must define the
        same keys and shapes.
    y0 : Float[Array, "S"]
        Full model state at ``t0`` (or the first observation time), of length
        ``S``. Built by the user's
        ``y0_fn`` hook when the experiment is constructed. The state may
        contain components that are never observed, so ``S`` need not equal
        the channel count. The framework never inspects ``S``.
    channels : dict[str, ChannelObs]
        Sparse observations, one entry per measured quantity. Must contain
        every name listed in ``make_dataset(..., output_channel_names=...)``.
    exp_id : str
        Identifier, unique within a dataset; ``make_dataset`` names an empty
        one ``experiment_<i>``. A static field, so it is not a JAX array
        leaf and never reaches a compiled kernel as data.
    t0 : float | None
        Initial state time, added to the union axis as an unobserved row.
        None means ``y0`` is the state at the first observation.
    """

    covariates: dict[str, Array]
    y0: Float[Array, " S"]
    channels: dict[str, ChannelObs]
    exp_id: str = eqx.field(static=True)
    t0: float | None = eqx.field(static=True, default=None)


class BucketPayload(NamedTuple):
    """One bucket of experiments, stacked into rectangular arrays for JAX.

    A bucket holds ``N`` experiments that share the same union-timestamp
    length ``T``. The bucketing rule fixes only that length. Two
    experiments in the same bucket can still have different observation
    times and different masks.

    A ``NamedTuple`` rather than an ``eqx.Module`` because every field is a
    stacked JAX array with no methods to hang on it, and a ``NamedTuple`` is
    the lightest pytree container JAX already recognises.

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
        ``True`` where the corresponding ``y_observed`` cell came from a
        real ``ChannelObs`` entry, ``False`` where the union axis carries a
        time at which that channel was not measured. Every loss reads this
        to know which cells count.
    covariates : dict[str, Array]
        Per-key covariate stacked across the bucket. Same keys as on
        ``Experiment.covariates``, with an ``N`` axis added; scalar values
        have shape ``[N]`` and vectors have shape ``[N, K]``.
    y0 : Float[Array, "N S"]
        Per-experiment full initial state, stacked.
    n_obs : Int[Array, ""]
        Total observed-cell count for the bucket (``mask.sum()``).

        No shipped loss reads it, and none should: it counts across *all*
        channels, while every loss reduces over a selected subset and needs
        its own denominator. Kept because examples and smoke scripts assert
        dataset shape with it (R-D4).
    """

    ts: Float[Array, "N T"]
    y_observed: Float[Array, "N T D"]
    yvar: Float[Array, "N T D"]
    mask: Bool[Array, "N T D"]
    covariates: dict[str, Array]
    y0: Float[Array, "N S"]
    n_obs: Int[Array, ""]


class Dataset(eqx.Module):
    """All buckets of a dataset, as pure data.

    ``bucket_payloads`` is the dispatch list, one compiled kernel per bucket
    shape. The ``Dataset`` carries no model-shaped callables: it never sees
    full simulator states, and ``state_to_output`` belongs to the model, so
    prediction and training take it as a separate argument.

    Attributes
    ----------
    bucket_payloads : tuple[BucketPayload, ...]
        One ``BucketPayload`` per distinct union-axis length, ordered
        ascending by ``T``.
    output_channel_names : tuple[str, ...]
        Channel order along the trailing ``D`` axis of every payload.
        ``make_dataset`` scatters values in this same order.
    covariate_names : tuple[str, ...]
        Covariate keys, sorted. Matches each ``Experiment.covariates`` key
        set. Sorting makes dict iteration deterministic.
    _experiments : tuple[Experiment, ...]
        Source experiments in input order, read through the
        :attr:`experiments` property. Empty when a ``Dataset`` is built by
        hand from raw payloads, and ``split_dataset`` then raises.
    """

    bucket_payloads: tuple[BucketPayload, ...]
    output_channel_names: tuple[str, ...] = eqx.field(static=True)
    covariate_names: tuple[str, ...] = eqx.field(static=True)
    _experiments: tuple[Experiment, ...] = ()

    @property
    def experiments(self) -> tuple[Experiment, ...]:
        """The source experiments in the order given to ``make_dataset``.

        Every ``exp_id`` is nonempty and unique. Empty for a dataset built
        by hand from raw payloads.
        """
        return self._experiments


def make_experiment(
    *,
    channels: dict[str, ChannelObs],
    covariates: dict[str, float | Array] | None = None,
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None = None,
    exp_id: str = "",
    t0: float | None = None,
    y0: Any = None,
) -> Experiment:
    """Build one ``Experiment`` from raw covariates, channels, and a state-init hook.

    ``y0_fn`` builds the model's full starting state from the covariates
    (already JAX arrays) and the channels, returning ``Float[Array, "S"]``.
    Where the observed channels are the whole state, a typical hook is
    ``lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])``.
    Unobserved state components are constructed there too, a population
    moment initialised to zero being the common case.

    Parameters
    ----------
    channels
        Sparse observations keyed by channel name.
    covariates
        Scalar or rank-1 vector run conditions, constant in time. Values are
        converted to JAX arrays; a given key must have one consistent shape
        across a dataset. Omit when the model takes no run conditions.
    y0_fn
        Hook ``(covariates, channels) -> [S]`` building the full initial
        state, where ``S`` is the state dimension the user's ``simulate_fn``
        integrates.
    exp_id
        Optional human-readable id copied to ``Experiment.exp_id``.
    t0
        Initial state time, at or before the first observation. Added as an
        unobserved integration point. None preserves the first-observation
        convention.
    y0
        Direct initial state, as an alternative to y0_fn. Supply exactly one.
    """
    cov_arr: dict[str, Array] = {}
    for key, value in (covariates or {}).items():
        array = jnp.asarray(value)
        if array.ndim not in (0, 1):
            raise ValueError(
                f"Experiment covariate {key!r} must be scalar or rank-1; got shape {array.shape}"
            )
        cov_arr[key] = array
    if (y0 is None) == (y0_fn is None):
        raise ValueError("Pass exactly one of y0= or y0_fn=")
    if t0 is not None and not np.isfinite(t0):
        raise ValueError("Experiment t0 must be finite")
    if y0_fn is not None:
        y0 = jnp.asarray(y0_fn(cov_arr, channels))
    else:
        y0 = jnp.asarray(y0, dtype=jnp.result_type(jnp.asarray(y0).dtype, float))
    if y0.ndim != 1:
        raise ValueError(f"Experiment y0 must be rank-1; got shape {y0.shape}")
    return Experiment(covariates=cov_arr, y0=y0, channels=channels, exp_id=exp_id, t0=t0)


def experiment_from_array(
    *,
    ts: Any,
    values: Any,
    channel_names: Sequence[str],
    exp_id: str,
    variance: Any = 1.0,
    covariates: dict[str, float | Array] | None = None,
    t0: float | None = None,
    y0: Any = None,
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None = None,
) -> Experiment:
    """Build one ``Experiment`` from a shared time grid and a ``[T, D]`` table of values.

    For data that arrives as a matrix, one row per sample time and one
    column per channel. A ``NaN`` cell means "not measured at this time",
    so each channel keeps only its own finite rows and channels may still be
    sampled at different times.

    Parameters
    ----------
    ts
        Sample times ``[T]``, no repeats.
    values
        Observations ``[T, D]``, columns in ``channel_names`` order. A
        rank-1 ``[T]`` array is accepted when there is one channel.
    channel_names
        The ``D`` channel names.
    exp_id
        Identifier, required so results stay traceable to the source run.
    variance
        A scalar for every cell, a length-``D`` sequence with one value per
        channel, or a ``[T, D]`` array. Read only at observed cells.
    covariates, t0, y0, y0_fn
        As for :func:`make_experiment`. Supply exactly one of ``y0`` and
        ``y0_fn``.
    """
    names = tuple(channel_names)
    times = np.asarray(ts)
    table = np.asarray(values, dtype=np.result_type(np.asarray(values).dtype, np.float32))
    if table.ndim == 1 and len(names) == 1:
        table = table[:, None]
    if times.ndim != 1 or table.shape != (times.shape[0], len(names)):
        raise ValueError(
            f"experiment_from_array({exp_id!r}): values must have shape [T, D] = "
            f"({times.shape[0] if times.ndim == 1 else '?'}, {len(names)}) to match ts "
            f"{times.shape} and channel_names {names}; got {table.shape}"
        )
    var = np.asarray(variance, dtype=table.dtype)
    if var.ndim == 1:
        if var.shape[0] != len(names):
            raise ValueError(
                f"experiment_from_array({exp_id!r}): a rank-1 variance has one entry per "
                f"channel ({len(names)}); got {var.shape[0]}"
            )
        var = np.broadcast_to(var[None, :], table.shape)
    elif var.ndim == 0:
        var = np.broadcast_to(var, table.shape)
    elif var.shape != table.shape:
        raise ValueError(
            f"experiment_from_array({exp_id!r}): variance must be a scalar, one value per "
            f"channel, or shape {table.shape}; got {var.shape}"
        )
    channels = {}
    for d, name in enumerate(names):
        observed = ~np.isnan(table[:, d])
        try:
            channels[name] = ChannelObs(
                ts=times[observed], values=table[observed, d], variance=var[observed, d]
            )
        except ValueError as exc:
            raise ValueError(f"Experiment {exp_id!r}, channel {name!r}: {exc}") from exc
    return make_experiment(
        channels=channels, covariates=covariates, y0_fn=y0_fn, exp_id=exp_id, t0=t0, y0=y0
    )


def _column_map(spec: Sequence[str] | Mapping[str, str]) -> dict[str, str]:
    """``{name: column}`` from a list of columns (named after themselves) or a mapping."""
    if isinstance(spec, str):
        return {spec: spec}
    if isinstance(spec, Mapping):
        return dict(cast(Mapping[str, str], spec))
    return {column: column for column in spec}


def experiments_from_dataframe(
    df: Any,
    *,
    time: str,
    experiment: str,
    channels: Sequence[str] | Mapping[str, str],
    covariates: Sequence[str] | Mapping[str, str] = (),
    variance: float | Mapping[str, str | float] | None = None,
    t0: float | str | None = None,
    y0: Any = None,
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None = None,
) -> list[Experiment]:
    """Build one ``Experiment`` per run from a wide pandas table.

    The table has one row per (experiment, sample time) and one column per
    measured channel, which is how lab spreadsheets usually look. ``NaN``
    marks a channel that was not measured at that row, so channels may be
    sampled at different times. A long table (one row per measurement, with
    ``channel`` and ``value`` columns) becomes wide with
    ``df.pivot_table(index=["run", "time"], columns="channel", values="value").reset_index()``.

    Experiments come back in order of first appearance, each built by
    :func:`experiment_from_array`.

    Parameters
    ----------
    df
        A ``pandas.DataFrame``. pandas itself is never imported here.
    time
        Column of sample times.
    experiment
        Column identifying the run. Its values, as strings, become ``exp_id``.
    channels
        Channel columns. A list names each channel after its column; a
        mapping ``{channel: column}`` renames.
    covariates
        Covariate columns, in the same two forms. Each must be constant
        within a run.
    variance
        None for unit variance, one number for every cell, or a mapping
        ``{channel: column or number}``. Channels missing from the mapping
        get unit variance.
    t0
        None, one initial time for every run, or a column holding each
        run's initial time (constant within the run).
    y0, y0_fn
        Exactly one. ``y0`` is either one state shared by every run or a
        mapping ``{exp_id: state}``. ``y0_fn`` is the
        ``(covariates, channels) -> [S]`` hook of :func:`make_experiment`.

    Examples
    --------
    ::

        experiments = hm.experiments_from_dataframe(
            df.replace(-1.0, np.nan),
            time="Time",
            experiment="Exp_ID",
            channels={"conc": "Concentration", "d43": "PS"},
            covariates={"temperature": "Temperature", "loading": "Loading"},
            variance={"conc": "Concentration_var", "d43": "PS_var"},
            y0_fn=lambda cov, ch: jnp.array([ch["conc"].values[0], 0.0, 0.0]),
        )
    """
    channel_cols = _column_map(channels)
    covariate_cols = _column_map(covariates)
    variance_spec: dict[str, str | float]
    if variance is None:
        variance_spec = {}
    elif isinstance(variance, Mapping):
        variance_spec = dict(cast(Mapping[str, str | float], variance))
    else:
        variance_spec = dict.fromkeys(channel_cols, variance)
    unknown = sorted(set(variance_spec) - set(channel_cols))
    if unknown:
        raise ValueError(
            f"variance names unknown channels {unknown}; channels are {list(channel_cols)}"
        )
    needed = [time, experiment, *channel_cols.values(), *covariate_cols.values()]
    needed += [v for v in variance_spec.values() if isinstance(v, str)]
    if isinstance(t0, str):
        needed.append(t0)
    missing = sorted({c for c in needed if c not in df.columns})
    if missing:
        raise KeyError(f"Columns {missing} not in the table; available: {list(df.columns)}")

    def constant(group: Any, column: str, exp_id: str) -> Any:
        values = group[column].unique()
        if len(values) != 1:
            raise ValueError(
                f"Column {column!r} must be constant within experiment {exp_id!r}; "
                f"found {list(values)}"
            )
        return values[0]

    def channel_variance(group: Any, name: str) -> np.ndarray[Any, Any]:
        spec = variance_spec.get(name, 1.0)
        if isinstance(spec, str):
            return group[spec].to_numpy(dtype=float)
        return np.full(len(group), float(spec))

    out = []
    for key, group in df.groupby(experiment, sort=False):
        exp_id = str(key)
        variances = [channel_variance(group, name) for name in channel_cols]
        out.append(
            experiment_from_array(
                ts=group[time].to_numpy(dtype=float),
                values=group[list(channel_cols.values())].to_numpy(dtype=float),
                channel_names=tuple(channel_cols),
                variance=np.stack(variances, axis=1),
                covariates={
                    name: float(constant(group, column, exp_id))
                    for name, column in covariate_cols.items()
                },
                t0=float(constant(group, t0, exp_id)) if isinstance(t0, str) else t0,
                y0=y0[exp_id] if isinstance(y0, Mapping) else y0,
                y0_fn=y0_fn,
                exp_id=exp_id,
            )
        )
    return out


class _ExperimentArrays(NamedTuple):
    """One experiment's union-axis tensors, before any bucket stacking.

    A name for what was a bare 4-tuple threaded through ``make_dataset`` and
    indexed positionally at both ends. Holds no invariant of its own beyond
    the shared leading ``T``, which is exactly what ``make_dataset`` buckets
    on.

    Attributes
    ----------
    ts : Float[Array, "T"]
        Sorted union timestamp axis.
    y_observed : Float[Array, "T D"]
        Channel values scattered onto ``ts``; ``0.0`` at unobserved cells.
    yvar : Float[Array, "T D"]
        Per-cell variance; ``1.0`` at unobserved cells (read only via mask).
    mask : Bool[Array, "T D"]
        ``True`` iff cell ``[t, d]`` came from a real ``ChannelObs`` entry.
    """

    ts: Float[Array, " T"]
    y_observed: Float[Array, "T D"]
    yvar: Float[Array, "T D"]
    mask: Bool[Array, "T D"]


def _per_experiment_arrays(
    experiment: Experiment,
    output_channel_names: tuple[str, ...],
) -> _ExperimentArrays:
    """Build the per-experiment union-axis tensors ``(ts, y_observed, yvar, mask)``.

    Computes ``T = len(union(ts_c) for c in output_channel_names)``,
    allocates ``[T]`` and ``[T, D]`` host buffers, then scatters each
    channel's ``(values, variance)`` into the rows matching its ``ts``.

    Uses ``numpy`` rather than ``jnp``: this runs once at data import,
    outside any compiled region, and staying on ``numpy`` keeps the
    set-membership cheap and avoids promoting the user's dtypes.

    Returns
    -------
    _ExperimentArrays
        See that class for the per-field shapes.
    """
    # One pass caches each channel's host-side arrays and accumulates dtypes.
    # The union pass below reuses the cached arrays instead of re-running
    # np.asarray on the eqx-leaf data.
    channel_arrays: list[
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ] = []
    union_ts: set[float] = set()
    val_dtypes: list[np.dtype[Any]] = []
    var_dtypes: list[np.dtype[Any]] = []
    ts_dtypes: list[np.dtype[Any]] = []
    for ch_name in output_channel_names:
        ch = experiment.channels[ch_name]
        ts_np = np.asarray(ch.ts)
        values_np = np.asarray(ch.values)
        variance_np = np.asarray(ch.variance)
        channel_arrays.append((ts_np, values_np, variance_np))
        ts_dtypes.append(ts_np.dtype)
        val_dtypes.append(values_np.dtype)
        var_dtypes.append(variance_np.dtype)

    # A decimal timestamp such as 0.1 has different exact binary values in
    # float32 and float64. Quantize keys to the least precise floating dtype
    # present so the same physical timestamp is not split into two union rows
    # merely because channels used different dtypes. Storage still uses the
    # result dtype selected below.
    float_ts_dtypes = [dtype for dtype in ts_dtypes if dtype.kind == "f"]
    if float_ts_dtypes:
        key_dtype = np.dtype(f"float{min(dtype.itemsize for dtype in float_ts_dtypes) * 8}")
    else:
        key_dtype = (
            np.result_type(*ts_dtypes, np.float32)
            if experiment.t0 is not None
            else np.result_type(*ts_dtypes)
            if ts_dtypes
            else np.dtype(np.float32)
        )
    for ts_np, _values_np, _variance_np in channel_arrays:
        union_ts.update(float(np.asarray(t, dtype=key_dtype)) for t in ts_np.tolist())

    if experiment.t0 is not None:
        start = float(np.asarray(experiment.t0, dtype=key_dtype))
        if union_ts and start > min(union_ts):
            raise ValueError(f"Experiment {experiment.exp_id!r}: t0 must precede observations")
        union_ts.add(start)
    sorted_ts = sorted(union_ts)
    ts_to_idx = {t: i for i, t in enumerate(sorted_ts)}

    T = len(sorted_ts)
    D = len(output_channel_names)
    val_dtype = np.result_type(*val_dtypes) if val_dtypes else np.dtype(np.float32)
    var_dtype = np.result_type(*var_dtypes) if var_dtypes else np.dtype(np.float32)
    ts_dtype = (
        np.result_type(*ts_dtypes, np.float32)
        if experiment.t0 is not None
        else np.result_type(*ts_dtypes)
        if ts_dtypes
        else np.dtype(np.float32)
    )

    y_observed = np.zeros((T, D), dtype=val_dtype)
    yvar = np.ones((T, D), dtype=var_dtype)
    mask = np.zeros((T, D), dtype=bool)

    for d, (ts_np, values_np, variance_np) in enumerate(channel_arrays):
        # A channel with *empty* values is a probe: its timestamps still
        # define where this experiment is integrated, but no cell is
        # marked observed, so the loss scores nothing there. This is how
        # "conditions to simulate with no y_true" (trajectory-penalty
        # probes) are expressed.
        if values_np.shape[0] == 0:
            continue
        for t, v, var in zip(ts_np.tolist(), values_np.tolist(), variance_np.tolist(), strict=True):
            idx = ts_to_idx[float(np.asarray(t, dtype=key_dtype))]
            y_observed[idx, d] = v
            yvar[idx, d] = var
            mask[idx, d] = True

    return _ExperimentArrays(
        ts=jnp.asarray(np.asarray(sorted_ts, dtype=ts_dtype)),
        y_observed=jnp.asarray(y_observed),
        yvar=jnp.asarray(yvar),
        mask=jnp.asarray(mask),
    )


def _renamed(experiment: Experiment, exp_id: str) -> Experiment:
    """Copy of ``experiment`` under a new ID.

    ``exp_id`` is a static field, which ``eqx.tree_at`` cannot replace, so
    the experiment is rebuilt from its fields.
    """
    return Experiment(
        covariates=experiment.covariates,
        y0=experiment.y0,
        channels=experiment.channels,
        exp_id=exp_id,
        t0=experiment.t0,
    )


def _named_experiments(experiments: Sequence[Experiment]) -> tuple[Experiment, ...]:
    """Name empty IDs ``experiment_<i>`` and refuse duplicates.

    Naming happens here, once, so every consumer of a dataset (prediction
    results, metrics, data frames) sees the same IDs.
    """
    named = tuple(
        exp if exp.exp_id else _renamed(exp, f"experiment_{i}") for i, exp in enumerate(experiments)
    )
    counts = Counter(exp.exp_id for exp in named)
    duplicates = sorted(exp_id for exp_id, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"make_dataset: duplicate exp_id values {duplicates}; IDs must be unique")
    return named


def _validate_experiments(
    experiments: Sequence[Experiment],
    output_channel_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Check the experiments agree with each other; return the covariate keys.

    Every experiment must carry the same covariate key set, because those keys
    become the stacked ``covariates`` dict and a missing one has no value to
    stack. Each must also define every requested output channel, because the
    trailing ``D`` axis is positional.

    Raising here, naming the offending ``exp_id``, is the point: both
    mismatches would otherwise surface much later as a shape error inside a
    ``jnp.stack``, with nothing to say which experiment caused it.
    """
    cov_keys = tuple(sorted(experiments[0].covariates.keys()))
    cov_shapes = {key: tuple(jnp.asarray(experiments[0].covariates[key]).shape) for key in cov_keys}
    for exp in experiments:
        exp_keys = tuple(sorted(exp.covariates.keys()))
        if exp_keys != cov_keys:
            raise ValueError(
                f"Inconsistent covariate keys: experiment {exp.exp_id!r} has "
                f"{list(exp_keys)}, expected {list(cov_keys)}"
            )
        for key in cov_keys:
            shape = tuple(jnp.asarray(exp.covariates[key]).shape)
            if shape != cov_shapes[key]:
                raise ValueError(
                    f"Inconsistent covariate shape for {key!r}: experiment "
                    f"{exp.exp_id!r} has {shape}, expected {cov_shapes[key]}"
                )
        missing = [c for c in output_channel_names if c not in exp.channels]
        if missing:
            raise ValueError(
                f"Experiment {exp.exp_id!r} missing requested output channels: {missing}"
            )
    return cov_keys


def _stack_bucket(
    items: list[tuple[_ExperimentArrays, Experiment]],
    cov_keys: tuple[str, ...],
) -> BucketPayload:
    """Stack same-``T`` experiments along a new leading ``N`` axis.

    Everything in ``items`` already shares its leading ``T``, which is what
    makes one ``jnp.stack`` per field legal and is the whole reason bucketing
    keys on ``T``.
    """
    arrays = [a for a, _exp in items]
    experiments = [exp for _a, exp in items]
    mask = jnp.stack([a.mask for a in arrays], axis=0)
    return BucketPayload(
        ts=jnp.stack([a.ts for a in arrays], axis=0),
        y_observed=jnp.stack([a.y_observed for a in arrays], axis=0),
        yvar=jnp.stack([a.yvar for a in arrays], axis=0),
        mask=mask,
        covariates={
            k: jnp.stack([jnp.asarray(exp.covariates[k]) for exp in experiments], axis=0)
            for k in cov_keys
        },
        y0=jnp.stack([exp.y0 for exp in experiments], axis=0),
        n_obs=mask.sum().astype(jnp.int32),
    )


def describe_buckets(dataset: Dataset) -> str:
    """One line per bucket: experiments, length, and how full the mask is.

    A quick human-readable census of the bucketed-irregular structure, for
    debugging and example output. Each line reports the bucket's ``N``
    (experiments), ``T`` (union timestamp axis), ``D`` (channels), and the
    fraction of ``[N, T, D]`` cells the mask marks as real observations.
    """
    lines = [f"{len(dataset.bucket_payloads)} bucket(s)"]
    for i, bp in enumerate(dataset.bucket_payloads):
        n, t, d = bp.y_observed.shape
        lines.append(
            f"  bucket {i}: N={n:2d} experiments, T={t:3d} timestamps, "
            f"D={d} channels, mask {float(bp.mask.mean()):.2f} full"
        )
    return "\n".join(lines)


def make_dataset(
    experiments: Sequence[Experiment],
    *,
    output_channel_names: tuple[str, ...] | list[str],
) -> Dataset:
    """Bucket and stack ``experiments`` into a JAX-traceable ``Dataset``.

    Three steps run in order.

    1. Validation. Experiment IDs must be unique; an empty one is named
       ``experiment_<i>`` after its position. All experiments must agree on
       the set of covariate keys, and each must define every requested
       output channel. A mismatch raises at once, naming the offending
       ``exp_id``.
    2. Per-experiment scattering. ``_per_experiment_arrays`` builds each
       experiment's union timestamp axis and its ``[T, D]`` observation,
       variance, and mask tensors.
    3. Bucketing. Experiments are grouped by their union-axis length ``T``,
       and each group is stacked along a new leading ``N`` axis into one
       ``BucketPayload``. Buckets come out in ascending ``T`` order.

    The ``Dataset`` is pure data: ``state_to_output``, being a property of
    the model, is passed to prediction and training separately.

    Parameters
    ----------
    experiments
        Non-empty sequence of ``Experiment`` objects, usually built with
        ``make_experiment``.
    output_channel_names
        Channel order for the trailing ``D`` axis. Coerced to a tuple before
        being stored statically on the ``Dataset``.

    Returns
    -------
    Dataset
        ``bucket_payloads`` ordered ascending by ``T``, with the (named)
        experiments kept on ``Dataset.experiments``.
    """
    if not experiments:
        raise ValueError("make_dataset requires at least one experiment")
    experiments = _named_experiments(experiments)
    output_channel_names = tuple(output_channel_names)
    if not output_channel_names:
        raise ValueError("make_dataset requires at least one output channel")
    if len(set(output_channel_names)) != len(output_channel_names):
        raise ValueError("output_channel_names must not contain duplicates")
    cov_keys = _validate_experiments(experiments, output_channel_names)

    by_len: dict[int, list[tuple[_ExperimentArrays, Experiment]]] = defaultdict(list)
    for exp in experiments:
        arrays = _per_experiment_arrays(exp, output_channel_names)
        if arrays.ts.shape[0] == 0:
            raise ValueError(
                f"Experiment {exp.exp_id!r} has no timestamps across the requested channels"
            )
        by_len[arrays.ts.shape[0]].append((arrays, exp))

    # Ascending T, so bucket order is a deterministic function of the data
    # rather than of dict insertion order.
    bucket_payloads = tuple(_stack_bucket(by_len[T], cov_keys) for T in sorted(by_len))

    return Dataset(
        bucket_payloads=bucket_payloads,
        output_channel_names=output_channel_names,
        covariate_names=cov_keys,
        _experiments=tuple(experiments),
    )


def _validate_fractions(train: float, val: float, test: float) -> None:
    """Reject split fractions that are out of range or do not sum to one.

    Called before the source-experiments check, so a caller who gets both
    wrong hears about the fractions first.
    """
    if any(f < 0.0 or f > 1.0 for f in (train, val, test)):
        raise ValueError("train, val, and test fractions must each be in [0, 1]")
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"train + val + test fractions must sum to 1.0, got {total}")


def _split_counts(n: int, train: float, val: float, test: float) -> tuple[int, int, int]:
    """Experiment counts per split, preserving explicitly empty splits.

    For the normal three-way split, train and val floor and test takes the
    remainder. When ``test == 0.0``, test is forced to stay empty and any
    rounding remainder is assigned to the larger fractional remainder of
    train or val.

    A positive fraction that floors to zero is refused. An empty split reads
    downstream as "no validation needed" rather than "lost to rounding", so a
    caller who means it has to ask for ``0.0`` explicitly.
    """
    n_train = int(np.floor(train * n))
    n_val = int(np.floor(val * n))
    if test == 0.0:
        remainder = n - n_train - n_val
        if remainder:
            fractional_train = train * n - n_train
            fractional_val = val * n - n_val
            if fractional_train >= fractional_val:
                n_train += remainder
            else:
                n_val += remainder
        n_test = 0
    else:
        n_test = n - n_train - n_val

    for name, frac, count in (
        ("train", train, n_train),
        ("val", val, n_val),
        ("test", test, n_test),
    ):
        if frac > 0.0 and count == 0:
            raise ValueError(
                f"split_dataset: {name} fraction {frac} produced 0 experiments out of "
                f"n={n} after floor rounding. Either increase n, raise the {name} "
                f"fraction, or set {name}=0.0 explicitly to skip this split."
            )
    return n_train, n_val, n_test


def _subset(
    dataset: Dataset,
    experiments: tuple[Experiment, ...],
    idxs: Sequence[int],
) -> Dataset:
    """Re-bucket the experiments at ``idxs``, or hand back an empty ``Dataset``.

    Bucketed from scratch rather than sliced out of the parent's buckets, so
    each split's bucket structure suits its own contents.

    An empty split carries no payloads and no experiments, which is what
    makes it unsplittable again rather than silently splittable into nothing.
    """
    if not idxs:
        return Dataset(
            bucket_payloads=(),
            output_channel_names=dataset.output_channel_names,
            covariate_names=dataset.covariate_names,
        )
    return make_dataset(
        [experiments[int(i)] for i in idxs],
        output_channel_names=dataset.output_channel_names,
    )


def make_bootstrap_dataset(
    dataset: Dataset,
    *,
    key: Array,
    n_experiments: int | None = None,
) -> Dataset:
    """Bootstrap resample the dataset's experiments, re-bucketing the result.

    Draws ``n_experiments`` experiments **with replacement** from the
    source (default: as many as the source holds), then re-buckets via
    :func:`make_dataset`. Irregular per-channel timestamps are handled
    automatically: re-bucketing regroups by union-axis length, and a
    duplicated experiment simply contributes more ``N`` rows to its bucket.
    Repeats are renamed ``<exp_id>#2``, ``<exp_id>#3`` and so on, since IDs
    within a dataset must be unique.

    This is the data half of a bagging ensemble: each call yields one
    resampled dataset, and training on several of them produces an
    ensemble whose members saw different resamples.

    Parameters
    ----------
    dataset
        Source dataset. Must carry its experiments (built by
        ``make_dataset``), or this raises.
    key
        Required ``jr.PRNGKey`` for the resample, never defaulted.
    n_experiments
        Number of experiments to draw. Defaults to the source size.
        Must be at least 1.

    Returns
    -------
    Dataset
        A new dataset of ``n_experiments`` experiments (some duplicated),
        re-bucketed from scratch.
    """
    experiments = dataset.experiments
    if not experiments:
        raise ValueError(
            "make_bootstrap_dataset requires the source experiments; the provided "
            "Dataset has none (was it built by hand rather than with make_dataset?)"
        )
    n = len(experiments)
    if n_experiments is None:
        n_experiments = n
    if n_experiments < 1:
        raise ValueError(f"make_bootstrap_dataset: n_experiments must be >= 1, got {n_experiments}")
    idxs = np.asarray(jr.choice(key, n, shape=(n_experiments,), replace=True)).tolist()
    # IDs must stay unique, so the k-th repeat of an experiment is renamed
    # ``<exp_id>#<k>``; the first draw keeps its ID.
    draws: dict[str, int] = defaultdict(int)
    resampled = []
    for i in idxs:
        exp = experiments[int(i)]
        draws[exp.exp_id] += 1
        k = draws[exp.exp_id]
        resampled.append(exp if k == 1 else _renamed(exp, f"{exp.exp_id}#{k}"))
    return make_dataset(resampled, output_channel_names=dataset.output_channel_names)


def split_dataset(
    dataset: Dataset,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    key: Array,
) -> tuple[Dataset, Dataset, Dataset]:
    """Permute experiments and re-bucket each split independently.

    Splitting happens at the ``Experiment`` level and each split is bucketed
    from scratch, so its bucket structure suits its own contents rather than
    the original bucket boundaries.

    Counts use ``floor(train*n)`` and ``floor(val*n)``, with test taking the
    remainder so the sizes sum to ``n``. An empty split comes back with no
    payloads and no experiments, so it cannot be split again.

    Parameters
    ----------
    dataset
        Source dataset. Must carry its experiments, or this raises.
    train, val, test
        Fractions in ``[0, 1]`` summing to ``1.0`` (within ``np.isclose``).
    key
        Required ``jr.PRNGKey`` for the permutation, never defaulted, so
        reproducibility does not rest on a hidden global.

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        ``(train_dataset, val_dataset, test_dataset)``.
    """
    _validate_fractions(train, val, test)

    experiments = dataset.experiments
    if not experiments:
        raise ValueError(
            "split_dataset requires the source experiments; the provided Dataset has none "
            "(was it built by hand rather than with make_dataset?)"
        )

    n = len(experiments)
    n_train, n_val, n_test = _split_counts(n, train, val, test)
    perm = np.asarray(jr.permutation(key, n)).tolist()
    splits_idx = (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val : n_train + n_val + n_test],
    )
    return cast(
        tuple[Dataset, Dataset, Dataset],
        tuple(_subset(dataset, experiments, idxs) for idxs in splits_idx),
    )
