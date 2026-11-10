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
no padding unless ``make_dataset(max_buckets=)`` asks for fewer. Other
layouts (regular grids, single-channel, ragged) are not modelled here;
pre-process such data into this shape.

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

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int


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
        Per-observation variance used by ``masked_mle`` / ``per_experiment_mle``. A
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
    observed : Float[Array, "N T D"]
        Channel observations scattered onto ``ts``. Cells where the channel
        was not observed at that timestamp hold ``0.0``; consumers must read
        ``mask`` to know which entries are real.
    variance : Float[Array, "N T D"]
        Per-observation variance (used by MLE losses). Defaults to ``1.0``
        at unobserved cells so masked positions never divide by zero.
    mask : Bool[Array, "N T D"]
        ``True`` where the corresponding ``observed`` cell came from a
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
    observed: Float[Array, "N T D"]
    variance: Float[Array, "N T D"]
    mask: Bool[Array, "N T D"]
    covariates: dict[str, Array]
    y0: Float[Array, "N S"]
    n_obs: Int[Array, ""]


def row_horizons(ts: Array, observed_rows: Array) -> Array:
    """The smallest ``horizon`` at which each row is scored, ``[N, T]``.

    ``ts`` is ``[N, T]`` and ``observed_rows`` is ``[N, T]``, true where any
    channel was measured. An experiment's timeline runs from its first row
    (``t0`` when set, else the earliest time of any output channel) to its
    last measurement,
    and a row's horizon is how far along that span it lies, from 0 to 1.
    Each experiment is measured against its own span, so one long
    experiment in a bucket does not shift the cutoff of the others. A ``t0``
    row starts the span; rows added by ``max_buckets`` padding hold no
    measurement and do not move it.

    Rows up to the first measurement get 0, so every horizon scores it:
    otherwise an experiment that starts at ``t0`` well before its first
    measurement would score nothing at a small horizon, and train nothing.
    The last measurement gets exactly 1, with no rounding at the edge, and so
    does any row after it. "Measured" here means in any channel, whatever
    the loss selects.
    """
    start = ts[:, :1]
    last = jnp.max(jnp.where(observed_rows, ts, -jnp.inf), axis=1, keepdims=True)
    span = last - start
    has_span = span > 0
    position = jnp.where(has_span, (ts - start) / jnp.where(has_span, span, 1.0), 0.0)
    first = jnp.min(jnp.where(observed_rows, position, jnp.inf), axis=1, keepdims=True)
    # A row after the last measurement, such as a probe channel's time, holds
    # nothing to score, but a multiple-shooting join there must still count
    # at the full horizon.
    return jnp.where(position <= first, 0.0, jnp.minimum(position, 1.0))


def horizon_rows(ts: Array, observed_rows: Array, horizon: ArrayLike) -> Array:
    """Which rows fall in the first ``horizon`` of their experiment's timeline.

    ``[N, T]`` booleans; see :func:`row_horizons`. ``horizon`` may be traced.
    """
    horizons = row_horizons(ts, observed_rows)
    return horizons <= jnp.asarray(horizon, dtype=horizons.dtype)


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
    _locations : tuple[tuple[int, int], ...]
        ``(bucket, row)`` of each source experiment, in the same order.
    _max_buckets : int | None
        The limit given to ``make_dataset``, kept so that a split or a
        resample applies it again to its own experiments.
    """

    bucket_payloads: tuple[BucketPayload, ...]
    output_channel_names: tuple[str, ...] = eqx.field(static=True)
    covariate_names: tuple[str, ...] = eqx.field(static=True)
    _experiments: tuple[Experiment, ...] = ()
    _locations: tuple[tuple[int, int], ...] = eqx.field(static=True, default=())
    _max_buckets: int | None = eqx.field(static=True, default=None)

    @property
    def experiments(self) -> tuple[Experiment, ...]:
        """The source experiments in the order given to ``make_dataset``.

        Every ``exp_id`` is nonempty and unique. Empty for a dataset built
        by hand from raw payloads.
        """
        return self._experiments


Data = Dataset | Sequence[Experiment]
"""What the model-level entry points accept as data: a ``Dataset`` or a list of experiments.

``HybridModel.fit``, ``fit_ensemble``, ``predict``, ``predict_ensemble``
and the trainers in :mod:`jaxhybridmodels.training` all take either. A list
goes through :func:`make_dataset` with the model's ``output_channel_names``.
"""


def make_experiment(
    *,
    channels: dict[str, ChannelObs],
    covariates: dict[str, float | Array] | None = None,
    exp_id: str = "",
    t0: float | None = None,
    y0: Any = None,
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None = None,
) -> Experiment:
    """Build one ``Experiment`` from raw covariates, channels, and a state-init hook.

    ``y0_fn`` builds the model's full starting state from the covariates
    (already JAX arrays) and the channels, returning ``Float[Array, "S"]``.
    Where the observed channels are the whole state, a typical hook is
    ``lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])``.
    The state belongs to the experiment's start: ``t0`` when set, else the
    earliest time of any channel the model outputs. ``values[0]`` is that
    state only when the channel's ``ts`` is sorted and the channel was
    measured at that start.
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
        ``make_dataset`` names an experiment that has none.
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
    variance: Any = 1.0,
    covariates: dict[str, float | Array] | None = None,
    exp_id: str = "",
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
    variance
        A scalar for every cell, a length-``D`` sequence with one value per
        channel, or a ``[T, D]`` array. Read only at observed cells.
    covariates, exp_id, t0, y0, y0_fn
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
    sampled at different times. For a table with one row per measurement, use
    :func:`experiments_from_long_dataframe`.

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
        A missing value raises ``ValueError``.
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
    _check_table(df, experiment, needed)

    def channel_variance(group: Any, name: str) -> np.ndarray[Any, Any]:
        spec = variance_spec.get(name, 1.0)
        if isinstance(spec, str):
            return group[spec].to_numpy(dtype=float)
        return np.full(len(group), float(spec))

    out = []
    for key, unsorted in df.groupby(experiment, sort=False):
        exp_id = str(key)
        # In time order, as the long loader gives them, so a y0_fn reading
        # ``values[0]`` gets each channel's earliest measurement.
        group = unsorted.sort_values(time, kind="stable")
        variances = [channel_variance(group, name) for name in channel_cols]
        out.append(
            experiment_from_array(
                ts=group[time].to_numpy(dtype=float),
                values=group[list(channel_cols.values())].to_numpy(dtype=float),
                channel_names=tuple(channel_cols),
                variance=np.stack(variances, axis=1),
                **_run_options(group, exp_id, covariate_cols, t0, y0, y0_fn),
            )
        )
    return out


def experiments_from_long_dataframe(
    df: Any,
    *,
    time: str,
    experiment: str,
    channel: str,
    value: str,
    variance: float | str | None = None,
    covariates: Sequence[str] | Mapping[str, str] = (),
    t0: float | str | None = None,
    y0: Any = None,
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None = None,
) -> list[Experiment]:
    """Build one ``Experiment`` per run from a long pandas table.

    The table has one row per measurement, naming its run, channel, time and
    value, so each channel keeps its own sample times. An experiment that
    never measured a channel gets it empty, so experiments still share their
    channels. Experiments come back in order of first appearance.

    Parameters
    ----------
    df
        A ``pandas.DataFrame``. pandas itself is never imported here.
    time, experiment, channel, value
        Columns holding the sample time, the run (its values, as strings,
        become ``exp_id``), the channel name and the measurement. A missing
        entry in any of them raises ``ValueError``, as does a repeated
        (experiment, channel, time).
    variance
        None for unit variance, one number for every cell, or a column.
    covariates, t0, y0, y0_fn
        As for :func:`experiments_from_dataframe`.

    Examples
    --------
    ::

        experiments = hm.experiments_from_long_dataframe(
            df,
            time="time",
            experiment="experiment_id",
            channel="channel",
            value="value",
            variance="variance",
            covariates=["temperature_C"],
            y0_fn=lambda covariates, channels: jnp.array([1.0]),
        )
    """
    covariate_cols = _column_map(covariates)
    needed = [time, experiment, channel, value, *covariate_cols.values()]
    if isinstance(variance, str):
        needed.append(variance)
    if isinstance(t0, str):
        needed.append(t0)
    _check_table(df, experiment, needed)
    for column in (time, channel, value):
        gaps = int(df[column].isna().sum())
        if gaps:
            raise ValueError(f"Column {column!r} has {gaps} missing value(s); every row needs one")

    names = [str(name) for name in df[channel].unique()]
    out = []
    for key, group in df.groupby(experiment, sort=False):
        exp_id = str(key)
        channels: dict[str, ChannelObs] = {}
        for name in names:
            rows = group[group[channel].astype(str) == name].sort_values(time, kind="stable")
            ts = rows[time].to_numpy(dtype=float)
            repeated = ts[1:][np.diff(ts) == 0]
            if repeated.size:
                raise ValueError(
                    f"Experiment {exp_id!r}, channel {name!r}: more than one row at time "
                    f"{float(repeated[0])}"
                )
            if isinstance(variance, str):
                cell_variance: Any = rows[variance].to_numpy(dtype=float)
            else:
                cell_variance = 1.0 if variance is None else float(variance)
            try:
                channels[name] = ChannelObs(
                    ts=ts, values=rows[value].to_numpy(dtype=float), variance=cell_variance
                )
            except ValueError as exc:
                raise ValueError(f"Experiment {exp_id!r}, channel {name!r}: {exc}") from exc
        out.append(
            make_experiment(
                channels=channels, **_run_options(group, exp_id, covariate_cols, t0, y0, y0_fn)
            )
        )
    return out


def _check_table(df: Any, experiment: str, needed: Sequence[str]) -> None:
    """Refuse a table missing a needed column, or a row with no experiment ID."""
    missing = sorted({c for c in needed if c not in df.columns})
    if missing:
        raise KeyError(f"Columns {missing} not in the table; available: {list(df.columns)}")
    missing_ids = df[experiment].isna()
    if missing_ids.any():
        # groupby would drop these rows silently, and a NaN in an integer
        # column turns every ID into a float ("1.0").
        raise ValueError(
            f"Column {experiment!r} has {int(missing_ids.sum())} missing experiment ID(s); "
            "every row needs one"
        )


def _run_options(
    group: Any,
    exp_id: str,
    covariate_cols: Mapping[str, str],
    t0: float | str | None,
    y0: Any,
    y0_fn: Callable[[dict[str, Array], dict[str, ChannelObs]], Array] | None,
) -> dict[str, Any]:
    """The per-run arguments both table loaders pass to an experiment builder."""

    def constant(column: str) -> Any:
        values = group[column].unique()
        if len(values) != 1:
            raise ValueError(
                f"Column {column!r} must be constant within experiment {exp_id!r}; "
                f"found {list(values)}"
            )
        value = values[0]
        # NaN of any float width, pandas' NA and None all count as missing.
        try:
            missing = value is None or bool(value != value)
        except TypeError:
            missing = True
        if missing:
            raise ValueError(f"Column {column!r} is missing (NaN) for experiment {exp_id!r}")
        return value

    return {
        "covariates": {name: float(constant(column)) for name, column in covariate_cols.items()},
        "t0": float(constant(t0)) if isinstance(t0, str) else t0,
        "y0": y0[exp_id] if isinstance(y0, Mapping) else y0,
        "y0_fn": y0_fn,
        "exp_id": exp_id,
    }


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
    observed : Float[Array, "T D"]
        Channel values scattered onto ``ts``; ``0.0`` at unobserved cells.
    variance : Float[Array, "T D"]
        Per-cell variance; ``1.0`` at unobserved cells (read only via mask).
    mask : Bool[Array, "T D"]
        ``True`` iff cell ``[t, d]`` came from a real ``ChannelObs`` entry.
    """

    ts: Float[Array, " T"]
    observed: Float[Array, "T D"]
    variance: Float[Array, "T D"]
    mask: Bool[Array, "T D"]


def _per_experiment_arrays(
    experiment: Experiment,
    output_channel_names: tuple[str, ...],
) -> _ExperimentArrays:
    """Build the per-experiment union-axis tensors ``(ts, observed, variance, mask)``.

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
        key_dtype = np.result_type(*ts_dtypes, np.float32)
    for ch_name, (ts_np, _values_np, _variance_np) in zip(
        output_channel_names, channel_arrays, strict=True
    ):
        keys = {float(np.asarray(t, dtype=key_dtype)) for t in ts_np.tolist()}
        if len(keys) < len(ts_np):
            # Two distinct times of one channel round onto one key, so one
            # measurement would silently overwrite the other.
            raise ValueError(
                f"Experiment {experiment.exp_id!r}: channel {ch_name!r} has times that "
                f"coincide in {key_dtype}, the least precise time dtype among its channels. "
                "Give every channel's ts the same dtype"
            )
        union_ts.update(keys)

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
    # Always floating: integer measurement times would otherwise floor every
    # float grid a caller predicts on.
    ts_dtype = np.result_type(*ts_dtypes) if ts_dtypes else np.dtype(np.float32)
    if ts_dtype.kind != "f":
        ts_dtype = np.result_type(ts_dtype, np.float32)

    observed = np.zeros((T, D), dtype=val_dtype)
    variance = np.ones((T, D), dtype=var_dtype)
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
            observed[idx, d] = v
            variance[idx, d] = var
            mask[idx, d] = True

    return _ExperimentArrays(
        ts=jnp.asarray(np.asarray(sorted_ts, dtype=ts_dtype)),
        observed=jnp.asarray(observed),
        variance=jnp.asarray(variance),
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
    results, metrics, data frames) sees the same IDs. A generated name that
    an explicit ID already uses gets a ``_<n>`` suffix instead, so only IDs
    the caller wrote can be reported as duplicates.
    """
    taken = {exp.exp_id for exp in experiments if exp.exp_id}
    named_list = []
    for i, exp in enumerate(experiments):
        if exp.exp_id:
            named_list.append(exp)
            continue
        name, n = f"experiment_{i}", 1
        while name in taken:
            n += 1
            name = f"experiment_{i}_{n}"
        taken.add(name)
        named_list.append(_renamed(exp, name))
    named = tuple(named_list)
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


def _choose_bucket_lengths(lengths: Sequence[int], max_buckets: int) -> tuple[int, ...]:
    """Pick at most ``max_buckets`` target lengths, minimising the padded size.

    A bucket pads each experiment up to its longest, so a bucket's cost is its
    experiment count times that length, the number of solver outputs per step.
    Experiments in a bucket are always a contiguous run of the sorted lengths,
    because swapping two across a boundary can only add padding. That leaves
    a small dynamic program over where the boundaries go. It is exact and
    deterministic, which a greedy merge of the closest lengths would not be.
    """
    counts = Counter(lengths)
    distinct = sorted(counts)
    if len(distinct) <= max_buckets:
        return tuple(distinct)
    # Cumulative experiment counts, so a run's size is a difference of two.
    totals = [0, *np.cumsum([counts[length] for length in distinct]).tolist()]
    n = len(distinct)

    # best[i] is the lowest cost of covering the first i lengths with the
    # buckets placed so far; cuts[b][i] is where the last of those buckets starts.
    best = [totals[i] * distinct[i - 1] if i else 0 for i in range(n + 1)]
    cuts: list[list[int]] = [[0] * (n + 1)]
    for _ in range(1, max_buckets):
        previous, best = best, list(best)
        row = list(cuts[-1])
        for i in range(1, n + 1):
            row[i] = i  # No new bucket ends at i until a cheaper split is found.
            for start in range(1, i):
                cost = previous[start] + (totals[i] - totals[start]) * distinct[i - 1]
                if cost < best[i]:
                    best[i], row[i] = cost, start
        cuts.append(row)

    chosen = []
    end = n
    for row in reversed(cuts):
        chosen.append(distinct[end - 1])
        end = row[end]
        if end == 0:
            break
    # A row that added no bucket at this boundary repeats the same length.
    return tuple(sorted(set(chosen)))


def _pad_to_length(arrays: _ExperimentArrays, target: int, exp_id: str) -> _ExperimentArrays:
    """Add unobserved rows at midpoints of the largest gaps until ``target`` rows.

    Midpoints rather than repeats of the last time, so ``ts`` stays strictly
    increasing and a ``simulate_fn`` stepping exactly through it keeps working,
    and rather than later times, so the integration span does not change. The
    new rows are masked, so no loss scores them.
    """
    length = arrays.ts.shape[0]
    if length == target:
        return arrays
    original = np.asarray(arrays.ts)
    # An integer time axis has no midpoints.
    float_dtype = original.dtype if original.dtype.kind == "f" else jnp.result_type(float)
    times = original.astype(float_dtype)
    for _ in range(target - length):
        gap = int(np.argmax(np.diff(times)))
        midpoint = times[gap] + (times[gap + 1] - times[gap]) / 2
        if not times[gap] < midpoint < times[gap + 1]:
            raise ValueError(
                f"Experiment {exp_id!r}: its times are too close together to pad to length "
                f"{target}; use more buckets, or x64"
            )
        times = np.insert(times, gap + 1, midpoint)
    rows = np.searchsorted(times, original.astype(times.dtype))
    width = arrays.observed.shape[1]
    observed = np.zeros((target, width), dtype=arrays.observed.dtype)
    variance = np.ones((target, width), dtype=arrays.variance.dtype)
    mask = np.zeros((target, width), dtype=bool)
    observed[rows] = np.asarray(arrays.observed)
    variance[rows] = np.asarray(arrays.variance)
    mask[rows] = np.asarray(arrays.mask)
    return _ExperimentArrays(
        ts=jnp.asarray(times),
        observed=jnp.asarray(observed),
        variance=jnp.asarray(variance),
        mask=jnp.asarray(mask),
    )


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
        observed=jnp.stack([a.observed for a in arrays], axis=0),
        variance=jnp.stack([a.variance for a in arrays], axis=0),
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
        n, t, d = bp.observed.shape
        lines.append(
            f"  bucket {i}: N={n:2d} experiments, T={t:3d} timestamps, "
            f"D={d} channels, mask {float(bp.mask.mean()):.2f} full"
        )
    return "\n".join(lines)


def make_dataset(
    experiments: Sequence[Experiment],
    *,
    output_channel_names: tuple[str, ...] | list[str],
    max_buckets: int | None = None,
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

    Every bucket compiles its own kernels, so many distinct lengths mean a slow
    start. ``max_buckets`` caps the count by padding shorter experiments with
    unobserved rows, at midpoints of their largest gaps, choosing the buckets
    that add the fewest rows. Predictions still report each experiment's own
    times. Only ``PredictionResult.buckets`` shows the padded grids.

    Capping does not change what ``mse`` or ``per_experiment_mse`` minimise: the trainers
    pool buckets by what each one averages over, and ``TrainingPhase.loss_horizon``
    cuts each experiment at a fraction of its own span, which the added rows
    do not move. Only a trajectory penalty sees the state at the added rows.

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
    max_buckets
        Most buckets to build. ``None`` keeps one per distinct length. An
        experiment with a single time has no gap to pad, so it stays in a
        bucket of its own, on top of this limit. Splits and resamples of the
        dataset apply the same limit to their own experiments.

    Returns
    -------
    Dataset
        ``bucket_payloads`` ordered ascending by ``T``, with the (named)
        experiments kept on ``Dataset.experiments``.
    """
    if not experiments:
        raise ValueError("make_dataset requires at least one experiment")
    if max_buckets is not None and (
        isinstance(max_buckets, bool) or not isinstance(max_buckets, int) or max_buckets < 1
    ):
        raise ValueError(f"make_dataset: max_buckets must be a positive int; got {max_buckets!r}")
    experiments = _named_experiments(experiments)
    output_channel_names = tuple(output_channel_names)
    if not output_channel_names:
        raise ValueError("make_dataset requires at least one output channel")
    if len(set(output_channel_names)) != len(output_channel_names):
        raise ValueError("output_channel_names must not contain duplicates")
    cov_keys = _validate_experiments(experiments, output_channel_names)

    per_experiment = []
    for exp in experiments:
        arrays = _per_experiment_arrays(exp, output_channel_names)
        if arrays.ts.shape[0] == 0:
            raise ValueError(
                f"Experiment {exp.exp_id!r} has no timestamps across the requested channels"
            )
        per_experiment.append(arrays)

    lengths = [arrays.ts.shape[0] for arrays in per_experiment]
    # One row spans no time, so there is nothing to pad into.
    paddable = [length for length in lengths if length > 1]
    targets = (
        _choose_bucket_lengths(paddable, max_buckets)
        if max_buckets is not None and paddable
        else tuple(sorted(set(paddable)))
    )
    by_len: dict[int, list[tuple[_ExperimentArrays, Experiment]]] = defaultdict(list)
    rows: list[tuple[int, int]] = []
    for arrays, exp, length in zip(per_experiment, experiments, lengths, strict=True):
        target = length if length == 1 else targets[bisect_left(targets, length)]
        rows.append((target, len(by_len[target])))
        by_len[target].append((_pad_to_length(arrays, target, exp.exp_id), exp))

    # Ascending T, so bucket order is a deterministic function of the data
    # rather than of dict insertion order.
    ordered = sorted(by_len)
    bucket_payloads = tuple(_stack_bucket(by_len[T], cov_keys) for T in ordered)
    index_of = {T: i for i, T in enumerate(ordered)}

    return Dataset(
        bucket_payloads=bucket_payloads,
        output_channel_names=output_channel_names,
        covariate_names=cov_keys,
        _experiments=tuple(experiments),
        _locations=tuple((index_of[T], row) for T, row in rows),
        _max_buckets=max_buckets,
    )


def _validate_fractions(train: float, validation: float, test: float) -> None:
    """Reject split fractions that are out of range or do not sum to one.

    Called before the source-experiments check, so a caller who gets both
    wrong hears about the fractions first.
    """
    if any(f < 0.0 or f > 1.0 for f in (train, validation, test)):
        raise ValueError("train, validation, and test fractions must each be in [0, 1]")
    total = train + validation + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"train + validation + test fractions must sum to 1.0, got {total}")


def _split_counts(n: int, train: float, validation: float, test: float) -> tuple[int, int, int]:
    """Experiment counts per split, preserving explicitly empty splits.

    For the normal three-way split, train and validation floor and test takes the
    remainder. When ``test == 0.0``, test is forced to stay empty and any
    rounding remainder is assigned to the larger fractional remainder of
    train or validation.

    A positive fraction that floors to zero is refused. An empty split reads
    downstream as "no validation needed" rather than "lost to rounding", so a
    caller who means it has to ask for ``0.0`` explicitly.
    """
    # A product such as 0.29 * 100 lands just under 29 in binary, so floor a
    # hair above it.
    n_train = int(np.floor(train * n + 1e-9))
    n_val = int(np.floor(validation * n + 1e-9))
    if test == 0.0:
        remainder = n - n_train - n_val
        if remainder:
            fractional_train = train * n - n_train
            fractional_val = validation * n - n_val
            if fractional_train >= fractional_val:
                n_train += remainder
            else:
                n_val += remainder
        n_test = 0
    else:
        n_test = n - n_train - n_val

    for name, frac, count in (
        ("train", train, n_train),
        ("validation", validation, n_val),
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
        max_buckets=dataset._max_buckets,
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
    # ``<exp_id>#<k>``; the first draw keeps its ID. Every source ID is
    # reserved up front, and k skips any name already taken, so a source ID
    # that itself looks like ``a#2`` cannot clash with a renamed repeat.
    draws: dict[str, int] = defaultdict(int)
    taken = {exp.exp_id for exp in experiments}
    resampled = []
    for i in idxs:
        exp = experiments[int(i)]
        draws[exp.exp_id] += 1
        if draws[exp.exp_id] == 1:
            resampled.append(exp)
            continue
        name = f"{exp.exp_id}#{draws[exp.exp_id]}"
        while name in taken:
            draws[exp.exp_id] += 1
            name = f"{exp.exp_id}#{draws[exp.exp_id]}"
        taken.add(name)
        resampled.append(_renamed(exp, name))
    return make_dataset(
        resampled,
        output_channel_names=dataset.output_channel_names,
        max_buckets=dataset._max_buckets,
    )


def split_dataset(
    dataset: Dataset,
    *,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
    key: Array,
) -> tuple[Dataset, Dataset, Dataset]:
    """Permute experiments and re-bucket each split independently.

    Splitting happens at the ``Experiment`` level and each split is bucketed
    from scratch, so its bucket structure suits its own contents rather than
    the original bucket boundaries.

    Counts use ``floor(train*n)`` and ``floor(validation*n)``, with test taking the
    remainder so the sizes sum to ``n``. With ``test=0`` the remainder goes to
    train or validation instead, whichever lost more to the floor. An empty split comes back with no
    payloads and no experiments, so it cannot be split again.

    Parameters
    ----------
    dataset
        Source dataset. Must carry its experiments, or this raises.
    train, validation, test
        Fractions in ``[0, 1]`` summing to ``1.0`` (within ``np.isclose``).
    key
        Required ``jr.PRNGKey`` for the permutation, never defaulted, so
        reproducibility does not rest on a hidden global.

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        ``(train_dataset, validation_dataset, test_dataset)``.
    """
    _validate_fractions(train, validation, test)

    experiments = dataset.experiments
    if not experiments:
        raise ValueError(
            "split_dataset requires the source experiments; the provided Dataset has none "
            "(was it built by hand rather than with make_dataset?)"
        )

    n = len(experiments)
    n_train, n_val, n_test = _split_counts(n, train, validation, test)
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
