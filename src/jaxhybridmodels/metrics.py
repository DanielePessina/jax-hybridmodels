"""Per-channel evaluation metrics for hybrid-model predictions.

Takes the ``PredictionResult`` returned by ``HybridModel.predict``, which
already carries each experiment's observed values and observation mask,
flattens the observed/predicted pairs under the mask, and reports MSE,
RMSE, MAE and R^2 **per channel**, pooled over the experiments or for each
one. The functions duck-type on that result rather than importing
``model.py``, so this module stays a leaf.

Per channel, never aggregated: the examples observe channels with different
units and dynamic ranges (``conc`` in mol/L, ``d43`` in micrometres), and
one MSE across both means nothing.

The arithmetic is JAX-native and every statistic is a JAX scalar, so the
results stay dtype-consistent with the training pipeline and
``ChannelMetrics`` is a pytree; call ``float(...)`` at the boundary when you
need Python floats.

Only cells the observation mask marks as real measurements count, matching
the losses' NaN discipline.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int


@partial(
    jtu.register_dataclass,
    data_fields=("n", "mse", "rmse", "mae", "r2"),
    meta_fields=("name",),
)
@dataclass(frozen=True)
class ChannelMetrics:
    """Metrics for a single output channel.

    Attributes
    ----------
    name : str
        Channel name, one of the model's ``output_channel_names``.
    n : Int[Array, ""]
        Scalar number of observed (mask=True) cells behind the stats. It is a
        JAX scalar so the complete metrics result can pass through ``jit``.
    mse, rmse, mae : Float[Array, ""]
        Error of ``predicted - observed`` over the masked cells.
    r2 : Float[Array, ""]
        ``1 - SS_res/SS_tot``; ``nan`` when the observations are constant
        and ``SS_tot`` is zero.

    With ``n == 0`` every statistic is ``nan``, so a channel nobody measured
    cannot read as a perfect fit.
    """

    name: str
    n: Int[Array, ""]
    mse: Float[Array, ""]
    rmse: Float[Array, ""]
    mae: Float[Array, ""]
    r2: Float[Array, ""]


def compute_metrics(
    predictions: Mapping[str, Any],
    *,
    by: Literal["channel", "experiment"] = "channel",
) -> dict[str, Any]:
    """Per-channel summary stats of a prediction against the data it was made on.

    Parameters
    ----------
    predictions : PredictionResult
        From ``HybridModel.predict`` or ``predict_ensemble`` (which is
        scored on its member mean), or any mapping from experiment ID to
        a record with ``outputs``, ``observed`` and ``observation_mask``
        dicts keyed by channel. Only cells the mask marks as measured
        count. Channels are read from the first experiment's ``outputs``.
    by : {"channel", "experiment"}
        ``"channel"`` pools every experiment. ``"experiment"`` scores each
        one alone, which is how to find the run a pooled number hides. Its
        R^2 compares against that experiment's own mean, so a good fit to a
        nearly flat run can still score low. A channel an experiment did not
        measure has ``n == 0`` and NaN statistics.

    Returns
    -------
    dict
        For ``"channel"``, ``{channel: ChannelMetrics}`` in the model's
        ``output_channel_names`` order. For ``"experiment"``,
        ``{exp_id: {channel: ChannelMetrics}}`` in experiment order.
    """
    if by not in ("channel", "experiment"):
        raise ValueError(f"compute_metrics by= must be 'channel' or 'experiment'; got {by!r}")
    if not isinstance(predictions, Mapping):
        raise TypeError(
            "compute_metrics takes the PredictionResult from HybridModel.predict or "
            f"predict_ensemble; got {type(predictions).__name__}."
        )
    experiments = list(predictions.values())
    if not experiments:
        raise ValueError("compute_metrics needs at least one experiment")
    if not experiments[0].observation_mask:
        raise ValueError(
            "compute_metrics needs observed values; pass the result of predict() on the "
            "measured grid, not simulate() or predict(ts=...) / predict(n_points=...)"
        )
    channels = list(experiments[0].outputs)
    if by == "experiment":
        return {
            exp_id: {name: _channel_metrics(name, [experiment]) for name in channels}
            for exp_id, experiment in predictions.items()
        }
    return {name: _channel_metrics(name, experiments) for name in channels}


def _channel_metrics(name: str, experiments: Sequence[Any]) -> ChannelMetrics:
    """One channel's statistics over the measured cells of ``experiments``, pooled."""
    mask = jnp.concatenate([jnp.ravel(e.observation_mask[name]) for e in experiments])
    observed = jnp.concatenate([jnp.ravel(e.observed[name]) for e in experiments])
    predicted = jnp.concatenate([jnp.ravel(e.outputs[name]) for e in experiments])
    # Unobserved cells hold NaN; zero them before any arithmetic so a
    # NaN never reaches a sum, even multiplied by a False mask.
    observed = jnp.where(mask, observed, 0.0)
    residuals = jnp.where(mask, predicted - observed, 0.0)
    n = mask.sum().astype(jnp.int32)
    mean_observed = observed.sum() / jnp.maximum(n, 1).astype(observed.dtype)
    ss_tot = jnp.sum(jnp.where(mask, (observed - mean_observed) ** 2, 0.0))
    return _stats_from_sufficient_stats(
        name, n, jnp.sum(residuals**2), jnp.sum(jnp.abs(residuals)), ss_tot
    )


def _stats_from_sufficient_stats(
    name: str,
    n: Array,
    ss_res: Array,
    abs_error: Array,
    ss_tot: Array,
) -> ChannelMetrics:
    """Build the canonical metrics from masked sufficient statistics."""
    denom = jnp.maximum(n, 1).astype(ss_res.dtype)
    # With no observations the sums are zero, and reporting an error of 0
    # would read as a perfect fit to a channel nobody measured.
    mse = jnp.where(n > 0, ss_res / denom, jnp.nan)
    rmse = jnp.sqrt(mse)
    mae = jnp.where(n > 0, abs_error / denom, jnp.nan)
    # ss_tot == 0 means constant observed values, so R^2 is undefined. It is
    # also undefined for a channel with no observations.
    r2 = jnp.where(n > 0, jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, jnp.nan), jnp.nan)
    return ChannelMetrics(name, n, mse, rmse, mae, r2)


def print_metrics(
    metrics: Mapping[str, Any],
    *,
    header: str | None = None,
    by: Literal["channel", "experiment"] = "channel",
) -> None:
    """Print a metrics table.

    Takes the output of :func:`compute_metrics`, or a ``PredictionResult``,
    which is passed through :func:`compute_metrics` with ``by`` first.

    Format::

        {header}
          channel        n          MSE         RMSE          MAE       R^2
          conc          42   1.2345e-03   3.5135e-02   2.7012e-02    0.9876
          d43           17   1.4321e+00   1.1967e+00   8.9120e-01    0.4231

    Scientific notation throughout, so one template stays readable across
    the example suite's scales, from ``omega ~ O(1)`` to nucleation rates
    spanning nine decades.

    ``by="experiment"``, or metrics computed that way, prints one table per
    channel with the worst RMSE first and leaves out experiments that did not
    measure the channel.
    """
    if by not in ("channel", "experiment"):
        raise ValueError(f"print_metrics by= must be 'channel' or 'experiment'; got {by!r}")
    values = list(metrics.values()) if isinstance(metrics, Mapping) else []
    if values and all(isinstance(m, ChannelMetrics) for m in values):
        if by == "experiment":
            raise ValueError(
                "print_metrics by='experiment' needs compute_metrics(predictions, "
                "by='experiment') or a PredictionResult; these metrics are pooled"
            )
    elif not (values and all(_is_per_channel(m) for m in values)):
        metrics = compute_metrics(metrics, by=by)
    if header:
        print(header)
    print("\n".join(format_metrics(metrics)))


def _is_per_channel(value: Any) -> bool:
    """Whether ``value`` is one experiment's ``{channel: ChannelMetrics}``."""
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(m, ChannelMetrics) for m in value.values())
    )


def _metric_columns(m: ChannelMetrics) -> str:
    r2_val = float(m.r2)
    r2_str = "    nan " if r2_val != r2_val else f"{r2_val:>8.4f}"
    return (
        f"{int(m.n):>6d} {float(m.mse):>13.4e} "
        f"{float(m.rmse):>13.4e} {float(m.mae):>13.4e} {r2_str}"
    )


_COLUMN_TITLES = f"{'n':>6} {'MSE':>13} {'RMSE':>13} {'MAE':>13} {'R^2':>8}"


def format_metrics(metrics: Mapping[str, Any]) -> list[str]:
    """The lines of :func:`print_metrics`, for pooled or per-experiment metrics."""
    first = next(iter(metrics.values()), None)
    if isinstance(first, ChannelMetrics):
        return [f"  {'channel':<12} {_COLUMN_TITLES}"] + [
            f"  {m.name:<12} {_metric_columns(m)}" for m in metrics.values()
        ]
    lines: list[str] = []
    for channel in first or ():
        rows = worst_first(metrics, channel)
        width = max([len("experiment"), *(len(exp_id) for exp_id, _ in rows)])
        lines.append(f"  {channel}")
        lines.append(f"    {'experiment':<{width}} {_COLUMN_TITLES}")
        lines.extend(f"    {exp_id:<{width}} {_metric_columns(m)}" for exp_id, m in rows)
    return lines


def worst_first(
    by_experiment: Mapping[str, Mapping[str, ChannelMetrics]], channel: str
) -> list[tuple[str, ChannelMetrics]]:
    """``(exp_id, metrics)`` for experiments that measured ``channel``, worst RMSE first."""
    rows = [
        (exp_id, per_channel[channel])
        for exp_id, per_channel in by_experiment.items()
        if int(per_channel[channel].n) > 0
    ]
    return sorted(rows, key=lambda row: -float(row[1].rmse))
