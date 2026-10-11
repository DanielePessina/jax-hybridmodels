"""Per-channel evaluation metrics for hybrid-model predictions.

Takes the ``PredictionResult`` returned by ``HybridModel.predict``, which
already carries each experiment's observed values and observation mask,
flattens the observed/predicted pairs under the mask, and reports MSE,
RMSE, MAE and R^2 **per channel**. The functions duck-type on that result
rather than importing ``model.py``, so this module stays a leaf.

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

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

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
    """

    name: str
    n: Int[Array, ""]
    mse: Float[Array, ""]
    rmse: Float[Array, ""]
    mae: Float[Array, ""]
    r2: Float[Array, ""]


def compute_metrics(predictions: Mapping[str, Any]) -> dict[str, ChannelMetrics]:
    """Per-channel summary stats of a prediction against the data it was made on.

    Parameters
    ----------
    predictions : PredictionResult
        From ``HybridModel.predict`` or ``PredictionResult.from_buckets``,
        or any mapping from experiment ID to
        a record with ``outputs``, ``observed`` and ``observation_mask``
        dicts keyed by channel. Only cells the mask marks as measured
        count. Channels are read from the first experiment's ``outputs``.

    Returns
    -------
    dict[str, ChannelMetrics]
        Keyed by channel, in the model's ``output_channel_names`` order.
    """
    if not isinstance(predictions, Mapping):
        raise TypeError(
            "compute_metrics takes the PredictionResult from HybridModel.predict; got "
            f"{type(predictions).__name__}. Wrap bucket arrays from predict_dataset or "
            "ensemble_predictions with PredictionResult.from_buckets(buckets, dataset)."
        )
    experiments = list(predictions.values())
    if not experiments:
        raise ValueError("compute_metrics needs at least one experiment")
    if not experiments[0].observation_mask:
        raise ValueError(
            "compute_metrics needs observed values; pass the result of predict(), not simulate()"
        )
    out: dict[str, ChannelMetrics] = {}
    for name in experiments[0].outputs:
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
        out[name] = _stats_from_sufficient_stats(
            name, n, jnp.sum(residuals**2), jnp.sum(jnp.abs(residuals)), ss_tot
        )
    return out


def _stats_from_sufficient_stats(
    name: str,
    n: Array,
    ss_res: Array,
    abs_error: Array,
    ss_tot: Array,
) -> ChannelMetrics:
    """Build the canonical metrics from masked sufficient statistics."""
    denom = jnp.maximum(n, 1).astype(ss_res.dtype)
    mse = ss_res / denom
    rmse = jnp.sqrt(mse)
    mae = abs_error / denom
    # ss_tot == 0 means constant observed values, so R^2 is undefined. It is
    # also undefined for a channel with no observations.
    r2 = jnp.where(n > 0, jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, jnp.nan), jnp.nan)
    return ChannelMetrics(name, n, mse, rmse, mae, r2)


def print_metrics(
    metrics: Mapping[str, Any],
    *,
    header: str | None = None,
) -> None:
    """Print a metrics table, one row per channel.

    Takes the output of :func:`compute_metrics`, or a ``PredictionResult``,
    which is passed through :func:`compute_metrics` first.

    Format::

        {header}
          channel        n          MSE         RMSE          MAE       R^2
          conc          42   1.2345e-03   3.5135e-02   2.7012e-02    0.9876
          d43           17   1.4321e+00   1.1967e+00   8.9120e-01    0.4231

    Scientific notation throughout, so one template stays readable across
    the example suite's scales, from ``omega ~ O(1)`` to nucleation rates
    spanning nine decades.
    """
    if not isinstance(metrics, Mapping) or not all(
        isinstance(m, ChannelMetrics) for m in metrics.values()
    ):
        metrics = compute_metrics(metrics)
    if header:
        print(header)
    print(f"  {'channel':<12} {'n':>6} {'MSE':>13} {'RMSE':>13} {'MAE':>13} {'R^2':>8}")
    for m in metrics.values():
        r2_val = float(m.r2)
        r2_str = "    nan " if r2_val != r2_val else f"{r2_val:>8.4f}"
        print(
            f"  {m.name:<12} {int(m.n):>6d} {float(m.mse):>13.4e} "
            f"{float(m.rmse):>13.4e} {float(m.mae):>13.4e} {r2_str}"
        )
