"""Per-channel evaluation metrics for hybrid-model predictions.

Takes the tuple returned by :func:`hybridmodels.prediction.predict_dataset`
and its ``Dataset``, flattens the observed/predicted pairs under each
bucket's mask, and reports MSE, RMSE, MAE and R^2 **per channel**.

Per channel, never aggregated: the examples observe channels with different
units and dynamic ranges (``conc`` in mol/L, ``d43`` in micrometres), and
one MSE across both means nothing.

Everything is JAX-native, so the metrics compose with ``jax.jit`` and stay
dtype-consistent with the training pipeline; call ``float(...)`` or
``np.asarray(...)`` at the boundary when you need Python floats.

Only cells the bucket's ``mask`` marks as real measurements count, matching
the losses' NaN discipline.
"""

# ruff: noqa: F722

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Float, Int

from hybridmodels.data import Dataset


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
        Channel name from ``dataset.output_channel_names``.
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


def compute_metrics(
    predictions: tuple[Float[Array, "N T D"], ...],
    dataset: Dataset,
) -> dict[str, ChannelMetrics]:
    """Collapse bucketed predictions into per-channel summary stats.

    Parameters
    ----------
    predictions : tuple of arrays, one per bucket
        From :func:`hybridmodels.prediction.predict_dataset`; each entry is
        ``[N_b, T_b, D]`` matching its ``BucketPayload``.
    dataset : Dataset
        The dataset that produced ``predictions``. Read for masks,
        observations, and channel names.

    Returns
    -------
    dict[str, ChannelMetrics]
        Keyed by channel, in ``dataset.output_channel_names`` order.
    """
    channels = dataset.output_channel_names
    if len(predictions) != len(dataset.bucket_payloads):
        raise ValueError(
            "predictions tuple length does not match dataset.bucket_payloads "
            f"({len(predictions)} vs {len(dataset.bucket_payloads)})"
        )

    out: dict[str, ChannelMetrics] = {}
    for d, name in enumerate(channels):
        n = jnp.asarray(0, dtype=jnp.int32)
        ss_res = jnp.asarray(0.0)
        abs_error = jnp.asarray(0.0)
        ss_tot = jnp.asarray(0.0)
        mean_observed = jnp.asarray(0.0)
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
            mask_d: Bool[Array, "N T"] = bp.mask[..., d]
            obs_d: Float[Array, "N T"] = bp.y_observed[..., d]
            pred_d: Float[Array, "N T"] = pred[..., d]
            obs_safe = jnp.where(mask_d, obs_d, 0.0)
            pred_safe = jnp.where(mask_d, pred_d, 0.0)
            residuals = pred_safe - obs_safe
            count = mask_d.sum().astype(jnp.int32)
            count_safe = jnp.maximum(count, 1).astype(obs_safe.dtype)
            mean_obs = obs_safe.sum() / count_safe
            combined_count = n + count
            combined_safe = jnp.maximum(combined_count, 1).astype(obs_safe.dtype)
            delta = mean_obs - mean_observed
            between = (
                delta**2 * n.astype(obs_safe.dtype) * count.astype(obs_safe.dtype) / combined_safe
            )
            mean_observed = jnp.where(
                combined_count > 0,
                (n.astype(obs_safe.dtype) * mean_observed + count.astype(obs_safe.dtype) * mean_obs)
                / combined_safe,
                0.0,
            )
            n = n + count
            ss_res = ss_res + jnp.sum(jnp.where(mask_d, residuals**2, 0.0))
            abs_error = abs_error + jnp.sum(jnp.where(mask_d, jnp.abs(residuals), 0.0))
            ss_tot = ss_tot + jnp.sum(jnp.where(mask_d, (obs_safe - mean_obs) ** 2, 0.0)) + between
        out[name] = _stats_from_sufficient_stats(name, n, ss_res, abs_error, ss_tot)
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
    # ss_tot == 0 means constant observed values — R^2 is undefined. It is
    # also undefined for a channel with no observations.
    r2 = jnp.where(n > 0, jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, jnp.nan), jnp.nan)
    return ChannelMetrics(name, n, mse, rmse, mae, r2)


def print_metrics(
    metrics: dict[str, ChannelMetrics],
    *,
    header: str | None = None,
) -> None:
    """Print a metrics table, one row per channel.

    Format::

        {header}
          channel        n          MSE         RMSE          MAE       R^2
          conc          42   1.2345e-03   3.5135e-02   2.7012e-02    0.9876
          d43           17   1.4321e+00   1.1967e+00   8.9120e-01    0.4231

    Scientific notation throughout, so one template stays readable across
    the example suite's scales, from ``omega ~ O(1)`` to nucleation rates
    spanning nine decades.
    """
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
