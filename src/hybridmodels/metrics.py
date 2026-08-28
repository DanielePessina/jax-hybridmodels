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

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from hybridmodels.data import Dataset


@dataclass(frozen=True)
class ChannelMetrics:
    """Metrics for a single output channel.

    Attributes
    ----------
    name : str
        Channel name from ``dataset.output_channel_names``.
    n : int
        Number of observed (mask=True) cells behind the stats.
    mse, rmse, mae : Float[Array, ""]
        Error of ``predicted - observed`` over the masked cells.
    r2 : Float[Array, ""]
        ``1 - SS_res/SS_tot``; ``nan`` when the observations are constant
        and ``SS_tot`` is zero.
    """

    name: str
    n: int
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
        obs_chunks: list[Array] = []
        pred_chunks: list[Array] = []
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
            mask_d: Bool[Array, "N T"] = bp.mask[..., d]
            obs_d: Float[Array, "N T"] = bp.y_observed[..., d]
            pred_d: Float[Array, "N T"] = pred[..., d]
            obs_chunks.append(obs_d[mask_d])
            pred_chunks.append(pred_d[mask_d])
        obs = jnp.concatenate(obs_chunks) if obs_chunks else jnp.empty(0)
        pred = jnp.concatenate(pred_chunks) if pred_chunks else jnp.empty(0)
        out[name] = _stats_from_pairs(name, obs, pred)
    return out


def _stats_from_pairs(name: str, obs: Array, pred: Array) -> ChannelMetrics:
    """Compute the canonical (MSE, RMSE, MAE, R^2) set from two flat arrays."""
    n = int(obs.shape[0])
    if n == 0:
        nan = jnp.asarray(float("nan"))
        return ChannelMetrics(name, 0, nan, nan, nan, nan)
    residuals = pred - obs
    mse = jnp.mean(residuals**2)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(residuals))
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((obs - jnp.mean(obs)) ** 2)
    # ss_tot == 0 means constant observed values — R^2 is undefined.
    r2 = jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, jnp.asarray(float("nan")))
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
            f"  {m.name:<12} {m.n:>6d} {float(m.mse):>13.4e} "
            f"{float(m.rmse):>13.4e} {float(m.mae):>13.4e} {r2_str}"
        )
