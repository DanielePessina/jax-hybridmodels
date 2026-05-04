"""Per-channel summary diagnostics for hybrid-model predictions.

Given the tuple returned by ``hybridmodels.predict_dataset`` and the
matching ``Dataset``, these helpers flatten the observed/predicted pairs
under each bucket's mask, then compute the standard regression triple
(MSE, RMSE, MAE) and the mask-aware coefficient of determination per
channel.

Why per-channel and not aggregate
---------------------------------
Hybrid examples in this repo deliberately observe multiple channels with
*different units and dynamic ranges* (e.g. ``conc`` in mol/L and ``d43``
in micrometres). A single MSE across both is not interpretable. We
therefore report one row per channel; aggregate stats are out of scope
for this helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChannelDiagnostics:
    """Diagnostics for a single output channel.

    Attributes
    ----------
    name : str
        Channel name from ``dataset.output_channel_names``.
    n : int
        Number of observed (mask=True) cells contributing to the stats.
    mse : float
        Mean squared error of ``predicted - observed`` over the masked cells.
    rmse : float
        ``sqrt(mse)`` for unit-matched reporting.
    mae : float
        Mean absolute error.
    r2 : float
        Coefficient of determination ``1 - SS_res/SS_tot`` where
        ``SS_tot`` uses the mean of the observed values. Returns ``nan``
        when ``SS_tot == 0`` (constant observations); a parity plot will
        still render but R^2 is undefined in that degenerate case.
    obs : np.ndarray
        Flattened observed values (shape ``[n]``); kept for parity plots.
    pred : np.ndarray
        Flattened predicted values (shape ``[n]``); kept for parity plots.
    """

    name: str
    n: int
    mse: float
    rmse: float
    mae: float
    r2: float
    obs: np.ndarray
    pred: np.ndarray


def compute_diagnostics(
    predictions: tuple[Any, ...],
    dataset: Any,
) -> dict[str, ChannelDiagnostics]:
    """Collapse bucketed predictions into per-channel summary stats.

    Parameters
    ----------
    predictions : tuple of arrays, one per bucket
        The tuple returned by ``hybridmodels.predict_dataset``. Each
        entry has shape ``[N_b, T_b, D]`` matching the corresponding
        ``BucketPayload``.
    dataset : hybridmodels.Dataset
        The dataset that produced ``predictions`` — its
        ``bucket_payloads`` (for masks/observations) and
        ``output_channel_names`` (for channel labels) are read here.

    Returns
    -------
    dict[str, ChannelDiagnostics]
        Insertion-order dict keyed by channel name in
        ``dataset.output_channel_names`` order.
    """
    channels = dataset.output_channel_names
    if len(predictions) != len(dataset.bucket_payloads):
        raise ValueError(
            "predictions tuple length does not match dataset.bucket_payloads "
            f"({len(predictions)} vs {len(dataset.bucket_payloads)})"
        )

    out: dict[str, ChannelDiagnostics] = {}
    for d, name in enumerate(channels):
        obs_chunks: list[np.ndarray] = []
        pred_chunks: list[np.ndarray] = []
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
            mask_d = np.asarray(bp.mask[..., d], dtype=bool)
            obs_d = np.asarray(bp.y_observed[..., d])
            pred_d = np.asarray(pred[..., d])
            obs_chunks.append(obs_d[mask_d])
            pred_chunks.append(pred_d[mask_d])
        obs_arr = np.concatenate(obs_chunks) if obs_chunks else np.empty(0)
        pred_arr = np.concatenate(pred_chunks) if pred_chunks else np.empty(0)
        out[name] = _stats_from_pairs(name, obs_arr, pred_arr)
    return out


def _stats_from_pairs(name: str, obs: np.ndarray, pred: np.ndarray) -> ChannelDiagnostics:
    """Compute the canonical (MSE, RMSE, MAE, R^2) triple from two flat arrays."""
    n = int(obs.shape[0])
    if n == 0:
        nan = float("nan")
        return ChannelDiagnostics(name, 0, nan, nan, nan, nan, obs, pred)
    residuals = pred - obs
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    # ss_tot == 0 means constant observed values — R^2 is undefined.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ChannelDiagnostics(name, n, mse, rmse, mae, r2, obs, pred)


def print_diagnostics(
    diag: dict[str, ChannelDiagnostics],
    *,
    header: str | None = None,
) -> None:
    """Pretty-print a diagnostics table to stdout.

    Format::

        {header}
          channel        n          MSE         RMSE          MAE       R^2
          conc          42   1.2345e-03   3.5135e-02   2.7012e-02    0.9876
          d43           17   1.4321e+00   1.1967e+00   8.9120e-01    0.4231

    The MSE/RMSE/MAE columns use scientific notation so the same template
    stays readable across the wildly different scales in the example
    suite (omega ~ O(1), conc ~ O(0.1), d43 ~ O(10), nucleation rates
    spanning ~9 decades after the predictor exponentiates them).
    """
    if header:
        print(header)
    print(f"  {'channel':<12} {'n':>6} {'MSE':>13} {'RMSE':>13} {'MAE':>13} {'R^2':>8}")
    for s in diag.values():
        r2_str = "    nan " if np.isnan(s.r2) else f"{s.r2:>8.4f}"
        print(f"  {s.name:<12} {s.n:>6d} {s.mse:>13.4e} {s.rmse:>13.4e} {s.mae:>13.4e} {r2_str}")
