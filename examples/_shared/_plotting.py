"""Default plots for the example scripts.

``parity_plot`` scatters predicted against observed, one subplot per
channel, from :func:`parity_diagnostics` of a ``PredictionResult``.
``trajectory_plot`` draws per-experiment time series of a ``HybridModel``,
one row per experiment and one column per channel.

Both take a ``save_path`` and return the ``Figure``. Neither applies
rcParams, which would stomp on a caller passing their own ``ax``, so call
:func:`apply_default_style` first. To extend either, copy and tweak.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jaxhybridmodels import Dataset, Experiment, HybridModel, PredictionResult, compute_metrics


def parity_diagnostics(predictions: PredictionResult) -> dict[str, SimpleNamespace]:
    """Observed and predicted values per channel, for :func:`parity_plot`.

    ``predictions`` is the result of ``model.predict(data)``. Only measured
    cells are kept. R^2 and RMSE come from ``hm.compute_metrics``.
    """
    metrics = compute_metrics(predictions)
    experiments = list(predictions.values())
    out: dict[str, SimpleNamespace] = {}
    for name, m in metrics.items():
        obs = np.concatenate(
            [
                np.asarray(e.observed[name])[np.asarray(e.observation_mask[name])]
                for e in experiments
            ]
        )
        pred = np.concatenate(
            [np.asarray(e.outputs[name])[np.asarray(e.observation_mask[name])] for e in experiments]
        )
        out[name] = SimpleNamespace(
            name=name, n=int(m.n), obs=obs, pred=pred, r2=float(m.r2), rmse=float(m.rmse)
        )
    return out


def parity_plot(
    diagnostics: dict[str, SimpleNamespace],
    *,
    channels: list[str] | None = None,
    title: str | None = "Parity",
    save_path: str | Path | None = None,
) -> Figure:
    """Predicted-vs-observed scatter, one subplot per channel.

    Each subplot carries the identity ``y = x`` line and the channel's R^2
    and RMSE in its title, so bias and spread read at a glance.

    Parameters
    ----------
    diagnostics : dict[str, SimpleNamespace]
        Output of :func:`parity_diagnostics`, one entry per channel with
        ``obs`` / ``pred`` / ``n`` / ``r2`` / ``rmse``.
    channels : list[str], optional
        Subset and order to plot. Defaults to every channel, in dataset
        order.
    title : str, optional
        Figure suptitle; ``None`` skips it.
    save_path : str | Path, optional
        Where to write the figure. It is returned either way.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if channels is None:
        channels = list(diagnostics.keys())
    n = len(channels)
    fig, axes_arr = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, layout="constrained")
    axes = axes_arr.flatten()

    for ax, name in zip(axes, channels, strict=True):
        s = diagnostics[name]
        if s.n == 0:
            ax.set_title(f"{name}\n(no observations)")
            ax.set_xlabel("observed")
            ax.set_ylabel("predicted")
            continue
        ax.scatter(s.obs, s.pred, s=12, alpha=0.6)
        # Span the combined min/max, so bias shows even when the cloud is
        # far from x = y.
        lo = float(min(s.obs.min(), s.pred.min()))
        hi = float(max(s.obs.max(), s.pred.max()))
        if lo == hi:
            # Degenerate, but still mark the point.
            pad = 1.0 if lo == 0.0 else abs(lo) * 0.1
            lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        r2_str = "nan" if np.isnan(s.r2) else f"{s.r2:.3f}"
        ax.set_title(f"{name}\nR$^2$ = {r2_str}, RMSE = {s.rmse:.3e}")

    if title:
        fig.suptitle(title)
    if save_path is not None:
        fig.savefig(str(save_path))
    return fig


def trajectory_plot(
    model: HybridModel,
    data: Dataset | Sequence[Experiment],
    *,
    predictions: PredictionResult | None = None,
    n_dense_points: int = 200,
    max_experiments: int = 6,
    title: str | None = "Trajectories",
    save_path: str | Path | None = None,
) -> Figure:
    """Per-experiment predicted-vs-observed time series.

    One row per experiment, one column per channel, for the first
    ``max_experiments`` experiments of ``data`` in order. Observations are
    scattered where they were measured. The predicted curve is re-simulated
    with ``model.simulate(experiment=...)`` on a dense grid, so it is smooth
    between sparse measurements.

    Parameters
    ----------
    model : HybridModel
        The model to draw, usually ``fit_result.model``.
    data : Dataset or sequence of Experiment
        The experiments to plot.
    predictions : PredictionResult, optional
        ``model.predict(data)``, if already computed. Supplies the observed
        values, and the curve when the dense path is off.
    n_dense_points : int, optional
        Size of the linspace part of the dense grid. The grid is that
        linspace unioned with the measured times, so the curve passes
        through every observation. ``0`` draws the prediction on the
        measured times only.
    max_experiments : int, optional
        Row cap, so a 50-experiment dataset does not render a giant figure.
    title : str, optional
        Figure suptitle; ``None`` skips it.
    save_path : str | Path, optional
        Where to write the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    channels = list(model.output_channel_names)
    experiments = list(data.experiments if isinstance(data, Dataset) else data)[:max_experiments]
    if predictions is None:
        predictions = model.predict(data)

    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        measured = predictions[experiment.exp_id]
        ts_obs = np.asarray(measured.times)
        row: dict[str, Any] = {
            "exp_id": experiment.exp_id,
            "ts_obs": ts_obs,
            "observed": {c: np.asarray(measured.observed[c]) for c in channels},
            "mask": {c: np.asarray(measured.observation_mask[c], dtype=bool) for c in channels},
            "ts_pred": ts_obs,
            "y_pred": {c: np.asarray(measured.outputs[c]) for c in channels},
        }
        if n_dense_points > 0 and ts_obs.size >= 2:
            # ``np.unique`` sorts and drops the measured times the linspace hits.
            dense = np.linspace(ts_obs[0], ts_obs[-1], n_dense_points)
            ts_dense = np.unique(np.concatenate([ts_obs, dense]))
            trajectory = model.simulate(experiment=experiment, ts=ts_dense)
            row["ts_pred"] = ts_dense
            row["y_pred"] = {c: np.asarray(trajectory.outputs[c]) for c in channels}
        rows.append(row)

    nrows = len(rows)
    ncols = len(channels)
    fig, axes_arr = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * max(nrows, 1)),
        squeeze=False,
        layout="constrained",
    )

    for r, row in enumerate(rows):
        for c, name in enumerate(channels):
            ax = axes_arr[r][c]
            ax.plot(row["ts_pred"], row["y_pred"][name], label="predicted", linewidth=1.5)
            mask = row["mask"][name]
            if bool(mask.any()):
                ax.scatter(
                    row["ts_obs"][mask],
                    row["observed"][name][mask],
                    s=18,
                    marker="o",
                    label="observed",
                    zorder=3,
                )
            if r == 0:
                ax.set_title(name)
            if r == nrows - 1:
                ax.set_xlabel("t")
            if c == 0:
                ax.set_ylabel(row["exp_id"])
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title)
    if save_path is not None:
        fig.savefig(str(save_path))
    return fig
