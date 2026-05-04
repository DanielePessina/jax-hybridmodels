"""Default plotting routines for the example scripts.

Two plot kinds are exposed:

* ``parity_plot``: predicted-vs-observed scatter, one subplot per channel,
  with the ``y = x`` identity line and per-panel R^2/RMSE in the title.
* ``trajectory_plot``: per-experiment time-series, one row per
  experiment, one column per channel — predicted curve plus observed
  scatter under the channel's mask.

Both functions accept a ``save_path`` and return the matplotlib
``Figure`` so the caller can show, save, or further customise. Callers
are expected to have applied :func:`apply_default_style` upstream — the
helpers do not force-apply rcParams (they would otherwise stomp on a
caller that passes their own ``ax``).

Extension pattern
-----------------
Both helpers are deliberately small and side-effect-light: they only
read from the diagnostics/dataset and call ``plt`` primitives. To extend,
copy a function and tweak — composition over a config-bag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._diagnostics import ChannelDiagnostics


def parity_plot(
    diagnostics: dict[str, ChannelDiagnostics],
    *,
    channels: list[str] | None = None,
    title: str | None = "Parity",
    save_path: str | Path | None = None,
) -> Figure:
    """Predicted-vs-observed scatter, one subplot per channel.

    Each subplot draws (a) the observed/predicted scatter and (b) the
    identity ``y = x`` reference line spanning the union range. Title
    carries the per-channel R^2 and RMSE so a single glance catches
    bias/spread.

    Parameters
    ----------
    diagnostics : dict[str, ChannelDiagnostics]
        Output of :func:`compute_diagnostics`. The ``obs``/``pred``
        arrays inside each entry are what gets scattered.
    channels : list[str], optional
        Subset and order of channels to plot. Defaults to every channel
        in the diagnostics dict (insertion order = dataset order).
    title : str, optional
        Figure suptitle. Pass ``None`` to skip.
    save_path : str | Path, optional
        If given, the figure is written there with the active
        ``savefig.dpi``. The figure is still returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if channels is None:
        channels = list(diagnostics.keys())
    n = len(channels)
    fig, axes_arr = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes_arr.flatten()

    for ax, name in zip(axes, channels, strict=True):
        s = diagnostics[name]
        if s.n == 0:
            ax.set_title(f"{name}\n(no observations)")
            ax.set_xlabel("observed")
            ax.set_ylabel("predicted")
            continue
        ax.scatter(s.obs, s.pred, s=12, alpha=0.6)
        # Identity line spans the combined min/max of obs and pred so
        # bias is visible even when the cloud is far from x = y.
        lo = float(min(s.obs.min(), s.pred.min()))
        hi = float(max(s.obs.max(), s.pred.max()))
        if lo == hi:
            # Degenerate: nothing useful to draw, but still mark the point.
            pad = 1.0 if lo == 0.0 else abs(lo) * 0.1
            lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        r2_str = "nan" if np.isnan(s.r2) else f"{s.r2:.3f}"
        ax.set_title(f"{name}\nR$^2$ = {r2_str}, RMSE = {s.rmse:.3e}")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path))
    return fig


def trajectory_plot(
    predictions: tuple[Any, ...],
    dataset: Any,
    *,
    max_experiments: int = 6,
    title: str | None = "Trajectories",
    save_path: str | Path | None = None,
) -> Figure:
    """Per-experiment predicted-vs-observed time series, one row per experiment.

    Layout: ``rows = min(max_experiments, total_experiments)``,
    ``cols = D`` (one per output channel). Predicted is drawn as a line
    on the bucket's union timestamp axis; observed values are scattered
    only at cells where ``mask`` is True (so missing-channel rows show
    only the predicted curve).

    Parameters
    ----------
    predictions : tuple of arrays
        Output of ``predict_dataset``; one ``[N_b, T_b, D]`` entry per
        bucket.
    dataset : hybridmodels.Dataset
        Used for ``output_channel_names`` and to walk the bucket-payload
        list in lockstep with ``predictions``.
    max_experiments : int, optional
        Upper cap on rows so a 50-experiment dataset doesn't render a
        gigantic figure by default. Experiments are walked in
        bucket-payload order.
    title : str, optional
        Figure suptitle. Pass ``None`` to skip.
    save_path : str | Path, optional
        If given, the figure is written there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    channels = list(dataset.output_channel_names)
    rows: list[dict[str, np.ndarray]] = []
    for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
        N = bp.ts.shape[0]
        for i in range(N):
            if len(rows) >= max_experiments:
                break
            rows.append(
                {
                    "ts": np.asarray(bp.ts[i]),
                    "y_obs": np.asarray(bp.y_observed[i]),
                    "mask": np.asarray(bp.mask[i], dtype=bool),
                    "y_pred": np.asarray(pred[i]),
                }
            )
        if len(rows) >= max_experiments:
            break

    nrows = len(rows)
    ncols = len(channels)
    fig, axes_arr = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * max(nrows, 1)),
        squeeze=False,
    )

    for r, row in enumerate(rows):
        for c, name in enumerate(channels):
            ax = axes_arr[r][c]
            ts = row["ts"]
            ax.plot(ts, row["y_pred"][:, c], label="predicted", linewidth=1.5)
            mask_c = row["mask"][:, c]
            if bool(mask_c.any()):
                ax.scatter(
                    ts[mask_c],
                    row["y_obs"][mask_c, c],
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
                ax.set_ylabel(f"exp {r}")
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path))
    return fig
