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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
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
    predictors: Any | None = None,
    simulate_fn: Callable[..., Any] | None = None,
    solver: Any | None = None,
    n_dense_points: int = 200,
    max_experiments: int = 6,
    title: str | None = "Trajectories",
    save_path: str | Path | None = None,
) -> Figure:
    """Per-experiment predicted-vs-observed time series, one row per experiment.

    Layout: ``rows = min(max_experiments, total_experiments)``,
    ``cols = D`` (one per output channel). Observed values are scattered
    at the bucket's union timestamps where ``mask`` is True. The
    predicted curve is drawn either:

    * on the dataset's union timestamp axis (using the pre-computed
      ``predictions`` argument) — the default when ``predictors`` /
      ``simulate_fn`` / ``solver`` are not all supplied; or
    * on a per-experiment dense grid (sorted unique union of the
      experiment's measured timestamps and ``n_dense_points`` linearly
      spaced samples between ``t0`` and ``t1``), re-simulated on the fly
      via ``simulate_fn`` and projected through
      ``dataset.state_to_output``. Activated when all three model
      components are passed; this is the recommended path because
      diffrax adaptive steps in the original ``predictions`` produce
      visibly piecewise-linear curves between sparse measurement times.

    The dense path runs ``simulate_fn`` eagerly (one call per plotted
    experiment, no jit). At ``max_experiments=6`` and a few hundred dense
    points this is cheap; if you bump either dramatically, expect a
    proportional cost.

    Parameters
    ----------
    predictions : tuple of arrays
        Output of ``predict_dataset``; one ``[N_b, T_b, D]`` entry per
        bucket. Used as the curve when the dense path is inactive, and
        always as the source of bucket-walk order so ``max_experiments``
        picks the same first ``k`` rows in either mode.
    dataset : hybridmodels.Dataset
        Used for ``output_channel_names``, ``state_to_output`` (dense
        path), and to walk the bucket-payload list in lockstep with
        ``predictions``.
    predictors, simulate_fn, solver : optional
        Trio that triggers the dense-grid re-simulation. Pass the same
        objects used for ``predict_dataset``. If any is omitted the
        helper falls back to plotting ``predictions`` as-is.
    n_dense_points : int, optional
        Target size of the linspace component of the dense grid. The
        actual per-experiment grid is the sorted-unique union of this
        linspace and the experiment's measured timestamps, so the
        plotted ts always passes through every measurement. Set to ``0``
        to disable the dense path even when the model trio is provided.
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
    state_to_output = dataset.state_to_output
    # Bundle the dense-path trio into a single optional so type narrowing
    # propagates from "is not None" into the loop body — keeping the three
    # individual args nullable (so callers can opt out by omission) while
    # giving the type checker a single witness to refine on.
    dense_bundle: tuple[Any, Callable[..., Any], Any] | None = (
        (predictors, simulate_fn, solver)
        if (
            predictors is not None
            and simulate_fn is not None
            and solver is not None
            and n_dense_points > 0
        )
        else None
    )

    rows: list[dict[str, np.ndarray]] = []
    for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
        N = bp.ts.shape[0]
        for i in range(N):
            if len(rows) >= max_experiments:
                break
            ts_obs = np.asarray(bp.ts[i])
            row: dict[str, np.ndarray] = {
                "ts_obs": ts_obs,
                "y_obs": np.asarray(bp.y_observed[i]),
                "mask": np.asarray(bp.mask[i], dtype=bool),
                "ts_pred": ts_obs,
                "y_pred": np.asarray(pred[i]),
            }
            if dense_bundle is not None and ts_obs.size >= 2:
                # Union of measured ts and a uniform linspace, deduped.
                # Including the measured ts guarantees the plotted curve
                # passes through every observation; the linspace fills
                # in the gaps so curvature between sparse measurements is
                # visible. ``np.unique`` also sorts.
                preds_d, sim_fn_d, solver_d = dense_bundle
                t0, t1 = float(ts_obs[0]), float(ts_obs[-1])
                dense_grid = np.linspace(t0, t1, n_dense_points)
                ts_dense = np.unique(np.concatenate([ts_obs, dense_grid]))
                ts_jax = jnp.asarray(ts_dense)
                cov_i = {k: v[i] for k, v in bp.covariates.items()}
                full_state = sim_fn_d(preds_d, ts_jax, cov_i, bp.y0[i], solver_d)
                y_dense = state_to_output(full_state)
                row["ts_pred"] = ts_dense
                row["y_pred"] = np.asarray(y_dense)
            rows.append(row)
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
            ax.plot(
                row["ts_pred"],
                row["y_pred"][:, c],
                label="predicted",
                linewidth=1.5,
            )
            mask_c = row["mask"][:, c]
            if bool(mask_c.any()):
                ax.scatter(
                    row["ts_obs"][mask_c],
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
