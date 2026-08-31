"""Default plots for the example scripts.

``parity_plot`` scatters predicted against observed, one subplot per
channel. ``trajectory_plot`` draws per-experiment time series, one row per
experiment and one column per channel.

Both take a ``save_path`` and return the ``Figure``. Neither applies
rcParams, which would stomp on a caller passing their own ``ax``, so call
:func:`apply_default_style` first. To extend either, copy and tweak.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from hybridmodels.metrics import compute_metrics


def parity_diagnostics(predictions: Any, dataset: Any) -> dict[str, SimpleNamespace]:
    """Masked obs/pred pairs per channel, for :func:`parity_plot`.

    ``compute_metrics`` keeps only the summary stats; the scatter needs the
    raw value pairs, so re-walk the mask here. This is the block every
    example used to inline.
    """
    metrics = compute_metrics(predictions, dataset)
    out: dict[str, SimpleNamespace] = {}
    for d, name in enumerate(dataset.output_channel_names):
        obs_chunks: list = []
        pred_chunks: list = []
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
            mask = bp.mask[..., d]
            obs_chunks.append(bp.y_observed[..., d][mask])
            pred_chunks.append(pred[..., d][mask])
        obs = jnp.concatenate(obs_chunks) if obs_chunks else jnp.empty(0)
        pred = jnp.concatenate(pred_chunks) if pred_chunks else jnp.empty(0)
        m = metrics[name]
        out[name] = SimpleNamespace(
            name=name, n=m.n, obs=obs, pred=pred, r2=float(m.r2), rmse=float(m.rmse)
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
    state_to_output: Callable[[Any], Any] | None = None,
    solver: Any | None = None,
    n_dense_points: int = 200,
    max_experiments: int = 6,
    title: str | None = "Trajectories",
    save_path: str | Path | None = None,
) -> Figure:
    """Per-experiment predicted-vs-observed time series.

    One row per experiment, one column per channel. Observations are
    scattered at the bucket's union timestamps where ``mask`` is True.
    The predicted curve is drawn one of two ways:

    * on the union timestamp axis, straight from ``predictions``. The
      fallback when the model pieces are not supplied, and visibly
      piecewise-linear between sparse measurement times.
    * on a per-experiment dense grid, re-simulated through
      ``simulate_fn`` and projected with ``state_to_output``. Preferred,
      and active when ``predictors``, ``simulate_fn``,
      ``state_to_output`` and ``solver`` are all passed. It runs eagerly
      without jit, one call per plotted experiment, which is cheap at the
      default caps and scales linearly if you raise them.

    Parameters
    ----------
    predictions : tuple of arrays
        Output of ``predict_dataset``, one ``[N_b, T_b, D]`` entry per
        bucket. Always sets the bucket-walk order, so ``max_experiments``
        picks the same rows in either mode.
    dataset : hybridmodels.Dataset
        Read for ``output_channel_names`` and the bucket payloads.
    predictors, simulate_fn, state_to_output, solver : optional
        The model pieces that turn on dense re-simulation. Pass the same
        objects used for ``predict_dataset``; omitting any one falls back.
    n_dense_points : int, optional
        Size of the linspace part of the dense grid. The grid itself is
        that linspace unioned with the measured timestamps, so the curve
        always passes through every observation. ``0`` disables the dense
        path.
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
    channels = list(dataset.output_channel_names)
    # Bundled into one optional so the type checker narrows once here
    # rather than four times inside the loop.
    dense_bundle: tuple[Any, Callable[..., Any], Callable[[Any], Any], Any] | None = (
        (predictors, simulate_fn, state_to_output, solver)
        if (
            predictors is not None
            and simulate_fn is not None
            and state_to_output is not None
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
                # Measured ts keep the curve on every observation, the
                # linspace fills the gaps. ``np.unique`` also sorts.
                preds_d, sim_fn_d, state_to_output_d, solver_d = dense_bundle
                t0, t1 = float(ts_obs[0]), float(ts_obs[-1])
                dense_grid = np.linspace(t0, t1, n_dense_points)
                ts_dense = np.unique(np.concatenate([ts_obs, dense_grid]))
                ts_jax = jnp.asarray(ts_dense)
                cov_i = {k: v[i] for k, v in bp.covariates.items()}
                full_state = sim_fn_d(preds_d, ts_jax, cov_i, bp.y0[i], solver_d)
                y_dense = state_to_output_d(full_state)
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
