"""Reporting machinery the examples would otherwise duplicate.

A high-DPI matplotlib style, per-channel diagnostics (MSE/RMSE/MAE/R^2),
and two default plots. Deliberately outside ``jaxhybridmodels``, which ships
no plotting.

Example scripts run as ``__main__`` under ``examples/<scenario>/``, so they
put the parent directory on ``sys.path`` before importing this package.
"""

from __future__ import annotations

from ._diagnostics import compute_diagnostics, print_diagnostics
from ._plotting import parity_diagnostics, parity_plot, trajectory_plot
from ._style import apply_default_style

__all__ = [
    "apply_default_style",
    "compute_diagnostics",
    "parity_diagnostics",
    "parity_plot",
    "print_diagnostics",
    "trajectory_plot",
]
