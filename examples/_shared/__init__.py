"""Plotting helpers the examples would otherwise duplicate.

A high-DPI matplotlib style and two default plots, parity and
trajectories. Deliberately outside ``jaxhybridmodels``, which ships no
plotting. Metrics come from the library itself (``hm.compute_metrics``).

Example scripts run as ``__main__`` under ``examples/<scenario>/``, so they
put the parent directory on ``sys.path`` before importing this package.
"""

from __future__ import annotations

from ._plotting import parity_diagnostics, parity_plot, trajectory_plot
from ._style import apply_default_style

__all__ = [
    "apply_default_style",
    "parity_diagnostics",
    "parity_plot",
    "trajectory_plot",
]
