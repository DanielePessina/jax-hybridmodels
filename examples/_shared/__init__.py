"""Shared example utilities — matplotlib style, diagnostics, default plots.

This sub-package is intentionally **outside** ``hybridmodels`` (per
``AGENTS.md`` "What's explicitly not your job: Adding plotting beyond what
an example script needs."). It bundles the small amount of reporting
machinery that every example otherwise duplicates: a high-DPI matplotlib
style, a per-channel diagnostics summary (MSE/RMSE/MAE/R^2), and two
default plots (parity, predicted-vs-observed time series).

Usage from an example script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from _shared import (
        apply_default_style,
        compute_diagnostics,
        print_diagnostics,
        parity_plot,
        trajectory_plot,
    )

The path tweak is needed because example scripts run as ``__main__``
under ``examples/<scenario>/`` and ``_shared`` is a sibling of those
scenario directories.
"""

from __future__ import annotations

from ._diagnostics import compute_diagnostics, print_diagnostics
from ._plotting import parity_plot, trajectory_plot
from ._style import apply_default_style

__all__ = [
    "apply_default_style",
    "compute_diagnostics",
    "parity_plot",
    "print_diagnostics",
    "trajectory_plot",
]
