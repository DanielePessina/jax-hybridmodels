"""Matplotlib style helper for the example scripts.

Mirrors the source package's thesis-figure style: high DPI, major and
minor gridlines, visible minor ticks, ``Dark2`` colour cycler. Font
registration is left out, so examples use the matplotlib default rather
than making every reader download a TTF.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler


def apply_default_style() -> None:
    """Install the example-default rcParams in place.

    Side-effecting, like ``plt.style.use``, and idempotent. Examples call
    it once near the top of ``main()``.
    """
    dark2 = list(plt.get_cmap("Dark2").colors)

    plt.rcParams.update(
        {
            # 500 matches the source package's thesis prints.
            "figure.dpi": 500,
            "savefig.dpi": 500,
            "savefig.bbox": "tight",
            # Faint dashed gridlines, so they do not dominate the data.
            "axes.grid": True,
            "axes.grid.which": "both",
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            # Gridlines for "both" do nothing unless the minor ticks are
            # visible, and matplotlib hides them by default.
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.prop_cycle": cycler(color=dark2),
        }
    )
