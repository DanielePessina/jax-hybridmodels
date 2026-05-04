"""Matplotlib style helper for the example scripts.

The settings mirror ``hybridcrystals/mpl_styles/plot.mplstyle`` (the source
package's thesis-figure style): high DPI, dual major+minor gridlines,
visible minor ticks. Default colour cycler is the qualitative ``Dark2``
palette (8 colours), set via the matplotlib ``cycler`` API.

Font registration is **not** copied across — bundling TTFs into this
example helper would force every reader to download fonts; the source
project keeps that machinery in its own package. We accept the matplotlib
default font here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler


def apply_default_style() -> None:
    """Install the project's example-default rcParams in place.

    Idempotent — calling repeatedly just re-applies the same dict.
    Intentionally side-effecting (mirrors ``plt.style.use``); examples
    invoke this once near the top of ``main()``.
    """
    dark2 = list(plt.get_cmap("Dark2").colors)

    plt.rcParams.update(
        {
            # Higher DPI for both interactive display and saved figures.
            # 500 is what hybridcrystals uses for thesis prints; we keep
            # parity so figures saved from examples look identical.
            "figure.dpi": 500,
            "savefig.dpi": 500,
            "savefig.bbox": "tight",
            # Major + minor gridlines, dashed and faint so they don't
            # dominate the data layer.
            "axes.grid": True,
            "axes.grid.which": "both",
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            # Make minor ticks visible — turning gridlines on for "both"
            # is silently ineffective if the minor ticks themselves are
            # hidden (matplotlib default).
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Default qualitative palette: Dark2 (ColorBrewer 8-colour).
            "axes.prop_cycle": cycler(color=dark2),
        }
    )
