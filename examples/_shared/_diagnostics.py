"""Compatibility shim: the per-channel diagnostics now live in ``jaxhybridmodels``.

The example train scripts migrated to ``jaxhybridmodels.compute_metrics`` /
``jaxhybridmodels.print_metrics``, so the real implementation lives in the
library. This module keeps the old ``compute_diagnostics`` /
``print_diagnostics`` / ``ChannelDiagnostics`` names importable for
remaining example code; it is a thin re-export and adds nothing.
"""

from __future__ import annotations

from jaxhybridmodels.metrics import ChannelMetrics as ChannelDiagnostics
from jaxhybridmodels.metrics import compute_metrics as compute_diagnostics
from jaxhybridmodels.metrics import print_metrics as print_diagnostics

__all__ = ["ChannelDiagnostics", "compute_diagnostics", "print_diagnostics"]