"""Compatibility shim: the per-channel diagnostics now live in ``hybridmodels``.

The example train scripts migrated to ``hybridmodels.compute_metrics`` /
``hybridmodels.print_metrics``, so the real implementation lives in the
library. This module keeps the old ``compute_diagnostics`` /
``print_diagnostics`` / ``ChannelDiagnostics`` names importable for
remaining example code; it is a thin re-export and adds nothing.
"""

from __future__ import annotations

from hybridmodels.metrics import ChannelMetrics as ChannelDiagnostics
from hybridmodels.metrics import compute_metrics as compute_diagnostics
from hybridmodels.metrics import print_metrics as print_diagnostics

__all__ = ["ChannelDiagnostics", "compute_diagnostics", "print_diagnostics"]