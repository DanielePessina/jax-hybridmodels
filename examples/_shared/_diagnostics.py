"""Old names for the library's per-channel metrics.

``compute_diagnostics``, ``print_diagnostics`` and ``ChannelDiagnostics``
are ``jaxhybridmodels.compute_metrics``, ``print_metrics`` and
``ChannelMetrics``. New code should import those directly.
"""

from __future__ import annotations

from jaxhybridmodels.metrics import ChannelMetrics as ChannelDiagnostics
from jaxhybridmodels.metrics import compute_metrics as compute_diagnostics
from jaxhybridmodels.metrics import print_metrics as print_diagnostics

__all__ = ["ChannelDiagnostics", "compute_diagnostics", "print_diagnostics"]
