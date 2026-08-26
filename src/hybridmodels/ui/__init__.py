"""Progress reporting for the training loops.

``TrainingUI`` and ``EvosaxUI`` are the callback protocols a training
loop calls into. ``RichTrainingUI`` and ``RichEvosaxUI`` are the live
terminal dashboards, selected by ``verbose=True`` on a training config.
``SilentUI`` implements both protocols and does nothing.

Pass ``ui=`` to a training function to use your own; it overrides
``verbose`` either way. See ``hybridmodels.ui.base`` for the events.
"""

from hybridmodels.ui.base import EvosaxUI, SilentUI, TrainingUI
from hybridmodels.ui.evosax import RichEvosaxUI
from hybridmodels.ui.optax import RichTrainingUI

__all__ = ["EvosaxUI", "RichEvosaxUI", "RichTrainingUI", "SilentUI", "TrainingUI"]
