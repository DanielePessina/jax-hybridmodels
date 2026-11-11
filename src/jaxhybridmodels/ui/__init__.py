"""Progress reporting for the training loops.

``TrainingUI`` and ``EvosaxUI`` are the callback protocols a training
loop calls into. ``RichTrainingUI`` and ``RichEvosaxUI`` are the live
terminal dashboards, selected by ``verbose=True`` on ``fit``.
``SilentUI`` implements both protocols and does nothing.

Pass ``ui=`` to a training function to use your own; it overrides
``verbose`` either way. See ``jaxhybridmodels.ui.base`` for the events.
"""

from jaxhybridmodels.ui.base import EvosaxUI, SilentUI, TrainingUI
from jaxhybridmodels.ui.evosax import RichEvosaxUI
from jaxhybridmodels.ui.optax import RichTrainingUI
from jaxhybridmodels.ui.plain import PlainEvosaxUI, PlainTrainingUI

__all__ = [
    "EvosaxUI",
    "PlainEvosaxUI",
    "PlainTrainingUI",
    "RichEvosaxUI",
    "RichTrainingUI",
    "SilentUI",
    "TrainingUI",
]
