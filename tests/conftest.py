"""Session fixtures: datasets and one shared fit, so tests assert instead of retraining.

Helpers and model builders live in ``tests/_models.py``; import them with
``from _models import ...`` (``tests/`` has no ``__init__.py``, so pytest
puts the directory on ``sys.path``).
"""

from __future__ import annotations

import jax.random as jr
import pytest
from _models import decay_dataset, parameter_model, quiet, temp_rate

from jaxhybridmodels.training import TrainingPhase
from jaxhybridmodels.ui.testing import RecordingUI


@pytest.fixture(scope="session")
def decay_data():
    """Constant-rate decay, ``k = TRUE_RATE``, four experiments in two buckets."""
    return decay_dataset()


@pytest.fixture(scope="session")
def temp_data():
    """Decay whose rate depends on the ``temp`` covariate."""
    return decay_dataset(temp_rate)


@pytest.fixture(scope="session")
def fitted(decay_data):
    """A two-phase fit of the Parameter model, with every UI event recorded.

    The first phase trains on half the horizon, the second on all of it.
    Read-only: tests that need a different config train their own model.
    """
    ui = RecordingUI()
    config = quiet(
        phases=(
            TrainingPhase(25, lr=0.1, horizon=0.5),
            TrainingPhase(25, lr=0.05),
        )
    )
    result = parameter_model().fit(decay_data, config, key=jr.key(0), ui=ui)
    return result, ui
