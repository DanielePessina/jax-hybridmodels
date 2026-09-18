"""The predictors a user builds with.

A *predictor* is an ``eqx.Module`` that maps an input array to the value
the user's ``simulate_fn`` needs, typically a rate or a coefficient
inside a differential equation. ``MLPPredictor``, ``KANPredictor`` and
``NeuralNPolynomial`` are the concrete families shipped here, and
``Predictor`` is the abstract marker to subclass for a fourth.

Wrap the network in ``BoundedPredictor`` to work in physical units. It
composes an inner predictor with two ``BoundScaler``s, one normalising
the named inputs and one squashing the output into its declared range,
so the network itself never handles a bound.

Multi-rate hybrid models pass a *tuple* of predictors to training and
unpack it inside ``simulate_fn``. There is no framework class for "two
rates" or "a list of rates", so each predictor's role is named where it
is used.
"""

from jaxhybridmodels.predictors.base import (
    BoundedPredictor,
    BoundScaler,
    Predictor,
    reinitialize_pytree_with_key,
    reinitialize_with_key,
)
from jaxhybridmodels.predictors.kan import KANPredictor
from jaxhybridmodels.predictors.mlp import MLPPredictor
from jaxhybridmodels.predictors.neural_npoly import NeuralNPolynomial

__all__ = [
    "BoundedPredictor",
    "BoundScaler",
    "KANPredictor",
    "MLPPredictor",
    "NeuralNPolynomial",
    "Predictor",
    "reinitialize_pytree_with_key",
    "reinitialize_with_key",
]
