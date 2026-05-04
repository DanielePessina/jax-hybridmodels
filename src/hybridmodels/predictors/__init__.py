"""Public predictor surface.

A *predictor* is an ``eqx.Module`` mapping a covariate vector to a real
or vector output that the user's ``simulate_fn`` consumes (typically as
a coefficient or rate inside a differential equation). The
``BoundedPredictor`` wrapper composes any inner predictor with a
``BoundScaler`` to constrain the output to a physically meaningful
range; ``MLPPredictor`` and ``KANPredictor`` are the two concrete
inner-predictor families shipped here.

Multi-rate hybrid models compose by passing a *tuple* of predictors to
training and unpacking that tuple inside ``simulate_fn`` — the framework
deliberately does not provide a single wrapper class for "two rates" or
"a list of rates", because the JIT-friendly thing is for the user's
simulator to know exactly what each predictor is for.

The file ``neural_npoly.py`` ships an experimental
``NeuralNPolynomial`` that is **not** re-exported here; it is held
in-tree as a future candidate but is not yet part of the supported
public API.
"""

from hybridmodels.predictors.base import (
    BoundedPredictor,
    BoundScaler,
    CovariateSelector,
    Predictor,
    reinitialize_pytree_with_key,
    reinitialize_with_key,
)
from hybridmodels.predictors.kan import KANPredictor
from hybridmodels.predictors.mlp import MLPPredictor

__all__ = [
    "BoundedPredictor",
    "BoundScaler",
    "CovariateSelector",
    "KANPredictor",
    "MLPPredictor",
    "Predictor",
    "reinitialize_pytree_with_key",
    "reinitialize_with_key",
]
