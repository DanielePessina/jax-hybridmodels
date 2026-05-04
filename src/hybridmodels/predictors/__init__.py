"""Public predictor surface for v1 (SPEC §4.4 / §5.2).

`RatePair` was removed in this round (R-A6: multi-rate models compose by
unpacking the predictors tuple at the simulate_fn boundary, no framework
wrapper). `NeuralNPolynomial` is deferred to post-v1 per SPEC §2.3 — the
implementation file ``neural_npoly.py`` is retained in-tree as a future
re-introduction candidate but is **not** part of the v1 public API.
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
