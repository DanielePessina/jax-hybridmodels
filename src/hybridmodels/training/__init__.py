"""The training loops and ensemble helpers.

:func:`train_with_optax` does gradient descent through the ODE solve. It
is the default, and the one to use for anything network-sized.

:func:`train_with_evosax` does gradient-free evolutionary search. Use it
for small kinetic predictors, roughly 4 to 10 trainable scalars, or when
the adjoint through the solve is unreliable.

They are separate entry points with no shared config. To combine them,
call evosax first and then feed its result to optax, passing the same
trainable mask to both.

For ensembles of hybrid models:

- :func:`train_seed_ensemble` reuses the tournament's cheap warm-start
  ranking to pick which seeds deserve a full training run, then fully
  trains the best few.
- :func:`train_bootstrap_ensemble` trains one (or a seed-set of) model(s)
  per bootstrap resample of the experiments, for bagging-style ensembles.
- :func:`hybridmodels.ensemble_predictions` averages the members'
  predictions; :func:`hybridmodels.make_bootstrap_dataset` is the data
  half of the bagging recipe.
"""

from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from hybridmodels.training.optax import (
    OptaxTrainingConfig,
    train_bootstrap_ensemble,
    train_seed_ensemble,
    train_with_optax,
)

__all__ = [
    "EvosaxTrainingConfig",
    "OptaxTrainingConfig",
    "train_bootstrap_ensemble",
    "train_seed_ensemble",
    "train_with_evosax",
    "train_with_optax",
]
