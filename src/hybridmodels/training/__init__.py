"""The two training loops.

:func:`train_with_optax` does gradient descent through the ODE solve. It
is the default, and the one to use for anything network-sized.

:func:`train_with_evosax` does gradient-free evolutionary search. Use it
for small kinetic predictors, roughly 4 to 10 trainable scalars, or when
the adjoint through the solve is unreliable.

They are separate entry points with no shared config. To combine them,
call evosax first and then feed its result to optax, passing the same
trainable mask to both.
"""

from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

__all__ = [
    "EvosaxTrainingConfig",
    "OptaxTrainingConfig",
    "train_with_evosax",
    "train_with_optax",
]
