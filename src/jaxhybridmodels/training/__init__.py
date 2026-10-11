"""The trainers behind ``HybridModel.fit``, and their configs.

:func:`train_with_optax` does gradient descent through the ODE solve. It
is the default, and the one to use for anything network-sized.

:func:`train_with_evosax` does gradient-free evolutionary search. Use it
for small kinetic predictors, roughly 4 to 10 trainable scalars, or when
the adjoint through the solve is unreliable.

Both take a ``HybridModel`` and the keywords of ``fit``, and return a
``FitResult``; ``model.fit(data, config, ...)`` picks one by the type of
``config``. What is minimised is a shared
:class:`~jaxhybridmodels.Objective`, passed to either as ``objective=``.
To combine them, fit with Evosax first and then fit its result's model
with Optax, passing the same objective and ``freeze`` to both.

Ensembles are ``HybridModel.fit_ensemble``; the kernels for a custom
training loop are in :mod:`jaxhybridmodels.training.kernels`.
"""

from jaxhybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from jaxhybridmodels.training.optax import OptaxTrainingConfig, TrainingPhase, train_with_optax

__all__ = [
    "EvosaxTrainingConfig",
    "OptaxTrainingConfig",
    "TrainingPhase",
    "train_with_evosax",
    "train_with_optax",
]
