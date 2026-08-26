"""Hybrid models: user-written ODE physics with trainable function approximators inside.

A model here is not one object. It is three pieces the user holds
together:

- ``predictors``, a pytree of trainable ``eqx.Module`` leaves, usually a
  tuple of ``BoundedPredictor``. Each one maps named inputs to a physical
  quantity, keeping it inside a declared box.
- ``simulate_fn``, a pure function the user writes. It integrates the
  dynamics for one experiment and returns the full state trajectory. The
  user owns the physics; the framework owns batching, compilation, and
  gradients.
- ``SolverConfig``, the diffrax settings for that integration.

Data arrives as ``Experiment`` records with sparse per-channel
observations. ``make_dataset`` groups them into buckets of equal
timestamp-axis length so JAX can compile one kernel per bucket shape.
``train_with_optax`` fits by gradient descent, ``train_with_evosax`` by
evolutionary search, and both take the same boolean mask saying which
parameters may move. ``save_run`` writes the result to disk.

Every public name below is imported lazily. The ``TYPE_CHECKING`` block
gives type checkers and IDEs the real symbols, while ``__getattr__``
resolves a name to its module only when someone actually reads it. That
keeps ``import hybridmodels`` from pulling in diffrax, optax, evosax, and
rich on a run that needs none of them.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hybridmodels.data import (
        BucketPayload,
        ChannelObs,
        Dataset,
        Experiment,
        make_dataset,
        make_experiment,
        split_dataset,
    )
    from hybridmodels.losses import (
        LOSS_REGISTRY,
        bal_mle,
        bal_mse,
        masked_mle,
        masked_mse,
    )
    from hybridmodels.penalties import (
        box_violation,
        clip_ste,
        soft_logit,
        softclip,
    )
    from hybridmodels.prediction import predict_bucket, predict_dataset
    from hybridmodels.predictors import (
        BoundedPredictor,
        BoundScaler,
        KANPredictor,
        MLPPredictor,
        Predictor,
        reinitialize_pytree_with_key,
        reinitialize_with_key,
    )
    from hybridmodels.rng import fold
    from hybridmodels.serialise import (
        load_predictors,
        load_run,
        save_predictors,
        save_run,
    )
    from hybridmodels.solver import (
        ADJOINT_REGISTRY,
        SOLVER_REGISTRY,
        SolverConfig,
        register_adjoint,
        register_solver,
    )
    from hybridmodels.trainable import (
        default_trainable,
        freeze_modules_of_type,
        freeze_paths,
        freeze_where,
        trainable_mask,
    )
    from hybridmodels.training import (
        EvosaxTrainingConfig,
        OptaxTrainingConfig,
        train_with_evosax,
        train_with_optax,
    )
    from hybridmodels.transforms import (
        BOUND_TRANSFORMS,
        WARPS,
        BoundTransform,
        Warp,
        register_bound_transform,
        register_warp,
    )
    from hybridmodels.ui import (
        EvosaxUI,
        RichEvosaxUI,
        RichTrainingUI,
        SilentUI,
        TrainingUI,
    )

__all__: list[str] = [
    "ADJOINT_REGISTRY",
    "BOUND_TRANSFORMS",
    "BoundedPredictor",
    "BoundScaler",
    "BoundTransform",
    "box_violation",
    "BucketPayload",
    "ChannelObs",
    "clip_ste",
    "Dataset",
    "EvosaxTrainingConfig",
    "EvosaxUI",
    "Experiment",
    "KANPredictor",
    "LOSS_REGISTRY",
    "MLPPredictor",
    "OptaxTrainingConfig",
    "Predictor",
    "RichEvosaxUI",
    "RichTrainingUI",
    "SOLVER_REGISTRY",
    "SilentUI",
    "SolverConfig",
    "TrainingUI",
    "bal_mle",
    "bal_mse",
    "default_trainable",
    "fold",
    "freeze_modules_of_type",
    "freeze_paths",
    "freeze_where",
    "load_predictors",
    "load_run",
    "make_dataset",
    "make_experiment",
    "masked_mle",
    "masked_mse",
    "predict_bucket",
    "predict_dataset",
    "register_solver",
    "reinitialize_pytree_with_key",
    "reinitialize_with_key",
    "save_predictors",
    "save_run",
    "register_adjoint",
    "register_bound_transform",
    "register_warp",
    "soft_inverse",
    "soft_logit",
    "softclip",
    "split_dataset",
    "train_with_evosax",
    "train_with_optax",
    "trainable_mask",
    "Warp",
    "WARPS",
]

_EXPORTS: dict[str, str] = {
    "ADJOINT_REGISTRY": "hybridmodels.solver",
    "BOUND_TRANSFORMS": "hybridmodels.transforms",
    "BoundTransform": "hybridmodels.transforms",
    "BoundedPredictor": "hybridmodels.predictors",
    "BoundScaler": "hybridmodels.predictors",
    "box_violation": "hybridmodels.penalties",
    "BucketPayload": "hybridmodels.data",
    "ChannelObs": "hybridmodels.data",
    "clip_ste": "hybridmodels.penalties",
    "Dataset": "hybridmodels.data",
    "EvosaxTrainingConfig": "hybridmodels.training",
    "EvosaxUI": "hybridmodels.ui",
    "Experiment": "hybridmodels.data",
    "KANPredictor": "hybridmodels.predictors",
    "LOSS_REGISTRY": "hybridmodels.losses",
    "MLPPredictor": "hybridmodels.predictors",
    "OptaxTrainingConfig": "hybridmodels.training",
    "Predictor": "hybridmodels.predictors",
    "RichEvosaxUI": "hybridmodels.ui",
    "RichTrainingUI": "hybridmodels.ui",
    "SOLVER_REGISTRY": "hybridmodels.solver",
    "SilentUI": "hybridmodels.ui",
    "SolverConfig": "hybridmodels.solver",
    "TrainingUI": "hybridmodels.ui",
    "bal_mle": "hybridmodels.losses",
    "bal_mse": "hybridmodels.losses",
    "default_trainable": "hybridmodels.trainable",
    "fold": "hybridmodels.rng",
    "freeze_modules_of_type": "hybridmodels.trainable",
    "freeze_paths": "hybridmodels.trainable",
    "freeze_where": "hybridmodels.trainable",
    "load_predictors": "hybridmodels.serialise",
    "load_run": "hybridmodels.serialise",
    "make_dataset": "hybridmodels.data",
    "make_experiment": "hybridmodels.data",
    "masked_mle": "hybridmodels.losses",
    "masked_mse": "hybridmodels.losses",
    "predict_bucket": "hybridmodels.prediction",
    "predict_dataset": "hybridmodels.prediction",
    "register_solver": "hybridmodels.solver",
    "reinitialize_pytree_with_key": "hybridmodels.predictors",
    "reinitialize_with_key": "hybridmodels.predictors",
    "save_predictors": "hybridmodels.serialise",
    "save_run": "hybridmodels.serialise",
    "register_adjoint": "hybridmodels.solver",
    "register_bound_transform": "hybridmodels.transforms",
    "register_warp": "hybridmodels.transforms",
    "soft_inverse": "hybridmodels.penalties",
    "soft_logit": "hybridmodels.penalties",
    "softclip": "hybridmodels.penalties",
    "split_dataset": "hybridmodels.data",
    "train_with_evosax": "hybridmodels.training",
    "train_with_optax": "hybridmodels.training",
    "trainable_mask": "hybridmodels.trainable",
    "Warp": "hybridmodels.transforms",
    "WARPS": "hybridmodels.transforms",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, name)
