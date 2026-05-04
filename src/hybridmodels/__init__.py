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
    from hybridmodels.prediction import predict_bucket, predict_dataset
    from hybridmodels.predictors import (
        BoundedPredictor,
        BoundScaler,
        CovariateSelector,
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
    from hybridmodels.solver import SOLVER_REGISTRY, SolverConfig, register_solver
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
    from hybridmodels.ui import (
        EvosaxUI,
        RichEvosaxUI,
        RichTrainingUI,
        SilentUI,
        TrainingUI,
    )

__all__: list[str] = [
    "BoundedPredictor",
    "BoundScaler",
    "BucketPayload",
    "ChannelObs",
    "CovariateSelector",
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
    "split_dataset",
    "train_with_evosax",
    "train_with_optax",
    "trainable_mask",
]

_EXPORTS: dict[str, str] = {
    "BoundedPredictor": "hybridmodels.predictors",
    "BoundScaler": "hybridmodels.predictors",
    "BucketPayload": "hybridmodels.data",
    "ChannelObs": "hybridmodels.data",
    "CovariateSelector": "hybridmodels.predictors",
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
    "split_dataset": "hybridmodels.data",
    "train_with_evosax": "hybridmodels.training",
    "train_with_optax": "hybridmodels.training",
    "trainable_mask": "hybridmodels.trainable",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, name)
