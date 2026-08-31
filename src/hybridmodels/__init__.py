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

Every public name below is imported lazily: the ``TYPE_CHECKING`` block
gives type checkers the real symbols, and ``__getattr__`` resolves a name
to its module only when read. That keeps ``import hybridmodels`` from
pulling in diffrax, optax, evosax and rich on a run that needs none.
"""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

try:
    __version__ = metadata.version("hybridmodels")
except metadata.PackageNotFoundError:
    # Same policy as serialise._resolve_version: a checkout that has not
    # been ``uv sync``'d has no installed package metadata, and that must
    # not break an import that otherwise works from source.
    __version__ = "unknown"

if TYPE_CHECKING:
    from hybridmodels.data import (
        BucketPayload,
        ChannelObs,
        Dataset,
        Experiment,
        describe_buckets,
        make_bootstrap_dataset,
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
        resolve_loss_fn,
    )
    from hybridmodels.metrics import ChannelMetrics, compute_metrics, print_metrics
    from hybridmodels.penalties import (
        PenaltyPointSource,
        attach_penalty_state,
        bound_penalty,
        box_grid,
        box_violation,
        clip_ste,
        data_penalty_points,
        length_mask_keep,
        penalty_integral,
        penalty_vector_field,
        select_penalty_points,
        soft_inverse,
        soft_logit,
        softclip,
        strip_penalty_state,
        trajectory_saturation_penalty,
        validate_penalty_points,
    )
    from hybridmodels.prediction import (
        ensemble_predictions,
        evaluate_predictor,
        predict_bucket,
        predict_dataset,
        predict_dense,
    )
    from hybridmodels.predictors import (
        BoundedPredictor,
        BoundScaler,
        KANPredictor,
        MLPPredictor,
        NeuralNPolynomial,
        Predictor,
        reinitialize_pytree_with_key,
        reinitialize_with_key,
    )
    from hybridmodels.profiles import (
        constant_profile,
        piecewise_linear_profile,
        ramp_profile,
        step_profile,
    )
    from hybridmodels.rng import fold
    from hybridmodels.schedules import annealing_schedule
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
        count_trainable_params,
        default_trainable,
        freeze_modules_of_type,
        freeze_paths,
        freeze_where,
        frozen_default_mask,
        trainable_mask,
    )
    from hybridmodels.training import (
        EvosaxTrainingConfig,
        OptaxTrainingConfig,
        train_bootstrap_ensemble,
        train_seed_ensemble,
        train_with_evosax,
        train_with_optax,
    )
    from hybridmodels.training.evosax import register_algorithm
    from hybridmodels.training.kernels import (
        apply_length_mask,
        build_apply_update,
        build_bucket_step,
        build_penalty_step,
        build_score_bucket,
        predict_bucket_obs,
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
    "BucketPayload",
    "ChannelObs",
    "Dataset",
    "EvosaxTrainingConfig",
    "EvosaxUI",
    "Experiment",
    "KANPredictor",
    "LOSS_REGISTRY",
    "MLPPredictor",
    "NeuralNPolynomial",
    "OptaxTrainingConfig",
    "PenaltyPointSource",
    "Predictor",
    "RichEvosaxUI",
    "RichTrainingUI",
    "SOLVER_REGISTRY",
    "SilentUI",
    "SolverConfig",
    "TrainingUI",
    "bal_mle",
    "bal_mse",
    "attach_penalty_state",
    "bound_penalty",
    "box_grid",
    "box_violation",
    "build_apply_update",
    "build_bucket_step",
    "build_penalty_step",
    "build_score_bucket",
    "ChannelMetrics",
    "clip_ste",
    "data_penalty_points",
    "length_mask_keep",
    "select_penalty_points",
    "constant_profile",
    "penalty_integral",
    "penalty_vector_field",
    "compute_metrics",
    "count_trainable_params",
    "default_trainable",
    "describe_buckets",
    "ensemble_predictions",
    "evaluate_predictor",
    "fold",
    "freeze_modules_of_type",
    "freeze_paths",
    "freeze_where",
    "frozen_default_mask",
    "load_predictors",
    "load_run",
    "make_bootstrap_dataset",
    "make_dataset",
    "make_experiment",
    "masked_mle",
    "masked_mse",
    "predict_bucket",
    "predict_bucket_obs",
    "predict_dataset",
    "predict_dense",
    "print_metrics",
    "ramp_profile",
    "register_adjoint",
    "register_algorithm",
    "register_bound_transform",
    "register_solver",
    "register_warp",
    "reinitialize_pytree_with_key",
    "reinitialize_with_key",
    "resolve_loss_fn",
    "save_predictors",
    "save_run",
    "soft_inverse",
    "step_profile",
    "strip_penalty_state",
    "soft_logit",
    "softclip",
    "trajectory_saturation_penalty",
    "split_dataset",
    "train_bootstrap_ensemble",
    "train_seed_ensemble",
    "train_with_evosax",
    "train_with_optax",
    "trainable_mask",
    "validate_penalty_points",
    "apply_length_mask",
    "Warp",
    "WARPS",
    "piecewise_linear_profile",
    "annealing_schedule",
]

_EXPORTS: dict[str, str] = {
    "ADJOINT_REGISTRY": "hybridmodels.solver",
    "BOUND_TRANSFORMS": "hybridmodels.transforms",
    "BoundTransform": "hybridmodels.transforms",
    "annealing_schedule": "hybridmodels.schedules",
    "BoundedPredictor": "hybridmodels.predictors",
    "BoundScaler": "hybridmodels.predictors",
    "BucketPayload": "hybridmodels.data",
    "ChannelObs": "hybridmodels.data",
    "Dataset": "hybridmodels.data",
    "EvosaxTrainingConfig": "hybridmodels.training",
    "EvosaxUI": "hybridmodels.ui",
    "Experiment": "hybridmodels.data",
    "KANPredictor": "hybridmodels.predictors",
    "LOSS_REGISTRY": "hybridmodels.losses",
    "MLPPredictor": "hybridmodels.predictors",
    "NeuralNPolynomial": "hybridmodels.predictors",
    "OptaxTrainingConfig": "hybridmodels.training",
    "PenaltyPointSource": "hybridmodels.penalties",
    "Predictor": "hybridmodels.predictors",
    "RichEvosaxUI": "hybridmodels.ui",
    "RichTrainingUI": "hybridmodels.ui",
    "SOLVER_REGISTRY": "hybridmodels.solver",
    "SilentUI": "hybridmodels.ui",
    "SolverConfig": "hybridmodels.solver",
    "TrainingUI": "hybridmodels.ui",
    "bal_mle": "hybridmodels.losses",
    "bal_mse": "hybridmodels.losses",
    "attach_penalty_state": "hybridmodels.penalties",
    "bound_penalty": "hybridmodels.penalties",
    "box_grid": "hybridmodels.penalties",
    "box_violation": "hybridmodels.penalties",
    "build_apply_update": "hybridmodels.training.kernels",
    "build_bucket_step": "hybridmodels.training.kernels",
    "build_penalty_step": "hybridmodels.training.kernels",
    "build_score_bucket": "hybridmodels.training.kernels",
    "ChannelMetrics": "hybridmodels.metrics",
    "clip_ste": "hybridmodels.penalties",
    "data_penalty_points": "hybridmodels.penalties",
    "length_mask_keep": "hybridmodels.penalties",
    "select_penalty_points": "hybridmodels.penalties",
    "constant_profile": "hybridmodels.profiles",
    "penalty_integral": "hybridmodels.penalties",
    "penalty_vector_field": "hybridmodels.penalties",
    "compute_metrics": "hybridmodels.metrics",
    "count_trainable_params": "hybridmodels.trainable",
    "default_trainable": "hybridmodels.trainable",
    "describe_buckets": "hybridmodels.data",
    "ensemble_predictions": "hybridmodels.prediction",
    "evaluate_predictor": "hybridmodels.prediction",
    "fold": "hybridmodels.rng",
    "freeze_modules_of_type": "hybridmodels.trainable",
    "freeze_paths": "hybridmodels.trainable",
    "freeze_where": "hybridmodels.trainable",
    "frozen_default_mask": "hybridmodels.trainable",
    "load_predictors": "hybridmodels.serialise",
    "load_run": "hybridmodels.serialise",
    "make_bootstrap_dataset": "hybridmodels.data",
    "make_dataset": "hybridmodels.data",
    "make_experiment": "hybridmodels.data",
    "masked_mle": "hybridmodels.losses",
    "masked_mse": "hybridmodels.losses",
    "piecewise_linear_profile": "hybridmodels.profiles",
    "predict_bucket": "hybridmodels.prediction",
    "predict_bucket_obs": "hybridmodels.training.kernels",
    "predict_dataset": "hybridmodels.prediction",
    "predict_dense": "hybridmodels.prediction",
    "print_metrics": "hybridmodels.metrics",
    "ramp_profile": "hybridmodels.profiles",
    "register_adjoint": "hybridmodels.solver",
    "register_algorithm": "hybridmodels.training.evosax",
    "register_bound_transform": "hybridmodels.transforms",
    "register_solver": "hybridmodels.solver",
    "register_warp": "hybridmodels.transforms",
    "reinitialize_pytree_with_key": "hybridmodels.predictors",
    "reinitialize_with_key": "hybridmodels.predictors",
    "resolve_loss_fn": "hybridmodels.losses",
    "save_predictors": "hybridmodels.serialise",
    "save_run": "hybridmodels.serialise",
    "soft_inverse": "hybridmodels.penalties",
    "step_profile": "hybridmodels.profiles",
    "strip_penalty_state": "hybridmodels.penalties",
    "soft_logit": "hybridmodels.penalties",
    "softclip": "hybridmodels.penalties",
    "trajectory_saturation_penalty": "hybridmodels.penalties",
    "split_dataset": "hybridmodels.data",
    "train_bootstrap_ensemble": "hybridmodels.training",
    "train_seed_ensemble": "hybridmodels.training",
    "train_with_evosax": "hybridmodels.training",
    "train_with_optax": "hybridmodels.training",
    "trainable_mask": "hybridmodels.trainable",
    "validate_penalty_points": "hybridmodels.penalties",
    "apply_length_mask": "hybridmodels.training.kernels",
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
