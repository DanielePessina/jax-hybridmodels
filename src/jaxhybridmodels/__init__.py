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
to its module only when read. That keeps ``import jaxhybridmodels`` from
pulling in diffrax, optax, evosax and rich on a run that needs none.
"""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

try:
    __version__ = metadata.version("jaxhybridmodels")
except metadata.PackageNotFoundError:
    # Same policy as serialise._resolve_version: a checkout that has not
    # been ``uv sync``'d has no installed package metadata, and that must
    # not break an import that otherwise works from source.
    __version__ = "unknown"

if TYPE_CHECKING:
    from jaxhybridmodels.data import (
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
    from jaxhybridmodels.losses import (
        LOSS_REGISTRY,
        bal_mle,
        bal_mse,
        masked_mle,
        masked_mse,
        resolve_loss_fn,
    )
    from jaxhybridmodels.metrics import ChannelMetrics, compute_metrics, print_metrics
    from jaxhybridmodels.penalties import (
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
    from jaxhybridmodels.prediction import (
        ensemble_predictions,
        evaluate_predictor,
        predict_bucket,
        predict_dataset,
        predict_dense,
    )
    from jaxhybridmodels.predictors import (
        BoundedPredictor,
        BoundScaler,
        KANPredictor,
        MLPPredictor,
        NeuralNPolynomial,
        Predictor,
        reinitialize_pytree_with_key,
        reinitialize_with_key,
    )
    from jaxhybridmodels.profiles import (
        constant_profile,
        piecewise_linear_profile,
        ramp_profile,
        step_profile,
    )
    from jaxhybridmodels.rng import fold
    from jaxhybridmodels.schedules import annealing_schedule
    from jaxhybridmodels.serialise import (
        load_predictors,
        load_run,
        save_predictors,
        save_run,
    )
    from jaxhybridmodels.solver import (
        ADJOINT_REGISTRY,
        SOLVER_REGISTRY,
        SolverConfig,
        register_adjoint,
        register_solver,
    )
    from jaxhybridmodels.trainable import (
        count_trainable_params,
        default_trainable,
        freeze_modules_of_type,
        freeze_paths,
        freeze_where,
        frozen_default_mask,
        trainable_mask,
    )
    from jaxhybridmodels.training import (
        EvosaxTrainingConfig,
        OptaxTrainingConfig,
        train_bootstrap_ensemble,
        train_seed_ensemble,
        train_with_evosax,
        train_with_optax,
    )
    from jaxhybridmodels.training.evosax import register_algorithm
    from jaxhybridmodels.training.kernels import (
        apply_length_mask,
        build_apply_update,
        build_bucket_step,
        build_penalty_step,
        build_score_bucket,
        predict_bucket_obs,
    )
    from jaxhybridmodels.transforms import (
        BOUND_TRANSFORMS,
        WARPS,
        BoundTransform,
        Warp,
        register_bound_transform,
        register_warp,
    )
    from jaxhybridmodels.ui import (
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
    "ADJOINT_REGISTRY": "jaxhybridmodels.solver",
    "BOUND_TRANSFORMS": "jaxhybridmodels.transforms",
    "BoundTransform": "jaxhybridmodels.transforms",
    "annealing_schedule": "jaxhybridmodels.schedules",
    "BoundedPredictor": "jaxhybridmodels.predictors",
    "BoundScaler": "jaxhybridmodels.predictors",
    "BucketPayload": "jaxhybridmodels.data",
    "ChannelObs": "jaxhybridmodels.data",
    "Dataset": "jaxhybridmodels.data",
    "EvosaxTrainingConfig": "jaxhybridmodels.training",
    "EvosaxUI": "jaxhybridmodels.ui",
    "Experiment": "jaxhybridmodels.data",
    "KANPredictor": "jaxhybridmodels.predictors",
    "LOSS_REGISTRY": "jaxhybridmodels.losses",
    "MLPPredictor": "jaxhybridmodels.predictors",
    "NeuralNPolynomial": "jaxhybridmodels.predictors",
    "OptaxTrainingConfig": "jaxhybridmodels.training",
    "PenaltyPointSource": "jaxhybridmodels.penalties",
    "Predictor": "jaxhybridmodels.predictors",
    "RichEvosaxUI": "jaxhybridmodels.ui",
    "RichTrainingUI": "jaxhybridmodels.ui",
    "SOLVER_REGISTRY": "jaxhybridmodels.solver",
    "SilentUI": "jaxhybridmodels.ui",
    "SolverConfig": "jaxhybridmodels.solver",
    "TrainingUI": "jaxhybridmodels.ui",
    "bal_mle": "jaxhybridmodels.losses",
    "bal_mse": "jaxhybridmodels.losses",
    "attach_penalty_state": "jaxhybridmodels.penalties",
    "bound_penalty": "jaxhybridmodels.penalties",
    "box_grid": "jaxhybridmodels.penalties",
    "box_violation": "jaxhybridmodels.penalties",
    "build_apply_update": "jaxhybridmodels.training.kernels",
    "build_bucket_step": "jaxhybridmodels.training.kernels",
    "build_penalty_step": "jaxhybridmodels.training.kernels",
    "build_score_bucket": "jaxhybridmodels.training.kernels",
    "ChannelMetrics": "jaxhybridmodels.metrics",
    "clip_ste": "jaxhybridmodels.penalties",
    "data_penalty_points": "jaxhybridmodels.penalties",
    "length_mask_keep": "jaxhybridmodels.penalties",
    "select_penalty_points": "jaxhybridmodels.penalties",
    "constant_profile": "jaxhybridmodels.profiles",
    "penalty_integral": "jaxhybridmodels.penalties",
    "penalty_vector_field": "jaxhybridmodels.penalties",
    "compute_metrics": "jaxhybridmodels.metrics",
    "count_trainable_params": "jaxhybridmodels.trainable",
    "default_trainable": "jaxhybridmodels.trainable",
    "describe_buckets": "jaxhybridmodels.data",
    "ensemble_predictions": "jaxhybridmodels.prediction",
    "evaluate_predictor": "jaxhybridmodels.prediction",
    "fold": "jaxhybridmodels.rng",
    "freeze_modules_of_type": "jaxhybridmodels.trainable",
    "freeze_paths": "jaxhybridmodels.trainable",
    "freeze_where": "jaxhybridmodels.trainable",
    "frozen_default_mask": "jaxhybridmodels.trainable",
    "load_predictors": "jaxhybridmodels.serialise",
    "load_run": "jaxhybridmodels.serialise",
    "make_bootstrap_dataset": "jaxhybridmodels.data",
    "make_dataset": "jaxhybridmodels.data",
    "make_experiment": "jaxhybridmodels.data",
    "masked_mle": "jaxhybridmodels.losses",
    "masked_mse": "jaxhybridmodels.losses",
    "piecewise_linear_profile": "jaxhybridmodels.profiles",
    "predict_bucket": "jaxhybridmodels.prediction",
    "predict_bucket_obs": "jaxhybridmodels.training.kernels",
    "predict_dataset": "jaxhybridmodels.prediction",
    "predict_dense": "jaxhybridmodels.prediction",
    "print_metrics": "jaxhybridmodels.metrics",
    "ramp_profile": "jaxhybridmodels.profiles",
    "register_adjoint": "jaxhybridmodels.solver",
    "register_algorithm": "jaxhybridmodels.training.evosax",
    "register_bound_transform": "jaxhybridmodels.transforms",
    "register_solver": "jaxhybridmodels.solver",
    "register_warp": "jaxhybridmodels.transforms",
    "reinitialize_pytree_with_key": "jaxhybridmodels.predictors",
    "reinitialize_with_key": "jaxhybridmodels.predictors",
    "resolve_loss_fn": "jaxhybridmodels.losses",
    "save_predictors": "jaxhybridmodels.serialise",
    "save_run": "jaxhybridmodels.serialise",
    "soft_inverse": "jaxhybridmodels.penalties",
    "step_profile": "jaxhybridmodels.profiles",
    "strip_penalty_state": "jaxhybridmodels.penalties",
    "soft_logit": "jaxhybridmodels.penalties",
    "softclip": "jaxhybridmodels.penalties",
    "trajectory_saturation_penalty": "jaxhybridmodels.penalties",
    "split_dataset": "jaxhybridmodels.data",
    "train_bootstrap_ensemble": "jaxhybridmodels.training",
    "train_seed_ensemble": "jaxhybridmodels.training",
    "train_with_evosax": "jaxhybridmodels.training",
    "train_with_optax": "jaxhybridmodels.training",
    "trainable_mask": "jaxhybridmodels.trainable",
    "validate_penalty_points": "jaxhybridmodels.penalties",
    "apply_length_mask": "jaxhybridmodels.training.kernels",
    "Warp": "jaxhybridmodels.transforms",
    "WARPS": "jaxhybridmodels.transforms",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, name)
