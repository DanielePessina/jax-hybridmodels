"""Hybrid models: user-written ODE physics with trainable function approximators inside.

A model has four pieces, bound together by ``HybridModel``:

- ``predictors``, a pytree of ``eqx.Module`` leaves, usually a dict.
  ``Parameter`` is a trainable physical value, ``Constant`` a fixed one,
  and ``BoundedPredictor`` a network mapping named inputs to a physical
  quantity inside a declared range.
- ``simulate_fn``, a pure function the user writes. It integrates the
  dynamics for one experiment and returns the full state trajectory. The
  user owns the physics; the framework owns batching, compilation, and
  gradients.
- ``state_to_output``, the map from simulator state to observed channels.
- ``SolverConfig``, the diffrax settings for that integration.

Data arrives as ``Experiment`` records with sparse per-channel
observations. ``make_dataset`` groups them into buckets of equal
timestamp-axis length so JAX can compile one kernel per bucket shape.
``HybridModel.fit`` minimises an ``Objective`` (what to minimise) with an
``OptaxTrainingConfig`` or an ``EvosaxTrainingConfig`` (how to search);
``predict`` and ``simulate`` return results keyed by experiment and
channel name.

The top level holds that workflow. Lower-level pieces (training kernels,
penalties, losses, transforms, UI classes, registries) live in their
submodules, such as ``jaxhybridmodels.penalties``.

Every name is imported lazily: the ``TYPE_CHECKING`` block gives type
checkers the real symbols, and ``__getattr__`` resolves a name to its
module only when read. That keeps ``import jaxhybridmodels`` from pulling
in diffrax, optax, evosax and rich on a run that needs none.
"""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

try:
    __version__ = metadata.version("jax-hybridmodels")
except metadata.PackageNotFoundError:
    # Same policy as serialise._resolve_version: a checkout that has not
    # been ``uv sync``'d has no installed package metadata, and that must
    # not break an import that otherwise works from source.
    __version__ = "unknown"

if TYPE_CHECKING:
    # Redundant aliases mark these as re-exports for linters.
    from jaxhybridmodels import (
        data as data,
    )
    from jaxhybridmodels import (
        losses as losses,
    )
    from jaxhybridmodels import (
        metrics as metrics,
    )
    from jaxhybridmodels import (
        model as model,
    )
    from jaxhybridmodels import (
        objective as objective,
    )
    from jaxhybridmodels import (
        penalties as penalties,
    )
    from jaxhybridmodels import (
        prediction as prediction,
    )
    from jaxhybridmodels import (
        predictors as predictors,
    )
    from jaxhybridmodels import (
        profiles as profiles,
    )
    from jaxhybridmodels import (
        schedules as schedules,
    )
    from jaxhybridmodels import (
        serialise as serialise,
    )
    from jaxhybridmodels import (
        solver as solver,
    )
    from jaxhybridmodels import (
        trainable as trainable,
    )
    from jaxhybridmodels import (
        training as training,
    )
    from jaxhybridmodels import (
        transforms as transforms,
    )
    from jaxhybridmodels import (
        ui as ui,
    )
    from jaxhybridmodels.data import (
        BucketPayload,
        ChannelObs,
        Dataset,
        Experiment,
        describe_buckets,
        experiment_from_array,
        experiments_from_dataframe,
        make_bootstrap_dataset,
        make_dataset,
        make_experiment,
        split_dataset,
    )
    from jaxhybridmodels.metrics import ChannelMetrics, compute_metrics, print_metrics
    from jaxhybridmodels.model import (
        ExperimentPrediction,
        FitHistory,
        FitResult,
        HybridModel,
        PredictionResult,
    )
    from jaxhybridmodels.objective import Objective
    from jaxhybridmodels.prediction import (
        ensemble_predictions,
        evaluate_predictor,
        predict_dataset,
        predict_dense,
    )
    from jaxhybridmodels.predictors import (
        BoundedPredictor,
        BoundScaler,
        Constant,
        KANPredictor,
        MLPPredictor,
        NeuralNPolynomial,
        Parameter,
        Predictor,
    )
    from jaxhybridmodels.profiles import (
        constant_profile,
        piecewise_linear_profile,
        ramp_profile,
        step_profile,
    )
    from jaxhybridmodels.serialise import load_predictors, load_run, save_predictors, save_run
    from jaxhybridmodels.solver import SolverConfig
    from jaxhybridmodels.trainable import (
        count_trainable_params,
        freeze_mask,
        freeze_modules_of_type,
        freeze_paths,
        freeze_where,
        trainable_mask,
    )
    from jaxhybridmodels.training import (
        EvosaxTrainingConfig,
        OptaxTrainingConfig,
        TrainingPhase,
        train_bootstrap_ensemble,
        train_seed_ensemble,
        train_with_evosax,
        train_with_optax,
    )

_EXPORTS: dict[str, str] = {
    # Model workflow.
    "HybridModel": "jaxhybridmodels.model",
    "FitResult": "jaxhybridmodels.model",
    "FitHistory": "jaxhybridmodels.model",
    "PredictionResult": "jaxhybridmodels.model",
    "ExperimentPrediction": "jaxhybridmodels.model",
    # Predictors.
    "Parameter": "jaxhybridmodels.predictors",
    "Constant": "jaxhybridmodels.predictors",
    "BoundedPredictor": "jaxhybridmodels.predictors",
    "BoundScaler": "jaxhybridmodels.predictors",
    "Predictor": "jaxhybridmodels.predictors",
    "MLPPredictor": "jaxhybridmodels.predictors",
    "KANPredictor": "jaxhybridmodels.predictors",
    "NeuralNPolynomial": "jaxhybridmodels.predictors",
    # Data.
    "ChannelObs": "jaxhybridmodels.data",
    "Experiment": "jaxhybridmodels.data",
    "Dataset": "jaxhybridmodels.data",
    "BucketPayload": "jaxhybridmodels.data",
    "make_experiment": "jaxhybridmodels.data",
    "experiment_from_array": "jaxhybridmodels.data",
    "experiments_from_dataframe": "jaxhybridmodels.data",
    "make_dataset": "jaxhybridmodels.data",
    "split_dataset": "jaxhybridmodels.data",
    "make_bootstrap_dataset": "jaxhybridmodels.data",
    "describe_buckets": "jaxhybridmodels.data",
    "constant_profile": "jaxhybridmodels.profiles",
    "step_profile": "jaxhybridmodels.profiles",
    "ramp_profile": "jaxhybridmodels.profiles",
    "piecewise_linear_profile": "jaxhybridmodels.profiles",
    # Solving and training.
    "SolverConfig": "jaxhybridmodels.solver",
    "Objective": "jaxhybridmodels.objective",
    "OptaxTrainingConfig": "jaxhybridmodels.training",
    "TrainingPhase": "jaxhybridmodels.training",
    "EvosaxTrainingConfig": "jaxhybridmodels.training",
    "train_with_optax": "jaxhybridmodels.training",
    "train_with_evosax": "jaxhybridmodels.training",
    "train_seed_ensemble": "jaxhybridmodels.training",
    "train_bootstrap_ensemble": "jaxhybridmodels.training",
    "trainable_mask": "jaxhybridmodels.trainable",
    "freeze_mask": "jaxhybridmodels.trainable",
    "freeze_modules_of_type": "jaxhybridmodels.trainable",
    "freeze_paths": "jaxhybridmodels.trainable",
    "freeze_where": "jaxhybridmodels.trainable",
    "count_trainable_params": "jaxhybridmodels.trainable",
    # Prediction and evaluation.
    "predict_dataset": "jaxhybridmodels.prediction",
    "predict_dense": "jaxhybridmodels.prediction",
    "ensemble_predictions": "jaxhybridmodels.prediction",
    "evaluate_predictor": "jaxhybridmodels.prediction",
    "ChannelMetrics": "jaxhybridmodels.metrics",
    "compute_metrics": "jaxhybridmodels.metrics",
    "print_metrics": "jaxhybridmodels.metrics",
    # Persistence.
    "save_run": "jaxhybridmodels.serialise",
    "load_run": "jaxhybridmodels.serialise",
    "save_predictors": "jaxhybridmodels.serialise",
    "load_predictors": "jaxhybridmodels.serialise",
}

# Submodules reachable as attributes, e.g. ``jaxhybridmodels.penalties``,
# without an explicit ``import jaxhybridmodels.penalties``.
_SUBMODULES: frozenset[str] = frozenset(
    {
        "data",
        "losses",
        "metrics",
        "model",
        "objective",
        "penalties",
        "prediction",
        "predictors",
        "profiles",
        "schedules",
        "serialise",
        "solver",
        "trainable",
        "training",
        "transforms",
        "ui",
    }
)

# Kept literal so linters and type checkers see the re-exports; a test
# checks it matches _EXPORTS.
__all__: list[str] = [
    "HybridModel",
    "FitResult",
    "FitHistory",
    "PredictionResult",
    "ExperimentPrediction",
    "Parameter",
    "Constant",
    "BoundedPredictor",
    "BoundScaler",
    "Predictor",
    "MLPPredictor",
    "KANPredictor",
    "NeuralNPolynomial",
    "ChannelObs",
    "Experiment",
    "Dataset",
    "BucketPayload",
    "make_experiment",
    "experiment_from_array",
    "experiments_from_dataframe",
    "make_dataset",
    "split_dataset",
    "make_bootstrap_dataset",
    "describe_buckets",
    "constant_profile",
    "step_profile",
    "ramp_profile",
    "piecewise_linear_profile",
    "SolverConfig",
    "Objective",
    "OptaxTrainingConfig",
    "TrainingPhase",
    "EvosaxTrainingConfig",
    "train_with_optax",
    "train_with_evosax",
    "train_seed_ensemble",
    "train_bootstrap_ensemble",
    "trainable_mask",
    "freeze_mask",
    "freeze_modules_of_type",
    "freeze_paths",
    "freeze_where",
    "count_trainable_params",
    "predict_dataset",
    "predict_dense",
    "ensemble_predictions",
    "evaluate_predictor",
    "ChannelMetrics",
    "compute_metrics",
    "print_metrics",
    "save_run",
    "load_run",
    "save_predictors",
    "load_predictors",
]


if not TYPE_CHECKING:
    # Hidden from type checkers: a module __getattr__ would make every
    # attribute look valid, so a removed name would no longer be reported.

    def __getattr__(name: str) -> Any:
        if name in _SUBMODULES:
            return import_module(f"{__name__}.{name}")
        try:
            module_name = _EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})
