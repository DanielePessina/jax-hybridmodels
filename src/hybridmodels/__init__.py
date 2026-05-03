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
    from hybridmodels.solver import SOLVER_REGISTRY, SolverConfig, register_solver

__all__: list[str] = [
    "BucketPayload",
    "ChannelObs",
    "Dataset",
    "Experiment",
    "SOLVER_REGISTRY",
    "SolverConfig",
    "make_dataset",
    "make_experiment",
    "register_solver",
    "split_dataset",
]

_EXPORTS: dict[str, str] = {
    "BucketPayload": "hybridmodels.data",
    "ChannelObs": "hybridmodels.data",
    "Dataset": "hybridmodels.data",
    "Experiment": "hybridmodels.data",
    "SOLVER_REGISTRY": "hybridmodels.solver",
    "SolverConfig": "hybridmodels.solver",
    "make_dataset": "hybridmodels.data",
    "make_experiment": "hybridmodels.data",
    "register_solver": "hybridmodels.solver",
    "split_dataset": "hybridmodels.data",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, name)
