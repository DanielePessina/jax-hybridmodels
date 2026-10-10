"""The public surface: what ``import jaxhybridmodels`` offers, and the release metadata.

Not a copy of ``__all__``. It pins the entry points the examples and docs
call, checks the lazy loader agrees with ``__all__``, and checks that
names removed in the API cleanup stay removed.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata

import pytest

import jaxhybridmodels

CORE = [
    "HybridModel",
    "Parameter",
    "Constant",
    "BoundScaler",
    "BoundedPredictor",
    "MLPPredictor",
    "KANPredictor",
    "make_experiment",
    "make_dataset",
    "split_dataset",
    "make_bootstrap_dataset",
    "ChannelObs",
    "SolverConfig",
    "OptaxTrainingConfig",
    "TrainingPhase",
    "EvosaxTrainingConfig",
    "train_with_optax",
    "train_with_evosax",
    "train_seed_ensemble",
    "train_bootstrap_ensemble",
    "predict_dataset",
    "predict_dense",
    "ensemble_predictions",
    "compute_metrics",
    "evaluate_predictor",
    "trainable_mask",
    "save_run",
    "load_run",
]


def test_core_entry_points_resolve_and_are_callable():
    missing = [name for name in CORE if not callable(getattr(jaxhybridmodels, name, None))]
    assert not missing


def test_all_matches_the_lazy_exports_and_every_export_resolves():
    assert set(jaxhybridmodels.__all__) == set(jaxhybridmodels._EXPORTS)
    for name, module in jaxhybridmodels._EXPORTS.items():
        assert getattr(jaxhybridmodels, name) is getattr(importlib.import_module(module), name)
    assert set(jaxhybridmodels.__all__) <= set(dir(jaxhybridmodels))


def test_submodules_resolve_as_attributes():
    # Lower-level pieces left the top level; they stay one attribute away.
    assert callable(jaxhybridmodels.penalties.box_grid)
    assert callable(jaxhybridmodels.losses.masked_mse)
    assert "Tsit5" in jaxhybridmodels.solver.SOLVER_REGISTRY


@pytest.mark.parametrize(
    "name",
    ["ConstantPredictor", "fit", "predict", "frozen_default_mask", "SOLVER_REGISTRY", "box_grid"],
)
def test_removed_and_demoted_names_are_not_top_level(name):
    with pytest.raises(AttributeError):
        getattr(jaxhybridmodels, name)


def test_distribution_metadata_matches_public_version():
    package = metadata.metadata("jax-hybridmodels")
    assert package["Name"] == "jax-hybridmodels"
    assert package["Version"] == "0.2.0b1"
    assert package["License-Expression"] == "BSD-3-Clause"
    assert "LICENSE" in (package.get_all("License-File") or [])
    assert "Development Status :: 4 - Beta" in (package.get_all("Classifier") or [])
    assert jaxhybridmodels.__version__ == package["Version"]
