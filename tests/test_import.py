"""The public surface resolves: the entry points examples and docs call work.

Deliberately small. This is not a re-assertion of ``jaxhybridmodels.__all__``
(a list that lives in the source and would have to be hand-kept in sync
here, which made the previous full-list version an implementation-pinning
test). Instead it pins the handful of stable, user-facing entry points —
the ones every example script calls — and checks they exist and are
callable, so a wholesale breakage of the public surface (a broken import,
a renamed trainer) fails loudly here rather than only inside the big
integration tests.
"""

from __future__ import annotations

import jaxhybridmodels


def test_core_entry_points_resolve_and_are_callable() -> None:
    core = [
        # training + prediction
        "train_with_optax",
        "train_seed_ensemble",
        "train_bootstrap_ensemble",
        "train_with_evosax",
        "predict_bucket",
        "predict_dataset",
        "predict_dense",
        "ensemble_predictions",
        # predictors + bounds
        "BoundScaler",
        "BoundedPredictor",
        "MLPPredictor",
        "KANPredictor",
        # data
        "make_experiment",
        "make_dataset",
        "split_dataset",
        "make_bootstrap_dataset",
        "ChannelObs",
        "Experiment",
        # solver + kernels (composed by custom loops)
        "SolverConfig",
        "build_bucket_step",
        "build_penalty_step",
        "build_score_bucket",
        "build_apply_update",
        # metrics + helpers
        "compute_metrics",
        "evaluate_predictor",
        "trainable_mask",
        "save_predictors",
        "load_predictors",
    ]
    missing = [name for name in core if not callable(getattr(jaxhybridmodels, name, None))]
    assert not missing, f"public entry points missing or not callable: {missing}"
