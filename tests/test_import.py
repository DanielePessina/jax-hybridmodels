def test_package_imports() -> None:
    import hybridmodels

    expected = {
        "BoundedPredictor",
        "BoundScaler",
        "BucketPayload",
        "ChannelObs",
        "CovariateSelector",
        "Dataset",
        "EvosaxUI",
        "Experiment",
        "LOSS_REGISTRY",
        "MLPPredictor",
        "OptaxTrainingConfig",
        "Predictor",
        "RatePair",
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
        "make_dataset",
        "make_experiment",
        "masked_mle",
        "masked_mse",
        "predict_bucket",
        "predict_dataset",
        "register_solver",
        "reinitialize_with_key",
        "split_dataset",
        "train_with_optax",
        "trainable_mask",
    }
    assert set(hybridmodels.__all__) == expected
    for name in expected:
        getattr(hybridmodels, name)
