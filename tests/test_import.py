def test_package_imports() -> None:
    import hybridmodels

    expected = {
        "BoundedPredictor",
        "BoundScaler",
        "BucketPayload",
        "ChannelObs",
        "CovariateSelector",
        "Dataset",
        "Experiment",
        "LOSS_REGISTRY",
        "MLPPredictor",
        "Predictor",
        "RatePair",
        "SOLVER_REGISTRY",
        "SolverConfig",
        "bal_mle",
        "bal_mse",
        "default_trainable",
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
        "trainable_mask",
    }
    assert set(hybridmodels.__all__) == expected
    for name in expected:
        getattr(hybridmodels, name)
