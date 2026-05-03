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
        "make_dataset",
        "make_experiment",
        "masked_mle",
        "masked_mse",
        "predict_bucket",
        "predict_dataset",
        "register_solver",
        "reinitialize_with_key",
        "split_dataset",
    }
    assert set(hybridmodels.__all__) == expected
    for name in expected:
        getattr(hybridmodels, name)
