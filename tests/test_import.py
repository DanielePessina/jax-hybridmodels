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
        "Predictor",
        "RatePair",
        "SOLVER_REGISTRY",
        "SolverConfig",
        "make_dataset",
        "make_experiment",
        "register_solver",
        "reinitialize_with_key",
        "split_dataset",
    }
    assert set(hybridmodels.__all__) == expected
    for name in expected:
        getattr(hybridmodels, name)
