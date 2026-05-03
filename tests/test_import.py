def test_package_imports() -> None:
    import hybridmodels

    expected = {
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
    }
    assert set(hybridmodels.__all__) == expected
    for name in expected:
        getattr(hybridmodels, name)
