def test_package_imports() -> None:
    import hybridmodels

    expected = {
        "BucketPayload",
        "ChannelObs",
        "Dataset",
        "Experiment",
        "make_dataset",
        "make_experiment",
        "split_dataset",
    }
    assert set(hybridmodels.__all__) == expected
    for name in expected:
        getattr(hybridmodels, name)
