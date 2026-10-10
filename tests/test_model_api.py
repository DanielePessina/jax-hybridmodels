"""User workflows: fixed physics, learned parameters, irregular prediction, and selection."""

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import jaxhybridmodels as hm


def _solver():
    return hm.SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=1024, dt0=0.05)


def _simulate(predictors, ts, covariates, y0, solver):
    del covariates
    rate = predictors["rate"].value
    fixed = predictors["fixed"].value
    return solver.diffeqsolve(diffrax.ODETerm(lambda t, y, args: -rate * y + fixed), ts, y0).ys


def _model(rate=0.3):
    return hm.HybridModel(
        predictors={
            "rate": hm.Parameter(rate, bounds=(0.0, None)),
            "fixed": hm.Constant(0.0),
        },
        simulate_fn=_simulate,
        state_to_output=lambda y: y,
        output_channel_names=("concentration",),
        solver=_solver(),
    )


def _dataset(rate=0.8):
    return hm.make_dataset(
        [
            hm.make_experiment(
                y0=[1.0],
                t0=0.0,
                exp_id=name,
                channels={"concentration": hm.ChannelObs(ts, jnp.exp(-rate * ts))},
            )
            for name, ts in [("long", jnp.array([0.5, 1.0, 2.0])), ("short", jnp.array([1.0, 2.0]))]
        ],
        output_channel_names=("concentration",),
    )


def test_fixed_scalar_tuple_and_list_survive_training_and_restart():
    constant = hm.Constant((1.0, [2.0, 3.0], (4.0, 5.0)))
    assert isinstance(constant.value, tuple)
    assert constant.value[0].shape == ()
    assert constant.value[1].shape == (2,)
    assert not any(jax.tree_util.tree_leaves(hm.trainable_mask(constant)))
    fresh = hm.predictors.reinitialize_pytree_with_key({"c": constant}, jr.key(0))["c"]
    for a, b in zip(
        jax.tree_util.tree_leaves(constant), jax.tree_util.tree_leaves(fresh), strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "bounds,value",
    [
        ((None, None), -2.0),
        ((0.0, None), 0.1),
        ((None, 1.0), 0.9),
        ((0.0, 1.0), 0.0001),
        ((-np.inf, np.inf), 2.0),
    ],
)
def test_parameter_initializes_in_physical_units_and_differentiates(bounds, value):
    parameter = hm.Parameter(value, bounds=bounds)
    np.testing.assert_allclose(parameter.value, value, rtol=1e-5, atol=1e-7)
    mask = hm.trainable_mask(parameter)
    assert mask.latent is True
    assert mask.scaler.temperature is False
    gradient = eqx.filter_grad(lambda p: p.value)(parameter)
    assert jnp.isfinite(gradient.latent).all()
    assert float(jnp.abs(gradient.latent).sum()) > 0
    np.testing.assert_allclose(eqx.filter_jit(lambda p: p.value)(parameter), value, rtol=1e-5)


def test_mixed_array_bounds_have_finite_gradients_and_zero_unbounded_penalty():
    parameter = hm.Parameter(
        [[0.2, 0.4], [-2.0, 0.8]], bounds=[(0.0, None), (None, 1.0), (None, None), (0.0, 1.0)]
    )
    assert parameter.value.shape == (2, 2)
    gradient = eqx.filter_grad(lambda p: p.value.sum())(parameter)
    assert jnp.isfinite(gradient.latent).all()
    unbounded = hm.Parameter([-100.0, 100.0])
    assert float(unbounded.scaler.saturation(unbounded.latent)) == 0.0


def test_log_warp_parameter_round_trips():
    rate = hm.Parameter(1e-3, bounds=(1e-6, 1e2), warp="log10")
    np.testing.assert_allclose(rate.value, 1e-3, rtol=1e-5)


def test_parameter_restart_draws_a_new_latent_and_keeps_the_scaler():
    rate = hm.Parameter(0.3, bounds=(0.0, 1.0))
    fresh = rate.initialized_with_key(jr.key(0))
    assert float(fresh.latent[0]) != float(rate.latent[0])
    assert fresh.scaler.bounds == rate.scaler.bounds
    assert float(fresh.scaler.temperature) == float(rate.scaler.temperature)


def test_network_open_input_scaling_is_identity():
    class Identity(hm.Predictor):
        in_size: int = eqx.field(static=True, default=2)
        out_size: int = eqx.field(static=True, default=2)

        def __call__(self, x):
            return x

    network = hm.BoundedPredictor(
        inner=Identity(), input_keys=("a", "b"), input_bounds=((0.0, None), (None, None))
    )
    np.testing.assert_allclose(network({"a": -3.0, "b": 4.0}), [-3.0, 4.0])


def test_phase_config_roundtrips(tmp_path):
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(3, lr=0.1, horizon=0.5), hm.TrainingPhase(4, lr=0.02))
    )
    assert config.total_steps == 7
    assert [phase.horizon for phase in config.phases] == [0.5, 1.0]
    single = hm.OptaxTrainingConfig(steps=3, lr=0.1)
    assert single.phases == (hm.TrainingPhase(3, lr=0.1),)
    model = _model()
    hm.save_run(tmp_path, predictors=model.predictors, solver=model.solver, optax_config=config)
    loaded = hm.load_run(
        tmp_path, predictors_template=model.predictors, optax_cls=hm.OptaxTrainingConfig
    )
    assert loaded["optax_config"].phases == config.phases
    np.testing.assert_allclose(loaded["predictors"]["rate"].value, 0.3, rtol=1e-6)
    with pytest.raises(ValueError, match="steps= or phases="):
        hm.OptaxTrainingConfig(steps=2, phases=(hm.TrainingPhase(3),))


def test_fit_reduces_loss_and_preserves_fixed_values_and_original_model():
    model = _model()
    dataset = _dataset()
    result = model.fit(
        dataset, hm.OptaxTrainingConfig(steps=35, lr=0.06, verbose=False), key=jr.key(1)
    )
    assert result.history.loss[-1] < result.history.loss[0] * 0.1
    assert result.best_step is not None
    assert result.stop_reason == "budget_exhausted"
    assert result.history.penalty == (0.0,) * 35
    np.testing.assert_allclose(model.predictors["rate"].value, 0.3, rtol=1e-6)
    assert result.model.predictors["fixed"].value == 0.0
    assert abs(float(result.model.predictors["rate"].value) - 0.8) < 0.08


def test_named_predictions_restore_experiment_order_and_initial_time():
    model = _model(0.8)
    dataset = _dataset()
    prediction = model.predict(dataset)
    assert list(prediction) == ["long", "short"]
    short = prediction["short"]
    np.testing.assert_allclose(
        short.outputs["concentration"], jnp.exp(-0.8 * short.times), rtol=1e-4
    )
    assert not short.observation_mask["concentration"][0]
    assert jnp.isnan(short.observed["concentration"][0])
    assert len(prediction.buckets) == 2


def test_simulation_and_dense_grid_keep_original_initial_time():
    model = _model(0.8)
    trajectory = model.simulate(ts=[1.0, 2.0], y0=[1.0], t0=0.0)
    np.testing.assert_allclose(
        trajectory.outputs["concentration"], jnp.exp(-0.8 * jnp.array([1.0, 2.0])), rtol=1e-4
    )
    dense = hm.predict_dense(
        model.predictors,
        _dataset(),
        ts_grid=jnp.array([1.0, 2.0]),
        simulate_fn=model.simulate_fn,
        state_to_output=model.state_to_output,
        solver=model.solver,
    )
    for bucket in dense:
        np.testing.assert_allclose(bucket[0, :, 0], trajectory.outputs["concentration"], rtol=1e-4)


def test_validation_selects_checkpoint_and_stops_for_validation_patience():
    model = _model()
    config = hm.OptaxTrainingConfig(steps=20, lr=0.03, patience=2, verbose=False)
    # Validation prefers the original value; fitting training data moves away.
    result = model.fit(_dataset(), config, validation=_dataset(0.3), key=jr.key(2))
    assert result.stop_reason == "patience"
    assert result.best_step == 0
    assert len(result.history.validation_loss) == 3
    np.testing.assert_allclose(result.model.predictors["rate"].value, 0.3, rtol=1e-6)


def test_value_and_input_validation():
    with pytest.raises(ValueError, match="strictly inside"):
        hm.Parameter(0.0, bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="nonempty and finite"):
        hm.Parameter(float("nan"))
    with pytest.raises(ValueError, match="t0"):
        _model().simulate(ts=[1.0, 2.0], y0=[1.0], t0=1.5)


def test_fit_histories_separate_data_trajectory_and_fixed_penalties():
    def constant_trajectory_penalty(state, bp):
        return jnp.asarray(2.0)

    def constant_fixed_penalty(predictors, points):
        return jnp.asarray(3.0)

    config = hm.OptaxTrainingConfig(
        steps=2,
        lr=0.01,
        penalty_weight=0.25,
        verbose=False,
        trajectory_penalty_fn=constant_trajectory_penalty,
        trajectory_penalty_weight=0.5,
        penalty_fn=constant_fixed_penalty,
    )
    result = _model().fit(_dataset(), config, key=jr.key(3))
    # The tracked loss carries the weighted trajectory penalty (0.5 * 2.0);
    # the fixed-point penalty is recorded separately, already weighted.
    plain = _model().fit(
        _dataset(), hm.OptaxTrainingConfig(steps=2, lr=0.01, verbose=False), key=jr.key(3)
    )
    np.testing.assert_allclose(result.history.loss[0] - plain.history.loss[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(result.history.penalty, 0.75, atol=1e-6)


def test_population_fit_uses_value_parameters_and_preserves_constants():
    model = hm.HybridModel(
        predictors={
            "theta": hm.Parameter([0.0, 0.0]),
            "fixed": hm.Constant((1.0, 2.0)),
        },
        simulate_fn=lambda p, ts, c, y0, solver: jnp.broadcast_to(p["theta"].value, (len(ts), 2)),
        state_to_output=lambda y: y,
        output_channel_names=("a", "b"),
        solver=_solver(),
    )
    dataset = hm.make_dataset(
        [
            hm.make_experiment(
                y0=[0.0, 0.0],
                channels={
                    "a": hm.ChannelObs([0.0, 1.0], [1.0, 1.0]),
                    "b": hm.ChannelObs([0.0, 1.0], [-1.0, -1.0]),
                },
            )
        ],
        output_channel_names=("a", "b"),
    )
    result = model.fit(
        dataset,
        hm.EvosaxTrainingConfig(
            population_size=16,
            num_generations=30,
            sigma_init=0.5,
            verbose=False,
        ),
        key=jr.key(4),
    )
    assert result.history.best_fitness[-1] < 0.05
    assert not result.history.loss
    assert result.model.predictors["fixed"].value[0] == 1.0
    assert len(result.history.best_fitness) == 30


def test_all_fixed_model_cannot_be_fitted():
    from dataclasses import replace

    model = replace(_model(), predictors={"rate": hm.Constant(0.3), "fixed": hm.Constant(0.0)})
    with pytest.raises(ValueError, match="no trainable"):
        model.fit(_dataset(), hm.OptaxTrainingConfig(steps=2), key=jr.key(0))


def test_initial_time_and_duplicate_experiment_ids_raise():
    exp = hm.make_experiment(
        y0=[1.0],
        t0=2.0,
        channels={"concentration": hm.ChannelObs([1.0, 3.0], [0.8, 0.6])},
    )
    with pytest.raises(ValueError, match="t0"):
        hm.make_dataset([exp], output_channel_names=("concentration",))
    dataset = _dataset()
    duplicate = hm.make_dataset(
        [dataset._experiments[0], dataset._experiments[0]], output_channel_names=("concentration",)
    )
    with pytest.raises(ValueError, match="Duplicate"):
        _model().predict(duplicate)


def test_dense_prediction_reuses_compilation_for_same_shapes():
    calls = []

    def simulate(predictors, ts, covariates, y0, solver):
        calls.append(ts.shape)
        return _simulate(predictors, ts, covariates, y0, solver)

    model = _model()
    dataset = _dataset()
    kwargs: dict[str, Any] = dict(
        simulate_fn=simulate, state_to_output=model.state_to_output, solver=model.solver
    )
    hm.predict_dense(model.predictors, dataset, n_points=10, **kwargs)
    compiled = len(calls)
    hm.predict_dense(model.predictors, dataset, n_points=10, **kwargs)
    assert compiled > 0 and len(calls) == compiled


def test_fractional_initial_time_with_integer_observations_and_state():
    experiment = hm.make_experiment(
        y0=[1], t0=0.5, channels={"concentration": hm.ChannelObs([1, 2], [0.8, 0.6])}
    )
    dataset = hm.make_dataset([experiment], output_channel_names=("concentration",))
    assert jnp.issubdtype(experiment.y0.dtype, jnp.floating)
    np.testing.assert_allclose(dataset.bucket_payloads[0].ts[0], [0.5, 1.0, 2.0])


def test_table_export_preserves_experiment_channel_and_observation_identity():
    pytest.importorskip("pandas")
    prediction = _model().predict(_dataset())
    frame = prediction.to_dataframe()
    assert list(frame.columns) == [
        "experiment_id",
        "time",
        "channel",
        "prediction",
        "observed",
        "is_observed",
    ]
    assert frame["experiment_id"].drop_duplicates().tolist() == ["long", "short"]
    assert frame["channel"].unique().tolist() == ["concentration"]
    assert frame["is_observed"].sum() == 5
    assert frame["observed"].isna().sum() == 2


def test_open_bounds_save_as_standard_json_and_load_with_matching_template(tmp_path):
    import json

    model = _model()
    hm.save_run(tmp_path, predictors=model.predictors, solver=model.solver)

    def reject_nonfinite(token):
        raise ValueError(f"Non-standard JSON token: {token}")

    json.loads((tmp_path / "metadata.json").read_text(), parse_constant=reject_nonfinite)
    loaded = hm.load_run(tmp_path, predictors_template=model.predictors)
    np.testing.assert_allclose(loaded["predictors"]["rate"].value, 0.3, rtol=1e-6)


@pytest.mark.parametrize("bounds", [((0.0, None),), ((None, 1.0),), ((None, None),)])
def test_open_scalers_accept_scalar_queries(bounds):
    scaler = hm.BoundScaler(bounds)
    value = jnp.array(0.2)
    latent = scaler.to_latent(value)
    np.testing.assert_allclose(scaler.from_latent(latent), [0.2], atol=1e-6)
    assert eqx.filter_jit(scaler.from_latent)(jnp.array(0.0)).shape == (1,)


def test_stop_reason_follows_the_final_phase():
    # Phase 1 cannot improve (lr=0), so patience cuts it short. Phase 2
    # improves every step after its first and runs to its budget, so the
    # run as a whole ended on budget, not patience.
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(10, lr=0.0), hm.TrainingPhase(3, lr=0.05)),
        patience=2,
        verbose=False,
    )
    result = _model().fit(_dataset(), config, key=jr.key(5))
    assert result.history.phase.count(0) < 10
    assert result.history.phase.count(1) == 3
    assert result.stop_reason == "budget_exhausted"
