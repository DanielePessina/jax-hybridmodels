"""Full workflows on one small system, the first-order reaction A -> B.

Each session fixture runs one workflow end to end, and the short tests below
it read properties off the result, so a fit runs once however many things are
checked. The system has a closed-form solution, ``A = exp(-K(t))`` and
``B = 1 - A``, which gives every check an oracle outside the framework.

The data is noise-free and deliberately awkward: channels sampled at different
times, an initial time before the first measurement, and two distinct grid
lengths, so masks and bucket dispatch are always exercised.
"""

from __future__ import annotations

from typing import Any, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd
import pytest
from jax import Array

import jaxhybridmodels as hm
import jaxhybridmodels.training.evosax as evosax_module
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.penalties import box_grid
from jaxhybridmodels.training.kernels import simulate_bucket
from jaxhybridmodels.ui.testing import RecordingUI

CHANNELS = ("a", "b")
SOLVER = hm.SolverConfig(rtol=1e-5, atol=1e-7, dt0=0.05, max_steps=256)
TRUE_RATE = 0.8
# The rate drops by half at t = 1 in the mechanistic workflow.
FEED = hm.step_profile(before=1.0, after=0.5, jump_at=1.0)

# (exp_id, temp, times for channel a, times for channel b). Grid lengths after
# the union with t0 are 4 and 5, so every dataset here has two buckets.
GRIDS: tuple[tuple[str, float, tuple[float, ...], tuple[float, ...]], ...] = (
    ("r0", 0.2, (0.5, 1.0, 2.0), (1.0,)),
    ("r1", 0.6, (0.25, 1.5, 2.5), (2.5,)),
    ("r2", 1.0, (0.5, 1.5, 2.0), (0.75,)),
    ("r3", 1.4, (0.3, 0.9, 1.8), (1.2,)),
    ("r4", 1.8, (0.4, 1.1, 2.2), (2.2,)),
    ("r5", 0.4, (0.6, 1.3, 2.1), (2.4,)),
    ("r6", 0.8, (0.2, 1.0, 1.9), (1.9,)),
    ("r7", 1.2, (0.7, 1.6, 2.3), (0.35,)),
    ("r8", 1.6, (0.45, 1.25, 2.05), (2.05,)),
    ("r9", 0.3, (0.55, 1.45, 2.35), (1.0,)),
)
# Measured cells per channel across all of GRIDS.
N_OBSERVED = {"a": 3 * len(GRIDS), "b": len(GRIDS)}


def rate_law(temp: Any) -> Any:
    """The covariate-dependent law the network workflow has to learn."""
    return 0.4 + 0.4 * temp


def feed_integral(ts: Array) -> Array:
    """``∫ FEED dt`` from 0: the time axis the mechanistic reaction runs on."""
    return jnp.where(ts < 1.0, ts, 1.0 + 0.5 * (ts - 1.0))


def experiments(extent: Any = None, rate: Any = None) -> list[hm.Experiment]:
    """Exact observations. ``extent(ts)`` is the integrated forcing, ``rate(temp)`` the law."""
    built = []
    for exp_id, temp, a_times, b_times in GRIDS:
        k = TRUE_RATE if rate is None else rate(temp)
        a_ts, b_ts = jnp.asarray(a_times), jnp.asarray(b_times)
        warp = (lambda t: t) if extent is None else extent
        built.append(
            hm.make_experiment(
                channels={
                    "a": hm.ChannelObs(ts=a_ts, values=jnp.exp(-k * warp(a_ts)), variance=1e-2),
                    "b": hm.ChannelObs(ts=b_ts, values=1 - jnp.exp(-k * warp(b_ts))),
                },
                covariates={"temp": temp},
                y0=[1.0, 0.0],
                t0=0.0,
                exp_id=exp_id,
            )
        )
    return built


def reaction(k: Array, y: Array) -> Array:
    return jnp.stack((-k * y[0], k * y[0]))


def mechanistic_field(t: Any, y: Array, predictors: Any, covariates: Any) -> Array:
    del covariates
    return reaction(predictors["rate"].value * predictors["gain"].value * FEED(t), y)


def network_field(t: Any, y: Array, predictors: Any, covariates: Any) -> Array:
    del t
    return reaction(predictors["rate"](covariates) * predictors["gain"].value, y)


def plain_field(t: Any, y: Array, predictors: Any, covariates: Any) -> Array:
    del t, covariates
    return reaction(predictors["rate"].value, y)


def model_of(field: Any, predictors: Any) -> hm.HybridModel:
    return hm.HybridModel.from_vector_field(
        vector_field=field,
        predictors=predictors,
        state_to_output=lambda state: state,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )


def mechanistic_model(rate: float = 0.3) -> hm.HybridModel:
    return model_of(
        mechanistic_field,
        {"rate": hm.Parameter(rate, bounds=(0.0, None)), "gain": hm.Constant(1.0)},
    )


def plain_model(rate: float = 0.3) -> hm.HybridModel:
    return model_of(
        plain_field,
        {"rate": hm.Parameter(rate, bounds=(0.05, 3.0)), "frozen": hm.Parameter(0.7)},
    )


def rate_network(key: Array, inner: Any = None) -> hm.BoundedPredictor:
    if inner is None:
        inner = hm.MLPPredictor(
            in_size=1, out_size=1, width_size=8, depth=1, activation_name="tanh", key=key
        )
    return hm.BoundedPredictor(
        inner=inner,
        input_keys=("temp",),
        input_bounds=((0.0, 2.0),),
        output_bounds=((0.05, 3.0),),
    )


def network_model(key: Array, inner: Any = None) -> hm.HybridModel:
    return model_of(
        network_field,
        {"rate": rate_network(key, inner), "gain": hm.Parameter(1.0, bounds=(0.5, 2.0))},
    )


def events_named(ui: RecordingUI, name: str) -> list[dict[str, Any]]:
    return [kwargs for event, kwargs in ui.events if event == name]


# ---------------------------------------------------------------------------
# Mechanistic: physical parameters, a schedule, a forced reaction
# ---------------------------------------------------------------------------

MECHANISTIC_CONFIG = hm.OptaxTrainingConfig(
    phases=(
        hm.TrainingPhase(40, lr=optax.linear_schedule(0.1, 0.02, 40), horizon=0.5),
        hm.TrainingPhase(60, lr=0.05, reset_optimizer=True),
    ),
    verbose=False,
)


@pytest.fixture(scope="session")
def mechanistic_data() -> hm.Dataset:
    return hm.make_dataset(experiments(extent=feed_integral), output_channel_names=CHANNELS)


@pytest.fixture(scope="session")
def mechanistic(mechanistic_data):
    ui = RecordingUI()
    objective = hm.Objective(channel_weights={"a": 1.0, "b": 0.5})
    model = mechanistic_model()
    result = model.fit(
        mechanistic_data, MECHANISTIC_CONFIG, objective=objective, key=jr.key(0), ui=ui
    )
    return model, result, ui


def fitted_rate(result: hm.FitResult) -> float:
    return float(result.model.predictors["rate"].value)


def test_mechanistic_fit_recovers_the_rate_and_leaves_the_input_alone(mechanistic):
    model, result, _ = mechanistic
    assert fitted_rate(result) == pytest.approx(TRUE_RATE, abs=0.02)
    assert float(model.predictors["rate"].value) == pytest.approx(0.3)
    assert float(result.model.predictors["gain"].value) == 1.0


def test_mechanistic_history_covers_every_step_of_every_phase(mechanistic):
    _, result, _ = mechanistic
    assert result.history.phase == (0,) * 40 + (1,) * 60
    assert len(result.history.loss) == len(result.history.penalty) == 100
    assert result.history.loss[-1] < result.history.loss[0]
    assert result.stop_reason == "budget_exhausted"
    # restore_best: the returned parameters carry the minimum of the last horizon.
    assert result.best_step is not None and result.best_step >= 40
    assert result.final_loss == min(result.history.loss[40:])


def test_mechanistic_ui_sees_the_lifecycle_in_order(mechanistic, mechanistic_data):
    _, _, ui = mechanistic
    names = [name for name, _ in ui.events]
    assert names[0] == "on_run_start" and names[-1] == "on_run_end"
    assert names.index("on_compile_done") < names.index("on_phase_start")
    assert names.count("on_step_end") == 100
    assert names.count("on_compile_start") == len(mechanistic_data.bucket_payloads) == 2
    phases = events_named(ui, "on_phase_start")
    assert [p["phase_idx"] for p in phases] == [0, 1]
    assert [p["lr"] for p in phases] == pytest.approx([0.1, 0.05])


def test_predict_matches_the_closed_form_on_the_measured_grid(mechanistic):
    _, result, _ = mechanistic
    predictions = result.model.predict(experiments(extent=feed_integral))
    assert list(predictions) == [grid[0] for grid in GRIDS]
    k = fitted_rate(result)
    for experiment in predictions.values():
        expected = jnp.exp(-k * feed_integral(experiment.times))
        np.testing.assert_allclose(experiment.outputs["a"], expected, atol=2e-3)
        np.testing.assert_allclose(experiment.outputs["b"], 1 - expected, atol=2e-3)
        # The t0 row was never measured, so it is NaN and masked in both channels.
        assert experiment.times[0] == 0.0
        assert not experiment.observation_mask["a"][0] and not experiment.observation_mask["b"][0]
        assert np.isnan(experiment.observed["a"][0])


def test_predict_on_a_requested_grid_has_no_observations(mechanistic):
    _, result, _ = mechanistic
    data = experiments(extent=feed_integral)
    even = result.model.predict(data, n_points=7)
    shared = result.model.predict(data, ts=jnp.array([0.5, 1.5]))
    k = fitted_rate(result)
    for experiment in even.values():
        assert experiment.times.shape == (7,) and experiment.observed == {}
        assert experiment.times[0] == 0.0
    for experiment in shared.values():
        expected = jnp.exp(-k * feed_integral(jnp.array([0.5, 1.5])))
        np.testing.assert_allclose(experiment.outputs["a"], expected, atol=2e-3)


def test_simulate_runs_new_conditions(mechanistic):
    _, result, _ = mechanistic
    ts = jnp.linspace(0.5, 3.0, 6)
    fresh = result.model.simulate(ts=ts, t0=0.0, y0=[0.5, 0.5])
    expected = 0.5 * jnp.exp(-fitted_rate(result) * feed_integral(ts))
    np.testing.assert_allclose(fresh.outputs["a"], expected, atol=2e-3)
    assert fresh.state is not None and fresh.state.shape == (6, 2)


def test_simulate_rereads_an_experiment_as_predict_does():
    # The smooth system: a jump in the forcing makes a solve sensitive to which
    # steps get accepted, which would swamp the agreement checked here.
    model = plain_model(TRUE_RATE)
    source = experiments()[0]
    again = model.simulate(source)
    on_data = model.predict([source])[source.exp_id]
    np.testing.assert_array_equal(again.times, on_data.times)
    np.testing.assert_allclose(again.outputs["a"], on_data.outputs["a"], atol=1e-5)


def test_metrics_count_only_measured_cells(mechanistic, mechanistic_data, capsys):
    _, result, _ = mechanistic
    predictions = result.model.predict(mechanistic_data)
    metrics = hm.compute_metrics(predictions)
    assert {name: int(m.n) for name, m in metrics.items()} == N_OBSERVED
    assert all(float(m.r2) > 0.99 for m in metrics.values())
    hm.print_metrics(predictions)
    assert "a" in capsys.readouterr().out


def test_a_fit_result_prints_as_one_line(mechanistic):
    _, result, _ = mechanistic
    printed = repr(result)
    assert "\n" not in printed and len(printed) < 120
    assert "stop_reason='budget_exhausted'" in printed and "steps=100" in printed


def test_dataframe_has_one_row_per_cell(mechanistic, mechanistic_data):
    _, result, _ = mechanistic
    frame = result.model.predict(mechanistic_data).to_dataframe()
    assert frame["is_observed"].sum() == sum(N_OBSERVED.values())
    assert frame.groupby("channel")["is_observed"].sum().to_dict() == N_OBSERVED
    assert set(frame["experiment_id"]) == {grid[0] for grid in GRIDS}
    assert frame.loc[~frame["is_observed"], "observed"].isna().all()


def test_every_data_builder_gives_the_same_experiment():
    ts = jnp.array([0.5, 1.0, 2.0])
    values = jnp.stack((jnp.exp(-ts), 1 - jnp.exp(-ts)), axis=1)
    by_hand = hm.make_experiment(
        channels={
            name: hm.ChannelObs(ts=ts, values=values[:, i]) for i, name in enumerate(CHANNELS)
        },
        covariates={"temp": 1.0},
        y0=[1.0, 0.0],
        t0=0.0,
        exp_id="run",
    )
    from_array = hm.experiment_from_array(
        ts=ts,
        values=values,
        channel_names=CHANNELS,
        covariates={"temp": 1.0},
        y0=[1.0, 0.0],
        t0=0.0,
        exp_id="run",
    )
    table = pd.DataFrame(
        {
            "run": "run",
            "t": np.asarray(ts),
            "a": np.asarray(values[:, 0]),
            "b": np.asarray(values[:, 1]),
            "temp": 1.0,
        }
    )
    (from_table,) = hm.experiments_from_dataframe(
        table,
        time="t",
        experiment="run",
        channels=CHANNELS,
        covariates=("temp",),
        t0=0.0,
        y0=[1.0, 0.0],
    )
    payloads = [
        hm.make_dataset([e], output_channel_names=CHANNELS).bucket_payloads[0]
        for e in (by_hand, from_array, from_table)
    ]
    for payload in payloads[1:]:
        np.testing.assert_allclose(payload.ts, payloads[0].ts)
        np.testing.assert_allclose(payload.y_observed, payloads[0].y_observed)
        np.testing.assert_array_equal(payload.mask, payloads[0].mask)


def test_trajectory_penalty_is_charged_on_top_of_the_data_loss(mechanistic_data):
    """A one-step fit's first loss is the data term plus the bucket-averaged charge."""

    def charge(state: Array, bp: Any) -> Array:
        del bp
        return jnp.mean(state[..., 1])

    one_step = hm.OptaxTrainingConfig(steps=1, lr=1e-3, verbose=False)
    model = mechanistic_model()
    plain = model.fit(mechanistic_data, one_step, key=jr.key(0))
    charged = model.fit(
        mechanistic_data,
        one_step,
        objective=hm.Objective(trajectory_penalty_fn=charge, trajectory_penalty_weight=0.25),
        key=jr.key(0),
    )
    by_bucket = [
        float(
            charge(
                simulate_bucket(
                    model.predictors, bp, simulate_fn=model.simulate_fn, solver=model.solver
                ),
                bp,
            )
        )
        for bp in mechanistic_data.bucket_payloads
    ]
    expected = plain.history.loss[0] + 0.25 * float(np.mean(by_bucket))
    assert charged.history.loss[0] == pytest.approx(expected, rel=1e-4)
    assert charged.history.penalty == (0.0,), "the bound penalty is reported separately"


@pytest.fixture(scope="session")
def overshoot(mechanistic_data):
    """A learning rate far too large from a good start, so the best step is an early one."""
    config = hm.OptaxTrainingConfig(steps=60, lr=0.6, patience=5, verbose=False)
    model = mechanistic_model(rate=TRUE_RATE + 0.01)
    keep_best = model.fit(mechanistic_data, config, key=jr.key(0))
    keep_last = model.fit(
        mechanistic_data,
        hm.OptaxTrainingConfig(steps=5, lr=0.6, restore_best=False, verbose=False),
        key=jr.key(0),
    )
    return keep_best, keep_last


def test_patience_stops_early_and_the_best_step_labels_the_model(overshoot, mechanistic_data):
    keep_best, _ = overshoot
    assert keep_best.stop_reason == "patience"
    assert len(keep_best.history.loss) < 60
    assert keep_best.best_step == int(np.argmin(keep_best.history.loss))
    # One more step at lr=0 measures the returned model; it must match its label.
    remeasured = keep_best.model.fit(
        mechanistic_data, hm.OptaxTrainingConfig(steps=1, lr=0.0, verbose=False), key=jr.key(0)
    )
    assert remeasured.history.loss[0] == pytest.approx(keep_best.final_loss, rel=1e-5)


def test_restore_best_off_returns_the_last_step(overshoot):
    _, keep_last = overshoot
    assert keep_last.best_step is None
    assert keep_last.final_loss == keep_last.history.loss[-1]


# ---------------------------------------------------------------------------
# Network: a learned law, a split, validation, a tournament, a bound penalty
# ---------------------------------------------------------------------------

NETWORK_STEPS = 150


@pytest.fixture(scope="session")
def network_splits() -> tuple[hm.Dataset, hm.Dataset, hm.Dataset]:
    data = hm.make_dataset(experiments(rate=rate_law), output_channel_names=CHANNELS)
    return hm.split_dataset(data, train=0.6, val=0.2, test=0.2, key=jr.key(1))


@pytest.fixture(scope="session")
def network_objective() -> hm.Objective:
    points = box_grid(rate_network(jr.key(0)).in_scaler, 5)
    return hm.Objective(penalty_weight=1e-3, penalty_points={"rate": points})


@pytest.fixture(scope="session")
def network_config() -> hm.OptaxTrainingConfig:
    return hm.OptaxTrainingConfig(
        steps=NETWORK_STEPS,
        lr=0.02,
        tournament_attempts=3,
        tournament_steps=5,
        tournament_lr=1e-3,
        verbose=False,
    )


@pytest.fixture(scope="session")
def network(network_splits, network_objective, network_config):
    train, val, _ = network_splits
    ui = RecordingUI()
    model = network_model(jr.key(0))
    result = model.fit(
        train,
        network_config,
        objective=network_objective,
        freeze="gain",
        validation=val,
        key=jr.key(2),
        ui=ui,
    )
    return model, result, ui


def test_split_partitions_the_experiments(network_splits):
    ids = [{e.exp_id for e in split.experiments} for split in network_splits]
    assert [len(split) for split in ids] == [6, 2, 2]
    assert set.union(*ids) == {grid[0] for grid in GRIDS}
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])


def test_network_learns_the_law_inside_its_bounds(network, network_splits):
    _, result, _ = network
    # Only at the temperatures it trained on: it has no reason to extrapolate well.
    temps = jnp.array([float(e.covariates["temp"]) for e in network_splits[0].experiments])
    learned = hm.evaluate_predictor(result.model.predictors["rate"], {"temp": temps})
    assert learned.shape == temps.shape
    np.testing.assert_allclose(learned, rate_law(temps), atol=0.1)
    sweep = hm.evaluate_predictor(
        result.model.predictors["rate"], {"temp": jnp.linspace(0.0, 2.0, 21)}
    )
    assert float(sweep.min()) > 0.05 and float(sweep.max()) < 3.0


def test_freeze_holds_the_named_predictor(network):
    model, result, _ = network
    assert float(result.model.predictors["gain"].value) == float(model.predictors["gain"].value)


def test_network_generalises_to_held_out_experiments(network, network_splits):
    _, result, _ = network
    metrics = hm.compute_metrics(result.model.predict(network_splits[2]))
    assert float(metrics["a"].r2) > 0.9


def test_validation_selects_the_returned_step(network):
    _, result, _ = network
    history = result.history
    assert len(history.validation_loss) == NETWORK_STEPS
    assert result.best_step == int(np.argmin(history.validation_loss))


def test_an_unsaturated_network_is_not_charged(network):
    # The law sits well inside the bounds, so the bound penalty has nothing to do.
    _, result, _ = network
    assert result.history.penalty == (0.0,) * NETWORK_STEPS


@pytest.fixture(scope="session")
def saturated_fit(network_splits, network_objective):
    """A readout scaled until the output squash is pinned against its bound.

    The first phase overrides the objective's weight with 0, the second uses it.
    """
    start = network_model(jr.key(0))
    pinned = eqx.tree_at(
        lambda m: m.predictors["rate"].inner.mlp.layers[-1].weight,
        start,
        start.predictors["rate"].inner.mlp.layers[-1].weight * 50.0,
    )
    ui = RecordingUI()
    objective = hm.Objective(penalty_weight=100.0, penalty_points=network_objective.penalty_points)
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(10, lr=0.05, penalty_weight=0.0), hm.TrainingPhase(20, lr=0.05)),
        verbose=False,
    )
    result = pinned.fit(network_splits[0], config, objective=objective, key=jr.key(2), ui=ui)
    return result, ui


def test_the_bound_penalty_follows_each_phases_weight(saturated_fit):
    result, ui = saturated_fit
    charged = result.history.penalty
    assert charged[:10] == (0.0,) * 10, "weight 0 in the first phase charges nothing"
    assert all(p > 0.0 for p in charged[10:15])
    # The history holds the weighted penalty, the UI event the unweighted one.
    raw = [event["penalty"] for event in events_named(ui, "on_step_end")]
    assert charged[10:] == pytest.approx([100.0 * p for p in raw[10:]])


def test_the_bound_penalty_pulls_the_output_off_its_bound(saturated_fit):
    _, ui = saturated_fit
    raw = [event["penalty"] for event in events_named(ui, "on_step_end")]
    assert raw[-1] < 0.5 * raw[10]


@pytest.fixture(scope="session")
def tournaments(plain_data):
    """The same key with more attempts: attempt keys are named, so it adds candidates."""

    def run(attempts: int) -> hm.FitResult:
        config = hm.OptaxTrainingConfig(
            steps=1,
            lr=1e-6,
            tournament_attempts=attempts,
            tournament_steps=3,
            tournament_lr=0.01,
            verbose=False,
        )
        return plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(9))

    return run(2), run(8)


def test_more_tournament_attempts_never_give_a_worse_start(tournaments, plain_data):
    few, many = tournaments
    assert many.history.loss[0] <= few.history.loss[0]
    # The winner replaced the starting value, and the tournament kept the frozen leaf.
    unchanged = plain_model().fit(
        plain_data, hm.OptaxTrainingConfig(steps=1, lr=0.0, verbose=False), key=jr.key(0)
    )
    assert many.history.loss[0] != unchanged.history.loss[0]
    assert float(many.model.predictors["frozen"].value) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Evosax
# ---------------------------------------------------------------------------

EVOSAX_RUNS: dict[str, dict[str, Any]] = {
    "cma-warm": {"algorithm": "CMA_ES", "init": "warm", "sigma_init": 0.5},
    "sep-uniform": {"algorithm": "Sep_CMA_ES", "init": "uniform_box", "sigma_init": 0.5},
    "simple-lhs": {"algorithm": "SimpleES", "init": "lhs_box", "sigma_init": 0.5},
}


@pytest.fixture(scope="session")
def plain_data() -> hm.Dataset:
    return hm.make_dataset(experiments(), output_channel_names=CHANNELS)


@pytest.fixture(scope="session")
def evolved(plain_data):
    runs = {}
    for name, settings in EVOSAX_RUNS.items():
        ui = RecordingUI()
        config = hm.EvosaxTrainingConfig(population_size=16, steps=25, verbose=False, **settings)
        result = plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(3), ui=ui)
        runs[name] = (result, ui)
    return runs


@pytest.mark.parametrize("run", EVOSAX_RUNS)
def test_evosax_improves_on_the_start_and_records_the_best_so_far(evolved, plain_data, run):
    result, ui = evolved[run]
    assert fitted_rate(result) == pytest.approx(TRUE_RATE, abs=0.15)
    assert len(result.history.loss) == 25
    assert list(result.history.loss) == sorted(result.history.loss, reverse=True)
    assert result.stop_reason == "budget_exhausted"
    assert len(events_named(ui, "on_generation_end")) == 25
    assert events_named(ui, "on_run_end")[0]["final_loss"] == pytest.approx(result.final_loss)
    start = plain_model().fit(
        plain_data, hm.OptaxTrainingConfig(steps=1, lr=0.0, verbose=False), key=jr.key(0)
    )
    assert result.final_loss < start.history.loss[0]


@pytest.mark.parametrize("run", EVOSAX_RUNS)
def test_evosax_keeps_frozen_leaves(evolved, run):
    result, _ = evolved[run]
    assert float(result.model.predictors["frozen"].value) == pytest.approx(0.7)


def test_evosax_validation_selects_a_generation(plain_data):
    result = plain_model().fit(
        plain_data,
        hm.EvosaxTrainingConfig(population_size=8, steps=8, sigma_init=0.5, verbose=False),
        validation=plain_data,
        freeze="frozen",
        key=jr.key(4),
    )
    assert len(result.history.validation_loss) == 8
    assert result.best_step == int(np.argmin(result.history.validation_loss))


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ensembles(plain_data):
    config = hm.OptaxTrainingConfig(
        steps=40, lr=0.05, tournament_steps=5, tournament_lr=0.05, verbose=False
    )
    model = plain_model()
    ui = RecordingUI()
    seeds = model.fit_ensemble(
        plain_data, config, n_members=2, n_seeds=3, freeze="frozen", key=jr.key(5), ui=ui
    )
    bootstrap = model.fit_ensemble(
        plain_data, config, n_members=2, bootstrap=True, freeze="frozen", key=jr.key(6)
    )
    return seeds, bootstrap, ui


def test_seed_ensemble_returns_trained_members_best_first(ensembles):
    seeds, _, ui = ensembles
    assert len(seeds) == 2
    assert seeds[0].final_loss <= seeds[1].final_loss
    assert all(fitted_rate(member) == pytest.approx(TRUE_RATE, abs=0.1) for member in seeds)
    assert len(events_named(ui, "on_run_end")) == 2, "each member is its own UI run"


def test_bootstrap_ensemble_trains_one_member_per_resample(ensembles):
    _, bootstrap, _ = ensembles
    assert len(bootstrap) == 2
    assert fitted_rate(bootstrap[0]) == pytest.approx(TRUE_RATE, abs=0.1)
    assert fitted_rate(bootstrap[0]) != fitted_rate(bootstrap[1])


@pytest.mark.parametrize("which", [0, 1], ids=["seeds", "bootstrap"])
def test_predict_ensemble_is_the_member_mean_with_its_spread(ensembles, plain_data, which):
    members = ensembles[which]
    combined = hm.predict_ensemble(members, plain_data)
    assert len(combined.members) == 2
    for exp_id, experiment in combined.items():
        stacked = jnp.stack([m.model.predict(plain_data)[exp_id].outputs["a"] for m in members])
        np.testing.assert_allclose(experiment.outputs["a"], stacked.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(experiment.std["a"], stacked.std(axis=0), atol=1e-5)
    assert "std" in combined.to_dataframe().columns
    assert {name: int(m.n) for name, m in hm.compute_metrics(combined).items()} == N_OBSERVED


@pytest.fixture(scope="session")
def validated_ensemble(plain_data):
    config = hm.OptaxTrainingConfig(steps=20, lr=0.05, verbose=False)
    # Scored on a set with a different bucket layout than the training data.
    validation = hm.make_dataset(experiments()[:3], output_channel_names=CHANNELS)
    return plain_model().fit_ensemble(
        plain_data, config, n_members=3, freeze="frozen", validation=validation, key=jr.key(7)
    )


def test_ensemble_validation_selects_each_members_step(validated_ensemble):
    for member in validated_ensemble:
        assert len(member.history.validation_loss) == 20
        assert member.best_step == int(np.argmin(member.history.validation_loss))


def test_ensemble_validation_ranks_the_members(validated_ensemble):
    scores = [m.history.validation_loss[m.best_step] for m in validated_ensemble]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Failure: a run that starts healthy and breaks partway
# ---------------------------------------------------------------------------

# The rate climbs from 0.3 towards 2.0 and crosses this limit on the way.
FAILURE_LIMIT = 1.0


def poison_past_limit(k: Array) -> Array:
    """1 below the limit, NaN past it, with no gradient.

    A plain ``jnp.where(ok, x, nan)`` puts NaN into the gradient on every step,
    healthy or not, and the run would break on its first update.
    """
    return jax.lax.stop_gradient(jnp.where(k < FAILURE_LIMIT, 1.0, jnp.nan))


def failing_field(t: Any, y: Array, predictors: Any, covariates: Any) -> Array:
    """Goes NaN past the limit, so the adaptive solver runs out of steps."""
    del t, covariates
    k = predictors["rate"].value
    return reaction(k, y) * poison_past_limit(k)


def failing_output(
    predictors: Any, ts: Array, covariates: Any, y0: Array, solver: hm.SolverConfig
) -> Array:
    """Solves without complaint and returns NaN past the limit: a loss with no solver error."""
    k = predictors["rate"].value
    del covariates
    term = diffrax.ODETerm(lambda t, y, args: reaction(k, y))
    states = solver.diffeqsolve(term, ts, y0).ys
    assert states is not None
    return states * poison_past_limit(k)


def failing_model(mode: str) -> hm.HybridModel:
    predictors = {"rate": hm.Parameter(0.3, bounds=(0.05, 3.0))}
    solver = hm.SolverConfig(rtol=1e-5, atol=1e-7, dt0=0.05, max_steps=64)
    if mode == "solver-error":
        return hm.HybridModel.from_vector_field(
            vector_field=failing_field,
            predictors=predictors,
            state_to_output=lambda state: state,
            output_channel_names=CHANNELS,
            solver=solver,
        )
    return hm.HybridModel(
        predictors=predictors,
        simulate_fn=failing_output,
        state_to_output=lambda state: state,
        output_channel_names=CHANNELS,
        solver=solver,
    )


@pytest.fixture(scope="session")
def fast_reaction_data() -> hm.Dataset:
    return hm.make_dataset(experiments(rate=lambda temp: 2.0), output_channel_names=CHANNELS)


@pytest.fixture(scope="session", params=["solver-error", "nan-loss"])
def failed(request, fast_reaction_data):
    ui = RecordingUI()
    config = hm.OptaxTrainingConfig(steps=300, lr=0.1, verbose=False)
    with pytest.warns(RuntimeWarning, match="diverged") as caught:
        result = failing_model(request.param).fit(fast_reaction_data, config, key=jr.key(0), ui=ui)
    return result, ui, str(caught[0].message), request.param


def test_a_failed_fit_returns_its_best_parameters(failed):
    result, _, _, _ = failed
    assert result.stop_reason == "diverged"
    assert 1 < len(result.history.loss) < 300
    assert all(np.isfinite(result.history.loss))
    assert result.best_step == int(np.argmin(result.history.loss))
    assert fitted_rate(result) < FAILURE_LIMIT


def test_a_failed_fit_can_still_predict(failed, fast_reaction_data):
    result, _, _, _ = failed
    outputs = result.model.predict(fast_reaction_data)
    assert all(np.isfinite(e.outputs["a"]).all() for e in outputs.values())


def test_a_failed_fit_says_why_and_closes_the_ui(failed):
    _, ui, message, mode = failed
    assert ("maximum number of solver steps" if mode == "solver-error" else "non-finite") in message
    assert events_named(ui, "on_message")[0]["level"] == "error"
    assert [name for name, _ in ui.events][-1] == "on_run_end"
    assert len(events_named(ui, "on_run_end")) == 1


def test_a_failed_fit_saves_and_reloads(failed, tmp_path):
    result, _, _, mode = failed
    hm.save(tmp_path, result)
    assert hm.load(tmp_path, like=failing_model(mode)).stop_reason == "diverged"


def test_restore_best_off_returns_the_last_measured_step(fast_reaction_data):
    config = hm.OptaxTrainingConfig(steps=300, lr=0.1, restore_best=False, verbose=False)
    with pytest.warns(RuntimeWarning, match="diverged"):
        result = failing_model("nan-loss").fit(fast_reaction_data, config, key=jr.key(0))
    assert result.best_step is None
    assert fitted_rate(result) < FAILURE_LIMIT
    remeasured = result.model.fit(
        fast_reaction_data, hm.OptaxTrainingConfig(steps=1, lr=0.0, verbose=False), key=jr.key(0)
    )
    assert remeasured.history.loss[0] == pytest.approx(result.history.loss[-1], rel=1e-4)


def test_a_failure_right_after_a_horizon_change_returns_a_measured_step(fast_reaction_data):
    # The running minimum resets at the boundary, leaving no best to restore. The
    # one update of the first phase jumps past the limit, so the second phase
    # fails on its first step and the only measured parameters are the start.
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(1, lr=3.0, horizon=0.5), hm.TrainingPhase(5, lr=0.1)),
        verbose=False,
    )
    with pytest.warns(RuntimeWarning, match="diverged"):
        result = failing_model("nan-loss").fit(fast_reaction_data, config, key=jr.key(0))
    assert result.stop_reason == "diverged"
    assert result.best_step is None
    assert len(result.history.loss) == 1
    assert result.final_loss == result.history.loss[0]
    assert fitted_rate(result) == pytest.approx(0.3, rel=1e-4)


@pytest.mark.parametrize("mode", ["solver-error", "nan-loss"])
def test_a_failure_on_the_first_step_raises(fast_reaction_data, mode):
    # Past the limit from the start, so no step can be measured.
    start = hm.Parameter(1.5, bounds=(0.05, 3.0))
    model = eqx.tree_at(lambda m: m.predictors["rate"], failing_model(mode), start)
    with pytest.raises((FloatingPointError, eqx.EquinoxRuntimeError)):
        model.fit(
            fast_reaction_data,
            hm.OptaxTrainingConfig(steps=5, lr=0.1, verbose=False),
            key=jr.key(0),
        )


class PoisonedStrategy:
    """An evosax strategy that asks for NaN candidates from a given generation on."""

    def __init__(self, strategy: Any, poison_from: int) -> None:
        self.strategy, self.poison_from, self.generation = strategy, poison_from, 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.strategy, name)

    def ask(self, key: Array, state: Any, params: Any) -> Any:
        population, state = self.strategy.ask(key, state, params)
        self.generation += 1
        return population * (jnp.nan if self.generation > self.poison_from else 1.0), state


@pytest.mark.parametrize("poison_from, reaches_history", [(3, True), (0, False)])
def test_evosax_returns_its_best_when_a_later_generation_fails(
    plain_data, monkeypatch, poison_from, reaches_history
):
    build = evosax_module._build_strategy

    def poisoned(**kwargs: Any) -> Any:
        strategy, params = build(**kwargs)
        return PoisonedStrategy(strategy, poison_from), params

    monkeypatch.setattr(evosax_module, "_build_strategy", poisoned)
    config = hm.EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.1, verbose=False)
    if not reaches_history:
        with pytest.raises((FloatingPointError, eqx.EquinoxRuntimeError)):
            plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(0))
        return
    with pytest.warns(RuntimeWarning, match="diverged"):
        result = plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(0))
    assert result.stop_reason == "diverged"
    assert len(result.history.loss) == poison_from
    assert np.isfinite(float(result.model.predictors["rate"].value))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_full_run_reloads_and_predicts_bit_exactly(
    network, network_splits, network_config, network_objective, tmp_path
):
    model, result, _ = network
    # A first save to the same directory is replaced, and missing parents are made.
    target = tmp_path / "not" / "yet" / "there"
    hm.save(target, model, extras={"attempt": 1})
    hm.save(
        target, result, config=network_config, objective=network_objective, extras={"attempt": 2}
    )
    saved = hm.load(target, like=model)

    assert isinstance(saved.config, hm.OptaxTrainingConfig) and saved.config == network_config
    assert isinstance(saved.objective, hm.Objective)
    assert saved.objective.penalty_weight == network_objective.penalty_weight
    assert saved.objective.penalty_points is not None
    assert network_objective.penalty_points is not None
    np.testing.assert_array_equal(
        saved.objective.penalty_points["rate"], network_objective.penalty_points["rate"]
    )
    assert saved.history == result.history
    assert (saved.best_step, saved.stop_reason) == (result.best_step, result.stop_reason)
    assert saved.extras == {"attempt": 2}

    test_set = network_splits[2]
    before, after = result.model.predict(test_set), saved.model.predict(test_set)
    for exp_id in before:
        np.testing.assert_array_equal(before[exp_id].outputs["a"], after[exp_id].outputs["a"])


def test_a_bare_model_saves_without_a_history(tmp_path):
    hm.save(tmp_path, mechanistic_model())
    saved = hm.load(tmp_path, like=mechanistic_model())
    assert saved.history == hm.FitHistory() and saved.stop_reason is None and saved.config is None


def test_callables_come_back_as_their_names(tmp_path):
    """A schedule and a custom loss cannot be rebuilt, so they reload as raw dicts."""
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(1, lr=optax.linear_schedule(0.1, 0.0, 1)),), verbose=False
    )
    objective = hm.Objective(loss=masked_mse)
    hm.save(tmp_path, mechanistic_model(), config=config, objective=objective)
    saved = hm.load(tmp_path, like=mechanistic_model())
    assert isinstance(saved.config, dict) and isinstance(saved.objective, dict)
    raw_config, raw_objective = (
        cast(dict[str, Any], saved.config),
        cast(dict[str, Any], saved.objective),
    )
    assert "__callable__" in raw_config["phases"][0]["lr"]
    assert raw_objective["loss"] == {"__callable__": "jaxhybridmodels.losses.masked_mse"}


# ---------------------------------------------------------------------------
# Custom parts: other predictors, a custom loss, an optimizer factory
# ---------------------------------------------------------------------------


class LinearReadout(hm.Predictor):
    """The smallest custom predictor: an affine map of the scaled input."""

    weight: Array
    bias: Array
    in_size: int = eqx.field(static=True, default=1)
    out_size: int = eqx.field(static=True, default=1)

    def __call__(self, x: Array) -> Array:
        return self.weight * x + self.bias


def kan_inner(key: Array) -> hm.Predictor:
    return hm.KANPredictor(in_size=1, out_size=1, hidden_widths=(4,), grid_size=3, key=key)


def polynomial_inner(key: Array) -> hm.Predictor:
    net = hm.MLPPredictor(in_size=1, out_size=2, width_size=4, depth=1, key=key)
    return hm.NeuralNPolynomial(coeff_net=net, exponents=(0.0, 1.0), in_size=1, out_size=1)


def custom_inner(key: Array) -> hm.Predictor:
    del key
    return LinearReadout(weight=jnp.array([0.1]), bias=jnp.array([0.0]))


def weighted_loss(predicted: Array, bp: Any) -> Array:
    """A plain custom loss, the signature ``Objective(loss=...)`` documents."""
    return jnp.sum(((predicted - bp.y_observed) * bp.mask) ** 2) / jnp.maximum(bp.mask.sum(), 1)


INNERS = {"kan": kan_inner, "polynomial": polynomial_inner, "custom": custom_inner}


@pytest.fixture(scope="session")
def custom_fits(network_splits):
    train, _, _ = network_splits
    config = hm.OptaxTrainingConfig(steps=30, lr=0.02, optimizer=optax.adam, verbose=False)
    fits = {}
    for name, inner in INNERS.items():
        model = network_model(jr.key(0), inner(jr.key(7)))
        fits[name] = model.fit(
            train, config, objective=hm.Objective(loss=weighted_loss), key=jr.key(8)
        )
    return fits


@pytest.mark.parametrize("name", INNERS)
def test_other_predictors_train_inside_a_bounded_predictor(custom_fits, name):
    result = custom_fits[name]
    assert result.final_loss < result.history.loss[0]
    assert hm.count_trainable_params(result.model.predictors) > 1


@pytest.mark.parametrize("name", INNERS)
def test_other_predictors_reload_bit_exactly(custom_fits, name, network_splits, tmp_path):
    result = custom_fits[name]
    hm.save(tmp_path, result)
    saved = hm.load(tmp_path, like=network_model(jr.key(0), INNERS[name](jr.key(7))))
    test_set = network_splits[2]
    before, after = result.model.predict(test_set), saved.model.predict(test_set)
    for exp_id in before:
        np.testing.assert_array_equal(before[exp_id].outputs["a"], after[exp_id].outputs["a"])
