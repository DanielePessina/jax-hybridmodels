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

import copy
import itertools
import json
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
from jaxhybridmodels._fit_summary import _loss_trend
from jaxhybridmodels.data import _choose_bucket_lengths
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.penalties import box_grid
from jaxhybridmodels.training.kernels import failure_details, simulate_bucket
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


def experiments(
    extent: Any = None, rate: Any = None, *, one_length: bool = False
) -> list[hm.Experiment]:
    """Exact observations. ``extent(ts)`` is the integrated forcing, ``rate(temp)`` the law.

    ``one_length`` keeps the experiments with a four-row grid, one bucket instead
    of two. A fit compiles per bucket, so workflows that do not exercise bucket
    dispatch use this to run in half the time.
    """
    built = []
    for exp_id, temp, a_times, b_times in GRIDS:
        if one_length and len({0.0, *a_times, *b_times}) != 4:
            continue
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


def test_metrics_by_experiment_add_up_to_the_pooled_ones(mechanistic, mechanistic_data, capsys):
    _, result, _ = mechanistic
    predictions = result.model.predict(mechanistic_data)
    pooled = hm.compute_metrics(predictions)
    by_experiment = hm.compute_metrics(predictions, by="experiment")
    assert list(by_experiment) == list(predictions)
    for channel, overall in pooled.items():
        parts = [per_channel[channel] for per_channel in by_experiment.values()]
        assert sum(int(m.n) for m in parts) == int(overall.n)
        # Each experiment's squared error, weighted by its cells, is the pool's.
        total = sum(float(m.mse) * int(m.n) for m in parts if int(m.n) > 0)
        assert total == pytest.approx(float(overall.mse) * int(overall.n), rel=1e-5)

    hm.print_metrics(by_experiment)
    printed = capsys.readouterr().out
    rows = [line.split() for line in printed.splitlines() if line.startswith("    ")]
    rmse = [float(row[3]) for row in rows if row[0] != "experiment"][: len(by_experiment)]
    assert rmse == sorted(rmse, reverse=True), "the worst experiment is listed first"


def test_summary_reports_the_run_and_its_parameters(mechanistic, mechanistic_data):
    _, result, _ = mechanistic
    text = result.summary()
    assert "ran its full budget" in text and "100 steps recorded" in text
    assert "rate" in text and f"{fitted_rate(result):.5g}" in text
    assert "near" not in text, "a rate in the middle of its box is not flagged"
    assert "Metrics" not in text, "no data, no prediction"

    with_data = result.summary(mechanistic_data)
    assert with_data.startswith(text)
    assert "Metrics" in with_data and "Worst experiment per channel" in with_data


def test_summary_flags_a_parameter_at_either_bound():
    predictors = {
        "low": hm.Parameter(0.001, bounds=(0.0, 1.0)),
        "open_low": hm.Parameter(0.999, bounds=(None, 1.0)),
        "high": hm.Parameter(0.999, bounds=(0.0, 1.0)),
        "open_side": hm.Parameter(1e-4, bounds=(0.0, None)),
        "log_high": hm.Parameter(9e2, bounds=(1e-3, 1e3), warp="log10"),
        "pair": hm.Parameter([0.5, 0.999], bounds=(0.0, 1.0)),
        "middle": hm.Parameter(0.5, bounds=(0.0, 1.0)),
    }
    model = eqx.tree_at(lambda m: m.predictors, mechanistic_model(), predictors)
    run = hm.FitResult(model=model, history=hm.FitHistory(), best_step=None, stop_reason="patience")
    rows = {line.split()[0]: line for line in run.summary().splitlines() if "(" in line}
    assert "near lower bound" in rows["low"]
    assert "near upper bound" in rows["open_low"], "a one-sided box can only near its one bound"
    assert "near upper bound" in rows["high"]
    assert "near lower bound" in rows["open_side"]
    assert "near upper bound" in rows["log_high"], "the side is judged in warped coordinates"
    assert "near" not in rows["pair[0]"] and "near upper bound" in rows["pair[1]"]
    assert "near" not in rows["middle"]
    assert "no fit recorded" in run.summary()


def test_summary_of_generations_has_no_phase_table():
    history = hm.FitHistory(loss=(2.0, 1.0, 1.0))
    run = hm.FitResult(
        model=mechanistic_model(), history=history, best_step=1, stop_reason="budget_exhausted"
    )
    text = run.summary()
    assert "3 generations recorded; returned generation 1" in text and "Phases" not in text


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

    one_step = hm.OptaxTrainingConfig(steps=1, lr=1e-3)
    model = mechanistic_model()
    plain = model.fit(mechanistic_data, one_step, key=jr.key(0), verbose=False)
    charged = model.fit(
        mechanistic_data,
        one_step,
        objective=hm.Objective(trajectory_penalty_fn=charge, trajectory_penalty_weight=0.25),
        key=jr.key(0),
        verbose=False,
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
    config = hm.OptaxTrainingConfig(steps=60, lr=0.6, patience=5)
    model = mechanistic_model(rate=TRUE_RATE + 0.01)
    data = hm.make_dataset(
        experiments(extent=feed_integral, one_length=True), output_channel_names=CHANNELS
    )
    keep_best = model.fit(data, config, key=jr.key(0), verbose=False)
    keep_last = model.fit(
        data,
        hm.OptaxTrainingConfig(steps=5, lr=0.6, restore_best=False),
        key=jr.key(0),
        verbose=False,
    )
    return keep_best, keep_last, data


def test_patience_stops_early_and_the_best_step_labels_the_model(overshoot):
    keep_best, _, data = overshoot
    assert keep_best.stop_reason == "patience"
    assert len(keep_best.history.loss) < 60
    assert keep_best.best_step == int(np.argmin(keep_best.history.loss))
    # One more step at lr=0 measures the returned model; it must match its label.
    remeasured = keep_best.model.fit(
        data, hm.OptaxTrainingConfig(steps=1, lr=0.0), key=jr.key(0), verbose=False
    )
    assert remeasured.history.loss[0] == pytest.approx(keep_best.final_loss, rel=1e-5)


def test_restore_best_off_returns_the_last_step(overshoot):
    _, keep_last, _ = overshoot
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
    )
    result = pinned.fit(network_splits[0], config, objective=objective, key=jr.key(2), ui=ui)
    return result, ui


def test_summary_shows_how_much_of_a_network_is_pinned(network, saturated_fit):
    def pinned_share(result: hm.FitResult) -> float:
        lines = result.summary().splitlines()
        row = next(line for line in lines if line.lstrip().startswith("rate "))
        return float(row.split("%")[0].split()[-1])

    assert pinned_share(saturated_fit[0]) > 10.0
    assert pinned_share(network[1] if isinstance(network, tuple) else network) == 0.0
    text = saturated_fit[0].summary()
    assert "Phases" in text and "bound penalty" in text


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
        )
        return plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(9), verbose=False)

    return run(2), run(8)


def test_more_tournament_attempts_never_give_a_worse_start(tournaments, plain_start_loss):
    few, many = tournaments
    assert many.history.loss[0] <= few.history.loss[0]
    # The winner replaced the starting value, and the tournament kept the frozen leaf.
    assert many.history.loss[0] != plain_start_loss
    assert float(many.model.predictors["frozen"].value) == pytest.approx(0.7)


def test_validation_drives_patience_as_well_as_the_returned_step(plain_data):
    # An unreachable validation target: its loss stops improving long before the
    # training loss does, so patience fires on it.
    far = hm.make_dataset(experiments(rate=lambda temp: 2.5), output_channel_names=CHANNELS)
    config = hm.OptaxTrainingConfig(steps=200, lr=0.05, patience=5)
    result = plain_model(TRUE_RATE).fit(
        plain_data, config, validation=far, freeze="frozen", key=jr.key(0), verbose=False
    )
    assert result.stop_reason == "patience" and len(result.history.loss) < 200
    assert result.best_step == int(np.argmin(result.history.validation_loss))


def test_the_same_key_reproduces_a_run_and_another_key_does_not(plain_data):
    config = hm.OptaxTrainingConfig(steps=3, lr=0.05, tournament_attempts=2, tournament_steps=2)
    first, second, other = (
        plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(seed), verbose=False)
        for seed in (11, 11, 12)
    )
    assert first.history == second.history
    assert first.history != other.history


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
    return hm.make_dataset(experiments(one_length=True), output_channel_names=CHANNELS)


@pytest.fixture(scope="session")
def plain_start_loss(plain_data) -> float:
    """The loss before any training, from a step that cannot move the parameters."""
    config = hm.OptaxTrainingConfig(steps=1, lr=0.0)
    return plain_model().fit(plain_data, config, key=jr.key(0), verbose=False).history.loss[0]


@pytest.fixture(scope="session")
def evolved(plain_data):
    runs = {}
    for name, settings in EVOSAX_RUNS.items():
        ui = RecordingUI()
        config = hm.EvosaxTrainingConfig(population_size=16, steps=25, **settings)
        result = plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(3), ui=ui)
        runs[name] = (result, ui)
    return runs


@pytest.mark.parametrize("run", EVOSAX_RUNS)
def test_evosax_improves_on_the_start_and_records_the_best_so_far(evolved, plain_start_loss, run):
    result, ui = evolved[run]
    assert fitted_rate(result) == pytest.approx(TRUE_RATE, abs=0.15)
    assert len(result.history.loss) == 25
    assert list(result.history.loss) == sorted(result.history.loss, reverse=True)
    assert result.stop_reason == "budget_exhausted"
    assert len(events_named(ui, "on_generation_end")) == 25
    assert events_named(ui, "on_run_end")[0]["final_loss"] == pytest.approx(result.final_loss)
    assert result.final_loss < plain_start_loss


@pytest.mark.parametrize("run", EVOSAX_RUNS)
def test_evosax_keeps_frozen_leaves(evolved, run):
    result, _ = evolved[run]
    assert float(result.model.predictors["frozen"].value) == pytest.approx(0.7)


def test_evosax_validation_selects_a_generation(plain_data):
    result = plain_model().fit(
        plain_data,
        hm.EvosaxTrainingConfig(population_size=8, steps=8, sigma_init=0.5),
        validation=plain_data,
        freeze="frozen",
        key=jr.key(4),
        verbose=False,
    )
    assert len(result.history.validation_loss) == 8
    assert result.best_step == int(np.argmin(result.history.validation_loss))


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ensembles(plain_data):
    config = hm.OptaxTrainingConfig(steps=40, lr=0.05, tournament_steps=5, tournament_lr=0.05)
    model = plain_model()
    ui = RecordingUI()
    seeds = model.fit_ensemble(
        plain_data, config, n_members=2, n_seeds=3, freeze="frozen", key=jr.key(5), ui=ui
    )
    bootstrap = model.fit_ensemble(
        plain_data,
        config,
        n_members=2,
        bootstrap=True,
        freeze="frozen",
        key=jr.key(6),
        verbose=False,
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
    measured = sum(int(bp.mask[..., 0].sum()) for bp in plain_data.bucket_payloads)
    assert int(hm.compute_metrics(combined)["a"].n) == measured


@pytest.fixture(scope="session")
def validated_ensemble(plain_data):
    config = hm.OptaxTrainingConfig(steps=20, lr=0.05)
    # Scored on a set with a different bucket layout than the training data.
    validation = hm.make_dataset(experiments()[-3:], output_channel_names=CHANNELS)
    return plain_model().fit_ensemble(
        plain_data,
        config,
        n_members=3,
        freeze="frozen",
        validation=validation,
        key=jr.key(7),
        verbose=False,
    )


def test_ensemble_validation_selects_each_members_step(validated_ensemble):
    for member in validated_ensemble:
        assert len(member.history.validation_loss) == 20
        assert member.best_step == int(np.argmin(member.history.validation_loss))


def test_ensemble_validation_ranks_the_members(validated_ensemble):
    scores = [m.history.validation_loss[m.best_step] for m in validated_ensemble]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Capped buckets: irregular data padded into fewer compiled shapes
# ---------------------------------------------------------------------------

CAPPED = 2


def many_lengths() -> list[hm.Experiment]:
    """Eight experiments with four different grid lengths."""
    sizes = (3, 3, 4, 5, 5, 7, 7, 3)
    rng = np.random.default_rng(0)
    built = []
    for i, size in enumerate(sizes):
        ts = jnp.asarray(np.sort(rng.uniform(0.1, 3.0, size)), dtype=jnp.float32)
        built.append(
            hm.make_experiment(
                channels={
                    "a": hm.ChannelObs(ts=ts, values=jnp.exp(-TRUE_RATE * ts)),
                    "b": hm.ChannelObs(ts=ts[:1], values=1 - jnp.exp(-TRUE_RATE * ts[:1])),
                },
                covariates={"temp": 1.0},
                y0=[1.0, 0.0],
                t0=0.0,
                exp_id=f"e{i}",
            )
        )
    return built


@pytest.fixture(scope="session")
def capped_data() -> hm.Dataset:
    return hm.make_dataset(many_lengths(), output_channel_names=CHANNELS, max_buckets=CAPPED)


@pytest.fixture(scope="session")
def capped_fit(capped_data):
    traces: list[Any] = []
    field = hm.HybridModel.from_vector_field(
        vector_field=plain_field,
        predictors=plain_model().predictors,
        state_to_output=lambda state: state,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    ).simulate_fn

    def counted(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Any:
        traces.append(ts.shape)
        return field(predictors, ts, covariates, y0, solver)

    model = hm.HybridModel(
        predictors=plain_model().predictors,
        simulate_fn=counted,
        state_to_output=lambda state: state,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    config = hm.OptaxTrainingConfig(steps=80, lr=0.05)
    return model.fit(capped_data, config, freeze="frozen", key=jr.key(0), verbose=False), traces


def test_capping_limits_the_buckets_and_keeps_the_data(capped_data):
    uncapped = hm.make_dataset(many_lengths(), output_channel_names=CHANNELS)
    assert len(uncapped.bucket_payloads) == 4
    assert len(capped_data.bucket_payloads) == CAPPED
    total = sum(int(bp.n_obs) for bp in capped_data.bucket_payloads)
    assert total == sum(int(bp.n_obs) for bp in uncapped.bucket_payloads)
    for bp in capped_data.bucket_payloads:
        # Strictly increasing, and the same start and end as the data asks for.
        assert bool((jnp.diff(bp.ts, axis=1) > 0).all())
        assert bool((bp.ts[:, 0] == 0.0).all())
        # Every padded row is unobserved in every channel, with neutral values.
        filler = ~bp.mask
        assert bool((bp.y_observed[filler] == 0.0).all()) and bool((bp.yvar[filler] == 1.0).all())


def test_capped_predictions_report_only_the_experiments_own_times(capped_data):
    data = many_lengths()
    model = plain_model(TRUE_RATE)
    capped = model.predict(capped_data)
    plain = model.predict(data)
    assert list(capped) == list(plain) == [e.exp_id for e in data]
    for exp_id in plain:
        np.testing.assert_array_equal(capped[exp_id].times, plain[exp_id].times)
        np.testing.assert_allclose(
            capped[exp_id].outputs["a"], plain[exp_id].outputs["a"], atol=1e-4
        )
        np.testing.assert_array_equal(
            capped[exp_id].observation_mask["a"], plain[exp_id].observation_mask["a"]
        )
    assert {k: int(m.n) for k, m in hm.compute_metrics(capped).items()} == {
        k: int(m.n) for k, m in hm.compute_metrics(plain).items()
    }
    # The raw arrays keep the padded grid, one row per experiment in its bucket.
    assert [b.shape[:2] for b in capped.buckets] == [
        bp.ts.shape for bp in capped_data.bucket_payloads
    ]


def test_capped_grids_keep_each_experiments_span(capped_data):
    model = plain_model(TRUE_RATE)
    for exp_id, experiment in model.predict(capped_data, n_points=5).items():
        source = next(e for e in capped_data.experiments if e.exp_id == exp_id)
        assert float(experiment.times[0]) == 0.0
        assert float(experiment.times[-1]) == pytest.approx(float(source.channels["a"].ts.max()))


def test_a_capped_fit_recovers_the_rate_and_compiles_per_bucket(capped_fit):
    result, traces = capped_fit
    assert fitted_rate(result) == pytest.approx(TRUE_RATE, abs=0.03)
    # One abstract shape check plus one trace per bucket. More would be a retrace.
    assert len(traces) <= CAPPED + 1


def test_splits_and_resamples_apply_the_same_cap(capped_data):
    for subset in hm.split_dataset(capped_data, train=0.6, val=0.2, test=0.2, key=jr.key(0)):
        assert len(subset.bucket_payloads) <= CAPPED
    resample = hm.make_bootstrap_dataset(capped_data, key=jr.key(0))
    assert len(resample.bucket_payloads) <= CAPPED


def test_a_single_time_stays_out_of_the_cap():
    one_point = hm.make_experiment(
        channels={"a": hm.ChannelObs(ts=jnp.array([1.0]), values=jnp.array([0.5]))},
        y0=[1.0],
        exp_id="lone",
    )
    longer = [
        hm.make_experiment(
            channels={"a": hm.ChannelObs(ts=jnp.arange(1.0, n + 1.0), values=jnp.zeros(n))},
            y0=[1.0],
            exp_id=f"n{n}",
        )
        for n in (2, 4)
    ]
    data = hm.make_dataset([*longer, one_point], output_channel_names=("a",), max_buckets=1)
    assert sorted(bp.ts.shape[1] for bp in data.bucket_payloads) == [1, 4]


def test_integer_times_pad_as_floats():
    experiment = hm.make_experiment(
        channels={"a": hm.ChannelObs(ts=jnp.array([1, 2, 4]), values=jnp.zeros(3))},
        y0=[1.0],
        exp_id="i",
    )
    longer = hm.make_experiment(
        channels={"a": hm.ChannelObs(ts=jnp.arange(1.0, 7.0), values=jnp.zeros(6))},
        y0=[1.0],
        exp_id="f",
    )
    data = hm.make_dataset([experiment, longer], output_channel_names=("a",), max_buckets=1)
    row = data.bucket_payloads[0].ts[0]
    assert bool((jnp.diff(row) > 0).all()) and int(data.bucket_payloads[0].mask[0].sum()) == 3


@pytest.mark.parametrize(
    "lengths, limit",
    [((3, 3, 4, 5, 5, 6, 7, 8, 9, 3), 2), ((2, 3, 10, 11, 30, 31, 31), 3), ((5, 9, 12, 20), 3)],
)
def test_the_chosen_buckets_are_the_cheapest(lengths, limit):
    # Brute force over every choice of targets. The longest length is always one.
    def cost(targets: tuple[int, ...]) -> int:
        return sum(min(t for t in targets if t >= length) for length in lengths)

    longest, others = max(lengths), sorted(set(lengths) - {max(lengths)})
    options = [(*c, longest) for n in range(limit) for c in itertools.combinations(others, n)]
    chosen = _choose_bucket_lengths(lengths, limit)
    assert len(chosen) <= limit
    assert cost(chosen) == min(cost(option) for option in options)


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
    return hm.make_dataset(
        experiments(rate=lambda temp: 2.0, one_length=True), output_channel_names=CHANNELS
    )


@pytest.fixture(scope="session", params=["solver-error", "nan-loss"])
def failed(request, fast_reaction_data):
    ui = RecordingUI()
    config = hm.OptaxTrainingConfig(steps=300, lr=0.1)
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
    config = hm.OptaxTrainingConfig(steps=300, lr=0.1, restore_best=False)
    with pytest.warns(RuntimeWarning, match="diverged"):
        result = failing_model("nan-loss").fit(
            fast_reaction_data, config, key=jr.key(0), verbose=False
        )
    assert result.best_step is None
    assert fitted_rate(result) < FAILURE_LIMIT
    remeasured = result.model.fit(
        fast_reaction_data, hm.OptaxTrainingConfig(steps=1, lr=0.0), key=jr.key(0), verbose=False
    )
    assert remeasured.history.loss[0] == pytest.approx(result.history.loss[-1], rel=1e-4)


def test_a_failure_right_after_a_horizon_change_returns_a_measured_step(fast_reaction_data):
    # The running minimum resets at the boundary, leaving no best to restore. The
    # one update of the first phase jumps past the limit, so the second phase
    # fails on its first step and the only measured parameters are the start.
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(1, lr=3.0, horizon=0.5), hm.TrainingPhase(5, lr=0.1)),
    )
    with pytest.warns(RuntimeWarning, match="diverged"):
        result = failing_model("nan-loss").fit(
            fast_reaction_data, config, key=jr.key(0), verbose=False
        )
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
            hm.OptaxTrainingConfig(steps=5, lr=0.1),
            key=jr.key(0),
            verbose=False,
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
    config = hm.EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.1)
    if not reaches_history:
        with pytest.raises((FloatingPointError, eqx.EquinoxRuntimeError)):
            plain_model().fit(plain_data, config, freeze="frozen", key=jr.key(0), verbose=False)
        return
    with pytest.warns(RuntimeWarning, match="diverged"):
        result = plain_model().fit(
            plain_data, config, freeze="frozen", key=jr.key(0), verbose=False
        )
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


def test_a_long_table_loads_like_the_same_experiments_built_by_hand():
    """Channels keep their own times, unsorted rows sort, and a run may lack a channel."""
    table = pd.DataFrame(
        {
            "run": ["a", "a", "a", "b", "a"],
            "time": [2.0, 1.0, 0.5, 1.0, 1.0],
            "channel": ["x", "x", "y", "x", "y"],
            "value": [0.2, 0.5, 9.0, 0.7, 8.0],
            "var": [0.1, 0.2, 0.3, 0.4, 0.5],
            "temp": [300.0, 300.0, 300.0, 310.0, 300.0],
        }
    )
    a, b = hm.experiments_from_long_dataframe(
        table,
        time="time",
        experiment="run",
        channel="channel",
        value="value",
        variance="var",
        covariates=["temp"],
        t0=0.0,
        y0=[1.0],
    )
    assert (a.exp_id, b.exp_id) == ("a", "b")
    np.testing.assert_array_equal(a.channels["x"].ts, [1.0, 2.0])
    np.testing.assert_allclose(a.channels["x"].values, [0.5, 0.2])
    np.testing.assert_allclose(a.channels["x"].variance, [0.2, 0.1])
    np.testing.assert_array_equal(a.channels["y"].ts, [0.5, 1.0])
    assert float(b.covariates["temp"]) == 310.0
    assert b.channels["y"].ts.size == 0
    # Both experiments must make one dataset, which needs the same channels.
    assert len(hm.make_dataset([a, b], output_channel_names=("x", "y")).experiments) == 2


def test_named_values_reads_parameters_and_constants():
    """Scalars are Python numbers, vectors stay arrays, and networks are left out."""
    inner = hm.MLPPredictor(in_size=1, out_size=1, width_size=4, depth=1, key=jr.key(0))
    model = eqx.tree_at(
        lambda m: m.predictors,
        mechanistic_model(),
        {
            "rate": hm.Parameter(0.3, bounds=(0.05, 3.0)),
            "vector": hm.Parameter([0.2, 0.4], bounds=(0.0, 1.0)),
            "g": hm.Constant(2.0),
            "net": hm.BoundedPredictor(inner=inner, input_bounds=(0, 1), output_bounds=(0, 1)),
        },
    )
    values = model.named_values()
    assert set(values) == {"rate", "vector", "g"}
    assert values["g"] == 2.0 and isinstance(values["g"], float)
    assert values["rate"] == pytest.approx(0.3)
    np.testing.assert_allclose(values["vector"], [0.2, 0.4], rtol=1e-5)


def stiff_field(t, state, predictors, covariates):
    return -covariates["k"] * predictors["rate"].value * state


def stiff_experiments() -> tuple[hm.Dataset, hm.SolverConfig]:
    """Two runs in one bucket, where only ``stiff`` runs out of explicit solver steps."""
    ts = jnp.linspace(0.1, 1.0, 4)
    runs = [
        hm.make_experiment(
            channels={"x": hm.ChannelObs(ts=ts, values=jnp.exp(-k * ts))},
            covariates={"k": k},
            y0=[1.0],
            t0=0.0,
            exp_id=name,
        )
        for name, k in [("gentle", 1.0), ("stiff", 1e6)]
    ]
    dataset = hm.make_dataset(runs, output_channel_names=("x",))
    return dataset, hm.SolverConfig(rtol=1e-5, atol=1e-7, dt0=0.01, max_steps=20000)


def test_a_failed_first_step_names_the_experiment_and_a_better_solver():
    dataset, solver = stiff_experiments()
    model = hm.HybridModel.from_vector_field(
        vector_field=stiff_field,
        predictors={"rate": hm.Parameter(0.3, bounds=(0.05, 3.0))},
        state_to_output=lambda state: state,
        output_channel_names=("x",),
        solver=solver,
    )
    with pytest.raises(eqx.EquinoxRuntimeError) as caught:
        model.fit(dataset, hm.OptaxTrainingConfig(steps=2, lr=1e-3), key=jr.key(0), verbose=False)
    note = " ".join(caught.value.__notes__)
    assert "Failing: stiff." in note
    assert "gentle" not in note
    assert "max_steps is already 20000" in note
    assert "Kvaerno3" in note


def test_failure_details_cannot_hide_the_original_error():
    """A ``simulate_fn`` that breaks on the re-solve still leaves usable advice."""
    dataset, solver = stiff_experiments()
    exc = RuntimeError("The maximum number of solver steps was reached.")

    def broken(*args, **kwargs):
        raise TypeError("not a fit failure")

    details = failure_details(exc, {}, dataset=dataset, simulate_fn=broken, solver=solver)
    assert details.startswith("max_steps is already 20000")


def test_summary_reports_whether_the_loss_was_still_moving():
    """Only the final phase counts, and a short run has too few steps to say."""
    falling = [1.0] * 50 + [float(0.99**i) for i in range(100)]
    history = hm.FitHistory(loss=tuple(falling), phase=(0,) * 50 + (1,) * 100)
    assert _loss_trend(history) == pytest.approx(0.99**10 - 1.0, rel=1e-6)
    assert _loss_trend(hm.FitHistory(loss=(1.0, 0.9))) is None
    # Evosax records no phases, and its best-so-far series can settle exactly.
    assert _loss_trend(hm.FitHistory(loss=(1.0,) * 40)) == 0.0


def test_a_prediction_result_says_where_per_experiment_fields_are():
    result = hm.PredictionResult({})
    with pytest.raises(AttributeError, match=r"result\[exp_id\]\.quantiles"):
        _ = result.quantiles
    with pytest.raises(AttributeError) as other:
        _ = result.no_such_field
    assert "keyed by experiment" not in str(other.value)
    assert copy.deepcopy(result) == result


def test_a_run_saved_with_the_old_verbose_field_still_loads(tmp_path):
    """0.2 wrote ``verbose`` into a saved config; ``load`` must ignore it."""
    model = mechanistic_model()
    config = hm.OptaxTrainingConfig(steps=3, lr=1e-3)
    hm.save(tmp_path, model, config=config)
    metadata = tmp_path / "metadata.json"
    raw = json.loads(metadata.read_text())
    raw["config"]["fields"]["verbose"] = False
    metadata.write_text(json.dumps(raw))
    assert hm.load(tmp_path, like=model).config == config


def test_short_form_warps_build_the_same_scalers_as_explicit_ones():
    """``input_warp`` and ``output_warp`` are shorthand for a warped ``BoundScaler``."""
    inner = hm.MLPPredictor(in_size=1, out_size=1, width_size=4, depth=1, key=jr.key(0))
    short = hm.BoundedPredictor(
        inner=inner,
        input_bounds=(0.1, 10.0),
        input_warp="log",
        output_bounds=(1e-6, 1e-2),
        output_warp="log10",
    )
    long = hm.BoundedPredictor(
        inner=inner,
        in_scaler=hm.BoundScaler(((0.1, 10.0),), warp="log"),
        out_scaler=hm.BoundScaler(((1e-6, 1e-2),), warp="log10"),
    )
    np.testing.assert_array_equal(short(jnp.array([2.0])), long(jnp.array([2.0])))
    assert (short.in_scaler.warp, short.out_scaler.warp) == ("log", "log10")


def test_an_array_experiment_needs_no_id():
    """``make_dataset`` names it, as it does for ``make_experiment``."""
    anonymous = hm.experiment_from_array(
        ts=[0.0, 1.0], values=[1.0, 0.5], channel_names=("x",), y0=[1.0]
    )
    (named,) = hm.make_dataset([anonymous], output_channel_names=("x",)).experiments
    assert named.exp_id


def test_a_bare_model_saves_without_a_history(tmp_path):
    hm.save(tmp_path, mechanistic_model())
    saved = hm.load(tmp_path, like=mechanistic_model())
    assert saved.history == hm.FitHistory() and saved.stop_reason is None and saved.config is None


def test_callables_come_back_as_their_names(tmp_path):
    """A schedule and a custom loss cannot be rebuilt, so they reload as raw dicts."""
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(1, lr=optax.linear_schedule(0.1, 0.0, 1)),)
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
def custom_fits():
    train = hm.make_dataset(
        experiments(rate=rate_law, one_length=True), output_channel_names=CHANNELS
    )
    config = hm.OptaxTrainingConfig(steps=15, lr=0.02, optimizer=optax.adam)
    fits = {}
    for name, inner in INNERS.items():
        model = network_model(jr.key(0), inner(jr.key(7)))
        fits[name] = model.fit(
            train, config, objective=hm.Objective(loss=weighted_loss), key=jr.key(8), verbose=False
        )
    return fits


@pytest.mark.parametrize("name", INNERS)
def test_other_predictors_train_inside_a_bounded_predictor(custom_fits, name):
    result = custom_fits[name]
    assert result.final_loss < result.history.loss[0]
    assert hm.count_trainable_params(result.model.predictors) > 1


@pytest.mark.parametrize("name", INNERS)
def test_other_predictors_reload_bit_exactly(custom_fits, name, tmp_path):
    result = custom_fits[name]
    hm.save(tmp_path, result)
    saved = hm.load(tmp_path, like=network_model(jr.key(0), INNERS[name](jr.key(7))))
    temps = {"temp": jnp.linspace(0.0, 2.0, 11)}
    np.testing.assert_array_equal(
        hm.evaluate_predictor(saved.model.predictors["rate"], temps),
        hm.evaluate_predictor(result.model.predictors["rate"], temps),
    )
