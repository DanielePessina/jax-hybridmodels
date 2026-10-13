"""The HybridModel workflow end to end: fit, predict, simulate, score, save, reload.

Every test runs on the decay fixtures in ``tests/_models.py``. Most read the
shared ``fitted`` session fixture; the few that need a different config
train their own model with a small step budget. Oracles sit outside the
framework where possible: the analytic decay curve, hand counts of
observed cells, and a trace-counting ``simulate_fn``.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from _models import (
    CHANNELS,
    SOLVER,
    TRUE_RATE,
    decay_dataset,
    identity,
    make_model,
    network_model,
    network_predictors,
    parameter_model,
    parameter_predictors,
    quiet,
    simulate_parameter,
)
from jax import Array
from rich.console import Console

import jaxhybridmodels as hm
import jaxhybridmodels.training.evosax as evosax_module
import jaxhybridmodels.training.optax as optax_module
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.serialise import load_predictors, save_predictors
from jaxhybridmodels.ui import EvosaxUI, RichEvosaxUI, RichTrainingUI, SilentUI, TrainingUI
from jaxhybridmodels.ui.testing import RecordingUI

# A number printed to at least four decimals, the precision the UIs promise.
_SIG4 = re.compile(r"\d\.\d{4,}")
# Observed cells in the decay dataset: 3 + 2 + 3 + 2 across experiments a-d.
_N_OBSERVED = 10


def _console() -> Console:
    """A non-TTY recording console: ``Live`` prints once, at ``stop()``."""
    return Console(record=True, force_terminal=False, width=120, file=StringIO())


def _analytic(rate: float, ts: Array) -> Array:
    return jnp.exp(-rate * ts)


def _rate(model: hm.HybridModel) -> float:
    return float(model.predictors["rate"].value)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def test_fit_recovers_the_rate_and_leaves_the_inputs_alone(fitted):
    """A fit returns a new model; the original and its Constant are untouched."""
    result, _ = fitted
    assert abs(_rate(result.model) - TRUE_RATE) < 0.05
    assert float(result.model.predictors["offset"].value) == 0.0
    # The fixture fits parameter_model(), which starts at rate 0.3. A fresh
    # build is the oracle because the original object is not exposed.
    assert result.model.predictors is not parameter_model().predictors
    assert result.model.solver is SOLVER


def test_fit_history_records_every_step_of_every_phase(fitted):
    """One history entry per step, tagged with its phase, and the best step indexed."""
    result, _ = fitted
    history = result.history
    assert len(history.loss) == len(history.penalty) == len(history.phase) == 50
    assert history.phase == (0,) * 25 + (1,) * 25
    assert history.penalty == (0.0,) * 50
    assert history.validation_loss == ()
    assert history.loss[-1] < 0.05 * history.loss[0]
    assert result.stop_reason == "budget_exhausted"
    # restore_best compares only within the final horizon, so the best
    # step is in phase 1 and holds the lowest loss there.
    assert result.best_step is not None and history.phase[result.best_step] == 1
    assert history.loss[result.best_step] == min(history.loss[25:])


def test_ui_sees_the_whole_lifecycle_in_order(fitted):
    """The trainer fires run, phase and step events in order, one step event per step."""
    _, ui = fitted
    names = [name for name, _ in ui.events]
    assert names[0] == "on_run_start"
    assert names[-1] == "on_run_end"
    assert ui.events[0][1] == {"total_steps": 50, "num_phases": 2}
    assert names.count("on_step_end") == 50
    assert names.count("on_phase_start") == names.count("on_phase_end") == 2
    assert names.index("on_compile_done") < names.index("on_step_end")
    phases = [kw["phase_idx"] for name, kw in ui.events if name == "on_phase_start"]
    assert phases == [0, 1]


@pytest.fixture(scope="module")
def validation_fit(decay_data):
    """Validation data generated at the starting rate, so training only hurts it."""
    validation = decay_dataset(lambda temp: 0.3)
    config = quiet(steps=20, lr=0.03, patience=2)
    return parameter_model().fit(decay_data, config, validation=validation, key=jr.key(2))


def test_validation_picks_the_returned_step_and_drives_patience(validation_fit):
    """With validation=, the validation loss chooses the checkpoint and stops the run."""
    result = validation_fit
    assert result.stop_reason == "patience"
    assert result.best_step == 0
    assert len(result.history.validation_loss) == len(result.history.loss) == 3
    np.testing.assert_allclose(_rate(result.model), 0.3, rtol=1e-6)


def test_stop_reason_follows_the_final_phase(decay_data, monkeypatch):
    """Patience in an early phase does not make the run's stop reason "patience".

    Also covers ``verbose=True``: with no ``ui`` the trainer builds a
    ``RichTrainingUI``, here wired to a recording console.
    """
    console = _console()
    monkeypatch.setattr(optax_module, "RichTrainingUI", lambda: RichTrainingUI(console=console))
    # Phase 0 cannot improve (lr=0), so patience cuts it short. Phase 1
    # improves every step and runs to its budget.
    config = hm.OptaxTrainingConfig(
        phases=(hm.TrainingPhase(10, lr=0.0), hm.TrainingPhase(3, lr=0.05)), patience=2
    )
    result = parameter_model().fit(decay_data, config, key=jr.key(5))
    assert result.history.phase.count(0) < 10
    assert result.history.phase.count(1) == 3
    assert result.stop_reason == "budget_exhausted"
    assert "final" in console.export_text().lower()


def test_history_separates_the_trajectory_and_fixed_point_penalties(
    decay_data, validation_fit, capsys
):
    """``loss`` carries the weighted trajectory penalty, ``penalty`` the weighted bound term.

    ``validation_fit`` starts from the same model on the same data, so its
    first loss is the unpenalised baseline. Also checks that a
    ``verbose=False`` fit prints nothing.
    """

    def trajectory_penalty(state, bp):
        return jnp.asarray(2.0)

    def fixed_penalty(predictors, points):
        return jnp.asarray(3.0)

    objective = hm.Objective(
        penalty_weight=0.25,
        penalty_fn=fixed_penalty,
        trajectory_penalty_fn=trajectory_penalty,
        trajectory_penalty_weight=0.5,
    )
    penalised = parameter_model().fit(
        decay_data, quiet(steps=1, lr=0.01), objective=objective, key=jr.key(3)
    )
    baseline = validation_fit.history.loss[0]
    np.testing.assert_allclose(penalised.history.loss[0] - baseline, 1.0, atol=1e-6)
    np.testing.assert_allclose(penalised.history.penalty, (0.75,), atol=1e-6)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.fixture(scope="module")
def evosax_fit(decay_data):
    """A population fit with ``verbose=True``, rendered to a recording console.

    The extra ``meta`` Constant holds a nested tuple, which the search must
    carry through unchanged even though ``simulate_parameter`` ignores it.
    """
    console = _console()
    predictors = {**parameter_predictors(), "meta": hm.Constant((1.0, [2.0, 3.0]))}
    model = eqx.tree_at(lambda m: m.predictors, parameter_model(), predictors)
    config = hm.EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.5)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            evosax_module,
            "RichEvosaxUI",
            lambda log_every=None: RichEvosaxUI(console=console, log_every=log_every),
        )
        result = model.fit(decay_data, config, key=jr.key(4))
    return result, console.export_text()


def test_population_fit_moves_values_and_keeps_constants(evosax_fit):
    """Evosax fills a monotone ``loss``, improves the rate, and leaves Constants alone."""
    result, _ = evosax_fit
    loss = result.history.loss
    assert len(loss) == 10
    assert all(b <= a for a, b in zip(loss, loss[1:], strict=False))
    assert result.history.penalty == result.history.phase == result.history.validation_loss == ()
    assert result.best_step == int(np.argmin(loss))
    assert abs(_rate(result.model) - TRUE_RATE) < abs(0.3 - TRUE_RATE)
    meta = result.model.predictors["meta"].value
    assert float(meta[0]) == 1.0
    np.testing.assert_array_equal(meta[1], [2.0, 3.0])


def test_network_fit_learns_a_covariate_dependent_rate(temp_data):
    """A BoundedPredictor rate fits data whose rate depends on ``temp``."""
    model = network_model()
    result = model.fit(temp_data, quiet(steps=50, lr=0.05), key=jr.key(1))
    assert result.history.loss[-1] < 0.1 * result.history.loss[0]
    rates = hm.evaluate_predictor(result.model.predictors["rate"], {"temp": jnp.array([0.5, 1.5])})
    assert rates.shape == (2,)
    slow, fast = (float(r) for r in rates)
    assert 0.05 < slow < fast < 3.0


# ---------------------------------------------------------------------------
# Predict and simulate
# ---------------------------------------------------------------------------


def test_predict_keeps_experiment_order_and_matches_the_analytic_curve(fitted, decay_data):
    """Results come back keyed in input order, not bucket order, on each own grid."""
    model = fitted[0].model
    prediction = model.predict(decay_data)
    # a and c share a bucket, as do b and d, so bucket order is a, c, b, d.
    assert list(prediction) == ["a", "b", "c", "d"]
    assert len(prediction.buckets) == 2
    rate = _rate(model)
    for exp in prediction.values():
        # t0 = 0 is integrated from and kept on the grid, but unobserved.
        assert float(exp.times[0]) == 0.0
        assert not exp.observation_mask["conc"][0]
        assert jnp.isnan(exp.observed["conc"][0])
        np.testing.assert_allclose(exp.outputs["conc"], _analytic(rate, exp.times), rtol=1e-3)
    np.testing.assert_allclose(prediction["b"].times, [0.0, 1.0, 2.0])


def test_dataframe_export_has_one_row_per_cell(fitted, decay_data):
    """The long table keeps experiment, channel and observation identity."""
    pytest.importorskip("pandas")
    frame = fitted[0].model.predict(decay_data).to_dataframe()
    assert list(frame.columns) == [
        "experiment_id",
        "time",
        "channel",
        "prediction",
        "observed",
        "is_observed",
    ]
    assert frame["experiment_id"].drop_duplicates().tolist() == ["a", "b", "c", "d"]
    assert frame["channel"].unique().tolist() == ["conc"]
    assert int(frame["is_observed"].sum()) == _N_OBSERVED
    assert int(frame["observed"].isna().sum()) == len(frame) - _N_OBSERVED


def test_simulate_integrates_from_an_earlier_t0(fitted):
    """``t0`` before the first requested time is integrated from but not returned."""
    model = fitted[0].model
    trajectory = model.simulate(ts=[1.0, 2.0], y0=[1.0], t0=0.0)
    np.testing.assert_allclose(trajectory.times, [1.0, 2.0])
    np.testing.assert_allclose(
        trajectory.outputs["conc"], _analytic(_rate(model), jnp.array([1.0, 2.0])), rtol=1e-3
    )
    assert trajectory.state is not None and trajectory.state.shape == (2, 1)


def test_vector_covariates_reach_each_simulation():
    """Each experiment's own covariate vector reaches its vmapped simulate_fn call."""

    def simulate(predictors, ts, covariates, y0, solver):
        return jnp.broadcast_to(covariates["features"].sum(), (ts.shape[0], 1))

    ts = jnp.array([0.0, 1.0])
    experiments = [
        hm.make_experiment(
            channels={"conc": hm.ChannelObs(ts=ts, values=jnp.zeros(2))},
            covariates={"features": jnp.array(features)},
            y0=[0.0],
            exp_id=name,
        )
        for name, features in [("x", [1.0, 2.0]), ("y", [3.0, 4.0])]
    ]
    dataset = hm.make_dataset(experiments, output_channel_names=CHANNELS)
    model = hm.HybridModel(
        predictors=parameter_predictors(),
        simulate_fn=simulate,
        state_to_output=identity,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    prediction = model.predict(dataset)
    np.testing.assert_array_equal(prediction["x"].outputs["conc"], [3.0, 3.0])
    np.testing.assert_array_equal(prediction["y"].outputs["conc"], [7.0, 7.0])


def test_prediction_compiles_once_per_bucket_shape(fitted, decay_data):
    """Same shapes reuse the trace, and new parameter values do not retrace.

    The explicit grid starts after ``t0``; each experiment still integrates
    from its own ``t0``, so every row matches the analytic curve.
    """
    traces = []

    def counting(predictors, ts, covariates, y0, solver):
        traces.append(ts.shape)
        return simulate_parameter(predictors, ts, covariates, y0, solver)

    def model_with(predictors: Any) -> hm.HybridModel:
        return hm.HybridModel(
            predictors=predictors,
            simulate_fn=counting,
            state_to_output=identity,
            output_channel_names=CHANNELS,
            solver=SOLVER,
        )

    trained, other = model_with(fitted[0].model.predictors), model_with(parameter_predictors(0.5))
    trained.predict(decay_data)
    assert len(traces) == 2, "expected one trace per distinct bucket shape"
    trained.predict(decay_data)
    other.predict(decay_data)
    assert len(traces) == 2

    grid = jnp.array([1.0, 2.0])
    on_grid = trained.predict(decay_data, ts=grid)
    compiled = len(traces)
    other.predict(decay_data, ts=grid)
    assert compiled == 4 and len(traces) == compiled
    expected = _analytic(_rate(fitted[0].model), grid)
    for experiment in on_grid.values():
        np.testing.assert_allclose(experiment.outputs["conc"], expected, rtol=1e-3)


def test_integer_observations_with_a_fractional_t0_are_promoted_to_float():
    """Integer times and states must not truncate a fractional ``t0``."""
    experiment = hm.make_experiment(
        y0=[1], t0=0.5, channels={"conc": hm.ChannelObs(ts=[1, 2], values=[0.8, 0.6])}
    )
    dataset = hm.make_dataset([experiment], output_channel_names=CHANNELS)
    assert jnp.issubdtype(experiment.y0.dtype, jnp.floating)
    np.testing.assert_allclose(dataset.bucket_payloads[0].ts[0], [0.5, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_count_only_observed_cells_and_score_a_good_fit(fitted, decay_data, capsys):
    """compute_metrics masks out padding and t0 rows; print_metrics prints them."""
    prediction = fitted[0].model.predict(decay_data)
    metrics = hm.compute_metrics(prediction)
    assert set(metrics) == {"conc"}
    conc = metrics["conc"]
    assert conc.n == _N_OBSERVED
    assert float(conc.rmse) < 0.02 and float(conc.r2) > 0.99

    bad = parameter_model(rate=2.0).predict(decay_data)
    assert float(hm.compute_metrics(bad)["conc"].mse) > float(conc.mse)

    hm.print_metrics(metrics, header="RESULTS")
    hm.print_metrics(prediction, header="DIRECT")
    out = capsys.readouterr().out
    assert "RESULTS" in out and "DIRECT" in out and out.count("conc") == 2
    assert "2 bucket(s)" in hm.describe_buckets(decay_data)


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


def _metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metadata.json").read_text())


def _edit_metadata(run_dir: Path, edit) -> None:
    metadata = _metadata(run_dir)
    edit(metadata)
    (run_dir / "metadata.json").write_text(json.dumps(metadata))


def test_a_fitted_run_reloads_and_predicts_identically(fitted, decay_data, tmp_path):
    """save then load gives back the same predictions, config and history."""
    result, _ = fitted
    config = quiet(
        phases=(hm.TrainingPhase(25, lr=0.1, horizon=0.5), hm.TrainingPhase(25, lr=0.05))
    )
    run_dir = tmp_path / "does" / "not" / "exist"
    # A first save to the same directory is overwritten by the second.
    hm.save(run_dir, parameter_model(), extras={"v": 1})
    hm.save(run_dir, result, config=config, extras={"v": 2})
    assert (run_dir / "predictors.eqx").exists()

    def reject_nonfinite(token):
        raise ValueError(f"Non-standard JSON token: {token}")

    # The rate's open upper bound must not leak Infinity into the JSON.
    metadata = json.loads((run_dir / "metadata.json").read_text(), parse_constant=reject_nonfinite)
    assert metadata["config"]["kind"] == "optax"
    assert [leaf["path"] for leaf in metadata["predictors"]["leaves"]] == [
        "['offset']",
        "['rate']",
    ]

    loaded = hm.load(run_dir, like=parameter_model())
    assert isinstance(loaded, hm.SavedRun)
    assert loaded.config == config
    assert loaded.model.solver.to_dict() == SOLVER.to_dict()
    assert loaded.history == result.history
    assert (loaded.best_step, loaded.stop_reason) == (result.best_step, result.stop_reason)
    assert loaded.objective is None
    assert loaded.extras == {"v": 2}
    for exp_id, exp in loaded.model.predict(decay_data).items():
        np.testing.assert_array_equal(
            exp.outputs["conc"], result.model.predict(decay_data)[exp_id].outputs["conc"]
        )


def test_a_bare_model_saves_without_history(tmp_path):
    """A HybridModel saves its weights alone; the history comes back empty."""
    model = parameter_model()
    hm.save(tmp_path, model)
    loaded = hm.load(tmp_path, like=parameter_model())
    assert loaded.history == hm.FitHistory()
    assert (loaded.best_step, loaded.stop_reason, loaded.config) == (None, None, None)
    assert _rate(loaded.model) == _rate(model)


def test_load_keeps_like_solver_and_save_accepts_any_solver(tmp_path):
    """The saved solver is a record; ``load`` keeps ``like.solver`` and checks it.

    The record names classes only, so rebuilding from it would drop
    constructor arguments such as ``checkpoints=7``. An unregistered solver
    still saves, and fails nothing half-way.
    """

    def with_solver(solver: hm.SolverConfig) -> hm.HybridModel:
        return hm.HybridModel(
            predictors=parameter_predictors(),
            simulate_fn=simulate_parameter,
            state_to_output=identity,
            output_channel_names=CHANNELS,
            solver=solver,
        )

    checkpointed = hm.SolverConfig(adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=7))
    hm.save(tmp_path / "a", with_solver(checkpointed))
    loaded = hm.load(tmp_path / "a", like=with_solver(checkpointed))
    assert loaded.model.solver == checkpointed
    with pytest.warns(RuntimeWarning, match="keeping like.solver"):
        loaded = hm.load(tmp_path / "a", like=parameter_model())
    assert loaded.model.solver == SOLVER

    stiff = hm.SolverConfig(solver=diffrax.Kvaerno5())
    hm.save(tmp_path / "b", with_solver(stiff))
    metadata = json.loads((tmp_path / "b" / "metadata.json").read_text())
    assert "Kvaerno5" in metadata["solver"]["unregistered"]
    assert hm.load(tmp_path / "b", like=with_solver(stiff)).model.solver == stiff


def test_nonfinite_history_entries_survive_a_save(tmp_path):
    """NaN and inf losses are written as markers and read back as floats."""
    history = hm.FitHistory(loss=(1.0, float("nan"), float("inf")), phase=(0, 0, 0))
    result = hm.FitResult(
        model=parameter_model(), history=history, best_step=0, stop_reason="patience"
    )
    hm.save(tmp_path, result)
    loss = hm.load(tmp_path, like=parameter_model()).history.loss
    assert loss[0] == 1.0 and np.isnan(loss[1]) and loss[2] == float("inf")


def test_save_rejects_what_it_cannot_write(tmp_path):
    """Only a model or a fit saves, and only a training config goes in config=."""
    with pytest.raises(TypeError, match="HybridModel or FitResult"):
        hm.save(tmp_path, parameter_predictors())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="OptaxTrainingConfig or EvosaxTrainingConfig"):
        hm.save(tmp_path, parameter_model(), config=Objective())  # ty: ignore[invalid-argument-type]


def test_network_predictors_round_trip_bit_exact(tmp_path):
    """save_predictors / load_predictors restore every leaf into a fresh template."""
    predictors = network_predictors(jr.key(0))
    path = str(tmp_path / "net.eqx")
    save_predictors(path, predictors)
    restored = load_predictors(path, like=network_predictors(jr.key(99)))
    leaves = jax.tree_util.tree_leaves(eqx.filter(predictors, eqx.is_inexact_array))
    back = jax.tree_util.tree_leaves(eqx.filter(restored, eqx.is_inexact_array))
    assert len(leaves) == len(back)
    for a, b in zip(leaves, back, strict=True):
        np.testing.assert_array_equal(a, b)


def test_an_evosax_config_reloads_as_its_own_type(tmp_path):
    """The config's kind is recorded, so an Evosax config comes back as one."""
    evosax = hm.EvosaxTrainingConfig(population_size=8, steps=2, verbose=False)
    hm.save(tmp_path, parameter_model(), config=evosax)
    assert _metadata(tmp_path)["config"]["kind"] == "evosax"
    assert hm.load(tmp_path, like=parameter_model()).config == evosax


def test_callables_in_a_config_are_described_not_pickled(tmp_path):
    """Callables save as a dotted name and load back as the raw dict."""

    def optimizer_factory(learning_rate):
        return optax.adam(learning_rate)

    def penalty_fn(predictors, points):
        return jnp.asarray(0.0)

    hm.save(
        tmp_path / "factory",
        parameter_model(),
        config=quiet(steps=1, optimizer=optimizer_factory),
        objective=Objective(loss=masked_mse, penalty_fn=penalty_fn),
    )
    saved = _metadata(tmp_path / "factory")
    assert "__callable__" in saved["config"]["fields"]["phases"][0]["optimizer"]
    assert saved["objective"]["loss"] == {"__callable__": "jaxhybridmodels.losses.masked_mse"}
    assert "__callable__" in saved["objective"]["penalty_fn"]
    raw = hm.load(tmp_path / "factory", like=parameter_model())
    assert isinstance(raw.config, dict) and isinstance(raw.objective, dict)
    raw_config, raw_objective = (
        cast(dict[str, Any], raw.config),
        cast(dict[str, Any], raw.objective),
    )
    assert "__callable__" in raw_config["phases"][0]["optimizer"]
    assert raw_objective["loss"] == {"__callable__": "jaxhybridmodels.losses.masked_mse"}

    schedule = quiet(phases=(hm.TrainingPhase(1, lr=optax.linear_schedule(0.1, 0.0, 1)),))
    hm.save(tmp_path / "schedule", parameter_model(), config=schedule)
    fields = _metadata(tmp_path / "schedule")["config"]["fields"]
    assert "__callable__" in fields["phases"][0]["lr"]

    raw_optimizer = quiet(steps=1, optimizer=optax.adam(1e-3))
    hm.save(tmp_path / "raw", parameter_model(), config=raw_optimizer)
    fields = _metadata(tmp_path / "raw")["config"]["fields"]
    assert "__opaque__" in fields["phases"][0]["optimizer"]


def test_an_objective_without_callables_round_trips(tmp_path):
    """Registry losses, weights by name and keyed penalty points come back as an Objective."""
    objective = Objective(
        loss="bal_mse",
        channel_weights={"conc": 2.0},
        penalty_weight=0.1,
        penalty_points={"rate": jnp.array([[0.5], [1.5]])},
    )
    hm.save(tmp_path, parameter_model(), objective=objective)
    loaded = hm.load(tmp_path, like=parameter_model()).objective
    assert isinstance(loaded, Objective)
    assert (loaded.loss, loaded.channel_weights, loaded.penalty_weight) == (
        "bal_mse",
        {"conc": 2.0},
        0.1,
    )
    assert loaded.penalty_points is not None
    np.testing.assert_array_equal(loaded.penalty_points["rate"], [[0.5], [1.5]])


class _Scaled(hm.Predictor):
    """A custom predictor whose static field changes what it computes."""

    weight: Array
    scale: float = eqx.field(static=True)

    def __init__(self, scale: float) -> None:
        self.weight = jnp.asarray([2.0])
        self.scale = scale

    def __call__(self, x: Array) -> Array:
        return self.weight * self.scale


def _bump_version(metadata: dict[str, Any]) -> None:
    metadata["version"] = "999.0.0"


def _add_config_field(metadata: dict[str, Any]) -> None:
    metadata["config"]["fields"]["future_field"] = True


def _rename_channel(metadata: dict[str, Any]) -> None:
    metadata["output_channel_names"] = ["other"]


@pytest.mark.parametrize(
    "saved,template,edit,match",
    [
        pytest.param(
            network_predictors,
            lambda: {
                **network_predictors(),
                "rate": hm.BoundedPredictor(
                    inner=network_predictors()["rate"].inner,
                    input_keys=("temp",),
                    input_bounds=((0.0, 10.0),),
                    output_bounds=((0.05, 3.0),),
                ),
            },
            None,
            "static",
            id="network-bounds-differ",
        ),
        pytest.param(
            parameter_predictors,
            lambda: {"rate": hm.Parameter(0.3, bounds=(0.0, 5.0)), "offset": hm.Constant(0.0)},
            None,
            "static",
            id="parameter-bounds-differ",
        ),
        pytest.param(lambda: _Scaled(3.0), lambda: _Scaled(99.0), None, "static", id="custom"),
        pytest.param(parameter_predictors, parameter_predictors, _bump_version, "version"),
        pytest.param(parameter_predictors, parameter_predictors, _add_config_field, "future_field"),
        pytest.param(
            parameter_predictors,
            parameter_predictors,
            _rename_channel,
            "output_channel_names",
        ),
    ],
)
def test_load_rejects_a_mismatched_run(tmp_path, saved, template, edit, match):
    """Loading fails loudly rather than silently mis-reading a different model."""
    hm.save(tmp_path, make_model(saved()), config=quiet(steps=1))
    if edit is not None:
        _edit_metadata(tmp_path, edit)
    with pytest.raises(ValueError, match=match):
        hm.load(tmp_path, like=make_model(template()))


# ---------------------------------------------------------------------------
# Rich dashboards
# ---------------------------------------------------------------------------


def test_training_dashboard_renders_a_real_run(fitted):
    """Replaying the fixture's events shows phases, optimiser, losses and a final summary."""
    _, recorded = fitted
    console = _console()
    ui = RichTrainingUI(console=console)
    for name, kwargs in recorded.events:
        getattr(ui, name)(**kwargs)
    text = console.export_text()
    lower = text.lower()
    assert "phase" in lower and "adamw" in lower
    assert "final" in lower[-len(lower) // 3 :]
    assert _SIG4.search(text) is not None
    # Piped output must end on a fresh line, or the next print joins the border.
    assert text.endswith("\n") and text.rstrip().endswith("╯")


def test_evosax_dashboard_renders_a_real_run(evosax_fit):
    """``verbose=True`` renders population size, generations and fitness to the console."""
    _, text = evosax_fit
    lower = text.lower()
    assert "8" in text
    assert "gen" in lower
    assert "final" in lower[-len(lower) // 3 :]
    assert _SIG4.search(text) is not None


def test_evosax_log_every_throttles_the_generation_table():
    """``log_every=2`` shows generations 0, 2 and 4 only."""
    console = _console()
    ui = RichEvosaxUI(console=console, log_every=2, recent_generations=10)
    ui.on_run_start(total_steps=5, population_size=4)
    bests = [0.7771, 0.6661, 0.5551, 0.4441, 0.3331]
    for i, best in enumerate(bests):
        ui.on_generation_end(gen_idx=i, best_loss=best, mean_loss=best + 0.1)
    ui.on_run_end(final_loss=0.2221)
    text = console.export_text()
    assert [f"{b:.4f}" in text for b in bests] == [True, False, True, False, True]


def test_dashboards_tolerate_out_of_order_events():
    """Events with no matching start are ignored, not raised."""
    training = RichTrainingUI(console=_console())
    training.on_run_start(total_steps=1, num_phases=1)
    training.on_phase_end(phase_idx=0)
    training.on_step_end(step_idx=0, phase_idx=0, loss=0.0)
    training.on_run_end(final_loss=0.0)
    evosax = RichEvosaxUI(console=_console())
    evosax.on_run_start(total_steps=3, population_size=4)
    evosax.on_run_end(final_loss=0.0)


@pytest.mark.parametrize(
    "ui", [RichTrainingUI(console=_console()), SilentUI(), RecordingUI()], ids=type
)
def test_training_uis_satisfy_the_protocol(ui):
    assert isinstance(ui, TrainingUI)


@pytest.mark.parametrize(
    "ui", [RichEvosaxUI(console=_console()), SilentUI(), RecordingUI()], ids=type
)
def test_evosax_uis_satisfy_the_protocol(ui):
    assert isinstance(ui, EvosaxUI)


# ---------------------------------------------------------------------------
# The model as an Equinox module, and its conveniences
# ---------------------------------------------------------------------------


def test_the_model_is_an_equinox_module():
    """``eqx.tree_at`` swaps one predictor; static fields carry over untouched."""
    model = parameter_model()
    swapped = eqx.tree_at(lambda m: m.predictors["rate"], model, hm.Parameter(0.9))
    assert _rate(swapped) == pytest.approx(0.9) and _rate(model) == pytest.approx(0.3)
    assert swapped.simulate_fn is simulate_parameter and swapped.solver is SOLVER
    # Only the predictors are array leaves.
    assert len(jax.tree_util.tree_leaves(model)) == len(jax.tree_util.tree_leaves(model.predictors))
    with pytest.raises(TypeError):
        hm.HybridModel(parameter_predictors(), simulate_parameter, identity, CHANNELS)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_filter_vmap_builds_a_batch_of_models():
    """Building models under ``eqx.filter_vmap`` gives one stacked model, one member per key."""
    batch = eqx.filter_vmap(network_model)(jr.split(jr.key(0), 3))
    weights = batch.predictors["rate"].inner.mlp.layers[0].weight
    assert weights.shape[0] == 3
    assert not np.allclose(weights[0], weights[1])
    member = jax.tree_util.tree_map(
        lambda x: x[1] if eqx.is_array(x) else x, batch, is_leaf=eqx.is_array
    )
    prediction = member.predict(decay_dataset())
    assert list(prediction) == ["a", "b", "c", "d"]


def test_experiments_stand_in_for_a_dataset(fitted, decay_data):
    """fit, predict and validation= accept a plain list of experiments."""
    experiments = list(decay_data.experiments)
    model = fitted[0].model
    from_list = model.predict(experiments)
    from_dataset = model.predict(decay_data)
    for exp_id in from_dataset:
        np.testing.assert_array_equal(
            from_list[exp_id].outputs["conc"], from_dataset[exp_id].outputs["conc"]
        )
    result = parameter_model().fit(
        experiments, quiet(steps=2, lr=0.05), validation=experiments[:2], key=jr.key(0)
    )
    assert len(result.history.validation_loss) == 2


def test_freeze_holds_named_predictors_fixed(decay_data):
    """``freeze=`` takes paths, classes or predicates on top of the default mask."""
    predictors = {"rate": hm.Parameter(0.3), "offset": hm.Parameter(0.0)}
    model = eqx.tree_at(lambda m: m.predictors, parameter_model(), predictors)
    result = model.fit(decay_data, quiet(steps=5, lr=0.1), freeze="offset", key=jr.key(0))
    assert float(result.model.predictors["offset"].value) == 0.0
    assert _rate(result.model) != pytest.approx(0.3)
    with pytest.raises(ValueError, match="no trainable"):
        model.fit(decay_data, quiet(steps=1), freeze=(hm.Parameter,), key=jr.key(0))
    with pytest.raises(ValueError, match="no trainable"):
        model.fit(
            decay_data,
            quiet(steps=1),
            freeze=lambda m: isinstance(m, hm.Parameter),
            key=jr.key(0),
        )


def test_evosax_fit_selects_by_validation(decay_data):
    """With validation=, Evosax records one validation score per generation."""
    config = hm.EvosaxTrainingConfig(population_size=4, steps=3, verbose=False)
    validation = decay_dataset(lambda temp: 0.3)
    result = parameter_model().fit(decay_data, config, validation=validation, key=jr.key(1))
    history = result.history
    assert len(history.loss) == len(history.validation_loss) == 3
    assert result.best_step == int(np.argmin(history.validation_loss))


def _decay_field(t, y, predictors, covariates):
    return -predictors["rate"].value * y + predictors["offset"].value


def test_from_vector_field_matches_a_hand_written_simulate_fn(decay_data):
    """The wrapped vector field returns a diffrax Solution, and the framework reads its ys."""
    model = hm.HybridModel.from_vector_field(
        vector_field=_decay_field,
        predictors=parameter_predictors(),
        state_to_output=identity,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    reference = parameter_model().predict(decay_data)
    for exp_id, exp in model.predict(decay_data).items():
        np.testing.assert_allclose(
            exp.outputs["conc"], reference[exp_id].outputs["conc"], rtol=1e-6
        )
    trajectory = model.simulate(ts=[0.0, 1.0], y0=[1.0])
    np.testing.assert_allclose(
        trajectory.outputs["conc"], _analytic(0.3, jnp.array([0.0, 1.0])), rtol=1e-3
    )


def test_simulate_reads_the_experiment(fitted, decay_data):
    """``simulate(experiment=)`` takes y0, t0 and covariates from it and defaults to its grid."""
    model = fitted[0].model
    experiment = decay_data.experiments[1]
    trajectory = model.simulate(experiment=experiment)
    predicted = model.predict(decay_data)[experiment.exp_id]
    np.testing.assert_allclose(trajectory.times, predicted.times)
    np.testing.assert_allclose(trajectory.outputs["conc"], predicted.outputs["conc"], rtol=1e-5)
    later = model.simulate(experiment=experiment, ts=[1.5, 2.0])
    np.testing.assert_allclose(
        later.outputs["conc"], _analytic(_rate(model), jnp.array([1.5, 2.0])), rtol=1e-3
    )


def test_fit_checks_output_shapes_before_compiling(decay_data):
    """A wrong ``state_to_output`` width fails up front, before the trainer starts."""
    ui = RecordingUI()
    with pytest.raises(ValueError, match=re.escape("fit: state_to_output returned shape (3, 2)")):
        _model(state_to_output=_doubled).fit(decay_data, quiet(steps=1), key=jr.key(0), ui=ui)
    assert ui.events == []


# ---------------------------------------------------------------------------
# Constants and API errors
# ---------------------------------------------------------------------------


def test_nested_constants_are_frozen_and_survive_a_restart():
    """A Constant of scalars, lists and tuples is never trainable or reinitialised."""
    constant = hm.Constant((1.0, [2.0, 3.0], (4.0, 5.0)))
    assert constant.value[0].shape == () and constant.value[1].shape == (2,)
    assert not any(jax.tree_util.tree_leaves(hm.trainable_mask(constant)))
    fresh = hm.predictors.reinitialize_pytree_with_key({"c": constant}, jr.key(0))["c"]
    for a, b in zip(
        jax.tree_util.tree_leaves(constant), jax.tree_util.tree_leaves(fresh), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def _duplicate_ids() -> list[hm.Experiment]:
    first = decay_dataset().experiments[0]
    return [first, first]


def _other_channels() -> hm.Dataset:
    exp = hm.make_experiment(
        channels={"other": hm.ChannelObs(ts=[1.0], values=[0.5])}, y0=[1.0], t0=0.0
    )
    return hm.make_dataset([exp], output_channel_names=("other",))


def _model(**overrides: Any) -> hm.HybridModel:
    """The Parameter model with some constructor arguments replaced."""
    kwargs: dict[str, Any] = dict(
        predictors=parameter_predictors(),
        simulate_fn=simulate_parameter,
        state_to_output=identity,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    return hm.HybridModel(**{**kwargs, **overrides})


_FROZEN = _model(predictors={"rate": hm.Constant(0.3), "offset": hm.Constant(0.0)})


def _doubled(state: Array) -> Array:
    return jnp.concatenate([state, state], axis=-1)


def _drops_last_row(predictors, ts, covariates, y0, solver):
    return simulate_parameter(predictors, ts, covariates, y0, solver)[:-1]


def _no_saved_state(predictors, ts, covariates, y0, solver):
    term = diffrax.ODETerm(lambda t, y, args: -y)
    # Dense output only: the solution holds an interpolant and no ys.
    saveat = diffrax.SaveAt(dense=True)
    return diffrax.diffeqsolve(term, solver.solver, ts[0], ts[-1], 0.1, y0, saveat=saveat)


@pytest.mark.parametrize(
    "call,match",
    [
        pytest.param(
            lambda: _FROZEN.fit(decay_dataset(), quiet(steps=2), key=jr.key(0)),
            "no trainable",
            id="all-constant",
        ),
        pytest.param(lambda: parameter_model().predict(_other_channels()), "channels", id="chan"),
        pytest.param(lambda: parameter_model().predict(_duplicate_ids()), "duplicate", id="dup"),
        pytest.param(
            lambda: parameter_model().simulate(ts=[1.0, 2.0], y0=[1.0], t0=1.5), "t0", id="t0"
        ),
        pytest.param(
            lambda: parameter_model().simulate(ts=[2.0, 1.0], y0=[1.0]), "increasing", id="ts"
        ),
        pytest.param(
            lambda: parameter_model().simulate(ts=[1.0], y0=[[1.0]]), "rank-1", id="y0-rank"
        ),
        pytest.param(
            lambda: _model(output_channel_names=("a", "a")), "unique", id="duplicate-channels"
        ),
        pytest.param(
            lambda: _model(state_to_output=_doubled).simulate(ts=[1.0], y0=[1.0], t0=0.0),
            re.escape("simulate: state_to_output returned shape (1, 2)"),
            id="output-width",
        ),
        pytest.param(
            lambda: _model(simulate_fn=_drops_last_row).simulate(ts=[1.0, 2.0], y0=[1.0]),
            re.escape("simulate: simulate_fn returned state of shape (1, 1)"),
            id="state-length",
        ),
        pytest.param(
            lambda: _model(state_to_output=_doubled).predict(decay_dataset()),
            re.escape("predict: state_to_output returned shape (3, 2)"),
            id="predict-output-width",
        ),
        pytest.param(
            lambda: _model(simulate_fn=_no_saved_state).simulate(ts=[1.0, 2.0], y0=[1.0]),
            "ys=None",
            id="solution-without-ys",
        ),
        pytest.param(
            lambda: parameter_model().fit(
                decay_dataset(), quiet(steps=1), key=jr.key(0), freeze="rate", trainable=True
            ),
            "not both",
            id="freeze-and-trainable",
        ),
        pytest.param(
            lambda: parameter_model().simulate(experiment=decay_dataset().experiments[0], y0=[1.0]),
            "experiment=",
            id="experiment-and-y0",
        ),
    ],
)
def test_model_api_rejects_bad_input(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_simulate_keeps_y0_at_the_experiment_start():
    """Without ``t0``, ``y0`` belongs to the first observation, whatever ``ts`` is passed."""
    model = parameter_model(rate=0.8)
    experiment = hm.make_experiment(
        channels={"conc": hm.ChannelObs(ts=jnp.array([1.0, 2.0, 3.0]), values=jnp.ones(3))},
        covariates={"temp": 1.0},
        y0=[1.0],
        exp_id="late",
    )
    full = model.simulate(experiment=experiment)
    later = model.simulate(experiment=experiment, ts=[2.0, 3.0])
    np.testing.assert_allclose(later.outputs["conc"], full.outputs["conc"][1:], rtol=1e-4)
    with pytest.raises(ValueError, match="t0 must be finite and at or before"):
        model.simulate(experiment=experiment, ts=[0.5, 1.0])


def test_compute_metrics_names_what_it_takes(fitted, decay_data):
    """Raw bucket arrays are not a result; the error names the entry points that make one."""
    buckets = fitted[0].model.predict(decay_data).buckets
    with pytest.raises(TypeError, match="predict_ensemble"):
        hm.compute_metrics(buckets)


def test_empty_penalty_points_survive_a_save(tmp_path):
    """An empty ``[0, n_inputs]`` grid reloads with its shape, so Objective accepts it."""
    objective = hm.Objective(penalty_points={"rate": jnp.zeros((0, 1)), "other": jnp.ones((2, 3))})
    hm.save(tmp_path, parameter_model(), objective=objective)
    loaded = hm.load(tmp_path, like=parameter_model()).objective
    assert isinstance(loaded, hm.Objective)
    points = loaded.penalty_points
    assert points is not None
    assert points["rate"].shape == (0, 1)
    np.testing.assert_array_equal(points["other"], np.ones((2, 3)))
