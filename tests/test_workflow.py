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
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

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
    assert history.validation_loss == () and history.best_fitness == ()
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

    config = quiet(
        steps=1,
        lr=0.01,
        penalty_weight=0.25,
        penalty_fn=fixed_penalty,
        trajectory_penalty_fn=trajectory_penalty,
        trajectory_penalty_weight=0.5,
    )
    penalised = parameter_model().fit(decay_data, config, key=jr.key(3))
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
    model = replace(parameter_model(), predictors=predictors)
    config = hm.EvosaxTrainingConfig(population_size=8, num_generations=10, sigma_init=0.5)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            evosax_module,
            "RichEvosaxUI",
            lambda log_every=None: RichEvosaxUI(console=console, log_every=log_every),
        )
        result = model.fit(decay_data, config, key=jr.key(4))
    return result, console.export_text()


def test_population_fit_moves_values_and_keeps_constants(evosax_fit):
    """Evosax fills ``best_fitness`` only, improves the rate, and leaves Constants alone."""
    result, _ = evosax_fit
    fitness = result.history.best_fitness
    assert len(fitness) == 10
    assert all(b <= a for a, b in zip(fitness, fitness[1:], strict=False))
    assert result.history.loss == ()
    assert result.best_step == int(np.argmin(fitness))
    assert abs(_rate(result.model) - TRUE_RATE) < abs(0.3 - TRUE_RATE)
    meta = result.model.predictors["meta"].value
    assert float(meta[0]) == 1.0
    np.testing.assert_array_equal(meta[1], [2.0, 3.0])


def test_network_fit_learns_a_covariate_dependent_rate(temp_data):
    """A BoundedPredictor rate fits data whose rate depends on ``temp``."""
    model = network_model()
    result = model.fit(temp_data, quiet(steps=50, lr=0.05), key=jr.key(1))
    assert result.history.loss[-1] < 0.1 * result.history.loss[0]
    slow = hm.evaluate_predictor(result.model.predictors["rate"], {"temp": 0.5})
    fast = hm.evaluate_predictor(result.model.predictors["rate"], {"temp": 1.5})
    assert isinstance(slow, float)
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
            channels={"conc": hm.ChannelObs(ts, jnp.zeros(2))},
            covariates={"features": jnp.array(features)},
            y0=[0.0],
            exp_id=name,
        )
        for name, features in [("x", [1.0, 2.0]), ("y", [3.0, 4.0])]
    ]
    dataset = hm.make_dataset(experiments, output_channel_names=CHANNELS)
    model = replace(parameter_model(), simulate_fn=simulate)
    prediction = model.predict(dataset)
    np.testing.assert_array_equal(prediction["x"].outputs["conc"], [3.0, 3.0])
    np.testing.assert_array_equal(prediction["y"].outputs["conc"], [7.0, 7.0])


def test_prediction_compiles_once_per_bucket_shape(fitted, decay_data):
    """Same shapes reuse the trace, and new parameter values do not retrace.

    The dense grid starts after ``t0``; each experiment still integrates
    from its own ``t0``, so every row matches the analytic curve.
    """
    traces = []

    def counting(predictors, ts, covariates, y0, solver):
        traces.append(ts.shape)
        return simulate_parameter(predictors, ts, covariates, y0, solver)

    kwargs: dict[str, Any] = dict(simulate_fn=counting, state_to_output=identity, solver=SOLVER)
    fitted_predictors = fitted[0].model.predictors
    hm.predict_dataset(fitted_predictors, decay_data, **kwargs)
    assert len(traces) == 2, "expected one trace per distinct bucket shape"
    hm.predict_dataset(fitted_predictors, decay_data, **kwargs)
    hm.predict_dataset(parameter_predictors(0.5), decay_data, **kwargs)
    assert len(traces) == 2

    grid = jnp.array([1.0, 2.0])
    dense = hm.predict_dense(fitted_predictors, decay_data, ts_grid=grid, **kwargs)
    compiled = len(traces)
    hm.predict_dense(parameter_predictors(0.5), decay_data, ts_grid=grid, **kwargs)
    assert compiled == 4 and len(traces) == compiled
    expected = _analytic(_rate(fitted[0].model), grid)
    for bucket in dense:
        for row in bucket:
            np.testing.assert_allclose(row[:, 0], expected, rtol=1e-3)


def test_integer_observations_with_a_fractional_t0_are_promoted_to_float():
    """Integer times and states must not truncate a fractional ``t0``."""
    experiment = hm.make_experiment(
        y0=[1], t0=0.5, channels={"conc": hm.ChannelObs([1, 2], [0.8, 0.6])}
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
    metrics = hm.compute_metrics(prediction.buckets, decay_data)
    assert set(metrics) == {"conc"}
    conc = metrics["conc"]
    assert conc.n == _N_OBSERVED
    assert float(conc.rmse) < 0.02 and float(conc.r2) > 0.99

    bad = parameter_model(rate=2.0).predict(decay_data)
    assert float(hm.compute_metrics(bad.buckets, decay_data)["conc"].mse) > float(conc.mse)

    hm.print_metrics(metrics, header="RESULTS")
    out = capsys.readouterr().out
    assert "RESULTS" in out and "conc" in out
    assert "2 bucket(s)" in hm.describe_buckets(decay_data)


def test_metrics_are_jit_safe_and_pool_observations_across_buckets():
    """R^2 uses the variance of all observations, not each bucket's own."""

    def payload(value: float) -> hm.BucketPayload:
        return hm.BucketPayload(
            ts=jnp.array([[0.0]]),
            y_observed=jnp.array([[[value]]]),
            yvar=jnp.ones((1, 1, 1)),
            mask=jnp.ones((1, 1, 1), dtype=bool),
            covariates={},
            y0=jnp.zeros((1, 1)),
            n_obs=jnp.array(1, dtype=jnp.int32),
        )

    dataset = hm.Dataset(
        bucket_payloads=(payload(0.0), payload(10.0)),
        output_channel_names=("x",),
        covariate_names=(),
    )
    zeros = tuple(jnp.zeros_like(bp.y_observed) for bp in dataset.bucket_payloads)
    metrics = jax.jit(hm.compute_metrics)(zeros, dataset)
    assert float(metrics["x"].r2) == pytest.approx(-1.0)
    # Constant observations leave R^2 undefined.
    flat = hm.Dataset(
        bucket_payloads=(payload(2.0), payload(2.0)),
        output_channel_names=("x",),
        covariate_names=(),
    )
    assert np.isnan(float(hm.compute_metrics(zeros, flat)["x"].r2))


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
    """save_run then load_run gives back the same predictions, config and history."""
    result, _ = fitted
    config = quiet(
        phases=(hm.TrainingPhase(25, lr=0.1, horizon=0.5), hm.TrainingPhase(25, lr=0.05))
    )
    run_dir = tmp_path / "does" / "not" / "exist"
    # A first save to the same directory is overwritten by the second.
    hm.save_run(run_dir, predictors=parameter_predictors(), solver=SOLVER, extras={"v": 1})
    hm.save_run(
        run_dir,
        predictors=result.model.predictors,
        solver=SOLVER,
        optax_config=config,
        loss_history=list(result.history.loss),
        extras={"v": 2},
    )
    assert (run_dir / "predictors.eqx").exists()

    def reject_nonfinite(token):
        raise ValueError(f"Non-standard JSON token: {token}")

    # The rate's open upper bound must not leak Infinity into the JSON.
    metadata = json.loads((run_dir / "metadata.json").read_text(), parse_constant=reject_nonfinite)
    assert metadata["loss_history_kind"] == "per_step_data"
    assert [leaf["path"] for leaf in metadata["predictors"]["leaves"]] == [
        "['offset']",
        "['rate']",
    ]

    loaded = hm.load_run(
        run_dir, predictors_template=parameter_predictors(), optax_cls=hm.OptaxTrainingConfig
    )
    assert loaded["optax_config"].phases == config.phases
    assert loaded["solver"].to_dict() == SOLVER.to_dict()
    assert loaded["loss_history"] == list(result.history.loss)
    assert loaded["extras"] == {"v": 2}
    reloaded = replace(result.model, predictors=loaded["predictors"])
    for exp_id, exp in reloaded.predict(decay_data).items():
        np.testing.assert_array_equal(
            exp.outputs["conc"], result.model.predict(decay_data)[exp_id].outputs["conc"]
        )

    raw = hm.load_run(run_dir, predictors_template=parameter_predictors())
    assert raw["optax_config"]["phases"][0]["steps"] == 25


def test_network_predictors_round_trip_bit_exact(tmp_path):
    """save_predictors / load_predictors restore every leaf into a fresh template."""
    predictors = network_predictors(jr.key(0))
    path = str(tmp_path / "net.eqx")
    hm.save_predictors(path, predictors)
    template = network_predictors(jr.key(99))
    restored = hm.load_predictors(path, template)
    leaves = jax.tree_util.tree_leaves(eqx.filter(predictors, eqx.is_inexact_array))
    back = jax.tree_util.tree_leaves(eqx.filter(restored, eqx.is_inexact_array))
    assert len(leaves) == len(back)
    for a, b in zip(leaves, back, strict=True):
        np.testing.assert_array_equal(a, b)


def test_history_kind_says_which_trainer_wrote_it(tmp_path):
    """Optax histories can rise, Evosax ones cannot; the metadata keeps them apart."""
    evosax = hm.EvosaxTrainingConfig(population_size=8, num_generations=2, verbose=False)
    cases: list[tuple[str, dict[str, Any], list[float] | None, str | None]] = [
        ("optax", {"optax_config": quiet(steps=1)}, [1.0], "per_step_data"),
        ("evosax", {"evosax_config": evosax}, [1.0], "best_so_far"),
        ("both", {"optax_config": quiet(steps=1), "evosax_config": evosax}, [1.0], "unknown"),
        ("none", {"optax_config": quiet(steps=1)}, None, None),
    ]
    for name, configs, history, kind in cases:
        hm.save_run(
            tmp_path / name,
            predictors=parameter_predictors(),
            solver=SOLVER,
            loss_history=history,
            **configs,
        )
        assert _metadata(tmp_path / name)["loss_history_kind"] == kind, name
    loaded = hm.load_run(
        tmp_path / "both",
        predictors_template=parameter_predictors(),
        optax_cls=hm.OptaxTrainingConfig,
        evosax_cls=hm.EvosaxTrainingConfig,
    )
    assert loaded["evosax_config"] == evosax


def test_callables_in_a_config_are_described_not_pickled(tmp_path):
    """Loss callables save as a dotted name; other callables refuse to load."""

    def optimizer_factory(learning_rate):
        return optax.adam(learning_rate)

    def penalty_fn(predictors, points):
        return jnp.asarray(0.0)

    hm.save_run(
        tmp_path / "loss",
        predictors=parameter_predictors(),
        solver=SOLVER,
        optax_config=quiet(steps=1, loss=masked_mse),
    )
    assert _metadata(tmp_path / "loss")["optax_config"]["loss"] == (
        "jaxhybridmodels.losses.masked_mse"
    )
    loaded = hm.load_run(
        tmp_path / "loss",
        predictors_template=parameter_predictors(),
        optax_cls=hm.OptaxTrainingConfig,
    )
    assert loaded["optax_config"].loss == "jaxhybridmodels.losses.masked_mse"

    factory = quiet(steps=1, optimizer=optimizer_factory, penalty_fn=penalty_fn)
    hm.save_run(
        tmp_path / "factory", predictors=parameter_predictors(), solver=SOLVER, optax_config=factory
    )
    saved = _metadata(tmp_path / "factory")["optax_config"]
    assert "__callable__" in saved["phases"][0]["optimizer"]
    assert "__callable__" in saved["penalty_fn"]
    with pytest.raises(ValueError, match="callable"):
        hm.load_run(
            tmp_path / "factory",
            predictors_template=parameter_predictors(),
            optax_cls=hm.OptaxTrainingConfig,
        )

    raw = quiet(steps=1, optimizer=optax.adam(1e-3))
    hm.save_run(
        tmp_path / "raw", predictors=parameter_predictors(), solver=SOLVER, optax_config=raw
    )
    assert "__opaque__" in _metadata(tmp_path / "raw")["optax_config"]["phases"][0]["optimizer"]


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
    metadata["optax_config"]["future_field"] = True


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
    ],
)
def test_load_run_rejects_a_mismatched_run(tmp_path, saved, template, edit, match):
    """Loading fails loudly rather than silently mis-reading a different model."""
    hm.save_run(tmp_path, predictors=saved(), solver=SOLVER, optax_config=quiet(steps=1))
    if edit is not None:
        _edit_metadata(tmp_path, edit)
    with pytest.raises(ValueError, match=match):
        hm.load_run(tmp_path, predictors_template=template(), optax_cls=hm.OptaxTrainingConfig)


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
    ui.on_run_start(num_generations=5, population_size=4)
    bests = [0.7771, 0.6661, 0.5551, 0.4441, 0.3331]
    for i, best in enumerate(bests):
        ui.on_generation_end(gen_idx=i, best_fitness=best, mean_fitness=best + 0.1)
    ui.on_run_end(best_fitness=0.2221)
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
    evosax.on_run_start(num_generations=3, population_size=4)
    evosax.on_run_end(best_fitness=0.0)


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


def _duplicate_ids() -> hm.Dataset:
    first = decay_dataset()._experiments[0]
    return hm.make_dataset([first, first], output_channel_names=CHANNELS)


def _other_channels() -> hm.Dataset:
    exp = hm.make_experiment(channels={"other": hm.ChannelObs([1.0], [0.5])}, y0=[1.0], t0=0.0)
    return hm.make_dataset([exp], output_channel_names=("other",))


_FROZEN = replace(
    parameter_model(), predictors={"rate": hm.Constant(0.3), "offset": hm.Constant(0.0)}
)


@pytest.mark.parametrize(
    "call,match",
    [
        pytest.param(
            lambda: _FROZEN.fit(decay_dataset(), quiet(steps=2), key=jr.key(0)),
            "no trainable",
            id="all-constant",
        ),
        pytest.param(
            lambda: parameter_model().fit(
                decay_dataset(),
                hm.EvosaxTrainingConfig(verbose=False),
                validation=decay_dataset(),
                key=jr.key(0),
            ),
            "Optax",
            id="evosax-validation",
        ),
        pytest.param(lambda: parameter_model().predict(_other_channels()), "channels", id="chan"),
        pytest.param(lambda: parameter_model().predict(_duplicate_ids()), "Duplicate", id="dup"),
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
            lambda: replace(parameter_model(), output_channel_names=("a", "a")),
            "unique",
            id="duplicate-channels",
        ),
        pytest.param(
            lambda: hm.HybridModel(
                predictors=parameter_predictors(),
                simulate_fn=simulate_parameter,
                state_to_output=lambda y: jnp.concatenate([y, y], axis=-1),
                output_channel_names=CHANNELS,
                solver=SOLVER,
            ).simulate(ts=[1.0], y0=[1.0], t0=0.0),
            "state_to_output",
            id="output-width",
        ),
    ],
)
def test_model_api_rejects_bad_input(call, match):
    with pytest.raises(ValueError, match=match):
        call()
