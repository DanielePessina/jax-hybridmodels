"""What a run does at its edges, on the small decay models in ``tests/_models.py``.

These train, unlike ``test_numerics.py``, but each pins one behaviour the main
workflows in ``test_workflows.py`` do not reach: a phase boundary, a failed
candidate, a dashboard, a raw optimizer.
"""

from __future__ import annotations

import json
import math
import re
from io import StringIO
from pathlib import Path
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.flatten_util as jfu
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
    parameter_model,
    parameter_predictors,
    simulate_parameter,
)
from jax import Array
from rich.console import Console

import jaxhybridmodels as hm
import jaxhybridmodels.training.evosax as evosax_module
import jaxhybridmodels.training.optax as optax_module
from jaxhybridmodels.data import Dataset
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.penalties import (
    attach_penalty_state,
    penalty_integral,
    penalty_vector_field,
    strip_penalty_state,
    trajectory_saturation_penalty,
)
from jaxhybridmodels.trainable import trainable_mask
from jaxhybridmodels.training import (
    EvosaxTrainingConfig,
    TrainingPhase,
    train_with_evosax,
    train_with_optax,
)
from jaxhybridmodels.training.evosax import (
    _build_single_eval,
)
from jaxhybridmodels.training.optax import _shared_tournament
from jaxhybridmodels.ui import (
    EvosaxUI,
    PlainEvosaxUI,
    PlainTrainingUI,
    RichEvosaxUI,
    RichTrainingUI,
    SilentUI,
    TrainingUI,
)
from jaxhybridmodels.ui.testing import RecordingUI

# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


N_ACC = 2  # saturation accumulator, input-violation accumulator


class Latent(hm.Predictor):
    """One trainable latent ``c``, ignoring its input.

    Restarts keep ``c``, so "training moved c" cannot be satisfied by a
    tournament re-draw alone.
    """

    c: Array

    def __init__(self, c: float) -> None:
        self.c = jnp.asarray(c, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:
        return self.c

    def initialized_with_key(self, key: Array) -> Latent:
        return self


def latent_predictors(c: float) -> tuple[hm.BoundedPredictor]:
    return (
        hm.BoundedPredictor(
            inner=Latent(c), input_keys=("u",), input_bounds=(-3.0, 3.0), output_bounds=(0.0, 1.0)
        ),
    )


def simulate_embedded(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """Decay whose state ``x`` feeds the predictor; two accumulators ride along.

    The accumulators integrate the predictor's output saturation and its
    input violation, so the final state holds both time integrals.
    """
    (p,) = predictors
    rate = jnp.reshape(p({"u": jnp.asarray(1.0)}), ())

    def physics(t, y, args):
        return -rate * y

    def penalty_rates(t, y, args):
        u = y[0]
        z = p.inner(p.in_scaler.to_latent(u))
        return jnp.stack([p.out_scaler.saturation(z), p.in_scaler.input_violation(u)])

    term = diffrax.ODETerm(penalty_vector_field(physics, penalty_rates))
    return solver.diffeqsolve(term, ts, y0).ys


def strip_accumulators(state: Array) -> Array:
    return strip_penalty_state(state, N_ACC)


def accumulated_penalty(full_state: Array, bp: Any) -> Array:
    return jnp.sum(penalty_integral(full_state, N_ACC))


EMBEDDED_KWARGS: dict[str, Any] = dict(
    simulate_fn=simulate_embedded, state_to_output=strip_accumulators, solver=SOLVER
)


def embedded_model(predictors: Any) -> hm.HybridModel:
    """The embedded-penalty model: one observed channel ``x``, accumulators stripped."""
    return hm.HybridModel(predictors=predictors, output_channel_names=("x",), **EMBEDDED_KWARGS)


def embedded_dataset(*, probe: bool = False) -> hm.Dataset:
    """One experiment observed at four times, or a probe with no observations.

    A probe keeps its time grid, so the solver still integrates over it, but
    its mask is all False: the data loss is exactly zero and only a
    trajectory penalty can produce a gradient.
    """
    ts = jnp.array([0.5, 1.0, 1.5, 2.0])
    values = jnp.array([]) if probe else jnp.exp(-0.5 * ts)
    experiment = hm.make_experiment(
        channels={"x": hm.ChannelObs(ts=ts, values=values)},
        y0=attach_penalty_state(jnp.array([1.0]), N_ACC),
        t0=0.0,
        exp_id="probe" if probe else "real",
    )
    return hm.make_dataset([experiment], output_channel_names=("x",))


TRAJECTORY = Objective(trajectory_penalty_fn=accumulated_penalty, trajectory_penalty_weight=1.0)


def test_evosax_ranks_by_the_trajectory_penalty():
    config = EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.5)
    result = train_with_evosax(
        embedded_model(latent_predictors(5.0)),
        embedded_dataset(),
        config,
        objective=TRAJECTORY,
        key=jr.key(0),
        verbose=False,
    )
    assert len(result.history.loss) == 10
    assert float(result.model.predictors[0].inner.c) < 5.0


def test_a_probe_alone_drives_every_bootstrap_member():
    # The probe has no observations, so the penalty is the only gradient.
    members = embedded_model(latent_predictors(5.0)).fit_ensemble(
        embedded_dataset(probe=True),
        hm.OptaxTrainingConfig(steps=10, lr=1e-1, restore_best=False),
        objective=TRAJECTORY,
        n_members=2,
        bootstrap=True,
        key=jr.key(0),
        verbose=False,
    )
    assert len(members) == 2
    for member in members:
        assert np.isfinite(member.final_loss)
        assert float(member.model.predictors[0].inner.c) < 4.5


def test_parallel_saturation_penalty_moves_the_latent_through_the_trainer():
    # Parallel hybrid: the predictor's output is the observed channel, so no
    # accumulators are needed; the hook inverts outputs back to latents.
    out_scaler = latent_predictors(0.0)[0].out_scaler

    def simulate(predictors, ts, covariates, y0, solver):
        out = predictors[0]({"u": jnp.asarray(0.0)})
        return jnp.broadcast_to(out[None], (ts.shape[0], 1))

    def saturation(full_state, bp):
        return trajectory_saturation_penalty(full_state, out_scaler)

    ts = jnp.linspace(0.0, 1.0, 4)
    experiment = hm.make_experiment(
        channels={"y": hm.ChannelObs(ts=ts, values=jnp.full_like(ts, 0.99))}, y0=[0.0], exp_id="e"
    )
    dataset = hm.make_dataset([experiment], output_channel_names=("y",))
    start = latent_predictors(8.0)
    # Anti-vacuity: the penalty is live at the start.
    assert float(saturation(jnp.full((4, 1), start[0]({"u": 0.0})), None)) > 0.0
    model = hm.HybridModel(
        predictors=start,
        simulate_fn=simulate,
        state_to_output=lambda s: s,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    result = train_with_optax(
        model,
        dataset,
        hm.OptaxTrainingConfig(steps=20, lr=1e-1, restore_best=False),
        objective=Objective(trajectory_penalty_fn=saturation, trajectory_penalty_weight=1.0),
        key=jr.key(0),
        verbose=False,
    )
    assert float(result.model.predictors[0].inner.c) < 8.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def data_loss(model: hm.HybridModel, dataset: Dataset) -> float:
    """The trainer's data term: per-bucket ``masked_mse``, averaged over buckets."""
    buckets = model.predict(dataset).buckets
    total = sum(
        float(masked_mse(pred, bp))
        for pred, bp in zip(buckets, dataset.bucket_payloads, strict=True)
    )
    return total / len(dataset.bucket_payloads)


def rate_of(model: hm.HybridModel) -> float:
    return float(model.predictors["rate"].value)


def nan_simulate(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """A model whose every evaluation is non-finite, so every candidate fails."""
    return jnp.full((ts.shape[0], 1), jnp.nan)


def test_patience_stops_early_and_weak_scalars_train(decay_data, capsys):
    """Patience cuts a flat run short, and a weak-typed leaf survives the settle path.

    ``jnp.asarray(1.0)`` is weak-typed, and JAX keys its compile cache on
    weak type, so the first update would retrace every kernel. The warm-up
    settle in ``training/optax.py`` absorbs that; the run must finish with
    finite losses. With ``lr=0`` nothing improves after step 0, so
    ``patience=3`` stops the run four steps in. No UI and ``verbose=False``
    must print nothing.
    """

    class WeakScale(eqx.Module):
        scale: Array

    predictors = {**parameter_predictors(), "weak": WeakScale(jnp.asarray(1.0))}
    assert jax.typeof(predictors["weak"].scale).weak_type
    result = train_with_optax(
        make_model(predictors),
        decay_data,
        hm.OptaxTrainingConfig(steps=20, lr=0.0, patience=3),
        key=jr.key(0),
        verbose=False,
    )
    assert len(result.history.loss) == 4
    assert result.stop_reason == "patience"
    assert all(math.isfinite(v) for v in result.history.loss)
    assert capsys.readouterr().out == ""


@pytest.fixture(scope="module")
def curriculum(temp_data):
    """A short-horizon phase then a full one, against data one rate cannot fit.

    The rate in ``temp_data`` depends on temperature, so a single
    ``Parameter`` has a loss floor over the full window. Over the first
    observation every curve looks alike and the loss is far lower.
    """
    short = TrainingPhase(10, lr=0.1, horizon=0.2)
    both = parameter_model().fit(
        temp_data,
        hm.OptaxTrainingConfig(phases=(short, TrainingPhase(15, lr=0.05))),
        key=jr.key(0),
        verbose=False,
    )
    short_only = parameter_model().fit(
        temp_data, hm.OptaxTrainingConfig(phases=(short,)), key=jr.key(0), verbose=False
    )
    return both, short_only


def test_raw_optimizer_after_a_reset_and_refused_lr_change(decay_data):
    """A raw ``GradientTransformation`` cannot take a new learning rate.

    After ``reset_optimizer`` it starts fresh and is fine. Reusing the same
    raw object across a learning-rate change is refused when the config is
    built, before any phase has trained.
    """
    raw = optax.adam(1e-2)
    result = parameter_model().fit(
        decay_data,
        hm.OptaxTrainingConfig(
            phases=(
                TrainingPhase(3, lr=0.05),
                TrainingPhase(3, lr=1e-2, optimizer=raw, reset_optimizer=True),
            )
        ),
        key=jr.key(0),
        verbose=False,
    )
    assert len(result.history.loss) == 6 and rate_of(result.model) != 0.3

    with pytest.raises(ValueError, match="learning rate cannot be changed"):
        hm.OptaxTrainingConfig(
            phases=(
                TrainingPhase(2, lr=1e-3, optimizer=raw),
                TrainingPhase(2, lr=1e-2, optimizer=raw),
            )
        )


def test_tournament_falls_back_when_every_attempt_fails(decay_data):
    # The fallback keeps the input model, whose own first step then fails too.
    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        with pytest.raises(FloatingPointError, match="non-finite"):
            make_model(parameter_predictors(), nan_simulate).fit(
                decay_data,
                hm.OptaxTrainingConfig(steps=1, tournament_attempts=2, tournament_steps=1),
                key=jr.key(0),
                verbose=False,
            )


def test_a_run_whose_every_attempt_fails_still_summarises(decay_data):
    """Only the starting rate solves, so every redrawn attempt fails and the run
    falls back to the start, trains at zero rate and must still summarise."""

    def only_the_start(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any):
        ys = simulate_parameter(predictors, ts, covariates, y0, solver)
        return jnp.where(jnp.abs(predictors["rate"].value - 0.3) < 1e-4, ys, jnp.nan)

    model = make_model(
        {"rate": hm.Parameter(0.3, bounds=(0.0, 1.0)), "offset": hm.Constant(0.0)}, only_the_start
    )
    config = hm.OptaxTrainingConfig(steps=3, lr=0.0, tournament_attempts=3, tournament_steps=1)
    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        result = model.fit(decay_data, config, key=jr.key(0), verbose=False)
    assert "all 3 attempts failed" in result.summary()


def test_tournament_survives_equinox_runtime_failures(decay_data):
    # A solver hitting max_steps raises EquinoxRuntimeError at runtime. The
    # tournament must treat it as a failed attempt, not crash the run.
    predictors = parameter_predictors()

    def score_bucket(_predictors, _bp, _fraction):
        raise eqx.EquinoxRuntimeError("solver failed")

    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        ranked, scores = _shared_tournament(
            predictors,
            decay_data,
            bucket_step=lambda *_args: (jnp.asarray(0.0), predictors),
            score_bucket=score_bucket,
            apply_update=lambda *args: args[:2],
            optimizer=optax.sgd(1e-3),
            trainable=trainable_mask(predictors),
            tournament_attempts=2,
            tournament_steps=0,
            tournament_lr=1e-3,
            key=jr.key(0),
            horizon=jnp.asarray(1.0),
            top_k=1,
        )
    assert ranked[0][0] == float("inf")
    assert scores == (float("inf"), float("inf"))


class BracketingUI(RecordingUI):
    """Records events and flags any that fall outside an open run bracket.

    Every ensemble member is its own ``on_run_start``/``on_run_end`` pair. A
    straddling or unclosed bracket freezes a live Rich dashboard.
    """

    def __init__(self) -> None:
        super().__init__()
        self.open = False
        self.invalid: list[str] = []

    def _inside(self, name: str) -> None:
        if not self.open:
            self.invalid.append(name)

    def on_run_start(self, **kwargs: Any) -> None:
        if self.open:
            self.invalid.append("nested on_run_start")
        self.open = True
        super().on_run_start(**kwargs)

    def on_compile_start(self, **kwargs: Any) -> None:
        self._inside("on_compile_start")
        super().on_compile_start(**kwargs)

    def on_phase_start(self, **kwargs: Any) -> None:
        self._inside("on_phase_start")
        super().on_phase_start(**kwargs)

    def on_step_end(self, **kwargs: Any) -> None:
        self._inside("on_step_end")
        super().on_step_end(**kwargs)

    def on_run_end(self, **kwargs: Any) -> None:
        self._inside("on_run_end")
        self.open = False
        super().on_run_end(**kwargs)

    def count(self, name: str) -> int:
        return sum(event == name for event, _ in self.events)


def test_failed_seed_ensemble_raises_and_closes_the_bracket(decay_data):
    ui = BracketingUI()
    with pytest.raises(RuntimeError, match="every seed failed"):
        make_model(parameter_predictors(), nan_simulate).fit_ensemble(
            decay_data,
            hm.OptaxTrainingConfig(steps=2, tournament_steps=1),
            key=jr.key(0),
            n_members=1,
            n_seeds=2,
            ui=ui,
        )
    assert ui.invalid == []
    assert ui.count("on_run_start") == ui.count("on_run_end") == 1


def failing_after_the_shape_check() -> Any:
    """A ``simulate_fn`` that passes the shape check and fails once training traces it."""
    calls = [0]

    def simulate(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
        calls[0] += 1
        if calls[0] > 1:
            raise RuntimeError("simulate_fn failed")
        return simulate_parameter(predictors, ts, covariates, y0, solver)

    return simulate


def train_with(trainer: str, model: hm.HybridModel, data: Dataset, **kwargs: Any) -> Any:
    """Run one of the four training entry points with two cheap steps."""
    if trainer == "optax":
        return train_with_optax(
            model, data, hm.OptaxTrainingConfig(steps=2), **kwargs, verbose=False
        )
    if trainer == "seed":
        return model.fit_ensemble(
            data, hm.OptaxTrainingConfig(steps=2), n_members=2, **kwargs, verbose=False
        )
    if trainer == "bootstrap":
        return model.fit_ensemble(
            data,
            hm.OptaxTrainingConfig(steps=2),
            n_members=2,
            bootstrap=True,
            **kwargs,
            verbose=False,
        )
    config = EvosaxTrainingConfig(population_size=4, steps=2)
    return train_with_evosax(model, data, config, **kwargs, verbose=False)


@pytest.mark.parametrize("trainer", ["optax", "seed", "bootstrap", "evosax"])
def test_bad_objective_raises_before_the_ui_opens(decay_data, trainer):
    # A Rich Live dashboard opened by on_run_start would be left on screen
    # if the objective check raised after it.
    ui = BracketingUI()
    bad = Objective(channel_weights={"missing": 1.0})
    with pytest.raises(ValueError, match="unknown channels"):
        train_with(trainer, parameter_model(), decay_data, objective=bad, key=jr.key(0), ui=ui)
    assert ui.events == []


@pytest.mark.parametrize("trainer", ["optax", "seed", "evosax"])
def test_error_after_run_start_closes_the_ui(decay_data, trainer):
    ui = BracketingUI()
    model = make_model(parameter_predictors(), failing_after_the_shape_check())
    with pytest.raises(RuntimeError, match="simulate_fn failed"):
        train_with(trainer, model, decay_data, key=jr.key(0), ui=ui)
    assert ui.count("on_run_start") == ui.count("on_run_end") == 1
    assert not ui.open
    assert math.isnan(ui.events[-1][1]["final_loss"])


def test_predict_on_explicit_and_even_grids(decay_data):
    """``ts`` and ``n_points`` change the grid; each experiment still starts at its ``t0``."""
    model = parameter_model(TRUE_RATE)
    measured = model.predict(decay_data)
    times = measured["b"].times
    on_ts = model.predict(decay_data, ts=times)
    for exp_id in measured:
        np.testing.assert_allclose(on_ts[exp_id].times, times)
        assert on_ts[exp_id].observed == {} and on_ts[exp_id].observation_mask == {}
    np.testing.assert_allclose(on_ts["b"].outputs["conc"], measured["b"].outputs["conc"], atol=1e-4)

    even = model.predict(list(decay_data.experiments), n_points=9)
    for exp_id, experiment in even.items():
        start, end = measured[exp_id].times[0], measured[exp_id].times[-1]
        np.testing.assert_allclose(experiment.times, jnp.linspace(start, end, 9), rtol=1e-6)
        np.testing.assert_allclose(
            experiment.outputs["conc"], jnp.exp(-TRUE_RATE * experiment.times), rtol=1e-3
        )
    with pytest.raises(ValueError, match="measured grid"):
        hm.compute_metrics(even)

    # simulate takes n_points for the same even grid over one experiment.
    first = decay_data.experiments[0]
    simulated = model.simulate(first, n_points=9)
    np.testing.assert_allclose(simulated.times, even[first.exp_id].times, rtol=1e-6)
    with pytest.raises(ValueError, match="not both"):
        model.simulate(first, ts=times, n_points=9)
    with pytest.raises(ValueError, match="needs experiment="):
        model.simulate(n_points=9, y0=[1.0])


# A no-ODE target for the cheap evosax checks: the "simulation" is the
# parameter vector itself, observed once.
THETA_STAR = jnp.array([1.0, -2.0, 3.0, -4.0])


QUADRATIC_CHANNELS = ("c0", "c1", "c2", "c3")


def simulate_quadratic(
    predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any
) -> Array:
    return jnp.broadcast_to(predictors.value, (ts.shape[0], 4))


def quadratic_model(simulate_fn: Any = simulate_quadratic) -> hm.HybridModel:
    return make_model(hm.Parameter(jnp.zeros(4)), simulate_fn, QUADRATIC_CHANNELS)


@pytest.fixture(scope="module")
def quadratic_data():
    ts = jnp.array([0.0])
    channels = {
        name: hm.ChannelObs(ts=ts, values=THETA_STAR[i : i + 1])
        for i, name in enumerate(QUADRATIC_CHANNELS)
    }
    experiment = hm.make_experiment(channels=channels, y0=jnp.zeros(4), exp_id="q")
    return hm.make_dataset([experiment], output_channel_names=QUADRATIC_CHANNELS)


def test_multi_bucket_fitness_averages_like_optax(quadratic_data):
    # Optax averages bucket losses with equal weight per bucket. Evosax must
    # use the same average, or the two trainers optimise different objectives.
    predictor = hm.Parameter(jnp.zeros(4))
    params, static = eqx.partition(predictor, trainable_mask(predictor))
    flat, unflatten = jfu.ravel_pytree(params)
    bp = quadratic_data.bucket_payloads[0]
    small = bp._replace(n_obs=jnp.asarray(1, dtype=bp.n_obs.dtype))
    large = bp._replace(n_obs=jnp.asarray(3, dtype=bp.n_obs.dtype))
    evaluate = _build_single_eval(
        static_predictor=static,
        unflatten=unflatten,
        bucket_payloads=(small, large),
        loss_fn=lambda _pred, bucket: bucket.n_obs.astype(jnp.float32),
        penalty_fn=lambda _p, _points: jnp.asarray(0.0),
        penalty_weight=0.0,
        simulate_fn=simulate_quadratic,
        state_to_output=identity,
        solver=SOLVER,
    )
    assert float(evaluate(flat)) == pytest.approx(2.0)


def test_channel_weights_by_name_fit_only_the_named_channels(quadratic_data):
    """A by-name mapping scores only the channels it names.

    The unnamed channels get no gradient, and adamw's decoupled weight decay
    leaves a zero leaf at exactly zero, so they stay put.
    """
    result = train_with_optax(
        quadratic_model(),
        quadratic_data,
        hm.OptaxTrainingConfig(steps=60, lr=0.2),
        objective=Objective(channel_weights={"c0": 1.0, "c1": 2.0}),
        key=jr.key(0),
        verbose=False,
    )
    value = result.model.predictors.value
    assert float(jnp.abs(value[:2] - THETA_STAR[:2]).max()) < 0.2
    assert value[2:].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# The model API and persistence
# ---------------------------------------------------------------------------


# A number printed to at least four decimals, the precision the UIs promise.
_SIG4 = re.compile(r"\d\.\d{4,}")


def _console() -> Console:
    """A non-TTY recording console: ``Live`` prints once, at ``stop()``."""
    return Console(record=True, force_terminal=False, width=120, file=StringIO())


def _rate(model: hm.HybridModel) -> float:
    return float(model.predictors["rate"].value)


@pytest.fixture(scope="module")
def validation_fit(decay_data):
    """Validation data generated at the starting rate, so training only hurts it."""
    validation = decay_dataset(lambda temp: 0.3)
    config = hm.OptaxTrainingConfig(steps=20, lr=0.03, patience=2)
    return parameter_model().fit(
        decay_data, config, validation=validation, key=jr.key(2), verbose=False
    )


def test_stop_reason_follows_the_final_phase(decay_data, monkeypatch):
    """Patience in an early phase does not make the run's stop reason "patience".

    Also covers ``verbose=True``: with no ``ui`` the trainer builds a
    ``RichTrainingUI``, here wired to a recording console.
    """
    console = _console()
    monkeypatch.setattr(optax_module, "RichTrainingUI", lambda: RichTrainingUI(console=console))
    monkeypatch.setattr(optax_module, "live_display_available", lambda: True)
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
        patch.setattr(evosax_module, "live_display_available", lambda: True)
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


def test_piped_output_gets_plain_lines(decay_data, capsys, monkeypatch):
    """No redrawing display is possible, so progress prints as greppable lines."""
    monkeypatch.setattr(optax_module, "live_display_available", lambda: False)
    monkeypatch.setattr(evosax_module, "live_display_available", lambda: False)

    parameter_model().fit(decay_data, hm.OptaxTrainingConfig(steps=20, lr=0.01), key=jr.key(0))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "training: 20 steps in 1 phase"
    assert lines[1].startswith("compiling")
    assert sum(line.startswith("step ") for line in lines) == 10
    assert any(line.startswith("step 20/20") for line in lines)
    assert lines[-1].startswith("done: final loss")

    config = hm.EvosaxTrainingConfig(population_size=4, steps=3, sigma_init=0.1)
    parameter_model().fit(decay_data, config, key=jr.key(0))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "searching: 3 generations of 4 candidates"
    assert sum(line.startswith("generation ") for line in lines) == 3

    parameter_model().fit(
        decay_data, hm.OptaxTrainingConfig(steps=2, lr=0.01), key=jr.key(0), verbose=False
    )
    assert capsys.readouterr().out == ""


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


def test_lr_schedule_counts_from_the_phase_start(decay_data):
    """A phase's schedule sees step 0 at the phase start, not the run start.

    Phase 1 runs five steps at zero. Phase 2's schedule is zero for its own
    first three steps and then trains. Each history entry is measured before
    that step's update, so the loss stays flat through step 8 (the fourth
    step of phase 2) and moves at step 9. Counting from the run start would
    already train at step 5.
    """
    held = optax.join_schedules([optax.constant_schedule(0.0), optax.constant_schedule(0.1)], [3])
    result = parameter_model().fit(
        decay_data,
        hm.OptaxTrainingConfig(
            phases=(TrainingPhase(5, lr=0.0), TrainingPhase(6, lr=held)), restore_best=False
        ),
        key=jr.key(0),
        verbose=False,
    )
    history = result.history.loss
    assert len(set(history[:9])) == 1
    assert history[9] != history[8]
    assert rate_of(result.model) != 0.3

    # The same schedule in NumPy cannot take a traced step, so it runs on the host.
    def host(step: Any) -> float:
        return float(np.where(step < 3, 0.0, 0.1))

    again = parameter_model().fit(
        decay_data,
        hm.OptaxTrainingConfig(
            phases=(TrainingPhase(5, lr=0.0), TrainingPhase(6, lr=host)), restore_best=False
        ),
        key=jr.key(0),
        verbose=False,
    )
    assert again.history.loss == pytest.approx(history)


def test_channel_selection_composes_with_a_plain_custom_loss(quadratic_data):
    """Unit weights by name become a pure selection, so a ``(pred_obs, bp)`` loss works.

    Non-unit weights cannot be applied inside a custom reduction and raise.
    """

    def plain(pred_obs, bp):
        return jnp.sum(jnp.where(bp.mask, (pred_obs - bp.y_observed) ** 2, 0.0))

    names = quadratic_data.output_channel_names
    bp = quadratic_data.bucket_payloads[0]
    pred = jnp.zeros_like(bp.y_observed)
    selected = Objective(loss=plain, channel_weights={"c2": 1.0}).loss_fn(names)
    assert float(selected(pred, bp)) == pytest.approx(float(THETA_STAR[2] ** 2))
    with pytest.raises(ValueError, match="channel_weights cannot be applied"):
        Objective(loss=plain, channel_weights={"c2": 2.0}).loss_fn(names)
    # Positional weights keep every channel; unit weights are the plain loss.
    assert Objective(loss=plain, channel_weights=(1.0,) * 4).loss_fn(names) is plain


def test_integer_observations_with_a_fractional_t0_are_promoted_to_float():
    """Integer times and states must not truncate a fractional ``t0``."""
    experiment = hm.make_experiment(
        y0=[1], t0=0.5, channels={"conc": hm.ChannelObs(ts=[1, 2], values=[0.8, 0.6])}
    )
    dataset = hm.make_dataset([experiment], output_channel_names=CHANNELS)
    assert jnp.issubdtype(experiment.y0.dtype, jnp.floating)
    np.testing.assert_allclose(dataset.bucket_payloads[0].ts[0], [0.5, 1.0, 2.0])


def _metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metadata.json").read_text())


def test_an_evosax_config_reloads_as_its_own_type(tmp_path):
    """The config's kind is recorded, so an Evosax config comes back as one."""
    evosax = hm.EvosaxTrainingConfig(population_size=8, steps=2)
    hm.save(tmp_path, parameter_model(), config=evosax)
    assert _metadata(tmp_path)["config"]["kind"] == "evosax"
    assert hm.load(tmp_path, like=parameter_model()).config == evosax


@pytest.mark.parametrize(
    "ui",
    [RichTrainingUI(console=_console()), PlainTrainingUI(), SilentUI(), RecordingUI()],
    ids=type,
)
def test_training_uis_satisfy_the_protocol(ui):
    assert isinstance(ui, TrainingUI)


@pytest.mark.parametrize(
    "ui", [RichEvosaxUI(console=_console()), PlainEvosaxUI(), SilentUI(), RecordingUI()], ids=type
)
def test_evosax_uis_satisfy_the_protocol(ui):
    assert isinstance(ui, EvosaxUI)
