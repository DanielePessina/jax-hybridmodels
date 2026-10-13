"""The trainers and ensembles, end to end on the decay model.

``train_with_optax`` and ``train_with_evosax`` take a ``HybridModel`` and
the keywords of ``fit``; ``fit`` dispatches to them. Every trainer call
compiles its kernels from scratch, which costs a couple of seconds, so runs
are shared through module fixtures and each test reads one property off a
run. Config validation lives in ``test_config.py`` and the bound and
trajectory penalties in ``test_penalties.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Literal, cast

import diffrax
import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import pytest
from _models import (
    SOLVER,
    TRUE_RATE,
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

import jaxhybridmodels as hm
from jaxhybridmodels.data import Dataset, make_bootstrap_dataset
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.predictors import BoundScaler, MLPPredictor
from jaxhybridmodels.solver import ADJOINT_REGISTRY
from jaxhybridmodels.trainable import count_trainable_params, freeze_mask, trainable_mask
from jaxhybridmodels.training import TrainingPhase, train_with_evosax, train_with_optax
from jaxhybridmodels.training.evosax import (
    EvosaxTrainingConfig,
    _box_population,
    _build_single_eval,
    _build_strategy,
)
from jaxhybridmodels.training.optax import _rank_key, _shared_tournament
from jaxhybridmodels.ui.testing import RecordingUI


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


# ---------------------------------------------------------------------------
# Optax: convergence, phases and the UI lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phased(decay_data):
    """Two phases, short horizon then full, with ``offset`` frozen by path.

    ``simulate_fn`` counts its traces, so the run also reports how often the
    training kernels were compiled.
    """
    traces = [0]

    def counted(predictors, ts, covariates, y0, solver):
        traces[0] += 1
        return simulate_parameter(predictors, ts, covariates, y0, solver)

    predictors = {"rate": hm.Parameter(0.3, bounds=(0.0, None)), "offset": hm.Parameter(0.0)}
    ui = RecordingUI()
    result = train_with_optax(
        make_model(predictors, counted),
        decay_data,
        quiet(phases=(TrainingPhase(20, lr=0.1, horizon=0.5), TrainingPhase(20, lr=0.05))),
        freeze="offset.latent",
        key=jr.key(0),
        ui=ui,
    )
    return result, ui, traces[0]


def test_two_phase_fit_recovers_the_rate(phased):
    result, _, _ = phased
    loss = result.history.loss
    assert len(loss) == 40
    assert rate_of(result.model) == pytest.approx(TRUE_RATE, abs=0.02)
    assert loss[-1] < loss[0]
    assert result.history.phase == (0,) * 20 + (1,) * 20


def test_a_horizon_change_does_not_recompile(phased):
    # The horizon is a traced fraction, not a static shape, so both phases
    # share one compiled kernel per bucket. The decay data has one bucket
    # per union length, plus one abstract trace from the shape check before
    # training; anything above that count is a retrace.
    _, _, traces = phased
    assert traces <= 3


def test_frozen_leaf_does_not_move_through_optax(phased):
    # adamw decays every leaf it is given, gradient or not, so a frozen
    # leaf that leaked into the optimiser would drift away from zero.
    result, _, _ = phased
    assert float(result.model.predictors["offset"].value) == 0.0


def test_ui_events_bracket_the_run_in_order(phased):
    _, ui, _ = phased
    names = [name for name, _ in ui.events]
    assert names[0] == "on_run_start"
    assert names[-1] == "on_run_end"
    assert names.count("on_step_end") == 40
    assert names.count("on_phase_start") == names.count("on_phase_end") == 2
    assert names.index("on_compile_done") < names.index("on_phase_start")


def test_train_with_optax_is_what_fit_runs(decay_data):
    """``fit`` dispatches to ``train_with_optax``, which takes a list of experiments too."""
    config = quiet(steps=5, lr=0.1)
    model = parameter_model()
    via_fit = model.fit(decay_data, config, key=jr.key(3))
    direct = train_with_optax(model, list(decay_data.experiments), config, key=jr.key(3))
    assert direct.history == via_fit.history
    assert rate_of(direct.model) == rate_of(via_fit.model)
    assert rate_of(model) == pytest.approx(0.3), "the input model was changed"
    with pytest.raises(TypeError, match="OptaxTrainingConfig"):
        train_with_optax(
            model,
            decay_data,
            EvosaxTrainingConfig(population_size=4, steps=1),  # ty: ignore[invalid-argument-type]
            key=jr.key(0),
        )


def test_missing_key_raises(decay_data):
    with pytest.raises(TypeError):
        train_with_optax(parameter_model(), decay_data, quiet(steps=1))  # ty: ignore[missing-argument]


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
        make_model(predictors), decay_data, quiet(steps=20, lr=0.0, patience=3), key=jr.key(0)
    )
    assert len(result.history.loss) == 4
    assert result.stop_reason == "patience"
    assert all(math.isfinite(v) for v in result.history.loss)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# restore_best
# ---------------------------------------------------------------------------

# Phase 1 overshoots at a high learning rate, so its best step is mid-phase.
# Phase 2 runs at zero and cannot improve on anything. Both share a horizon.
OVERSHOOT_PHASES = (TrainingPhase(15, lr=0.6), TrainingPhase(3, lr=0.0))


@pytest.fixture(scope="module")
def overshoot(decay_data):
    return {
        restore: parameter_model().fit(
            decay_data, quiet(phases=OVERSHOOT_PHASES, restore_best=restore), key=jr.key(0)
        )
        for restore in (True, False)
    }


def test_restore_best_returns_the_pre_update_snapshot(overshoot, decay_data):
    """The returned model scores exactly the history minimum.

    Each step's loss is measured before that step's update. Snapshotting
    after the update would return a model whose loss is not the reported
    minimum.
    """
    best = overshoot[True]
    history = best.history.loss
    argmin = min(range(len(history)), key=history.__getitem__)
    assert argmin < OVERSHOOT_PHASES[0].steps - 1, "phase 1 was meant to overshoot"
    assert best.best_step == argmin and best.final_loss == min(history)
    assert data_loss(best.model, decay_data) == pytest.approx(min(history), rel=1e-4)


def test_restore_best_false_returns_the_final_step(overshoot, decay_data):
    # restore_best changes what is handed back, not how training runs.
    best, last = overshoot[True], overshoot[False]
    assert best.history.loss == last.history.loss
    assert last.best_step is None and last.final_loss == last.history.loss[-1]
    assert rate_of(last.model) != rate_of(best.model)
    # Same horizon in both phases, so the running minimum carries across
    # the boundary and the mid-phase-1 best beats the endpoint.
    assert data_loss(best.model, decay_data) < data_loss(last.model, decay_data)


@pytest.fixture(scope="module")
def curriculum(temp_data):
    """A short-horizon phase then a full one, against data one rate cannot fit.

    The rate in ``temp_data`` depends on temperature, so a single
    ``Parameter`` has a loss floor over the full window. Over the first
    observation every curve looks alike and the loss is far lower.
    """
    short = TrainingPhase(10, lr=0.1, horizon=0.2)
    both = parameter_model().fit(
        temp_data, quiet(phases=(short, TrainingPhase(15, lr=0.05))), key=jr.key(0)
    )
    short_only = parameter_model().fit(temp_data, quiet(phases=(short,)), key=jr.key(0))
    return both, short_only


def test_restore_best_does_not_compare_across_horizons(curriculum, temp_data):
    """The best step is chosen within the final horizon only.

    Losses over different horizons measure different things. One running
    minimum across both phases lands in the short phase, handing back the
    least-trained model of the run.
    """
    both, short_only = curriculum
    history = both.history.loss
    # Anti-vacuity: the global minimum of the history is in the short phase.
    assert min(range(len(history)), key=history.__getitem__) < 10
    assert data_loss(both.model, temp_data) < data_loss(short_only.model, temp_data)


# ---------------------------------------------------------------------------
# Optimisers, losses and predictor pytrees
# ---------------------------------------------------------------------------


def test_factory_optimizer_custom_loss_and_network_dict(temp_data):
    """An optimiser factory, a custom loss callable, and a network in a dict.

    The factory receives each phase's learning rate. The custom loss
    replaces ``"mse"`` everywhere the trainer scores a bucket.
    """
    calls = []

    def loss(pred_obs, bp):
        calls.append(1)
        return jnp.sum(jnp.where(bp.mask, (pred_obs - bp.y_observed) ** 2, 0.0))

    result = train_with_optax(
        network_model(),
        temp_data,
        quiet(
            steps=30,
            lr=0.05,
            optimizer=lambda learning_rate: optax.chain(
                optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
            ),
        ),
        objective=Objective(loss=loss),
        key=jr.key(0),
    )
    assert calls, "the custom loss was never traced"
    assert result.history.loss[-1] < 0.5 * result.history.loss[0]
    assert set(result.model.predictors) == {"rate", "offset"}


def test_raw_optimizer_after_a_reset_and_refused_lr_change(decay_data):
    """A raw ``GradientTransformation`` cannot take a new learning rate.

    After ``reset_optimizer`` it starts fresh and is fine. Reusing the same
    raw object across a learning-rate change passes the config check (the
    specs are equal) and must be refused when the phase begins.
    """
    raw = optax.adam(1e-2)
    result = parameter_model().fit(
        decay_data,
        quiet(
            phases=(
                TrainingPhase(3, lr=0.05),
                TrainingPhase(3, lr=1e-2, optimizer=raw, reset_optimizer=True),
            )
        ),
        key=jr.key(0),
    )
    assert len(result.history.loss) == 6 and rate_of(result.model) != 0.3

    refused = quiet(
        phases=(TrainingPhase(2, lr=1e-3, optimizer=raw), TrainingPhase(2, lr=1e-2, optimizer=raw))
    )
    with pytest.raises(ValueError, match="cannot be re-hyperparametrised"):
        parameter_model().fit(decay_data, refused, key=jr.key(0))


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
        quiet(phases=(TrainingPhase(5, lr=0.0), TrainingPhase(6, lr=held)), restore_best=False),
        key=jr.key(0),
    )
    history = result.history.loss
    assert len(set(history[:9])) == 1
    assert history[9] != history[8]
    assert rate_of(result.model) != 0.3


def test_lr_schedule_on_the_single_phase_shorthand_and_after_a_float_phase(decay_data):
    """``steps=, lr=schedule`` trains, and a float phase after it resets the lr.

    A schedule that decays to zero halfway leaves the second half flat; the
    float phase that follows must move the parameter again, so the schedule's
    last value cannot leak past its phase.
    """
    to_zero = optax.linear_schedule(0.1, 0.0, 4)
    history = (
        parameter_model()
        .fit(decay_data, quiet(steps=8, lr=to_zero, restore_best=False), key=jr.key(0))
        .history.loss
    )
    assert history[1] != history[0] and len(set(history[5:])) == 1

    history = (
        parameter_model()
        .fit(
            decay_data,
            quiet(phases=(TrainingPhase(6, lr=to_zero), TrainingPhase(3, lr=0.05))),
            key=jr.key(0),
        )
        .history.loss
    )
    assert history[5] == history[6] and history[8] != history[7]


def test_validation_set_drives_selection(decay_data, temp_data):
    """``validation=`` scores every step with the training objective.

    Validating on data the model is not trained on means the selection
    series differs from the training loss, and the returned step is the
    validation minimum. Validation may be a list of experiments.
    """
    result = train_with_optax(
        parameter_model(),
        decay_data,
        quiet(steps=12, lr=0.1),
        key=jr.key(0),
        validation=list(temp_data.experiments),
    )
    history = result.history
    assert len(history.validation_loss) == len(history.loss) == 12
    assert history.validation_loss != history.loss
    assert result.best_step == int(np.argmin(history.validation_loss))
    assert data_loss(result.model, temp_data) == pytest.approx(
        min(history.validation_loss), rel=1e-4
    )

    mismatched = hm.make_dataset(
        [
            hm.make_experiment(
                channels={"other": hm.ChannelObs(ts=jnp.array([1.0]), values=jnp.array([0.5]))},
                y0=[1.0],
                exp_id="x",
            )
        ],
        output_channel_names=("other",),
    )
    with pytest.raises(ValueError, match="validation: dataset channels"):
        train_with_optax(
            parameter_model(), decay_data, quiet(steps=1), key=jr.key(0), validation=mismatched
        )


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


def tournament_config(attempts: int):
    # One main step at zero learning rate, so the returned model is the
    # tournament's pick and nothing else.
    return quiet(
        steps=1,
        lr=0.0,
        tournament_attempts=attempts,
        tournament_steps=3,
        tournament_lr=0.05,
        restore_best=False,
    )


@pytest.fixture(scope="module")
def tournament_losses(decay_data):
    """Winner losses for pools of two and three attempts under one key.

    Key 12 is chosen because its third attempt beats the first two, so the
    larger pool has something to find. Under most keys the best candidate
    is already among the first two and the comparison would be flat.
    """
    return {
        attempts: data_loss(
            parameter_model().fit(decay_data, tournament_config(attempts), key=jr.key(12)).model,
            decay_data,
        )
        for attempts in (2, 3)
    }


def test_more_attempts_give_a_better_winner(tournament_losses):
    """The tournament returns the best candidate, not the first survivor.

    Attempt keys are folded from the attempt index, so the pool of three
    contains the pool of two. Under first-survivor-wins both runs would
    return attempt 0 and score the same.
    """
    assert tournament_losses[3] < tournament_losses[2] - 1e-9, tournament_losses


def test_tournament_falls_back_when_every_attempt_fails(decay_data):
    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        make_model(parameter_predictors(), nan_simulate).fit(
            decay_data, quiet(steps=1, tournament_attempts=2, tournament_steps=1), key=jr.key(0)
        )


def test_tournament_survives_equinox_runtime_failures(decay_data):
    # A solver hitting max_steps raises EquinoxRuntimeError at runtime. The
    # tournament must treat it as a failed attempt, not crash the run.
    predictors = parameter_predictors()

    def score_bucket(_predictors, _bp, _fraction):
        raise eqx.EquinoxRuntimeError("solver failed")

    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        ranked = _shared_tournament(
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
            length_mask_fraction=jnp.asarray(1.0),
            top_k=1,
        )
    assert ranked[0][0] == float("inf")


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


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


SEED_CONFIG = quiet(steps=8, lr=0.05, tournament_steps=2)


@pytest.fixture(scope="module")
def seed_ensembles(decay_data):
    """Under one key: the best two of three seeds, and all three (``n_seeds`` defaulted)."""
    ui = BracketingUI()
    top_two = parameter_model().fit_ensemble(
        decay_data, SEED_CONFIG, key=jr.key(0), n_members=2, n_seeds=3, ui=ui
    )
    all_three = parameter_model().fit_ensemble(
        list(decay_data.experiments), SEED_CONFIG, key=jr.key(0), n_members=3
    )
    return top_two, all_three, ui


def test_seed_ensemble_ranks_the_best_members(seed_ensembles, decay_data):
    top_two, all_three, _ = seed_ensembles
    assert len(top_two) == 2 and len(all_three) == 3
    assert all(isinstance(r, hm.FitResult) for r in top_two)
    losses = [r.final_loss for r in top_two]
    assert losses == sorted(losses)
    assert losses[0] < data_loss(parameter_model(), decay_data)
    assert all(len(r.history.loss) == SEED_CONFIG.total_steps for r in top_two)


def test_seed_ensemble_is_deterministic_in_the_key(seed_ensembles):
    # Members are seeded by rank, so the top two of a three-member run with
    # the same key are the same models as a two-member run.
    top_two, all_three, _ = seed_ensembles
    assert [rate_of(r.model) for r in top_two] == [rate_of(r.model) for r in all_three[:2]]


def test_reported_loss_labels_the_returned_member(seed_ensembles, decay_data):
    # Under restore_best each member comes back from its best step, so the
    # reported loss must be measured at those parameters, not the last step.
    top_two, _, _ = seed_ensembles
    for result in top_two:
        assert result.final_loss == pytest.approx(data_loss(result.model, decay_data), abs=1e-5)


def test_each_seed_member_is_its_own_ui_run(seed_ensembles):
    _, _, ui = seed_ensembles
    assert ui.invalid == []
    assert ui.count("on_run_start") == ui.count("on_run_end") == 2
    assert ui.count("on_phase_start") == 2


@pytest.fixture(scope="module")
def bootstrap_ensemble(decay_data):
    """Two bootstrap members, each the better of two seeds on its resample."""
    ui = BracketingUI()
    members = parameter_model().fit_ensemble(
        decay_data,
        quiet(steps=4, lr=0.05, tournament_steps=1),
        key=jr.key(0),
        n_members=2,
        n_seeds=2,
        bootstrap=True,
        ui=ui,
    )
    return members, ui


def test_bootstrap_ensemble_trains_one_member_per_resample(bootstrap_ensemble):
    members, ui = bootstrap_ensemble
    assert len(members) == 2
    assert members[0].final_loss <= members[1].final_loss
    assert rate_of(members[0].model) != rate_of(members[1].model)
    assert ui.invalid == []
    assert ui.count("on_run_start") == ui.count("on_run_end") == 2


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        (dict(n_members=3, n_seeds=2), ValueError, "exceeds n_seeds"),
        (dict(n_members=0), ValueError, "n_members must be >= 1"),
        (dict(n_members=2, n_seeds=0, bootstrap=True), ValueError, "n_seeds must be >= 1"),
        (dict(n_members=2.0), TypeError, "must be an int"),
    ],
)
def test_fit_ensemble_rejects_bad_sizes(decay_data, kwargs, error, match):
    with pytest.raises(error, match=match):
        parameter_model().fit_ensemble(decay_data, quiet(steps=1), key=jr.key(0), **kwargs)


def test_the_tournament_keeps_frozen_leaves(temp_data):
    """Re-initialising a candidate must not redraw what the mask freezes.

    The frozen first layer and the default-frozen scalers keep the original
    values in a tournament run and in every ensemble member.
    """
    model = network_model(jr.key(1))
    frozen = "rate.inner.mlp.layers.0"
    config = quiet(steps=2, lr=1e-2, tournament_attempts=2, tournament_steps=1)
    fitted = model.fit(temp_data, config, key=jr.key(0), freeze=frozen).model
    members = model.fit_ensemble(
        temp_data, quiet(steps=2, lr=1e-2), key=jr.key(0), n_members=2, freeze=frozen
    )

    def kept(new: hm.HybridModel, select: Any) -> bool:
        old_leaves = jtu.tree_leaves(eqx.filter(select(model.predictors), eqx.is_array))
        new_leaves = jtu.tree_leaves(eqx.filter(select(new.predictors), eqx.is_array))
        return all(np.array_equal(a, b) for a, b in zip(old_leaves, new_leaves, strict=True))

    def layer0(p: Any) -> Any:
        return p["rate"].inner.mlp.layers[0]

    def scalers(p: Any) -> Any:
        return [
            n
            for n in jtu.tree_leaves(p, is_leaf=lambda x: isinstance(x, BoundScaler))
            if isinstance(n, BoundScaler)
        ]

    for new in (fitted, *(r.model for r in members)):
        assert kept(new, layer0) and kept(new, scalers)
        assert not kept(new, lambda p: p["rate"].inner.mlp.layers[1])


def test_ensemble_ranking_puts_a_nan_loss_last():
    """A NaN final loss would scramble a plain sort; it ranks as ``inf``."""
    losses = [0.5, float("nan"), 0.1]
    members = [cast(hm.FitResult, SimpleNamespace(final_loss=v)) for v in losses]
    ranked = [m.final_loss for m in sorted(members, key=_rank_key)]
    assert ranked[:2] == [0.1, 0.5] and np.isnan(ranked[2])


def test_fit_ensemble_is_optax_only(decay_data):
    config = EvosaxTrainingConfig(population_size=4, steps=1, verbose=False)
    with pytest.raises(TypeError, match="Optax only"):
        parameter_model().fit_ensemble(
            decay_data,
            config,  # ty: ignore[invalid-argument-type]
            key=jr.key(0),
            n_members=1,
        )


def test_failed_seed_ensemble_raises_and_closes_the_bracket(decay_data):
    ui = BracketingUI()
    with pytest.raises(RuntimeError, match="every seed failed"):
        make_model(parameter_predictors(), nan_simulate).fit_ensemble(
            decay_data,
            quiet(steps=2, tournament_steps=1),
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
        return train_with_optax(model, data, quiet(steps=2), **kwargs)
    if trainer == "seed":
        return model.fit_ensemble(data, quiet(steps=2), n_members=2, **kwargs)
    if trainer == "bootstrap":
        return model.fit_ensemble(data, quiet(steps=2), n_members=2, bootstrap=True, **kwargs)
    config = EvosaxTrainingConfig(population_size=4, steps=2, verbose=False)
    return train_with_evosax(model, data, config, **kwargs)


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


# ---------------------------------------------------------------------------
# Prediction grids and ensemble prediction
# ---------------------------------------------------------------------------


def test_predict_ensemble_is_the_member_mean_with_its_spread(seed_ensembles, decay_data):
    top_two, _, _ = seed_ensembles
    models = [r.model for r in top_two]
    ensemble = hm.predict_ensemble(models, decay_data)
    singles = [m.predict(decay_data) for m in models]
    assert list(ensemble) == list(singles[0]) and len(ensemble.members) == 2
    for exp_id, experiment in ensemble.items():
        a, b = (s[exp_id].outputs["conc"] for s in singles)
        np.testing.assert_allclose(experiment.outputs["conc"], (a + b) / 2, rtol=1e-6)
        np.testing.assert_allclose(experiment.std["conc"], jnp.abs(a - b) / 2, atol=1e-6)
        np.testing.assert_array_equal(
            experiment.observed["conc"], singles[0][exp_id].observed["conc"]
        )
        np.testing.assert_allclose(ensemble.members[1][exp_id].outputs["conc"], b)
    for bucket, s0, s1 in zip(ensemble.buckets, *(s.buckets for s in singles), strict=True):
        np.testing.assert_allclose(bucket, (s0 + s1) / 2, rtol=1e-6)

    # compute_metrics scores the mean, so the ensemble's error is the mean's.
    metrics = hm.compute_metrics(ensemble)["conc"]
    mean_only = hm.PredictionResult(
        {
            exp_id: hm.ExperimentPrediction(
                times=e.times,
                outputs=e.outputs,
                observed=e.observed,
                observation_mask=e.observation_mask,
            )
            for exp_id, e in ensemble.items()
        }
    )
    assert float(metrics.mse) == pytest.approx(float(hm.compute_metrics(mean_only)["conc"].mse))

    pytest.importorskip("pandas")
    frame = ensemble.to_dataframe()
    assert list(frame.columns)[-1] == "std" and frame["std"].notna().all()

    # The FitResults from fit_ensemble go in as is.
    from_results = hm.predict_ensemble(top_two, decay_data)
    np.testing.assert_array_equal(from_results["a"].outputs["conc"], ensemble["a"].outputs["conc"])

    dense = hm.predict_ensemble(models, list(decay_data.experiments), n_points=7)
    assert dense["a"].times.shape == (7,) and dense["a"].std["conc"].shape == (7,)
    assert dense["a"].observed == {}


def test_predict_ensemble_needs_matching_channels(decay_data):
    other = hm.HybridModel(
        predictors=parameter_predictors(),
        simulate_fn=simulate_parameter,
        state_to_output=identity,
        output_channel_names=("other",),
        solver=SOLVER,
    )
    with pytest.raises(ValueError, match="same channels"):
        hm.predict_ensemble([parameter_model(), other], decay_data)
    with pytest.raises(ValueError, match="at least one model"):
        hm.predict_ensemble([], decay_data)
    with pytest.raises(TypeError, match="member 1 is a dict"):
        hm.predict_ensemble(
            [parameter_model(), parameter_predictors()],  # ty: ignore[invalid-argument-type]
            decay_data,
        )


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


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        (dict(ts=[0.0, 1.0], n_points=3), ValueError, "not both"),
        (dict(n_points=0), ValueError, "at least 1"),
        (dict(n_points=2.5), TypeError, "must be an int"),
        (dict(ts=[1.0, 0.5]), ValueError, "strictly increasing"),
        (dict(ts=[-1.0, 1.0]), ValueError, "before the start"),
    ],
)
def test_predict_rejects_bad_grids(decay_data, kwargs, error, match):
    with pytest.raises(error, match=match):
        parameter_model().predict(decay_data, **kwargs)


def test_bootstrap_dataset_resamples_with_replacement(decay_data):
    boot = make_bootstrap_dataset(decay_data, key=jr.key(0), n_experiments=8)
    ids = [e.exp_id for e in boot.experiments]
    # Pigeonhole: eight draws from four experiments repeat one, and a repeat
    # is renamed "<id>#<k>" so experiment IDs stay unique.
    sources = [i.split("#")[0] for i in ids]
    assert len(ids) == len(set(ids)) == 8 and len(set(sources)) < 8
    assert sum(int(bp.y0.shape[0]) for bp in boot.bucket_payloads) == 8
    same = make_bootstrap_dataset(decay_data, key=jr.key(0), n_experiments=8)
    assert [e.exp_id for e in same.experiments] == ids
    assert len(make_bootstrap_dataset(decay_data, key=jr.key(1)).experiments) == 4

    bare = Dataset(
        bucket_payloads=decay_data.bucket_payloads,
        output_channel_names=decay_data.output_channel_names,
        covariate_names=decay_data.covariate_names,
    )
    with pytest.raises(ValueError, match="source experiments"):
        make_bootstrap_dataset(bare, key=jr.key(0))


# ---------------------------------------------------------------------------
# Evosax
# ---------------------------------------------------------------------------


def simulate_nested(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """Decay with predictors held as ``((rate,), offset)``."""
    (rate,), offset = predictors
    k, c = rate.value, offset.value
    return solver.diffeqsolve(diffrax.ODETerm(lambda t, y, args: -k * y + c), ts, y0).ys


@pytest.fixture(scope="module")
def evolved(decay_data):
    """CMA-ES on a nested-tuple pytree with the offset frozen by path."""
    model = make_model(
        ((hm.Parameter(0.3, bounds=(0.0, None)),), hm.Parameter(0.0)), simulate_nested
    )
    ui = RecordingUI()
    result = train_with_evosax(
        model,
        list(decay_data.experiments),
        EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.5, verbose=False),
        freeze="1.latent",
        key=jr.key(0),
        ui=ui,
    )
    return model, result, ui


def test_evosax_improves_and_keeps_the_best_ever(evolved, decay_data):
    model, result, ui = evolved
    history = result.history.loss
    assert all(b <= a for a, b in zip(history, history[1:], strict=False))
    assert history[-1] < data_loss(model, decay_data)
    # The returned candidate is the best ever seen, so no worse than the
    # final generation's mean.
    final_mean = next(kw["mean_loss"] for n, kw in reversed(ui.events) if n == "on_generation_end")
    assert data_loss(result.model, decay_data) <= final_mean + 1e-7


def test_frozen_leaf_does_not_move_through_evosax(evolved):
    _, result, _ = evolved
    trained = result.model.predictors
    assert float(trained[1].value) == 0.0
    assert isinstance(trained[0], tuple) and len(trained[0]) == 1


def test_evosax_ui_reports_every_generation(evolved):
    _, _, ui = evolved
    names = [n for n, _ in ui.events]
    assert names[0] == "on_run_start" and names[-1] == "on_run_end"
    generations = [kw["gen_idx"] for n, kw in ui.events if n == "on_generation_end"]
    assert generations == list(range(10))


def test_train_with_evosax_is_what_fit_runs(decay_data):
    config = EvosaxTrainingConfig(population_size=4, steps=3, verbose=False)
    model = parameter_model()
    via_fit = model.fit(decay_data, config, key=jr.key(1))
    direct = train_with_evosax(model, decay_data, config, key=jr.key(1))
    assert direct.history == via_fit.history and direct.best_step == via_fit.best_step
    assert rate_of(direct.model) == rate_of(via_fit.model)
    with pytest.raises(TypeError, match="EvosaxTrainingConfig"):
        train_with_evosax(model, decay_data, quiet(steps=1), key=jr.key(0))  # ty: ignore[invalid-argument-type]


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


@pytest.mark.parametrize("algorithm", ["CMA_ES", "Sep_CMA_ES", "SimpleES"])
def test_registered_algorithms_run(quadratic_data, algorithm):
    result = train_with_evosax(
        quadratic_model(),
        quadratic_data,
        EvosaxTrainingConfig(algorithm=algorithm, population_size=8, steps=3, verbose=False),
        key=jr.key(0),
    )
    history = result.history.loss
    assert len(history) == 3 and all(math.isfinite(v) for v in history)


def test_evosax_rejects_nonfinite_fitness_and_a_missing_key(quadratic_data):
    config = EvosaxTrainingConfig(population_size=4, steps=1, verbose=False)
    with pytest.raises(TypeError):
        train_with_evosax(quadratic_model(), quadratic_data, config)  # ty: ignore[missing-argument]

    def nan_quadratic(predictors, ts, covariates, y0, solver):
        return jnp.full((ts.shape[0], 4), jnp.nan)

    with pytest.raises(FloatingPointError, match="non-finite"):
        train_with_evosax(quadratic_model(nan_quadratic), quadratic_data, config, key=jr.key(0))


def test_box_init_modes_spread_wider_than_a_warm_start():
    """``uniform_box`` and ``lhs_box`` cover the requested extent; ``warm`` stays tight.

    These are the helpers ``train_with_evosax`` calls at generation 0.
    """
    flat = jnp.zeros(4)
    key = jr.key(7)

    def config(init: Literal["warm", "uniform_box", "lhs_box"]) -> EvosaxTrainingConfig:
        return EvosaxTrainingConfig(
            population_size=24, init=init, init_box_extent=2.0, sigma_init=0.5, verbose=False
        )

    def spread(population: Array) -> float:
        return float(jnp.linalg.norm(population[:, None] - population[None], axis=-1).max())

    strategy, params = _build_strategy(config=config("warm"), flat=flat)
    warm, _ = strategy.ask(key, strategy.init(key, flat, params), params)
    for init in ("uniform_box", "lhs_box"):
        population = _box_population(flat=flat, config=config(init), key=key)
        assert population.shape == (24, 4)
        assert float(jnp.abs(population).max()) <= 2.0 + 1e-6
        assert spread(population) >= 1.5 * spread(warm)


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


def test_evosax_validation_picks_the_generation_best_it_scores_lowest(quadratic_data):
    """With ``validation=``, the returned individual is the validation argmin.

    Validating against a shifted target makes the validation and training
    rankings disagree, so the selection is visibly not the training best.
    """
    shifted = hm.make_dataset(
        [
            hm.make_experiment(
                channels={
                    name: hm.ChannelObs(ts=jnp.array([0.0]), values=THETA_STAR[i : i + 1] * 0.0)
                    for i, name in enumerate(QUADRATIC_CHANNELS)
                },
                y0=jnp.zeros(4),
                exp_id="v",
            )
        ],
        output_channel_names=QUADRATIC_CHANNELS,
    )
    result = train_with_evosax(
        quadratic_model(),
        quadratic_data,
        EvosaxTrainingConfig(population_size=8, steps=6, sigma_init=0.5, verbose=False),
        key=jr.key(0),
        validation=shifted,
    )
    history = result.history
    assert len(history.loss) == len(history.validation_loss) == 6
    assert all(b <= a for a, b in zip(history.loss, history.loss[1:], strict=False))
    assert result.best_step == int(np.argmin(history.validation_loss))
    assert result.best_step is not None
    returned = float(jnp.mean(result.model.predictors.value**2))
    assert returned == pytest.approx(history.validation_loss[result.best_step], rel=1e-5)


def test_channel_weights_by_name_fit_only_the_named_channels(quadratic_data):
    """A by-name mapping scores only the channels it names.

    The unnamed channels get no gradient, and adamw's decoupled weight decay
    leaves a zero leaf at exactly zero, so they stay put.
    """
    result = train_with_optax(
        quadratic_model(),
        quadratic_data,
        quiet(steps=60, lr=0.2),
        objective=Objective(channel_weights={"c0": 1.0, "c1": 2.0}),
        key=jr.key(0),
    )
    value = result.model.predictors.value
    assert float(jnp.abs(value[:2] - THETA_STAR[:2]).max()) < 0.2
    assert value[2:].tolist() == [0.0, 0.0]


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


# ---------------------------------------------------------------------------
# Adjoints
# ---------------------------------------------------------------------------


def solution_sum(rate: Array, adjoint: str, *, threaded: bool) -> Array:
    """Sum of the decay solution at a few times, differentiable in ``rate``.

    Backsolve re-integrates an adjoint ODE and can only differentiate
    values passed through ``args``, so ``threaded=True`` routes the rate
    there instead of closing over it.
    """
    solver = hm.SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-6,
        atol=1e-8,
        max_steps=32,
        dt0=0.1,
        adjoint=ADJOINT_REGISTRY[adjoint](),
    )
    ts = jnp.array([0.0, 0.5, 1.0, 2.0])
    y0 = jnp.array([1.0])
    if threaded:
        term = diffrax.ODETerm(lambda t, y, k: -k * y)
        return solver.diffeqsolve(term, ts, y0, args=rate).ys.sum()
    term = diffrax.ODETerm(lambda t, y, args: -rate * y)
    return solver.diffeqsolve(term, ts, y0).ys.sum()


@pytest.mark.parametrize(
    "adjoint,threaded",
    [("Direct", False), ("RecursiveCheckpoint", False), ("Backsolve", True)],
)
def test_adjoint_gradients_match_the_analytic_derivative(adjoint, threaded):
    # y(t) = exp(-k t), so d/dk sum_t y(t) = -sum_t t exp(-k t). An exact
    # anchor, so adjoints that shared a bug would still fail.
    rate = jnp.asarray(TRUE_RATE)
    ts = jnp.array([0.0, 0.5, 1.0, 2.0])
    expected = float(-(ts * jnp.exp(-rate * ts)).sum())
    grad = jax.grad(lambda k: solution_sum(k, adjoint, threaded=threaded))(rate)
    assert float(grad) == pytest.approx(expected, rel=1e-4)


def test_backsolve_rejects_closed_over_values():
    # Pinned so switching SolverConfig.adjoint to Backsolve on the usual
    # closed-over simulate_fn gives an error, not a silently wrong gradient.
    with pytest.raises(Exception, match="closed-over"):
        jax.grad(lambda k: solution_sum(k, "Backsolve", threaded=False))(jnp.asarray(1.0))


# ---------------------------------------------------------------------------
# Trainable masks
# ---------------------------------------------------------------------------


def flags(mask: Any) -> list[bool]:
    return [bool(leaf) for leaf in jtu.tree_leaves(mask)]


def test_default_mask_trains_networks_and_parameters_only():
    predictors = network_predictors()
    predictors["k"] = hm.Parameter(0.5, bounds=(0.0, 1.0))
    mask = trainable_mask(predictors)
    assert jtu.tree_structure(mask) == jtu.tree_structure(predictors)
    rate = mask["rate"]
    assert rate.in_scaler.temperature is False and rate.out_scaler.temperature is False
    assert not any(flags(mask["offset"]))
    assert mask["k"].latent is True and mask["k"].scaler.temperature is False
    inner_leaves = jtu.tree_leaves(predictors["rate"].inner)
    assert flags(rate.inner) == [eqx.is_inexact_array(leaf) for leaf in inner_leaves]

    scalers_too = trainable_mask(predictors, train_scalers=True)
    assert scalers_too["rate"].in_scaler.temperature is True
    assert not any(flags(scalers_too["offset"])), "a Constant is never trainable"
    assert not any(flags(trainable_mask(predictors, predicate=lambda _: False)))


def test_mask_helpers_take_predictors_first_and_chain_through_mask():
    """Every helper takes the predictors first and an optional base mask=.

    A chained call freezes on top of the base without mutating it, and
    count_trainable_params counts against the same default mask.
    """
    predictors = network_predictors()["rate"]
    base = trainable_mask(predictors, train_scalers=True)

    no_scalers = freeze_mask(predictors, BoundScaler, mask=base)
    by_fn = freeze_mask(predictors, lambda m: isinstance(m, BoundScaler), mask=base)
    for frozen in (no_scalers, by_fn):
        assert frozen.in_scaler.temperature is False and frozen.out_scaler.temperature is False
        assert flags(frozen.inner) == flags(base.inner)
    assert base.in_scaler.temperature is True, "the base mask was mutated"

    nothing = freeze_mask(predictors, MLPPredictor, mask=no_scalers)
    assert not any(flags(nothing))
    assert count_trainable_params(predictors, mask=nothing) == 0

    one = freeze_mask(predictors, "in_scaler.temperature", mask=base)
    assert one.in_scaler.temperature is False and one.out_scaler.temperature is True
    assert sum(flags(one)) == sum(flags(base)) - 1
    assert flags(freeze_mask(predictors, (), mask=base)) == flags(base)

    # The default base is trainable_mask(predictors), scalers frozen.
    assert flags(freeze_mask(predictors, ())) == flags(trainable_mask(predictors))
    n_inner = sum(
        int(leaf.size)
        for leaf in jtu.tree_leaves(eqx.filter(predictors.inner, eqx.is_inexact_array))
    )
    assert count_trainable_params(predictors) == n_inner
    assert count_trainable_params(predictors, mask=base) == n_inner + 2


def test_freeze_mask_refuses_entries_that_match_nothing_and_suggests_paths():
    # A silent no-op on a typo trains a leaf the caller believes is frozen.
    predictors = network_predictors()["rate"]
    with pytest.raises(ValueError, match="no leaf matches"):
        freeze_mask(predictors, ("in_scaler.temperature", "nope.nope"))
    with pytest.raises(ValueError, match="Closest matches"):
        freeze_mask(predictors, "in_scaler.temperatur")
