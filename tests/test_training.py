"""The functional trainers, end to end on the decay model.

Every trainer call compiles its kernels from scratch, which costs a couple
of seconds, so runs are shared through module fixtures and each test reads
one property off a run. Config validation lives in ``test_config.py`` and
the bound and trajectory penalties in ``test_penalties.py``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

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
    NETWORK_KWARGS,
    PARAMETER_KWARGS,
    SOLVER,
    TRUE_RATE,
    identity,
    network_predictors,
    parameter_predictors,
    quiet,
    simulate_parameter,
)
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.data import Dataset, make_bootstrap_dataset
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.prediction import predict_dataset, predict_dense
from jaxhybridmodels.predictors import BoundScaler, MLPPredictor
from jaxhybridmodels.solver import ADJOINT_REGISTRY
from jaxhybridmodels.trainable import (
    freeze_modules_of_type,
    freeze_paths,
    freeze_where,
    trainable_mask,
)
from jaxhybridmodels.training import TrainingPhase, train_with_optax
from jaxhybridmodels.training.evosax import (
    EvosaxTrainingConfig,
    _box_population,
    _build_single_eval,
    _build_strategy,
    _train_with_evosax,
    train_with_evosax,
)
from jaxhybridmodels.training.optax import (
    _shared_tournament,
    _train_with_optax,
    train_bootstrap_ensemble,
    train_seed_ensemble,
)
from jaxhybridmodels.ui.testing import RecordingUI


def data_loss(predictors: Any, dataset: Dataset, **kwargs: Any) -> float:
    """The trainer's data term: per-bucket ``masked_mse``, averaged over buckets."""
    kwargs = kwargs or PARAMETER_KWARGS
    predictions = predict_dataset(predictors, dataset, **kwargs)
    total = sum(
        float(masked_mse(pred, bp))
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True)
    )
    return total / len(dataset.bucket_payloads)


def rate_of(predictors: Any) -> float:
    return float(predictors["rate"].value)


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
    mask = freeze_paths(trainable_mask(predictors), ("offset.latent",))
    ui = RecordingUI()
    history, trained = train_with_optax(
        predictors,
        decay_data,
        quiet(phases=(TrainingPhase(20, lr=0.1, horizon=0.5), TrainingPhase(20, lr=0.05))),
        simulate_fn=counted,
        state_to_output=identity,
        solver=SOLVER,
        trainable=mask,
        key=jr.key(0),
        ui=ui,
    )
    return history, trained, ui, traces[0]


def test_two_phase_fit_recovers_the_rate(phased):
    history, trained, _, _ = phased
    assert len(history) == 40
    assert rate_of(trained) == pytest.approx(TRUE_RATE, abs=0.02)
    assert history[-1] < history[0]


def test_a_horizon_change_does_not_recompile(phased):
    # The horizon is a traced fraction, not a static shape, so both phases
    # share one compiled kernel per bucket. The decay data has one bucket
    # per union length; anything above that count is a retrace.
    _, _, _, traces = phased
    assert traces <= 2


def test_frozen_leaf_does_not_move_through_optax(phased):
    # adamw decays every leaf it is given, gradient or not, so a frozen
    # leaf that leaked into the optimiser would drift away from zero.
    _, trained, _, _ = phased
    assert float(trained["offset"].value) == 0.0


def test_ui_events_bracket_the_run_in_order(phased):
    _, _, ui, _ = phased
    names = [name for name, _ in ui.events]
    assert names[0] == "on_run_start"
    assert names[-1] == "on_run_end"
    assert names.count("on_step_end") == 40
    assert names.count("on_phase_start") == names.count("on_phase_end") == 2
    assert names.index("on_compile_done") < names.index("on_phase_start")


def test_missing_key_raises(decay_data):
    with pytest.raises(TypeError):
        train_with_optax(parameter_predictors(), decay_data, quiet(steps=1), **PARAMETER_KWARGS)


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
    history, _ = train_with_optax(
        predictors,
        decay_data,
        quiet(steps=20, lr=0.0, patience=3),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    assert len(history) == 4
    assert all(math.isfinite(v) for v in history)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# restore_best
# ---------------------------------------------------------------------------

# Phase 1 overshoots at a high learning rate, so its best step is mid-phase.
# Phase 2 runs at zero and cannot improve on anything. Both share a horizon.
OVERSHOOT_PHASES = (TrainingPhase(15, lr=0.6), TrainingPhase(3, lr=0.0))


@pytest.fixture(scope="module")
def overshoot(decay_data):
    runs = {}
    for restore in (True, False):
        runs[restore] = train_with_optax(
            parameter_predictors(),
            decay_data,
            quiet(phases=OVERSHOOT_PHASES, restore_best=restore),
            key=jr.key(0),
            **PARAMETER_KWARGS,
        )
    return runs


def test_restore_best_returns_the_pre_update_snapshot(overshoot, decay_data):
    """The returned model scores exactly the history minimum.

    Each step's loss is measured before that step's update. Snapshotting
    after the update would return a model whose loss is not the reported
    minimum.
    """
    history, best = overshoot[True]
    argmin = min(range(len(history)), key=history.__getitem__)
    assert argmin < OVERSHOOT_PHASES[0].steps - 1, "phase 1 was meant to overshoot"
    assert data_loss(best, decay_data) == pytest.approx(min(history), rel=1e-4)


def test_restore_best_false_returns_the_final_step(overshoot, decay_data):
    # restore_best changes what is handed back, not how training runs.
    best_history, best = overshoot[True]
    last_history, last = overshoot[False]
    assert best_history == last_history
    assert rate_of(last) != rate_of(best)
    # Same horizon in both phases, so the running minimum carries across
    # the boundary and the mid-phase-1 best beats the endpoint.
    assert data_loss(best, decay_data) < data_loss(last, decay_data)


@pytest.fixture(scope="module")
def curriculum(temp_data):
    """A short-horizon phase then a full one, against data one rate cannot fit.

    The rate in ``temp_data`` depends on temperature, so a single
    ``Parameter`` has a loss floor over the full window. Over the first
    observation every curve looks alike and the loss is far lower.
    """
    short = TrainingPhase(10, lr=0.1, horizon=0.2)
    both = train_with_optax(
        parameter_predictors(),
        temp_data,
        quiet(phases=(short, TrainingPhase(15, lr=0.05))),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    short_only = train_with_optax(
        parameter_predictors(),
        temp_data,
        quiet(phases=(short,)),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    return both, short_only


def test_restore_best_does_not_compare_across_horizons(curriculum, temp_data):
    """The best step is chosen within the final horizon only.

    Losses over different horizons measure different things. One running
    minimum across both phases lands in the short phase, handing back the
    least-trained model of the run.
    """
    (history, restored), (_, short_only) = curriculum
    # Anti-vacuity: the global minimum of the history is in the short phase.
    assert min(range(len(history)), key=history.__getitem__) < 10
    assert data_loss(restored, temp_data) < data_loss(short_only, temp_data)


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

    predictors = network_predictors()
    history, trained = train_with_optax(
        predictors,
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
        **NETWORK_KWARGS,
    )
    assert calls, "the custom loss was never traced"
    assert history[-1] < 0.5 * history[0]
    assert set(trained) == {"rate", "offset"}


def test_raw_optimizer_after_a_reset_and_refused_lr_change(decay_data):
    """A raw ``GradientTransformation`` cannot take a new learning rate.

    After ``reset_optimizer`` it starts fresh and is fine. Reusing the same
    raw object across a learning-rate change passes the config check (the
    specs are equal) and must be refused when the phase begins.
    """
    raw = optax.adam(1e-2)
    history, trained = train_with_optax(
        parameter_predictors(),
        decay_data,
        quiet(
            phases=(
                TrainingPhase(3, lr=0.05),
                TrainingPhase(3, lr=1e-2, optimizer=raw, reset_optimizer=True),
            )
        ),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    assert len(history) == 6 and rate_of(trained) != 0.3

    refused = quiet(
        phases=(TrainingPhase(2, lr=1e-3, optimizer=raw), TrainingPhase(2, lr=1e-2, optimizer=raw))
    )
    with pytest.raises(ValueError, match="cannot be re-hyperparametrised"):
        train_with_optax(
            parameter_predictors(), decay_data, refused, key=jr.key(0), **PARAMETER_KWARGS
        )


def test_lr_schedule_counts_from_the_phase_start(decay_data):
    """A phase's schedule sees step 0 at the phase start, not the run start.

    Phase 1 runs five steps at zero. Phase 2's schedule is zero for its own
    first three steps and then trains. Each history entry is measured before
    that step's update, so the loss stays flat through step 8 (the fourth
    step of phase 2) and moves at step 9. Counting from the run start would
    already train at step 5.
    """
    held = optax.join_schedules([optax.constant_schedule(0.0), optax.constant_schedule(0.1)], [3])
    history, trained = train_with_optax(
        parameter_predictors(),
        decay_data,
        quiet(phases=(TrainingPhase(5, lr=0.0), TrainingPhase(6, lr=held)), restore_best=False),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    assert len(set(history[:9])) == 1
    assert history[9] != history[8]
    assert rate_of(trained) != 0.3


def test_lr_schedule_on_the_single_phase_shorthand_and_after_a_float_phase(decay_data):
    """``steps=, lr=schedule`` trains, and a float phase after it resets the lr.

    A schedule that decays to zero halfway leaves the second half flat; the
    float phase that follows must move the parameter again, so the schedule's
    last value cannot leak past its phase.
    """
    to_zero = optax.linear_schedule(0.1, 0.0, 4)
    history, _ = train_with_optax(
        parameter_predictors(),
        decay_data,
        quiet(steps=8, lr=to_zero, restore_best=False),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    assert history[1] != history[0] and len(set(history[5:])) == 1

    history, _ = train_with_optax(
        parameter_predictors(),
        decay_data,
        quiet(phases=(TrainingPhase(6, lr=to_zero), TrainingPhase(3, lr=0.05))),
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    assert history[5] == history[6] and history[8] != history[7]


def test_validation_set_drives_selection_through_the_functional_layer(decay_data, temp_data):
    """``validation=`` scores every step with the training objective.

    Validating on data the model is not trained on means the selection
    series differs from the training loss, and the returned step is the
    validation minimum.
    """
    run = _train_with_optax(
        parameter_predictors(),
        decay_data,
        quiet(steps=12, lr=0.1),
        key=jr.key(0),
        validation=temp_data,
        **PARAMETER_KWARGS,
    )
    assert len(run.selection) == len(run.loss) == 12
    assert run.selection != run.loss
    assert run.best_step == int(np.argmin(run.selection))
    assert data_loss(run.predictors, temp_data) == pytest.approx(min(run.selection), rel=1e-4)

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
    with pytest.raises(ValueError, match="validation channels"):
        train_with_optax(
            parameter_predictors(),
            decay_data,
            quiet(steps=1),
            key=jr.key(0),
            validation=mismatched,
            **PARAMETER_KWARGS,
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
    losses = {}
    for attempts in (2, 3):
        _, winner = train_with_optax(
            parameter_predictors(),
            decay_data,
            tournament_config(attempts),
            key=jr.key(12),
            **PARAMETER_KWARGS,
        )
        losses[attempts] = data_loss(winner, decay_data)
    return losses


def test_more_attempts_give_a_better_winner(tournament_losses):
    """The tournament returns the best candidate, not the first survivor.

    Attempt keys are folded from the attempt index, so the pool of three
    contains the pool of two. Under first-survivor-wins both runs would
    return attempt 0 and score the same.
    """
    assert tournament_losses[3] < tournament_losses[2] - 1e-9, tournament_losses


def test_tournament_falls_back_when_every_attempt_fails(decay_data):
    with pytest.warns(RuntimeWarning, match="all attempts failed"):
        train_with_optax(
            parameter_predictors(),
            decay_data,
            quiet(steps=1, tournament_attempts=2, tournament_steps=1),
            simulate_fn=nan_simulate,
            state_to_output=identity,
            solver=SOLVER,
            key=jr.key(0),
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
    """The same key with ``k_best=2`` and with the default ``k_best = n_seeds``."""
    ui = BracketingUI()
    top_two = train_seed_ensemble(
        parameter_predictors(),
        decay_data,
        SEED_CONFIG,
        n_seeds=3,
        k_best=2,
        key=jr.key(0),
        ui=ui,
        **PARAMETER_KWARGS,
    )
    all_three = train_seed_ensemble(
        parameter_predictors(),
        decay_data,
        SEED_CONFIG,
        n_seeds=3,
        key=jr.key(0),
        **PARAMETER_KWARGS,
    )
    return top_two, all_three, ui


def test_seed_ensemble_ranks_the_k_best_members(seed_ensembles, decay_data):
    top_two, all_three, _ = seed_ensembles
    assert len(top_two) == 2 and len(all_three) == 3
    losses = [loss for loss, _ in top_two]
    assert losses == sorted(losses)
    assert losses[0] < data_loss(parameter_predictors(), decay_data)


def test_seed_ensemble_is_deterministic_in_the_key(seed_ensembles):
    # Members are seeded by rank, so the top two of a three-member run with
    # the same key are the same models as a two-member run.
    top_two, all_three, _ = seed_ensembles
    assert [rate_of(p) for _, p in top_two] == [rate_of(p) for _, p in all_three[:2]]


def test_reported_loss_labels_the_returned_member(seed_ensembles, decay_data):
    # Under restore_best each member comes back from its best step, so the
    # reported loss must be measured at those parameters, not the last step.
    top_two, _, _ = seed_ensembles
    for loss, member in top_two:
        assert loss == pytest.approx(data_loss(member, decay_data), abs=1e-5)


def test_each_seed_member_is_its_own_ui_run(seed_ensembles):
    _, _, ui = seed_ensembles
    assert ui.invalid == []
    assert ui.count("on_run_start") == ui.count("on_run_end") == 2
    assert ui.count("on_phase_start") == 2


def test_bootstrap_members_are_ui_bracketed(decay_data):
    ui = BracketingUI()
    members = train_bootstrap_ensemble(
        parameter_predictors(),
        decay_data,
        quiet(steps=4, lr=0.05),
        n_bootstraps=2,
        key=jr.key(0),
        ui=ui,
        **PARAMETER_KWARGS,
    )
    assert len(members) == 2
    assert ui.invalid == []
    assert ui.count("on_run_start") == ui.count("on_run_end") == 2


def test_failed_seed_ensemble_falls_back_and_closes_the_bracket(decay_data):
    ui = BracketingUI()
    with pytest.warns(RuntimeWarning, match="every seed failed"):
        members = train_seed_ensemble(
            parameter_predictors(),
            decay_data,
            quiet(steps=2, tournament_steps=1),
            simulate_fn=nan_simulate,
            state_to_output=identity,
            solver=SOLVER,
            n_seeds=2,
            k_best=1,
            key=jr.key(0),
            ui=ui,
        )
    assert len(members) == 1 and members[0][0] == float("inf")
    assert ui.invalid == []
    assert ui.count("on_run_start") == ui.count("on_run_end") == 1


def raising_simulate(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """Fails while the training kernels trace, after the UI run has opened."""
    raise RuntimeError("simulate_fn failed")


@pytest.mark.parametrize("trainer", ["optax", "seed", "bootstrap", "evosax"])
def test_bad_objective_raises_before_the_ui_opens(decay_data, trainer):
    # A Rich Live dashboard opened by on_run_start would be left on screen
    # if the objective check raised after it.
    ui = BracketingUI()
    bad = Objective(channel_weights={"missing": 1.0})
    common: dict[str, Any] = dict(objective=bad, key=jr.key(0), ui=ui, **PARAMETER_KWARGS)
    with pytest.raises(ValueError, match="unknown channels"):
        if trainer == "optax":
            train_with_optax(parameter_predictors(), decay_data, quiet(steps=2), **common)
        elif trainer == "seed":
            train_seed_ensemble(
                parameter_predictors(), decay_data, quiet(steps=2), n_seeds=2, **common
            )
        elif trainer == "bootstrap":
            train_bootstrap_ensemble(
                parameter_predictors(), decay_data, quiet(steps=2), n_bootstraps=2, **common
            )
        else:
            config = EvosaxTrainingConfig(population_size=4, steps=2, verbose=False)
            train_with_evosax(parameter_predictors(), decay_data, config, **common)
    assert ui.events == []


@pytest.mark.parametrize("trainer", ["optax", "seed", "evosax"])
def test_error_after_run_start_closes_the_ui(decay_data, trainer):
    ui = BracketingUI()
    kwargs: dict[str, Any] = dict(
        simulate_fn=raising_simulate, state_to_output=identity, solver=SOLVER
    )
    with pytest.raises(RuntimeError, match="simulate_fn failed"):
        if trainer == "optax":
            train_with_optax(
                parameter_predictors(), decay_data, quiet(steps=2), key=jr.key(0), ui=ui, **kwargs
            )
        elif trainer == "seed":
            train_seed_ensemble(
                parameter_predictors(),
                decay_data,
                quiet(steps=2),
                n_seeds=2,
                key=jr.key(0),
                ui=ui,
                **kwargs,
            )
        else:
            config = EvosaxTrainingConfig(population_size=4, steps=2, verbose=False)
            train_with_evosax(
                parameter_predictors(), decay_data, config, key=jr.key(0), ui=ui, **kwargs
            )
    assert ui.count("on_run_start") == ui.count("on_run_end") == 1
    assert not ui.open
    assert math.isnan(ui.events[-1][1]["final_loss"])


def test_ensemble_predictions_average_and_dense_grids_agree(seed_ensembles, decay_data):
    top_two, _, _ = seed_ensembles
    members = tuple(p for _, p in top_two)
    mean = hm.ensemble_predictions(members, decay_data, **PARAMETER_KWARGS)
    singles = [predict_dataset(p, decay_data, **PARAMETER_KWARGS) for p in members]
    for b, bucket_mean in enumerate(mean):
        expected = (singles[0][b] + singles[1][b]) / 2
        assert jnp.allclose(bucket_mean, expected)

    # A dense grid at the measured times reproduces predict_dataset.
    bucket = decay_data.bucket_payloads[0]
    dense = predict_dense(members[0], decay_data, ts_grid=bucket.ts[0], **PARAMETER_KWARGS)
    assert jnp.allclose(dense[0][0], singles[0][0][0], atol=1e-4)


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
    predictors = ((hm.Parameter(0.3, bounds=(0.0, None)),), hm.Parameter(0.0))
    mask = freeze_paths(trainable_mask(predictors), ("1.latent",))
    ui = RecordingUI()
    history, trained = train_with_evosax(
        predictors,
        decay_data,
        EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.5, verbose=False),
        simulate_fn=simulate_nested,
        state_to_output=identity,
        solver=SOLVER,
        trainable=mask,
        key=jr.key(0),
        ui=ui,
    )
    return predictors, history, trained, ui


def test_evosax_improves_and_keeps_the_best_ever(evolved, decay_data):
    predictors, history, trained, ui = evolved
    kwargs = dict(simulate_fn=simulate_nested, state_to_output=identity, solver=SOLVER)
    assert all(b <= a for a, b in zip(history, history[1:], strict=False))
    assert history[-1] < data_loss(predictors, decay_data, **kwargs)
    # The returned candidate is the best ever seen, so no worse than the
    # final generation's mean.
    final_mean = next(kw["mean_loss"] for n, kw in reversed(ui.events) if n == "on_generation_end")
    assert data_loss(trained, decay_data, **kwargs) <= final_mean + 1e-7


def test_frozen_leaf_does_not_move_through_evosax(evolved):
    _, _, trained, _ = evolved
    assert float(trained[1].value) == 0.0
    assert isinstance(trained[0], tuple) and len(trained[0]) == 1


def test_evosax_ui_reports_every_generation(evolved):
    _, _, _, ui = evolved
    names = [n for n, _ in ui.events]
    assert names[0] == "on_run_start" and names[-1] == "on_run_end"
    generations = [kw["gen_idx"] for n, kw in ui.events if n == "on_generation_end"]
    assert generations == list(range(10))


# A no-ODE target for the cheap evosax checks: the "simulation" is the
# parameter vector itself, observed once.
THETA_STAR = jnp.array([1.0, -2.0, 3.0, -4.0])
QUADRATIC_KWARGS: dict[str, Any] = dict(
    simulate_fn=lambda p, ts, cov, y0, solver: jnp.broadcast_to(p.value, (ts.shape[0], 4)),
    state_to_output=identity,
    solver=SOLVER,
)


@pytest.fixture(scope="module")
def quadratic_data():
    ts = jnp.array([0.0])
    channels = {f"c{i}": hm.ChannelObs(ts=ts, values=THETA_STAR[i : i + 1]) for i in range(4)}
    experiment = hm.make_experiment(channels=channels, y0=jnp.zeros(4), exp_id="q")
    return hm.make_dataset([experiment], output_channel_names=tuple(channels))


@pytest.mark.parametrize("algorithm", ["CMA_ES", "Sep_CMA_ES", "SimpleES"])
def test_registered_algorithms_run(quadratic_data, algorithm):
    history, _ = train_with_evosax(
        hm.Parameter(jnp.zeros(4)),
        quadratic_data,
        EvosaxTrainingConfig(algorithm=algorithm, population_size=8, steps=3, verbose=False),
        key=jr.key(0),
        **QUADRATIC_KWARGS,
    )
    assert len(history) == 3 and all(math.isfinite(v) for v in history)


def test_evosax_rejects_nonfinite_fitness_and_a_missing_key(quadratic_data):
    config = EvosaxTrainingConfig(population_size=4, steps=1, verbose=False)
    with pytest.raises(TypeError):
        train_with_evosax(hm.Parameter(jnp.zeros(4)), quadratic_data, config, **QUADRATIC_KWARGS)
    with pytest.raises(FloatingPointError, match="non-finite"):
        train_with_evosax(
            hm.Parameter(jnp.zeros(4)),
            quadratic_data,
            config,
            simulate_fn=lambda p, ts, cov, y0, solver: jnp.full((ts.shape[0], 4), jnp.nan),
            state_to_output=identity,
            solver=SOLVER,
            key=jr.key(0),
        )


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
        **QUADRATIC_KWARGS,
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
                    f"c{i}": hm.ChannelObs(ts=jnp.array([0.0]), values=THETA_STAR[i : i + 1] * 0.0)
                    for i in range(4)
                },
                y0=jnp.zeros(4),
                exp_id="v",
            )
        ],
        output_channel_names=quadratic_data.output_channel_names,
    )
    run = _train_with_evosax(
        hm.Parameter(jnp.zeros(4)),
        quadratic_data,
        EvosaxTrainingConfig(population_size=8, steps=6, sigma_init=0.5, verbose=False),
        key=jr.key(0),
        validation=shifted,
        **QUADRATIC_KWARGS,
    )
    assert len(run.loss) == len(run.validation) == 6
    assert all(b <= a for a, b in zip(run.loss, run.loss[1:], strict=False))
    assert run.best_step == int(np.argmin(run.validation))
    returned = float(jnp.mean(run.predictors.value**2))
    assert returned == pytest.approx(run.validation[run.best_step], rel=1e-5)


def test_channel_weights_by_name_fit_only_the_named_channels(quadratic_data):
    """A by-name mapping scores only the channels it names.

    The unnamed channels get no gradient, and adamw's decoupled weight decay
    leaves a zero leaf at exactly zero, so they stay put.
    """
    _, trained = train_with_optax(
        hm.Parameter(jnp.zeros(4)),
        quadratic_data,
        quiet(steps=60, lr=0.2),
        objective=Objective(channel_weights={"c0": 1.0, "c1": 2.0}),
        key=jr.key(0),
        **QUADRATIC_KWARGS,
    )
    value = trained.value
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


def test_freeze_helpers_return_new_masks():
    predictors = network_predictors()["rate"]
    mask = trainable_mask(predictors, train_scalers=True)

    by_type = freeze_modules_of_type(mask, BoundScaler)
    by_fn = freeze_where(mask, lambda m: isinstance(m, BoundScaler))
    for frozen in (by_type, by_fn):
        assert frozen.in_scaler.temperature is False and frozen.out_scaler.temperature is False
        assert flags(frozen.inner) == flags(mask.inner)
    assert mask.in_scaler.temperature is True, "the input mask was mutated"

    both = freeze_modules_of_type(by_type, MLPPredictor)
    assert not any(flags(both))

    one = freeze_paths(mask, ("in_scaler.temperature",))
    assert one.in_scaler.temperature is False and one.out_scaler.temperature is True
    assert sum(flags(one)) == sum(flags(mask)) - 1
    assert flags(freeze_paths(mask, ())) == flags(mask)


def test_freeze_paths_refuses_unknown_paths_and_suggests_matches():
    # A silent no-op on a typo trains a leaf the caller believes is frozen.
    mask = trainable_mask(network_predictors()["rate"], train_scalers=True)
    with pytest.raises(ValueError, match="no leaf matches"):
        freeze_paths(mask, ("in_scaler.temperature", "nope.nope"))
    with pytest.raises(ValueError, match="Closest matches"):
        freeze_paths(mask, ("in_scaler.temperatur",))
