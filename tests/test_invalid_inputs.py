"""Every way of misusing the public API that the package promises to reject.

Mostly tables: each row is a mistake a user can make, with the error it gets, so
a row documents a message as much as it tests one. Calls fail before compiling
unless a row needs a real prediction or fit to reach its check.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd
import pytest
from _models import (
    CHANNELS,
    SOLVER,
    decay_dataset,
    identity,
    make_model,
    network_predictors,
    parameter_model,
    parameter_predictors,
    simulate_network,
    simulate_parameter,
)
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.data import (
    BucketPayload,
    Dataset,
    experiment_from_array,
    experiments_from_dataframe,
)
from jaxhybridmodels.losses import (
    masked_mse,
    resolve_loss_fn,
)
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.predictors import (
    BoundedPredictor,
    KANPredictor,
    MLPPredictor,
    NeuralNPolynomial,
)
from jaxhybridmodels.profiles import (
    piecewise_linear_profile,
    ramp_profile,
)
from jaxhybridmodels.rng import check_key
from jaxhybridmodels.schedules import annealing_schedule
from jaxhybridmodels.trainable import freeze_mask
from jaxhybridmodels.training import (
    EvosaxTrainingConfig,
    OptaxTrainingConfig,
    TrainingPhase,
    train_with_optax,
)

TS = jnp.array([0.5, 1.0, 2.0])


def obs(**overrides: Any) -> hm.ChannelObs:
    return hm.ChannelObs(**{"ts": TS, "values": jnp.exp(-TS), **overrides})


def experiment(**overrides: Any) -> hm.Experiment:
    fields: dict[str, Any] = {"channels": {"conc": obs()}, "y0": [1.0], "exp_id": "x", **overrides}
    return hm.make_experiment(**fields)


def fit(model: hm.HybridModel | None = None, **kwargs: Any) -> Any:
    model = parameter_model() if model is None else model
    fields: dict[str, Any] = {"key": jr.key(0), **kwargs}
    return model.fit(decay_dataset(), hm.OptaxTrainingConfig(steps=1), **fields)


def unchecked(value: Any) -> Any:
    """``value`` typed ``Any``, for rows that break a signature on purpose."""
    return value


def unchecked_model() -> Any:
    """A model typed ``Any``, for rows that break the signature on purpose."""
    return parameter_model()


def bad_output(state: Any) -> Any:
    return state[..., 0]


CASES: list[tuple[str, Callable[[], Any], type[Exception], str]] = [
    # Data
    ("channel-lengths", lambda: obs(values=TS[:2]), ValueError, "same length"),
    ("channel-nan", lambda: obs(values=TS.at[0].set(jnp.nan)), ValueError, "finite"),
    ("repeated-time", lambda: obs(ts=jnp.array([1.0, 1.0, 2.0])), ValueError, "must not repeat"),
    (
        "no-initial-state",
        lambda: hm.make_experiment(channels={"conc": obs()}),
        ValueError,
        "exactly one of y0",
    ),
    (
        "t0-after-data",
        lambda: hm.make_dataset([experiment(t0=5.0)], output_channel_names=("conc",)),
        ValueError,
        "t0 must precede",
    ),
    (
        "missing-channel",
        lambda: hm.make_dataset([experiment()], output_channel_names=("other",)),
        ValueError,
        "missing requested output channels",
    ),
    (
        "duplicate-id",
        lambda: hm.make_dataset([experiment(), experiment()], output_channel_names=("conc",)),
        ValueError,
        "duplicate exp_id",
    ),
    (
        "zero-buckets",
        lambda: hm.make_dataset([experiment()], output_channel_names=("conc",), max_buckets=0),
        ValueError,
        "max_buckets must be a positive int",
    ),
    (
        "times-too-close-to-pad",
        lambda: hm.make_dataset(
            [
                hm.make_experiment(
                    channels={
                        "conc": hm.ChannelObs(
                            ts=jnp.array([1.0, jnp.nextafter(1.0, 2.0)]), values=jnp.zeros(2)
                        )
                    },
                    y0=[1.0],
                    exp_id="tight",
                ),
                experiment(),
            ],
            output_channel_names=("conc",),
            max_buckets=1,
        ),
        ValueError,
        "too close together",
    ),
    # Predictors
    ("reversed-bounds", lambda: hm.Parameter(0.5, bounds=(2.0, 1.0)), ValueError, "low < high"),
    ("outside-bounds", lambda: hm.Parameter(5.0, bounds=(0.0, 1.0)), ValueError, "strictly inside"),
    (
        "evaluate-array",
        lambda: hm.evaluate_predictor(
            hm.BoundedPredictor(inner=mlp(jr.key(0)), input_bounds=((0, 1),) * 2),
            jnp.zeros((5, 2)),  # ty: ignore[invalid-argument-type]
        ),
        TypeError,
        "inputs are .'x1', 'x2'.",
    ),
    # Configuration
    ("no-budget", lambda: hm.OptaxTrainingConfig(), ValueError, "Supply steps="),
    ("lr-without-steps", lambda: hm.OptaxTrainingConfig(lr=0.1), ValueError, "need steps="),
    (
        "lr-with-phases",
        lambda: hm.OptaxTrainingConfig(lr=0.1, phases=(hm.TrainingPhase(1),)),
        ValueError,
        "per TrainingPhase",
    ),
    (
        "tournament-without-steps",
        lambda: hm.OptaxTrainingConfig(steps=1, tournament_attempts=3),
        ValueError,
        "tournament_steps=0",
    ),
    (
        "steps-and-phases",
        lambda: hm.OptaxTrainingConfig(steps=1, phases=(hm.TrainingPhase(1),)),
        ValueError,
        "not both",
    ),
    ("zero-steps", lambda: hm.TrainingPhase(0), ValueError, "at least 1"),
    ("bad-horizon", lambda: hm.TrainingPhase(1, horizon=2.0), ValueError, "horizon"),
    ("zero-segments", lambda: hm.TrainingPhase(1, segments=0), ValueError, "segments"),
    ("float-segments", lambda: hm.TrainingPhase(1, segments=2.5), TypeError, "segments"),  # ty: ignore[invalid-argument-type]
    (
        "negative-continuity",
        lambda: hm.TrainingPhase(1, segments=2, continuity_weight=-1.0),
        ValueError,
        "continuity_weight",
    ),
    (
        "ends-segmented",
        lambda: hm.OptaxTrainingConfig(phases=(hm.TrainingPhase(1, segments=4),)),
        ValueError,
        "single shooting",
    ),
    (
        "optimizer-change-no-reset",
        lambda: hm.OptaxTrainingConfig(
            phases=(hm.TrainingPhase(1), hm.TrainingPhase(1, optimizer="adabelief"))
        ),
        ValueError,
        "reset_optimizer",
    ),
    # Fit
    (
        "one-experiment",
        lambda: unchecked_model().fit(experiment(), hm.OptaxTrainingConfig(steps=1), key=jr.key(0)),
        TypeError,
        "not one Experiment",
    ),
    (
        "no-key",
        lambda: unchecked_model().fit(decay_dataset(), hm.OptaxTrainingConfig(steps=1)),
        TypeError,
        "key",
    ),
    ("none-key", lambda: fit(key=None), TypeError, "must be a JAX PRNG key"),
    ("int-key", lambda: fit(key=0), TypeError, "must be a JAX PRNG key"),
    (
        "ensemble-int-key",
        lambda: parameter_model().fit_ensemble(
            decay_dataset(), hm.OptaxTrainingConfig(steps=1), key=unchecked(0), n_members=1
        ),
        TypeError,
        "must be a JAX PRNG key",
    ),
    (
        "evosax-int-key",
        lambda: parameter_model().fit(
            decay_dataset(), hm.EvosaxTrainingConfig(steps=1), key=unchecked(0)
        ),
        TypeError,
        "must be a JAX PRNG key",
    ),
    (
        "wrong-config-type",
        lambda: unchecked_model().fit(decay_dataset(), {"steps": 1}, key=jr.key(0)),
        TypeError,
        "OptaxTrainingConfig",
    ),
    (
        "unknown-channel-weight",
        lambda: fit(objective=hm.Objective(channel_weights={"zz": 1.0})),
        ValueError,
        "unknown channels",
    ),
    ("unknown-loss", lambda: fit(objective=hm.Objective(loss="rmse")), ValueError, "Unknown loss"),
    ("freeze-typo", lambda: fit(freeze="ratee"), ValueError, "no leaf matches"),
    ("all-frozen", lambda: fit(freeze="rate"), ValueError, "no trainable"),
    (
        "output-shape",
        lambda: fit(
            hm.HybridModel(
                predictors=parameter_predictors(),
                simulate_fn=parameter_model().simulate_fn,
                state_to_output=bad_output,
                output_channel_names=("conc",),
            )
        ),
        ValueError,
        "state_to_output returned shape",
    ),
    (
        "model-channels",
        lambda: hm.HybridModel(
            predictors=parameter_predictors(),
            simulate_fn=parameter_model().simulate_fn,
            state_to_output=identity,
            output_channel_names="conc",
        ),
        TypeError,
        "not a string",
    ),
    # Ensembles
    (
        "ensemble-evosax",
        lambda: unchecked_model().fit_ensemble(
            decay_dataset(), hm.EvosaxTrainingConfig(steps=1), key=jr.key(0), n_members=1
        ),
        TypeError,
        "Optax only",
    ),
    (
        "ensemble-members",
        lambda: parameter_model().fit_ensemble(
            decay_dataset(), hm.OptaxTrainingConfig(steps=1), key=jr.key(0), n_members=3, n_seeds=2
        ),
        ValueError,
        "exceeds n_seeds",
    ),
    # Prediction and persistence
    (
        "two-grids",
        lambda: parameter_model().predict(decay_dataset(), ts=TS, n_points=3),
        ValueError,
        "not both",
    ),
    (
        "predict-one-experiment",
        lambda: unchecked_model().predict(experiment()),
        TypeError,
        "not one Experiment",
    ),
    (
        "load-missing",
        lambda: hm.load("/nonexistent/run", like=parameter_model()),
        FileNotFoundError,
        "metadata.json",
    ),
]


@pytest.mark.parametrize(
    "call, error, match",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_rejected(call: Callable[[], Any], error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        call()


def test_both_kinds_of_jax_key_are_accepted() -> None:
    check_key(jr.key(0), "test")
    check_key(jr.PRNGKey(0), "test")


def test_a_lone_tournament_budget_warns() -> None:
    # A legacy key too, to show one reaches a real fit.
    config = hm.OptaxTrainingConfig(steps=1, tournament_steps=2)
    with pytest.warns(RuntimeWarning, match="tournament_attempts=1"):
        parameter_model().fit(decay_dataset(), config, key=jr.PRNGKey(0))


def _missing_covariate(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
    del predictors, solver, y0
    return jnp.zeros((ts.shape[0], 1)) * covariates["pressure"]


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.fit(decay_dataset(), hm.OptaxTrainingConfig(steps=1), key=jr.key(0)),
        lambda m: m.predict(decay_dataset()),
        lambda m: m.simulate(ts=TS, y0=[1.0], covariates={"temp": 1.0}),
    ],
    ids=["fit", "predict", "simulate"],
)
def test_an_error_in_user_code_lists_the_names_that_exist(call: Any) -> None:
    model = hm.HybridModel(
        predictors=parameter_predictors(),
        simulate_fn=_missing_covariate,
        state_to_output=identity,
        output_channel_names=("conc",),
    )
    with pytest.raises(KeyError, match="pressure") as caught:
        call(model)
    note = "\n".join(caught.value.__notes__)
    assert "['rate', 'offset']" in note or "['offset', 'rate']" in note
    assert "['temp']" in note and "(1,)" in note


def test_a_missing_pandas_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    prediction = parameter_model().predict(decay_dataset())
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=r"jax-hybridmodels\[pandas\]"):
        prediction.to_dataframe()


# ---------------------------------------------------------------------------
# Configuration, schedules, profiles and solvers
# ---------------------------------------------------------------------------


_TSIT5 = diffrax.Tsit5()


def _solver(
    *,
    solver: diffrax.AbstractSolver = _TSIT5,
    rtol: float = 1e-4,
    atol: float | tuple[float, ...] = 1e-6,
    max_steps: int = 100,
    dt0: float | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
) -> hm.SolverConfig:
    return hm.SolverConfig(
        solver=solver,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        dt0=dt0,
        adjoint=adjoint,
        pcoeff=pcoeff,
        icoeff=icoeff,
    )


def _trajectory_penalty(state: Array, bp: object) -> Array:
    del bp
    return jnp.sum(state)


class _Unregistered(diffrax.Heun):
    pass


class _UnregisteredAdjoint(diffrax.DirectAdjoint):
    pass


_NAN = float("nan")


_INF = float("inf")


INVALID = [
    # TrainingPhase
    pytest.param(lambda: TrainingPhase(0), "steps", id="phase-steps-0"),
    pytest.param(lambda: TrainingPhase(-1), "steps", id="phase-steps-negative"),
    pytest.param(lambda: TrainingPhase(2, lr=-1.0), "lr", id="phase-lr-negative"),
    pytest.param(lambda: TrainingPhase(2, lr=_NAN), "lr", id="phase-lr-nan"),
    # A ready-made transformation owns its learning rate, so a schedule
    # paired with one would be accepted and then ignored.
    pytest.param(
        lambda: TrainingPhase(2, lr=optax.constant_schedule(1e-3), optimizer=optax.sgd(1e-3)),
        "owns its learning rate",
        id="phase-schedule-with-raw-optimizer",
    ),
    # The horizon indexes a prefix of the observation times: zero scores
    # nothing, above one claims more timestamps than exist.
    pytest.param(lambda: TrainingPhase(2, horizon=0.0), "horizon", id="phase-horizon-0"),
    pytest.param(lambda: TrainingPhase(2, horizon=-0.1), "horizon", id="phase-horizon-neg"),
    pytest.param(lambda: TrainingPhase(2, horizon=1.5), "horizon", id="phase-horizon-1.5"),
    pytest.param(lambda: TrainingPhase(2, penalty_weight=-1.0), "non-negative", id="phase-pw-neg"),
    pytest.param(lambda: TrainingPhase(2, penalty_weight=_NAN), "finite", id="phase-pw-nan"),
    pytest.param(lambda: TrainingPhase(2, penalty_weight=_INF), "finite", id="phase-pw-inf"),
    # OptaxTrainingConfig: the phase schedule. Zero phases would run nothing
    # and return an empty history with no hint why.
    pytest.param(lambda: OptaxTrainingConfig(), "steps= for one phase", id="optax-no-phases"),
    pytest.param(lambda: OptaxTrainingConfig(phases=()), "steps= for one phase", id="optax-empty"),
    pytest.param(
        lambda: OptaxTrainingConfig(steps=2, phases=(TrainingPhase(3),)),
        "steps= or phases=",
        id="optax-steps-and-phases",
    ),
    pytest.param(
        lambda: OptaxTrainingConfig(phases=(TrainingPhase(2),), lr=1e-2),
        "per TrainingPhase",
        id="optax-phases-with-shorthand-lr",
    ),
    pytest.param(lambda: OptaxTrainingConfig(steps=0), "steps", id="optax-steps-0"),
    # Optimiser state belongs to the optimiser that built it; switching
    # without a reset used to be accepted and silently keep the old one.
    pytest.param(
        lambda: OptaxTrainingConfig(
            phases=(TrainingPhase(2), TrainingPhase(2, optimizer="adabelief"))
        ),
        "reset_optimizer",
        id="optax-switch-without-reset",
    ),
    # Objective: what is minimised.
    pytest.param(lambda: Objective(penalty_weight=-1.0), "non-negative", id="objective-pw-neg"),
    pytest.param(lambda: Objective(penalty_weight=_NAN), "finite", id="objective-pw-nan"),
    pytest.param(
        lambda: Objective(trajectory_penalty_weight=-1.0), "non-negative", id="objective-traj-neg"
    ),
    pytest.param(
        lambda: Objective(
            trajectory_penalty_fn=_trajectory_penalty, trajectory_penalty_weight=_INF
        ),
        "finite",
        id="objective-traj-inf",
    ),
    pytest.param(
        lambda: Objective(trajectory_penalty_weight=1.0),
        "trajectory_penalty_fn is None",
        id="objective-traj-weight-without-fn",
    ),
    # Penalty points are [G, n_inputs] physical input vectors.
    pytest.param(
        lambda: Objective(penalty_points={"rate": jnp.zeros(3)}), "rank-2", id="points-rank-1"
    ),
    pytest.param(
        lambda: Objective(penalty_points={"rate": jnp.zeros((2, 2, 1))}),
        "rank-2",
        id="points-rank-3",
    ),
    pytest.param(
        lambda: Objective(channel_weights={"a": -1.0}), "non-negative", id="weights-negative"
    ),
    pytest.param(lambda: Objective(channel_weights=(1.0, _NAN)), "finite", id="weights-nan"),
    pytest.param(lambda: Objective(channel_weights={}), "at least one", id="weights-empty"),
    pytest.param(lambda: Objective(channel_weights="conc"), "mapping", id="weights-string"),  # ty: ignore[invalid-argument-type]
    # Channel names are checked against the dataset's when the loss resolves.
    pytest.param(
        lambda: Objective(channel_weights={"nope": 1.0}).loss_fn(("a", "b")),
        "unknown channels",
        id="weights-unknown-name",
    ),
    pytest.param(
        lambda: Objective(channel_weights=(1.0,)).loss_fn(("a", "b")),
        "one weight per channel",
        id="weights-wrong-length",
    ),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, tournament_attempts=0), "attempts"),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, tournament_steps=-1), "tournament_steps"),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, tournament_lr=_NAN), "tournament_lr"),
    pytest.param(lambda: OptaxTrainingConfig(steps=2, patience=-1), "patience"),
    # EvosaxTrainingConfig
    pytest.param(lambda: EvosaxTrainingConfig(algorithm="nope"), "available", id="evo-algorithm"),
    pytest.param(lambda: EvosaxTrainingConfig(init="nope"), "available", id="evo-init"),  # ty: ignore[invalid-argument-type]
    pytest.param(lambda: EvosaxTrainingConfig(population_size=0), "population_size"),
    pytest.param(lambda: EvosaxTrainingConfig(steps=0), "steps", id="evo-steps-0"),
    pytest.param(lambda: EvosaxTrainingConfig(init_box_extent=-1.0), "init_box_extent"),
    pytest.param(lambda: EvosaxTrainingConfig(sigma_init=0.0), "sigma_init"),
    pytest.param(lambda: EvosaxTrainingConfig(log_every=0), "log_every"),
    # SolverConfig
    pytest.param(lambda: _solver(rtol=0.0), "rtol", id="solver-rtol-0"),
    pytest.param(lambda: _solver(rtol=-1.0), "rtol", id="solver-rtol-neg"),
    pytest.param(lambda: _solver(atol=0.0), "atol", id="solver-atol-0"),
    pytest.param(lambda: _solver(atol=(1e-6, 0.0)), "atol", id="solver-atol-tuple"),
    pytest.param(lambda: _solver(max_steps=0), "max_steps", id="solver-max-steps"),
    pytest.param(lambda: _solver(dt0=0.0), "dt0", id="solver-dt0-0"),
    pytest.param(lambda: _solver(dt0=-0.1), "dt0", id="solver-dt0-neg"),
    pytest.param(
        lambda: hm.SolverConfig.from_dict(
            {"solver": "DoesNotExist", "rtol": 1e-4, "atol": 1e-5, "max_steps": 10, "dt0": None}
        ),
        "solver",
        id="solver-from-dict-unknown",
    ),
    pytest.param(
        lambda: _solver(solver=_Unregistered()).to_dict(),
        "SOLVER_REGISTRY",
        id="solver-unregistered",
    ),
    pytest.param(
        lambda: _solver(adjoint=_UnregisteredAdjoint()).to_dict(),
        "ADJOINT_REGISTRY",
        id="adjoint-unregistered",
    ),
    # Schedules and profiles
    pytest.param(lambda: annealing_schedule("cosine", total_epochs=0), "total_epochs"),
    pytest.param(
        lambda: annealing_schedule("warmup_cosine", total_epochs=10, warmup_epochs=10), "warmup"
    ),
    pytest.param(lambda: annealing_schedule("cosine", total_epochs=10, init_value=0.0), "init"),
    pytest.param(lambda: annealing_schedule("cosine", total_epochs=10, end_value=1.5), "end"),
    pytest.param(lambda: annealing_schedule("nope", total_epochs=10), "kind", id="schedule-kind"),
    pytest.param(lambda: ramp_profile(t0=3.0, t1=3.0, v0=0.0, v1=1.0), "t1", id="ramp-t1"),
    pytest.param(lambda: piecewise_linear_profile(((0.0, 1.0),)), "at least two", id="pw-one"),
    pytest.param(
        lambda: piecewise_linear_profile(((0.0, 1.0), (1.0, 2.0), (0.5, 3.0))),
        "strictly increasing",
        id="pw-decreasing",
    ),
]


@pytest.mark.parametrize("build, match", INVALID)
def test_invalid_configuration_raises(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_configs_reject_stray_positional_arguments():
    """Only ``TrainingPhase.steps`` may be positional."""
    assert hm.TrainingPhase(5, lr=1e-3).steps == 5
    with pytest.raises(TypeError):
        hm.TrainingPhase(5, 1e-3)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError):
        hm.OptaxTrainingConfig(5)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError):
        hm.EvosaxTrainingConfig("CMA_ES")  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("config", [hm.OptaxTrainingConfig, hm.EvosaxTrainingConfig])
def test_the_display_switch_is_not_a_config_field(config):
    """``verbose`` belongs to ``fit``, so old code fails loudly and not silently."""
    with pytest.raises(TypeError, match="verbose"):
        config(verbose=False)


def test_objective_names_the_mapping_form_for_old_tuple_points():
    """The removed tuple form of ``penalty_points`` raises a TypeError saying what to pass."""
    with pytest.raises(TypeError, match="mapping from predictor path"):
        Objective(penalty_points=(jnp.zeros((3, 1)),))  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Data, splits, metrics and losses
# ---------------------------------------------------------------------------


def _obs(ts, values, **kwargs) -> hm.ChannelObs:
    return hm.ChannelObs(ts=jnp.asarray(ts), values=jnp.asarray(values), **kwargs)


def _exp(exp_id: str = "e", covariates: Any = None, **channels) -> hm.Experiment:
    """One experiment with a single-state ``y0``; channels as ``name=(ts, values)``."""
    return hm.make_experiment(
        channels={name: _obs(*spec) for name, spec in channels.items()},
        covariates={"a": 1.0} if covariates is None else covariates,
        y0=[0.0],
        exp_id=exp_id,
    )


def _two_channel() -> Dataset:
    """Two experiments in one bucket: ``x`` dense, ``y`` sparse in the second."""
    full = _exp("full", x=([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]), y=([0.0, 1.0, 2.0], [5.0, 6.0, 7.0]))
    sparse = _exp("sparse", x=([0.0, 1.0, 2.0], [2.0, 1.0, 0.5]), y=([1.0], [4.0]))
    return hm.make_dataset([full, sparse], output_channel_names=("x", "y"))


@pytest.mark.parametrize(
    "build, match",
    [
        pytest.param(lambda: _obs([0.0, 0.0], [1.0, 2.0]), "repeat", id="duplicate-ts"),
        pytest.param(lambda: _obs([0.0], [1.0], variance=0.0), "variance", id="zero-variance"),
        pytest.param(lambda: _obs([0.0], [jnp.nan]), "finite", id="nan-value"),
        pytest.param(lambda: _obs([0.0], [jnp.inf]), "finite", id="inf-value"),
        pytest.param(lambda: _obs([jnp.inf], [1.0]), "finite", id="inf-ts"),
        pytest.param(lambda: _obs([0.0, 1.0], [1.0]), "same length", id="length-mismatch"),
        pytest.param(lambda: _obs([[0.0]], [[1.0]]), "rank-1", id="rank-2"),
        pytest.param(
            lambda: _obs([0.0, 1.0], [1.0, 2.0], variance=jnp.ones(3)),
            "variance",
            id="variance-length",
        ),
    ],
)
def test_channel_obs_rejects_bad_observations(build, match):
    with pytest.raises(ValueError, match=match):
        build()


_X = {"x": _obs([1.0], [1.0])}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"covariates": {"a": jnp.zeros((2, 2))}, "y0": [0.0]}, "rank-1", id="cov"),
        pytest.param({}, "exactly one", id="no-y0"),
        pytest.param({"y0": [0.0], "y0_fn": lambda c, ch: jnp.zeros(1)}, "exactly one", id="both"),
        pytest.param({"y0": [[0.0]]}, "y0", id="y0-rank-2"),
        pytest.param({"y0_fn": lambda c, ch: jnp.zeros((1, 1))}, "y0", id="y0_fn-rank-2"),
        pytest.param({"y0": [0.0], "t0": float("nan")}, "t0", id="t0-nan"),
    ],
)
def test_make_experiment_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hm.make_experiment(channels=_X, **kwargs)


@pytest.mark.parametrize(
    "experiments, names, match",
    [
        pytest.param([], ("x",), "at least one experiment", id="empty"),
        pytest.param([_exp(x=([0.0], [1.0]))], (), "output channel", id="no-channels"),
        pytest.param([_exp(x=([0.0], [1.0]))], ("x", "x"), "duplicates", id="dup-names"),
        pytest.param([_exp(x=([0.0], [1.0]))], ("x", "y"), "missing", id="missing-channel"),
        pytest.param(
            [_exp("a", x=([0.0], [1.0])), _exp("b", {"b": 1.0}, x=([0.0], [1.0]))],
            ("x",),
            "covariate keys",
            id="covariate-keys",
        ),
        pytest.param(
            [
                _exp("a", {"f": jnp.zeros(2)}, x=([0.0], [1.0])),
                _exp("b", {"f": jnp.zeros(3)}, x=([0.0], [1.0])),
            ],
            ("x",),
            "shape",
            id="covariate-shape",
        ),
        pytest.param([_exp(x=([], []))], ("x",), "no timestamps", id="no-timestamps"),
        pytest.param(
            [
                hm.make_experiment(
                    channels={"x": _obs([1.0], [1.0])}, y0=[0.0], t0=2.0, exp_id="late"
                )
            ],
            ("x",),
            "t0 must precede",
            id="t0-after-data",
        ),
    ],
)
def test_make_dataset_rejects_inconsistent_experiments(experiments, names, match):
    with pytest.raises(ValueError, match=match):
        hm.make_dataset(experiments, output_channel_names=names)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"values": np.ones((3, 1))}, r"shape \[T, D\]", id="values-shape"),
        pytest.param({"variance": (1.0, 1.0, 1.0)}, "one entry per channel", id="variance-len"),
        pytest.param({"variance": np.ones((3, 2))}, "variance must be", id="variance-shape"),
        pytest.param({"variance": 0.0}, "channel 'x'.*variance", id="bad-variance"),
    ],
)
def test_experiment_from_array_rejects_bad_shapes(kwargs, match):
    base: dict[str, Any] = {
        "ts": [0.0, 1.0],
        "values": np.ones((2, 2)),
        "channel_names": ("x", "y"),
    }
    with pytest.raises(ValueError, match=match):
        experiment_from_array(**{**base, **kwargs}, y0=[0.0], exp_id="e")


def _lab_table() -> pd.DataFrame:
    """Two runs in the shape of a lab spreadsheet: one row per sample time."""
    return pd.DataFrame(
        {
            "Exp": [7, 7, 7, 3, 3],
            "Time": [0.0, 30.0, 60.0, 0.0, 45.0],
            "Temp": [21.0, 21.0, 21.0, 25.0, 25.0],
            "Conc": [19.1, 18.4, 13.5, 12.3, 11.9],
            "Conc_var": [0.03, 0.16, 2.4, 0.06, 0.07],
            "PS": [np.nan, np.nan, 13.0, np.nan, 15.0],
        }
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        pytest.param({"channels": ["Nope"]}, KeyError, "Nope", id="missing-column"),
        pytest.param(
            {"channels": ["Conc"], "covariates": ["Conc"]},
            ValueError,
            "constant within experiment '7'",
            id="varying-covariate",
        ),
        pytest.param(
            {"channels": ["Conc"], "variance": {"other": 1.0}},
            ValueError,
            "unknown channels",
            id="variance-key",
        ),
    ],
)
def test_experiments_from_dataframe_rejects_bad_tables(kwargs, error, match):
    with pytest.raises(error, match=match):
        experiments_from_dataframe(_lab_table(), time="Time", experiment="Exp", y0=[0.0], **kwargs)


def _numbered(n: int) -> Dataset:
    """``n`` experiments with grids of 2, 3 and 4 points in rotation."""
    experiments = []
    for i in range(n):
        ts = jnp.linspace(0.0, 1.0, 2 + i % 3)
        experiments.append(_exp(f"e{i}", {"a": float(i)}, c=(ts, ts * i)))
    return hm.make_dataset(experiments, output_channel_names=("c",))


@pytest.mark.parametrize(
    "dataset, fractions, match",
    [
        pytest.param(_numbered(10), {"train": 0.5, "val": 0.2, "test": 0.2}, "sum", id="sum"),
        pytest.param(_numbered(10), {"train": 0.5, "val": -0.2, "test": 0.7}, r"\[0, 1\]"),
        pytest.param(_numbered(10), {"train": 1.2, "val": 0.0, "test": -0.2}, r"\[0, 1\]"),
        # A positive fraction that floors to zero experiments would silently
        # drop a split the caller asked for.
        pytest.param(_numbered(7), {}, "val fraction", id="n7-val-empty"),
        pytest.param(_numbered(3), {}, "val fraction", id="n3-val-empty"),
        pytest.param(
            Dataset(bucket_payloads=(), output_channel_names=("c",), covariate_names=()),
            {},
            "experiments",
            id="manual-dataset",
        ),
    ],
)
def test_split_rejects_bad_fractions(dataset, fractions, match):
    with pytest.raises(ValueError, match=match):
        hm.split_dataset(dataset, key=jr.key(0), **fractions)


@pytest.mark.parametrize(
    "dataset, n, match",
    [
        pytest.param(_numbered(3), 0, "n_experiments", id="zero"),
        pytest.param(
            Dataset(bucket_payloads=(), output_channel_names=("c",), covariate_names=()),
            None,
            "source experiments",
            id="manual-dataset",
        ),
    ],
)
def test_bootstrap_rejects_bad_inputs(dataset, n, match):
    with pytest.raises(ValueError, match=match):
        hm.make_bootstrap_dataset(dataset, key=jr.key(0), n_experiments=n)


def test_metrics_refuse_results_without_observations():
    bare = hm.PredictionResult(
        {"a": hm.ExperimentPrediction(times=jnp.zeros(1), outputs={"x": jnp.zeros(1)})}
    )
    with pytest.raises(ValueError, match="observed values"):
        hm.compute_metrics(bare)
    with pytest.raises(ValueError, match="at least one"):
        hm.compute_metrics(hm.PredictionResult({}))


def _two_channel_scored() -> tuple[jax.Array, BucketPayload]:
    bp = _two_channel().bucket_payloads[0]
    offsets = jnp.arange(bp.y_observed.size, dtype=jnp.float32).reshape(bp.y_observed.shape)
    return bp.y_observed + 0.1 * offsets, bp


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(lambda: resolve_loss_fn("rmse", None, None), "bal_mle", id="unknown-name"),
        # Weights must act inside the loss's own reduction, which an opaque
        # (pred_obs, bp) callable cannot do.
        pytest.param(
            lambda: resolve_loss_fn(lambda p, bp: p.sum(), None, (0.5, 0.5)),
            "channel_weights",
            id="weights-on-plain-callable",
        ),
        pytest.param(
            lambda: masked_mse(*_two_channel_scored(), channel_idx=(2,)),
            "channel_idx",
            id="idx-out-of-range",
        ),
        pytest.param(
            lambda: masked_mse(*_two_channel_scored(), channel_idx=(-1,)),
            "channel_idx",
            id="idx-negative",
        ),
        pytest.param(
            lambda: resolve_loss_fn(lambda p, bp: p.sum(), (9,), None)(*_two_channel_scored()),
            "channel_idx",
            id="idx-out-of-range-plain",
        ),
    ],
)
def test_loss_configuration_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "rows, match",
    [
        pytest.param({"value": [1.0, np.nan]}, "'value' has 1 missing", id="missing-value"),
        pytest.param({"time": [1.0, 1.0]}, "more than one row at time 1.0", id="repeated-cell"),
        pytest.param({"run": ["a", None]}, "missing experiment ID", id="missing-id"),
    ],
)
def test_experiments_from_long_dataframe_rejects_bad_rows(rows, match):
    table = {"run": ["a", "a"], "time": [1.0, 2.0], "channel": ["x", "x"], "value": [1.0, 2.0]}
    with pytest.raises(ValueError, match=match):
        hm.experiments_from_long_dataframe(
            pd.DataFrame({**table, **rows}),
            time="time",
            experiment="run",
            channel="channel",
            value="value",
            y0=[0.0],
        )


def test_experiments_from_dataframe_rejects_missing_ids():
    """A row with no experiment ID raises instead of vanishing in groupby."""
    df = pd.DataFrame(
        {"run": [1, 1, 2, np.nan], "time": [0.0, 1.0, 0.0, 1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.raises(ValueError, match="missing experiment ID"):
        hm.experiments_from_dataframe(df, time="time", experiment="run", channels=["x"], y0=[0.0])


# ---------------------------------------------------------------------------
# Scalers and parameters
# ---------------------------------------------------------------------------


def test_positive_warp_rejects_nonpositive_runtime_inputs():
    scaler = hm.BoundScaler(((1e-3, 1.0),), warp="log10")
    with pytest.raises(ValueError, match="positive"):
        scaler.to_latent(jnp.array([0.0]))
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive"):
        eqx.filter_jit(scaler.to_latent)(jnp.array([0.0]))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(bounds=((0.0, 1.0),), transform="tanh"), "algebraic"),
        (dict(bounds=((0.0, 1.0),), warp="nope"), "linear"),
        (dict(bounds=((math.nan, 1.0),)), "NaN"),
        (dict(bounds=((10.0, 0.0),)), "low < high"),
        (dict(bounds=((5.0, 5.0),)), "low < high"),
        (dict(bounds=((0.0, 10.0),), warp="log10"), "undefined at or below zero"),
        (dict(bounds=((None, None),), warp="log10"), "Unbounded values require"),
        (dict(bounds=((0.0, 1.0),), temperature=0.0), "temperature"),
        (dict(bounds=((0.0, 1.0),), temperature=math.nan), "temperature"),
        (dict(bounds=((0.0, 1.0),), temperature=jnp.array([1.0, 2.0])), "temperature"),
        (dict(bounds=((0.0, 1.0),), logit_eps=0.5), "logit_eps"),
        (dict(bounds=((0.0, 1.0),), logit_eps=0.0), "logit_eps"),
        (dict(bounds=((0.0, 1.0),), z_knee=-1.0), "z_knee"),
        (dict(bounds=((0.0, 1.0, 2.0),)), "one \\(low, high\\) pair"),
        (dict(bounds=(0.0, 1.0, 2.0)), "one \\(low, high\\) pair"),
    ],
)
def test_scaler_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hm.BoundScaler(**kwargs)


@pytest.mark.parametrize(
    "value,bounds,match",
    [
        (math.nan, None, "finite"),
        ([], None, "nonempty"),
        (0.0, (0.0, 1.0), "strictly inside"),
        (2.0, (None, 1.0), "strictly inside"),
        # 0.1 in float32 rounds onto the bound 0.1, where the latent is infinite.
        (0.1, (0.1, 0.2), "strictly inside"),
        ([0.1, 0.2], [(0.0, 1.0)] * 3, "Expected 2 bound pairs"),
    ],
)
def test_parameter_validation(value, bounds, match):
    with pytest.raises(ValueError, match=match):
        hm.Parameter(value, bounds=bounds)


# ---------------------------------------------------------------------------
# Predictors and masks
# ---------------------------------------------------------------------------

IN, OUT = 2, 1


def mlp(key: Array) -> MLPPredictor:
    return MLPPredictor(in_size=IN, out_size=OUT, width_size=4, depth=2, key=key)


@pytest.mark.parametrize(
    "build,match",
    [
        (
            lambda: MLPPredictor(
                in_size=1, out_size=1, width_size=2, depth=1, activation_name="nope", key=jr.key(0)
            ),
            "not supported",
        ),
        (
            lambda: KANPredictor(
                in_size=IN,
                out_size=OUT,
                hidden_widths=(3,),
                grid_size=3,
                basis="nope",
                key=jr.key(0),
            ),
            "not supported",
        ),
        (
            lambda: NeuralNPolynomial(
                coeff_net=MLPPredictor(in_size=3, out_size=5, width_size=2, depth=1, key=jr.key(0)),
                exponents=(0.0, 1.0, 2.0),
                in_size=3,
                out_size=2,
            ),
            "out_size",
        ),
        (
            lambda: NeuralNPolynomial(
                coeff_net=mlp(jr.key(0)), exponents=(-1.0,), in_size=IN, out_size=OUT
            ),
            "exponents",
        ),
        (
            lambda: NeuralNPolynomial(
                coeff_net=mlp(jr.key(0)), exponents=(0.5,), in_size=IN, out_size=OUT
            ),
            "exponents",
        ),
        (lambda: BoundedPredictor(inner=mlp(jr.key(0)), input_keys=("a",)), "input dimension"),
        (
            lambda: BoundedPredictor(inner=mlp(jr.key(0)), output_bounds=((0, 1), (0, 1))),
            "output dimension",
        ),
        (lambda: BoundedPredictor(inner=mlp(jr.key(0)), output_warp="log"), "needs output_bounds"),
        (
            lambda: BoundedPredictor(
                inner=mlp(jr.key(0)),
                in_scaler=hm.BoundScaler(((1.0, 2.0),) * IN),
                input_warp="log",
            ),
            "needs input_bounds",
        ),
        (
            lambda: BoundedPredictor(
                inner=mlp(jr.key(0)), output_bounds=(-1.0, 1.0), output_warp="log"
            ),
            "positive",
        ),
        (
            lambda: BoundedPredictor(
                inner=mlp(jr.key(0)), input_keys=("a",), input_bounds=((0, 1), (0, 1))
            ),
            "input_keys length",
        ),
        (
            lambda: BoundedPredictor(inner=mlp(jr.key(0)), input_bounds=(), output_bounds=(0, 1)),
            "at least one input",
        ),
        (
            lambda: BoundedPredictor(
                inner=mlp(jr.key(0)),
                in_scaler=hm.BoundScaler(((0, 1), (0, 1))),
                input_bounds=(0, 1),
            ),
            "not both",
        ),
    ],
)
def test_construction_errors(build, match):
    with pytest.raises(ValueError, match=match):
        build()


class Identity(hm.Predictor):
    """Latent in, latent out, so the bounded output equals the input."""

    def __call__(self, x: Array) -> Array:
        return x


def identity_bounded(keys: tuple[str, ...] | None = ("a", "b")) -> BoundedPredictor:
    return BoundedPredictor(
        inner=Identity(),
        input_keys=keys,
        input_bounds=((0.0, 1.0),) * 2,
        output_bounds=((0.0, 1.0),) * 2,
    )


@pytest.mark.parametrize(
    "inputs,error",
    [
        ({"a": 0.4}, KeyError),
        (jnp.array([0.4, 0.7, 0.1]), eqx.EquinoxTracetimeError),
        (jnp.array([[0.4, 0.7]]), eqx.EquinoxTracetimeError),
    ],
    ids=["missing-key", "wrong-length", "wrong-rank"],
)
def test_bounded_predictor_rejects_malformed_inputs(inputs, error):
    with pytest.raises(error):
        identity_bounded()(inputs)


def _model_predictors() -> dict[str, eqx.Module]:
    return {
        "rate": BoundedPredictor(inner=mlp(jr.key(0)), input_keys=("a", "b"), output_bounds=(0, 1)),
        "offset": hm.Parameter(0.5, bounds=(0.0, 1.0)),
        "gain": hm.Parameter(2.0, bounds=(0.0, None)),
        "fixed": hm.Constant(1.0),
    }


@pytest.mark.parametrize(
    "freeze, error, match",
    [
        (("rate.inner.nope",), ValueError, "no leaf matches"),
        ((KANPredictor,), ValueError, "no leaf matches"),
        ((lambda m: False,), ValueError, "no leaf matches"),
        ((3,), TypeError, "freeze entries"),
    ],
    ids=["path", "class", "predicate", "type"],
)
def test_freeze_mask_rejects_entries_that_match_nothing(freeze, error, match):
    with pytest.raises(error, match=match):
        freeze_mask(_model_predictors(), freeze)


def test_bounded_predictor_names_missing_inputs():
    """A misspelled input key lists the keys the predictor needs."""
    inner = hm.MLPPredictor(
        in_size=1, out_size=1, width_size=4, depth=1, activation_name="tanh", key=jr.key(0)
    )
    rate = hm.BoundedPredictor(
        inner=inner, input_keys=("temp",), input_bounds=(0.0, 2.0), output_bounds=(0.05, 3.0)
    )
    with pytest.raises(KeyError, match=r"missing \['temp'\], got \['T'\]"):
        rate({"T": 1.0})


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def simulate_pressure(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """``simulate_network`` feeding ``temp`` to a rate network that calls it ``pressure``."""
    return simulate_network(predictors, ts, {"pressure": covariates["temp"]}, y0, solver)


def test_an_uncovered_leaf_raises_before_training(temp_data):
    # input_keys that match no covariate leave the leaf with no measured
    # points. With the penalty on, that must fail loudly, not skip the leaf.
    predictors = network_predictors()
    inner = predictors["rate"].inner
    predictors["rate"] = hm.BoundedPredictor(
        inner=inner, input_keys=("pressure",), input_bounds=(0.0, 2.0), output_bounds=(0.05, 3.0)
    )
    with pytest.raises(ValueError, match="'rate'.*trajectory_penalty_fn"):
        train_with_optax(
            make_model(predictors, simulate_pressure),
            temp_data,
            hm.OptaxTrainingConfig(steps=2),
            objective=Objective(penalty_weight=1e-2),
            key=jr.key(0),
        )
    # A phase override switches the penalty on for that phase alone, and
    # the coverage check follows it.
    with pytest.raises(ValueError, match="no penalty points"):
        train_with_optax(
            make_model(predictors, simulate_pressure),
            temp_data,
            hm.OptaxTrainingConfig(
                phases=(TrainingPhase(1), TrainingPhase(1, penalty_weight=1e-2))
            ),
            key=jr.key(0),
        )


@pytest.mark.parametrize(
    "points,match",
    [
        ({"rates": jnp.zeros((2, 1))}, r"valid paths are \['rate'\]"),
        ({"rate": jnp.zeros((2, 3))}, "input columns"),
    ],
)
def test_malformed_penalty_points_raise(temp_data, points, match):
    objective = Objective(penalty_weight=1e-2, penalty_points=points)
    with pytest.raises(ValueError, match=match):
        train_with_optax(
            make_model(network_predictors(), simulate_network),
            temp_data,
            hm.OptaxTrainingConfig(steps=2),
            objective=objective,
            key=jr.key(0),
        )


# ---------------------------------------------------------------------------
# The model API and persistence
# ---------------------------------------------------------------------------


def _metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metadata.json").read_text())


def _edit_metadata(run_dir: Path, edit) -> None:
    metadata = _metadata(run_dir)
    edit(metadata)
    (run_dir / "metadata.json").write_text(json.dumps(metadata))


def test_save_rejects_what_it_cannot_write(tmp_path):
    """Only a model or a fit saves, and only a training config goes in config=."""
    with pytest.raises(TypeError, match="HybridModel or FitResult"):
        hm.save(tmp_path, parameter_predictors())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="OptaxTrainingConfig or EvosaxTrainingConfig"):
        hm.save(tmp_path, parameter_model(), config=Objective())  # ty: ignore[invalid-argument-type]


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
    hm.save(tmp_path, make_model(saved()), config=hm.OptaxTrainingConfig(steps=1))
    if edit is not None:
        _edit_metadata(tmp_path, edit)
    with pytest.raises(ValueError, match=match):
        hm.load(tmp_path, like=make_model(template()))


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
            lambda: _FROZEN.fit(decay_dataset(), hm.OptaxTrainingConfig(steps=2), key=jr.key(0)),
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
                decay_dataset(),
                hm.OptaxTrainingConfig(steps=1),
                key=jr.key(0),
                freeze="rate",
                trainable=True,
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


def test_metrics_reject_a_bad_grouping(fitted, decay_data):
    predictions = fitted[0].model.predict(decay_data)
    with pytest.raises(ValueError, match="by= must be"):
        hm.compute_metrics(predictions, by="bucket")  # ty: ignore[no-matching-overload]


def test_compute_metrics_names_what_it_takes(fitted, decay_data):
    """Raw bucket arrays are not a result; the error names the entry points that make one."""
    buckets = fitted[0].model.predict(decay_data).buckets
    with pytest.raises(TypeError, match="predict_ensemble"):
        hm.compute_metrics(buckets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
        parameter_model().fit_ensemble(
            decay_data, hm.OptaxTrainingConfig(steps=1), key=jr.key(0), **kwargs
        )


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
