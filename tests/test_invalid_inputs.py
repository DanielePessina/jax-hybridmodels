"""Every way of misusing the public API that the package promises to reject.

One table, one assertion: the call raises the named error, and its message
matches. Each row is a mistake a user can make, so an entry here documents an
error message as much as it tests one. Nothing trains: every call fails before
compiling.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest
from _models import decay_dataset, identity, parameter_model, parameter_predictors, quiet

import jaxhybridmodels as hm

TS = jnp.array([0.5, 1.0, 2.0])


def obs(**overrides: Any) -> hm.ChannelObs:
    return hm.ChannelObs(**{"ts": TS, "values": jnp.exp(-TS), **overrides})


def experiment(**overrides: Any) -> hm.Experiment:
    fields: dict[str, Any] = {"channels": {"conc": obs()}, "y0": [1.0], "exp_id": "x", **overrides}
    return hm.make_experiment(**fields)


def fit(model: hm.HybridModel | None = None, **kwargs: Any) -> Any:
    model = parameter_model() if model is None else model
    fields: dict[str, Any] = {"key": jr.key(0), **kwargs}
    return model.fit(decay_dataset(), quiet(steps=1), **fields)


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
    # Predictors
    ("reversed-bounds", lambda: hm.Parameter(0.5, bounds=(2.0, 1.0)), ValueError, "low < high"),
    ("outside-bounds", lambda: hm.Parameter(5.0, bounds=(0.0, 1.0)), ValueError, "strictly inside"),
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
        lambda: unchecked_model().fit(experiment(), quiet(steps=1), key=jr.key(0)),
        TypeError,
        "not one Experiment",
    ),
    ("no-key", lambda: unchecked_model().fit(decay_dataset(), quiet(steps=1)), TypeError, "key"),
    ("none-key", lambda: fit(key=None), TypeError, "must be a JAX PRNG key"),
    ("int-key", lambda: fit(key=0), TypeError, "must be a JAX PRNG key"),
    (
        "ensemble-int-key",
        lambda: parameter_model().fit_ensemble(
            decay_dataset(), quiet(steps=1), key=unchecked(0), n_members=1
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
            decay_dataset(), quiet(steps=1), key=jr.key(0), n_members=3, n_seeds=2
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
    for key in (jr.key(0), jr.PRNGKey(0)):
        fit(key=key)


def test_a_lone_tournament_budget_warns() -> None:
    config = quiet(steps=1, tournament_steps=2)
    with pytest.warns(RuntimeWarning, match="tournament_attempts=1"):
        parameter_model().fit(decay_dataset(), config, key=jr.key(0))


def _missing_covariate(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
    del predictors, solver, y0
    return jnp.zeros((ts.shape[0], 1)) * covariates["pressure"]


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.fit(decay_dataset(), quiet(steps=1), key=jr.key(0)),
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
