"""Smoke coverage for the PPO batch reactor example.

The example lives in ``examples/`` and is not part of the installed package, so
these tests import it by path. They exist to catch the two ways it can rot
silently: the ``rlax`` dependency disappearing or changing shape, and the
policy split drifting away from ``BoundedPredictor.__call__``.

Deliberately small. The example's own verification checkpoints do the
scientific work; this only asserts that the machinery turns over.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import diffrax
import jax.numpy as jnp
import jax.random as jr
import pytest

from jaxhybridmodels import SolverConfig

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "batch_reactor"


def _load_example():
    """Import ``train_rl_deactivation`` by path, with its own directory importable.

    The script resolves ``_model`` and ``_shared`` as top-level modules because
    it normally runs as ``__main__`` from its own directory. Both paths go on
    ``sys.path`` here to reproduce that.
    """
    for path in (_EXAMPLE_DIR, _EXAMPLE_DIR.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "train_rl_deactivation", _EXAMPLE_DIR / "train_rl_deactivation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing. Equinox modules are dataclasses, and
    # dataclasses resolves string annotations through sys.modules[cls.__module__];
    # a module that is not there yet makes that lookup return None.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


@pytest.fixture(scope="module")
def solver():
    return SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05)


@pytest.fixture(scope="module")
def episodes(example):
    """Two aged experiments against an untrained trunk.

    An untrained trunk is fine here. Nothing in these tests depends on the trunk
    being right, only on the plumbing running, and it removes the dependency on
    the saved artefact.
    """
    train, _ = example._build_aged_datasets(doe_seed=0, noise_key=jr.PRNGKey(0))
    trunk = example.build_predictors(key=jr.PRNGKey(1))
    return example.episodes_from_experiments(train[:2], trunk)


class TestRewardReference:
    """The pre-learning checkpoint the example rests on."""

    def test_the_true_law_beats_doing_nothing(self, example, episodes, solver):
        truth, static = example.reference_returns(episodes, solver)
        assert truth > 2.0 * static, (
            f"the true deactivation law scored {truth:.4f} against {static:.4f} for no "
            "deactivation; if these are close there is nothing for a policy to learn"
        )

    def test_the_interval_mean_is_the_exact_hold_target(self, example):
        """Holding activity at its interval mean reproduces the continuous truth.

        For ``dCa/dt = -k a(t) Ca`` only the integral of ``a`` over the interval
        enters the solution, so the mean is exact and the left endpoint is not.
        This is the fact the reward reference is built on.
        """
        t0, t1, pH = 1.0, 1.5, 5.5
        grid = jnp.linspace(t0, t1, 20_001)
        expected = jnp.trapezoid(example._activity_true(grid, pH), grid) / (t1 - t0)
        assert float(example._activity_interval_mean(t0, t1, pH)) == pytest.approx(
            float(expected), rel=1e-6
        )
        # The left endpoint overestimates on a falling curve, which is exactly
        # why scoring the truth that way understated it.
        assert float(example._activity_true(t0, pH)) > float(expected)


class TestPolicySplit:
    """The thesis: the actor is split at ``out_scaler`` and nowhere else."""

    def test_the_halves_recombine_into_bounded_predictor_call(self, example):
        """``from_latent(_policy_mean(obs))`` must equal ``policy(obs)`` exactly.

        PPO samples in latent space and applies ``out_scaler.from_latent``
        inside the rollout. If that split ever stops matching
        ``BoundedPredictor.__call__``, the trained artefact silently stops being
        the model that was trained.
        """
        agent = example.build_agent(key=jr.PRNGKey(3))
        obs = jnp.array([0.6, 25.0, 6.0])
        via_split = agent.policy.out_scaler.from_latent(example._policy_mean(agent, obs))
        via_call = agent.policy({"Ca": obs[0], "temperature_C": obs[1], "pH": obs[2]})
        assert jnp.allclose(via_split, via_call, atol=0.0, rtol=0.0)

    def test_actions_stay_inside_the_activity_box(self, example, episodes, solver):
        """Bounds are structural, so no sampled latent can escape the box."""
        agent = example.build_agent(key=jr.PRNGKey(4))
        # A deliberately huge spread: with a clip this would pin to the edges,
        # and with a reparameterisation it cannot leave the interior at all.
        agent = agent._replace(log_std=jnp.full((1,), 3.0))
        rollout = example.rollout_batch(
            agent, episodes, jr.PRNGKey(5), solver=solver, penalty_weight=0.0, n_samples=8
        )
        low, high = example.ACTIVITY_BOUNDS[0]
        assert float(jnp.min(rollout.activity)) > low
        assert float(jnp.max(rollout.activity)) < high


class TestPPOTurnsOver:
    def test_a_short_run_improves_the_return(self, example, episodes, solver):
        """A handful of updates must move the return upward.

        Not a convergence test. It fails if ``rlax`` changes signature, if the
        agent stops being a valid scan carry, or if the gradient never reaches
        the policy.
        """
        agent = example.build_agent(key=jr.PRNGKey(6))
        truth, _ = example.reference_returns(episodes, solver)
        _, _, returns, _, losses = example.train_ppo(
            agent,
            episodes,
            solver=solver,
            n_updates=12,
            n_samples=8,
            lr=3e-3,
            penalty_weight=1e-3,
            truth_return=truth,
            val_episodes=episodes,
            key=jr.PRNGKey(7),
        )
        assert all(jnp.isfinite(jnp.asarray(losses))), f"non-finite PPO loss: {losses}"
        assert max(returns) > returns[0], (
            f"the return never improved on its starting value {returns[0]:.4f}; "
            f"best was {max(returns):.4f}"
        )
