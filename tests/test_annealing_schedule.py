"""Tests for ``annealing_schedule``: epoch-scaled schedule multipliers.

The stock trainers express changing hyperparameters as *phases* (fixed
lr/weight blocks). A custom loop that wants a smooth within-run schedule —
learning-rate decay, a ramping penalty weight — composes this factory:
``annealing_schedule`` returns an optax-style callable
``(step: int) -> float``, normalised to a multiplier in
``[end_value, init_value]``, with the run length baked in as
``total_epochs``. A library *step* is one full pass over every bucket,
i.e. one epoch, so "epochs" and "steps" are synonyms here.

Each test pins one property of the contract:

- the four kinds (cosine, linear, warmup_cosine, exponential) hit their
  endpoint and mid-run values;
- validation refuses degenerate configurations;
- the callable is ``jit``-safe (a traced step index works);
- composition: ``lr = base * schedule(step)`` inside a custom loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hybridmodels.schedules import annealing_schedule


@pytest.mark.parametrize(
    "kind, kwargs, start_value",
    [
        ("cosine", {}, 1.0),
        ("linear", {}, 1.0),
        ("warmup_cosine", {"warmup_epochs": 10}, 0.0),
        ("exponential", {}, 1.0),
    ],
)
def test_all_kinds_are_epoch_scaled_multipliers(kind, kwargs, start_value):
    """The schedule maps an integer step in ``[0, total_epochs]`` to a float
    multiplier. Every kind is a multiplier; warmup_cosine starts at 0 by
    design (its ``init_value`` is the peak)."""
    sched = annealing_schedule(kind, total_epochs=100, **kwargs)
    assert float(sched(0)) == pytest.approx(start_value)


def test_cosine_decays_to_alpha_and_hits_midpoint():
    sched = annealing_schedule("cosine", total_epochs=100)
    # Cosine with alpha=0: halfway is exactly 0.5, the end is ~0.
    assert float(sched(50)) == pytest.approx(0.5, abs=1e-2)
    assert float(sched(100)) == pytest.approx(0.0, abs=1e-3)
    # Monotone decreasing.
    vals = [float(sched(i)) for i in range(0, 101, 10)]
    assert vals == sorted(vals, reverse=True)


def test_cosine_honours_end_value():
    sched = annealing_schedule("cosine", total_epochs=50, end_value=0.1)
    assert float(sched(0)) == pytest.approx(1.0)
    assert float(sched(50)) == pytest.approx(0.1, abs=1e-3)


def test_linear_reaches_end_value_exactly():
    sched = annealing_schedule("linear", total_epochs=40, end_value=0.2)
    assert float(sched(0)) == pytest.approx(1.0)
    assert float(sched(20)) == pytest.approx(0.6)
    assert float(sched(40)) == pytest.approx(0.2)


def test_warmup_cosine_rises_then_decays():
    sched = annealing_schedule("warmup_cosine", total_epochs=100, warmup_epochs=20)
    assert float(sched(0)) == pytest.approx(0.0)
    assert float(sched(20)) == pytest.approx(1.0, abs=1e-2)
    assert float(sched(100)) == pytest.approx(0.0, abs=1e-3)


def test_exponential_decays_geometrically_to_end_value():
    # The per-epoch rate is derived from (init, end, total), so the curve is
    # geometric *and* lands exactly on end_value at the run end.
    sched = annealing_schedule("exponential", total_epochs=10, end_value=0.01)
    assert float(sched(0)) == pytest.approx(1.0)
    ratio_1 = float(sched(2)) / float(sched(1))
    ratio_2 = float(sched(3)) / float(sched(2))
    assert ratio_1 == pytest.approx(ratio_2, abs=1e-3)  # geometric
    assert float(sched(10)) == pytest.approx(0.01, abs=1e-4)
    # The tail stays at the end value instead of decaying forever.
    assert float(sched(50)) == pytest.approx(0.01, abs=1e-4)


def test_step_holds_after_the_run_ends():
    """A schedule evaluated past its budget stays at the end value, so a
    loop that runs a couple of steps beyond ``total_epochs`` cannot blow up
    or turn the multiplier negative."""
    sched = annealing_schedule("linear", total_epochs=10, end_value=0.0)
    assert float(sched(10)) == pytest.approx(0.0, abs=1e-3)
    assert float(sched(50)) == pytest.approx(0.0, abs=1e-3)


def test_validation_rejects_degenerate_configs():
    with pytest.raises(ValueError, match="total_epochs"):
        annealing_schedule("cosine", total_epochs=0)
    with pytest.raises(ValueError, match="warmup"):
        annealing_schedule("warmup_cosine", total_epochs=10, warmup_epochs=10)
    with pytest.raises(ValueError, match="init_value"):
        annealing_schedule("cosine", total_epochs=10, init_value=0.0)
    with pytest.raises(ValueError, match="end_value"):
        annealing_schedule("cosine", total_epochs=10, end_value=1.5)
    with pytest.raises(ValueError, match="kind"):
        annealing_schedule("not_a_kind", total_epochs=10)


def test_schedule_is_jit_safe():
    """A traced step index (the count inside a jitted loop) must work."""
    sched = annealing_schedule("cosine", total_epochs=100)

    @jax.jit
    def lr_at(step):
        return sched(step)

    assert float(lr_at(50)) == pytest.approx(0.5, abs=1e-2)


def test_composes_as_a_multiplier_in_a_custom_loop():
    """The canonical use: ``lr = base_lr * schedule(step)``. The result is
    a plain traced scalar, so it plugs into any optax-style call."""
    base_lr = 1e-2
    sched = annealing_schedule("warmup_cosine", total_epochs=50, warmup_epochs=5)
    lrs = [base_lr * float(sched(s)) for s in (0, 5, 25, 50)]
    assert lrs[0] == pytest.approx(0.0)  # warmup starts at zero
    assert lrs[1] == pytest.approx(base_lr)  # peak exactly at warmup end
    # After the peak the tail is non-increasing down to the end value.
    assert lrs[1:] == sorted(lrs[1:], reverse=True)
    assert lrs[2] < base_lr  # well into the decay tail
    assert jnp.isfinite(jnp.asarray(lrs)).all()