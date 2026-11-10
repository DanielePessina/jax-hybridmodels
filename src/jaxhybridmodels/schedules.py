"""Epoch-scaled schedule multipliers.

The stock Optax trainer changes hyperparameters per **phase**, a block of
steps sharing an lr or penalty weight, and within a phase
``TrainingPhase.lr`` also takes any optax schedule, counted from the phase's
first step. :func:`annealing_schedule` returns such a callable
``(step: int) -> float``, normalised to a multiplier with the run length
baked in. Scale it into a learning rate for a phase, or use it to ramp a
penalty weight in a custom loop:

    phase = hm.TrainingPhase(steps=steps, lr=lambda step: 1e-2 * cosine(step))

    bound_penalty_weight = 0.5 * annealing_schedule("linear", total_steps=steps)(step)

where ``cosine = annealing_schedule("cosine", total_steps=steps)``. A
library *step* is one full pass over every bucket, i.e. one epoch
(CONTEXT.md, "step (training)"), so ``total_steps`` is the step budget of
the phase or run it scales.

The name is deliberate: it is an *annealing* schedule, which only
smooths a training hyperparameter. It has nothing to do with a physical
temperature, and is distinct from ``BoundScaler.temperature``, so the
word "temperature" is never used here (it collides with chemistry).
"""
# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import optax
from jax.typing import ArrayLike
from jaxtyping import Array

_SCHEDULE_KINDS: frozenset[str] = frozenset(("cosine", "linear", "warmup_cosine", "exponential"))


def annealing_schedule(
    kind: str = "cosine",
    *,
    total_steps: int,
    init_value: float = 1.0,
    end_value: float = 0.0,
    warmup_steps: int = 0,
) -> Callable[[ArrayLike], Array]:
    """Return ``schedule(step) -> Array``, a scalar multiplier over ``[0, total_steps]``.

    Built on optax's own schedule helpers, so the returned callable is a
    plain pure function of the integer step count: ``jit``-safe, usable
    inside a traced loop. Every kind lands on ``end_value``, up to float
    rounding, at ``step == total_steps`` and stays there afterwards, so the run length
    is genuinely baked in.

    Parameters
    ----------
    kind : str
        One of ``"cosine"`` (default), ``"linear"``, ``"warmup_cosine"``,
        ``"exponential"``. The exponential kind decays geometrically with
        the per-epoch rate *derived* from ``(init_value, end_value,
        total_steps)``, so its end point and run length are honoured like
        every other kind. It needs ``end_value > 0``.
    total_steps : int
        Run length in steps (one step == one epoch). Must be at least 1.
    init_value : float
        Value at ``step=0``, except ``"warmup_cosine"`` where it is the
        **peak** the schedule rises to after ``warmup_steps``. Must be
        positive.
    end_value : float
        Value at ``step=total_steps``, where every kind arrives. Must lie
        in ``[0, init_value]``.
    warmup_steps : int
        ``"warmup_cosine"`` only: steps from 0 to ``init_value`` before
        the decay. Must lie in ``[0, total_steps)``.

    Returns
    -------
    optax.Schedule
        ``schedule(step)`` in ``[end_value, init_value]``, except during
        the ``"warmup_cosine"`` warmup, which starts at 0. Compose as a
        multiplier: ``lr = base_lr * schedule(step)``.
    """
    if total_steps < 1:
        raise ValueError(f"annealing_schedule: total_steps must be at least 1, got {total_steps}")
    if kind not in _SCHEDULE_KINDS:
        raise ValueError(
            f"annealing_schedule: unknown kind={kind!r}; expected one of {sorted(_SCHEDULE_KINDS)}"
        )
    if init_value <= 0.0:
        raise ValueError(f"annealing_schedule: init_value must be positive, got {init_value}")
    if not 0.0 <= end_value <= init_value:
        raise ValueError(
            f"annealing_schedule: end_value={end_value} must lie in [0, init_value={init_value}]"
        )
    if kind == "exponential" and end_value == 0.0:
        # A geometric decay never reaches zero, and the stand-in used to be a
        # straight line under the exponential name.
        raise ValueError(
            "annealing_schedule: an exponential decay needs end_value > 0, since it never "
            "reaches zero; use kind='linear' or 'cosine' to end at 0"
        )
    if kind == "warmup_cosine" and not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"annealing_schedule: warmup_steps={warmup_steps} must lie in "
            f"[0, total_steps={total_steps})"
        )

    schedule = _optax_schedule(kind, total_steps, init_value, end_value, warmup_steps)

    def multiplier(step: ArrayLike) -> Array:
        return jnp.asarray(schedule(step))

    return multiplier


def _optax_schedule(
    kind: str, total_steps: int, init_value: float, end_value: float, warmup_steps: int
) -> optax.Schedule:
    """The optax schedule behind :func:`annealing_schedule`, inputs already validated."""
    if kind == "cosine":
        # optax's cosine lands exactly on alpha * init_value at decay_steps.
        return optax.cosine_decay_schedule(
            init_value=init_value,
            decay_steps=total_steps,
            alpha=end_value / init_value,
        )
    if kind == "linear":
        return optax.linear_schedule(
            init_value=init_value,
            end_value=end_value,
            transition_steps=total_steps,
        )
    if kind == "warmup_cosine":
        # init_value doubles as the peak the schedule rises to.
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=init_value,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end_value,
        )
    # exponential: the per-epoch geometric factor is derived from the end
    # point, so the curve lands exactly on end_value at step == total_steps
    # (same contract as every other kind). optax's exponential_decay does
    # not flatten past its budget, so clamp the tail at end_value.
    rate = (end_value / init_value) ** (1.0 / total_steps)
    raw = optax.exponential_decay(
        init_value=init_value,
        transition_steps=1,
        decay_rate=rate,
    )

    def clamped(step):
        return jnp.maximum(raw(step), end_value)

    return clamped
