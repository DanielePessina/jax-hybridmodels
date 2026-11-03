"""Ready-made time profiles for exogenous, time-varying quantities.

A hybrid model's vector field often needs a quantity that changes over
time: a reactor temperature profile, a feed flow that steps on at some
moment, a ramp between two set-points. ``simulate_fn`` already receives
per-experiment covariates, so the *parameters* of the profile (a set
point, a ramp rate, a step time) travel as ordinary covariates and the
profile itself is a pure callable ``t -> Array`` used inside the user's
vector field:

    def vector_field(t, y, args):
        T = ramp_profile(start=t0, end=t1, before=25.0, after=60.0)(t)  # flat, ramp, flat
        ...

Everything here is a pure JAX function, safe under ``jit``, ``vmap``
and ``grad``, so a profile composes with the framework exactly like
any other term in the vector field. The functions are factories: they
close over their parameters and return a :class:`Profile`, a ``t -> Array``
callable. ``step_profile`` and ``ramp_profile`` take their parameters by
keyword, since several same-typed numbers in a row are easy to swap.

A profile that jumps or bends makes the vector field non-smooth there. An
adaptive solver's error estimate assumes smoothness, so it either rejects
steps around the change or, for a feature shorter than a step, steps over it
without noticing. Each profile lists those times in ``jump_ts``; pass them to
:meth:`~jaxhybridmodels.SolverConfig.diffeqsolve` so the solver stops on them::

    sol = solver.diffeqsolve(term, ts, y0, jump_ts=temperature.jump_ts)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array


class Profile:
    """A time profile: call it with ``t``, and read where it jumps or bends.

    ``jump_ts`` holds the times where the profile or its slope changes, empty
    for a smooth one. Concatenate several profiles' ``jump_ts`` to pass them
    to one solve.
    """

    def __init__(self, fn: Callable[[ArrayLike], Array], jump_ts: ArrayLike | None = None) -> None:
        self._fn = fn
        self.jump_ts = (
            jnp.zeros(0) if jump_ts is None else jnp.atleast_1d(jnp.asarray(jump_ts, dtype=float))
        )

    def __call__(self, t: ArrayLike) -> Array:
        return self._fn(t)


def constant_profile(value: ArrayLike) -> Profile:
    """Return ``t -> value``: a profile that never changes.

    The degenerate case, included so a code path can treat every
    quantity uniformly (a quantity is either a covariate or
    ``constant_profile(v)`` evaluated at ``t``).
    """

    def profile(t: ArrayLike) -> Array:
        return jnp.asarray(value)

    return Profile(profile)


def step_profile(*, at: ArrayLike, before: ArrayLike, after: ArrayLike) -> Profile:
    """Return ``t -> before if t < at else after``: a step change.

    For a reactor, the moment a feed valve opens or a heater switches.
    Discontinuous at ``at``, which is its ``jump_ts``.
    """

    def profile(t: ArrayLike) -> Array:
        return jnp.where(jnp.asarray(t) < at, jnp.asarray(before), jnp.asarray(after))

    return Profile(profile, jnp.asarray([at]))


def ramp_profile(
    *, start: ArrayLike, end: ArrayLike, before: ArrayLike, after: ArrayLike
) -> Profile:
    """Return a flat-ramp-flat profile: ``before`` until ``start``, linear to ``after`` by ``end``.

    The reactor heat-up shape: hold at the initial set point, ramp to the
    final set point, hold there. Values before ``start`` and after ``end``
    are exactly ``before`` and ``after``, the names ``step_profile`` uses
    for the same two levels.

    ``end > start`` is required; equal times would be a discontinuity. The
    check runs when the times are host-side values; inside ``jit``/``vmap``
    (per-experiment parameters from traced covariates) it is skipped, so
    the factory stays trace-safe.
    """
    try:
        degenerate = bool(jnp.asarray(end) <= jnp.asarray(start))
    except jax.errors.TracerBoolConversionError:
        degenerate = False
    if degenerate:
        raise ValueError(
            f"ramp_profile requires end > start, got start={start}, end={end}; "
            "equal times would be a discontinuity."
        )

    t_start, t_end = jnp.asarray(start), jnp.asarray(end)
    low, high = jnp.asarray(before), jnp.asarray(after)

    def profile(t: ArrayLike) -> Array:
        rate = (high - low) / (t_end - t_start)
        # jnp.minimum/maximum, not min()/max(): the edge values may be
        # traced (per-experiment parameters from covariates), and the
        # Python builtins would bool-convert a tracer.
        return jnp.clip(low + rate * (t - t_start), jnp.minimum(low, high), jnp.maximum(low, high))

    # Continuous, but its slope jumps at both corners.
    return Profile(profile, jnp.stack([t_start, t_end]))


def piecewise_linear_profile(knots: tuple[tuple[ArrayLike, ArrayLike], ...]) -> Profile:
    """Return a piecewise-linear interpolation of ``(t, v)`` knots.

    The general shape: any finite profile that is linear between its
    knots. Constant on both edges (the first and last values extend
    outward). Knots must be strictly increasing in ``t`` and contain at
    least two entries. Its slope changes at every knot, so the knot times are
    its ``jump_ts``.
    """
    if len(knots) < 2:
        raise ValueError("piecewise_linear_profile requires at least two (t, v) knots")
    ts = jnp.asarray([k[0] for k in knots])
    vs = jnp.asarray([k[1] for k in knots])
    # The check runs only for host-side knot times, mirroring
    # ramp_profile: traced knots (a factory built inside jit/vmap) skip it
    # rather than bool-converting a tracer.
    try:
        increasing = bool(jnp.all(jnp.diff(ts) > 0))
    except jax.errors.TracerBoolConversionError:
        increasing = True
    if not increasing:
        raise ValueError("piecewise_linear_profile knot times must be strictly increasing")

    def profile(t: ArrayLike) -> Array:
        t = jnp.asarray(t)
        idx = jnp.searchsorted(ts, t, side="right") - 1
        i0 = jnp.clip(idx, 0, len(ts) - 2)
        # Clamped fraction: t outside the knot range sits exactly on the
        # first/last value (the two flat edges), never extrapolated.
        frac = jnp.clip((t - ts[i0]) / (ts[i0 + 1] - ts[i0]), 0.0, 1.0)
        return vs[i0] + frac * (vs[i0 + 1] - vs[i0])

    return Profile(profile, ts)
