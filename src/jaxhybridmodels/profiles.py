"""Ready-made time profiles for exogenous, time-varying quantities.

A hybrid model's vector field often needs a quantity that changes over
time: a reactor temperature profile, a feed flow that steps on at some
moment, a ramp between two set-points. ``simulate_fn`` already receives
per-experiment covariates, so the *parameters* of the profile (a set
point, a ramp rate, a step time) travel as ordinary covariates and the
profile itself is a pure callable ``t -> Array`` used inside the user's
vector field:

    def vector_field(t, y, args):
        T = ramp_profile(t0=t0, t1=t1, v0=25.0, v1=60.0)(t)  # flat, ramp, flat
        ...

Everything here is a pure JAX function, safe under ``jit``, ``vmap``
and ``grad``, so a profile composes with the framework exactly like
any other term in the vector field. The functions are factories: they
close over their parameters and return the ``t -> Array`` callable.
``step_profile`` and ``ramp_profile`` take their parameters by keyword,
since several same-typed numbers in a row are easy to swap.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

# A profile maps a time (Python float or traced scalar) to a value.
Profile = Callable[[ArrayLike], Array]


def constant_profile(value: ArrayLike) -> Profile:
    """Return ``t -> value``: a profile that never changes.

    The degenerate case, included so a code path can treat every
    quantity uniformly (a quantity is either a covariate or
    ``constant_profile(v)`` evaluated at ``t``).
    """

    def profile(t: ArrayLike) -> Array:
        return jnp.asarray(value)

    return profile


def step_profile(*, before: ArrayLike, after: ArrayLike, jump_at: ArrayLike) -> Profile:
    """Return ``t -> before if t < jump_at else after``: a step change.

    For a reactor, the moment a feed valve opens or a heater switches.
    Discontinuous at ``jump_at``; an adaptive solver sees a kink, so
    place the jump at a known event time or accept a short transient.
    """

    def profile(t: ArrayLike) -> Array:
        return jnp.where(jnp.asarray(t) < jump_at, jnp.asarray(before), jnp.asarray(after))

    return profile


def ramp_profile(*, t0: ArrayLike, t1: ArrayLike, v0: ArrayLike, v1: ArrayLike) -> Profile:
    """Return a flat-ramp-flat profile: ``v0`` until ``t0``, linear to ``v1`` by ``t1``, then flat.

    The reactor heat-up shape: hold at the initial set point, ramp to the
    final set point, hold there. Values before ``t0`` and after ``t1``
    are exactly the edge values (the two flat profiles on either edge).

    ``t1 > t0`` is required; equal times would be a discontinuity. The
    check runs when the times are host-side values; inside ``jit``/``vmap``
    (per-experiment parameters from traced covariates) it is skipped, so
    the factory stays trace-safe.
    """
    try:
        degenerate = bool(jnp.asarray(t1) <= jnp.asarray(t0))
    except jax.errors.TracerBoolConversionError:
        degenerate = False
    if degenerate:
        raise ValueError(
            f"ramp_profile requires t1 > t0, got t0={t0}, t1={t1}; "
            "equal times would be a discontinuity."
        )

    start, end = jnp.asarray(t0), jnp.asarray(t1)
    low, high = jnp.asarray(v0), jnp.asarray(v1)

    def profile(t: ArrayLike) -> Array:
        rate = (high - low) / (end - start)
        # jnp.minimum/maximum, not min()/max(): the edge values may be
        # traced (per-experiment parameters from covariates), and the
        # Python builtins would bool-convert a tracer.
        return jnp.clip(low + rate * (t - start), jnp.minimum(low, high), jnp.maximum(low, high))

    return profile


def piecewise_linear_profile(knots: tuple[tuple[ArrayLike, ArrayLike], ...]) -> Profile:
    """Return a piecewise-linear interpolation of ``(t, v)`` knots.

    The general shape: any finite profile that is linear between its
    knots. Constant on both edges (the first and last values extend
    outward). Knots must be strictly increasing in ``t`` and contain at
    least two entries.
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

    return profile
