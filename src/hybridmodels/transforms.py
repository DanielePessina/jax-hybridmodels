"""Registries for the two independent axes of a bound scaler.

``BoundScaler`` turns a physical box into an unbounded latent in two
steps, and they answer different questions.

The **warp** decides what "halfway between the bounds" means. A rate
constant bounded by ``(1e-6, 1e2)`` has its plausible values spread over
eight decades, and a linear warp puts 99.999% of the latent range above
``1e-2``. The whole low end collapses into a sliver you cannot resolve.
``"log10"`` makes the midpoint ``1e-2`` instead of ``50``, which is what
anyone fitting kinetics means by the middle of that range.

The **transform** decides how the squash saturates near the bounds, and
therefore how fast the gradient dies once a predictor pushes against one.

The two compose freely. Bounds are always declared in physical units
whichever warp is chosen.

Both are name-keyed registries following ``SOLVER_REGISTRY`` in
``solver.py`` and ``LOSS_REGISTRY`` in ``losses.py``. The scaler stores a
string, so it stays JSON-friendly and round-trips through
``eqx.tree_serialise_leaves`` with no change to its leaf structure. That
is why these are plain ``NamedTuple`` records rather than modules: they
hold no arrays, so composing one into ``BoundScaler`` would add a pytree
node that serialises to nothing and could only be rebuilt from the
load-side template.

Register your own with :func:`register_bound_transform` and
:func:`register_warp`.

Gradient decay is the reason to care about the transform choice.
Measured float32 ``|du/dz|``, and the ``z`` at which it underflows to
exactly ``0.0``:

    sigmoid    e^-|z|        4.5e-5 at z=10,  dead at z=16.8
    algebraic  |z|^-3 / 2    4.9e-4 at z=10,  dead at z~3.0e3
    softsign   |z|^-2 / 2    4.1e-3 at z=10,  dead at z~1.1e7

Sigmoid's last usable gradient (3.6e-7, at z=15) is matched by algebraic
at z~110 and by softsign at z~1178. Polynomial decay does not remove the
need for the saturation penalty. Escape time under a gradient of c/z^2
scales as z^3, so recovering from z=100 is 1e6 times slower than from
z=1. It turns an impossible recovery into a slow one.

``tanh`` is absent on purpose. ``(1 + tanh z) / 2`` is exactly
``sigmoid(2z)``, so it would be the sigmoid entry at half temperature and
nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

__all__ = (
    "BoundTransform",
    "Warp",
    "BOUND_TRANSFORMS",
    "WARPS",
    "register_bound_transform",
    "register_warp",
)


class BoundTransform(NamedTuple):
    """A latent to unit-interval squash, its inverse, and the metadata around them.

    Attributes
    ----------
    forward : Callable
        ``R -> (0, 1)``. What ``from_latent`` applies.
    inverse : Callable
        ``(0, 1) -> R``. What ``to_latent`` inverts with.
    inverse_slope : Callable
        ``d(inverse)/ds``. Used to build the linear continuation that keeps
        ``to_latent`` differentiable outside the box.
    knee : float
        Latent at which ``forward`` reaches 0.95, the outer 5% of the box.
        This is the default for ``BoundScaler.z_knee``. It has to come from
        the transform: sharing sigmoid's 2.944 with softsign would start
        charging the saturation penalty at 12.5% from the bound instead of
        5%, roughly 2.7 times more aggressive in physical terms.
    """

    forward: Callable[[Array], Array]
    inverse: Callable[[Array], Array]
    inverse_slope: Callable[[Array], Array]
    knee: float


class Warp(NamedTuple):
    """A monotone reparameterisation of the physical axis before normalising.

    Attributes
    ----------
    forward : Callable
        Physical to warped coordinate. Must accept a Python float as well
        as an array, because :func:`warp_bounds` calls it on the static
        edges at construction time.
    inverse : Callable
        Warped coordinate back to physical. Must invert ``forward`` exactly
        on the declared box.
    requires_positive : bool
        Whether the warp is undefined at or below zero. Checked against the
        declared bounds at construction, where it can raise a useful error,
        rather than at trace time where it would surface as a silent nan.
    """

    forward: Callable[[ArrayLike], Array]
    inverse: Callable[[ArrayLike], Array]
    requires_positive: bool


def _softsign(z: Array) -> Array:
    return 0.5 * (1.0 + z / (1.0 + jnp.abs(z)))


def _softsign_inverse(s: Array) -> Array:
    y = 2.0 * s - 1.0
    return y / (1.0 - jnp.abs(y))


def _softsign_inverse_slope(s: Array) -> Array:
    y = 2.0 * s - 1.0
    return 2.0 / (1.0 - jnp.abs(y)) ** 2


def _algebraic(z: Array) -> Array:
    return 0.5 * (1.0 + z * jax.lax.rsqrt(1.0 + z * z))


def _algebraic_inverse(s: Array) -> Array:
    y = 2.0 * s - 1.0
    return y * jax.lax.rsqrt(1.0 - y * y)


def _algebraic_inverse_slope(s: Array) -> Array:
    y = 2.0 * s - 1.0
    return 2.0 * (1.0 - y * y) ** -1.5


BOUND_TRANSFORMS: dict[str, BoundTransform] = {
    "sigmoid": BoundTransform(
        forward=jax.nn.sigmoid,
        inverse=jax.scipy.special.logit,
        inverse_slope=lambda s: 1.0 / (s * (1.0 - s)),
        knee=2.9444389,  # logit(0.95)
    ),
    "algebraic": BoundTransform(
        forward=_algebraic,
        inverse=_algebraic_inverse,
        inverse_slope=_algebraic_inverse_slope,
        knee=2.0647416,  # 0.9 / sqrt(1 - 0.81)
    ),
    "softsign": BoundTransform(
        forward=_softsign,
        inverse=_softsign_inverse,
        inverse_slope=_softsign_inverse_slope,
        knee=9.0,  # 0.9 / (1 - 0.9)
    ),
}
"""Name to :class:`BoundTransform`. Extend via :func:`register_bound_transform`.

``algebraic`` is the recommended alternative to ``sigmoid``. It buys 180
times the latent runway and is C-infinity. ``softsign`` buys far more
runway again, but it is C^1 and not C^2, and the second-derivative jump
sits at the *box midpoint*, a point solver steps straddle routinely. Pick
it when a predictor is expected to live near its bounds and the dynamics
are not stiff.
"""


WARPS: dict[str, Warp] = {
    # jnp.asarray, not the identity: warp_bounds calls forward on a Python
    # float and reads it back with float(), and the array path needs an
    # Array out either way.
    "linear": Warp(forward=jnp.asarray, inverse=jnp.asarray, requires_positive=False),
    "log": Warp(forward=jnp.log, inverse=jnp.exp, requires_positive=True),
    "log10": Warp(
        forward=jnp.log10,
        inverse=lambda w: jnp.power(10.0, w),
        requires_positive=True,
    ),
}
"""Name to :class:`Warp`. Extend via :func:`register_warp`.

``log10`` is usually the one you want for a quantity quoted in decades,
because the latent then reads in decades too and a bound of ``(1e-6, 1e2)``
has a midpoint of ``1e-2``. ``log`` is the same reparameterisation in
nats; it changes the latent scale, not which physical values are
reachable.
"""


def register_bound_transform(name: str, transform: BoundTransform) -> None:
    """Register a squash under ``name`` for use by ``BoundScaler``.

    Mirrors :func:`~hybridmodels.solver.register_solver`. A scaler stores
    only the name, so a custom transform must be registered before a saved
    scaler that references it can be rebuilt. Re-registering an existing
    name overwrites without warning.
    """
    BOUND_TRANSFORMS[name] = transform


def register_warp(name: str, warp: Warp) -> None:
    """Register an axis warp under ``name`` for use by ``BoundScaler``.

    Same contract as :func:`register_bound_transform`. A warp must be
    monotone on the declared box and ``inverse`` must undo ``forward``
    there, or the scaler's round trip stops being the identity.
    """
    WARPS[name] = warp


def warp_bounds(
    bounds: tuple[tuple[float, float], ...], warp_name: str
) -> tuple[tuple[float, float], ...]:
    """Map each ``(low, high)`` pair into warped coordinates, as Python floats.

    Call this once when a scaler is constructed and keep the result in a
    static field. Doing it per call would fail under ``jit``: every ``jnp``
    operation inside a trace is staged out, so ``jnp.log(1e-6)`` returns a
    tracer rather than a number and cannot be read back with ``float()``.
    Bounds are static, so the warped edges are compile-time constants and
    there is nothing to recompute anyway.

    A custom warp's ``forward`` therefore has to accept a Python float and
    return something ``float()`` can read. Every array library's
    elementwise function does, outside a trace.
    """
    forward = WARPS[warp_name].forward
    return tuple((float(forward(low)), float(forward(high))) for low, high in bounds)


def check_bounds(bounds: tuple[tuple[float, float], ...], warp_name: str) -> None:
    """Validate a bounds tuple against a warp, raising with the offending pair.

    Three failures are worth catching here rather than at trace time.

    Non-finite edges construct silently and then break three ways: the
    width is ``inf``, so every finite input normalises to 0 and the
    predictor sees one constant; ``from_latent`` returns ``inf``; and
    ``box_violation`` divides by ``inf`` and returns ``nan``. None of that
    raises.

    A non-positive edge under ``log``/``log10`` gives ``-inf`` or ``nan``
    edges, with the same silence.

    ``low >= high`` gives a zero or negative width, so the normalisation
    divides by zero or flips orientation.
    """
    warp = WARPS[warp_name]
    for low, high in bounds:
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(
                f"BoundScaler bounds must be finite; got ({low}, {high}). "
                "One-sided quantities are not supported: use a finite box "
                "wide enough to hold the plausible range, with warp='log10' "
                "if it spans decades."
            )
        if low >= high:
            raise ValueError(f"BoundScaler bounds require low < high; got ({low}, {high}).")
        if warp.requires_positive and low <= 0.0:
            raise ValueError(
                f"warp={warp_name!r} is undefined at or below zero, but bounds "
                f"({low}, {high}) start at {low}. Raise the lower bound to a "
                "small positive value, or use warp='linear'."
            )
