"""Keeping quantities inside their physical bounds without killing the gradient.

Some quantities in a hybrid model have known physical bounds. A rate
constant is positive, a concentration cannot be negative, a temperature
sits inside the range the rig can reach. ``BoundScaler`` enforces such a
box by reparameterisation. It maps the physical box onto an unbounded
*latent* coordinate ``z`` through a squashing function (sigmoid by
default) and its inverse, so an out-of-box physical value simply has no
latent that represents it. Feasibility comes for free that way. Usable
gradient does not, and this module repairs the two places it goes
missing.

Saturation, on the output side. ``from_latent`` has derivative
``(high - low) / T * sigma'(z / T)``, where ``T`` is the scaler's
temperature. That factor is 4.5e-2 at ``z = 3``, 4.5e-5 at ``z = 10``, and
underflows to exactly 0.0 past ``|z / T| = 15`` in float32. A predictor
that far out is pinned against its bound with no gradient left to pull it
back. :meth:`BoundScaler.saturation` charges a hinge on the latent
magnitude to stop it getting there.

Penalise the latent, never the physical value. A penalty written against
the physical output picks up that same ``sigma'`` factor on the backward
pass, so it dies exactly where saturation is worst, and then reports
itself satisfied while the predictor is dead. Hinging on ``|z| / T`` gives
push-back linear in the overshoot, which never underflows.

Excursion, on the input side. ``to_latent`` must keep ``logit`` away from
its poles at 0 and 1. A hard ``jnp.clip`` does that, but its derivative
outside the box is exactly zero, and because the guard sits in the middle
of the computation graph that zero propagates back to every upstream
parameter. Predictor inputs are often derived from the ODE state, so a
clipped input silently drops a real sensitivity from the adjoint (the
backward solve that produces gradients through the ODE).
:func:`soft_logit` continues linearly instead, and :func:`box_violation`
supplies push-back beyond its reach.

Everything here is pure and safe under ``jit``, ``vmap`` and ``grad``.
Hinges use ``jnp.maximum(., 0.0) ** 2``. The squared hinge is C^1, so an
adaptive ODE step-size controller does not chatter at the crossing, and it
has no pole, so it needs none of the double-``where`` guarding that
``losses.py`` applies around ``log`` and division.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

if TYPE_CHECKING:
    from hybridmodels.predictors import BoundedPredictor

__all__ = (
    "soft_inverse",
    "soft_logit",
    "softclip",
    "clip_ste",
    "box_violation",
    "collocation_grids",
    "bound_penalty",
)


def _bounded_leaves(predictors: Any) -> list[BoundedPredictor]:
    """Every ``BoundedPredictor`` in ``predictors``, outermost first.

    Recurses into each match rather than stopping at it. A traversal that
    stops at the first match halts at the outermost ``BoundedPredictor``,
    and one nested as another's ``inner`` would then declare a box that
    never gets penalised. Nesting is supported (``inner`` is typed
    ``Predictor``, and ``BoundedPredictor`` is one), so the walk keeps going.

    Order is outer-then-inner, depth first, matching pytree traversal for
    the sibling case. :func:`collocation_grids` relies on it to return a
    positional tuple.

    ``BoundedPredictor`` is imported lazily because ``predictors.base``
    imports this module for :func:`soft_logit`; a module-level import back
    would be circular.
    """
    from hybridmodels.predictors import BoundedPredictor

    is_bp = lambda node: isinstance(node, BoundedPredictor)  # noqa: E731

    found: list[BoundedPredictor] = []
    for leaf in jtu.tree_leaves(predictors, is_leaf=is_bp):
        if is_bp(leaf):
            found.append(leaf)
            found.extend(_bounded_leaves(leaf.inner))
    return found


def soft_inverse(
    s: Array,
    inverse: Callable[[Array], Array],
    inverse_slope: Callable[[Array], Array],
    eps: float = 1e-3,
) -> Array:
    """``inverse(s)``, extended linearly outside ``[eps, 1 - eps]``.

    Generalises :func:`soft_logit` to the inverse of any squashing
    function. Every such inverse has a pole at each end of the unit
    interval, and the reason a hard clip is unacceptable there is the same
    whichever squash it is. A zero derivative in the middle of the graph
    propagates back to every upstream parameter and drops state-derived
    sensitivities from the ODE adjoint without raising (R-P2).

    Exact in value and derivative inside the band, and C^1 across the
    junction, because the continuation uses the inverse's own slope at the
    crossing. The ``stop_gradient`` on the clamp is load-bearing. Without
    it the correction term picks up a contribution through the clip and the
    interior derivative comes out wrong.
    """
    s_clamped = jax.lax.stop_gradient(jnp.clip(s, eps, 1.0 - eps))
    return inverse(s_clamped) + inverse_slope(s_clamped) * (s - s_clamped)


def soft_logit(s: Array, eps: float = 1e-3) -> Array:
    """``logit(s)``, extended linearly outside ``[eps, 1 - eps]``.

    ``s`` is a physical value already normalised into ``[0, 1]`` across its
    declared box, and ``logit`` is the inverse of the sigmoid squash, so
    this is the step that turns a bounded quantity into an unbounded latent.

    Exact in value and derivative for ``s`` inside the band, and C^1 across
    the junction, since the continuation uses logit's own slope at the
    crossing. Outside the band the result grows linearly instead of blowing
    up at the pole, and the derivative is a finite constant instead of zero.

    Replaces the ``logit(jnp.clip(s, eps, 1 - eps))`` idiom, whose
    derivative outside the band is exactly zero. The module docstring
    explains why that is a silent correctness bug.

    :func:`softclip` cannot do this job. Its interior error is
    ``O(1 / beta)`` in the units of ``s``, and ``s`` spans only ``[0, 1]``,
    so any ``beta`` gentle enough to keep gradient far outside the box also
    distorts the middle of it. Splitting into an exact band and a linear
    continuation avoids that trade, with no interior distortion at any
    threshold.

    ``eps`` sets the continuation slope, ``1 / (eps * (1 - eps))``, roughly
    ``1 / eps``. It is the one tuning knob. At ``eps = 1e-6`` a 1% overshoot
    maps to ``|z| ~ 1e4``, which saturates or overflows the inner network.
    Larger ``eps`` shrinks the exact band. The default 1e-3 maps a 1%
    overshoot to ``|z| ~ 10``, outside the sigmoid's linear region but still
    a number a network can consume.
    """
    from hybridmodels.transforms import BOUND_TRANSFORMS

    t = BOUND_TRANSFORMS["sigmoid"]
    return soft_inverse(s, t.inverse, t.inverse_slope, eps)


def softclip(x: Array, lo: float | Array, hi: float | Array, beta: float = 20.0) -> Array:
    """Smoothly clamp ``x`` into ``[lo, hi]`` with a derivative that never hits zero.

    Built from two softplus shoulders, so the derivative is
    ``sigmoid(beta * (x - lo)) - sigmoid(beta * (x - hi))``, which lies in
    ``(0, 1)`` analytically. The interior is reproduced to ``O(1 / beta)``
    and the output asymptotes to the bounds rather than meeting them.

    Larger ``beta`` tracks a hard clip more closely but underflows sooner
    outside the box. The default 20 holds interior error below about 0.05
    box widths and keeps usable gradient roughly one width out.

    This repairs the near field only. Several widths out the derivative
    underflows just as a hard clip's does. Pair it with
    :func:`box_violation`, which supplies the unbounded push-back.
    """
    scaled_softplus = lambda u: jax.nn.softplus(beta * u) / beta  # noqa: E731
    return lo + scaled_softplus(x - lo) - scaled_softplus(x - hi)


def clip_ste(x: Array, lo: float | Array, hi: float | Array) -> Array:
    """Hard-clip on the forward pass, identity on the backward pass.

    This is the straight-through estimator. Use it when downstream code
    genuinely requires a feasible number, say a concentration that must not
    go negative before a ``log``, while the task loss should keep flowing as
    though the clip were not there.

    The identity gradient is a deliberate fiction. It propagates whatever
    the data loss asks for, including "go further out of bounds", forever.
    A straight-through clip never pushes back on its own. Pair it with
    :func:`box_violation` on the pre-clip value for the restoring force.
    """
    return x + jax.lax.stop_gradient(jnp.clip(x, lo, hi) - x)


def box_violation(x: Array, lows: Array, highs: Array) -> Array:
    """Width-normalised squared hinge measuring how far ``x`` falls outside its box.

    Returns a scalar. Zero in value and gradient strictly inside the box,
    so it never perturbs the feasible interior. Outside it grows
    quadratically, giving a restoring gradient linear in the overshoot,
    which does not vanish the way a reparameterised bound's does.

    Each component is normalised by its own width ``high - low`` so one
    penalty weight works across channels. Bounds in this package run from
    fractions of a unit to hundreds of kelvin, and an unnormalised hinge
    would let the widest channel dominate on units alone.

    Parameters
    ----------
    x : Array
        Values in physical units; broadcast against ``lows`` / ``highs``.
    lows, highs : Array
        Per-component box edges, as produced by ``BoundScaler._lows_highs``.

    Returns
    -------
    Array
        Scalar sum of squared fractional violations.
    """
    width = highs - lows
    below = jnp.maximum((lows - x) / width, 0.0)
    above = jnp.maximum((x - highs) / width, 0.0)
    return jnp.sum(below**2 + above**2)


def collocation_grids(predictors: Any, n_per_dim: int = 5) -> tuple[Array, ...]:
    """Build one *collocation grid* per ``BoundedPredictor`` leaf of ``predictors``.

    A collocation grid is a fixed set of input points at which a predictor
    is evaluated for inspection, chosen up front rather than taken from any
    trajectory. Each grid here is a tensor product of ``n_per_dim`` evenly
    spaced points along every dimension of that predictor's
    ``in_scaler.bounds``, so it has shape
    ``[n_per_dim ** n_inputs, n_inputs]`` and covers the whole box the
    predictor declares.

    Call this once on the host before the training loop. The grids depend
    only on static ``bounds``, so rebuilding them per step adds compiled
    work that computes a constant. The returned tuple is positional and
    matches the leaf order :func:`bound_penalty` walks, the same
    ``jtu.tree_leaves(..., is_leaf=...)`` order used by
    ``reinitialize_pytree_with_key`` and ``freeze_modules_of_type``.

    The grid is deterministic rather than sampled because ``restore_best``
    compares raw loss values across steps. A resampled penalty would make
    that comparison noisy and could pick a "best" that drew an easy sample.

    Point count grows exponentially in input dimension. ``n_per_dim=5``
    over six inputs is 15625 points. Predictors here take two or three
    named inputs, so the grid is 25 to 125 forward passes and negligible
    beside an ODE solve. Lower ``n_per_dim`` if that stops holding.

    Returns
    -------
    tuple[Array, ...]
        One ``[G, n_inputs]`` grid per ``BoundedPredictor`` leaf, in
        traversal order. Empty if the pytree holds no such leaf.
    """
    grids: list[Array] = []
    for leaf in _bounded_leaves(predictors):
        axes = [jnp.linspace(low, high, n_per_dim) for low, high in leaf.in_scaler.bounds]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        grids.append(jnp.stack([m.reshape(-1) for m in mesh], axis=-1))
    return tuple(grids)


def bound_penalty(predictors: Any, grids: tuple[Array, ...]) -> Array:
    """Mean output-squash saturation over every ``BoundedPredictor`` in a pytree.

    For each leaf, evaluates ``inner(in_scaler.to_latent(x))`` across that
    leaf's collocation grid and charges
    :meth:`~hybridmodels.predictors.BoundScaler.saturation` on the
    resulting latents. Leaves are summed.

    This is the default way to penalise bound behaviour here, and it is
    deliberately trajectory-blind. Saturation is a property the predictor
    has as a function on its declared input box, whatever any particular
    solve does. Three consequences follow. It needs no cooperation from
    ``simulate_fn``, ``predict_bucket`` or the loss protocol, so none of
    their signatures change. It never observes the call site, so it behaves
    the same whether the predictor runs inside a vector field or above one.
    It walks the pytree by leaf, so arbitrary nesting works for free.

    Trajectory-blindness cuts both ways. The penalty reports saturation
    anywhere in the declared box, including regions no training trajectory
    visited, which catches extrapolation failure before deployment. It
    cannot answer "did this solve push an input out of range", since that
    needs the penalty computed where the state actually went.

    Parameters
    ----------
    predictors : PyTree[eqx.Module]
        Any pytree shape. Only ``BoundedPredictor`` leaves contribute.
    grids : tuple[Array, ...]
        Output of :func:`collocation_grids` for this same pytree.

    Returns
    -------
    Array
        Non-negative scalar. Exactly zero when no leaf saturates.
    """
    leaves = _bounded_leaves(predictors)
    if len(leaves) != len(grids):
        raise ValueError(
            f"grids has {len(grids)} entries but predictors holds "
            f"{len(leaves)} BoundedPredictor leaves; rebuild the grids "
            "with collocation_grids(predictors) after changing the pytree."
        )
    total = jnp.asarray(0.0)
    for leaf, grid in zip(leaves, grids, strict=True):
        latents = jax.vmap(lambda x, _l=leaf: _l.inner(_l.in_scaler.to_latent(x)))(grid)
        total = total + leaf.out_scaler.saturation(latents)
    return total
