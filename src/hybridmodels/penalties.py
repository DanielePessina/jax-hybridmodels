"""Keeping quantities inside their physical bounds without killing the gradient.

``BoundScaler`` enforces a physical box by reparameterisation: it maps the
box onto an unbounded *latent* ``z`` through a squashing function, so an
out-of-box physical value has no latent that represents it. Feasibility
comes for free that way, usable gradient does not. This module repairs the
two places it goes missing.

Output side, saturation. ``from_latent`` has derivative
``(high - low) / T * sigma'(z / T)``, which underflows to exactly 0.0 past
``|z / T| = 15`` in float32. A predictor that far out is pinned against its
bound with nothing left to pull it back. :meth:`BoundScaler.saturation`
hinges on the latent magnitude to stop it getting there. Hinge on the
latent, never the physical value: a penalty written against the output
picks up that same ``sigma'`` factor and so dies exactly where saturation
is worst.

Input side, excursion. ``to_latent`` must keep ``logit`` away from its
poles at 0 and 1. A hard ``jnp.clip`` does that, but its derivative outside
the box is exactly zero, and that zero propagates back through the ODE
adjoint to every upstream parameter. :func:`soft_logit` continues linearly
instead, and :func:`box_violation` supplies push-back beyond its reach.

Everything here is pure and safe under ``jit``, ``vmap`` and ``grad``.
Hinges are ``jnp.maximum(., 0.0) ** 2``: C^1, so an adaptive ODE step-size
controller does not chatter at the crossing, and pole-free, so they need
none of the double-``where`` guarding ``losses.py`` applies around ``log``.
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

    Recurses into each match rather than stopping at it: nesting is
    supported (``inner`` is typed ``Predictor``), and a traversal that
    stopped at the outer match would leave the inner box unpenalised.

    Order is outer-then-inner, depth first. :func:`collocation_grids`
    relies on it to return a positional tuple.

    ``BoundedPredictor`` is imported lazily because ``predictors.base``
    imports this module for :func:`soft_logit`.
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
    interval, where a hard clip would zero the derivative and silently drop
    state-derived sensitivities from the ODE adjoint (R-P2).

    Exact in value and derivative inside the band, and C^1 across the
    junction, since the continuation uses the inverse's own slope there.
    The ``stop_gradient`` on the clamp is load-bearing: without it the
    correction term picks up a contribution through the clip and the
    interior derivative comes out wrong.
    """
    s_clamped = jax.lax.stop_gradient(jnp.clip(s, eps, 1.0 - eps))
    return inverse(s_clamped) + inverse_slope(s_clamped) * (s - s_clamped)


def soft_logit(s: Array, eps: float = 1e-3) -> Array:
    """``logit(s)``, extended linearly outside ``[eps, 1 - eps]``.

    ``s`` is a physical value already normalised into ``[0, 1]`` across its
    declared box, so this is the step that turns a bounded quantity into an
    unbounded latent. Outside the band the result grows linearly instead of
    blowing up at the pole, and the derivative is a finite constant instead
    of the exact zero ``logit(jnp.clip(s, eps, 1 - eps))`` would give.

    :func:`softclip` cannot do this job: its interior error is
    ``O(1 / beta)`` and ``s`` spans only ``[0, 1]``, so any ``beta`` gentle
    enough to keep gradient far outside the box also distorts the middle.

    ``eps`` sets the continuation slope, roughly ``1 / eps``, and is the one
    tuning knob. The default 1e-3 maps a 1% overshoot to ``|z| ~ 10``, a
    number a network can still consume; 1e-6 would map it to ``|z| ~ 1e4``.
    """
    from hybridmodels.transforms import BOUND_TRANSFORMS

    t = BOUND_TRANSFORMS["sigmoid"]
    return soft_inverse(s, t.inverse, t.inverse_slope, eps)


def softclip(x: Array, lo: float | Array, hi: float | Array, beta: float = 20.0) -> Array:
    """Smoothly clamp ``x`` into ``[lo, hi]`` with a derivative that never hits zero.

    Built from two softplus shoulders, so the derivative is
    ``sigmoid(beta * (x - lo)) - sigmoid(beta * (x - hi))``, analytically in
    ``(0, 1)``. The interior is reproduced to ``O(1 / beta)``. Larger
    ``beta`` tracks a hard clip more closely but underflows sooner outside
    the box; the default 20 holds interior error below about 0.05 box widths
    and keeps usable gradient roughly one width out.

    Repairs the near field only. Several widths out the derivative
    underflows as a hard clip's does, so pair it with :func:`box_violation`
    for unbounded push-back.
    """
    scaled_softplus = lambda u: jax.nn.softplus(beta * u) / beta  # noqa: E731
    return lo + scaled_softplus(x - lo) - scaled_softplus(x - hi)


def clip_ste(x: Array, lo: float | Array, hi: float | Array) -> Array:
    """Hard-clip on the forward pass, identity on the backward pass.

    The straight-through estimator. Use it when downstream code genuinely
    requires a feasible number, say a concentration that must not go
    negative before a ``log``, while the task loss keeps flowing as though
    the clip were not there.

    The identity gradient is a deliberate fiction: it propagates whatever
    the data loss asks for, including "go further out of bounds", forever.
    Pair it with :func:`box_violation` on the pre-clip value for the
    restoring force.
    """
    return x + jax.lax.stop_gradient(jnp.clip(x, lo, hi) - x)


def box_violation(x: Array, lows: Array, highs: Array) -> Array:
    """Width-normalised squared hinge measuring how far ``x`` falls outside its box.

    Zero in value and gradient strictly inside the box, so it never
    perturbs the feasible interior. Outside it grows quadratically, giving
    a restoring gradient linear in the overshoot, which does not vanish the
    way a reparameterised bound's does.

    Each component is normalised by its own width ``high - low`` so one
    penalty weight works across channels. Bounds here run from fractions of
    a unit to hundreds of kelvin, and an unnormalised hinge would let the
    widest channel dominate on units alone.

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
    trajectory. Each grid is a tensor product of ``n_per_dim`` evenly spaced
    points along every dimension of that predictor's ``in_scaler.bounds``,
    shape ``[n_per_dim ** n_inputs, n_inputs]``.

    Call this once on the host before the training loop: the grids depend
    only on static ``bounds``, so rebuilding them per step compiles work
    that computes a constant. The returned tuple is positional and matches
    the leaf order :func:`bound_penalty` walks.

    The grid is deterministic rather than sampled because ``restore_best``
    compares raw loss values across steps, and a resampled penalty could
    pick a "best" that drew an easy sample.

    Point count grows exponentially in input dimension. Predictors here take
    two or three inputs, so ``n_per_dim=5`` is 25 to 125 forward passes and
    negligible beside an ODE solve. Lower it if that stops holding.

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

    The default way to penalise bound behaviour here, and deliberately
    trajectory-blind: saturation is a property of the predictor as a
    function on its declared box, whatever any particular solve does. So it
    needs no cooperation from ``simulate_fn``, ``predict_bucket`` or the
    loss protocol, behaves the same inside a vector field or above one, and
    handles arbitrary nesting by walking leaves.

    That cuts both ways. It reports saturation anywhere in the declared box,
    including regions no training trajectory visited, which catches
    extrapolation failure early. It cannot answer "did this solve push an
    input out of range", which needs the penalty computed where the state
    actually went.

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
