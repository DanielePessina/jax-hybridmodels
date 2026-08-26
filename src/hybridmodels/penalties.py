"""Gradient-safe primitives for soft bound handling.

The framework enforces bounds by *reparameterisation* — ``BoundScaler``
maps a physical box onto an unbounded latent via logit/sigmoid, so an
out-of-box physical value is unrepresentable by construction. That is a
strong guarantee, but it buys feasibility at the cost of two distinct
gradient pathologies that this module exists to repair.

**Saturation (output side).** ``from_latent`` has derivative
``(high - low) / T * sigma'(z / T)``, which decays exponentially:
``sigma'(3) ~ 4.5e-2``, ``sigma'(10) ~ 4.5e-5``, and past ``|z / T| ~ 15``
it underflows to exactly ``0.0`` in float32. A predictor whose latent has
drifted that far is pinned against its bound with no gradient left to pull
it back. The repair is :meth:`BoundScaler.saturation`, a hinge on the
*latent* magnitude.

The choice of latent space is the load-bearing decision here, not a
detail. A penalty written against the *physical* output is multiplied by
that same ``sigma'`` factor on the way back, so it vanishes precisely when
saturation is worst — a penalty that silently reads as satisfied is worse
than no penalty at all, because nothing distinguishes "healthy" from
"dead". Penalising ``|z| / T`` instead gives push-back that grows linearly
in the overshoot and never underflows.

**Excursion (input side).** ``to_latent`` must keep ``logit`` away from its
poles at the closed endpoints. Doing that with a hard ``jnp.clip`` gives
*exactly* zero derivative outside the box, and because the guard sits
mid-graph that zero propagates to every upstream parameter on the path.
Predictor inputs are routinely state-derived (supersaturation in the
crystallisation example), so a clipped input drops a real sensitivity out
of the ODE adjoint without raising anything. :func:`softclip` replaces the
hard guard with a smooth one, and :func:`box_violation` supplies the
push-back the softclip's far field cannot.

Every function here is a pure ``Array -> Array``, safe under ``jit``,
``vmap``, and ``grad``. Hinges are built from ``jnp.maximum(., 0.0) ** 2``
rather than ``abs`` or ``sqrt`` deliberately: the squared hinge is ``C^1``
(so an adaptive ODE controller does not chatter at the crossing) and
pole-free (so it needs none of the double-``where`` guarding that
``losses.py`` applies around ``log`` and division).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

if TYPE_CHECKING:
    from hybridmodels.predictors import BoundedPredictor

__all__ = (
    "soft_logit",
    "softclip",
    "clip_ste",
    "box_violation",
    "collocation_grids",
    "bound_penalty",
)


def _bounded_leaves(predictors: Any) -> list[BoundedPredictor]:
    """Every ``BoundedPredictor`` in ``predictors``, in pytree traversal order.

    Imported lazily to keep the dependency one-way: ``predictors.base``
    imports this module for :func:`soft_logit`, so a module-level import
    back would be circular.

    Uses the same ``is_leaf``-stopped traversal as
    ``reinitialize_pytree_with_key`` and ``freeze_modules_of_type``, which
    is what makes leaf *order* consistent across all three and lets
    :func:`collocation_grids` return a plain positional tuple.
    """
    from hybridmodels.predictors import BoundedPredictor

    is_bp = lambda node: isinstance(node, BoundedPredictor)  # noqa: E731
    return [leaf for leaf in jtu.tree_leaves(predictors, is_leaf=is_bp) if is_bp(leaf)]


def soft_logit(s: Array, eps: float = 1e-3) -> Array:
    """``logit(s)``, extended linearly outside ``[eps, 1 - eps]``.

    Exact — value *and* derivative — for ``s`` inside the threshold band,
    and ``C^1`` across the junction, because the continuation uses
    ``logit`` 's own slope at the crossing point. Outside the band the
    result therefore grows linearly instead of blowing up at the pole, and
    the derivative is a finite constant instead of zero.

    This replaces the ``logit(jnp.clip(s, eps, 1 - eps))`` idiom, whose
    derivative outside the band is exactly zero — see the module docstring
    for why a mid-graph zero derivative is a silent correctness bug rather
    than a numerical nuisance.

    A plain :func:`softclip` cannot be used for this. Its error in the
    interior is ``O(1 / beta)`` in the units of ``s``, and ``s`` here is
    normalised to ``[0, 1]``, so any ``beta`` gentle enough to retain
    gradient far outside the box also visibly distorts the middle of it.
    The two-regime construction sidesteps the trade-off entirely: no
    interior distortion at any threshold.

    ``eps`` sets the continuation slope, which is ``1 / (eps * (1 - eps))``
    — roughly ``1 / eps``. That is the one real tuning knob: too small and
    a modest excursion maps to an enormous latent (``eps = 1e-6`` sends a
    1% overshoot to ``|z| ~ 1e4``, which then saturates or overflows the
    inner network); too large and the exact-interior band shrinks. The
    default ``1e-3`` maps a 1% overshoot to ``|z| ~ 10`` — firmly outside
    the sigmoid's linear region, so the push-back is felt, but still a
    number a network can consume.
    """
    s_clamped = jax.lax.stop_gradient(jnp.clip(s, eps, 1.0 - eps))
    value = jax.scipy.special.logit(s_clamped)
    slope = 1.0 / (s_clamped * (1.0 - s_clamped))
    # Inside the band ``s - s_clamped`` is zero, so this reduces to
    # ``logit(s)`` with its true derivative; ``stop_gradient`` on the clamp
    # is what keeps the correction term's derivative equal to ``slope``
    # rather than picking up a spurious contribution through the clip.
    return value + slope * (s - s_clamped)


def softclip(x: Array, lo: float | Array, hi: float | Array, beta: float = 20.0) -> Array:
    """Smoothly clamp ``x`` into ``[lo, hi]`` with a derivative that never hits zero.

    Built from two softplus shoulders, so the derivative is
    ``sigmoid(beta * (x - lo)) - sigmoid(beta * (x - hi))``, which lies in
    ``(0, 1)`` analytically. The interior is reproduced to ``O(1 / beta)``
    and the output asymptotes to the bounds rather than meeting them.

    ``beta`` trades interior fidelity against far-field gradient: larger
    values track a hard clip more closely but decay to underflow sooner
    outside the box. The default of ``20`` keeps the interior error below
    roughly ``0.05`` box widths while retaining usable gradient about one
    width out.

    This is a *near-field* repair only. Several widths outside the box the
    derivative underflows just as a hard clip's does, so ``softclip``
    should be paired with :func:`box_violation` whenever the input can
    stray far — the hinge is what supplies unbounded push-back.
    """
    scaled_softplus = lambda u: jax.nn.softplus(beta * u) / beta  # noqa: E731
    return lo + scaled_softplus(x - lo) - scaled_softplus(x - hi)


def clip_ste(x: Array, lo: float | Array, hi: float | Array) -> Array:
    """Hard-clip on the forward pass, identity on the backward pass.

    The straight-through estimator: use it when downstream code genuinely
    requires a feasible number (a concentration that must not go negative
    before a ``log``, say) but the task loss should keep flowing as though
    the clip were not there.

    The identity gradient is a deliberate fiction. It propagates whatever
    the data loss asks for, including "keep going further out of bounds",
    indefinitely — a straight-through clip *never* pushes back on its own.
    Pair it with :func:`box_violation` on the pre-clip value, which is the
    term that actually supplies the restoring force.
    """
    return x + jax.lax.stop_gradient(jnp.clip(x, lo, hi) - x)


def box_violation(x: Array, lows: Array, highs: Array) -> Array:
    """Width-normalised squared hinge measuring how far ``x`` falls outside its box.

    Returns a scalar; exactly zero (value *and* gradient) strictly inside
    the box, so the penalty never perturbs the feasible interior. Outside,
    it grows quadratically, giving a restoring gradient that is linear in
    the overshoot and therefore does not vanish the way a reparameterised
    bound's does.

    Normalising each component by its own width ``high - low`` is what
    makes a single penalty weight portable: bounds in this package range
    from fractions of a unit to hundreds of kelvin, and an unnormalised
    hinge would let the widest channel dominate the term purely through
    its units.

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
    """Build one input grid per ``BoundedPredictor`` leaf of ``predictors``.

    Each grid is a tensor product of ``n_per_dim`` evenly spaced points
    across every dimension of that predictor's ``in_scaler.bounds``, so it
    has shape ``[n_per_dim ** n_inputs, n_inputs]`` and spans the box the
    predictor declares it covers.

    Call this once, on the host, before the training loop: the grids
    depend only on static ``bounds``, so recomputing them per step would
    add trace work for a constant. The returned tuple is positional and
    aligns with the leaf order :func:`bound_penalty` walks, which is the
    same ``jtu.tree_leaves(..., is_leaf=...)`` order used by
    ``reinitialize_pytree_with_key`` and ``freeze_modules_of_type``.

    A deterministic grid is preferred over random sampling because
    ``restore_best`` compares raw loss values across steps; a penalty that
    resampled each step would make that comparison noisy and could pick a
    "best" that merely drew an easy sample.

    Note the exponential growth in input dimension — ``n_per_dim=5`` over
    six inputs is 15625 points. Predictors in this package take two or
    three named inputs, where the grid is 25 to 125 forward passes and
    negligible beside an ODE solve; lower ``n_per_dim`` if that stops
    being true.

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

    This is the *default* way to penalise bound behaviour in this package,
    and it is deliberately trajectory-blind. Saturation is a property of
    the predictor as a function on its declared input box, not of any
    particular solve, so it needs no cooperation from ``simulate_fn``,
    ``predict_bucket``, or the loss protocol — none of their signatures
    change. It is equally correct whether the predictor is called inside a
    vector field or hoisted above one, because it never observes the call
    site. And because it walks the pytree by leaf, arbitrary nesting works
    for free (ADR-0006).

    The flip side of trajectory-blindness: it reports saturation anywhere
    in the declared box, including regions the training trajectories never
    visited. For catching extrapolation failure before deployment that is
    a feature. For "did this particular solve push an input out of range",
    it is the wrong instrument — that question needs the penalty computed
    where the state actually is.

    Parameters
    ----------
    predictors : PyTree[eqx.Module]
        Any pytree shape; only ``BoundedPredictor`` leaves contribute.
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
