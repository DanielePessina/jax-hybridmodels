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

The bound penalty (:func:`bound_penalty`) is charged at *points*, not on a
box sweep. The default points are the measured ones: the input vectors the
loss actually sees at observed cells, gathered from the dataset by
:func:`data_penalty_points` and following the same length-mask prefix as
the loss (:func:`length_mask_keep`). User-supplied penalty-only points
(no measurements needed) extend coverage where it matters — a deployment
region, a future operating point — and :func:`box_grid` is the
collocation-as-extension recipe: a deterministic sweep of the input box,
uniform in *warped* coordinates so a log warp covers decades evenly.

Everything here is pure and safe under ``jit``, ``vmap`` and ``grad``.
Hinges are ``jnp.maximum(., 0.0) ** 2``: C^1, so an adaptive ODE step-size
controller does not chatter at the crossing, and pole-free, so they need
none of the double-``where`` guarding ``losses.py`` applies around ``log``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from hybridmodels.data import Dataset

if TYPE_CHECKING:
    from hybridmodels.predictors import BoundedPredictor

__all__ = (
    "soft_inverse",
    "soft_logit",
    "softclip",
    "clip_ste",
    "box_violation",
    "PenaltyPointSource",
    "box_grid",
    "data_penalty_points",
    "length_mask_keep",
    "select_penalty_points",
    "validate_penalty_points",
    "bound_penalty",
    "attach_penalty_state",
    "penalty_vector_field",
    "strip_penalty_state",
    "penalty_integral",
    "trajectory_saturation_penalty",
)


def _bounded_leaves(predictors: Any) -> list[BoundedPredictor]:
    """Every ``BoundedPredictor`` in ``predictors``, outermost first.

    Recurses into each match rather than stopping at it: nesting is
    supported (``inner`` is typed ``Predictor``), and a traversal that
    stopped at the outer match would leave the inner box unpenalised.

    Order is outer-then-inner, depth first. :func:`data_penalty_points`
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


class PenaltyPointSource(NamedTuple):
    """One ``BoundedPredictor`` leaf's gathered measured points.

    Produced by :func:`data_penalty_points` for leaves whose ``input_keys``
    all resolve to dataset covariates. One entry per *observed cell*: the
    input vector the loss actually sees at that cell, in ``input_keys``
    column order, in physical units.

    Attributes
    ----------
    points : Float[Array, "G n_inputs"]
        The measured input vectors.
    cell_ts : Int[Array, " G"]
        Timestamp index of each cell inside its own experiment.
    cell_T : Int[Array, " G"]
        Length of that experiment's time grid. ``cell_ts`` and ``cell_T``
        let :func:`length_mask_keep` apply the loss's prefix mask to the
        penalty, so the two never disagree about which points are live.
    """

    points: Array
    cell_ts: Array
    cell_T: Array


def box_grid(in_scaler: Any, n_per_dim: int = 5) -> Array:
    """Build a deterministic sweep of an input box, uniform in *warped* coordinates.

    The collocation-as-extension recipe: a tensor product of ``n_per_dim``
    evenly spaced points along every dimension of ``in_scaler``'s box, in
    the box's *warped* coordinates, mapped back to physical units. A linear
    warp therefore reproduces a plain physical ``linspace`` sweep
    bit-for-bit, while ``log`` / ``log10`` warps cover each decade evenly
    instead of starving the low end of the box.

    ``n_per_dim`` must be at least 2 so both edges of every input box are
    represented. Deterministic: it depends only on the scaler's static
    ``bounds`` and ``warp``, so ``restore_best`` compares raw loss values
    across steps with no resampling noise.

    Point count grows exponentially in input dimension. Predictors here take
    two or three inputs, so ``n_per_dim=5`` is 25 to 125 forward passes and
    negligible beside an ODE solve. Lower it if that stops holding.

    Parameters
    ----------
    in_scaler : BoundScaler
        The input scaler of the predictor the sweep is for (``leaf.in_scaler``).

    Returns
    -------
    Array
        ``[n_per_dim ** n_inputs, n_inputs]`` physical points, positional
        in the scaler's input dimension order.
    """
    if n_per_dim < 2:
        raise ValueError(
            "box_grid n_per_dim must be at least 2 "
            f"(one point per box edge); got {n_per_dim}"
        )

    from hybridmodels.transforms import WARPS

    warp = WARPS[in_scaler.warp]
    lows = jnp.asarray([b[0] for b in in_scaler.warped_bounds])
    highs = jnp.asarray([b[1] for b in in_scaler.warped_bounds])
    axes = [
        warp.inverse(jnp.linspace(low, high, n_per_dim))
        for low, high in zip(lows, highs, strict=True)
    ]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.reshape(-1) for m in mesh], axis=-1)


def data_penalty_points(
    predictors: Any, dataset: Dataset
) -> tuple[PenaltyPointSource | None, ...]:
    """Gather one :class:`PenaltyPointSource` per ``BoundedPredictor`` leaf.

    A leaf is *resolvable* when every ``input_keys`` name is a dataset
    covariate (in every bucket). Its measured points are then the input
    vectors at the observed cells — the cells the loss actually charges —
    with a ``None`` entry for an unresolvable leaf. A resolvable leaf whose
    dataset holds no observed cells (all probes) yields a source with zero
    points, which contributes nothing until extras are added.

    Call this once on the host before the training loop: the points depend
    only on the dataset and the leaf's static ``input_keys``.

    Covariates must be scalar per experiment (constant in time, the v1
    contract); a per-experiment covariate with extra dimensions has no
    unambiguous column to feed an input key and raises.
    """
    leaves = _bounded_leaves(predictors)
    if not leaves:
        return ()
    cov_keys = set().union(*(set(bp.covariates) for bp in dataset.bucket_payloads))

    sources: list[PenaltyPointSource | None] = []
    for leaf in leaves:
        keys = leaf.input_keys
        if not all(key in cov_keys for key in keys):
            sources.append(None)
            continue
        point_chunks: list[Array] = []
        ts_chunks: list[Array] = []
        t_chunks: list[Array] = []
        for bp in dataset.bucket_payloads:
            vectors = jnp.stack([bp.covariates[key] for key in keys], axis=-1)
            if vectors.ndim != 2:
                raise ValueError(
                    "penalty: covariates feeding a BoundedPredictor's penalty points "
                    "must be scalar per experiment (constant in time); got shape "
                    f"{vectors.shape} for input_keys {keys!r}"
                )
            observed = jnp.argwhere(bp.mask.any(axis=-1))
            if observed.shape[0] == 0:
                continue
            point_chunks.append(vectors[observed[:, 0]])
            ts_chunks.append(observed[:, 1])
            t_chunks.append(jnp.full(observed.shape[0], bp.ts.shape[1], dtype=observed.dtype))
        if not point_chunks:
            sources.append(
                PenaltyPointSource(
                    points=jnp.zeros((0, len(keys))),
                    cell_ts=jnp.zeros(0, jnp.int32),
                    cell_T=jnp.zeros(0, jnp.int32),
                )
            )
            continue
        sources.append(
            PenaltyPointSource(
                points=jnp.concatenate(point_chunks),
                cell_ts=jnp.concatenate(ts_chunks),
                cell_T=jnp.concatenate(t_chunks),
            )
        )
    return tuple(sources)


def length_mask_keep(source: PenaltyPointSource, length_mask_fraction: float) -> Array:
    """Boolean keep-vector over a source's cells, matching the loss's prefix mask.

    Applies the same prefix semantics :func:`~hybridmodels.training.kernels.apply_length_mask`
    gives the loss: a cell is kept when its timestamp index is below
    ``ceil(T * fraction)``, clamped at 1 so a phase never scores nothing.
    The penalty therefore follows the curriculum exactly, disagreeing with
    the loss about which points are live only by mistake.

    ``length_mask_fraction`` must be a Python float, not a traced array:
    the keep-vector is used to *index* the gathered points, and JAX rejects
    boolean indexing with tracers. The penalty kernel retraces only when
    the fraction changes (a phase boundary), never per step.
    """
    cutoff = jnp.maximum(
        jnp.ceil(jnp.float32(source.cell_T) * length_mask_fraction).astype(jnp.int32),
        jnp.int32(1),
    )
    return source.cell_ts < cutoff


def validate_penalty_points(
    predictors: Any,
    sources: tuple[PenaltyPointSource | None, ...],
    extras: tuple[Array, ...],
    *,
    enabled: bool,
) -> None:
    """Raise when an enabled penalty has a leaf no point set reaches.

    ``sources`` is the per-leaf output of :func:`data_penalty_points`
    (``None`` for unresolvable leaves); ``extras`` the user-supplied
    penalty-only points, positional per leaf. ``enabled`` is whether any
    phase charges the penalty.

    With the penalty on, every ``BoundedPredictor`` leaf must be covered by
    measured points, extras, or both; an uncovered leaf would otherwise be
    a silent no-penalty, the failure mode this penalty exists to prevent.
    For embedded predictors (state-derived inputs) the trajectory penalty
    (ADR-0009) is the instrument, and the error says so.

    Point-shape mismatches are checked unconditionally, so a user cannot
    carry a silently-wrong extras array into a run that later enables it.
    """
    leaves = _bounded_leaves(predictors)
    if not leaves:
        return
    if len(extras) not in (0, len(leaves)):
        raise ValueError(
            f"penalty_points has {len(extras)} entries but predictors holds "
            f"{len(leaves)} BoundedPredictor leaves; provide one point array "
            "per leaf (in traversal order)."
        )
    for idx, leaf in enumerate(leaves):
        extra = extras[idx] if idx < len(extras) else None
        if extra is not None:
            if extra.ndim != 2:
                raise ValueError(
                    f"penalty_points[{idx}] must be a rank-2 [G, n_inputs] array "
                    f"of physical input vectors; got shape {extra.shape}"
                )
            n_inputs = len(leaf.in_scaler.bounds)
            if extra.shape[-1] != n_inputs:
                raise ValueError(
                    f"penalty_points[{idx}] has {extra.shape[-1]} input columns but "
                    f"leaf {idx} (input_keys={leaf.input_keys!r}) takes {n_inputs}; "
                    "columns follow input_keys order."
                )
        if not enabled:
            continue
        covered_by_data = sources[idx] is not None and sources[idx].points.shape[0] > 0
        covered_by_extra = extra is not None and extra.shape[0] > 0
        if not (covered_by_data or covered_by_extra):
            raise ValueError(
                f"penalty: leaf {idx} (input_keys={leaf.input_keys!r}) has no penalty "
                "points: its inputs do not all resolve to dataset covariates and no "
                "entry was given in penalty_points for it. Add penalty_points for "
                "this leaf, or use trajectory_penalty_fn (see ADR-0009) for "
                "embedded predictors."
            )


def select_penalty_points(
    sources: tuple[PenaltyPointSource | None, ...],
    extras: tuple[Array, ...],
    length_mask_fraction: float,
) -> tuple[Array, ...]:
    """Per-leaf point arrays for one phase: length-mask-kept cells ∪ extras.

    ``sources`` is the per-leaf output of :func:`data_penalty_points`
    (``None`` for unresolvable leaves); ``extras`` the user-supplied
    penalty-only points, positional per leaf. Returns one ``[G, n_inputs]``
    array per leaf in traversal order: the measured cells kept by
    :func:`length_mask_keep` concatenated with that leaf's extras.

    This is host-side selection: the keep-vector must be concrete to index
    the gathered points, so call it with the phase's Python float outside
    any jit (the stock trainers do, once per phase). The returned arrays
    are what a ``penalty_step`` receives.
    """
    built: list[Array] = []
    for idx, src in enumerate(sources):
        extra = extras[idx] if idx < len(extras) else None
        if src is not None and src.points.shape[0] > 0:
            kept = src.points[length_mask_keep(src, length_mask_fraction)]
            if extra is not None and extra.shape[0] > 0:
                built.append(jnp.concatenate([kept, extra]))
            else:
                built.append(kept)
        elif extra is not None and extra.shape[0] > 0:
            built.append(extra)
        else:
            built.append(jnp.zeros((0, 0)))
    return tuple(built)


def bound_penalty(predictors: Any, points: tuple[Array, ...]) -> Array:
    """Mean output-squash saturation over every ``BoundedPredictor`` in a pytree.

    For each leaf, evaluates ``inner(in_scaler.to_latent(x))`` across the
    leaf's point set and charges
    :meth:`~hybridmodels.predictors.BoundScaler.saturation` on the
    resulting latents. Leaves are summed.

    ``points`` is positional, one ``[G, n_inputs]`` array per leaf in
    traversal order: measured points from :func:`data_penalty_points`,
    user-supplied penalty-only points, or a :func:`box_grid` sweep. An
    empty per-leaf array contributes exactly zero, so a leaf the penalty
    cannot reach does not NaN a run.

    The default way to penalise bound behaviour here, and deliberately
    evaluated at fixed input points rather than along a solve: saturation
    is a property of the predictor as a function, whatever any particular
    solve does. So it needs no cooperation from ``simulate_fn``,
    ``predict_bucket`` or the loss protocol, behaves the same inside a
    vector field or above one, and handles arbitrary nesting by walking
    leaves.

    That cuts both ways. It reports saturation wherever the point set
    reaches, including regions no training trajectory visited when the
    points say so (a ``box_grid`` sweep, user extras); for "did *this*
    solve push an input out of range" the trajectory-aware penalty
    (ADR-0009) is the right instrument.

    Parameters
    ----------
    predictors : PyTree[eqx.Module]
        Any pytree shape. Only ``BoundedPredictor`` leaves contribute.
    points : tuple[Array, ...]
        One ``[G, n_inputs]`` array per leaf, in traversal order: the
        output of :func:`data_penalty_points` (with any user extras
        concatenated), user penalty-only points, or a :func:`box_grid`
        sweep. An empty per-leaf array contributes zero.

    Returns
    -------
    Array
        Non-negative scalar. Exactly zero when no leaf saturates.
    """
    leaves = _bounded_leaves(predictors)
    if len(leaves) != len(points):
        raise ValueError(
            f"points has {len(points)} entries but predictors holds "
            f"{len(leaves)} BoundedPredictor leaves; rebuild the points "
            "after changing the pytree."
        )
    total = jnp.asarray(0.0)
    for leaf, leaf_points in zip(leaves, points, strict=True):
        if leaf_points.shape[0] == 0:
            continue
        latents = jax.vmap(lambda x, _l=leaf: _l.inner(_l.in_scaler.to_latent(x)))(leaf_points)
        total = total + leaf.out_scaler.saturation(latents)
    return total


# --------------------------------------------------------------------------- #
# Trajectory-aware penalties (embedded hybrid models).
#
# ``bound_penalty`` above is trajectory-blind: it evaluates saturation at
# fixed input points, never along a solve. For an *embedded* hybrid model —
# the predictor runs inside the user's vector field and its output feeds the
# dynamics (a kinetic parameter, a shape factor) — what matters is whether
# the model saturates along the trajectories it actually simulates. The
# penalty then has to be collected inside the solve, as extra ODE state.
#
# The recipe (all helpers here are greppable ``penalty*`` names):
#
#   1. ``attach_penalty_state(y0, n)`` appends ``n`` zero accumulators to
#      the initial state, so ``y0`` becomes ``[S_physics, n]``.
#   2. ``penalty_vector_field(base_rhs, penalty_rhs)`` wraps the physics
#      with a ``penalty_rhs(t, y, args) -> [n]`` giving the per-call rate
#      (e.g. ``[saturation(z), input_violation(x)]``). The solver integrates
#      those rates alongside the physics; the accumulated value is the
#      *time-integral* of the penalty along the trajectory.
#   3. ``strip_penalty_state(state, n)`` drops the accumulators in
#      ``state_to_output`` so the loss sees only the physics.
#   4. A training config's ``trajectory_penalty_fn`` reads
#      ``penalty_integral(full_state, n)`` and returns the scalar to charge.
#
# Only the solver's state widens; nothing about ``simulate_fn``'s signature,
# ``state_to_output``'s signature, or the loss protocol changes.
# --------------------------------------------------------------------------- #


def _validate_penalty_count(n: int) -> None:
    if n < 1:
        raise ValueError(f"penalty accumulator count must be at least one; got {n}")


def attach_penalty_state(y0: Array, n: int = 1) -> Array:
    """Append ``n`` zero-valued penalty accumulators to ``y0``.

    The first step of the trajectory-penalty recipe: the ODE state widens
    from ``[S]`` to ``[S + n]``, where the trailing components are
    integrated penalty rates supplied by :func:`penalty_vector_field`.
    ``y0_fn`` should return ``attach_penalty_state(physics_y0, n)``.
    """
    _validate_penalty_count(n)
    zeros = jnp.zeros((n,), dtype=y0.dtype)
    return jnp.concatenate([jnp.asarray(y0), zeros])


def penalty_vector_field(
    base_rhs: Callable[[Array, Array, Any], Array],
    penalty_rhs: Callable[[Array, Array, Any], Array],
) -> Callable[[Array, Array, Any], Array]:
    """Wrap a physics vector field with per-call penalty rates.

    Returns ``(t, y, args) -> [physics_dot, penalty_rates]``. ``base_rhs``
    is the user's vector field on the physical components; ``penalty_rhs``
    returns the ``n`` penalty rates (e.g. ``[saturation(z),
    input_violation(x)]``) at the current state. The solver integrates both,
    so the trailing accumulators carry the *time-integral* of the penalty
    along the trajectory.

    The rates must read the physical components (typically ``y[:-n]``) and
    are closed over the user's predictors — only the user knows the latent
    ``z`` or input ``x`` of an embedded predictor at call time.
    """

    def wrapped(t: Array, y: Array, args: Any) -> Array:
        rates = jnp.asarray(penalty_rhs(t, y, args))
        if rates.ndim != 1:
            raise ValueError(f"penalty_rhs must return a rank-1 vector; got shape {rates.shape}")
        n = rates.shape[0]
        if n < 1 or n >= y.shape[0]:
            raise ValueError(
                "penalty_rhs must return at least one rate and leave at least one "
                f"physical state component; got {n} rates for state shape {y.shape}."
            )
        phys = base_rhs(t, y[:-n], args)
        return jnp.concatenate([jnp.asarray(phys), rates])

    return wrapped


def strip_penalty_state(state: Array, n: int = 1) -> Array:
    """Drop the trailing ``n`` penalty accumulators from a full-state trajectory.

    Use in ``state_to_output``: the loss and the observed channels should
    see only the physics, not the integrated penalty components. ``state``
    is ``[..., S + n]``; the result is ``[..., S]``.
    """
    _validate_penalty_count(n)
    return state[..., :-n]


def penalty_integral(state: Array, n: int = 1) -> Array:
    """The accumulated (time-integrated) penalty values at the trajectory's end.

    ``state`` is the full state trajectory ``[..., T, S + n]`` as produced by
    a :func:`penalty_vector_field` solve. Returns the trailing ``n``
    components at the final time, ``[..., n]``. The training hook charges
    these; divide by the time span to get the time-mean instead of the
    integral.
    """
    _validate_penalty_count(n)
    return state[..., -1, -n:]


def trajectory_saturation_penalty(state: Array, out_scaler: Any) -> Array:
    """Sum over time of output saturation for a predictor whose output *is* the state.

    For a **parallel** hybrid model — the predictor is outside the solver and
    its output is a predicted channel — the full state already holds the
    physical outputs. Invert them back to latents with the predictor's
    ``out_scaler`` and charge :meth:`BoundScaler.saturation` at every time
    step, summed over time. This is the trajectory-aware counterpart of
    ``bound_penalty`` for the hoisted case: it fires only where the model
    actually predicted, not across a synthetic grid.

    ``state`` is ``[..., T, D]`` (or ``[..., T, S]`` projected to the
    predictor's channel); ``out_scaler`` is the ``BoundScaler`` whose
    ``from_latent`` produced those outputs.
    """
    latents = out_scaler.to_latent(state)
    return jnp.sum(out_scaler.saturation(latents))
