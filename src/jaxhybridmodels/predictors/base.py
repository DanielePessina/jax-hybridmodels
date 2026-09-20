"""Predictors and the wrappers that compose them.

A *predictor* is a trainable ``eqx.Module`` whose ``__call__`` maps one
array to another. That is the whole contract. A predictor knows nothing
about covariate names, physical units, or bounds.

Everything else is composition rather than inheritance. ``Predictor``
is an abstract marker, concrete predictors (``MLPPredictor``,
``KANPredictor``, ...) are final, and shared concerns live in wrappers
that hold a ``Predictor`` as a field. ``BoundScaler`` maps a physical
range onto an unbounded latent. ``BoundedPredictor`` chains an input
scaler, an inner predictor, and an output scaler. Adding a predictor
family means subclassing ``Predictor`` and writing ``__call__``. No
method is ever overridden.

Named-input ordering lives on ``BoundedPredictor.input_keys``, a static
``tuple[str, ...]``. The keys travel through
``eqx.tree_serialise_leaves`` as static metadata, so a reload site can
read which inputs a saved predictor expects, and in which order, without
consulting the code that built it. ``__call__`` accepts either a
``dict[str, Array]`` (subset extraction in ``input_keys`` order) or a
rank-1 ``Array``.

Multi-rate models (simultaneous nucleation and growth, say) compose as a
*tuple* of predictors the user unpacks at the top of their
``simulate_fn``. There is deliberately no framework "rate-pair" class:
unpacking names each predictor's role in user code, and adding a third
rate is one more tuple entry.

The trainable component handed to training kernels is therefore a
``PyTree[eqx.Module]``: a tuple by convention, but any pytree shape
works. :func:`reinitialize_pytree_with_key` walks that pytree and gives
each ``eqx.Module`` leaf an independent subkey, so identical-shape
siblings get different fresh weights when the tournament restarts a run.
"""

# ruff: noqa: F722

from __future__ import annotations

import math
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array

from jaxhybridmodels.penalties import box_violation, soft_inverse
from jaxhybridmodels.transforms import BOUND_TRANSFORMS, WARPS, check_bounds, warp_bounds

# What this module exports.
__all__ = (
    "Predictor",
    "BoundScaler",
    "BoundedPredictor",
    "reinitialize_with_key",
    "reinitialize_pytree_with_key",
)


class Predictor(eqx.Module):
    """Abstract marker for a trainable ``Array -> Array`` module.

    Carries no behaviour. It exists so the rest of the framework can say
    "this leaf is a trainable function approximator" and so composition
    wrappers have one type to accept. The base ``__call__`` raises.

    To add your own family, subclass ``Predictor``, declare trainable
    arrays as ordinary fields and hyperparameters as
    ``eqx.field(static=True)``, and implement
    ``__call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]``.
    Dynamic leaves must be JAX float arrays and static fields must be
    JSON-encodable, so the module round-trips through
    ``eqx.tree_serialise_leaves``. Optionally implement
    ``initialized_with_key(self, key) -> Self`` to control how the
    tournament restarts your weights; without it,
    :func:`reinitialize_with_key` replaces every float leaf with a
    standard-normal sample, which skews any considered init scheme.

    Do not subclass to add bound handling or named inputs.
    ``BoundedPredictor`` supplies both by composition.
    """

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError("Predictor is abstract; concrete subclasses implement __call__")


class BoundScaler(eqx.Module):
    """Two-way map between a physical range ``[low, high]`` and an unbounded latent.

    Why this exists
    ---------------
    Physical quantities have ranges, and an ODE solver handed a value
    outside one either fails or returns nonsense. An optimiser knows
    none of that; it proposes whatever number lowers the loss. Clipping
    the proposal is not usable, because a clip has exactly zero
    derivative outside the range, so a parameter that leaves the box has
    no gradient to pull it back. This scaler reparameterises instead: the
    inner predictor reads and writes a *latent* value, any real number,
    and the scaler squashes it into the physical range. An out-of-range
    physical value has no latent, so there is nothing to clip.

    Vocabulary
    ----------
    latent
        The unbounded real number the inner predictor works in. Written
        ``z`` below.
    physical
        The value in the units the simulator uses. Always inside
        ``[low, high]``.
    warp
        Decides what "halfway between the bounds" means. ``"linear"``
        puts the midpoint of ``(1e-6, 1e2)`` at 50 and collapses the
        eight-decade low end into a sliver; ``"log10"`` puts it at
        ``1e-2``. Bounds are declared in physical units either way.
        Name-keyed registry in ``transforms.py``.
    squash
        The map from latent onto ``(0, 1)``, applied before the affine
        rescale onto the box. ``"sigmoid"`` (default), ``"algebraic"``,
        ``"softsign"``. Same module, ``BOUND_TRANSFORMS``. Stored as the
        ``transform`` field.
    temperature
        Divides the latent before the squash. A larger ``T`` spreads the
        same box over a wider latent range, so the squash saturates more
        slowly. ``T = 1.0`` is the plain squash.
    knee
        The ``z_knee`` field. Latent magnitude past which
        :meth:`saturation` starts charging. Derived per transform from
        one physical criterion, the outer 5% of the box.

    The two maps
    ------------
    :meth:`to_latent` warps the physical value, normalises it to
    ``[0, 1]`` against the warped bounds, applies the squash inverse, and
    multiplies by ``T``. :meth:`from_latent` inverts that. With the
    default ``"linear"`` warp and ``"sigmoid"`` squash they read
    ``z = logit((x - low) / (high - low)) * T`` and
    ``x = low + (high - low) * sigmoid(z / T)``.

    The round trip is the identity strictly inside the open box. Outside
    a narrow band at the endpoints ``to_latent`` continues linearly (see
    its docstring), so it deviates there rather than hitting a pole.

    The cost
    --------
    Bounds hold by construction; the price is gradient. The squash
    derivative decays as ``|z|`` grows, so a predictor pinned against a
    bound has little signal left to pull it back. Sigmoid's decay is
    exponential and dies at ``z = 16.8`` in float32; ``"algebraic"`` and
    ``"softsign"`` decay polynomially and buy far more runway (numbers in
    ``transforms.py``). Runway alone is not enough, since escape time
    still grows fast with ``|z|``. :meth:`saturation` charges the output
    end for sitting deep in the squash, :meth:`input_violation` charges
    the input end for arriving outside its box. Both are pure queries;
    ``__call__`` invokes neither.

    ``temperature`` is a dynamic leaf, so it could be trained. Freeze it
    instead, for example with
    ``freeze_modules_of_type(mask, predictor, BoundScaler)``. The scaler
    defines the activation shape and the inner predictor learns inside
    it; a trainable ``T`` moves gradient between the two and tends to
    slow convergence.

    Attributes
    ----------
    bounds : tuple[tuple[float, float], ...]
        Per-component ``(low, high)`` pairs in physical units. Length
        sets the input/output dimension and the pairs apply elementwise
        to the last axis. Must be finite and ordered ``low < high``.
    transform : str
        Static. Squash name, a key of ``BOUND_TRANSFORMS``. Register your
        own with ``register_bound_transform``.
    temperature : Array
        Scalar (or per-component) latent sharpness. ``T = 1.0`` recovers
        the plain squash and its inverse.
    warp : str
        Static. Warp name, a key of ``WARPS``. Register your own with
        ``register_warp``.
    warped_bounds : tuple[tuple[float, float], ...]
        Static. ``bounds`` pushed through the warp once at construction.
        Resolved eagerly so no call has to warp the edges under a trace.
    logit_eps : float
        Static. Half-width of the band at each end of ``[0, 1]`` outside
        which ``to_latent`` continues linearly instead of running into
        the squash inverse's pole. Also sets the continuation slope
        (roughly ``1 / logit_eps``).
    z_knee : float
        Static. The knee. Defaults to the transform's own value, which
        for sigmoid is ``2.944`` (``logit(0.95)``), the outer 5% of the
        box. Do not share one number across transforms: 2.944 is 12.5%
        from the bound on softsign, which would charge 2.7x too hard.
    """

    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    transform: str = eqx.field(static=True)
    temperature: Array
    warp: str = eqx.field(static=True)
    warped_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    logit_eps: float = eqx.field(static=True)
    z_knee: float = eqx.field(static=True)

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        transform: str = "sigmoid",
        temperature: Any = 1.0,
        warp: str = "linear",
        logit_eps: float = 1e-3,
        z_knee: float | None = None,
    ) -> None:
        if transform not in BOUND_TRANSFORMS:
            raise ValueError(
                f"Unknown transform {transform!r}; available: "
                f"{sorted(BOUND_TRANSFORMS)}. Register custom squashes via "
                "register_bound_transform(name, transform)."
            )
        if warp not in WARPS:
            raise ValueError(
                f"Unknown warp {warp!r}; available: {sorted(WARPS)}. "
                "Register custom warps via register_warp(name, warp)."
            )
        bounds = tuple((float(low), float(high)) for low, high in bounds)
        check_bounds(bounds, warp)
        temperature_arr = jnp.asarray(temperature)
        # Kick the weak type. ``jnp.asarray(1.0)`` is weak-typed, and JAX's
        # compiled cache keys 0-d leaves by value *and* weak type, so the
        # first optimiser update would flip it strong and force a one-time
        # full retrace of every training kernel. A scalar this framework
        # constructs is strong from birth, so default models never pay that
        # (see ``_has_weak_scalar_trainable`` in ``training/optax.py``).
        if jax.typeof(temperature_arr).weak_type:
            temperature_arr = jnp.asarray(temperature_arr, dtype=temperature_arr.dtype)
        if temperature_arr.ndim not in (0, 1) or (
            temperature_arr.ndim == 1 and temperature_arr.shape[0] != len(bounds)
        ):
            raise ValueError(
                "BoundScaler temperature must be a positive scalar or a vector "
                f"with one entry per bound; got shape {temperature_arr.shape}."
            )
        try:
            valid_temperature = bool(
                jnp.all(jnp.isfinite(temperature_arr)) & jnp.all(temperature_arr > 0.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BoundScaler temperature must be finite and strictly positive"
            ) from exc
        if not valid_temperature:
            raise ValueError("BoundScaler temperature must be finite and strictly positive")
        if not math.isfinite(float(logit_eps)) or not 0.0 < float(logit_eps) < 0.5:
            raise ValueError(
                "BoundScaler logit_eps must be finite and lie strictly between 0 and 0.5"
            )
        self.bounds = bounds
        self.transform = transform
        self.warp = warp
        # Resolved once here, not per call: jnp ops inside a jit trace are
        # staged out, so warping the edges lazily would hand back tracers.
        self.warped_bounds = warp_bounds(bounds, warp)
        self.temperature = temperature_arr
        self.logit_eps = float(logit_eps)
        # Resolved to a float now, not looked up per call. The static field
        # stays JSON-friendly, and a saved scaler keeps the knee it trained
        # with even if the registry default later changes.
        self.z_knee = float(BOUND_TRANSFORMS[transform].knee) if z_knee is None else float(z_knee)
        if not math.isfinite(self.z_knee) or self.z_knee < 0.0:
            raise ValueError("BoundScaler z_knee must be finite and non-negative")

    def _lows_highs(self) -> tuple[Array, Array]:
        """Box edges in *physical* units, as two ``[len(bounds)]`` arrays."""
        lows = jnp.asarray([b[0] for b in self.bounds])
        highs = jnp.asarray([b[1] for b in self.bounds])
        return lows, highs

    def _warped_edges(self) -> tuple[Array, Array]:
        """Box edges in *warped* coordinates, as two ``[len(bounds)]`` arrays."""
        lows = jnp.asarray([b[0] for b in self.warped_bounds])
        highs = jnp.asarray([b[1] for b in self.warped_bounds])
        return lows, highs

    def to_latent(self, x: Array) -> Array:
        """Map a physical value to its latent representative.

        Warp ``x``, normalise it to ``[0, 1]`` against the warped bounds,
        apply the squash inverse through
        :func:`~jaxhybridmodels.penalties.soft_inverse`, multiply by
        ``temperature``.

        The squash inverse has a pole at each end of ``[0, 1]``, guarded by
        a linear continuation rather than a hard ``jnp.clip``. A clip has
        zero derivative outside the box, and since the guard sits mid-graph
        that zero propagates to every upstream parameter. Predictor inputs
        are often state-derived (supersaturation in the crystallisation
        example), so a clipped input silently drops a real sensitivity from
        the adjoint.

        Inside ``[logit_eps, 1 - logit_eps]`` the map is exactly the plain
        inverse in value and derivative. Outside it continues linearly at
        the inverse's slope at the crossing: finite values, non-zero
        constant gradient, and a C^1 join so an adaptive step controller
        sees no kink.

        The continuation reports direction, not magnitude. Pair it with
        :meth:`input_violation` when an input can leave its box.
        """
        warp = WARPS[self.warp]
        if warp.requires_positive:
            nonpositive = jnp.any(x <= 0.0)
            try:
                if bool(nonpositive):
                    raise ValueError(
                        f"BoundScaler warp={self.warp!r} requires strictly positive runtime inputs"
                    )
            except jax.errors.TracerBoolConversionError:
                x = eqx.error_if(
                    x,
                    nonpositive,
                    f"BoundScaler warp={self.warp!r} requires strictly positive runtime inputs",
                )
        lows, highs = self._warped_edges()
        normalized = (warp.forward(x) - lows) / (highs - lows)
        t = BOUND_TRANSFORMS[self.transform]
        return soft_inverse(normalized, t.inverse, t.inverse_slope, self.logit_eps) * (
            self.temperature
        )

    def input_violation(self, x: Array) -> Array:
        """Scalar squared hinge on how far ``x`` fell outside ``bounds``.

        Zero in value *and* gradient strictly inside the box, so adding it
        to a loss never perturbs the feasible interior. Outside, it grows
        quadratically in the width-normalised overshoot.

        The push-back half of a pair: :meth:`to_latent` keeps the forward
        pass finite and differentiable near the box, this term supplies a
        restoring force that keeps working far outside it. Pure, so the
        caller decides whether and where to pay for it.
        """
        lows, highs = self._lows_highs()
        return box_violation(x, lows, highs)

    def saturation(self, z: Array) -> Array:
        """Scalar squared overshoot of ``|z / temperature|`` past ``z_knee``.

        Measures how hard the output squash is pinned against its bound.
        For sigmoid the knee is ``2.944``, where ``sigmoid(2.944) = 0.95``,
        the outer 5% of the physical box on each side.

        The argument is the latent, never the physical value it maps to.
        ``from_latent``'s derivative carries a ``sigma'(z / T)`` factor that
        underflows to 0.0 past ``|z / T| ~ 15``, so a penalty written
        against the physical output dies exactly where saturation is worst.
        Reading ``|z| / T`` gives a gradient linear in the overshoot.

        Reduced with ``mean``, not ``sum``, so one weight means the same for
        a one-output and a six-output predictor.
        """
        u = jnp.abs(z / self.temperature)
        return jnp.mean(jnp.maximum(u - self.z_knee, 0.0) ** 2)

    def from_latent(self, z: Array) -> Array:
        """Map a latent value back into the physical box ``[low, high]``.

        Squash ``z / temperature`` into ``(0, 1)``, affine rescale onto the
        warped box, then unwarp. The result is finite and inside the box for
        any finite ``z``, so this direction needs no guard.
        """
        warp = WARPS[self.warp]
        lows, highs = self._warped_edges()
        squashed = BOUND_TRANSFORMS[self.transform].forward(z / self.temperature)
        return warp.inverse(lows + (highs - lows) * squashed)


class BoundedPredictor(Predictor):
    """A predictor in physical units, built from a network that never sees a bound.

    This is the thing a user's vector field calls. It takes named inputs
    in physical units, returns an output in physical units, and keeps
    both inside their declared ranges.

    Three stages
    ------------
    1. **Normalise the input.** ``in_scaler.to_latent`` maps each input
       from its physical range onto an unbounded latent, so the network
       receives numbers of comparable size whether the input was a
       temperature in the tens or a concentration in the thousandths.
    2. **Run the network.** The trainable ``inner`` ``Predictor`` maps
       latent to latent. It is handed no bound information at all.
    3. **Squash the output.** ``out_scaler.from_latent`` maps the
       network's unbounded output into the physical output box.

    The network is better off never seeing a bound. Given one it would
    have to enforce the range itself, with either a clip (zero gradient
    outside, so a parameter that leaves cannot come back) or a final
    squash it would have to learn to aim. Moving the squash into
    ``out_scaler`` makes an out-of-range output unrepresentable, leaves
    ``inner`` free to be any ``Array -> Array`` function, and lets the
    same network be reused under different bounds.

    Calling it
    ----------
    The user builds predictor inputs in the vector field by mixing
    constant covariates with state-derived or exogenous time-dependent
    values (CONTEXT.md, "Predictor inputs"). ``__call__`` accepts:

    - ``dict[str, Array]``. Extra keys are allowed; only the subset named
      in ``self.input_keys`` is pulled, in declared order. A missing key
      raises ``KeyError``.
    - ``Array``, rank-1 of length ``len(input_keys)``, shape-checked.

    Construction
    ------------
    ``input_keys`` must match ``in_scaler.bounds`` in length, and that
    length must be at least 1: a zero-input predictor has no training
    signal. Omitting it auto-fills ``("x1", ..., "xN")``, so a saved
    predictor always describes its own input contract.

    Attributes
    ----------
    input_keys : tuple[str, ...]
        Static. Declared order of the physical-units inputs; one entry per
        ``in_scaler.bounds`` row. Drives subset extraction when ``__call__``
        receives a dict.
    in_scaler : BoundScaler
        Maps physical-space inputs into the inner network's latent space.
    inner : Predictor
        Trainable Array -> Array module operating in latent space.
    out_scaler : BoundScaler
        Maps the inner network's latent output back to physical units.
    """

    input_keys: tuple[str, ...] = eqx.field(static=True)
    in_scaler: BoundScaler
    inner: Predictor
    out_scaler: BoundScaler

    def __init__(
        self,
        *,
        in_scaler: BoundScaler,
        inner: Predictor,
        out_scaler: BoundScaler,
        input_keys: tuple[str, ...] | None = None,
    ) -> None:
        n = len(in_scaler.bounds)
        if n < 1:
            raise ValueError(
                "BoundedPredictor requires at least one input "
                "(in_scaler.bounds is empty); a zero-input predictor has no training signal."
            )
        if input_keys is None:
            resolved_keys: tuple[str, ...] = tuple(f"x{i + 1}" for i in range(n))
        else:
            resolved_keys = tuple(input_keys)
            if len(resolved_keys) != n:
                raise ValueError(
                    f"input_keys length {len(resolved_keys)} does not match "
                    f"in_scaler.bounds length {n}; they must agree."
                )
        inner_in_size = getattr(inner, "in_size", None)
        if inner_in_size is not None and int(inner_in_size) != n:
            raise ValueError(
                "BoundedPredictor inner input dimension must match in_scaler.bounds; "
                f"got inner.in_size={inner_in_size} and {n} input bounds."
            )
        inner_out_size = getattr(inner, "out_size", None)
        n_outputs = len(out_scaler.bounds)
        if inner_out_size is not None and int(inner_out_size) != n_outputs:
            raise ValueError(
                "BoundedPredictor inner output dimension must match out_scaler.bounds; "
                f"got inner.out_size={inner_out_size} and {n_outputs} output bounds."
            )
        self.input_keys = resolved_keys
        self.in_scaler = in_scaler
        self.inner = inner
        self.out_scaler = out_scaler

    def __call__(self, inputs: dict[str, Array] | Array) -> Array:
        n = len(self.input_keys)
        if isinstance(inputs, dict):
            # A missing key raises KeyError from the dict access itself.
            # eqx.error_if does not fit here: it needs an array to attach the
            # check to, and the failure happens before any array exists.
            x = jnp.stack([jnp.asarray(inputs[k]) for k in self.input_keys])
        else:
            x = jnp.asarray(inputs)
            # Array path contract: rank-1 with length matching input_keys. The
            # predicate is Python-static (shapes are known at trace time), so
            # this raises at trace time under jit.
            x = eqx.error_if(
                x,
                x.ndim != 1 or x.shape[0] != n,
                f"BoundedPredictor expected rank-1 array of length {n} "
                f"(matching input_keys={self.input_keys}); got shape {x.shape}.",
            )
        z_in = self.in_scaler.to_latent(x)
        z_out = jnp.asarray(self.inner(z_in))
        if z_out.ndim == 0 and len(self.out_scaler.bounds) == 1:
            z_out = jnp.reshape(z_out, (1,))
        z_out = eqx.error_if(
            z_out,
            z_out.ndim != 1 or z_out.shape[0] != len(self.out_scaler.bounds),
            "BoundedPredictor inner output must be rank-1 with one entry per "
            f"out_scaler bound; got shape {z_out.shape}.",
        )
        return self.out_scaler.from_latent(z_out)

    def initialized_with_key(self, key: Array) -> BoundedPredictor:
        """Re-initialise ``inner`` only, leaving both scalers untouched.

        Without this method, ``reinitialize_with_key`` takes its generic
        branch and replaces every inexact leaf, ``BoundScaler.temperature``
        included, with a sample from ``N(0, 1)``. A temperature near zero
        or negative inverts and blows up both ``to_latent`` and
        ``from_latent``.

        Delegating to the free function also restores the inner predictor's
        own init scheme rather than leaf-level normal sampling. The scalers
        hold bound geometry, not learned state, so a restart has no reason
        to touch them.
        """
        return eqx.tree_at(
            lambda bp: bp.inner,
            self,
            reinitialize_with_key(self.inner, key),
        )


def reinitialize_with_key(predictor: eqx.Module, key: Array) -> eqx.Module:
    """Return a fresh copy of ``predictor`` with its float leaves re-initialised.

    Single-Module helper. If ``predictor`` implements
    ``initialized_with_key`` (a ``self -> key -> self`` method for classes
    that want their own re-init scheme, such as a KAN that has to rebuild
    its grid), this delegates to it. Otherwise every inexact-array leaf is
    replaced with a standard-normal sample of matching shape and dtype.

    For a *pytree* of predictors, the convention at the ``simulate_fn``
    boundary, use :func:`reinitialize_pytree_with_key`.
    """
    if hasattr(predictor, "initialized_with_key"):
        return cast(eqx.Module, predictor.initialized_with_key(key))

    leaves, treedef = jtu.tree_flatten(predictor)
    inexact_indices = [i for i, leaf in enumerate(leaves) if eqx.is_inexact_array(leaf)]
    if not inexact_indices:
        return predictor
    subkeys = jr.split(key, len(inexact_indices))
    new_leaves = list(leaves)
    for k, idx in zip(subkeys, inexact_indices, strict=True):
        leaf = leaves[idx]
        new_leaves[idx] = jr.normal(k, leaf.shape, dtype=leaf.dtype)
    return cast(eqx.Module, jtu.tree_unflatten(treedef, new_leaves))


def reinitialize_pytree_with_key(predictors: Any, key: Array) -> Any:
    """Per-``eqx.Module``-leaf re-initialisation across a ``predictors`` pytree.

    Used by the training tournament to escape bad initial weights: when
    an attempt diverges or stalls, the loop draws a fresh per-attempt key,
    calls this, and restarts. Each ``eqx.Module`` leaf gets its own
    independent subkey, so identical-shape siblings re-init differently.

    Subkeys are handed out in traversal order, which is enough because the
    pytree shape is fixed across re-inits within one run. Path-derived
    subkeys would be path-stable at the cost of a hash per leaf.

    Accepts any pytree shape ``jax.tree_util`` can walk, including a bare
    ``eqx.Module``. Returns a structurally identical pytree with fresh
    weights on every ``eqx.Module`` leaf.
    """

    def _is_module(node: Any) -> bool:
        return isinstance(node, eqx.Module)

    module_leaves = [
        leaf for leaf in jtu.tree_leaves(predictors, is_leaf=_is_module) if _is_module(leaf)
    ]
    n_modules = len(module_leaves)
    if n_modules == 0:
        return predictors

    subkeys = jr.split(key, n_modules)
    keys_iter = iter(subkeys)

    def _per_leaf(node: Any) -> Any:
        if _is_module(node):
            return reinitialize_with_key(node, next(keys_iter))
        return node

    return jtu.tree_map(_per_leaf, predictors, is_leaf=_is_module)
