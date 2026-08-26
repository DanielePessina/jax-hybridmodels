"""Predictor primitives and composition wrappers.

The framework prefers composition over inheritance: ``Predictor`` is an
abstract marker for trainable ``Array -> Array`` modules, concrete
predictors (``MLPPredictor``, ``KANPredictor``, ...) are final, and
shared concerns — bound-aware input/output scaling, named-input ordering
— live in standalone wrappers (``BoundScaler``, ``BoundedPredictor``)
that hold a ``Predictor`` as a field. Adding a new predictor family is
therefore "subclass ``Predictor`` and implement ``__call__``"; the
wrapping pieces stay reusable.

Named-input ordering is carried by ``BoundedPredictor.input_keys`` (a
static ``tuple[str, ...]``) rather than a separate selector module. The
keys travel through ``eqx.tree_serialise_leaves`` as static metadata, so a
saved predictor remains self-describing — reload sites know which inputs
the predictor expects, in which order, without consulting external code.
The earlier draft modelled this as a standalone ``CovariateSelector``
``eqx.Module`` composed inside ``BoundedPredictor``; that class held no
trainable leaves and a single stack operation, so it was folded into the
parent. ``__call__`` is polymorphic — it accepts a ``dict[str, Array]``
(subset extraction in ``input_keys`` order) or a rank-1 ``Array`` (passed
through), letting users construct inputs in either named or positional
form at the vector-field boundary.

Multi-rate models (e.g. simultaneous nucleation and growth rates)
compose as a *tuple* of predictors that the user unpacks at the top of
their ``simulate_fn``. There is deliberately no framework "rate-pair"
wrapper class — keeping the unpacking explicit at the simulator
boundary means each predictor's role is named in user code, and adding
a third rate is just appending to the tuple.

The trainable component handed to training kernels is therefore a
``PyTree[eqx.Module]``: a tuple by convention, but any pytree shape
(list, dict, NamedTuple, single Module) works. The re-initialisation
helper :func:`reinitialize_pytree_with_key` walks that pytree and
gives each ``eqx.Module`` leaf an independent subkey, so identical-shape
sibling predictors get genuinely different fresh weights when the
training tournament restarts a run.
"""

# ruff: noqa: F722

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array

from hybridmodels.penalties import box_violation, soft_logit

# Public re-export surface from this module.
__all__ = (
    "Predictor",
    "BoundScaler",
    "BoundedPredictor",
    "reinitialize_with_key",
    "reinitialize_pytree_with_key",
)

_SUPPORTED_TRANSFORMS: tuple[str, ...] = ("sigmoid",)


class Predictor(eqx.Module):
    """Abstract marker for trainable Array -> Array modules.

    Concrete subclasses (MLPPredictor, KANPredictor, ...) implement `__call__`
    with signature ``Float[Array, "in"] -> Float[Array, "out"]``. The
    composition wrapper (`BoundedPredictor`) holds a `Predictor` as a field
    and exposes a richer call signature (``dict[str, Array] | Array -> Array``)
    without subclassing it.

    Multi-rate models do not need a framework wrapper: multiple
    predictors compose as a tuple at the ``simulate_fn`` boundary and
    the user unpacks them at the top of the vector field, naming each
    one in their own code (``rate_growth, rate_nucleation = predictors``).
    """

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError("Predictor is abstract; concrete subclasses implement __call__")


class BoundScaler(eqx.Module):
    """Bidirectional sigmoid scaler between physical ``[low, high]`` and an unbounded latent.

    The trainable inner predictor sees no bounds and outputs an
    unbounded latent value; this scaler translates between that latent
    and the physical box the simulator actually needs.

    The forward map is
    ``physical -> latent = logit((x - low) / (high - low)) * T``; the
    inverse is
    ``latent -> physical = low + (high - low) * sigmoid(z / T)``.
    Composing inverse with forward is the identity strictly inside the
    open box ``(low, high)``; outside a narrow band at the endpoints
    ``to_latent`` switches to a linear continuation (see method doc), so
    the round trip deviates there rather than saturating.

    Bounds are enforced by *construction* — the inner predictor emits an
    unbounded latent and ``from_latent`` squashes it — so a physical
    violation is unrepresentable and there is nothing to clip. What that
    costs is gradient: the squash derivative decays exponentially, so a
    predictor pinned against a bound has no signal left to pull it back.
    :meth:`saturation` and :meth:`input_violation` are the optional
    penalty queries that repair the two ends of that problem; both are
    pure and neither is invoked by ``__call__``.

    Sigmoid is the only transform supported here; alternative transforms
    can be introduced by extending ``_SUPPORTED_TRANSFORMS`` and adding
    matching forward/inverse maps.

    The temperature ``T`` is a leaf, not a static field, so it could in
    principle be trained. The recommended convention is to freeze it
    (e.g. via ``freeze_modules_of_type(mask, predictor, BoundScaler)``)
    because the scaler is meant to define the activation shape, not
    learn it; leaving it trainable shifts the gradient signal between
    the scaler and the inner predictor and tends to slow convergence.

    Attributes
    ----------
    bounds : tuple[tuple[float, float], ...]
        Per-component ``(low, high)`` pairs. Length sets the I/O dimension;
        applies elementwise to the last axis of inputs.
    transform : str
        Name of the scaling transform; ``"sigmoid"`` is currently the
        only supported value.
    temperature : Array
        Scalar (or per-component) sharpness multiplier in latent space.
        ``T = 1.0`` recovers the standard logit/sigmoid pair.
    logit_eps : float
        Static. Half-width of the band at each end of ``[0, 1]`` outside
        which ``to_latent`` continues linearly instead of running into
        ``logit``'s pole. Sets the continuation slope (``~1 / logit_eps``).
    z_knee : float
        Static. Latent magnitude past which :meth:`saturation` starts
        charging. ``3.0`` is the outer 5% of the physical box.
    """

    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    transform: str = eqx.field(static=True)
    temperature: Array
    logit_eps: float = eqx.field(static=True)
    z_knee: float = eqx.field(static=True)

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        transform: str = "sigmoid",
        temperature: Any = 1.0,
        logit_eps: float = 1e-3,
        z_knee: float = 3.0,
    ) -> None:
        if transform not in _SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"Unsupported transform {transform!r}; "
                f"supported transforms: {list(_SUPPORTED_TRANSFORMS)}"
            )
        self.bounds = tuple((float(low), float(high)) for low, high in bounds)
        self.transform = transform
        self.temperature = jnp.asarray(temperature)
        self.logit_eps = float(logit_eps)
        self.z_knee = float(z_knee)

    def _lows_highs(self) -> tuple[Array, Array]:
        """Return ``bounds`` as two parallel ``[len(bounds)]`` arrays of lows and highs."""
        lows = jnp.asarray([b[0] for b in self.bounds])
        highs = jnp.asarray([b[1] for b in self.bounds])
        return lows, highs

    def to_latent(self, x: Array) -> Array:
        """Map a physical-space value to its latent representative.

        Steps: normalise to ``[0, 1]`` against ``bounds``, apply
        :func:`~hybridmodels.penalties.soft_logit`, scale by ``temperature``.

        The ``logit`` pole guard is a linear continuation, not a hard
        ``jnp.clip``. A hard clip has *exactly* zero derivative outside the
        box, and because this guard sits mid-graph that zero propagates to
        every upstream parameter on the path. Predictor inputs are
        routinely state-derived — supersaturation in the crystallisation
        example is a traced function of the ODE state — so a clipped input
        silently drops a real sensitivity out of the ODE adjoint with
        nothing raised and nothing logged.

        Inside ``[logit_eps, 1 - logit_eps]`` the map is *exactly* the old
        ``logit``, value and derivative both, so models trained before this
        change keep their numerics wherever they were behaving. Outside it,
        the map continues linearly at ``logit``'s own slope at the
        crossing: finite values, constant non-zero gradient, ``C^1`` across
        the junction so an adaptive ODE controller sees no kink.

        The continuation still only reports *direction*, not magnitude —
        it cannot tell a small excursion from a catastrophic one in a way a
        loss can act on. Pair it with :meth:`input_violation` when an input
        can leave its declared box.
        """
        lows, highs = self._lows_highs()
        normalized = (x - lows) / (highs - lows)
        return soft_logit(normalized, self.logit_eps) * self.temperature

    def input_violation(self, x: Array) -> Array:
        """Scalar squared hinge on how far ``x`` fell outside ``bounds``.

        Zero in value *and* gradient strictly inside the box, so adding it
        to a loss never perturbs the feasible interior. Outside, it grows
        quadratically in the width-normalised overshoot.

        This is the push-back half of the pair whose forward half is the
        softclip in :meth:`to_latent`: the softclip keeps the forward pass
        finite and differentiable near the box, this term supplies a
        restoring force that keeps working arbitrarily far from it.

        Pure and side-effect free — emitting a penalty is a separate query,
        never a side effect of calling the scaler, which is what lets the
        caller decide whether and where to pay for it.
        """
        lows, highs = self._lows_highs()
        return box_violation(x, lows, highs)

    def saturation(self, z: Array) -> Array:
        """Scalar squared overshoot of ``|z / temperature|`` past ``z_knee``.

        Measures how hard the output squash is pinned against its bound.
        ``z_knee`` defaults to ``3.0``, i.e. ``sigmoid(3) ~ 0.953`` — the
        outer 5% of the physical box on each side.

        Deliberately a function of the *latent*, not of the physical value
        it maps to. ``from_latent``'s derivative carries a ``sigma'(z / T)``
        factor that decays to ``4.5e-5`` by ``|z / T| = 10`` and underflows
        to exactly ``0.0`` past roughly ``15``; a penalty written against
        the physical output inherits that factor on the backward pass and
        so dies exactly where saturation is worst. Reading ``|z| / T``
        directly gives a gradient linear in the overshoot that never
        underflows.

        Reduced with ``mean`` rather than ``sum`` so the term does not
        scale with the number of output components — one penalty weight
        then means the same thing for a one-output and a six-output
        predictor.
        """
        u = jnp.abs(z / self.temperature)
        return jnp.mean(jnp.maximum(u - self.z_knee, 0.0) ** 2)

    def from_latent(self, z: Array) -> Array:
        """Map a latent value back into the physical box ``[low, high]``.

        Apply ``sigmoid(z / temperature)`` to land in ``(0, 1)``, then affine
        rescale to ``[low, high]``. The output is finite for any finite ``z``
        (no clipping required on the inverse direction).
        """
        lows, highs = self._lows_highs()
        return lows + (highs - lows) * jax.nn.sigmoid(z / self.temperature)


class BoundedPredictor(eqx.Module):
    """Composition wrapper: ``in_scaler.to_latent -> inner -> out_scaler.from_latent``.

    The full physical-units forward pass for a bound-scaled predictor.
    The user constructs predictor inputs in the vector field by mixing
    constant covariates with state-derived or exogenous time-dependent
    values (CONTEXT.md "Predictor inputs"). ``__call__`` accepts that
    construction in two equivalent forms:

    - ``dict[str, Array]`` — the dict may carry extra keys; only the
      named subset listed in ``self.input_keys`` is pulled, in declared
      order. Missing keys raise ``KeyError``.
    - ``Array`` (rank-1, length ``len(input_keys)``) — passed through
      after a shape check (``eqx.error_if``). Useful when the user
      prefers to stack positionally at the call site.

    From there, ``in_scaler`` maps each value into the inner network's
    latent input space, the trainable ``Predictor`` runs in unbounded
    latent space, and ``out_scaler`` maps its output back into the
    physical output box. ``inner`` therefore sees no bound information
    and never has to clamp itself.

    Construction
    ------------
    ``input_keys`` is required to match ``len(in_scaler.bounds)`` and
    that length must be at least 1 (a predictor with zero inputs has no
    training signal). When ``input_keys`` is omitted (``None``), the
    constructor auto-fills ``("x1", "x2", ..., "xN")`` so the static
    field is always populated and the saved predictor remains
    self-describing — the user can still call it with a positional
    ``Array`` even if they never wrote a names tuple.

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
        self.input_keys = resolved_keys
        self.in_scaler = in_scaler
        self.inner = inner
        self.out_scaler = out_scaler

    def __call__(self, inputs: dict[str, Array] | Array) -> Array:
        n = len(self.input_keys)
        if isinstance(inputs, dict):
            # Subset extraction: dict may carry extra keys; only input_keys are
            # pulled. Missing keys raise a natural ``KeyError`` from the dict
            # access — eqx.error_if doesn't fit here because it requires an
            # array to attach the check to, and the failure happens before any
            # array is constructed.
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
        z_out = self.inner(z_in)
        return self.out_scaler.from_latent(z_out)


def reinitialize_with_key(predictor: eqx.Module, key: Array) -> eqx.Module:
    """Return a fresh copy of `predictor` with inexact-float leaves re-initialised.

    Single-Module helper. If `predictor` implements the
    ``initialized_with_key`` protocol (a method ``self -> key -> self``
    used by predictor classes that want a custom re-init scheme — for
    example a KAN that needs to rebuild its grid), it is delegated to.
    Otherwise every inexact-array leaf in the pytree is replaced with a
    standard-normal sample of matching shape and dtype; non-inexact
    leaves and static fields are left untouched.

    For re-initialising a *pytree* of predictors (the convention at the
    ``simulate_fn`` boundary — typically a tuple of ``BoundedPredictor``s),
    use :func:`reinitialize_pytree_with_key` so each ``eqx.Module`` leaf
    gets its own independently-derived key.
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

    Used by the training tournament to escape bad initial weights:
    when an attempt diverges or stalls, the loop draws a fresh
    per-attempt key, calls this function, and restarts. Each
    ``eqx.Module`` leaf gets its *own* independent subkey, so two
    sibling predictors with identical shapes still re-init to
    different random weights.

    The split is done by **traversal order**: we count the
    ``eqx.Module`` leaves with a Module-stopped traversal, call
    ``jr.split(key, n_module_leaves)`` once, and hand out subkeys in
    that order. The alternative — folding the per-leaf path string —
    would give path-stable subkeys but cost a hash per leaf and
    produce a less obvious correspondence between subkeys and
    pytree positions; traversal-order splitting is simpler and
    sufficient because the pytree shape is fixed across re-inits
    within a single training run.

    Accepts any pytree shape: the conventional
    ``tuple[BoundedPredictor, ...]``, a bare ``eqx.Module`` (a
    one-leaf pytree, equivalent to calling
    :func:`reinitialize_with_key` directly), ``dict[str, ...]``,
    ``NamedTuple`` subclasses, and any nested combinations
    ``jax.tree_util`` can walk.

    Returns a structurally identical pytree with fresh weights on every
    ``eqx.Module`` leaf.
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
