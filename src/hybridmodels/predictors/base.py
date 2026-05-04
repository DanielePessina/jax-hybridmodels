"""Predictor primitives and composition wrappers.

Per SPEC §5.2 and ADR-0001 / ADR-0003 / ADR-0006: `Predictor` is an abstract
marker; concrete predictors are final per Equinox's pattern. Bound-scaling
and covariate selection are decoupled into composition wrappers
(`BoundedPredictor`) rather than baked into a class hierarchy. Multi-rate
models (e.g. nucleation + growth) compose as a tuple of predictors at the
``simulate_fn`` boundary — there is no framework `RatePair` wrapper (R-A6).

The pytree contract for the trainable component (R-A2) lives at the
``simulate_fn`` boundary: the first argument is a `PyTree[eqx.Module]` —
runtime-permissive, with ``tuple`` as the canonical convention shown in
examples. ``reinitialize_pytree_with_key`` (R-T8) is the per-leaf
re-initialiser used by the tournament; it splits the per-attempt key by
traversal order across `eqx.Module` leaves so identical-shape sibling
predictors get *different* re-init weights.
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

# Public re-export surface from this module.
__all__ = (
    "Predictor",
    "CovariateSelector",
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
    and exposes a richer call signature (`dict[str, Array] -> Array`) without
    subclassing it.

    Multi-rate models do not need a framework wrapper (no `RatePair`):
    multiple predictors compose as a tuple at the ``simulate_fn`` boundary
    and the user unpacks them at the top of the vector field. See R-A6.
    """

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError("Predictor is abstract; concrete subclasses implement __call__")


class CovariateSelector(eqx.Module):
    """Pulls a fixed-order subset of named covariates and stacks them into a 1-D array.

    Used as the input adapter inside ``BoundedPredictor``: the experiment's
    covariates dict carries every name (``temperature_C``, ``loading``, ...)
    while a given predictor only consumes a subset in a specific order. The
    static ``keys`` field is what makes that order explicit and serialisable.

    Calling contract
    ----------------
    Input  : ``dict[str, Array]`` — covariates dict with at least ``self.keys``.
    Output : ``Float[Array, "len(keys)"]`` — 0-d covariate values stacked
             along axis 0 in declared order.
    """

    keys: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, covariates: dict[str, Array]) -> Array:
        return jnp.stack([jnp.asarray(covariates[k]) for k in self.keys])


class BoundScaler(eqx.Module):
    """Bidirectional sigmoid scaler between physical ``[low, high]`` and an unbounded latent.

    Sigmoid is the only transform supported in v1 (R-A4 / CONTEXT.md). The
    forward map is ``physical -> latent = logit((x - low) / (high - low)) * T``;
    the inverse is ``latent -> physical = low + (high - low) * sigmoid(z / T)``.
    Composing inverse with forward is the identity strictly inside the open
    box ``(low, high)``; values at or near the closed endpoints are clipped
    by ``to_latent`` (see method doc) so the round-trip can deviate by up to
    one clip width at the extremes.

    The temperature ``T`` is a leaf, not a static field, so it can in
    principle be trained — but every example freezes it via
    ``freeze_modules_of_type(mask, predictor, BoundScaler)``.

    Attributes
    ----------
    bounds : tuple[tuple[float, float], ...]
        Per-component ``(low, high)`` pairs. Length sets the I/O dimension;
        applies elementwise to the last axis of inputs.
    transform : str
        Name of the scaling transform; ``"sigmoid"`` only in v1.
    temperature : Array
        Scalar (or per-component) sharpness multiplier in latent space.
        ``T = 1.0`` recovers the standard logit/sigmoid pair.
    """

    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    transform: str = eqx.field(static=True)
    temperature: Array

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        transform: str = "sigmoid",
        temperature: Any = 1.0,
    ) -> None:
        if transform not in _SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"Unsupported transform {transform!r}; "
                f"supported in v1: {list(_SUPPORTED_TRANSFORMS)}"
            )
        self.bounds = tuple((float(low), float(high)) for low, high in bounds)
        self.transform = transform
        self.temperature = jnp.asarray(temperature)

    def _lows_highs(self) -> tuple[Array, Array]:
        """Return ``bounds`` as two parallel ``[len(bounds)]`` arrays of lows and highs."""
        lows = jnp.asarray([b[0] for b in self.bounds])
        highs = jnp.asarray([b[1] for b in self.bounds])
        return lows, highs

    def to_latent(self, x: Array) -> Array:
        """Map a physical-space value to its latent representative.

        Steps: normalise to ``[0, 1]`` against ``bounds``, clip to
        ``[1e-6, 1 - 1e-6]`` (so ``logit`` does not produce ``±inf`` at the
        closed endpoints), apply ``logit``, scale by ``temperature``. The
        clip is the only source of round-trip error; for inputs strictly
        inside the box it is a no-op.
        """
        lows, highs = self._lows_highs()
        normalized = (x - lows) / (highs - lows)
        clipped = jnp.clip(normalized, 1e-6, 1.0 - 1e-6)
        return jax.scipy.special.logit(clipped) * self.temperature

    def from_latent(self, z: Array) -> Array:
        """Map a latent value back into the physical box ``[low, high]``.

        Apply ``sigmoid(z / temperature)`` to land in ``(0, 1)``, then affine
        rescale to ``[low, high]``. The output is finite for any finite ``z``
        (no clipping required on the inverse direction).
        """
        lows, highs = self._lows_highs()
        return lows + (highs - lows) * jax.nn.sigmoid(z / self.temperature)


class BoundedPredictor(eqx.Module):
    """Composition wrapper: ``selector -> in_scaler.to_latent -> inner -> out_scaler.from_latent``.

    The full physical-units forward pass for a covariate-conditioned predictor
    with bounded inputs and outputs. Inputs are pulled from a *predictor input
    dict* by name (which the user composes inside the vector field — covariates
    plus state-derived plus exogenous time-dependent values; see CONTEXT.md
    "Predictor inputs"), mapped to the inner network's latent input space,
    run through the trainable ``Predictor``, then mapped back into the physical
    output box. ``inner`` sees no bound information and never has to clamp
    itself.

    Calling contract
    ----------------
    Input  : ``dict[str, Array]`` — predictor input dict (any keyed scalars,
             not necessarily covariates; provenance is intentionally invisible
             to this wrapper).
    Output : ``Float[Array, "out"]`` — physical-units prediction; ``out`` is
             determined by ``out_scaler.bounds`` (and the ``inner`` network's
             configured output dimension).
    """

    selector: CovariateSelector
    in_scaler: BoundScaler
    inner: Predictor
    out_scaler: BoundScaler

    def __call__(self, inputs: dict[str, Array]) -> Array:
        x = self.selector(inputs)
        z_in = self.in_scaler.to_latent(x)
        z_out = self.inner(z_in)
        return self.out_scaler.from_latent(z_out)


def reinitialize_with_key(predictor: eqx.Module, key: Array) -> eqx.Module:
    """Return a fresh copy of `predictor` with inexact-float leaves re-initialised.

    Single-Module helper. If `predictor` implements the `initialized_with_key`
    protocol (R-T8), it is delegated to. Otherwise every inexact-array leaf
    in the PyTree is replaced with a standard-normal sample of matching shape
    and dtype; non-inexact leaves and static fields are left untouched.

    For re-initialising a *pytree* of predictors (the convention at the
    `simulate_fn` boundary — typically a tuple of `BoundedPredictor`s), use
    :func:`reinitialize_pytree_with_key` so each `eqx.Module` leaf gets its
    own independently-derived key.
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
    """Per-`eqx.Module`-leaf re-init across a `predictors` pytree (R-T8).

    Used by the tournament loop to escape bad initial weights. Splits ``key``
    by **traversal order** (``jr.split(key, n_module_leaves)``) into one key
    per `eqx.Module` leaf, then calls :func:`reinitialize_with_key` on each
    leaf with its dedicated subkey. Identical-shape sibling predictors get
    *different* re-init weights — this is the (β) split-by-traversal design
    locked during the 2026-05-04 grilling.

    Accepts any pytree shape: the canonical ``tuple[BoundedPredictor, ...]``
    convention, a bare `eqx.Module` (one-leaf pytree, equivalent to calling
    `reinitialize_with_key` directly), `dict[str, ...]`, NamedTuple subclasses,
    nested combinations — anything ``jax.tree_util`` can walk.

    Returns a structurally-identical pytree with fresh weights on every
    `eqx.Module` leaf.
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
