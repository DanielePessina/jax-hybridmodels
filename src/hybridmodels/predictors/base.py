"""Predictor primitives and composition wrappers.

Per SPEC §5.2 and ADR-0001/ADR-0003: `Predictor` is an abstract marker; concrete
predictors are final per Equinox's pattern. Bound-scaling, covariate selection,
and rate pairing are decoupled into composition wrappers (`BoundedPredictor`,
`RatePair`) rather than baked into a class hierarchy.
"""

# ruff: noqa: F722

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float

_SUPPORTED_TRANSFORMS: tuple[str, ...] = ("sigmoid",)


class Predictor(eqx.Module):
    """Abstract marker for trainable Array -> Array modules.

    Concrete subclasses (MLPPredictor, KANPredictor, ...) implement `__call__`
    with signature ``Float[Array, "in"] -> Float[Array, "out"]``. Composition
    wrappers (`BoundedPredictor`, `RatePair`) hold a `Predictor` as a field and
    expose richer call signatures without subclassing it.
    """

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError(
            "Predictor is abstract; concrete subclasses implement __call__"
        )


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
    with bounded inputs and outputs. Inputs are pulled from a covariates dict
    by name, mapped to the inner network's latent input space, run through
    the trainable ``Predictor``, then mapped back into the physical output
    box. ``inner`` sees no bound information and never has to clamp itself.

    Calling contract
    ----------------
    Input  : ``dict[str, Array]`` — covariates dict.
    Output : ``Float[Array, "out"]`` — physical-units prediction; ``out`` is
             determined by ``out_scaler.bounds`` (and the ``inner`` network's
             configured output dimension).
    """

    selector: CovariateSelector
    in_scaler: BoundScaler
    inner: Predictor
    out_scaler: BoundScaler

    def __call__(self, covariates: dict[str, Array]) -> Array:
        x = self.selector(covariates)
        z_in = self.in_scaler.to_latent(x)
        z_out = self.inner(z_in)
        return self.out_scaler.from_latent(z_out)


class RatePair(eqx.Module):
    """Stack two ``BoundedPredictor``s' outputs into a ``[2]`` array.

    Used by the crystallisation example to produce a ``(nucleation_rate,
    growth_rate)`` pair from one covariate dict — a domain-specific shape
    convenient enough that we promote it here, but not a framework concept
    (R-A1: no ``Model`` wrapper). Other domains compose their own pair/tuple
    structures the same way.

    The squeeze on each branch is defensive: ``inner`` networks are typically
    configured with ``out_size=1`` (e.g. ``MLPPredictor(out_size=1)`` returns
    ``[1]``), and ``stack`` along the last axis would otherwise produce ``[2, 1]``
    instead of ``[2]``. Removing the trailing singleton makes the output
    shape ``[2]`` regardless of whether the branch returned a scalar or a
    length-1 vector.

    Calling contract
    ----------------
    Input  : ``dict[str, Array]``.
    Output : ``Float[Array, "2"]`` — ``[nucleation, growth]``.
    """

    nucleation: BoundedPredictor
    growth: BoundedPredictor

    def __call__(self, covariates: dict[str, Array]) -> Float[Array, " 2"]:
        n = self.nucleation(covariates)
        g = self.growth(covariates)
        if n.ndim > 0 and n.shape[-1] == 1:
            n = jnp.squeeze(n, axis=-1)
        if g.ndim > 0 and g.shape[-1] == 1:
            g = jnp.squeeze(g, axis=-1)
        return jnp.stack((n, g), axis=-1)


def reinitialize_with_key(predictor: eqx.Module, key: Array) -> eqx.Module:
    """Return a fresh copy of `predictor` with inexact-float leaves re-initialised.

    If the predictor implements the `initialized_with_key` protocol (R-T8), it is
    delegated to. Otherwise every inexact-array leaf in the PyTree is replaced
    with a standard-normal sample of matching shape and dtype; non-inexact
    leaves and static fields are left untouched.
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
