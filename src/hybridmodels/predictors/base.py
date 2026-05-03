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
    keys: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, covariates: dict[str, Array]) -> Array:
        return jnp.stack([jnp.asarray(covariates[k]) for k in self.keys])


class BoundScaler(eqx.Module):
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
        lows = jnp.asarray([b[0] for b in self.bounds])
        highs = jnp.asarray([b[1] for b in self.bounds])
        return lows, highs

    def to_latent(self, x: Array) -> Array:
        lows, highs = self._lows_highs()
        normalized = (x - lows) / (highs - lows)
        clipped = jnp.clip(normalized, 1e-6, 1.0 - 1e-6)
        return jax.scipy.special.logit(clipped) * self.temperature

    def from_latent(self, z: Array) -> Array:
        lows, highs = self._lows_highs()
        return lows + (highs - lows) * jax.nn.sigmoid(z / self.temperature)


class BoundedPredictor(eqx.Module):
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
