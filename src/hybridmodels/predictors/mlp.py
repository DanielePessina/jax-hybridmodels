"""Concrete ``MLPPredictor`` wrapping ``eqx.nn.MLP`` (SPEC §5.2).

All hyperparameters are static fields so the module pickles to a JSON-friendly
shape and round-trips through ``eqx.tree_serialise_leaves`` (R-A5). Only the
inner ``mlp`` carries dynamic float leaves; ``activation_name`` is a string
key into ``_ACTIVATION_MAP`` rather than a callable so it serialises cleanly.
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hybridmodels.predictors.base import Predictor

_ACTIVATION_MAP: dict[str, Callable[[Array], Array]] = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "elu": jax.nn.elu,
    "leaky_relu": jax.nn.leaky_relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "softplus": jax.nn.softplus,
}


def _resolve_activation(name: str) -> Callable[[Array], Array]:
    try:
        return _ACTIVATION_MAP[name]
    except KeyError as exc:
        available = sorted(_ACTIVATION_MAP)
        raise ValueError(
            f"Activation {name!r} not supported. Available: {available}"
        ) from exc


class MLPPredictor(Predictor):
    """Multi-layer perceptron ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

    Thin wrapper around ``eqx.nn.MLP`` exposing the constructor hyperparameters
    as static fields (so they survive serialisation and tournament re-init).
    Only the inner ``mlp`` field carries trainable weights; the rest is metadata.

    Attributes
    ----------
    mlp : eqx.nn.MLP
        The actual trainable network; weights and biases are inexact-array
        leaves picked up by ``default_trainable``.
    in_size, out_size : int
        Input / output dimensionality (static).
    width_size, depth : int
        Hidden width and number of hidden layers (static).
    activation_name : str
        Key into ``_ACTIVATION_MAP``; stored as a string rather than the
        callable so the module is JSON-serialisable.
    """

    mlp: eqx.nn.MLP
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    activation_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        key: Array,
        activation_name: str = "tanh",
    ) -> None:
        """Build the MLP, materialising the activation callable from its name.

        ``key`` is required (R-R1) and threaded into ``eqx.nn.MLP`` for layer
        weight initialisation. ``activation_name`` must be a key of
        ``_ACTIVATION_MAP``; an unknown name raises with the supported set.
        """
        activation = _resolve_activation(activation_name)
        self.in_size = int(in_size)
        self.out_size = int(out_size)
        self.width_size = int(width_size)
        self.depth = int(depth)
        self.activation_name = activation_name
        self.mlp = eqx.nn.MLP(
            in_size=self.in_size,
            out_size=self.out_size,
            width_size=self.width_size,
            depth=self.depth,
            activation=activation,
            key=key,
        )

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        """Forward pass: ``[in_size] -> [out_size]``.

        ``jnp.asarray`` ensures the return type is a concrete ``jax.Array``;
        ``eqx.nn.MLP.__call__`` returns ``Any`` in some Equinox versions and
        the cast keeps the public type signature clean.
        """
        return jnp.asarray(self.mlp(x))

    def initialized_with_key(self, key: Array) -> MLPPredictor:
        """Return a fresh ``MLPPredictor`` with the same architecture, new weights.

        Implements the R-T8 protocol consumed by ``reinitialize_with_key``
        and the optax tournament. Re-instantiating the whole module is
        cleaner than reinitialising leaves in place because ``eqx.nn.MLP``
        owns its own per-layer init logic (Glorot/normal scaling, bias zero
        init); leaf-level standard-normal sampling would skew the
        distribution.
        """
        return MLPPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            width_size=self.width_size,
            depth=self.depth,
            activation_name=self.activation_name,
            key=key,
        )
