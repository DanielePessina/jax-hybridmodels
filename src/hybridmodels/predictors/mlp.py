"""Concrete ``MLPPredictor`` wrapping ``eqx.nn.MLP``.

All hyperparameters are stored as static ``eqx.field``s so the module
serialises cleanly: the structural metadata travels with the JSON
sidecar, only the inner ``mlp``'s weights and biases are dynamic leaves
written into the binary checkpoint, and the activation function is
referenced by *name* (a string key into ``_ACTIVATION_MAP``) rather than
by its callable identity, so loading does not require re-importing or
guessing at the activation.
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
        raise ValueError(f"Activation {name!r} not supported. Available: {available}") from exc


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

        ``key`` is required and threaded into ``eqx.nn.MLP`` for layer
        weight initialisation — the framework refuses silent default keys
        so reproducibility never relies on a hidden global RNG.
        ``activation_name`` must be a key of ``_ACTIVATION_MAP``; an
        unknown name raises with the supported set.
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

        Implements the re-init protocol consumed by
        :func:`reinitialize_with_key` and by the training tournament loop
        when it restarts a stalled attempt. Re-instantiating the whole
        module is cleaner than reinitialising leaves in place because
        ``eqx.nn.MLP`` owns its own per-layer init logic (Glorot/normal
        scaling, zero biases); leaf-level standard-normal sampling would
        skew the distribution and break that scheme.
        """
        return MLPPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            width_size=self.width_size,
            depth=self.depth,
            activation_name=self.activation_name,
            key=key,
        )

    def with_zero_final_head(self) -> MLPPredictor:
        """Return a copy whose final ``Linear`` layer's weight and bias are zero.

        Hidden layers retain their LeCun-uniform random init, so the input
        feature transformation is non-degenerate; only the readout layer is
        forced to zero. Composed inside a ``BoundedPredictor``, the latent
        zero produced for any input maps via ``out_scaler.from_latent(0)``
        to the *exact midpoint* of the physical bound box — a known-good
        starting output that is independent of the random key. This makes
        training reproducible across seeds when the rate bounds span many
        decades and an unlucky standard-normal readout draw could otherwise
        place the initial output too far off midpoint for the downstream
        ODE solver to handle.

        Returns a structurally identical predictor; only the trailing
        ``eqx.nn.Linear``'s ``weight`` and ``bias`` arrays change.
        """
        final = self.mlp.layers[-1]
        # ``eqx.nn.Linear.bias`` is ``Array | None`` — typed as a union
        # because ``use_bias=False`` is allowed. ``eqx.nn.MLP``'s default
        # is ``use_final_bias=True`` so in practice ``final.bias`` is an
        # Array here, but the no-bias case is supported for free by
        # narrowing the where-tuple before calling ``tree_at``.
        if final.bias is None:
            zeroed = eqx.tree_at(
                lambda layer: layer.weight,
                final,
                jnp.zeros_like(final.weight),
            )
        else:
            zeroed = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                final,
                (jnp.zeros_like(final.weight), jnp.zeros_like(final.bias)),
            )
        return eqx.tree_at(lambda mp: mp.mlp.layers[-1], self, zeroed)
