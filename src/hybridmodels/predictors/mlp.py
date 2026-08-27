"""``MLPPredictor``, a dense feedforward network wrapping ``eqx.nn.MLP``.

The default inner predictor. Give it an input and output width, a
hidden width, a depth, and a PRNG key.

Every hyperparameter is a static ``eqx.field`` so the module serialises
cleanly. Structural metadata travels in the JSON sidecar, and the only
dynamic leaves written to the binary checkpoint are the inner ``mlp``'s
weights and biases. The activation is stored by *name*, a string key
into ``_ACTIVATION_MAP``, rather than by callable identity, so loading
never has to re-import or guess at a function.
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

        ``key`` is required, not defaulted, so reproducibility never rests
        on a hidden global RNG. ``activation_name`` must be a key of
        ``_ACTIVATION_MAP``; an unknown name raises, listing the supported
        set.
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

        ``jnp.asarray`` pins the return type: ``eqx.nn.MLP.__call__``
        returns ``Any`` in some Equinox versions.
        """
        return jnp.asarray(self.mlp(x))

    def initialized_with_key(self, key: Array) -> MLPPredictor:
        """Return a fresh ``MLPPredictor`` with the same architecture, new weights.

        The re-init protocol consumed by :func:`reinitialize_with_key` and
        by the tournament when it restarts a stalled attempt.
        Re-instantiating beats reinitialising leaves in place because
        ``eqx.nn.MLP`` owns its per-layer init (LeCun-uniform weights
        scaled by fan-in, zero biases), which leaf-level normal sampling
        would replace with a badly scaled scheme.
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

        Hidden layers keep their LeCun-uniform init, so the feature
        transformation stays non-degenerate; only the readout is zeroed.
        Inside a ``BoundedPredictor`` the resulting zero latent maps
        through ``out_scaler.from_latent(0)`` to the exact midpoint of the
        physical output box, a known-good start independent of the key.
        That matters when rate bounds span decades, where an unlucky
        readout draw can stall the solver on step one.

        Structurally identical predictor; only the trailing
        ``eqx.nn.Linear``'s ``weight`` and ``bias`` change.
        """
        final = self.mlp.layers[-1]
        # ``eqx.nn.Linear.bias`` is ``Array | None``: MLP defaults to
        # ``use_final_bias=True``, but narrowing covers the no-bias case.
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
