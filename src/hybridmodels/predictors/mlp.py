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
        return jnp.asarray(self.mlp(x))

    def initialized_with_key(self, key: Array) -> MLPPredictor:
        return MLPPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            width_size=self.width_size,
            depth=self.depth,
            activation_name=self.activation_name,
            key=key,
        )
