# ruff: noqa: F722

from __future__ import annotations

import io
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import Array, Float

from hybridmodels.predictors import (
    BoundedPredictor,
    BoundScaler,
    CovariateSelector,
    Predictor,
    RatePair,
)


class _LinearPredictor(Predictor):
    weights: Float[Array, "in_dim out_dim"]

    def __call__(self, x: Array) -> Array:
        return x @ self.weights


def _bounded_predictor() -> BoundedPredictor:
    return _bounded_predictor_with_key(jr.PRNGKey(0))


def _bounded_predictor_with_key(key: Array) -> BoundedPredictor:
    return BoundedPredictor(
        selector=CovariateSelector(keys=("a", "b")),
        in_scaler=BoundScaler(
            bounds=((0.0, 1.0), (0.0, 2.0)),
            transform="sigmoid",
            temperature=1.5,
        ),
        inner=_LinearPredictor(weights=jr.normal(key, (2, 3))),
        out_scaler=BoundScaler(
            bounds=((0.0, 5.0), (0.0, 10.0), (-1.0, 1.0)),
            transform="sigmoid",
        ),
    )


def _rate_pair() -> RatePair:
    return RatePair(nucleation=_bounded_predictor(), growth=_bounded_predictor())


def _different_template(predictor: eqx.Module) -> eqx.Module:
    if isinstance(predictor, RatePair):
        return RatePair(
            nucleation=_bounded_predictor_with_key(jr.PRNGKey(1)),
            growth=_bounded_predictor_with_key(jr.PRNGKey(2)),
        )
    return _bounded_predictor_with_key(jr.PRNGKey(1))


PREDICTOR_FACTORIES: list[tuple[str, Callable[[], eqx.Module]]] = [
    ("bounded_predictor", _bounded_predictor),
    ("rate_pair", _rate_pair),
]


@pytest.mark.parametrize(
    "factory",
    [f for _, f in PREDICTOR_FACTORIES],
    ids=[name for name, _ in PREDICTOR_FACTORIES],
)
def test_serialise_round_trip_bit_exact(factory: Callable[[], eqx.Module]) -> None:
    predictor = factory()
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, predictor)
    buffer.seek(0)
    template = _different_template(predictor)
    template_leaves = jtu.tree_leaves(template)
    original_leaves = jtu.tree_leaves(predictor)
    assert any(
        hasattr(original, "shape") and not jnp.array_equal(original, template_leaf)
        for original, template_leaf in zip(original_leaves, template_leaves, strict=True)
    )
    restored = eqx.tree_deserialise_leaves(buffer, template)

    leaves_original = original_leaves
    leaves_restored = jtu.tree_leaves(restored)
    assert len(leaves_original) == len(leaves_restored)
    for original, recovered in zip(leaves_original, leaves_restored, strict=True):
        if hasattr(original, "shape"):
            assert jnp.array_equal(original, recovered)
        else:
            assert original == recovered


@pytest.mark.parametrize(
    "factory",
    [f for _, f in PREDICTOR_FACTORIES],
    ids=[name for name, _ in PREDICTOR_FACTORIES],
)
def test_serialise_round_trip_preserves_static_fields(
    factory: Callable[[], eqx.Module],
) -> None:
    predictor = factory()
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, predictor)
    buffer.seek(0)
    template = factory()
    restored = eqx.tree_deserialise_leaves(buffer, template)
    assert jtu.tree_structure(predictor) == jtu.tree_structure(restored)
