"""Trainability filter smoke run.

Run with:

    uv run python scripts/smoke_trainable.py
"""

from __future__ import annotations

import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu

from jaxhybridmodels import (
    BoundedPredictor,
    BoundScaler,
    MLPPredictor,
    count_trainable_params,
    trainable_mask,
)
from jaxhybridmodels.trainable import freeze_mask


def _trainable_leaf_count(mask) -> int:
    return sum(1 for leaf in jtu.tree_leaves(mask) if bool(leaf))


def main() -> None:
    bp = BoundedPredictor(
        input_keys=("temperature_C", "loading"),
        in_scaler=BoundScaler(bounds=((20.0, 40.0), (0.05, 0.30)), transform="sigmoid"),
        inner=MLPPredictor(
            in_size=2,
            out_size=2,
            width_size=8,
            depth=2,
            activation_name="tanh",
            key=jr.PRNGKey(0),
        ),
        out_scaler=BoundScaler(bounds=((1e8, 1e14), (1e-8, 1e-4)), transform="sigmoid"),
    )

    # Every helper takes the predictors first and chains through ``mask=``.
    with_scalers = trainable_mask(bp, train_scalers=True)
    # Freezing the scalers by class gives back the default mask.
    no_scalers = freeze_mask(bp, BoundScaler, mask=with_scalers)
    default_mask = trainable_mask(bp)
    assert jtu.tree_leaves(no_scalers) == jtu.tree_leaves(default_mask)
    # A predicate entry, on the inner network.
    frozen_inner = freeze_mask(bp, lambda m: isinstance(m, MLPPredictor), mask=no_scalers)

    stages = [
        ("train_scalers=True", with_scalers),
        ("freeze BoundScaler (the default)", no_scalers),
        ("+ freeze inner MLPPredictor", frozen_inner),
    ]
    for label, mask in stages:
        n_leaves = _trainable_leaf_count(mask)
        n_params = count_trainable_params(bp, mask=mask)
        print(f"\n[{label}] trainable leaves: {n_leaves} | trainable parameters: {n_params}")
        bool_only = jtu.tree_map(lambda x: bool(x), mask)
        print(f"  structure: {eqx.tree_pformat(bool_only, short_arrays=True)}")


if __name__ == "__main__":
    main()
