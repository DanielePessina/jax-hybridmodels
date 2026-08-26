"""Trainability filter smoke run.

Run with:

    uv run python scripts/smoke_trainable.py
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    MLPPredictor,
    freeze_modules_of_type,
    freeze_where,
    trainable_mask,
)


def _trainable_leaf_count(mask) -> int:
    return sum(1 for leaf in jtu.tree_leaves(mask) if bool(leaf))


def _trainable_param_count(mask, predictor) -> int:
    total = 0
    for m, p in zip(jtu.tree_leaves(mask), jtu.tree_leaves(predictor), strict=True):
        if not bool(m):
            continue
        if hasattr(p, "shape"):
            total += int(jnp.size(p))
        else:
            total += 1
    return total


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

    default_mask = trainable_mask(bp)
    no_scalers = freeze_modules_of_type(default_mask, bp, BoundScaler)
    # CovariateSelector no longer exists -- named-input ordering folded
    # into BoundedPredictor.input_keys, a static field with no leaves to
    # freeze. Demonstrate freeze_where on the inner network instead.
    frozen_inner = freeze_where(no_scalers, bp, lambda m: isinstance(m, MLPPredictor))

    stages = [
        ("default", default_mask),
        ("freeze BoundScaler", no_scalers),
        ("+ freeze inner MLPPredictor", frozen_inner),
    ]
    for label, mask in stages:
        n_leaves = _trainable_leaf_count(mask)
        n_params = _trainable_param_count(mask, bp)
        print(f"\n[{label}] trainable leaves: {n_leaves} | trainable parameters: {n_params}")
        bool_only = jtu.tree_map(lambda x: bool(x), mask)
        print(f"  structure: {eqx.tree_pformat(bool_only, short_arrays=True)}")


if __name__ == "__main__":
    main()
