"""Smoke run for the named-fold RNG helper.

Run with:

    uv run python scripts/smoke_rng.py
"""

from __future__ import annotations

import jax.random as jr
import numpy as np
from jax.typing import ArrayLike

from jaxhybridmodels import fold

NAMES = ("init", "tournament", "phase_0", "phase_1", "evosax_init", "evosax_ask_0")


def _hex(key: ArrayLike) -> str:
    return np.asarray(key).astype(np.uint32).tobytes().hex()


def main() -> None:
    root = jr.PRNGKey(42)
    print("root: PRNGKey(42)")
    print(f"{'name':<16} {'key (hex)':<20} key (uint32)")
    for name in NAMES:
        key = fold(root, name)
        print(f"{name:<16} {_hex(key):<20} {np.asarray(key).tolist()}")


if __name__ == "__main__":
    main()
