"""Named-fold RNG helper. Salt is `zlib.crc32(name.encode("utf-8"))` for cross-process stability."""

from __future__ import annotations

import zlib

import jax.random as jr
from jax import Array


def fold(root_key: Array, name: str) -> Array:
    return jr.fold_in(root_key, zlib.crc32(name.encode("utf-8")))
