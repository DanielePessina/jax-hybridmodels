"""Named-fold RNG helper (SPEC §5.6 / R-R2).

The framework derives every internal subkey from the user-supplied root key
via ``jr.fold_in(root, salt)`` where ``salt`` is a stable hash of a string
name (``"init"``, ``"tournament"``, ``"phase_{i}"``, ``"evosax_init"``,
``"evosax_ask_{gen}"``). Naming the consumer rather than chaining
``jr.split`` calls means inserting or reordering a consumer does not shift
every downstream key — adding a ``"warmup"`` fold leaves ``"phase_0"``
unchanged.

``zlib.crc32`` is the salt because Python's built-in ``hash("name")`` is
randomised per process (PEP 456). CRC32 is deterministic across processes,
which the spec requires for reproducibility (R-R2).
"""

from __future__ import annotations

import zlib

import jax.random as jr
from jax import Array


def fold(root_key: Array, name: str) -> Array:
    """Derive a stable subkey from ``root_key`` named ``name``.

    Equivalent to ``jr.fold_in(root_key, crc32(name.encode("utf-8")))``.
    Calling ``fold(root, "init")`` always yields the same key for the same
    root, and different names always yield different keys (collisions only
    on CRC32 collisions across the framework's small constant set of names).
    """
    return jr.fold_in(root_key, zlib.crc32(name.encode("utf-8")))
