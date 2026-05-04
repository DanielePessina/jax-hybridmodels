"""Named-fold RNG helper.

The framework derives every internal subkey from a user-supplied root key
via ``jr.fold_in(root, salt)``, where ``salt`` is a stable hash of a
short string name. Examples used internally include ``"init"``,
``"tournament"``, ``"phase_{i}"``, ``"evosax_init"``, and
``"evosax_ask_{gen}"``.

Why name-based folding rather than the more familiar ``jr.split`` chain?
A chained split makes each subkey depend on the *order* in which keys
were taken: inserting a new consumer between two existing ones shifts
every downstream key and silently breaks reproducibility. Naming the
consumer instead means adding a ``"warmup"`` fold leaves ``"phase_0"``
exactly where it was — old runs still reproduce bit-for-bit.

The salt is computed with ``zlib.crc32`` because Python's built-in
``hash("name")`` is randomised per process (PEP 456). CRC32 is
deterministic across processes, which is what reproducibility requires.
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
