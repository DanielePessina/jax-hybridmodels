"""Deriving random-number keys by name, so runs stay reproducible.

JAX has no global random state. Randomness comes from an explicit key
that the caller passes around, and a new key is derived from an existing
one whenever fresh randomness is needed. This framework takes one root
key from the user and derives every internal key from it with
``jr.fold_in(root, salt)``, where ``salt`` is a stable hash of a short
name. The names used internally are ``"tournament"``,
``"tournament_attempt_{i}"``, ``"evosax_init"``, ``"evosax_box_init"``,
``"evosax_ask_{gen}"``, and ``"evosax_tell_{gen}"``.

Names rather than the more familiar ``jr.split`` chain. A chained split
makes each subkey depend on the order in which keys were taken, so
inserting a new consumer between two existing ones shifts every
downstream key and quietly breaks reproducibility. Naming the consumer
means adding a ``"warmup"`` fold leaves ``"evosax_ask_0"`` exactly where
it was, and old runs still reproduce bit for bit.

The salt uses ``zlib.crc32`` because Python's built-in ``hash("name")``
is randomised per process (PEP 456). CRC32 gives the same number in every
process, which is what reproducibility needs.
"""

from __future__ import annotations

import zlib

import jax.random as jr
from jax import Array


def fold(root_key: Array, name: str) -> Array:
    """Derive a stable subkey from ``root_key`` named ``name``.

    Equivalent to ``jr.fold_in(root_key, crc32(name.encode("utf-8")))``.
    ``fold(root, "tournament")`` always returns the same key for the same root,
    and two different names return different keys unless their CRC32 values
    collide, which none of the framework's fixed set of names do.
    """
    return jr.fold_in(root_key, zlib.crc32(name.encode("utf-8")))
