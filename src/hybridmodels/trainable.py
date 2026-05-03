"""Composable trainability filters.

Per SPEC §5.5 / §4.5 and ADR-0003: trainability is a boolean PyTree mask matching
the predictor's tree structure. The default predicate marks every inexact-array
leaf as trainable; freezers are free functions that consume a mask + predictor
and return a new mask, never mutating the input.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu


def default_trainable(leaf: Any) -> bool:
    """Default leaf-trainability predicate (R-F2): ``True`` for inexact-array leaves only.

    "Inexact" means JAX arrays with float (or complex) dtype — every other
    leaf (ints, bools, Python scalars, static fields' frozen values) is
    treated as non-trainable. This matches what gradient-based optimisers
    can actually update.
    """
    return bool(eqx.is_inexact_array(leaf))


def trainable_mask(
    predictor: Any,
    predicate: Callable[[Any], bool] = default_trainable,
) -> Any:
    """Build a boolean PyTree mask matching ``predictor``'s structure (R-F1).

    Maps ``predicate`` over every leaf of ``predictor`` to produce a mask of
    the same tree shape with ``bool`` leaves. The result is consumed unchanged
    by both Optax (``eqx.filter_value_and_grad(..., filter_spec=mask)``) and
    Evosax (``eqx.partition(predictor, mask)``).
    """
    return jtu.tree_map(predicate, predictor)


def _path_to_dotted(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jtu.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, jtu.FlattenedIndexKey):
            parts.append(str(key.key))
        else:
            parts.append(str(key))
    return ".".join(parts)


def freeze_paths(mask: Any, paths: tuple[str, ...]) -> Any:
    """Return a new mask with leaves at `paths` set to ``False``.

    Path syntax is dot-joined segments addressing the mask PyTree from its root.
    Each segment is the bare key produced by :func:`jax.tree_util.tree_flatten_with_path`:
    attribute names for ``eqx.Module`` fields, integer indices for tuples and
    lists, and string keys for dicts. Example: ``"inner.mlp.layers.0.weight"``
    addresses ``mask.inner.mlp.layers[0].weight``. Unknown paths are silently
    ignored.
    """
    target = set(paths)

    def _f(path: tuple[Any, ...], leaf: Any) -> Any:
        return False if _path_to_dotted(path) in target else leaf

    return jtu.tree_map_with_path(_f, mask)


def _zero_subtree(submask: Any) -> Any:
    return jtu.tree_map(lambda _: False, submask)


def freeze_modules_of_type(mask: Any, predictor: Any, cls: type) -> Any:
    """Return a new mask with every leaf inside any subtree of type ``cls`` set to ``False``.

    Walks ``mask`` and ``predictor`` in lockstep; when a node in ``predictor``
    is an instance of ``cls``, the corresponding sub-mask is replaced
    wholesale by an all-``False`` subtree. Typical use:
    ``freeze_modules_of_type(mask, predictor, BoundScaler)`` to freeze every
    bound scaler's ``temperature`` leaf (the recommended convention; see
    CONTEXT.md).
    """

    def _is_target(node: Any) -> bool:
        return isinstance(node, cls)

    def _f(submask: Any, sub_pred: Any) -> Any:
        if _is_target(sub_pred):
            return _zero_subtree(submask)
        return submask

    return jtu.tree_map(_f, mask, predictor, is_leaf=_is_target)


def freeze_where(
    mask: Any, predictor: Any, fn: Callable[[eqx.Module], bool]
) -> Any:
    """Return a new mask with every leaf inside any submodule satisfying ``fn`` set to ``False``.

    ``fn`` should be a structural predicate (``isinstance`` checks, static-field
    inspection); applying it to leaf-value comparisons is undefined since the
    mask copy at a node carries boolean leaves while the predictor carries
    arrays.
    """

    def _is_target(node: Any) -> bool:
        return isinstance(node, eqx.Module) and fn(node)

    def _f(submask: Any, sub_pred: Any) -> Any:
        if _is_target(sub_pred):
            return _zero_subtree(submask)
        return submask

    return jtu.tree_map(_f, mask, predictor, is_leaf=_is_target)
