"""Composable trainability filters.

Trainability is represented as a boolean PyTree *mask* whose structure
mirrors the ``predictors`` pytree (any container shape: tuple, dict,
NamedTuple, single Module). Optimisers consume the mask alongside the
parameters: Optax via ``eqx.filter_value_and_grad(..., filter_spec=mask)``,
Evosax via ``eqx.partition(predictors, mask)``. ``True`` marks a leaf as
trainable, ``False`` freezes it.

The default predicate marks every inexact-array leaf as trainable; the
freezers in this module are free functions ``(mask, predictors) -> mask``
— they never mutate their inputs, so masks compose by chaining. A
typical pipeline reads:

    mask = trainable_mask(predictors)
    mask = freeze_modules_of_type(mask, predictors, BoundScaler)
    mask = freeze_paths(mask, ("inner.bias",))

Each step returns a fresh mask of the same pytree shape.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu


def default_trainable(leaf: Any) -> bool:
    """Default leaf-trainability predicate: ``True`` for inexact-array leaves only.

    "Inexact" means JAX arrays with float (or complex) dtype — every other
    leaf (ints, bools, Python scalars, static fields' frozen values) is
    treated as non-trainable. This matches what gradient-based optimisers
    can actually update.
    """
    return bool(eqx.is_inexact_array(leaf))


def trainable_mask(
    predictors: Any,
    predicate: Callable[[Any], bool] = default_trainable,
) -> Any:
    """Build a boolean PyTree mask matching ``predictors``'s structure.

    Maps ``predicate`` over every leaf of the ``predictors`` pytree to produce
    a mask of the same tree shape with ``bool`` leaves. Accepts any
    container shape (tuple, dict, NamedTuple, single ``eqx.Module``). The
    result is consumed unchanged by both Optax
    (``eqx.filter_value_and_grad(..., filter_spec=mask)``) and Evosax
    (``eqx.partition(predictors, mask)``).
    """
    return jtu.tree_map(predicate, predictors)


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
    addresses ``mask.inner.mlp.layers[0].weight``.

    A path matching nothing raises. Silently ignoring it meant a typo left
    a leaf the caller believed was frozen training normally, which shows up
    as a wrong experiment rather than a wrong program. The error lists the
    closest realised paths, since the usual cause is one wrong segment.
    """
    target = set(paths)
    seen: set[str] = set()

    def _f(path: tuple[Any, ...], leaf: Any) -> Any:
        dotted = _path_to_dotted(path)
        seen.add(dotted)
        return False if dotted in target else leaf

    out = jtu.tree_map_with_path(_f, mask)

    missing = sorted(target - seen)
    if missing:
        suggestions = difflib.get_close_matches(missing[0], sorted(seen), n=3, cutoff=0.4)
        hint = f" Closest matches: {suggestions}." if suggestions else ""
        raise ValueError(
            f"freeze_paths: no leaf matches {missing}.{hint} The mask has {len(seen)} leaves."
        )
    return out


def _zero_subtree(submask: Any) -> Any:
    return jtu.tree_map(lambda _: False, submask)


def freeze_modules_of_type(mask: Any, predictors: Any, cls: type) -> Any:
    """Return a new mask with every leaf inside any subtree of type ``cls`` set to ``False``.

    Walks ``mask`` and ``predictors`` in lockstep; when a node in
    ``predictors`` is an instance of ``cls``, the corresponding sub-mask is
    replaced wholesale by an all-``False`` subtree. Typical use:
    ``freeze_modules_of_type(mask, predictors, BoundScaler)`` to freeze
    every bound scaler's ``temperature`` leaf — the convention recommended
    for hybrid models where the scaler defines the activation shape and
    is not meant to drift during training.
    """

    def _is_target(node: Any) -> bool:
        return isinstance(node, cls)

    def _f(submask: Any, sub_pred: Any) -> Any:
        if _is_target(sub_pred):
            return _zero_subtree(submask)
        return submask

    return jtu.tree_map(_f, mask, predictors, is_leaf=_is_target)


def freeze_where(mask: Any, predictors: Any, fn: Callable[[eqx.Module], bool]) -> Any:
    """Return a new mask with every leaf inside any submodule satisfying ``fn`` set to ``False``.

    ``fn`` should be a structural predicate (``isinstance`` checks,
    static-field inspection); applying it to leaf-value comparisons is
    undefined since the mask copy at a node carries boolean leaves while the
    predictors pytree carries arrays.
    """

    def _is_target(node: Any) -> bool:
        return isinstance(node, eqx.Module) and fn(node)

    def _f(submask: Any, sub_pred: Any) -> Any:
        if _is_target(sub_pred):
            return _zero_subtree(submask)
        return submask

    return jtu.tree_map(_f, mask, predictors, is_leaf=_is_target)
