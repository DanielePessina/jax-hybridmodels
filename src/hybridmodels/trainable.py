"""Saying which parameters train and which stay fixed.

Trainability is a boolean *mask*: a pytree with exactly the structure of
``predictors``, holding a ``bool`` wherever the predictors have a
parameter array. ``True`` means the optimiser may update that leaf.

Both optimisers take the mask directly. Optax partitions the predictors
into the leaves the mask marks ``True`` and differentiates only those,
via ``eqx.filter_value_and_grad``; Evosax partitions to derive the flat
parameter vector via ``eqx.partition(predictors, mask)``. Either way the
mask's pytree shape is what the partitioner walks.

The default predicate marks every inexact (floating-point) array leaf as
trainable. The freezers here are free functions ``(mask, predictors) ->
mask`` that never mutate their input, so masks compose by chaining:

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
import jax.numpy as jnp
import jax.tree_util as jtu


def default_trainable(leaf: Any) -> bool:
    """Default trainability rule. ``True`` only for inexact-array leaves.

    "Inexact" means a JAX array with a float or complex dtype. Everything
    else is fixed, including ints, bools, Python scalars and static-field
    values, which is what a gradient-based optimiser can actually update.
    """
    return bool(eqx.is_inexact_array(leaf))


def trainable_mask(
    predictors: Any,
    predicate: Callable[[Any], bool] = default_trainable,
) -> Any:
    """Build a boolean mask matching the structure of ``predictors``.

    Applies ``predicate`` to every leaf, returning a tree of the same shape
    whose leaves are ``bool``. Both optimisers take the result unchanged.
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

    A path matching nothing raises, listing the closest real paths. Ignoring
    it quietly would leave a leaf the caller believed frozen training as
    normal, which shows up as a wrong experiment, not a wrong program.
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

    Walks ``mask`` and ``predictors`` in lockstep. When a node in
    ``predictors`` is an instance of ``cls``, the matching sub-mask is
    replaced wholesale by an all-``False`` subtree.

    The common use is
    ``freeze_modules_of_type(mask, predictors, BoundScaler)``, which freezes
    every scaler's ``temperature``. The temperature sets how sharply the
    squash saturates and is not meant to drift while the model trains.
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

    ``fn`` must be a structural test, such as an ``isinstance`` check or a
    look at a static field. It must not compare leaf values. The walk pairs
    a mask node, whose leaves are booleans, with a predictors node, whose
    leaves are arrays, so a value comparison has no defined meaning here.
    """

    def _is_target(node: Any) -> bool:
        return isinstance(node, eqx.Module) and fn(node)

    def _f(submask: Any, sub_pred: Any) -> Any:
        if _is_target(sub_pred):
            return _zero_subtree(submask)
        return submask

    return jtu.tree_map(_f, mask, predictors, is_leaf=_is_target)


def frozen_default_mask(predictors: Any, *classes: type) -> Any:
    """The default mask with every leaf of ``classes`` frozen.

    Shortcut for the composition every example writes by hand::

        mask = trainable_mask(predictors)
        mask = freeze_modules_of_type(mask, predictors, BoundScaler)

    Freezing ``BoundScaler`` leaves (their ``temperature``) is the common
    case, so ``frozen_default_mask(predictors, BoundScaler)`` is the
    conventional starting mask for a hybrid ODE fit.
    """
    mask = trainable_mask(predictors)
    for cls in classes:
        mask = freeze_modules_of_type(mask, predictors, cls)
    return mask


def count_trainable_params(predictors: Any, mask: Any) -> int:
    """Number of trainable scalar parameters selected by ``mask``.

    Sums the sizes of every leaf the mask marks ``True``. Useful for
    reporting the effective search dimension before a run (e.g. to sanity
    check an evosax budget or a phase-transition threshold).
    """
    total = 0
    for leaf, m in zip(
        jtu.tree_leaves(predictors), jtu.tree_leaves(mask), strict=True
    ):
        if eqx.is_inexact_array(leaf) and bool(m):
            total += int(jnp.size(leaf))
    return total
