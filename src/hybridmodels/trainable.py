"""Saying which parameters train and which stay fixed.

Trainability is a boolean *mask*. The mask is a pytree (a nested
container of leaves that JAX can flatten and rebuild) with exactly the
structure of the ``predictors`` pytree, but with a ``bool`` wherever the
predictors have a parameter array. ``True`` means the optimiser may
update that leaf, ``False`` freezes it. Any container shape works: tuple,
dict, NamedTuple, or a single Module.

Both optimisers take the mask directly. Optax reads it through
``eqx.filter_value_and_grad(..., filter_spec=mask)``. Evosax reads it
through ``eqx.partition(predictors, mask)``, which splits the pytree into
a trainable half and a frozen half.

The default predicate marks every inexact (floating-point) array leaf as
trainable. The freezers here are free functions of the form
``(mask, predictors) -> mask``. None of them mutates its input, so masks
compose by chaining. A typical pipeline reads:

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
    """Default trainability rule. ``True`` only for inexact-array leaves.

    "Inexact" means a JAX array with a float or complex dtype. Every other
    leaf is treated as fixed, including ints, bools, Python scalars, and
    the frozen values of static fields. That matches what a gradient-based
    optimiser can actually update.
    """
    return bool(eqx.is_inexact_array(leaf))


def trainable_mask(
    predictors: Any,
    predicate: Callable[[Any], bool] = default_trainable,
) -> Any:
    """Build a boolean mask matching the structure of ``predictors``.

    Applies ``predicate`` to every leaf of the ``predictors`` pytree,
    returning a tree of the same shape whose leaves are ``bool``. Any
    container shape works (tuple, dict, NamedTuple, single ``eqx.Module``).
    Both Optax (``eqx.filter_value_and_grad(..., filter_spec=mask)``) and
    Evosax (``eqx.partition(predictors, mask)``) take the result unchanged.
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

    A path matching nothing raises. Ignoring it quietly meant a typo left a
    leaf the caller believed was frozen training as normal, and that shows
    up as a wrong experiment rather than a wrong program. The error lists
    the closest real paths, since the usual cause is one wrong segment.
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
    every bound scaler's ``temperature`` leaf. Every example does this,
    because the temperature sets how sharply the scaler's squash saturates
    and is not meant to drift while the model trains.
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
