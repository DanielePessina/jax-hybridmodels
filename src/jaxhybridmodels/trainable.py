"""Saying which parameters train and which stay fixed.

Trainability is a boolean *mask*: a pytree with exactly the structure of
``predictors``, holding a ``bool`` wherever the predictors have a
parameter array. ``True`` means the optimiser may update that leaf.

Both optimisers take the mask directly. Optax partitions the predictors
into the leaves the mask marks ``True`` and differentiates only those,
via ``eqx.filter_value_and_grad``; Evosax partitions to derive the flat
parameter vector via ``eqx.partition(predictors, mask)``. Either way the
mask's pytree shape is what the partitioner walks.

The default mask, :func:`trainable_mask`, trains inexact parameter arrays
and freezes ``Constant`` and ``BoundScaler`` subtrees. ``fit(freeze=...)``
covers the common case in one argument. :func:`freeze_mask` builds the
same mask by hand, for inspection or for ``trainable=``. Every function
here takes ``predictors`` first and never mutates its input, so masks
chain through ``mask=``::

    mask = freeze_mask(predictors, MLPPredictor)
    mask = freeze_mask(predictors, "rate.inner.mlp.layers.0.bias", mask=mask)
    count_trainable_params(predictors, mask=mask)
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Collection, Sequence
from typing import Any, cast

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
    *,
    predicate: Callable[[Any], bool] = default_trainable,
    train_scalers: bool = False,
) -> Any:
    """Build a boolean mask matching the structure of ``predictors``.

    Applies ``predicate`` to parameter leaves, returning a tree of the same
    shape whose leaves are ``bool``. ``Constant`` subtrees are always
    fixed. ``BoundScaler`` subtrees are fixed unless ``train_scalers`` is
    set: a trainable temperature moves gradient between the scaler and the
    inner network and tends to slow convergence. Both optimisers take the
    result unchanged.
    """
    from jaxhybridmodels.predictors.base import BoundScaler, Constant

    def fixed(node: Any) -> bool:
        return isinstance(node, Constant) or (not train_scalers and isinstance(node, BoundScaler))

    return jtu.tree_map(
        lambda node: jtu.tree_map(lambda _: False, node) if fixed(node) else predicate(node),
        predictors,
        is_leaf=fixed,
    )


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


def _freeze_prefixes(mask: Any, prefixes: Collection[str]) -> tuple[Any, set[str]]:
    """Set every mask leaf at or below one of ``prefixes`` to ``False``.

    A prefix names a leaf or a whole subtree: ``"rate"`` covers
    ``"rate.inner.mlp.layers.0.weight"`` and every other leaf under it, and
    the empty string covers the root. Returns the new mask and the prefixes
    that matched at least one leaf, so callers decide whether a miss is an
    error.
    """
    matched: set[str] = set()

    def _covering(dotted: str) -> list[str]:
        return [p for p in prefixes if p in ("", dotted) or dotted.startswith(p + ".")]

    def _f(path: tuple[Any, ...], leaf: Any) -> Any:
        hits = _covering(_path_to_dotted(path))
        matched.update(hits)
        return False if hits else leaf

    return jtu.tree_map_with_path(_f, mask), matched


def _module_paths(tree: Any, test: Callable[[Any], bool]) -> list[str]:
    """Dotted paths of the outermost ``eqx.Module`` nodes in ``tree`` passing ``test``."""

    def _is_target(node: Any) -> bool:
        return isinstance(node, eqx.Module) and bool(test(node))

    flat, _ = jtu.tree_flatten_with_path(tree, is_leaf=_is_target)
    return [_path_to_dotted(path) for path, node in flat if _is_target(node)]


def _no_match_error(caller: str, missing: Any, mask: Any) -> ValueError:
    """Name the entries that matched nothing, with the closest real paths."""
    paths = sorted(_path_to_dotted(path) for path, _ in jtu.tree_flatten_with_path(mask)[0])
    first = missing[0] if isinstance(missing[0], str) else ""
    suggestions = difflib.get_close_matches(first, paths, n=3, cutoff=0.4) if first else []
    hint = f" Closest matches: {suggestions}." if suggestions else ""
    return ValueError(
        f"{caller}: no leaf matches {missing}.{hint} The mask has {len(paths)} leaves."
    )


FreezeSpec = str | type | Callable[[Any], bool]
"""One ``freeze=`` entry: a dotted path, a module class, or a module predicate."""


def freeze_mask(
    predictors: Any,
    freeze: Sequence[FreezeSpec] | FreezeSpec,
    *,
    mask: Any = None,
) -> Any:
    """The mask behind ``fit(freeze=...)``: ``mask`` with the ``freeze`` entries set to ``False``.

    ``mask`` defaults to :func:`trainable_mask` of ``predictors``. Each
    ``freeze`` entry is one of

    - a dotted path naming a leaf or a whole subtree: ``"rate"`` for
      ``predictors["rate"]``, ``"rate.inner.mlp.layers.0.weight"`` for one
      leaf. Segments are the bare keys of
      :func:`jax.tree_util.tree_flatten_with_path`: attribute names for
      ``eqx.Module`` fields, integer indices for tuples and lists, and dict
      keys;
    - a module class, freezing every instance (``hm.Constant`` is already
      frozen; ``hm.MLPPredictor`` is the usual one);
    - a predicate ``(module) -> bool``, called on the real modules of
      ``predictors``, so it may read their values.

    A single entry may be passed bare. An entry that matches no leaf
    raises, listing the closest real paths: ignoring it would leave a leaf
    the caller believed frozen training as normal, which shows up as a
    wrong experiment rather than a wrong program.

    Examples
    --------
    ::

        mask = freeze_mask(predictors, ("offset", hm.MLPPredictor))
        # Chain onto a mask that also trains the scalers.
        base = trainable_mask(predictors, train_scalers=True)
        mask = freeze_mask(predictors, "offset", mask=base)
    """
    # A string is a Sequence too, so it is checked first.
    entries = cast(
        tuple[FreezeSpec, ...],
        tuple(freeze)
        if isinstance(freeze, Sequence) and not isinstance(freeze, str)
        else (freeze,),
    )
    out = trainable_mask(predictors) if mask is None else mask
    for entry in entries:
        if isinstance(entry, str):
            prefixes = [entry]
        elif isinstance(entry, type):
            prefixes = _module_paths(predictors, lambda m, cls=entry: isinstance(m, cls))
        elif callable(entry):
            prefixes = _module_paths(predictors, entry)
        else:
            raise TypeError(
                "freeze entries must be dotted path strings, module classes, or "
                f"predicates (module) -> bool; got {entry!r}"
            )
        out, matched = _freeze_prefixes(out, prefixes)
        if not matched:
            raise _no_match_error("freeze", [entry], out)
    return out


def count_trainable_params(predictors: Any, *, mask: Any = None) -> int:
    """Number of trainable scalar parameters ``mask`` selects in ``predictors``.

    ``mask`` defaults to :func:`trainable_mask` of ``predictors``. Sums the
    sizes of every leaf the mask marks ``True``. Useful for reporting the
    effective search dimension before a run, say to check an Evosax budget.
    """
    if mask is None:
        mask = trainable_mask(predictors)
    total = 0
    for leaf, m in zip(jtu.tree_leaves(predictors), jtu.tree_leaves(mask), strict=True):
        if eqx.is_inexact_array(leaf) and bool(m):
            total += int(jnp.size(leaf))
    return total
