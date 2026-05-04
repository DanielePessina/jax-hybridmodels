# Trainable Masks: Freezing Leaves

Trainability is encoded as a boolean PyTree mask matching the predictors pytree's structure. The training loop calls `eqx.partition(predictors, mask)` once at start, optimises only the `True` leaves, and re-combines.

[`trainable_mask`](#trainable_mask) builds the default mask (every inexact-array leaf trainable). The `freeze_*` helpers compose to zero out subsets — by path, by module type, or by arbitrary predicate.

## Quick links

- [`default_trainable`](#default_trainable)
- [`trainable_mask`](#trainable_mask)
- [`freeze_paths`](#freeze_paths)
- [`freeze_modules_of_type`](#freeze_modules_of_type)
- [`freeze_where`](#freeze_where)

---

<a id="default_trainable"></a>

### `default_trainable()`

<small>`from hybridmodels.trainable import default_trainable` &nbsp;·&nbsp; also re-exported as `hybridmodels.default_trainable`</small>

```python
default_trainable(leaf: 'Any') -> 'bool'
```

Default leaf-trainability predicate: ``True`` for inexact-array leaves only.

"Inexact" means JAX arrays with float (or complex) dtype — every other
leaf (ints, bools, Python scalars, static fields' frozen values) is
treated as non-trainable. This matches what gradient-based optimisers
can actually update.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L31)</small>

---

<a id="trainable_mask"></a>

### `trainable_mask()`

<small>`from hybridmodels.trainable import trainable_mask` &nbsp;·&nbsp; also re-exported as `hybridmodels.trainable_mask`</small>

```python
trainable_mask(
    predictors: 'Any',
    predicate: 'Callable[[Any], bool]' = <default_trainable>,
) -> Any
```

Build a boolean PyTree mask matching ``predictors``'s structure.

Maps ``predicate`` over every leaf of the ``predictors`` pytree to produce
a mask of the same tree shape with ``bool`` leaves. Accepts any
container shape (tuple, dict, NamedTuple, single ``eqx.Module``). The
result is consumed unchanged by both Optax
(``eqx.filter_value_and_grad(..., filter_spec=mask)``) and Evosax
(``eqx.partition(predictors, mask)``).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L42)</small>

---

<a id="freeze_paths"></a>

### `freeze_paths()`

<small>`from hybridmodels.trainable import freeze_paths` &nbsp;·&nbsp; also re-exported as `hybridmodels.freeze_paths`</small>

```python
freeze_paths(mask: 'Any', paths: 'tuple[str, ...]') -> 'Any'
```

Return a new mask with leaves at `paths` set to ``False``.

Path syntax is dot-joined segments addressing the mask PyTree from its root.
Each segment is the bare key produced by :func:`jax.tree_util.tree_flatten_with_path`:
attribute names for ``eqx.Module`` fields, integer indices for tuples and
lists, and string keys for dicts. Example: ``"inner.mlp.layers.0.weight"``
addresses ``mask.inner.mlp.layers[0].weight``. Unknown paths are silently
ignored.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L74)</small>

---

<a id="freeze_modules_of_type"></a>

### `freeze_modules_of_type()`

<small>`from hybridmodels.trainable import freeze_modules_of_type` &nbsp;·&nbsp; also re-exported as `hybridmodels.freeze_modules_of_type`</small>

```python
freeze_modules_of_type(mask: 'Any', predictors: 'Any', cls: 'type') -> 'Any'
```

Return a new mask with every leaf inside any subtree of type ``cls`` set to ``False``.

Walks ``mask`` and ``predictors`` in lockstep; when a node in
``predictors`` is an instance of ``cls``, the corresponding sub-mask is
replaced wholesale by an all-``False`` subtree. Typical use:
``freeze_modules_of_type(mask, predictors, BoundScaler)`` to freeze
every bound scaler's ``temperature`` leaf — the convention recommended
for hybrid models where the scaler defines the activation shape and
is not meant to drift during training.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L96)</small>

---

<a id="freeze_where"></a>

### `freeze_where()`

<small>`from hybridmodels.trainable import freeze_where` &nbsp;·&nbsp; also re-exported as `hybridmodels.freeze_where`</small>

```python
freeze_where(
    mask: 'Any',
    predictors: 'Any',
    fn: 'Callable[[eqx.Module], bool]',
) -> Any
```

Return a new mask with every leaf inside any submodule satisfying ``fn`` set to ``False``.

``fn`` should be a structural predicate (``isinstance`` checks,
static-field inspection); applying it to leaf-value comparisons is
undefined since the mask copy at a node carries boolean leaves while the
predictors pytree carries arrays.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L119)</small>
