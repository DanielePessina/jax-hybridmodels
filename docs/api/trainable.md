# Trainable Masks: Freezing Leaves

Trainability is encoded as a boolean PyTree mask matching the predictors pytree's structure. The training loop calls `eqx.partition(predictors, mask)` once at start, optimises only the `True` leaves, and re-combines.

[`trainable_mask`](#trainable_mask) builds the default mask (every inexact-array leaf trainable). The `freeze_*` helpers compose to zero out subsets — by path, by module type, or by arbitrary predicate.

## Quick links

- [`default_trainable`](#default_trainable)
- [`trainable_mask`](#trainable_mask)
- [`freeze_paths`](#freeze_paths)
- [`freeze_modules_of_type`](#freeze_modules_of_type)
- [`freeze_where`](#freeze_where)
- [`frozen_default_mask`](#frozen_default_mask)
- [`count_trainable_params`](#count_trainable_params)

---

<a id="default_trainable"></a>

### `default_trainable()`

<small>`from hybridmodels.trainable import default_trainable` &nbsp;·&nbsp; also re-exported as `hybridmodels.default_trainable`</small>

```python
default_trainable(leaf: 'Any') -> 'bool'
```

Default trainability rule. ``True`` only for inexact-array leaves.

"Inexact" means a JAX array with a float or complex dtype. Everything
else is fixed, including ints, bools, Python scalars and static-field
values, which is what a gradient-based optimiser can actually update.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L35)</small>

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

Build a boolean mask matching the structure of ``predictors``.

Applies ``predicate`` to every leaf, returning a tree of the same shape
whose leaves are ``bool``. Both optimisers take the result unchanged.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L45)</small>

---

<a id="freeze_paths"></a>

### `freeze_paths()`

<small>`from hybridmodels.trainable import freeze_paths` &nbsp;·&nbsp; also re-exported as `hybridmodels.freeze_paths`</small>

```python
freeze_paths(mask: 'Any', paths: 'tuple[str, ...]') -> 'Any'
```

Return a new mask with leaves at `paths` set to ``False``.

Path syntax is dot-joined segments addressing the mask PyTree from its root.
Each segment is the bare key produced by `jax.tree_util.tree_flatten_with_path`:
attribute names for ``eqx.Module`` fields, integer indices for tuples and
lists, and string keys for dicts. Example: ``"inner.mlp.layers.0.weight"``
addresses ``mask.inner.mlp.layers[0].weight``.

A path matching nothing raises, listing the closest real paths. Ignoring
it quietly would leave a leaf the caller believed frozen training as
normal, which shows up as a wrong experiment, not a wrong program.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L73)</small>

---

<a id="freeze_modules_of_type"></a>

### `freeze_modules_of_type()`

<small>`from hybridmodels.trainable import freeze_modules_of_type` &nbsp;·&nbsp; also re-exported as `hybridmodels.freeze_modules_of_type`</small>

```python
freeze_modules_of_type(mask: 'Any', predictors: 'Any', cls: 'type') -> 'Any'
```

Return a new mask with every leaf inside any subtree of type ``cls`` set to ``False``.

Walks ``mask`` and ``predictors`` in lockstep. When a node in
``predictors`` is an instance of ``cls``, the matching sub-mask is
replaced wholesale by an all-``False`` subtree.

The common use is
``freeze_modules_of_type(mask, predictors, BoundScaler)``, which freezes
every scaler's ``temperature``. The temperature sets how sharply the
squash saturates and is not meant to drift while the model trains.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L110)</small>

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

``fn`` must be a structural test, such as an ``isinstance`` check or a
look at a static field. It must not compare leaf values. The walk pairs
a mask node, whose leaves are booleans, with a predictors node, whose
leaves are arrays, so a value comparison has no defined meaning here.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L134)</small>

---

<a id="frozen_default_mask"></a>

### `frozen_default_mask()`

<small>`from hybridmodels.trainable import frozen_default_mask` &nbsp;·&nbsp; also re-exported as `hybridmodels.frozen_default_mask`</small>

```python
frozen_default_mask(predictors: 'Any', *classes: 'type') -> 'Any'
```

The default mask with every leaf of ``classes`` frozen.

Shortcut for the composition every example writes by hand::

    mask = trainable_mask(predictors)
    mask = freeze_modules_of_type(mask, predictors, BoundScaler)

Freezing ``BoundScaler`` leaves (their ``temperature``) is the common
case, so ``frozen_default_mask(predictors, BoundScaler)`` is the
conventional starting mask for a hybrid ODE fit.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L154)</small>

---

<a id="count_trainable_params"></a>

### `count_trainable_params()`

<small>`from hybridmodels.trainable import count_trainable_params` &nbsp;·&nbsp; also re-exported as `hybridmodels.count_trainable_params`</small>

```python
count_trainable_params(predictors: 'Any', mask: 'Any') -> 'int'
```

Number of trainable scalar parameters selected by ``mask``.

Sums the sizes of every leaf the mask marks ``True``. Useful for
reporting the effective search dimension before a run (e.g. to sanity
check an evosax budget or a phase-transition threshold).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/trainable.py#L172)</small>
