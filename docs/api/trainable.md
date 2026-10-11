# Trainable Masks: Freezing Leaves

Trainability is encoded as a boolean PyTree mask matching the predictors pytree's structure. The training loop calls `eqx.partition(predictors, mask)` once at start, optimises only the `True` leaves, and re-combines.

[`trainable_mask`](#trainable_mask) builds the default mask (every inexact-array leaf trainable). The `freeze_*` helpers compose to zero out subsets — by path, by module type, or by arbitrary predicate.

## Quick links

- [`default_trainable`](#default_trainable)
- [`trainable_mask`](#trainable_mask)
- [`freeze_mask`](#freeze_mask)
- [`freeze_paths`](#freeze_paths)
- [`freeze_modules_of_type`](#freeze_modules_of_type)
- [`freeze_where`](#freeze_where)
- [`count_trainable_params`](#count_trainable_params)

---

<a id="default_trainable"></a>

### `default_trainable()`

<small>`from jaxhybridmodels.trainable import default_trainable`</small>

```python
default_trainable(leaf: 'Any') -> 'bool'
```

Default trainability rule. ``True`` only for inexact-array leaves.

"Inexact" means a JAX array with a float or complex dtype. Everything
else is fixed, including ints, bools, Python scalars and static-field
values, which is what a gradient-based optimiser can actually update.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L40)</small>

---

<a id="trainable_mask"></a>

### `trainable_mask()`

<small>`from jaxhybridmodels.trainable import trainable_mask` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.trainable_mask`</small>

```python
trainable_mask(
    predictors: 'Any',
    predicate: 'Callable[[Any], bool]' = <default_trainable>,
    train_scalers: 'bool' = False,
) -> Any
```

Build a boolean mask matching the structure of ``predictors``.

Applies ``predicate`` to parameter leaves, returning a tree of the same
shape whose leaves are ``bool``. ``Constant`` subtrees are always
fixed. ``BoundScaler`` subtrees are fixed unless ``train_scalers`` is
set: a trainable temperature moves gradient between the scaler and the
inner network and tends to slow convergence. Both optimisers take the
result unchanged.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L50)</small>

---

<a id="freeze_mask"></a>

### `freeze_mask()`

<small>`from jaxhybridmodels.trainable import freeze_mask` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.freeze_mask`</small>

```python
freeze_mask(
    predictors: 'Any',
    freeze: 'Sequence[FreezeSpec] | FreezeSpec',
    mask: 'Any' = None,
) -> Any
```

The mask behind ``fit(freeze=...)``: ``mask`` with the ``freeze`` entries set to ``False``.

``mask`` defaults to [`trainable_mask`](/api/trainable#trainable_mask) of ``predictors``. Each
``freeze`` entry is one of

- a dotted path, as for [`freeze_paths`](/api/trainable#freeze_paths), naming a leaf or a whole
  subtree (``"rate"`` for ``predictors["rate"]``);
- a module class, freezing every instance (``hm.Constant`` is already
  frozen; ``hm.MLPPredictor`` is the usual one);
- a predicate ``(module) -> bool``, called on the real modules of
  ``predictors``, so it may read their values.

A single entry may be passed bare. An entry that matches no leaf raises,
for the same reason [`freeze_paths`](/api/trainable#freeze_paths) does.

**Examples**

::

    mask = freeze_mask(predictors, ("offset", hm.MLPPredictor))

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L188)</small>

---

<a id="freeze_paths"></a>

### `freeze_paths()`

<small>`from jaxhybridmodels.trainable import freeze_paths` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.freeze_paths`</small>

```python
freeze_paths(mask: 'Any', paths: 'str | Sequence[str]') -> 'Any'
```

Return a new mask with leaves at or below ``paths`` set to ``False``.

Path syntax is dot-joined segments addressing the mask PyTree from its root.
Each segment is the bare key produced by `jax.tree_util.tree_flatten_with_path`:
attribute names for ``eqx.Module`` fields, integer indices for tuples and
lists, and string keys for dicts. ``"inner.mlp.layers.0.weight"``
addresses one leaf, ``mask.inner.mlp.layers[0].weight``. A path may also
stop at a subtree: ``"rate"`` freezes everything under ``mask["rate"]``.
One path may be passed bare, as a string.

A path matching nothing raises, listing the closest real paths. Ignoring
it quietly would leave a leaf the caller believed frozen training as
normal, which shows up as a wrong experiment, not a wrong program.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L136)</small>

---

<a id="freeze_modules_of_type"></a>

### `freeze_modules_of_type()`

<small>`from jaxhybridmodels.trainable import freeze_modules_of_type` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.freeze_modules_of_type`</small>

```python
freeze_modules_of_type(mask: 'Any', cls: 'type') -> 'Any'
```

Return a new mask with every leaf inside any subtree of type ``cls`` set to ``False``.

Reads the types off the mask itself: a mask built by
[`trainable_mask`](/api/trainable#trainable_mask) keeps the module classes of the predictors it was
built from, with booleans in place of arrays. For example,
``freeze_modules_of_type(mask, MLPPredictor)`` freezes every MLP and
leaves the rest of the mask as it was. Matching nothing is not an error.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L160)</small>

---

<a id="freeze_where"></a>

### `freeze_where()`

<small>`from jaxhybridmodels.trainable import freeze_where` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.freeze_where`</small>

```python
freeze_where(mask: 'Any', fn: 'Callable[[eqx.Module], bool]') -> 'Any'
```

Return a new mask with every leaf inside any submodule satisfying ``fn`` set to ``False``.

``fn`` is called on the mask's copy of each module, whose array leaves
have been replaced by booleans. Use a structural test, such as an
``isinstance`` check or a look at a static field. To test the real
modules (their values, say), pass the predicate to [`freeze_mask`](/api/trainable#freeze_mask).
Matching nothing is not an error.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L172)</small>

---

<a id="count_trainable_params"></a>

### `count_trainable_params()`

<small>`from jaxhybridmodels.trainable import count_trainable_params` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.count_trainable_params`</small>

```python
count_trainable_params(predictors: 'Any', mask: 'Any') -> 'int'
```

Number of trainable scalar parameters selected by ``mask``.

Sums the sizes of every leaf the mask marks ``True``. Useful for
reporting the effective search dimension before a run (e.g. to sanity
check an evosax budget or a phase-transition threshold).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L241)</small>
