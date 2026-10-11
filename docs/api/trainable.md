# Trainable Masks: Freezing Leaves

Trainability is a boolean pytree mask with the structure of the predictors. Training partitions the predictors on it once and optimises only the `True` leaves.

[`trainable_mask`](#trainable_mask) builds the default mask (every inexact-array leaf except `Constant`s and scalers). [`freeze_mask`](#freeze_mask) is what `fit(freeze=...)` calls: it sets the leaves matched by paths, classes or predicates to `False`, on top of an optional `mask=`. Pass its result as `fit(trainable=...)` when `freeze=` is not enough.

## Quick links

- [`default_trainable`](#default_trainable)
- [`trainable_mask`](#trainable_mask)
- [`freeze_mask`](#freeze_mask)
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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L36)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L46)</small>

---

<a id="freeze_mask"></a>

### `freeze_mask()`

<small>`from jaxhybridmodels.trainable import freeze_mask`</small>

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

- a dotted path naming a leaf or a whole subtree: ``"rate"`` for
  ``predictors["rate"]``, ``"rate.inner.mlp.layers.0.weight"`` for one
  leaf. Segments are the bare keys of
  `jax.tree_util.tree_flatten_with_path`: attribute names for
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

**Examples**

::

    mask = freeze_mask(predictors, ("offset", hm.MLPPredictor))
    # Chain onto a mask that also trains the scalers.
    base = trainable_mask(predictors, train_scalers=True)
    mask = freeze_mask(predictors, "offset", mask=base)

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L136)</small>

---

<a id="count_trainable_params"></a>

### `count_trainable_params()`

<small>`from jaxhybridmodels.trainable import count_trainable_params` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.count_trainable_params`</small>

```python
count_trainable_params(predictors: 'Any', *, mask: 'Any' = None) -> 'int'
```

Number of trainable scalar parameters ``mask`` selects in ``predictors``.

``mask`` defaults to [`trainable_mask`](/api/trainable#trainable_mask) of ``predictors``. Sums the
sizes of every leaf the mask marks ``True``. Useful for reporting the
effective search dimension before a run, say to check an Evosax budget.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/trainable.py#L198)</small>
