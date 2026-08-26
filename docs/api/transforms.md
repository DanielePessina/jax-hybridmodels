# Transforms: Squash Shapes and Axis Warps

## Quick links

- [`BoundTransform`](#boundtransform)
- [`BOUND_TRANSFORMS`](#bound_transforms)
- [`register_bound_transform`](#register_bound_transform)
- [`Warp`](#warp)
- [`WARPS`](#warps)
- [`register_warp`](#register_warp)

---

<a id="boundtransform"></a>

### `BoundTransform`

<small>`from hybridmodels.transforms import BoundTransform` &nbsp;·&nbsp; also re-exported as `hybridmodels.BoundTransform`</small>

```python
BoundTransform(
    forward: ForwardRef('Callable[[Array], Array]'),
    inverse: ForwardRef('Callable[[Array], Array]'),
    inverse_slope: ForwardRef('Callable[[Array], Array]'),
    knee: ForwardRef('float'),
)
```

A latent to unit-interval squash, its inverse, and the metadata around them.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `forward` | `Callable` | ``R -> (0, 1)``. What ``from_latent`` applies. |
| `inverse` | `Callable` | ``(0, 1) -> R``. What ``to_latent`` inverts with. |
| `inverse_slope` | `Callable` | ``d(inverse)/ds``. Used to build the linear continuation that keeps ``to_latent`` differentiable outside the box. |
| `knee` | `float` | Latent at which ``forward`` reaches 0.95, the outer 5% of the box. This is the default for ``BoundScaler.z_knee``. It has to come from the transform: sharing sigmoid's 2.944 with softsign would start charging the saturation penalty at 12.5% from the bound instead of 5%, roughly 2.7 times more aggressive in physical terms. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/transforms.py#L70)</small>

---

<a id="bound_transforms"></a>

### `BOUND_TRANSFORMS`

<small>`from hybridmodels.transforms import BOUND_TRANSFORMS` &nbsp;·&nbsp; also re-exported as `hybridmodels.BOUND_TRANSFORMS`</small>

```python
BOUND_TRANSFORMS = {
  'algebraic': BoundTransform
  'sigmoid': BoundTransform
  'softsign': BoundTransform
}
```

dict() -> new empty dictionary

dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

---

<a id="register_bound_transform"></a>

### `register_bound_transform()`

<small>`from hybridmodels.transforms import register_bound_transform` &nbsp;·&nbsp; also re-exported as `hybridmodels.register_bound_transform`</small>

```python
register_bound_transform(name: 'str', transform: 'BoundTransform') -> 'None'
```

Register a squash under ``name`` for use by ``BoundScaler``.

Mirrors :func:`~hybridmodels.solver.register_solver`. A scaler stores
only the name, so a custom transform must be registered before a saved
scaler that references it can be rebuilt. Re-registering an existing
name overwrites without warning.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/transforms.py#L195)</small>

---

<a id="warp"></a>

### `Warp`

<small>`from hybridmodels.transforms import Warp` &nbsp;·&nbsp; also re-exported as `hybridmodels.Warp`</small>

```python
Warp(
    forward: ForwardRef('Callable[[Array], Array]'),
    inverse: ForwardRef('Callable[[Array], Array]'),
    requires_positive: ForwardRef('bool'),
)
```

A monotone reparameterisation of the physical axis before normalising.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `forward` | `Callable` | Physical to warped coordinate. |
| `inverse` | `Callable` | Warped coordinate back to physical. Must invert ``forward`` exactly on the declared box. |
| `requires_positive` | `bool` | Whether the warp is undefined at or below zero. Checked against the declared bounds at construction, where it can raise a useful error, rather than at trace time where it would surface as a silent nan. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/transforms.py#L96)</small>

---

<a id="warps"></a>

### `WARPS`

<small>`from hybridmodels.transforms import WARPS` &nbsp;·&nbsp; also re-exported as `hybridmodels.WARPS`</small>

```python
WARPS = {
  'linear': Warp
  'log': Warp
  'log10': Warp
}
```

dict() -> new empty dictionary

dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

---

<a id="register_warp"></a>

### `register_warp()`

<small>`from hybridmodels.transforms import register_warp` &nbsp;·&nbsp; also re-exported as `hybridmodels.register_warp`</small>

```python
register_warp(name: 'str', warp: 'Warp') -> 'None'
```

Register an axis warp under ``name`` for use by ``BoundScaler``.

Same contract as :func:`register_bound_transform`. A warp must be
monotone on the declared box and ``inverse`` must undo ``forward``
there, or the scaler's round trip stops being the identity.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/transforms.py#L206)</small>
