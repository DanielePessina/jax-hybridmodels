# Custom predictors

The library ships [`MLPPredictor`](/api/predictors#mlppredictor),
[`KANPredictor`](/api/predictors#kanpredictor) and
[`NeuralNPolynomial`](/api/predictors#neuralnpolynomial). Write your own
when the function you want has a known form a generic network would only
approximate: a basis expansion, a monotone map, random features, a layer
from a paper. The [custom-predictor example](/examples/custom-predictor)
builds one end to end. For a trainable constant, use `Parameter` instead.

## The contract

1. Subclass [`Predictor`](/api/predictors#predictor). It is an
   [`eqx.Module`](https://docs.kidger.site/equinox/), so instances are
   frozen dataclasses and JAX pytrees.
2. Implement `__call__(self, x) -> Array`, rank-1 in and rank-1 out. A
   module with one output may return a scalar instead.
3. Trainable values are plain fields holding float arrays.
   Hyperparameters are `eqx.field(static=True)`.
4. Leave bounds, named inputs and physical units out. `BoundedPredictor`
   adds them by wrapping your module.

## A minimal predictor

An affine map:

```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array

import jaxhybridmodels as hm


class Affine(hm.Predictor):
    weight: Array
    bias: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, *, key: Array) -> None:
        self.weight = jr.normal(key, (out_size, in_size)) / jnp.sqrt(in_size)
        self.bias = jnp.zeros(out_size)
        self.in_size = in_size
        self.out_size = out_size

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias
```

That is complete. Training, prediction and saving all work without a
registration step, because none of them inspect the class.

## Trainable and static fields

| | Array field | `eqx.field(static=True)` |
|---|---|---|
| Example | `weight: Array` | `in_size: int = eqx.field(static=True)` |
| Optimiser | updates it | never sees it |
| Checkpoint | written | not written; the template supplies it |
| Good values | float arrays | ints, floats, strings, tuples of those |

Every float array is trainable by default, including ones you meant as
fixed data such as a grid or a random feature bank. Freeze those with a
mask (below).

Static fields are structure. They are not saved, so loading needs a
template built with the same values. Do not put JAX arrays or lists there.

## Re-initialisation

The [tournament](/guide/training#the-tournament) restarts every predictor
from a new key. By default each float array is replaced with a
standard-normal sample of the same shape, which ignores any scaling your
constructor applied. Define `initialized_with_key` to rebuild properly:

```python
    def initialized_with_key(self, key: Array) -> "Affine":
        return Affine(self.in_size, self.out_size, key=key)
```

The result must have the same pytree shape: same fields, array shapes and
static values. The trainable mask built before the tournament is reused
after it. The method is found with `hasattr`; nothing to register.

## Freezing fixed arrays

A module cannot mark its own arrays as fixed. The caller does, with `freeze=`:

```python
result = model.fit(data, config, freeze=("rate.inner.frequencies",), key=key)
```

The path reads "the `rate` entry, its `inner` module, its `frequencies`
field". See [Freezing leaves](/guide/training#freezing-leaves) for the
other helpers.

## Bounds and named inputs

Wrap the module:

```python
rate = hm.BoundedPredictor(
    inner=Affine(1, 1, key=jr.key(0)),
    input_keys=("temperature",),
    input_bounds=(280.0, 360.0),
    output_bounds=(1e-2, 3.0),
)
```

Your module sees latent inputs of roughly unit scale and its output is
squashed into the output box, so unit-scale initialisation is the right
default whatever the physical units. With one output bound the wrapper
returns a scalar. To pick a `transform` or `warp`, pass
`in_scaler`/`out_scaler` instead; see
[Predictors and bounds](/guide/predictors).

## Checklist

- Subclasses `Predictor` and implements `__call__`.
- Trainable arrays are plain fields; hyperparameters are static.
- `initialized_with_key` is defined if the constructor does anything
  smarter than standard-normal weights, and keeps the pytree shape.
- Arrays that must stay fixed are frozen with `freeze=` or a mask.
- Saving and loading into a fresh template gives the same outputs; see
  [Saving and loading](/guide/serialization).
