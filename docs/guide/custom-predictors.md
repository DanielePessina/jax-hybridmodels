# Custom predictors

A **predictor** is a trainable module that takes an array and returns an
array. The library includes [`MLPPredictor`](/api/predictors#mlppredictor),
a dense feedforward network, [`KANPredictor`](/api/predictors#kanpredictor),
a Kolmogorov-Arnold network, and `NeuralNPolynomial`. You can add another
predictor family by subclassing `Predictor`.

Write a custom predictor when a generic network does not match the function
you want to fit: for example, a physical constant, a basis expansion, a
monotone function, a random-features regressor, or a layer from a paper. The
[custom-predictor example](/examples/custom-predictor) does the last of
these end to end.

## The contract

A predictor must satisfy four rules. Everything else is free.

1. It **must** subclass [`Predictor`](/api/predictors#predictor), which
   is an [`eqx.Module`](https://docs.kidger.site/equinox/) and therefore
   a frozen dataclass and a JAX pytree.
2. It **must** implement `__call__(self, x) -> Array`, mapping a rank-1
   array to a rank-1 array.
3. Trainable values **must** be plain fields holding JAX float arrays.
   Hyperparameters **must** be `eqx.field(static=True)`.
4. It **must not** subclass to add bounds, named inputs, or physical
   units. [`BoundedPredictor`](/api/predictors#boundedpredictor) supplies
   those by composition.

Keep bounds, named inputs, and physical units in `BoundedPredictor`.
Concrete predictors are composed; their methods are not overridden.

## A minimal predictor

This one holds a single trainable scalar and ignores its input, which is
enough when every experiment shares the same unknown constant.

```python
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from hybridmodels import Predictor

class OmegaPredictor(Predictor):
    """One trainable scalar; ignores its input."""

    omega_lat: Array

    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:
        return self.omega_lat[None]
```

That is a complete, trainable predictor. It works in
`train_with_optax`, `train_with_evosax`, `predict_dataset`, and
`save_predictors` with no registration step, because none of them
inspect the class.

## Trainable arrays versus static fields

Equinox splits a module's fields in two, and the split decides what the
optimiser touches and what ends up in a checkpoint.

| | Trainable field | `eqx.field(static=True)` |
|---|---|---|
| Declared as | `weights: Array` | `depth: int = eqx.field(static=True)` |
| Holds | JAX float arrays | shape and configuration metadata |
| Optimiser | updates it | never sees it |
| Checkpoint | written to the binary | not written; you supply it again |
| Good values | `jnp` arrays | ints, floats, strings, bools, tuples of those |

Two consequences follow.

**Anything that is a float array is trainable by default.** That includes
arrays you meant to hold fixed, such as a precomputed grid or a random
feature bank. Freezing them is a separate step, covered below.

**Static fields are structure, not data.** They are not written to the
checkpoint file. When you reload, you rebuild the module with the same
static values and the file fills in the arrays. Keep static fields to
plain Python scalars and tuples. Equinox warns if you put a JAX array
there, and a mutable list is a trap for the same reason.

## Re-initialisation

The [restart tournament](/guide/training#the-shared-tournament) runs
several short attempts from different starting weights and keeps the
best. To restart, it needs fresh weights for your module.

By default, [`reinitialize_with_key`](/api/predictors#reinitialize_with_key)
replaces every floating-point array with a standard normal sample of the
same shape. That is a reasonable fallback and a bad fit for any module
with a considered initialisation scheme: it ignores your scaling, your
distributions, and your static hyperparameters.

Define `initialized_with_key` to take control. The method takes a PRNG
key and returns a new instance of the same class:

```python
def initialized_with_key(self, key: Array) -> OmegaPredictor:
    return OmegaPredictor(jr.normal(key))
```

The rebuilt module **must** keep the same pytree shape. Same fields,
same array shapes, same static values. A trainability mask built before
training is reused after a restart, and a changed shape breaks it.

The name is discovered by `hasattr`, so there is nothing to register.

## Freezing arrays you never want trained

Trainability is a boolean pytree matching your predictors, one flag per
array leaf, not a property of a class.
A module therefore cannot declare one of its own arrays fixed. The
caller freezes it:

```python
from hybridmodels import freeze_paths, trainable_mask

mask = trainable_mask(predictors)                    # every float array: True
mask = freeze_paths(mask, ("0.inner.frequencies",))  # that one: False

history, trained = train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, trainable=mask, key=key,
)
```

Paths are dot-joined from the root of the predictors pytree: tuple
indices become numbers and module fields become attribute names, so
`"0.inner.frequencies"` reads as "element 0, its `inner`, its
`frequencies`". A path matching no leaf raises and lists the closest
matches. See [Freezing leaves](/guide/training#freezing-leaves) for
`freeze_modules_of_type` and `freeze_where`, which work on whole
subtrees.

## Serialisation

Every predictor **must** round-trip through
`eqx.tree_serialise_leaves`. The library has no builder registry and
never imports user code by name, so loading works like this:

1. Rebuild a template with the same class, the same container shape, and
   the same static fields. Any key will do; its arrays are placeholders.
2. Call [`load_predictors`](/api/serialise#load_predictors) with the file
   and the template.

```python
from hybridmodels import load_predictors, save_predictors

save_predictors("model.eqx", trained)

template = (build_predictor(jr.PRNGKey(0)),)   # same code path as training
restored = load_predictors("model.eqx", template)
```

A template whose static fields differ from the saved model raises a
shape error. [`save_run`](/api/serialise#save_run) writes a JSON sidecar
recording the pytree structure and each leaf's `"{module}.{qualname}"`
class name, which turns that generic error into a readable mismatch.

## Bounds and named inputs

Do not put either in your predictor. Wrap it:

```python
from hybridmodels import BoundedPredictor, BoundScaler

BoundedPredictor(
    input_keys=("temperature",),
    in_scaler=BoundScaler(bounds=((280.0, 360.0),), transform="sigmoid"),
    inner=YourPredictor(...),
    out_scaler=BoundScaler(bounds=((1e-2, 3.0),), transform="sigmoid", warp="log10"),
)
```

The wrapper pulls named covariates in declared order, maps them into an
unbounded latent space, calls your module, and maps the result back into
physical units. Your module sees latent values of roughly unit scale in
and writes latent values out. It never handles a bound, and it cannot
produce one that is out of range. See
[Predictors and bounds](/guide/predictors) for what the scalers do
and how to choose a warp.

This is also why unit-scale defaults are the right choice inside a
custom predictor. The input distribution is set by the scaler, not by
your data.

## Checklist

Before you train:

- [ ] Subclasses `Predictor`, implements `__call__`, overrides nothing.
- [ ] Trainable arrays are plain fields; hyperparameters are
      `eqx.field(static=True)`.
- [ ] `initialized_with_key` is defined if the module has a considered
      init scheme, and preserves the pytree shape.
- [ ] Arrays that must stay fixed are frozen with a mask at the call site.
- [ ] A save and reload against a freshly built template returns the same
      outputs.
- [ ] Bounds and named inputs live in a `BoundedPredictor`, not in the
      class.

