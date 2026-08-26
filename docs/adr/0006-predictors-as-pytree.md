# `predictors` is a pytree: runtime-permissive contract, tuple convention

The first argument of `simulate_fn`, `train_with_optax`, `train_with_evosax`, `predict_bucket`, and `predict_dataset` is `predictors: PyTree[eqx.Module]`. The runtime accepts any pytree shape: tuple, list, dict, NamedTuple, or a single `eqx.Module` (which is itself a one-leaf pytree). The canonical convention shown in CONTEXT.md and the primary `examples/crystallisation/train_kinetic.py` script is a tuple, with the single-predictor case written `(BP,)`.

```python
# Single predictor: convention is a 1-tuple, not a bare Module.
predictors = (BoundedPredictor(...),)

# Multi-rate: tuple unpacked at the top of the vector field.
predictors = (growth_BP, nucleation_BP)

def vector_field(t, y, args):
    predictors, covariates = args
    growth, nucleation = predictors
    ...
```

Dict and NamedTuple shapes are valid alternatives demonstrated in secondary examples. They pass through `eqx.partition`, `eqx.filter_value_and_grad`, `eqx.tree_serialise_leaves`, `jax.tree_util.tree_map` identically. The framework never inspects the container type at runtime; only leaves are walked.

## Why this is non-obvious

A future reader will look at the signature `def simulate_fn(predictors, ...)` and likely reach for one of two stricter interpretations: (a) "must be a `dict[str, Predictor]`", or (b) "must be a single `Predictor`". Neither is correct. The surprising part is the gap between the *runtime contract* (any pytree) and the *convention shown in examples* (tuple).

The split exists because:

- The runtime cannot meaningfully constrain shape. `eqx.partition`, `eqx.filter_value_and_grad`, `eqx.tree_serialise_leaves`, and `jax.tree_util.tree_map` all work on leaves regardless of container type. A type check that rejects `dict` in favour of `tuple` (or vice versa) would be runtime cosmetics; it would not protect any framework invariant.
- A convention is still useful. Without one, every example diverges in idiom (one uses a dict, another a tuple, a third a NamedTuple), and the user reading example code can't form a single mental model. Locking *tuple as convention* gives uniform examples while leaving the runtime free.
- Tuple is the simplest minimal commitment. Single-predictor case `(BP,)` is one character of overhead vs. a bare `BP`. Multi-predictor case `(g, n)` reads as a flat list of trainable components. No string keys to manage. Adding a new predictor in the middle does shift unpacking, a real cost in dimensional growth, but at v1 scale (typically 2–3 predictors per example) the pain is small.
- Tournament re-init falls out cleanly. Per-leaf split-by-traversal (R-T8) `jr.split(attempt_key, n_module_leaves)` doesn't care whether the pytree is a tuple, dict, or NamedTuple, because `jax.tree_util.tree_leaves(predictors, is_leaf=eqx.is_module)` walks all of them uniformly. Identical-shape sibling predictors get *different* re-init weights regardless of container choice.

## Considered alternatives

- Locked runtime contract, single `Predictor` (no pytree). Rejected because multi-rate models would need a wrapper class (`RatePair`, `RateTriple`, ...) for each cardinality; that's the source package's failure pattern. The pytree contract widens the framework while making it *simpler*.
- Locked runtime contract, `dict[str, Predictor]`. Rejected because (1) it forbids tuple/NamedTuple/single-Module users without buying any safety, (2) string keys add ceremony for the trivial 1-predictor case, (3) the framework code becomes "branch on dict vs Module" with no benefit. The flat-dict idea was floated and discarded mid-grilling on 2026-05-04.
- No convention at all, examples diverge per author. Rejected because users reading example code form their mental model from those examples; without a uniform shape, every example feels arbitrary. Locking *tuple* in CONTEXT.md and the primary kinetic example is enough. Secondary examples may use other shapes to demonstrate flexibility.
- NamedTuple as convention. Considered seriously, since it gives both named access (`predictors.growth`) and immutability. Rejected because each example would have to write a one-line subclass, and users with one predictor would have to define a one-field NamedTuple to follow the convention. Tuple wins on minimum-ceremony.
