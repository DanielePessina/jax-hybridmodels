# Concepts

The short version of each interface. The linked pages have details and
examples.

## A model is a set of functions and data

`HybridModel` binds the first four pieces and takes the data per call, as
a `Dataset` or a list of experiments:

| Object | What it contains |
| --- | --- |
| `predictors` | A PyTree of trainable values and networks, usually a dict. |
| `simulate_fn` | The ODE solve for one experiment. |
| `state_to_output` | The map from full state to measured channels. |
| `SolverConfig` | Diffrax solver settings. |
| `Dataset` | Experiments converted into rectangular bucket payloads. |

The functional training and prediction calls take the same pieces as
separate arguments. Either way the dataset knows nothing about the state
layout or the observation map.

See [Model interface](/guide/model-interface).

## Experiments and channels

An **experiment** is one run of a physical system. It contains its initial
state, conditions, and observations.

A **channel** is one measured quantity with its own timestamps, values and
variance. Channels in one experiment may be sampled at different times.

A **covariate** is a condition that is constant during one experiment. A
covariate may be a scalar or a rank-1 vector. For example:

```python
covariates = {
    "temperature_C": 25.0,
    "feed_composition": jnp.array([0.2, 0.5, 0.3]),
}
```

See [Data and buckets](/guide/data).

## Buckets and masks

The data layer builds one union timestamp axis per experiment. It places each
channel's values on that axis and creates a boolean mask. A mask cell is
`True` only when that channel was measured at that time.

Experiments with the same union-axis length form a **bucket**. Experiments in
one bucket may still have different timestamp values and masks. Bucketing
provides fixed shapes for JAX compilation without padding or interpolation.

Regular data is the simple case: experiments with the same time grid form one
bucket with an all-`True` mask.

## Predictors and bounds

The `predictors` PyTree holds three kinds of leaf:

| Leaf | Use it for | Read it with |
| --- | --- | --- |
| `Constant(value)` | A fixed physical value. Never trained. | `.value` |
| `Parameter(value, bounds=...)` | A trainable physical value. | `.value` |
| `BoundedPredictor(inner=..., ...)` | A network of named physical inputs. | `predictor(inputs)` |

`Parameter` and `BoundedPredictor` train an unbounded latent and map it into
the declared physical range, so no iterate can leave it. A
`BoundedPredictor` also scales its inputs before the inner network sees them:

```text
physical input → latent input → inner network → latent output → physical output
```

A `BoundedPredictor` with one output returns a scalar.
`evaluate_predictor` evaluates any predictor over a grid of named inputs.
The optional saturation penalty discourages a network's latent output from
sitting near a bound, where its physical gradient is small.

See [Predictors and bounds](/guide/predictors).

## Embedded and parallel predictors

In an **embedded** model, the vector field calls a predictor and uses its
output in the ODE:

```python
def vector_field(t, y, args):
    rate = predictors["rate"]({"temperature": temperature})
    return physics(t, y, rate)
```

In a **parallel** model, the predictor output is simulated or compared as an
observed channel outside the mechanistic dynamics.

The distinction matters for trajectory-aware penalties. An embedded predictor
can carry a penalty as extra ODE state. A parallel predictor can be penalised
from its predicted output.

## Training steps and phases

An Optax **step** visits every bucket, averages the bucket losses and
gradients, and applies one optimizer update. A bucket is not a step.

A **phase** is a run of steps with one learning rate (a number or an optax
schedule), optimizer and loss horizon, written as a `TrainingPhase`.
`OptaxTrainingConfig(steps=..., lr=...)` is one phase;
`OptaxTrainingConfig(phases=(...))` is several. Evosax runs a flat loop of
`steps` generations instead.

What is minimised is separate from how: an `Objective` holds the loss,
channel weights and penalties, and both configs take the same one through
`fit(objective=...)`.

See [Training](/guide/training).

## Trainability masks

A trainability mask is a boolean PyTree with the same structure as
`predictors`. `True` selects an array for optimization. The same mask works
with Optax and Evosax. `trainable_mask(predictors)` is the default: every
float array except `Constant` values and bound scalers.

`fit(freeze=...)` freezes more by path, class or predicate. For full
control, build a mask with `jaxhybridmodels.trainable.freeze_mask`, which
takes the same entries, and pass it as `trainable=`. The mask is data, not
state on a predictor.

## JAX boundaries

The Python loop over buckets stays outside compiled code, and each bucket
shape compiles once.

The user callbacks are called inside JAX transformations. They must return
fixed-shape arrays and must not depend on Python side effects or traced values
in Python control flow.

## Random keys

Training and network constructors require an explicit JAX key; `Parameter`
and `Constant` do not. Internal randomness comes from named folds of the root
key, so a run is a deterministic function of that key.

## Serialization

Saving writes the array leaves of the predictors PyTree. Loading needs a
template with the same container shape, predictor types and static fields.
Simulators, observation maps, datasets and masks are not stored; they are
code, rebuilt by the script that loads the run.

See [Troubleshooting](/guide/troubleshooting) for common shape, tracing, and
solver failures.
