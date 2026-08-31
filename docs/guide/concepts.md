# Concepts

`hybridmodels` combines a user-written ODE with trainable predictors. This
page gives the short version of the interfaces. Use the linked pages for
details and examples.

## A model is a set of functions and data

The framework does not define a `Model` wrapper class. A working setup has:

| Object | What it contains |
| --- | --- |
| `predictors` | One predictor or a PyTree of predictors. |
| `simulate_fn` | The ODE solve for one experiment. |
| `state_to_output` | The map from full state to measured channels. |
| `SolverConfig` | Diffrax solver settings. |
| `Dataset` | Experiments converted into rectangular bucket payloads. |

Training and prediction receive these objects as separate arguments. This
keeps the data independent from the model's state layout and observation map.

See [Model interface](/guide/model-interface).

## Experiments and channels

An **experiment** is one run of a physical system. It contains its initial
state, conditions, and observations.

A **channel** is one measured quantity. A channel has its own timestamps,
values, and variance. Channels in one experiment may be sampled at different
times.

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

A **predictor** is an Equinox module that maps an input array to an output
array. A `BoundedPredictor` composes an input scaler, an inner predictor, and
an output scaler:

```text
physical input → latent input → inner predictor → latent output → physical output
```

The output reparameterisation keeps the physical value inside its declared
box. The optional saturation penalty discourages the latent output from
staying close to a bound, where its physical gradient is small.

See [Predictors and bounds](/guide/predictors).

## Embedded and parallel predictors

In an **embedded** model, the vector field calls a predictor and uses its
output in the ODE:

```python
def vector_field(t, y, args):
    rate = predictors[0]({"temperature": temperature})
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

A **phase** is a contiguous group of steps with one learning rate, optimizer,
and loss horizon. The phase fields are tuples of equal length. Evosax uses a
flat generation loop instead of phases.

See [Training](/guide/training).

## Trainability masks

A trainability mask is a boolean PyTree with the same structure as
`predictors`. `True` selects an array for optimization. The same mask works
with Optax and Evosax.

Use `freeze_paths`, `freeze_modules_of_type`, or `freeze_where` to derive a
new mask. The mask is data, not mutable state on a predictor.

## JAX boundaries

The library keeps the Python bucket loop outside compiled kernels. It then
compiles one kernel per bucket shape.

The user callbacks are called inside JAX transformations. They must return
fixed-shape arrays and must not depend on Python side effects or traced values
in Python control flow.

## Random keys

Training and predictor constructors require explicit JAX keys. Internal
randomness is derived from named folds of the root key. This makes a run a
deterministic function of its supplied root key.

## Serialization

Equinox serializes the array leaves of a predictor PyTree. Loading requires a
template with the same container shape, predictor types, and static fields.
Simulation functions, observation maps, datasets, and trainability masks are
not stored in a run.

See [Troubleshooting](/guide/troubleshooting) for common shape, tracing, and
solver failures.
