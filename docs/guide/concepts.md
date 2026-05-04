# Concepts

This page is the package's vocabulary. Every term shows up in API docstrings, error messages, and example comments — knowing what each one is, and where the boundary is, makes the rest of the docs trivial to navigate.

## Predictor

A narrow, trainable `eqx.Module` whose `__call__` is `Array → Array`. It knows nothing about covariate names, bounds, or experiments. v1 ships two concrete subclasses:

- [`MLPPredictor`](/api/predictors#mlppredictor) — wraps `eqx.nn.MLP`.
- [`KANPredictor`](/api/predictors#kanpredictor) — wraps a `jaxkan` model.

Concrete subclasses are **final** — there is no inheritance chain, you compose. Writing your own predictor is a matter of subclassing [`Predictor`](/api/predictors#predictor) directly with whatever fields you need (the [pendulum example](/examples/pendulum) does exactly this with a one-leaf `OmegaPredictor`).

## BoundScaler

An `eqx.Module` that maps between physical `[low, high]` units and a latent space via a sigmoid:

$$
x = \text{low} + (\text{high} - \text{low}) \cdot \sigma(z)
$$

It has two methods, [`to_latent`](/api/predictors#boundscaler) and `from_latent`, so the conversion is bidirectional. The transform is **sigmoid only** in v1 (tanh deferred). The optional `temperature` field is a leaf, but **frozen by convention** in every example via [`freeze_modules_of_type`](/api/trainable#freeze_modules_of_type).

## BoundedPredictor

The composition wrapper that gives a predictor a physical-units interface:

```
input_keys → in_scaler.to_latent → inner Predictor → out_scaler.from_latent → physical Array
```

Two things make `BoundedPredictor` pleasant to use:

- **Polymorphic call.** It accepts either `dict[str, Array]` (subset extraction in `input_keys` order; extra keys ignored, missing keys raise `KeyError`) or a rank-1 `Array` of length `len(input_keys)`. Inside a `simulate_fn` the dict form is idiomatic — it lets you mix covariates, state-derived values, and exogenous time-dependent values without committing to an ordering.
- **Self-describing.** The `input_keys` static field is required to match `len(in_scaler.bounds)` and is auto-filled to `("x1", ..., "xN")` when omitted, so the saved predictor always carries its own input contract.

The inner predictor never has to clamp itself; bounds are owned at the wrapper level. See [`BoundedPredictor`](/api/predictors#boundedpredictor).

## simulate_fn

A pure user-written function that, for one experiment, integrates the dynamics and returns the **full state trajectory**. The framework owns `vmap` over the bucket, `jit` per bucket shape, and `grad` through the integrator; you own the physics.

The signature is **mandatory** — passing a function with a different one breaks `predict_bucket` / `train_with_optax` / `train_with_evosax`:

```python
def simulate_fn(
    predictors,                       # PyTree[eqx.Module] — convention: tuple of BoundedPredictor leaves
    ts: Float[Array, "T"],            # observation times for this experiment
    covariates: dict[str, Array],     # named, constant-in-time scalars
    y0: Float[Array, "S"],            # full initial state
    solver: SolverConfig,             # static; diffrax bits
) -> Float[Array, "T S"]:             # full state at each ts
    ...
```

Inside, you build a per-timestep input dict for each predictor, derive any state-dependent quantities, and call `diffrax.diffeqsolve` with `adjoint=diffrax.DirectAdjoint()` so gradients flow back into the predictors.

### Predictor inputs vs covariates

These two terms are not interchangeable.

- **Covariates** are constant-in-time per-experiment scalars. They live on `Experiment.covariates` and are passed into `simulate_fn` unchanged.
- **Predictor inputs** are the dict your vector field hands to a `BoundedPredictor` at each timestep. It is a *superset* of covariates: you mix in state-derived values (`y[CONC_IDX] / c_sat`), exogenous time-dependent values (`T_now = T0 + r*t`), or whatever else the bounded predictor expects.

Key collision is intentional: a time-varying input can override a covariate's name. The framework treats every key as a named scalar regardless of provenance.

```python
def vector_field(t, y, args):
    predictors, covariates = args
    inputs = {
        "temperature_C": covariates["temperature_C"],          # constant covariate
        "loading": covariates["loading"],                      # constant covariate
        "supersaturation": y[CONC_IDX] / covariates["c_sat"],  # state-derived, time-varying
    }
    log10_G = predictors[0](inputs)
    log10_J = predictors[1](inputs)
    ...
```

## state_to_output

A pure callable mapping the full-state trajectory `[T, S]` to the observed channels `[T, D]`. Crystallisation: `[mu0, mu1, mu2, mu3, mu4, conc] → [conc, d43]`. Stored as a static field on [`Dataset`](/api/data#dataset); applied by [`predict_bucket`](/api/prediction#predict_bucket) before any loss computation.

## y0_fn

A user-supplied hook invoked at **data-import time** by [`make_experiment`](/api/data#make_experiment) to build the per-experiment full initial state from raw covariates and channel observations. Two common shapes:

- **State == observed:** `lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])`.
- **Hidden state with observed initial value:** `lambda c, ch: jnp.concatenate([jnp.zeros(5), ch["conc"].values[0:1]])` (crystallisation: five moments start at zero, concentration starts at the first observation).

The hook runs once per experiment at dataset-build time, not at training time — so it can use `pandas` lookups, side lookups, anything Python.

## Experiment, ChannelObs, Dataset, BucketPayload

A small data hierarchy that turns sparse, irregular runs into JIT-traceable batches.

- [`ChannelObs`](/api/data#channelobs): per-channel triple `(ts, values, variance)`. Each channel has its **own** `Tc` timestamps.
- [`Experiment`](/api/data#experiment): one record holding `covariates: dict[str, Array]`, `y0: [S]`, `channels: dict[str, ChannelObs]`, and a string `exp_id`.
- [`Dataset`](/api/data#dataset): a tuple of [`BucketPayload`](/api/data#bucketpayload) objects, the `state_to_output` callable, and channel/covariate metadata.
- [`BucketPayload`](/api/data#bucketpayload): a `NamedTuple` of stacked `[N, T, ...]` arrays — `ts`, `y_observed`, `yvar`, `mask`, `covariates`, `y0`. One bucket per distinct `len(union_ts)`.

The boundary is meaningful: **users build `Experiment`s; the framework builds `Dataset`s and `BucketPayload`s.** You never write mask code by hand. [`make_dataset`](/api/data#make_dataset) computes per-experiment union timestamps, derives the boolean mask, and stacks experiments grouped by `T = len(union_ts)`.

## SolverConfig

A frozen `eqx.Module` with all-static fields wrapping a `diffrax` solver instance plus tolerances. Static-only because the config is closed over by jitted functions and contributes to their static signature without re-tracing on value changes (changes do trigger a recompile, which is what we want). JSON round-trip works through [`SOLVER_REGISTRY`](/api/solver#solver_registry).

## Bucket, step, phase

Three terms that look similar but are very different in scope:

- A **bucket** is a group of experiments with the same `len(union_ts)`. Buckets are JIT cache keys: one compiled trace per bucket *shape*.
- A **step** is one full pass over **all** buckets, accumulating gradients across them, followed by **one** `optimizer.update`. Equivalently: one epoch.
- A **phase** is a contiguous block of N steps sharing the same hyperparameters. A run is a tuple of phases — see [Training](/guide/training).

A common confusion: a step is *not* one bucket. The bucket-dispatch loop happens **inside** every step.

## predictors pytree

The first argument of `simulate_fn` is a `PyTree[eqx.Module]`. The framework is runtime-permissive: any pytree of leaves works (single Module, tuple, dict, NamedTuple, custom container). The *convention* is to wrap in a tuple — a single-predictor case is `(predictor,)` — so the surrounding code never branches on container type.

`eqx.partition`, `eqx.filter_value_and_grad`, and `eqx.tree_serialise_leaves` walk the leaves uniformly, so the framework never inspects the container.

## Trainability mask

A PyTree of booleans matching the predictors pytree's structure. The training loop calls `eqx.partition(predictors, mask)` once at start, optimises only the `True` leaves, and re-combines. Default predicate: `eqx.is_inexact_array` — every float array trainable.

[`trainable_mask`](/api/trainable#trainable_mask) builds the default; [`freeze_paths`](/api/trainable#freeze_paths), [`freeze_modules_of_type`](/api/trainable#freeze_modules_of_type), and [`freeze_where`](/api/trainable#freeze_where) compose to zero out subsets. There is **no per-class registry** — adding behaviour means adding a free function.

## RNG discipline

Reproducibility is built on **named folds**. Internal subkeys are derived from the user-supplied root key via `jr.fold_in(root, _id("name"))` where `_id` is a stable hash of the consumer name. Names used internally: `"init"`, `"tournament"`, `"phase_{i}"`, `"evosax_init"`, `"evosax_ask_{gen}"`. Reordering or inserting a consumer doesn't shift downstream keys — compare to chained `jr.split`, which is positional and very fragile under refactors.

The framework **never silently defaults** to `jr.PRNGKey(0)`; the `key=` kwarg is required on both training entry points. See [`fold`](/api/rng#fold).

## What's next

- [Training](/guide/training) — phases, shared tournament, evosax.
- [Recommendations](/guide/recommendations) — practical guidance for bounds, solvers, freezing, and pitfalls.
- [API Reference](/api/) — full surface, by module.
