# Concepts

Every term below shows up in the API docs, in error messages, and in the
examples. This page defines each one and says what problem it exists to
solve. You can read it without having read anything else.

## Experiment, channel, covariate

An **experiment** is one run of the real thing: one crystallisation
batch, one reactor charge, one released pendulum. It carries everything
that distinguishes that run from the others.

A **covariate** is a condition that stays fixed for the whole run.
Temperature setpoint, pH, initial loading. Covariates are named scalars
in a dict, and every experiment in a dataset must declare the same
names. Time-varying covariates are out of scope; if a quantity changes
during a run, compute it inside your vector field instead (see
[predictor inputs vs covariates](#predictor-inputs-vs-covariates)).

A **channel** is one measured quantity, with its own timestamps. In a
crystallisation run, concentration and mean particle diameter are two
channels. They are measured by two different instruments at two
different cadences, and the library expects exactly that. A channel is a
[`ChannelObs`](/api/data#channelobs): the triple
`(ts, values, variance)`, where `variance` is either one number per
observation or a single number broadcast across all of them.

You build experiments with
[`make_experiment`](/api/data#make_experiment). It also takes a
[`y0_fn`](#y0-fn).

## Buckets, the union timestamp axis, and the mask

### The problem

An ODE solver wants to save its solution at a known list of times, and
JAX wants those lists to be rectangular arrays so it can run every
experiment at once. Real data is neither.

Take one experiment with two channels. Concentration is sampled at
`0, 30, 60, ..., 270` minutes. Particle diameter is measured once, at
270 minutes. There is no single time axis, and the two channels do not
line up.

Now take twelve experiments. They ran for different durations, so their
time axes have different lengths. Two standard fixes both cost you
something:

- **Pad every experiment to the longest axis.** The solver then
  integrates dead time for the short runs, and you pay for it on every
  step of training.
- **Interpolate every channel onto a shared grid.** That invents
  measurements you did not take, and the loss then fits your
  interpolation as if it were data.

### What the library does instead

For each experiment, [`make_dataset`](/api/data#make_dataset) computes
the **union timestamp axis**: the sorted set of every time at which any
channel of that experiment was measured. In the example above that is
`0, 30, ..., 270`, length 9, with the diameter measurement landing on
the existing `270` entry.

It then scatters each channel's values onto that axis and builds a
**mask**: a boolean array, `True` where a cell holds a real measurement
and `False` where that channel was not measured at that time. Every loss
reads the mask, so unmeasured cells contribute nothing. You never write
mask code.

Finally it groups experiments into **buckets**. A bucket is the set of
experiments whose union axes happen to have the same length. That length
is the only thing bucketing groups on. Experiments in one bucket can
still have entirely different times and entirely different masks.

Bucketing exists because of just-in-time compilation. JAX compiles a
function once per distinct input shape. All experiments in a bucket
share one shape, so the training kernel is compiled once for that bucket
and reused for the whole run. Three distinct axis lengths means three
compilations, then full speed.

Nothing is padded and nothing is dropped. Regularly sampled data is the
degenerate case: every experiment lands in one bucket with an all-`True`
mask.

### The types

- [`ChannelObs`](/api/data#channelobs). One channel: `(ts, values, variance)`.
- [`Experiment`](/api/data#experiment). One run: `covariates`, `y0`,
  `channels`, and a string `exp_id`.
- [`BucketPayload`](/api/data#bucketpayload). One bucket, as stacked
  arrays with a leading `N` axis: `ts [N, T]`, `y_observed [N, T, D]`,
  `yvar [N, T, D]`, `mask [N, T, D]`, `covariates`, `y0 [N, S]`.
- [`Dataset`](/api/data#dataset). The tuple of buckets plus
  `state_to_output` and the channel and covariate names.

You build `Experiment`s. The library builds `BucketPayload`s and the
`Dataset`.

## Bound scaling

### The problem

A fitted rate constant must be positive. A concentration must not exceed
solubility. A growth velocity has a plausible range spanning eight
decades, and outside it the ODE goes stiff and the solver gives up.

The obvious fix is to clip the network's output to `[low, high]`. It
does not work, and the reason is worth being precise about. Clipping has
a derivative of exactly zero everywhere outside the range. During
training the network will push outside the range, get clipped, and
receive a gradient of zero: no signal telling it which way to move back.
The parameter is then stuck, permanently, at the value that first went
out of bounds. Clipping does not constrain the fit. It destroys the
gradient that would have recovered the parameter.

### The fix

A [`BoundScaler`](/api/predictors#boundscaler) maps between the physical
box `[low, high]` and an unbounded **latent** space. **Latent** means
"the coordinates the network actually works in": a real number with no
constraints. **Physical** means "the units your ODE needs": metres per
second, moles per litre.

The inner network emits a latent value. The scaler squashes it into the
box. Since the squash is smooth and its output is always inside the box,
an out-of-range value cannot be represented and there is nothing to
clip. The network never sees a bound and never has to respect one.

The scaler is bidirectional. `to_latent` goes physical to latent (used
for the predictor's inputs). `from_latent` goes latent to physical (used
for its output).

Two things follow. The gradient no longer dies at the boundary, it
merely gets small, and the library gives you a
[penalty](#the-saturation-penalty) for the case where it gets too small.
And the box edges are declared once, in physical units, where a domain
expert can read them.

### Warp and squash are two independent choices

A `BoundScaler` composes two maps, and they answer different questions.
You pick each one separately.

The **warp** decides what "halfway between the bounds" means. It
reparameterises the physical axis before anything else happens.

- `"linear"` (default). Halfway between `1e-6` and `1e2` is `50`.
- `"log10"`. Halfway is `1e-2`.
- `"log"`. The same reparameterisation in natural units, which changes
  the latent scale but not which physical values are reachable.

This matters because a freshly initialised network sits near latent zero
and therefore emits the box midpoint. For a rate constant spanning eight
decades, a linear warp puts almost the entire latent range above `1e-2`
and collapses the low end into a sliver you cannot resolve. `log10`
spreads the decades evenly. Bounds stay in physical units whichever warp
you pick. Log warps reject non-positive bounds at construction.

The **squash** (the `transform` argument) decides how the map saturates
as it approaches a bound, and therefore how fast the gradient decays
once a predictor pushes against one.

| transform | gradient at `z = 10` | gradient reaches exactly `0.0` (float32) |
|---|---|---|
| `"sigmoid"` (default) | 4.5e-5 | `z = 16.8` |
| `"algebraic"` | 4.9e-4 | `z ≈ 3e3` |
| `"softsign"` | 4.1e-3 | `z ≈ 1.1e7` |

Once the gradient is exactly zero, the predictor is stuck for the same
reason clipping leaves it stuck. `algebraic` is the recommended
alternative to `sigmoid`: it buys 180 times the latent runway and is
smooth to all orders. `softsign` buys far more runway again, but it is
not twice differentiable, and the kink sits at the box midpoint, which
solver steps cross routinely. Pick it when a predictor is expected to
live near its bounds and the dynamics are not stiff.

`tanh` is deliberately absent. `(1 + tanh z) / 2` is exactly
`sigmoid(2z)`, so it would be the sigmoid entry at half temperature.

Register your own on either axis with
[`register_warp`](/api/transforms#register_warp) and
[`register_bound_transform`](/api/transforms#register_bound_transform).
A box that straddles zero, for example, rules out `log10` and wastes a
linear box's resolution on large corrections that should never happen; a
signed-logarithmic warp registered in user code fixes both. A scaler
stores its warp by name, so a custom warp must be registered before a
saved model referencing it can be loaded.

### The saturation penalty

**Saturation** is how hard a predictor is pinned against a bound,
measured in latent coordinates. `bound_penalty` sweeps a **collocation
grid** (a fixed lattice of points spanning each predictor's declared
input box, independent of any trajectory) and charges the model for
saturation anywhere on it. Turn it on with `penalty_weight` on either
training config.

The penalty is computed on the latent value, never on the physical
output. The physical output's derivative carries a factor that
underflows to exactly zero past a latent of about 15, so a penalty
written against the physical value would die precisely where saturation
is worst.

The grid is deliberately blind to which states a solve actually visited.
That is what makes it report a predictor pressed against a bound in a
corner of its input box that no experiment reached, which the training
loss cannot see. It also means the penalty cannot tell you whether a
particular solve pushed an input out of range.

The word is "saturation", not "bound violation". With this
reparameterisation a physical violation cannot be represented, so there
is nothing to violate.

## Predictor

A **predictor** is a trainable [`eqx.Module`](https://docs.kidger.site/equinox/)
whose `__call__` takes an array and returns an array. That is all. It
knows nothing about covariate names, bounds, or experiments.

Two ship with the library:

- [`MLPPredictor`](/api/predictors#mlppredictor). A standard multi-layer
  perceptron.
- [`KANPredictor`](/api/predictors#kanpredictor). A Kolmogorov-Arnold
  network, which learns activation shapes on the edges rather than
  weights on the nodes.

Both constructors are keyword-only and both require `key=`. To write
your own, subclass [`Predictor`](/api/predictors#predictor) directly
with whatever fields you need. The
[getting-started example](/guide/getting-started#a-runnable-example)
does this with a one-scalar `OmegaPredictor`, and
[Custom predictors](/guide/custom-predictors) is the full procedure:
static fields, re-initialisation, freezing, and serialisation. Concrete
predictors are final; you compose them rather than overriding their
methods.

## BoundedPredictor

[`BoundedPredictor`](/api/predictors#boundedpredictor) is the wrapper
that gives a predictor a physical-units interface:

```
named inputs → in_scaler.to_latent → inner Predictor → out_scaler.from_latent → physical array
```

Its constructor is keyword-only: `in_scaler`, `inner`, `out_scaler`, and
an optional `input_keys`.

`input_keys` names each input slot in order. Its length must match
`len(in_scaler.bounds)` and must be at least one. Omit it and the
constructor fills in `("x1", ..., "xN")`, so a saved predictor always
describes its own input contract.

Calling it accepts either form:

- A `dict[str, Array]`. Only the keys listed in `input_keys` are pulled,
  in declared order. Extra keys are ignored. A missing key raises
  `KeyError`.
- A rank-1 array of length `len(input_keys)`, passed straight through.

Inside a vector field the dict form is the one to use, because it lets
you mix constant covariates with values computed from the current state
without committing to an ordering.

## simulate_fn

Your function. It integrates the dynamics for one experiment and
returns the full state trajectory. The library handles running it across
every experiment in a bucket at once, compiling it, and differentiating
through it. You handle the physics.

The signature is mandatory. A function with a different one breaks
`predict_bucket`, `train_with_optax`, and `train_with_evosax`.

```python
def simulate_fn(
    predictors,                       # your networks, in any pytree container
    ts: Float[Array, "T"],            # times to save at, for this experiment
    covariates: dict[str, Array],     # named scalars, constant over the run
    y0: Float[Array, "S"],            # full initial state
    solver: SolverConfig,             # solver instance and tolerances
) -> Float[Array, "T S"]:             # full state at each time in ts
    ...
```

Inside, you build the input dict for each predictor, derive whatever
depends on the current state, and call `diffrax.diffeqsolve`. Use
`solver.stepsize_controller()` and `solver.adjoint` so the settings on
your `SolverConfig` are the ones that take effect.

### Where a predictor sits relative to the solver

You decide, and the decision costs real time and memory.

A predictor whose inputs are all covariates does not change during a
trajectory. Call it once, above `diffeqsolve`, and close over the
result. Its cost is then independent of how many steps the solver takes,
and it never lands on the tape the backward pass has to store.

A predictor that reads the current state has to run inside the vector
field, once per solver step. That puts it on the tape, and the adjoint
choice starts to matter for memory. See
[`SolverConfig`](#solverconfig).

A hybrid model usually has both. Hoist what you can.

### Predictor inputs vs covariates

These two are not the same thing, and conflating them is the most common
early mistake.

**Covariates** are constant over a run. They live on
`Experiment.covariates` and reach `simulate_fn` unchanged.

**Predictor inputs** are the dict your vector field hands to a
`BoundedPredictor` at each solver step. It is a superset of the
covariates. You mix in:

- values derived from the current state, such as
  `y[CONC_IDX] / covariates["c_sat"]`;
- values that depend on time from outside the model, such as a
  programmed temperature ramp `T0 + rate * t`.

Key collision is intentional. A time-varying input may reuse a
covariate's name, which is how you feed `T(t)` to a predictor that
declared `"temperature_C"`. The library treats every key as a named
scalar and never asks where it came from.

```python
def vector_field(t, y, args):
    predictors, covariates = args
    inputs = {
        "temperature_C": covariates["temperature_C"],           # constant covariate
        "loading": covariates["loading"],                       # constant covariate
        "supersaturation": y[CONC_IDX] / covariates["c_sat"],   # from the state, varies in time
    }
    log10_G = predictors[0](inputs)
    log10_J = predictors[1](inputs)
    ...
```

## state_to_output

A pure function mapping the full state trajectory `[T, S]` to the
channels you measured `[T, D]`, in the order given by
`output_channel_names`. It exists because the state almost always
carries components no instrument sees.

For crystallisation the state is five population moments plus
concentration, and the output is `[conc, d43]`, where `d43` is derived
as `mu4 / mu3`. So this is a projection and a derivation, not just a
slice.

You pass it to [`make_dataset`](/api/data#make_dataset), which stores it
on the [`Dataset`](/api/data#dataset) and applies it before any loss.

## y0_fn

A hook that builds one experiment's full initial state from its
covariates and channels. `make_experiment` calls it once, when you build
the experiment, so it can do anything Python can do. It never runs
during training.

Its signature is `(covariates, channels) -> Array` of shape `[S]`. Two
common shapes:

```python
# State is exactly what you measured.
y0_fn = lambda c, ch: jnp.array([ch["x"].values[0], ch["v"].values[0]])

# Hidden components start at zero; one component starts at its first
# observation. (Crystallisation: five moments, then concentration.)
y0_fn = lambda c, ch: jnp.concatenate([jnp.zeros(5), ch["conc"].values[0:1]])
```

## SolverConfig

[`SolverConfig`](/api/solver#solverconfig) holds the Diffrax solver
instance, `rtol` and `atol` tolerances, `max_steps`, `dt0`, and the
adjoint. Every field is static, meaning it never becomes a JAX array and
is instead baked into the compiled code. Changing a value triggers a
recompile, which is the intended behaviour.

`atol` may be a single number or a tuple with one entry per state
component. Per-component tolerances matter when your state spans many
orders of magnitude. Call `solver.stepsize_controller()` rather than
building a `PIDController` yourself; it converts a tuple `atol` to the
array form Diffrax needs.

The **adjoint** is how Diffrax computes gradients back through the
integration, and it decides the memory cost of the backward pass.
[`ADJOINT_REGISTRY`](/api/solver#adjoint_registry) holds four:

| adjoint | trade |
|---|---|
| `Direct` (default) | Stores the whole forward tape. Cheapest to differentiate, most memory. |
| `RecursiveCheckpoint` | Stores checkpoints and recomputes between them. The one to use when a network runs inside the vector field. |
| `Backsolve` | Integrates the adjoint ODE backwards. Constant memory, but the reverse solve can be inaccurate. |
| `ForwardMode` | Forward-mode differentiation. Good when parameters are few. |

Configs round-trip to JSON through
[`SOLVER_REGISTRY`](/api/solver#solver_registry) and
`ADJOINT_REGISTRY`.

## Bucket, step, phase

Three words that sound alike and mean different things.

- A **bucket** is a group of experiments whose union timestamp axes have
  the same length. It is a compilation unit: one compiled kernel per
  bucket shape.
- A **step** is one full pass over every bucket, accumulating gradients
  across all of them, followed by one optimiser update. It is what other
  frameworks call an epoch.
- A **phase** is a contiguous block of steps sharing one set of
  hyperparameters. A run is a tuple of phases.

A step is not one bucket. The loop over buckets happens inside every
step.

## The predictors pytree

`simulate_fn`'s first argument is your networks, in whatever container
you chose. Any pytree works: a single module, a tuple, a dict, a
NamedTuple. The library never inspects the container, because
`eqx.partition`, `eqx.filter_value_and_grad`, and
`eqx.tree_serialise_leaves` all walk the leaves uniformly.

The convention is a tuple, so a single-predictor model is
`(predictor,)` and the surrounding code never has to branch on container
type. The examples all follow it.

## Trainability mask

A **trainability mask** is a pytree of booleans with the same structure
as your predictors, one boolean per array leaf. Training optimises only
the `True` leaves.

[`trainable_mask`](/api/trainable#trainable_mask) builds the default,
where every floating-point array is trainable and integers, booleans,
and static fields stay frozen. Three free functions return a modified
copy:

- [`freeze_paths`](/api/trainable#freeze_paths) takes dot-joined leaf
  paths, for example `"0.inner.mlp.layers.0.weight"`. A path matching no
  leaf raises, with the closest matches listed.
- [`freeze_modules_of_type`](/api/trainable#freeze_modules_of_type)
  freezes every leaf inside any module of a given class.
- [`freeze_where`](/api/trainable#freeze_where) freezes every leaf inside
  any module satisfying a predicate you supply.

They compose. There is no per-class registry; adding behaviour means
adding a function.

## RNG discipline

JAX random number generation is explicit: you pass a `key` around rather
than relying on a global seed.

The library never defaults that key. `key=` is required and
keyword-only on both trainers, so every run states its seed at the call
site.

Internally it derives subkeys by **named fold** rather than by chained
splitting: `jr.fold_in(root, _id(name))`, where `_id` is a stable hash of
the consumer's name. The names in use are `"init"`, `"tournament"`,
`"phase_{i}"`, `"evosax_init"`, and `"evosax_ask_{gen}"`. Chained
`jr.split` is positional, so inserting or reordering a consumer shifts
every key after it and silently changes a run you thought was fixed.
Named folds do not. Use [`fold`](/api/rng#fold) for the same trick in
your own code.

## What's next

- [Training](/guide/training). Phases, restart tournaments, population
  search, freezing.
- [Recommendations](/guide/recommendations). Bounds, tolerances, and the
  traps.
- [API reference](/api/). Every public symbol, by module.
