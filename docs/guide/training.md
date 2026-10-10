# Training

Choose a training loop based on the number and type of parameters:

| Loop | Use it for |
| --- | --- |
| `train_with_optax` | Differentiable predictors and medium or large parameter sets. |
| `train_with_evosax` | Small parameter sets or objectives with difficult gradients. |

```python
history, trained = train_with_optax(predictors, dataset, config, *, simulate_fn, state_to_output, solver, trainable=None, key, ui=None)
history, trained = train_with_evosax(predictors, dataset, config, *, simulate_fn, state_to_output, solver, trainable=None, key, ui=None)
```

`train_with_optax` differentiates through the ODE solve.
`train_with_evosax` searches a population of parameter vectors and does not
compute gradients. Both return `(loss_history, trained_predictors)` and both
require an explicit keyword-only `key=`.

The two histories mean different things. Optax returns the raw loss at
each step, which can go up. Evosax returns best-so-far per generation,
which cannot. Do not plot them on one axis.

## Model methods and fit results

```python
config = hm.OptaxTrainingConfig(steps=1000, lr=1e-3)
result = model.fit(dataset, config, key=key)

result.model                    # New model with the fitted predictors
result.history.loss             # Data + weighted trajectory penalty, per step
result.history.penalty          # Weighted bound penalty, per step
result.history.phase            # Phase index of each step
result.best_step                # Index of the returned step; None with restore_best=False
result.stop_reason              # "budget_exhausted" or "patience"
```

`loss + penalty` is the objective the optimiser minimised. Each entry is
measured at the parameters before that step's update.

For validation-based model selection:

```python
result = model.fit(train_data, config, validation=validation_data, key=key)
result.history.validation_loss
```

With `validation=`, the validation data loss over the full horizon picks the
returned step and drives `patience`, in place of the training loss. It costs
one extra forward pass over the validation set per step. Validation is
supported for Optax only.

Pass an `EvosaxTrainingConfig` to `model.fit` for population search. Its result
fills `history.best_fitness` instead of the Optax series.

## Optax: gradient-based, multi-phase

[`train_with_optax`](/api/training#train_with_optax) runs a gradient
loop. One **step** is one full pass over every bucket, accumulating
gradients across all of them, then one optimiser update. A **bucket** is
a group of experiments whose merged time axes have the same length; see
[Data and buckets](/guide/data).
A step is not one bucket.

### Phases

A **phase** is a contiguous block of steps sharing hyperparameters. A single
phase uses `steps=`, `lr=`, `optimizer=` and `penalty_weight=` directly. A
schedule uses `phases=`, a tuple of `TrainingPhase`:

```python
from jaxhybridmodels import OptaxTrainingConfig, TrainingPhase

config = OptaxTrainingConfig(
    phases=(
        TrainingPhase(steps=200, lr=1e-2, horizon=0.5),
        TrainingPhase(steps=800, lr=1e-3, reset_optimizer=True),
    ),
    loss="mse",
)
```

That is 200 warm-up steps at `lr=1e-2` seeing the first half of every
trajectory, then 800 fine-tuning steps at `lr=1e-3` seeing all of it.
The two forms do not mix: with `phases=`, set `lr`, `optimizer` and
`penalty_weight` on each `TrainingPhase`. Either way the config stores
`config.phases`, and `config.total_steps` sums their steps.

**`horizon`** is the fraction of each trajectory the loss looks at, in the
interval `(0, 1]`. It masks the loss, never the integration. The solver
still runs the full trajectory; only the first `int(T * horizon)`
observation times are scored. Use it to get the early dynamics right before
asking the model to fit the tail, which stops a long-horizon divergence from
drowning the gradient. Because it is a runtime mask rather than a shape
change, crossing a phase boundary costs no recompile. The default `1.0`
scores everything.

**`reset_optimizer=True`** rebuilds the optimiser at the start of that
phase, discarding its momentum. Set it when you switch optimiser type.
Switching type without a reset raises, because optimiser state belongs to
the optimiser that built it. Set it also when a horizon jump makes the
accumulated momentum meaningless.

Two counters reset at every phase boundary. `patience` (the early-stop
budget) resets because a new phase is a new regime. The running minimum
behind `restore_best` resets whenever the horizon changes, because a phase
with a different horizon measures a different loss. Without that reset the
minimum would land in the shortest-horizon phase and the returned model
would be the least-trained one in the run.

### The saturation penalty

`penalty_weight` charges the objective for predictors pinned against
their bounds, evaluated at the measured points plus any penalty-only
points. It is set per phase.

```python
config = OptaxTrainingConfig(
    steps=1000,
    lr=1e-3,
    penalty_weight=1e-3,
    penalty_points=(sweep,),  # optional per-leaf penalty-only points
)
```

`loss_history`, `restore_best`, early stopping, and the tournament score
track the data term plus any configured trajectory penalty. The fixed-point
regulariser (`penalty_fn`, including the default bound penalty) is excluded
from those measurements. The optimiser descends
`data + trajectory_weight * trajectory_penalty + weight * penalty`, but
reporting the combined value would let
"best" move when only the penalty weight changed, and would make runs
with different weights incomparable.

### The shared tournament

Random initialisation can be unlucky. A bad draw can put the integrator
in a stiff regime it cannot escape within `max_steps`, and training
never gets started. The **tournament** re-initialises the predictors
several times from different keys, trains each candidate for a few
steps, scores it with a forward-only pass on the data term plus any configured
trajectory penalty, and keeps the **best-scoring** candidate. The fixed-point
bound penalty is excluded. If an attempt raises a diffrax error or produces a
non-finite score it is dropped and the next key is tried, up to
`tournament_attempts` times. Ties keep the earlier attempt,
so the result is a deterministic function of the seed.

```python
config = OptaxTrainingConfig(steps=1000, lr=1e-3, loss="mse", tournament_attempts=8, tournament_steps=20, tournament_lr=1e-2)
```

It turns on implicitly when `tournament_steps > 0` and
`tournament_attempts > 1`. The defaults (`0`, `1`) leave it off.

"Shared" means it reuses the main loop's already-compiled step function,
so it costs no extra compilation. Only the shared mode exists;
vmapped and separately-compiled serial modes are out of scope.

Only two failures are caught: a Diffrax runtime error and a non-finite
loss. Anything else, a shape bug or a mistyped `input_keys`, reaches you
instead of being silently retried. If every attempt fails, training
falls back to your original predictors and emits a `RuntimeWarning`
rather than crashing. Re-initialisation splits the key by traversal
order, so two identically shaped sibling predictors get different
weights.

### Choosing a loss

Pass a string to look one up in
[`LOSS_REGISTRY`](/api/losses#loss_registry):

```python
config = OptaxTrainingConfig(..., loss="mse")     # masked_mse
config = OptaxTrainingConfig(..., loss="bal_mle") # bal_mle
```

Or pass any callable matching
`(pred_obs: [N, T, D], bp: BucketPayload) -> scalar`. The built-ins take
two optional weighting arguments, forwarded from the config:

```python
config = OptaxTrainingConfig(
    ...,
    loss="mse",
    channel_idx=(0, 1),
    channel_weights=(1.0, 0.1),  # count channel 1 at a tenth of channel 0
)
```

## Evosax: population-based search

[`train_with_evosax`](/api/training#train_with_evosax) evaluates a
population of candidate parameter vectors each generation and moves the
population towards the good ones. It never differentiates. Use it when
the loss has many local minima, or when the trainable part is a handful
of kinetic constants rather than a network.

```python
from jaxhybridmodels import EvosaxTrainingConfig, train_with_evosax

config = EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=64,
    num_generations=200,
    init="lhs_box",         # Latin hypercube, computed on the host with scipy.qmc
    init_box_extent=0.5,    # half-width of the latent box the population starts in
    sigma_init=0.3,         # CMA-ES initial step size
    loss="mse",
    verbose=True,
)
history, trained = train_with_evosax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, key=key,
)
```

There are no phases here. The loop is flat: `num_generations` of
`population_size` individuals.

Three initialisation modes set where generation 0 lives in latent space.
`"warm"` (the default) centres it on the parameters you passed in.
`"uniform_box"` samples uniformly within `±init_box_extent` of that
centre. `"lhs_box"` uses a Latin hypercube over the same box, which
spreads a small population more evenly across the corners.

Design notes:

- Sized for small parameter vectors, roughly 4 to 10 dimensions. It is
  not tuned for network-sized search.
- Bounds are not enforced during the search. CMA-ES wanders freely in
  latent space and `BoundedPredictor`'s squash keeps every physical
  value inside its box regardless.
- Best-ever is tracked on the host with `jnp.argmin` per generation,
  rather than through a strategy-specific field.
- There is no per-individual error handling. One individual that hits a
  Diffrax error or produces a non-finite loss brings down the
  generation. Guard against it with a conservative `sigma_init` and
  `init_box_extent`.

### Running both

There is no polishing option on `OptaxTrainingConfig`. Call the two in
sequence instead.

```python
# Coarse global search.
hist1, predictors = train_with_evosax(predictors, dataset, evo_config,
    simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver, key=k1)
# Local refinement with gradients.
hist2, predictors = train_with_optax(predictors, dataset, opt_config,
    simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver, key=k2)
```

Both accept the same `trainable=` mask, so freezing carries across
unchanged. The [batch reactor script](/examples/batch-reactor) is a worked
two-phase fit in exactly this shape.

## Freezing leaves

`trainable=` takes a boolean pytree matching your predictors, one
boolean per array. Omit it and floating-point parameter arrays train;
`Constant` values and scaler temperatures stay fixed. Pass one to hold more
of the model fixed, or build it with `trainable_mask(predictors,
train_scalers=True)` to train scaler temperatures.

```python
from jaxhybridmodels import freeze_paths, trainable_mask

mask = trainable_mask(predictors)  # Constants and scalers already fixed
mask = freeze_paths(mask, ("0.inner.mlp.layers.0.weight",))           # one specific array

history, trained = train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, trainable=mask, key=key,
)
```

`freeze_paths` addresses individual leaves by dot-joined path from the
root of the tree: tuple indices become numbers, module fields become
attribute names. A path that matches no leaf raises and lists the
closest matches, because a silently ignored typo means a leaf you
believed was frozen trained anyway.

`freeze_modules_of_type` and
[`freeze_where`](/api/trainable#freeze_where) work on whole subtrees
instead, which is usually what you want. All three return a new mask and
compose in any order. See [Trainable masks](/api/trainable).

A predictor you wrote yourself may hold arrays that must never train, a
fixed basis or a precomputed grid. It cannot declare them fixed; the
mask does it. See
[Custom predictors](/guide/custom-predictors#freezing-arrays-you-never-want-trained).

## Progress display

`ui=` overrides the config's `verbose` flag.

- `verbose=True` gives [`RichTrainingUI`](/api/ui#richtrainingui), a live
  terminal dashboard.
- `verbose=False` gives [`SilentUI`](/api/ui#silentui).
- `ui=YourUI()` takes any object implementing the
  [`TrainingUI`](/api/ui#trainingui) protocol.

Compilation is reported through its own hooks
(`on_compile_start`, `on_compile_progress`, `on_compile_done`), so the
first step of a run shows per-bucket compile time instead of appearing
to hang.

## Reproducibility checklist

- Seed the root key explicitly: `key = jr.PRNGKey(seed)`.
- Pass `key=` at every `train_with_*` call site.
- Use [`fold`](/api/rng#fold) (`from jaxhybridmodels.rng import fold`)
  instead of `jr.split` for named subkeys in
  your own code, so inserting a consumer does not shift the others.
