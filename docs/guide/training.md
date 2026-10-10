# Training

`HybridModel.fit` trains a model and returns a new one. The config picks
the optimiser:

| Config | Method | Use it for |
| --- | --- | --- |
| `OptaxTrainingConfig` | Gradient descent through the ODE solve | Networks, and the default for everything else |
| `EvosaxTrainingConfig` | Evolutionary search, no gradients | Roughly 4 to 10 kinetic scalars, or a multi-modal loss |

```python
config = hm.OptaxTrainingConfig(steps=1000, lr=1e-3)
result = model.fit(dataset, config, key=jr.key(0))

result.model                # new HybridModel with the fitted predictors
result.history.loss         # data loss plus weighted trajectory penalty, per step
result.history.penalty      # weighted bound penalty, per step
result.history.phase        # phase index of each step
result.best_step            # step whose parameters were returned
result.stop_reason          # "budget_exhausted" or "patience"
```

Each history entry is measured at the parameters before that step's
update, and `loss + penalty` is what the optimiser minimised. An Evosax fit
fills `history.best_fitness` instead: best-so-far per generation, so it
never rises. Do not plot it against an Optax `loss`.

`key=` is keyword-only and required, so the seed is visible at every call.

### Validation

```python
result = model.fit(train, config, validation=val, key=key)
result.history.validation_loss
```

The full-horizon data loss on `validation` then picks the returned step and
drives `patience`. It costs one extra forward pass per step. Optax only.

### The functional trainers

`model.fit` wraps `train_with_optax` and `train_with_evosax`. Call them
directly when you have predictors and callables but no `HybridModel`, or
when you need the [ensemble helpers](/guide/ensembles):

```python
history, trained = hm.train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver,
    key=key,
)
```

Both return `(loss_history, trained_predictors)`. The Optax history is the
raw per-step `loss` above; the Evosax history is best-so-far.

## Optax

One **step** is one pass over every bucket, accumulating gradients, then
one optimiser update. A bucket is a group of experiments whose merged time
axes have the same length; see [Data and buckets](/guide/data).

### Phases

A single phase takes `steps=`, `lr=`, `optimizer=` and `penalty_weight=`
on the config. A schedule takes `phases=`, a tuple of `TrainingPhase`, and
those four settings move onto each phase:

```python
from jaxhybridmodels import OptaxTrainingConfig, TrainingPhase

config = OptaxTrainingConfig(
    phases=(
        TrainingPhase(steps=200, lr=1e-2, horizon=0.5),
        TrainingPhase(steps=800, lr=1e-3, reset_optimizer=True),
    ),
)
```

That is 200 steps at `lr=1e-2` scored on the first half of every
trajectory, then 800 at `lr=1e-3` scored on all of it. Mixing `steps=`
with `phases=` raises. `config.total_steps` sums the phases.

**`horizon`**, in `(0, 1]`, is the fraction of each trajectory the loss
scores. The solver still integrates the whole trajectory. Fitting the early
dynamics first keeps a diverging tail from swamping the gradient. It is a
runtime mask, so changing it between phases costs no recompile.

**`reset_optimizer=True`** discards the optimiser state at the start of the
phase. It is required when the phase changes optimiser, and useful after a
large horizon jump. Changing `lr` alone keeps the state.

`patience` counts steps without improvement and resets at every phase. The
running best behind `restore_best=True` (the default) resets whenever the
horizon changes, since losses over different horizons are not comparable.
The returned model therefore always comes from the last horizon trained.

### What "loss" means

The recorded loss, `restore_best`, `patience` and the tournament all use
the data term plus any trajectory penalty. The bound penalty, or your
`penalty_fn`, is left out. The optimiser still descends the full objective,
but leaving the penalty out of the recorded loss keeps runs with different
penalty weights comparable.

### The tournament

A bad initialisation can put the solver in a stiff regime it never leaves.
The tournament re-initialises the predictors from `tournament_attempts`
keys, trains each for `tournament_steps` at `tournament_lr`, scores each
with a forward pass, and keeps the best:

```python
config = OptaxTrainingConfig(
    steps=1000, lr=1e-3,
    tournament_attempts=8, tournament_steps=20, tournament_lr=1e-2,
)
```

It runs only when `tournament_steps > 0` and `tournament_attempts > 1`. It
reuses the compiled training step, so it adds no compile time. An attempt
that hits a Diffrax runtime error or a non-finite loss is dropped; any
other error propagates. If every attempt fails, training continues from
your original predictors with a `RuntimeWarning`. Ties keep the earlier
attempt, so the result is deterministic in the key. `Constant` leaves are
never re-initialised.

### Losses

`loss` takes a name from [`LOSS_REGISTRY`](/api/losses#loss_registry)
(`"mse"`, `"mle"`, `"bal_mse"`, `"bal_mle"`) or any callable
`(pred_obs: [N, T, D], bp: BucketPayload) -> scalar`. `channel_idx` and
`channel_weights` select and weight channels:

```python
config = OptaxTrainingConfig(
    steps=1000,
    loss="mse",
    channel_idx=(0, 1),
    channel_weights=(1.0, 0.1),   # channel 1 counts a tenth as much
)
```

[Extending](/guide/extending#losses) covers how these combine with a
custom loss.

## Evosax

```python
config = hm.EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=64,
    num_generations=200,
    init="lhs_box",       # Latin hypercube over the start box
    init_box_extent=0.5,  # half-width of that box, in latent units
    sigma_init=0.3,       # CMA-ES initial step size
)
result = model.fit(dataset, config, key=key)
```

There are no phases: the run is `num_generations` generations of
`population_size` candidates. `init` sets where generation 0 sits in
latent space: `"warm"` (the default) centres it on the current parameters,
`"uniform_box"` samples uniformly within `±init_box_extent` of them, and
`"lhs_box"` uses a Latin hypercube over that box, which covers it more
evenly with a small population.

The search runs on latents, so `Parameter` and `BoundedPredictor` values
stay inside their bounds without any constraint handling. There is no
per-candidate error handling: one candidate that hits a Diffrax error or a
non-finite loss fails the generation. Keep `sigma_init` and
`init_box_extent` conservative.

### Evosax, then Optax

Run the coarse search first, then refine with gradients:

```python
coarse = model.fit(dataset, evo_config, key=k1)
result = coarse.model.fit(dataset, optax_config, key=k2)
```

Pass the same `trainable=` mask to both if you use one. The
[batch reactor example](/examples/batch-reactor) does this.

## Freezing leaves

`trainable=` takes a boolean pytree matching the predictors. The default,
`trainable_mask(predictors)`, trains floating-point arrays and freezes
`Constant` values and `BoundScaler` temperatures. Pass
`train_scalers=True` to train the temperatures too. Three helpers return a
narrower mask:

```python
mask = hm.trainable_mask(predictors)
mask = hm.freeze_paths(mask, ("rate.inner.mlp.layers.0.weight",))   # one array
mask = hm.freeze_modules_of_type(mask, predictors, hm.MLPPredictor)  # every MLP
mask = hm.freeze_where(                                              # any matching module
    mask, predictors, lambda m: isinstance(m, hm.BoundedPredictor) and "temp" in m.input_keys
)

result = model.fit(dataset, config, trainable=mask, key=key)
```

`freeze_paths` takes dot-joined paths to array leaves: dict keys and
module fields by name, tuple indices as numbers. A path that matches
nothing raises and suggests the closest matches.

`freeze_modules_of_type` and `freeze_where` freeze the whole subtree of
every matching module, so matching an outer `BoundedPredictor` freezes
everything inside it. The `freeze_where` predicate sees the mask's
modules, not your predictors, so test types and static fields such as
`input_keys`, never identity or array values.

The mask is an ordinary pytree, so a dict entry can also be frozen by
hand: `mask["rate"] = jax.tree.map(lambda _: False, mask["rate"])`.

`model.fit` raises if the mask leaves nothing to train.

## Progress display

`verbose=True` (the default) shows [`RichTrainingUI`](/api/ui#richtrainingui),
a live terminal dashboard that also reports per-bucket compile time.
`verbose=False` uses [`SilentUI`](/api/ui#silentui). `ui=` overrides both
with any object implementing [`TrainingUI`](/api/ui#trainingui).

## Reproducibility

Every random choice comes from the `key=` you pass. For named subkeys in
your own code, [`fold`](/api/rng#fold) (`from jaxhybridmodels.rng import
fold`) derives a key from a name, so adding a consumer does not shift the
others the way `jr.split` would.
