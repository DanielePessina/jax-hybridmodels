# Training

`HybridModel.fit` trains a model and returns a new one. Two arguments say
what happens: `objective=` is what is minimised, and the config is how.

| Config | Method | Use it for |
| --- | --- | --- |
| `OptaxTrainingConfig` | Gradient descent through the ODE solve | Networks, and the default for everything else |
| `EvosaxTrainingConfig` | Evolutionary search, no gradients | Roughly 4 to 10 kinetic scalars, or a multi-modal loss |

```python
config = hm.OptaxTrainingConfig(steps=1000, lr=1e-3)
result = model.fit(experiments, config, key=jr.key(0))

result.model                # new HybridModel with the fitted predictors
result.history.loss         # data loss plus weighted trajectory penalty, per step
result.history.penalty      # weighted bound penalty, per step
result.history.phase        # phase index of each step
result.best_step            # step whose parameters were returned
result.stop_reason          # "budget_exhausted" or "patience"
```

The data may be a list of experiments or a `Dataset`. Each history entry is
measured at the parameters before that step's update, and `loss + penalty`
is what the optimiser minimised. `key=` is keyword-only and required, so the
seed is visible at every call.

## The objective

`hm.Objective` holds every term of the training objective. `Objective()`,
the default, is MSE over every channel with no penalty.

```python
objective = hm.Objective(
    loss="bal_mse",                                  # or "mse", "mle", "bal_mle", a callable
    channel_weights={"concentration": 1.0, "size": 0.1},
    penalty_weight=1e-3,
)
result = model.fit(experiments, config, objective=objective, key=key)
```

`loss` takes a name from [`LOSS_REGISTRY`](/api/losses#loss_registry) or a
callable `(pred_obs: [N, T, D], bp: BucketPayload) -> scalar`.

`channel_weights` keyed by channel name scores only the channels it names,
so `{"concentration": 1.0}` fits one channel of several. A sequence gives
one weight per output channel, in `output_channel_names` order. Names are
checked against the data when training starts, and an unknown one raises
before anything compiles. A named channel with weight `0.0` adds nothing
but still counts in the loss's denominator, which scales the loss down;
leave the name out to drop the channel. [Extending](/guide/extending#losses) covers how
weights combine with a custom loss.

`penalty_weight`, `penalty_points` and `penalty_fn` set the bound penalty,
and `trajectory_penalty_fn` and `trajectory_penalty_weight` the trajectory
penalty; see [Penalties](/guide/penalties).

The configs hold no loss settings, so one objective can drive an Evosax
search and then an Optax refinement. Validation is scored with the same
objective, minus the bound penalty.

### Validation

```python
result = model.fit(train, config, objective=objective, validation=val, key=key)
result.history.validation_loss
```

The full-horizon data loss on `validation` then picks the returned step and
drives `patience`. It costs one extra forward pass per step. Evosax takes
`validation=` too; see [below](#evosax).

### The functional trainers

`model.fit` wraps `train_with_optax` and `train_with_evosax`. Call them
directly when you have predictors and callables but no `HybridModel`, or
when you need the [ensemble helpers](/guide/ensembles):

```python
history, trained = hm.train_with_optax(
    predictors, dataset, config,
    objective=objective,
    simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver,
    key=key,
)
```

Both take a `Dataset` and return `(loss_history, trained_predictors)`. The
Optax history is the per-step `loss` above; the Evosax history is
best-so-far.

## Optax

One **step** is one pass over every bucket, accumulating gradients, then
one optimiser update. A bucket is a group of experiments whose merged time
axes have the same length; see [Data and buckets](/guide/data).

### Phases

A single phase takes `steps=`, `lr=` and `optimizer=` on the config. A
schedule takes `phases=`, a tuple of `TrainingPhase`, and those settings
move onto each phase:

```python
config = hm.OptaxTrainingConfig(
    phases=(
        hm.TrainingPhase(steps=200, lr=1e-2, horizon=0.5),
        hm.TrainingPhase(steps=800, lr=1e-3, reset_optimizer=True),
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

**`penalty_weight`** on a phase overrides `Objective.penalty_weight` for
that phase; `None`, the default, keeps the objective's. Use it to switch
the bound penalty on only after a warm-up phase.

`patience` counts steps without improvement and resets at every phase. The
running best behind `restore_best=True` (the default) resets whenever the
horizon changes, since losses over different horizons are not comparable.
The returned model therefore always comes from the last horizon trained.

### Learning-rate schedules

`lr` also takes an optax schedule, a function `step -> lr`. Its step count
starts at 0 at the start of the phase, whatever ran before:

```python
import optax

config = hm.OptaxTrainingConfig(
    steps=1000,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-2, warmup_steps=50, decay_steps=1000
    ),
)
```

For several curves inside one phase, chain them with
`optax.join_schedules`. A schedule needs an optimiser name or a factory
taking `learning_rate`. A ready-made `optax.GradientTransformation` owns its
learning rate, so pairing one with a schedule raises.

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
config = hm.OptaxTrainingConfig(
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

## Evosax

```python
config = hm.EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=64,
    steps=200,            # generations
    init="lhs_box",       # Latin hypercube over the start box
    init_box_extent=0.5,  # half-width of that box, in latent units
    sigma_init=0.3,       # CMA-ES initial step size
)
result = model.fit(experiments, config, objective=objective, key=key)
result.history.loss       # best-so-far loss, one entry per generation
```

There are no phases: the run is `steps` generations of `population_size`
candidates. `history.loss` holds the best loss found so far, so it never
rises; with a bound penalty on, it includes the penalty. Do not plot it
against an Optax `loss`. `history.penalty` and `history.phase` stay empty.

With `validation=`, each generation's best candidate is scored on the
validation data, `history.validation_loss` records that score, and the
returned model is the candidate with the lowest one.

`init` sets where generation 0 sits in latent space: `"warm"` (the
default) centres it on the current parameters, `"uniform_box"` samples
uniformly within `±init_box_extent` of them, and `"lhs_box"` uses a Latin
hypercube over that box, which covers it more evenly with a small
population.

The search runs on latents, so `Parameter` and `BoundedPredictor` values
stay inside their bounds without any constraint handling. There is no
per-candidate error handling: one candidate that hits a Diffrax error or a
non-finite loss fails the generation. Keep `sigma_init` and
`init_box_extent` conservative.

### Evosax, then Optax

Run the coarse search first, then refine with gradients under the same
objective:

```python
coarse = model.fit(data, evo_config, objective=objective, key=k1)
result = coarse.model.fit(data, optax_config, objective=objective, key=k2)
```

Each `fit` takes its own `freeze=`. The
[batch reactor example](/examples/batch-reactor) runs Evosax then Optax and
freezes a different half of the predictors in each fit.

## Freezing leaves

The default mask, `trainable_mask(predictors)`, trains floating-point
arrays and freezes `Constant` values and `BoundScaler` temperatures.
`freeze=` on `fit` holds more fixed. Each entry is a dotted path, a module
class, or a predicate on modules:

```python
result = model.fit(data, config, freeze=("offset",), key=key)                 # one predictor
result = model.fit(data, config, freeze=(hm.MLPPredictor,), key=key)          # every MLP
result = model.fit(
    data, config, key=key,
    freeze=(lambda m: isinstance(m, hm.BoundedPredictor) and "temp" in m.input_keys,),
)
```

Paths are dot-joined: dict keys and module fields by name, tuple indices as
numbers. A path may stop at a subtree (`"rate"`, `"rate.inner"`) or name
one array (`"rate.inner.mlp.layers.0.weight"`). A class or predicate
freezes the whole subtree of every module it matches, so matching an outer
`BoundedPredictor` freezes everything inside it. An entry that matches
nothing raises, and a path suggests the closest matches.

For full control pass an explicit boolean mask as `trainable=` instead;
passing both raises. The mask is a pytree matching the predictors, built
with the same helpers:

```python
mask = hm.trainable_mask(predictors, train_scalers=True)    # also train scaler temperatures
mask = hm.freeze_paths(mask, ("rate.inner.mlp.layers.0.weight",))
mask = hm.freeze_modules_of_type(mask, hm.MLPPredictor)
mask = hm.freeze_mask(predictors, ("offset",), mask=mask)    # the freeze= rules on a mask
result = model.fit(data, config, trainable=mask, key=key)
```

`freeze_modules_of_type` and `freeze_where` read module types off the mask
itself, so their predicate sees booleans in place of arrays: test types and
static fields such as `input_keys`, never array values. A predicate passed
to `freeze=` or `freeze_mask` sees the real modules.

`model.fit` raises if nothing is left to train.

## Progress display

`verbose=True` (the default) shows [`RichTrainingUI`](/api/ui#richtrainingui),
a live terminal dashboard that also reports per-bucket compile time.
`verbose=False` uses [`SilentUI`](/api/ui#silentui). `ui=` overrides both
with any object implementing [`TrainingUI`](/api/ui#trainingui). If a run
raises after the dashboard opens, the UI gets `on_run_end` with a NaN loss
before the error propagates.

## Reproducibility

Every random choice comes from the `key=` you pass. For named subkeys in
your own code, [`fold`](/api/rng#fold) (`from jaxhybridmodels.rng import
fold`) derives a key from a name, so adding a consumer does not shift the
others the way `jr.split` would.
