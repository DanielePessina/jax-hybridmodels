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
result.stop_reason          # "budget_exhausted", "patience" or "diverged"
```

The data may be a list of experiments or a `Dataset`. Each history entry is
measured at the parameters before that step's update, and `loss + penalty`
is what the optimiser minimised. `key=` is keyword-only and required, so the
seed is visible at every call.

A solver error or a non-finite loss partway through a run ends it. `fit` warns
and returns the best parameters so far, with `stop_reason="diverged"`, so a long
run keeps its work. If the first step fails there is nothing to return, so `fit`
raises. Errors in your own code always raise.

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
callable `(pred_obs: [N, T, D], bp: BucketPayload) -> scalar`. To use a callable
with `channel_weights` other than 1, give it the keywords the built-in losses
take, `(pred_obs, bp, *, channel_idx=None, channel_weights=None)`, and apply
them in its own reduction. A loss that wraps a built-in one passes them on:

```python
def my_loss(pred_obs, bp, *, channel_idx=None, channel_weights=None):
    data = hm.losses.bal_mle(
        pred_obs, bp, channel_idx=channel_idx, channel_weights=channel_weights
    )
    return data + my_extra_term(pred_obs, bp)
```

`channel_weights` keyed by channel name scores only the channels it names,
so `{"concentration": 1.0}` fits one channel of several. A sequence gives
one weight per output channel, in `output_channel_names` order. Names are
checked against the data when training starts, and an unknown one raises
before anything compiles. A named channel with weight `0.0` adds nothing
but still counts in the loss's denominator, which scales the loss down;
leave the name out to drop the channel.

The `bal_*` losses average each channel over time within an experiment before
summing channels, so a channel with one point per run counts as much as one
with ten. The `mle` losses divide by each point's variance, which handles
channels on different scales. A weight is left for a model that cannot fit
every channel at once: raising one channel's weight trades fit on the others,
and the best value is found by trying a few. [Extending](/guide/extending#losses) covers how
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

`model.fit` picks `train_with_optax` or `train_with_evosax` from
`jaxhybridmodels.training` by the type of `config`. They take the model
first and then `fit`'s arguments, and return the same `FitResult`:

```python
from jaxhybridmodels.training import train_with_optax

result = train_with_optax(model, data, config, objective=objective, key=key)
```

For a loop of your own, the compiled pieces they are built from are in
`jaxhybridmodels.training.kernels`; see [Extending](/guide/extending).

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
keys, trains each for `tournament_steps`, scores each with a forward pass,
and keeps the best:

```python
config = hm.OptaxTrainingConfig(
    steps=1000, lr=1e-3,
    tournament_attempts=8, tournament_steps=30,
)
```

It runs only when `tournament_steps > 0` and `tournament_attempts > 1`.
The attempts train with the first phase's optimiser, learning rate and
horizon, since the winner goes on to train in that phase. `tournament_lr`
overrides the rate. Keep it close to the phase's: at a hundredth of it the
attempts barely move, and the tournament ranks random starting points
rather than where each is heading.

The tournament reuses the compiled training step, so it adds no compile
time. An attempt that hits a Diffrax runtime error or a non-finite loss is
dropped; any other error propagates. If every attempt fails, training
continues from your original predictors with a `RuntimeWarning`. Ties keep
the earlier attempt, so the result is deterministic in the key.

Every attempt's score is in `result.history.tournament_scores`, and the
summary prints the best and worst. A wide spread means the starting point
matters for this model. A narrow one means the tournament is not buying
much, and its steps are better spent training.

What gets redrawn:

- Networks get fresh weights.
- A `Parameter` with two finite bounds gets a random value across the middle
  of its box. One without keeps its starting value, since there is no range
  to draw from.
- `Constant` leaves are never re-initialised, and frozen leaves go back to
  their given values after each redraw, so a restart cannot leave a frozen
  array at random values. To compare different frozen arrays, build a model
  for each.

### Multiple shooting

Single shooting solves each experiment once, from `y0` to its last sample.
From a poor guess, an oscillating or stiff system or a long horizon can
drift away from the data early. The loss then stops measuring how wrong the
parameters are, and training stalls in the nearest dip. Multiple shooting
splits each trajectory into segments. Segment 0 starts from `y0`, and every
later segment starts from its own trainable state, a node. A gap term pulls
the segments back into one trajectory:

<<< @/../examples/multiple_shooting/oscillator.py#configs

- **Segments are counted in samples.** Consecutive segments share one sample.
  A bucket with too few samples uses fewer segments.
- **A node is the full state**, unmeasured components included. When a
  segmented phase starts, the nodes are read off a solve at the current
  predictors, so the segments start joined. An experiment whose solve fails
  starts its nodes at `y0`.
- **The loss** is the data term on the joined-up segments, plus
  `continuity_weight * mean((gap / scale)^2)`. The scale is each state's
  largest size along that starting solve, so the gap term has no units. A
  plain `"mse"` loss is in the units of the data, so the balance between
  the two depends on the data's scale.
- **The schedule must end with `segments=1`.** The nodes are dropped when
  the run ends, and the returned model has been fitted as the single solve
  that `predict` runs. `restore_best` picks from that last phase. With
  `validation=`, it picks by the validation score instead, which is always
  single shooting at the full horizon, so it can pick from any phase.
- **A change in `segments` resets the optimiser**, because the nodes change
  shape. Expect a short spike in the loss there.
- **`history.continuity`** records the gap term per step, and the summary's
  phase table shows it at the end of each segmented phase.

Segments must be short compared with how fast the model at its starting
guess moves. A rule of thumb for the first phase:

- each segment spans 5 to 10 samples, so the union length divided by 5 to
  10 gives the count;
- each segment is no longer than about half the period of the fastest
  oscillation, or a decay time, at the starting guess and in the data;
- then cut the count by about four per phase, such as 16, 4 and then 1, with
  a few hundred steps each.

On 81 predator-prey samples from a poor guess, 16, 4 and 1 recovered the
parameters where single shooting stalled. If a segmented fit still settles in
the wrong place, use more segments. Too many is safe but slow to tie back
together, since every node must be pulled into line. The [multiple shooting
example](/examples/multiple-shooting) fits a frequency that single shooting
gets wrong from two of three starts.

A tournament whose first phase is segmented trains its attempts with
segments too. Evosax and `predict` always use single shooting.

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
stay inside their bounds without any constraint handling.

A wide box often holds candidates the solver cannot finish, such as a stiff
one under an explicit solver. Such a candidate ranks last in its generation
and the search goes on. `result.history.failed_candidates` counts them per
generation, the progress display notes the first one, and `summary()` gives the
total. Only a generation where every candidate fails ends the run, with the
best parameters so far. A `simulate_fn` that calls `diffrax.diffeqsolve`
itself must pass `throw=solver.throw`, or a failed solve still ends the run.

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
passing both raises. The mask is a pytree matching the predictors.
`freeze_mask` in `jaxhybridmodels.trainable` is what `freeze=` calls, and
chains onto any base mask through `mask=`:

```python
from jaxhybridmodels.trainable import freeze_mask

mask = hm.trainable_mask(predictors, train_scalers=True)    # also train scaler temperatures
mask = freeze_mask(predictors, ("offset", hm.MLPPredictor), mask=mask)
result = model.fit(data, config, trainable=mask, key=key)
hm.count_trainable_params(predictors, mask=mask)
```

`freeze_mask(predictors, freeze, *, mask=None)` takes the same entries as
`freeze=`, a single one or a sequence, and its predicates see the real
modules.

`model.fit` raises if nothing is left to train.

## Progress display

`fit(..., verbose=True)`, the default, shows
[`RichTrainingUI`](/api/ui#richtrainingui), a live terminal dashboard that
also reports per-bucket compile time. `verbose=False` uses
[`SilentUI`](/api/ui#silentui). When output is not a terminal, such as a log
file or CI, `verbose=True` prints a plain line about every 10% of the run
instead. `ui=` overrides all of these with any object implementing [`TrainingUI`](/api/ui#trainingui). If a run
raises after the dashboard opens, the UI gets `on_run_end` with a NaN loss
before the error propagates.

## Reproducibility

Every random choice comes from the `key=` you pass. For named subkeys in
your own code, [`fold`](/api/rng#fold) (`from jaxhybridmodels.rng import
fold`) derives a key from a name, so adding a consumer does not shift the
others the way `jr.split` would.
