# Training

`hybridmodels` ships two training entry points with **the same signature**:

```python
history, trained = train_with_optax(predictors, dataset, config, *, simulate_fn, solver, trainable=None, key, ui=None)
history, trained = train_with_evosax(predictors, dataset, config, *, simulate_fn, solver, trainable=None, key, ui=None)
```

Both return `(loss_history, trained_predictors)`. Both demand a `key` — the framework never silently defaults to `jr.PRNGKey(0)`, so reproducibility is explicit.

## Optax: gradient-based, multi-phase

[`train_with_optax`](/api/training#train_with_optax) runs an Optax loop with a **multi-phase schedule**. Each step is one full pass over all buckets, accumulating gradients across them, then one `optimizer.update`. (A step is *not* one bucket — see [Concepts → Bucket, step, phase](/guide/concepts#bucket-step-phase).)

### Phases

A phase is a contiguous block of N steps sharing the same hyperparameters. A run is a tuple of phases. Every phase-keyed field on [`OptaxTrainingConfig`](/api/training#optaxtrainingconfig) is a **required tuple of equal length** — there is no scalar broadcast.

```python
from hybridmodels.training.optax import OptaxTrainingConfig

config = OptaxTrainingConfig(
    steps=(200, 800),
    lr=(1e-2, 1e-3),
    optimizer=("adamw", "adamw"),
    reset_optimiser_state=(False, True),
    length_schedule=(0.5, 1.0),
    loss="mse",
    verbose=True,
)
```

This runs 200 warm-up steps at `lr=1e-2` on the **first half** of every trajectory (`length_schedule=0.5`), then 800 fine-tuning steps at `lr=1e-3` on the full trajectory.

`length_schedule` is a per-phase fraction in `(0, 1]` applied as a **runtime mask cutoff** — the loss simply ignores observations beyond `int(T * length_schedule[phase])`. There is **no JIT recompile across phase boundaries**; the same compiled `make_step` is reused.

`reset_optimiser_state=True` rebuilds the optimiser state at a phase boundary. Set it whenever you switch optimiser type or whenever a `length_schedule` jump invalidates the running momentum.

### The shared tournament

Random init can be brutal — a bad omega draw can land the integrator in a stiff regime that `max_steps` cannot escape. The shared tournament runs N serial warm-up candidates, picks the one with the best loss, and continues normal training from there.

```python
config = OptaxTrainingConfig(
    steps=(1000,), lr=(1e-3,), optimizer=("adamw",),
    reset_optimiser_state=(False,), length_schedule=(1.0,),
    loss="mse",
    tournament_attempts=8,
    tournament_steps=20,
    tournament_lr=1e-2,
)
```

Implementation notes:

- **Shared** means it reuses the **main loop's compiled `make_step` and `apply_update`** — there is no extra JIT compile cost. (vmapped and serial-with-fresh-jit modes are explicitly out of scope; see ADR 0002.)
- **Per-attempt failure** (diffrax error / non-finite loss) drops the candidate and tries a fresh RNG.
- **All attempts failing** falls back to the original predictors with a `RuntimeWarning` rather than crashing.
- **Re-init splits the key by traversal order**: identical-shape sibling predictors get *different* re-init weights, not the same ones.

The tournament is enabled implicitly when `tournament_steps > 0 AND tournament_attempts > 1` — leave both at their defaults (`0`, `1`) for a single-shot run.

### Loss selection

Pass a string to look up a built-in via [`LOSS_REGISTRY`](/api/losses#loss_registry):

```python
config = OptaxTrainingConfig(..., loss="mse")     # masked_mse
config = OptaxTrainingConfig(..., loss="bal_mle") # bal_mle
```

Or pass a callable matching the signature `(pred_obs: [N, T, D], bp: BucketPayload) → scalar` for a custom loss. `channel_idx=` and `channel_weights=` weight specific channels:

```python
config = OptaxTrainingConfig(
    ...,
    loss="mse",
    channel_idx=(0, 1),
    channel_weights=(1.0, 0.1),  # de-prioritise channel 1
)
```

## Evosax: population-based search

[`train_with_evosax`](/api/training#train_with_evosax) reaches into population-based optimisation when the loss landscape has many local minima or when the "predictor" is a small kinetic-parameter vector that gradient methods over-fit.

```python
from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax

config = EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=64,
    num_generations=200,
    init="lhs_box",         # Latin Hypercube via scipy.qmc, host-side
    init_box_extent=0.5,    # ±0.5 in latent space around the warm mean
    sigma_init=0.3,
    loss="mse",
    verbose=True,
)
history, trained = train_with_evosax(
    predictors, dataset, config, simulate_fn=simulate_fn, solver=solver, key=key,
)
```

Design notes:

- **Targeted at small-parameter predictors** (~4-10 dims). Not optimised for NN-sized search in v1.
- **Single-eval JIT boundary**: `population_eval = eqx.filter_jit(jax.vmap(single_eval))`. The bucket-dispatch Python loop unrolls inside the trace.
- **Init modes**: `"warm"` (mean = current params, default), `"uniform_box"` (latent ±extent uniform), `"lhs_box"` (Latin Hypercube).
- **Bounds during search are not enforced.** CMA_ES wanders latent space; `BoundedPredictor`'s sigmoid keeps physical outputs in range.
- **Best-ever tracked host-side** (`jnp.argmin(fitnesses)` per generation), not via strategy-specific best-member fields.
- **No graceful per-individual error handling.** A diffrax/non-finite individual crashes the generation; mitigation is conservative `sigma_init` and `init_box_extent`.

### Composing Optax + Evosax

There is no polishing field on `OptaxTrainingConfig`. Compose by calling them in sequence:

```python
# Coarse search.
hist1, predictors = train_with_evosax(predictors, dataset, evo_config,
                                       simulate_fn=simulate_fn, solver=solver, key=k1)
# Fine-tune with gradients.
hist2, predictors = train_with_optax(predictors, dataset, opt_config,
                                      simulate_fn=simulate_fn, solver=solver, key=k2)
```

Both calls accept the same `trainable=` mask, so freezing carries forward unchanged.

## Freezing leaves

The `trainable=` keyword on both training functions accepts a boolean PyTree mask matching the predictors structure. Default behaviour (omit it) marks every inexact-float leaf trainable; pass a custom mask to hold subsets fixed.

```python
from hybridmodels import (
    BoundScaler, freeze_modules_of_type, freeze_paths, trainable_mask,
)

mask = trainable_mask(predictors)
mask = freeze_modules_of_type(mask, predictors, BoundScaler)  # freeze all scalers
mask = freeze_paths(mask, ("0", "inner"))                      # freeze the first predictor's inner net

history, trained = train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, solver=solver, trainable=mask, key=key,
)
```

Compose freezers — they all return a new mask. See [Trainable Masks](/api/trainable).

## UIs and logging

Pass `ui=` to override the config's `verbose` flag:

- `verbose=True` → [`RichTrainingUI`](/api/ui#richtrainingui) live dashboard.
- `verbose=False` → [`SilentUI`](/api/ui#silentui).
- `ui=YourUI()` → custom UI implementing the [`TrainingUI`](/api/ui#trainingui) protocol.

Lifecycle hooks include compile-phase progress (`on_compile_start` / `on_compile_progress` / `on_compile_done`), so per-bucket-shape compile time is **visible**, not silent.

## Reproducibility checklist

- Seed your root key explicitly: `key = jr.PRNGKey(seed)`.
- Pass `key=` to every `train_with_*` call.
- Don't rely on dict ordering anywhere downstream of training (Python's insertion order saves you, but pin it explicitly when you can).
- Use [`fold`](/api/rng#fold) — not `jr.split` — when you need named subkeys in your own code.

## What's next

- [Recommendations](/guide/recommendations) — practical guidance for bounds, solver tolerances, and common pitfalls.
- [Crystallisation walkthrough](/examples/crystallisation) — sees the multi-phase config in a real workflow.
- [Training API](/api/training) — full reference for both configs and trainers.
