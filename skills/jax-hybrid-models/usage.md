# Usage reference

## Contents

- Data layer (experiments, channels, dataset)
- `simulate_fn` and the vector field
- Predictors and bounds
- Training with Optax
- Training with evosax
- Prediction and metrics
- Serialisation
- Losses
- Time profiles and penalties (pointers)

## Data layer

`make_experiment` — all keyword-only:

```python
exp = hm.make_experiment(
    covariates={"temperature_C": 25.0, "feed": jnp.array([0.2, 0.5, 0.3])},  # scalars or rank-1 vectors, constant in time
    channels={
        "conc": hm.ChannelObs(ts=ts_conc, values=y_conc, variance=var_conc),  # var: array or float
        "d43":  hm.ChannelObs(ts=ts_d43,  values=y_d43,  variance=var_d43),   # channels may be sampled differently
    },
    y0_fn=lambda c, ch: jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, ch["conc"].values[0]]),  # full state [S], may exceed channels
    exp_id="run_1",
)
```

- `y0_fn` receives `(covariates_dict, channels_dict)` and must return the **full** initial state `[S]`, including components nobody measured.
- A channel with `values=jnp.array([])` is a **probe**: its `ts` defines the integration grid, the mask is all-False, the data loss is exactly zero (used for trajectory penalties / steer-to scenarios).
- Covariate keys must be identical across experiments; a given key must have one shape across the dataset.

```python
dataset = hm.make_dataset(experiments, output_channel_names=("conc", "d43"))
# validates: same covariate keys everywhere; EVERY experiment defines EVERY output channel.
# buckets experiments by len(union_ts); mask is computed for you, never hand-written.
train_ds, val_ds, test_ds = hm.split_dataset(dataset, train=0.8, val=0.1, test=0.1, key=key)
boot_ds = hm.make_bootstrap_dataset(dataset, key=key, n_experiments=None)  # one resample, with replacement
# or train an ensemble directly:
members = hm.train_bootstrap_ensemble(predictors, dataset, config, simulate_fn=simulate_fn,
    state_to_output=state_to_output, solver=solver, n_bootstraps=50, n_seeds=1,
    k_best=None, trainable=mask, key=key)  # -> list[(score, predictors)]
```

`BucketPayload` is a NamedTuple of stacked `[N, T, ...]` arrays — one per bucket. You rarely touch it directly.

## `simulate_fn` and the vector field

Fixed signature, pure, JAX-transformable (no Python `if` on traced values, no side effects):

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    growth, nucleation = predictors  # tuple-unpack at the top; one BoundedPredictor per rate

    def vector_field(t, y, args):
        # Build ONE inputs dict per step, pass it to every predictor.
        # Mix constant covariates with state-derived and time-varying values.
        inputs = {
            "temperature_C": covariates["temperature_C"],
            "supersaturation": y[CONC_IDX] / covariates["c_sat"],
        }
        G = growth(inputs)
        J = nucleation(inputs)
        return physics_rhs(t, y, G, J, covariates)

    return solver.diffeqsolve(
        diffrax.ODETerm(vector_field), ts, y0,
        adjoint=solver.adjoint,            # configured on SolverConfig
        stepsize_controller=solver.stepsize_controller(),  # handles per-state atol arrays
    ).ys
```

- `BoundedPredictor.__call__` accepts the dict (extracts `input_keys` in declared order; missing keys raise, extras ignored) or a rank-1 Array.
- Time-varying exogenous values: evaluate a profile factory in the vector field, e.g. `hm.ramp_profile(t0=cov["t0"], t1=cov["t1"], v0=cov["T_lo"], v1=cov["T_hi"])(t)` — parameters travel as ordinary covariates.
- `state_to_output(state)` maps `[T, S]` (or `[N, T, S]` when vmapped) → `[T, D]`. It is passed to training/prediction, never stored on the Dataset. Guarded divisions belong here.

## Predictors and bounds

`Predictor` subclasses: `MLPPredictor(in_size, out_size, width_size, depth, activation_name, key)`, `KANPredictor`, `NeuralNPolynomial`. Custom predictors subclass `hm.Predictor` (eqx.Module): fields are JAX arrays or static JSON-encodable values — **every predictor must round-trip `eqx.tree_serialise_leaves`** (all leaves arrays, static fields JSON-encodable primitives/tuples, no closures in non-static positions).

`BoundedPredictor` composition: `input_keys (named input order) → in_scaler → inner → out_scaler` → single Array in physical units.

- `input_keys` order must match `in_scaler.bounds` cardinality (≥ 1); auto-fills `("x1", ...)` if omitted.
- `BoundScaler(bounds=((lo1, hi1), (lo2, hi2)), warp=..., transform=...)` — one tuple per input/output slot.
  - `warp`: `"linear"` (default), `"log"`, `"log10"` — use log warps when a box spans decades. Log warps reject non-positive bounds at construction.
  - `transform`: `"sigmoid"` (default), `"algebraic"`, `"softsign"`. Gradient death at latent: sigmoid 16.8, algebraic ~3e3, softsign ~1.1e7 (softsign is not C^2; kink at box midpoint).
  - `to_latent` uses a linear continuation (`soft_inverse`) outside the box, never a hard clip.
- Zero-initialised final head: `inner = inner.with_zero_final_head()` (MLP and KAN) — initial output is exactly the box midpoint; reach for it when `max_steps exceeded` shows up on some seeds only.

Trainability mask (same structure as `predictors`, default = all float arrays):

```python
mask = hm.trainable_mask(predictors)
mask = hm.freeze_modules_of_type(mask, predictors, hm.BoundScaler)  # freeze temperature: ALWAYS
mask = hm.freeze_paths(mask, predictors, ...)                       # paths = dotted names
mask = hm.freeze_where(mask, predictors, lambda m: isinstance(m, hm.MLPPredictor))
```

`freeze_where`'s predicate runs on every module **node**; a matching node freezes its whole subtree. `BoundedPredictor` is the outermost node, so a broad predicate there freezes everything — narrow to the inner class.

## Training with Optax

```python
config = hm.OptaxTrainingConfig(
    steps=(1000,), lr=(1e-3,), optimizer=("adamw",),   # phase-keyed: equal-length tuples, NO scalar broadcast
    reset_optimiser_state=(False,), length_schedule=(1.0,),  # (0,1] fraction of each trajectory the loss sees
    loss="mse",                     # "mse" | "mle" | "bal_mse" | "bal_mle" | callable
    channel_idx=(0, 1),             # optional: restrict loss to channels
    penalty_weight=(1e-3,),         # bound-saturation penalty; length-1 broadcasts; 0.0 disables
    penalty_points=(sweep,),        # optional [G, n_inputs] per leaf, in traversal order
    tournament_attempts=8, tournament_steps=20, tournament_lr=1e-4,  # on only when attempts>1 AND steps>0
    patience=0, restore_best=True,
    verbose=False,                  # True -> RichTrainingUI; explicit ui=... always wins
)
history, trained = hm.train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, trainable=mask, key=key,   # key: keyword-only, required
)
```

Semantics that matter:

- **A step = one epoch**: full pass over all buckets, gradients accumulated, one `optimizer.update`. The Python bucket loop drives JIT-cached per-bucket-shape kernels — the loop is idiomatic, not a smell.
- **Phases**: each phase gets its own `(steps, lr, optimizer, reset_optimiser_state, length_schedule)`. `optimizer` entries: name string, factory taking `learning_rate` (chains/clipping compose), or a ready-made `optax.GradientTransformation`. Changing `lr` with a raw instance requires `reset_optimiser_state=True` there.
- `restore_best`/`patience` reset at phase boundaries (a `length_schedule` change redefines what the loss measures).
- **Bound penalty** (saturation, charged in latent space): added to the optimiser's objective, but `losses_history`/`restore_best`/early stopping/tournament track the data term alone.
- **Tournament**: fresh re-init candidates trained briefly then scored forward-only, reusing the compiled step kernels (no extra compile). Triggers on diffrax runtime errors / non-finite scores; all-fail falls back with a `RuntimeWarning`.
- **Trajectory penalty** (opt-in `trajectory_penalty_fn(full_state, bp)` + weight): rides in the per-step loss. Embedded predictors accumulate it as ODE state (`attach_penalty_state`, `penalty_vector_field`, `strip_penalty_state`, `penalty_integral`); parallel predictors use `trajectory_saturation_penalty`.

## Training with evosax

Use when the trainable set is small (~4–10 dims), kinetic-parameter-shaped, or multi-modal. Run evosax first, refine with Optax; both take the same `trainable=` mask.

```python
config = hm.EvosaxTrainingConfig(
    algorithm="CMA_ES", population_size=64, num_generations=200,
    init="warm",               # "warm" | "uniform_box" | "lhs_box" (init_box_extent=2.0)
    sigma_init=0.1,            # conservative; no per-individual error handling in v1
    loss="mse", penalty_weight=0.0,
    verbose=False,
)
history, trained = hm.train_with_evosax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, trainable=mask, key=key,
)
```

Bounds are **not** enforced during search — CMA-ES wanders latent space; the sigmoid reparameterisation keeps physical outputs in range. A diffrax/non-finite individual crashes the generation.

## Prediction and metrics

```python
pred = hm.predict_dataset(trained, dataset, simulate_fn=simulate_fn,
                          state_to_output=state_to_output, solver=solver)   # tuple of per-bucket [N, T, D] arrays
dense = hm.predict_dense(trained, dataset, ...)                              # on a dense grid
value = hm.evaluate_predictor(trained[0], {"temperature_C": 25.0})           # one scalar readout as float
ens = hm.ensemble_predictions([trained_a, trained_b], dataset, ...)
metrics = hm.compute_metrics(pred, dataset); hm.print_metrics(metrics)
```

`predict_bucket` is jitted separately from training (forward-only, one trace per bucket shape).

## Serialisation

```python
hm.save_run("run_dir", predictors=trained, solver=solver,
            optax_config=config, loss_history=history, extras={"note": "..."})
out = hm.load_run("run_dir", predictors_template=untrained_skeleton, optax_cls=hm.OptaxTrainingConfig)
# template must share container shape, module types, and static fields (bounds, warps, widths, depth...)
hm.save_predictors("p.eqx", trained); restored = hm.load_predictors("p.eqx", template)
```

- Saved: predictors `.eqx` + metadata JSON. Not saved: `simulate_fn`, `state_to_output`, Dataset, masks.
- Configs with executable callables return as metadata markers — rebind `trajectory_penalty_fn` etc. explicitly.
- `SolverConfig` round-trips via name registries (`SOLVER_REGISTRY`, `ADJOINT_REGISTRY`); custom solvers/adjoints need `register_solver`/`register_adjoint` first.

## Losses

Built-ins: `masked_mse`, `masked_mle`, `bal_mse`, `bal_mle` (selected by name string; `bal_*` weighs by channel variance). Custom loss signature: `loss(pred_obs: [N, T, D], bp: BucketPayload) -> scalar` — pass the callable in `config.loss`.

## Penalties and profiles quick reference

- `hm.box_grid(predictor, n_per_dim)` — deterministic warp-uniform sweep of a predictor's input box; feed as `penalty_points`.
- `hm.data_penalty_points` — the measured points the loss sees; gathered automatically.
- `hm.box_violation`, `hm.saturation`, `hm.penalty_integral`, `hm.clip_ste` — building blocks for custom penalty fns.
- Profiles: `constant_profile`, `step_profile`, `ramp_profile` (exact flat edges), `piecewise_linear_profile` (extends edges outward) — all return `t -> Array`, jit/vmap-safe, host-side validated.
- `hm.annealing_schedule(total_epochs, init_value, end_value)` — epoch-scaled multiplier for custom loops.