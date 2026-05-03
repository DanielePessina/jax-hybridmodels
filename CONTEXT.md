# jax-hybridmodels

A JAX/Equinox library for **hybrid models** — composing trainable function approximators (MLP, KAN, ...) with user-written ODE dynamics, trained on irregular time-series experiments. Crystallisation kinetics is the canonical example, not the scope.

## Language

**Predictor**:
A narrow trainable `eqx.Module` whose `__call__` is `Array → Array`. Knows nothing about covariate names, bounds, or experiments. Concrete examples: `MLPPredictor`, `KANPredictor`, `NeuralNPolynomial` (the last composes another `Predictor` for coefficients).
_Avoid_: Regressor (the previous package's overloaded term — bundled bound-scaling, rate-pair semantics, and trainable weights together).

**CovariateSelector**:
An `eqx.Module` that pulls a fixed tuple of named covariates out of the dict and stacks them in declared order. Static `keys: tuple[str, ...]`.

**BoundScaler**:
An `eqx.Module` providing bidirectional sigmoid scaling between physical `[low, high]` and a latent space. Sigmoid only for v1; tanh deferred. No "temperature" knob in v1 unless we re-grill (default sigmoid: `low + (high-low) * sigmoid(z)`).

**BoundedPredictor**:
A composition wrapper: `selector → in_scaler → inner Predictor → out_scaler`. Returns a single `Array` (the physical-units output). **No penalty term in v1** — the bound-excursion penalty machinery from the source package was never properly wired and is dropped.

**RatePair**:
Composition of two `BoundedPredictor`s (`nucleation`, `growth`). Stacks their outputs. Used by the crystallisation example; not built into the framework.

**simulate_fn**:
A pure user-written function with a mandatory signature that, for one experiment, integrates the dynamics and returns the **full state trajectory**. The framework owns vmapping, jitting, and gradient flow; the user owns physics.
_Avoid_: "ODE function" (ambiguous — could mean the vector field), "solve" (overlap with diffrax).

Mandatory signature:
```python
def simulate_fn(
    predictor,                    # trainable eqx.Module
    ts: Float[Array, "T"],         # observation times for this experiment
    covariates: dict[str, Array],  # named, constant-in-time scalars
    y0: Float[Array, "S"],         # full initial state
    solver: SolverConfig,          # static; diffrax bits
) -> Float[Array, "T S"]:           # full state at each ts
    ...
```

**state_to_output**:
A pure callable mapping a full-state trajectory `[T, S]` to the observed output channels `[T, D]` (e.g. crystallisation: `[mu0..mu4, conc] → [conc, d43]`). Applied externally to `simulate_fn`'s output, before loss computation.

**y0_fn**:
A user-supplied hook invoked at **data-import time** to construct the per-experiment full initial state from the raw row + covariates. Default for "state == observed": `lambda row, c: row.y[0]`. Crystallisation-style: `lambda row, c: jnp.array([0,0,0,0,0, row.y[0,0]])`.

**SolverConfig**:
A frozen `eqx.Module` whose fields are all `eqx.field(static=True)` — diffrax solver instance, rtol, atol (scalar or per-state tuple), max_steps, dt0. Static so it doesn't enter the pytree leaves. JSON-serialisable via a small solver class-name registry.

**Covariates**:
Named scalars that are **constant in time** for an experiment (e.g. `temperature_C`, `loading`). Always passed as `dict[str, Array]` — no canonical-order array. Time-varying covariates are out of scope for v1.

**Bucket**:
A group of experiments sharing the same `len(union_ts)`. Within a bucket, individual experiments may have different `ts` values and different masks (mask is a per-experiment array). Not promoted to a class — `BucketPayload` is just a `NamedTuple` of stacked `[N, T, ...]` arrays produced once by `make_dataset`.

**Experiment**:
A user-built record holding **per-channel sparse observations**. Each channel has its own `(ts_channel, values, variance)` triple, allowing arbitrary sparsity per channel. The framework computes the per-experiment union timestamp axis and the resulting mask automatically at `make_dataset` time — users never write mask code.

**ChannelObs**:
Per-channel observation triple: `(ts: [Tc], values: [Tc], variance: [Tc] | float)`. A scalar variance broadcasts over `Tc`.

**step** (training):
One full pass over all buckets → accumulate gradients across buckets → **one** `optimizer.update`. Equivalently: one epoch. *Bucket ≠ step.*

**phase** (training):
A contiguous block of N steps sharing the same hyperparameters (lr, optimizer, …). A run is a tuple of phases.

**bucket dispatch loop**:
The Python `for bp in bucket_payloads:` that drives JIT-cached per-bucket kernels. Idiomatic; not a performance smell.

## Relationships

- A **simulate_fn** consumes a **Predictor**, a **SolverConfig**, and one **Experiment**'s `(ts, covariates, y0)`; returns a full state trajectory.
- **state_to_output** is composed externally: `loss(state_to_output(simulate_fn(predictor, ...)), y_observed, mask)`.
- A **Bucket** holds N **Experiments** with identical `len(ts)`; the training loop vmaps `simulate_fn` over the bucket.
- The **Predictor** is the *only* component that is binary-serialised (eqx.tree_serialise_leaves). **simulate_fn** and **state_to_output** are code (re-imported); **SolverConfig** is JSON.

## Optax training

- **Phases** are tuples; **all phase-keyed fields are required tuples of equal length** (no scalar broadcast). Fields: `steps, lr, optimizer, reset_optimiser_state, length_schedule`.
- **`length_schedule`** is a per-phase fraction in `(0, 1]`, applied as a **runtime mask cutoff** — no JIT recompile across phase boundaries (default `(1.0,)` for one phase = no scheduling).
- **`reset_optimiser_state`** is per-phase (tuple of bool); a True entry rebuilds the optimiser at that phase boundary (used when switching optimiser type or when length-schedule changes invalidate momentum).
- **`make_step(predictor, opt_state, bucket_payload)` is jitted** and returns `(loss, grads)`. Optimiser update happens in a separate jitted `apply_update(predictor, accumulated_grads, opt_state)` *after* the Python for-loop over buckets has accumulated.
- **Shared tournament only.** Enabled implicitly when `tournament_steps > 0 AND tournament_attempts > 1`. Serial candidate evaluation that *reuses the main loop's compiled `make_step` and `apply_update`* (no extra JIT compile cost). On per-attempt failure (diffrax error / non-finite loss), drop and try a fresh RNG; if all fail, fall back to the original predictor with a `RuntimeWarning`.
- **`Predictor.initialized_with_key(key)`** is a documented protocol used by the tournament. Default implementation is a free function `reinitialize_with_key(predictor, key)` that re-inits inexact-float leaves only.

## Loss interface

- A loss is a pure function `loss(pred_obs: [N, T, D], bp: BucketPayload) → scalar`.
- The framework wraps it: `loss_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(_simulate_then_loss))`, where `_simulate_then_loss` does the vmap of `simulate_fn`, applies `state_to_output`, and calls the user's loss.
- Built-ins: `masked_mse`, `masked_mle`, `bal_mse`, `bal_mle`. Built-ins accept optional `channel_idx` and `channel_weights` kwargs (carried over from the source package; may be reworked later).
- User-supplied losses just match the signature.

## UI

- **Callback-based**, not log-based. Training functions accept `ui: TrainingUI` (or `EvosaxUI`); two protocols, no merged supertype.
- **Default selection**: `verbose: bool = True` on each training config picks `RichTrainingUI` / `RichEvosaxUI`; `False` picks `SilentUI`. An explicit `ui=...` parameter on the training function overrides the config flag.
- **Lifecycle events** include compile-phase progress (`on_compile_start/_progress/_done`) so per-bucket-shape compile time is visible, not silent.
- **Concrete shipped UIs**: `SilentUI`, `RichTrainingUI`, `RichEvosaxUI`. Single `rich.live.Live` per training run; panels swap as run progresses.
- **Out of scope for v1**: live loss plots, ETA columns, per-bucket per-step breakdowns, notebook-specific layouts.

## Serialisation

- **Hard design constraint** (applies *now* to every Predictor we design): every `Predictor` / `BoundedPredictor` must round-trip through `eqx.tree_serialise_leaves` ↔ `eqx.tree_deserialise_leaves` with zero JAX/Equinox conflicts. Concretely: all dynamic leaves are JAX arrays; all static fields are JSON-encodable primitives or tuples thereof; no closures in non-static positions.
- **Implementation is the last shipped feature.** The save/load helpers (`save_predictor`, `load_predictor`, optional `save_run`/`load_run`) are written after the rest of the framework is verified.
- **Save format (when implemented)**: a directory containing `predictor.eqx` + `metadata.json` (timestamp, version, predictor class path, simulate_fn module hint, user `extras`).
- **Not serialised by the framework**: `simulate_fn` and `state_to_output` (pure functions, re-imported), `Dataset`, the trainable mask (rebuild from user code), training configs (user owns).
- **No builder registry in v1.** Loading requires a user-supplied `template` predictor, per `eqx.tree_deserialise_leaves`. Builder registry is deferred until friction proves real.

## RNG discipline

- **Root key is user-supplied.** Framework never silently defaults `jr.PRNGKey(0)`; missing key raises.
- **Named folds.** Internal subkeys derived via `jr.fold_in(root, _id("name"))` where `_id` is a stable hash of the consumer name. Avoids the chained-split fragility where reordering or inserting a consumer shifts every downstream key. Names: `"init"`, `"tournament"`, `"phase_{i}"`, `"evosax_init"`, `"evosax_ask_{gen}"`.
- **Bucket visit order is fixed**, not shuffled. Full-batch gradient accumulation is commutative; the source package's per-step shuffle is dropped.
- **No stochasticity in v1**: no dropout, no augmentation. Predictors are deterministic given inputs; key plumbing is reserved for tournament inits and evosax sampling.

## Evosax training

- **Separate top-level entry point** from Optax — no polishing field on `OptaxTrainingConfig`. User composes by calling them in sequence with the same trainable mask.
- **Targeted at small-parameter (kinetic) predictors** (~4–10 dims). Not optimised for NN-sized search in v1.
- **Single-eval JIT boundary**: `population_eval = eqx.filter_jit(jax.vmap(single_eval))`. `single_eval` closes over `static`, `dataset`, `simulate_fn`, `state_to_output`, `solver`, `loss_fn`; the bucket dispatch loop unrolls inside the trace.
- **Flatten contract**: `eqx.partition(predictor, trainable_mask) → (params, static); jax.flatten_util.ravel_pytree(params) → (flat, unflatten)`. `static` is closed over (callables can't pass through `vmap`).
- **Init modes**: `"warm"` (mean = current params, default — covers user-supplied kinetic init and midpoint-init via instantiating predictor at zero latent), `"uniform_box"` (latent ±extent uniform), `"lhs_box"` (Latin Hypercube via `scipy.qmc`, host-side).
- **Bounds during search**: not enforced. CMA_ES wanders latent space; `BoundedPredictor`'s sigmoid keeps physical outputs in range.
- **Best-ever tracked host-side** (`jnp.argmin(fitnesses)` per generation), not via strategy-specific best-member fields.
- **No graceful per-individual error handling** in v1: a diffrax/non-finite individual crashes the generation. Mitigation: conservative `sigma_init` and `init_box_extent`. Per-individual NaN sentinels deferred.

## JIT boundaries (training and prediction kept separate)

- `loss_and_grad(predictor, bucket_payload)` — **jitted, one trace per bucket shape**. Vmaps `simulate_fn` across the bucket; applies `state_to_output`; computes masked loss; returns `(loss, grads)`.
- `apply_update(predictor, accumulated_grads, opt_state)` — jitted; one shape (no bucket dependence).
- `predict_bucket(predictor, bucket_payload)` — jitted **separately** from training, one trace per bucket shape. No backward pass.
- The Python `for` loop over buckets is the dispatch driver, *not* part of the jitted region.

## Trainability filter

- A **trainable mask** is a PyTree of booleans matching the predictor's tree structure. Same shape used for both Optax (passed to `eqx.filter_value_and_grad(..., filter_spec=mask)`) and Evosax (passed to `eqx.partition(predictor, mask)` to derive the flat parameter vector).
- Default predicate: `eqx.is_inexact_array` — every float array is trainable; ints/bools/static fields stay frozen.
- **Freezers are free functions** that return a new mask: `freeze_paths`, `freeze_modules_of_type`, `freeze_where`. Replaces the source package's per-class `_build_filter_spec` case statement. Adding behaviour = adding a function, never a class.
- `BoundScaler.temperature` is a leaf but **frozen by convention** via `freeze_modules_of_type(mask, predictor, BoundScaler)` recommended in every example.
- No `predictor.set_trainable(...)` method (violates Equinox's no-overriding pattern and feels mutable).

## Out of scope (explicitly removed from the source package)

- Gaussian process regressors and the entire `bayes/` (variational inference) submodule.
- System embeddings (`EmbeddedMLP*`) and the `SystemConditionedRatePredictor` protocol.
- Padded-batched-experiments interface (`UnscaledBatchedExperiments`).
- Time-varying covariates and ODE-internal NN inputs (state-derived inputs to the predictor) — deferred.
- A `Model` wrapper class. The "model" is the loose triple `(predictor, simulate_fn, solver_config)`; serialisation handles each piece appropriately.
- **Temperature annealing** of any kind: `cosine_temperature_annealing`, `use_temp_annealing`, `initial_temperature`, `temp_cosine_fraction`, `temp_indices`. The bound-scaler's `temperature` is just a (typically frozen) parameter.
- The `_build_filter_spec` per-class registry from the source package — replaced by composable freezer functions.
- Bound-excursion penalty machinery (`bound_penalty_weight`, `_penalty()`, `call_with_penalty`) — never properly wired in source.

## Flagged ambiguities

- "Model" was used in the source package both for the trainable Equinox module and for the simulate-able physics object. Resolved: the trainable thing is a **Predictor**; the integrable physics is **simulate_fn**; nothing is called "Model".
- "Regressor" in the source was overloaded with bound-scaling. Resolved: scaling is decoupled from **Predictor** (interface for that branch is still being grilled).
