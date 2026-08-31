# jax-hybridmodels

A JAX/Equinox library for hybrid models: trainable function approximators (MLP, KAN, ...) composed with user-written ODE dynamics and trained on irregular time-series experiments. Crystallisation kinetics is the canonical example, not the scope.

## Language

**Predictor**:
A narrow trainable `eqx.Module` whose `__call__` is `Array → Array`. Knows nothing about covariate names, bounds, or experiments. Concrete examples in v1: `MLPPredictor`, `KANPredictor`, `NeuralNPolynomial` (a per-channel polynomial in `sum(x)` whose coefficients come from an inner network).
_Avoid_: Regressor (the previous package's overloaded term, which bundled bound-scaling, rate-pair semantics, and trainable weights together).

**BoundScaler**:
An `eqx.Module` mapping a physical box onto an unbounded latent. Two independent axes, both name-keyed registries in `transforms.py`, both stored as static strings so the scaler still round-trips through `eqx.tree_serialise_leaves`.

**warp** decides what "halfway between the bounds" means. `"linear"` (default), `"log"`, `"log10"`. A rate constant bounded by `(1e-6, 1e2)` spans eight decades; a linear warp puts its midpoint at 50 and collapses the whole low end into a sliver. `log10` puts the midpoint at `1e-2`. Bounds stay in physical units whichever warp is chosen. Log warps reject non-positive bounds at construction.

**transform** decides how the squash saturates, and therefore how fast gradient dies once a predictor pushes against a bound. `"sigmoid"` (default), `"algebraic"`, `"softsign"`. Measured float32 gradient death: sigmoid at `z=16.8`, algebraic at `z~3e3` (180x the runway), softsign at `z~1.1e7`. `algebraic` is the recommended alternative and is C-infinity; `softsign` buys more runway but is not C^2 and its kink sits at the box midpoint, which solver steps straddle routinely.

`tanh` is deliberately absent. `(1+tanh z)/2` is exactly `sigmoid(2z)`, so it would be the sigmoid entry at half temperature.

`to_latent` guards the squash inverse against its poles with a **linear continuation** (`soft_inverse`), not a hard clip. The clip it replaced had exactly zero derivative outside the box; sitting mid-graph, that zero propagated to every upstream parameter, so a state-derived input straying out of range silently dropped a real sensitivity from the ODE adjoint. Inside `[logit_eps, 1-logit_eps]` the map is exactly the old one, value and derivative both.

`z_knee` is derived from the transform, not shared. It encodes a physical criterion, the outer 5% of the box; sigmoid's 2.944 means 12.5% from the bound on softsign, so a shared constant would charge 2.7x too aggressively.

Two pure penalty queries hang off it: `input_violation` (how far a physical input fell outside its box) and `saturation` (how hard the output squash is pinned). Neither is called by `__call__`; emitting a penalty is a query, never a side effect.

Bounds must be finite and ordered `low < high`. Infinite bounds used to construct silently and then fail three ways: every finite input mapped to one latent, `from_latent` returned `inf`, and `input_violation` returned `nan`.

**BoundedPredictor**:
A composition wrapper: `input_keys (named-input order) → in_scaler → inner Predictor → out_scaler`. Returns a single `Array` (the physical-units output). `input_keys` is a static `tuple[str, ...]` that names each input slot in declared order; cardinality must match `in_scaler.bounds` and be ≥ 1. Auto-fills to `("x1", ..., "xN")` when omitted, so the saved predictor is always self-describing. `__call__` accepts either `dict[str, Array]` (subset extraction in `input_keys` order, extra keys allowed, missing keys raise) or rank-1 `Array` (passed through). Penalties are opt-in and default to zero. See "Bound penalty" below.

**simulate_fn**:
A pure user-written function with a mandatory signature that, for one experiment, integrates the dynamics and returns the full state trajectory. The framework owns vmapping, jitting, and gradient flow; the user owns physics. Inside the user's vector field, multi-rate models compose their predictors directly (`growth, nucleation = predictors`). There is no framework wrapper for "the pair of rate predictors"; the source package's `RatePair` is dropped.
_Avoid_: "ODE function" (ambiguous, since it could mean the vector field), "solve" (overlap with diffrax).

Mandatory signature:
```python
def simulate_fn(
    predictors,                       # PyTree[eqx.Module], convention: tuple of BoundedPredictor leaves
    ts: Float[Array, "T"],            # observation times for this experiment
    covariates: dict[str, Array],     # named, constant-in-time scalars or vectors
    y0: Float[Array, "S"],            # full initial state
    solver: SolverConfig,             # static; diffrax bits
) -> Float[Array, "T S"]:              # full state at each ts
    ...
```

The first argument is a pytree of `eqx.Module` leaves. It is runtime-permissive (any pytree works for autodiff: tuple, list, dict, NamedTuple, custom Module), with a fixed convention: always wrap in a tuple, single-predictor case = `(BP,)`. This gives examples a uniform shape and lets `eqx.partition` / `eqx.filter_value_and_grad` / `eqx.tree_serialise_leaves` walk the leaves uniformly. Dict and NamedTuple are valid alternatives demonstrated in secondary examples; the framework never inspects the container type.

**state_to_output**:
A pure callable mapping a full-state trajectory `[T, S]` to the observed output channels `[T, D]` (e.g. crystallisation: `[mu0..mu4, conc] → [conc, d43]`). Applied externally to `simulate_fn`'s output, before loss computation. It belongs to the **model setup**, not the data: the Dataset never sees raw states, and which states exist depends on the model. `predict_bucket`/`predict_dataset` and the training loops take it as a parameter; the Dataset carries only data (experiments, channels, masks, the union timestamp axis, y0).

**y0_fn**:
A user-supplied hook invoked at data-import time to construct the per-experiment full initial state from the raw row + covariates. Default for "state == observed": `lambda row, c: row.y[0]`. Crystallisation-style: `lambda row, c: jnp.array([0,0,0,0,0, row.y[0,0]])`.

**SolverConfig**:
A frozen `eqx.Module` whose fields are all `eqx.field(static=True)`: diffrax solver instance, rtol, atol (scalar or per-state tuple), max_steps, dt0. Static so it doesn't enter the pytree leaves. JSON-serialisable via a small solver class-name registry.

**Covariates**:
Named scalar or rank-1 vector conditions that are constant in time for an experiment (e.g. `temperature_C`, `loading`, `c_sat`, or a feed-composition vector). Stored on `Experiment.covariates` and passed into `simulate_fn` unchanged. Always passed as `dict[str, Array]`, with no canonical-order array. A given key must have one shape across a dataset. Time-varying *values* come from the profile factories (see "Time profiles" below).

**Time profiles**:
Factory callables in `hybridmodels.profiles` (`constant_profile`, `step_profile`, `ramp_profile`, `piecewise_linear_profile`) that return a pure-JAX function `t -> Array` for a quantity changing over time. The profile *parameters* (set points, jump/ramp times) travel as ordinary scalar covariates; the callable is evaluated inside the user's vector field at the solver's continuous `t`:

```python
def vector_field(t, y, args):
    T_C = hm.ramp_profile(t0=cov["ramp_t0"], t1=cov["ramp_t1"],
                          v0=cov["T_lo"], v1=cov["T_hi"])(t)
    inputs = {"temperature_C": T_C, "pH": cov["pH"]}
    ...
```

`ramp_profile` has two exact flat edges (`v0` before `t0`, `v1` after `t1`); `piecewise_linear_profile` extends its first/last values outward. Factories validate host-side parameters and skip validation for traced values, so they are `jit`/`vmap`-safe. This is the shipped answer to the older "time-varying covariate hooks" deferral: no data-layer change, no `simulate_fn` signature change (R-D9).

**Predictor inputs**:
The dict that the user's vector field actually feeds into a `BoundedPredictor` at call time. A *superset* of `covariates`. The user constructs it inside the vector field by mixing constant covariates with time-varying values:
- state-derived values (`y[CONC_IDX]`, supersaturation `S = c / c_sat`, etc.),
- exogenous time-dependent values from a profile factory (e.g. `hm.ramp_profile(...)(t)` — see "Time profiles").

```python
def vector_field(t, y, args):
    predictors, covariates = args
    inputs = {
        "temperature_C": covariates["temperature_C"],     # constant covariate
        "supersaturation": y[CONC_IDX] / covariates["c_sat"],  # state-derived, time-varying
    }
    G = predictors[0](inputs)
```

Key collision is intentional: the dict can override a covariate's name with a time-varying value (e.g. `T(t)`). The `BoundedPredictor.input_keys`, the `BoundScaler.bounds`, and the dict mixing are *unaware* of provenance. Every key is treated as a named scalar regardless of whether it came from `covariates`, `y`, or `t`. This collapses "state-derived inputs to the predictor", "exogenous time-dependent inputs", and "covariate inputs" into one mechanism.

**Bucket**:
A group of experiments sharing the same `len(union_ts)`. Within a bucket, individual experiments may have different `ts` values and different masks (mask is a per-experiment array). Not promoted to a class; `BucketPayload` is just a `NamedTuple` of stacked `[N, T, ...]` arrays produced once by `make_dataset`.

**Experiment**:
A user-built record holding per-channel sparse observations. Each channel has its own `(ts_channel, values, variance)` triple, allowing arbitrary sparsity per channel. The framework computes the per-experiment union timestamp axis and the resulting mask automatically at `make_dataset` time; users never write mask code.

**ChannelObs**:
Per-channel observation triple: `(ts: [Tc], values: [Tc], variance: [Tc] | float)`. A scalar variance broadcasts over `Tc`.

**step** (training):
One full pass over all buckets → accumulate gradients across buckets → one `optimizer.update`. Equivalently, one epoch. *Bucket ≠ step.*

**phase** (training):
A contiguous block of N steps sharing the same hyperparameters (lr, optimizer, …). A run is a tuple of phases.

**bucket dispatch loop**:
The Python `for bp in bucket_payloads:` that drives JIT-cached per-bucket kernels. Idiomatic; not a performance smell.

## Relationships

- A simulate_fn consumes a predictors pytree (typically a tuple of `BoundedPredictor`s), a SolverConfig, and one Experiment's `(ts, covariates, y0)`; returns a full state trajectory.
- state_to_output is composed externally: `loss(state_to_output(simulate_fn(predictors, ...)), y_observed, mask)`.
- A Bucket holds N Experiments with identical `len(ts)`; the training loop vmaps simulate_fn over the bucket.
- The predictors pytree is the *only* component that is binary-serialised (eqx.tree_serialise_leaves walks any pytree of leaves). simulate_fn and state_to_output are code (re-imported); SolverConfig is JSON.
- state_to_output lives with the model triple, not the Dataset: prediction and training receive it as a parameter; the Dataset is pure data.

## Optax training

- Phases are tuples, and all phase-keyed fields are required tuples of equal length (no scalar broadcast). Fields: `steps, lr, optimizer, reset_optimiser_state, length_schedule`.
- `optimizer` is a per-phase tuple whose entries may be a name string (`"adamw"`, `"adabelief"`), a factory taking a `learning_rate` keyword and returning an `optax.GradientTransformation` (so chains, clipping, schedules compose), or a ready-made transformation instance. Names and factories are wrapped in `optax.inject_hyperparams`, so a phase boundary can move the learning rate without a rebuild; a raw instance is returned as-is, so a phase that changes `lr` on one must also set `reset_optimiser_state` (the loop refuses otherwise).
- `length_schedule` is a per-phase fraction in `(0, 1]`, applied as a runtime mask cutoff, so there is no JIT recompile across phase boundaries (default `(1.0,)` for one phase = no scheduling).
- A phase that changes `length_schedule` changes what the loss measures, so `restore_best`'s running minimum resets there. Without that, the minimum lands in the shortest-horizon phase and the returned model is the least-trained one in the run. `patience` resets at every phase boundary for the same kind of reason.
- `reset_optimiser_state` is per-phase (tuple of bool); a True entry rebuilds the optimiser at that phase boundary (used when switching optimiser type or when length-schedule changes invalidate momentum).
- `bucket_step(predictors, bucket_payload, length_mask_fraction)` is jitted and returns `(loss, grads)`. Optimiser update happens in a separate jitted `apply_update(predictors, accumulated_grads, opt_state)` *after* the Python for-loop over buckets has accumulated.
- Shared tournament only. Enabled implicitly when `tournament_steps > 0 AND tournament_attempts > 1`. Serial candidate evaluation that *reuses the main loop's compiled `bucket_step` and `apply_update`* (no extra JIT compile cost). Each candidate is a fresh re-init trained for `tournament_steps` at `tournament_lr`, then scored with a forward-only pass on the data term — including any configured trajectory penalty, which the phases also charge — and *excluding* the bound penalty, so a candidate cannot win by drifting where the bound penalty likes rather than by fitting. On per-attempt failure (diffrax error / non-finite loss), drop and try a fresh RNG; if all fail, fall back to the original predictors with a `RuntimeWarning`.
- `Predictor.initialized_with_key(key)` is a documented per-leaf protocol used by the tournament. Default implementation is a free function `reinitialize_with_key(predictor, key)` that re-inits inexact-float leaves only. Across the predictors pytree, the tournament splits the attempt key by traversal order (`jr.split(attempt_key, n_module_leaves)`) and applies `reinitialize_with_key` to each `eqx.Module` leaf independently, so identical-shape sibling predictors get *different* re-init weights.

## Loss interface

- A loss is a pure function `loss(pred_obs: [N, T, D], bp: BucketPayload) → scalar`.
- The framework wraps it: `loss_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(_simulate_then_loss))`, where `_simulate_then_loss` does the vmap of `simulate_fn`, applies `state_to_output`, and calls the user's loss.
- Built-ins: `masked_mse`, `masked_mle`, `bal_mse`, `bal_mle`. Built-ins accept optional `channel_idx` and `channel_weights` kwargs (carried over from the source package; may be reworked later).
- User-supplied losses just match the signature.

## UI

- Callback-based rather than log-based. Training functions accept `ui: TrainingUI` (or `EvosaxUI`); two protocols, no merged supertype.
- Default selection: `verbose: bool = True` on each training config picks `RichTrainingUI` / `RichEvosaxUI`; `False` picks `SilentUI`. An explicit `ui=...` parameter on the training function overrides the config flag.
- Lifecycle events include compile-phase progress (`on_compile_start/_progress/_done`) so per-bucket-shape compile time is visible rather than silent.
- Concrete shipped UIs: `SilentUI`, `RichTrainingUI`, `RichEvosaxUI`. Single `rich.live.Live` per training run; panels swap as run progresses.
- Out of scope for v1: live loss plots, ETA columns, per-bucket per-step breakdowns, notebook-specific layouts.

## Serialisation

- Hard design constraint (applies *now* to every Predictor we design): every `Predictor` / `BoundedPredictor` must round-trip through `eqx.tree_serialise_leaves` ↔ `eqx.tree_deserialise_leaves` with zero JAX/Equinox conflicts. The constraint extends naturally from a single Predictor to the full `predictors` pytree, because `eqx.tree_serialise_leaves` walks any pytree of leaves. Concretely: all dynamic leaves are JAX arrays; all static fields are JSON-encodable primitives or tuples thereof; no closures in non-static positions.
- Implementation is the last shipped feature. The save/load helpers (`save_predictor`, `load_predictor`, optional `save_run`/`load_run`) are written after the rest of the framework is verified.
- Save format (when implemented): a directory containing `predictors.eqx` + `metadata.json` (timestamp, version, container shape hint, simulate_fn module hint, user `extras`).
- Not serialised by the framework: `simulate_fn` and `state_to_output` (pure functions, re-imported), `Dataset`, the trainable mask (rebuild from user code), training configs (user owns).
- No builder registry in v1. Loading requires a user-supplied `template` predictors pytree (same container shape and Module types), per `eqx.tree_deserialise_leaves`. Builder registry is deferred until friction proves real.

## RNG discipline

- Root key is user-supplied. Framework never silently defaults `jr.PRNGKey(0)`; missing key raises.
- Named folds. Internal subkeys derived via `jr.fold_in(root, _id("name"))` where `_id` is a stable hash of the consumer name. Avoids the chained-split fragility where reordering or inserting a consumer shifts every downstream key. Names: `"tournament"`, `"tournament_attempt_{i}"`, `"seed_ensemble_tournament"`, `"bootstrap_{s}"`, `"bootstrap_seeds_{s}"`, `"bootstrap_sample_tournament"`, `"evosax_init"`, `"evosax_box_init"`, `"evosax_ask_{gen}"`, `"evosax_tell_{gen}"`.
- Bucket visit order is fixed rather than shuffled. Full-batch gradient accumulation is commutative; the source package's per-step shuffle is dropped.
- No stochasticity in v1: no dropout, no augmentation. Predictors are deterministic given inputs; key plumbing is reserved for tournament inits and evosax sampling.

## Evosax training

- Separate top-level entry point from Optax, with no polishing field on `OptaxTrainingConfig`. User composes by calling them in sequence with the same trainable mask.
- Targeted at small-parameter (kinetic) predictors (~4–10 dims). Not optimised for NN-sized search in v1.
- Single-eval JIT boundary: `population_eval = eqx.filter_jit(jax.vmap(single_eval))`. `single_eval` closes over `static`, `dataset`, `simulate_fn`, `state_to_output`, `solver`, `loss_fn`; the bucket dispatch loop unrolls inside the trace.
- Flatten contract: `eqx.partition(predictors, trainable_mask) → (params, static); jax.flatten_util.ravel_pytree(params) → (flat, unflatten)`. The pytree shape of `predictors` is irrelevant here, since `eqx.partition` walks leaves uniformly. `static` is closed over (callables can't pass through `vmap`).
- Init modes: `"warm"` (mean = current params, the default, covering user-supplied kinetic init and midpoint-init via instantiating predictor at zero latent), `"uniform_box"` (latent ±extent uniform), `"lhs_box"` (Latin Hypercube via `scipy.qmc`, host-side).
- Bounds during search: not enforced. CMA_ES wanders latent space; `BoundedPredictor`'s sigmoid keeps physical outputs in range.
- Best-ever is tracked host-side (`jnp.argmin(fitnesses)` per generation), not via strategy-specific best-member fields.
- No graceful per-individual error handling in v1: a diffrax/non-finite individual crashes the generation. Mitigation: conservative `sigma_init` and `init_box_extent`. Per-individual NaN sentinels deferred.

## JIT boundaries (training and prediction kept separate)

- `loss_and_grad(predictors, bucket_payload)`: jitted, one trace per bucket shape. Vmaps `simulate_fn` across the bucket; applies `state_to_output`; computes masked loss; returns `(loss, grads)`.
- `apply_update(predictors, accumulated_grads, opt_state)`: jitted; one shape (no bucket dependence).
- `predict_bucket(predictors, bucket_payload)`: jitted separately from training, one trace per bucket shape. No backward pass.
- The Python `for` loop over buckets is the dispatch driver, *not* part of the jitted region.

## Trainability filter

- A trainable mask is a PyTree of booleans matching the `predictors` pytree structure (any container: tuple, dict, NamedTuple, single Module). Same shape used for both Optax (passed to `eqx.filter_value_and_grad(..., filter_spec=mask)`) and Evosax (passed to `eqx.partition(predictors, mask)` to derive the flat parameter vector).
- Default predicate: `eqx.is_inexact_array`, so every float array is trainable and ints/bools/static fields stay frozen.
- Freezers are free functions that return a new mask: `freeze_paths`, `freeze_modules_of_type`, `freeze_where`. Replaces the source package's per-class `_build_filter_spec` case statement. Adding behaviour = adding a function, never a class.
- `BoundScaler.temperature` is a leaf but frozen by convention via `freeze_modules_of_type(mask, predictor, BoundScaler)` recommended in every example.
- No `predictor.set_trainable(...)` method (violates Equinox's no-overriding pattern and feels mutable).

## Out of scope (explicitly removed from the source package)

- Gaussian process regressors and the entire `bayes/` (variational inference) submodule.
- System embeddings (`EmbeddedMLP*`) and the `SystemConditionedRatePredictor` protocol.
- Padded-batched-experiments interface (`UnscaledBatchedExperiments`).
- Time-varying *covariates* at the data layer. `Experiment.covariates` stays constant in time, whether scalar or vector. Time-varying *values* are first-class via the `hybridmodels.profiles` factories evaluated in the vector field (see "Time profiles" above); state-derived and exogenous time-dependent *predictor inputs* mix into the per-call dict inside the vector field.
- A `Model` wrapper class. The "model" is the loose triple `(predictors, simulate_fn, solver_config)`; serialisation handles each piece appropriately.
- Temperature annealing of any kind in the *bound scaler*: `cosine_temperature_annealing`, `use_temp_annealing`, `initial_temperature`, `temp_cosine_fraction`, `temp_indices`. The bound-scaler's `temperature` is just a (typically frozen) parameter. This is distinct from `annealing_schedule`, which anneals a training hyperparameter (lr/weight) and lives in `hybridmodels.schedules`.
- The `_build_filter_spec` per-class registry from the source package, replaced by composable freezer functions.
- ~~Bound-excursion penalty machinery~~ is reinstated, properly wired this time. The source package's version was dropped because it was never plumbed through to the loss, not because the idea was wrong. See "Bound penalty".
- `RatePair` framework class (deleted in this round of design). Multi-rate models compose by unpacking the predictors tuple at the top of the user's vector field.
- ~~`NeuralNPolynomial` framework class~~ is now public. The supersaturation-polynomial form (`sum_i c_i(T) * (S - 1)^p_i`) is also implemented directly in user vector-field code in `examples/crystallisation/train_kinetic.py`, where the `coeffs` come from a `BoundedPredictor`.

**annealing schedule**:
An epoch-scaled multiplier for a custom training loop, from `hybridmodels.schedules` (`annealing_schedule`): a callable `(step: int) -> float` in `[end_value, init_value]`, built on optax's own schedule helpers, with the run length baked in as `total_epochs` (one step == one epoch). Composes as `lr = base_lr * schedule(step)` or `weight = w0 * schedule(step)`. The name is deliberately not "temperature": that word collides with chemistry and with `BoundScaler.temperature`; this schedule only ever anneals a *training* hyperparameter.

**Bound penalty**:
A scalar added to the training objective that charges a `BoundedPredictor` for saturating its output squash. Evaluated at *points*: the measured points (the input vectors the loss actually sees at observed cells, gathered by `data_penalty_points` and following the length-mask prefix) plus any user-supplied penalty-only points (`OptaxTrainingConfig.penalty_points`, positional per leaf, no measurements needed). `box_grid` is the collocation-as-extension recipe: a deterministic warp-uniform sweep of a predictor's input box. Weighted by `OptaxTrainingConfig.penalty_weight` (per-phase tuple, length-1 broadcasts) or `EvosaxTrainingConfig.penalty_weight` (scalar); both are relative to the per-bucket-averaged data term, mean-aggregated and charged once per step, so the same weight means the same thing whatever the dataset size.

Three properties make this the default rather than an aux-threading scheme:

- No signature changes. `simulate_fn`, `BoundedPredictor.__call__`, `predict_bucket`, and the `loss(pred_obs, bp)` contract are all untouched.
- Nesting-invariant. It walks the pytree by leaf, the same `is_leaf`-stopped traversal `reinitialize_pytree_with_key` and `freeze_modules_of_type` use, so arbitrary nesting works for free (ADR-0006).
- Call-site blind. Equally correct whether the predictor runs inside a vector field or is hoisted above one.

The penalty hinges on the latent, never the physical output. `from_latent`'s derivative carries a `sigma'` factor that underflows to exactly `0.0` past `|z/T| ~ 15`, so a penalty written against the physical value dies precisely where saturation is worst.

A leaf with no point set when the penalty is enabled is a hard error (`validate_penalty_points`), pointing at the trajectory penalty for embedded predictors: a silent no-penalty is the failure mode this penalty exists to prevent.
_Avoid_: "bound violation penalty". With sigmoid reparameterisation a physical violation is unrepresentable; what is being charged is saturation.

**Trajectory penalty**:
A scalar charged on `trajectory_penalty_fn(full_state, bp)` (configs' opt-in hook; scalar weight), added to the data loss inside the training step's single forward pass. `full_state` is `[N, T, S]` *before* `state_to_output`, so it can carry extra ODE components. For an **embedded** hybrid model (predictor inside the vector field) the penalty rides in the state as accumulators whose derivative is the per-call penalty rate — `saturation(z)` and `input_violation(x)` kept as two separate components — and the charge is their time-integral (`penalty_integral`). Recipe helpers: `attach_penalty_state`, `penalty_vector_field`, `strip_penalty_state`. For a **parallel** hybrid model (predictor output is the measured channel) `trajectory_saturation_penalty` inverts outputs to latents and charges saturation over time, no state change. Probe conditions — scenarios to steer toward with no measurements — are experiments whose channels carry `values=jnp.array([])`: the `ts` still defines the integration grid, the mask is all-False, and the data loss is exactly zero. See [ADR-0009](./docs/adr/0009-trajectory-aware-penalties.md).

**Data loss vs. objective**:
`losses_history`, `restore_best`, early stopping, and the tournament score all track the data term alone — plus any configured trajectory penalty, which rides in the per-step loss. The **bound** penalty is the one excluded: it is reported separately via `TrainingUI.on_step_end(penalty=...)`, and the combined objective (`data + weight * penalty`) is what the optimiser descends but never what "best" is measured on. Reporting the bound penalty too would let "best" move when only the penalty weight ramped, and would make runs with different weights incomparable. (Evosax is the exception: it ranks by one scalar with no aux channel, so its history folds the bound penalty in when configured.)

## Flagged ambiguities

- "Model" was used in the source package both for the trainable Equinox module and for the simulate-able physics object. Resolved: the trainable thing is a predictors pytree (typically a tuple of `BoundedPredictor`s); the integrable physics is simulate_fn; nothing is called "Model".
- "Regressor" in the source was overloaded with bound-scaling. Resolved: scaling is decoupled from Predictor via `BoundedPredictor` composition (`input_keys → in_scaler → inner → out_scaler`).
- "Covariate" vs "predictor input" was historically conflated. Resolved: a covariate is a constant-in-time per-experiment scalar or rank-1 vector at the data layer (`Experiment.covariates`); a predictor input is the per-call dict (or pre-stacked Array) the vector field hands to a predictor (covariates ∪ state-derived ∪ exogenous-time-dependent). `BoundedPredictor` named inputs remain one scalar per key; vector covariates can be indexed or passed through its array interface.
- `CovariateSelector` was originally a separate `eqx.Module` composed inside `BoundedPredictor` to translate a named-covariates dict into a stacked Array. It held no trainable leaves and one operation, so it was folded into `BoundedPredictor` as the `input_keys` static field. The saved predictor still self-describes its input contract; `__call__` is now polymorphic (dict or Array) so users can construct inputs in either named or positional form at the vector-field boundary.
