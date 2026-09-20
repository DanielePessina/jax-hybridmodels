# jax-hybridmodels architecture specification

**Status:** v1 build order (§8) complete and green; the package is in pre-1.0 refinement. Locked through interactive grilling on 2026-05-03; amended 2026-08-26 to reinstate bound penalties (§2.2, R-P1..R-P7); amended 2026-08-31 to reverse the collocation default for measured points plus user penalty-only points (R-P4), and to document the evosax history divergence (R-P6).

This spec is the architectural source of truth. It pairs with [`CONTEXT.md`](./CONTEXT.md), the domain glossary. Read CONTEXT.md first if any term here looks unfamiliar.

---

## 1. Purpose

A JAX/Equinox library for composing trainable function approximators (MLP, KAN, polynomial bases) with user-written ODE dynamics, and training the result against irregular time-series experimental data.

The package is a strongly refactored port of the existing `hybridcrystals` package, dropping its Bayesian/GP/embeddings machinery and consolidating its many regressor base classes into a flat composable design. Crystallisation kinetics is the *canonical example*, not the scope.

---

## 2. REQUIREMENTS

This section is the contract. Implementation is judged against these line by line.

### 2.1 Hard requirements (must hold in v1)

#### Architecture

- **R-A1**: There is no `Model` wrapper class. A "model" is the loose triple `(predictors, simulate_fn, solver_config)` where `predictors` is a `PyTree[eqx.Module]`.
- **R-A2**: `simulate_fn` is a pure user-written function with the mandatory signature in §4.2. Its first argument is `predictors: PyTree[eqx.Module]`, runtime-permissive (any pytree shape: tuple, list, dict, NamedTuple, single Module). The canonical convention shown in CONTEXT.md and `examples/crystallisation/train_kinetic.py` is a tuple, single-predictor case = `(BP,)`. The framework supplies vmap, jit, and gradient flow; the user supplies physics. The function is passed into training/prediction routines.
- **R-A3**: Composition over inheritance everywhere. Predictors follow Equinox's abstract/final pattern. No method overriding.
- **R-A4**: Bound-scaling is decoupled from `Predictor`. Implemented as composition (`BoundedPredictor` wraps a `Predictor` with input/output `BoundScaler`s).
- **R-A5**: Predictors must round-trip through `eqx.tree_serialise_leaves` ↔ `eqx.tree_deserialise_leaves` cleanly. The constraint extends to the full `predictors` pytree (eqx walks any pytree of leaves). This applies *now*, regardless of when the save/load helpers are implemented.
- **R-A6**: There is no framework wrapper for "a pair of rate predictors". The source package's `RatePair` is dropped; multi-rate models compose by unpacking the `predictors` tuple at the top of the user's vector field.

#### Data

- **R-D1**: Bucketed-irregular is the only data interface. No padded `UnscaledBatchedExperiments` pathway.
- **R-D2**: `Experiment` accepts per-channel sparse observations (`ChannelObs` with its own `ts/values/variance`). The framework computes the per-experiment union timestamp axis and the resulting mask automatically at `make_dataset` time. Users never write mask code.
- **R-D3**: Bucketing groups by `len(union_ts)`. Within a bucket, individual experiments may have different `ts` values and different masks (mask is a per-experiment array).
- **R-D4**: `BucketPayload` is not promoted to a class. It is a `NamedTuple` of stacked `[N, T, ...]` arrays.
- **R-D5**: Covariates are passed as `dict[str, float | Array]` (scalar or rank-1 vector per experiment, constant in time, named, with no canonical-order packed array). Stored as 0-d or 1-d JAX arrays after `make_experiment`. Each covariate key must have the same shape across a dataset.
- **R-D6**: `y0` is the full model state, constructed at data-import time via a user-supplied `y0_fn` hook and stored on `Experiment`.
- **R-D7**: `state_to_output` maps a full state trajectory to observed channels and is applied externally to `simulate_fn`'s output, before loss. It is a property of the *model*, passed to prediction and training as a keyword argument — not stored on the `Dataset`.
- **R-D8**: `split_dataset(dataset, *, train, val, test, key)` is provided.
- **R-D9**: Exogenous time-varying quantities enter through **profile factories** (`jaxhybridmodels.profiles`: `constant_profile`, `step_profile`, `ramp_profile`, `piecewise_linear_profile`). A profile is a pure-JAX callable `t -> Array` evaluated inside the user's vector field at the solver's continuous `t`; its *parameters* travel as ordinary R-D5 covariates, so the data layer, bucketing, and the mandatory `simulate_fn` signature are untouched. The two flat edges are exact: `ramp_profile(t0, t1, v0, v1)` returns `v0` before `t0` and `v1` after `t1`; `piecewise_linear_profile` extends the first/last values outward. Factories validate host-side parameters (`t1 > t0`, strictly increasing knots) and skip validation for traced values so they stay `jit`/`vmap`-safe.

#### Training (Optax)

- **R-T1**: Training step = full pass over all buckets → accumulate gradients → one `optimizer.update`. Bucket ≠ step.
- **R-T2**: The phase-keyed config fields (`steps, lr, optimizer, reset_optimiser_state, length_schedule`) are tuples of equal length (no scalar broadcast). `steps`, `lr`, `optimizer`, and `reset_optimiser_state` are required with no default; `length_schedule` defaults to `(1.0,)` for a single phase (scoring everything).
- **R-T3**: `length_schedule` (per-phase fraction in `(0, 1]`) is implemented as a runtime mask cutoff to avoid JIT recompile across phase boundaries.
- **R-T4**: `reset_optimiser_state` per phase rebuilds the optimiser at that phase boundary. It is a required per-phase tuple with no default: every phase states explicitly whether it resets, so a phase switch cannot silently keep a stale optimiser.
- **R-T5**: `bucket_step(predictors, bucket_payload, length_mask_fraction)` is `eqx.filter_jit`-compiled per bucket shape and returns `(loss, grads)`. It takes no `opt_state`: the optimiser update lives in a separate jitted `apply_update`, which is what the rest of this requirement already says. The bound penalty is *not* computed here; it is charged once per step by `build_penalty_step` (public in `jaxhybridmodels.training.kernels`), outside the bucket loop.
- **R-T6**: Shared tournament only. Implicitly enabled when `tournament_steps > 0 AND tournament_attempts > 1`. Reuses the main loop's compiled `bucket_step` and `apply_update`. Every surviving candidate is scored on the data term with a forward-only pass and the **lowest-scoring** one is returned; ties keep the earlier attempt, so the result is a deterministic function of `key`.
- **R-T7**: Tournament failure handling: on per-attempt failure (diffrax error, non-finite loss), drop and try a fresh RNG; if all fail, fall back to the original `predictors` pytree with a `RuntimeWarning`.
- **R-T8**: `Predictor.initialized_with_key(key)` is a documented per-leaf protocol used by the tournament; default free-function implementation is `reinitialize_with_key(predictor, key)` for one Module. Across the `predictors` pytree, the tournament splits the per-attempt key by traversal order (`jr.split(attempt_key, n_module_leaves)`) and applies `reinitialize_with_key` to each `eqx.Module` leaf independently. Identical-shape sibling predictors get *different* re-init weights.
- **R-T9**: Per-phase state resets. `patience` and the `restore_best` running minimum are both scoped to a horizon, not to the run. `patience` resets at every phase boundary, so a plateau at the end of one phase cannot stop the next before its new learning rate acts. `best_loss`/`best_predictors` reset at a boundary where `length_schedule` changes, because losses measured over a prefix and losses measured over the full window are not comparable and one running minimum across both lands in the shortest phase. The restored model therefore always comes from the final horizon.
- **R-T10**: `annealing_schedule` (in `jaxhybridmodels.schedules`) provides epoch-scaled schedule multipliers for custom loops: `schedule(step) -> float` in `[end_value, init_value]` with the run length baked in, built on optax schedule helpers, with kinds `"cosine"`, `"linear"`, `"warmup_cosine"`, `"exponential"`. The stock trainers do not take it (they express strategy changes as phases, R-T2); it composes as `lr = base_lr * schedule(step)`. Named `annealing_schedule` — never "temperature" — to stay distinct from chemistry and `BoundScaler.temperature`.

#### Training (Evosax)

- **R-E1**: Separate top-level entry point from Optax. No polishing field on `OptaxTrainingConfig`. Composition is by the user.
- **R-E2**: Targeted at small-parameter (kinetic) predictors (~4–10 dims). Not optimised for NN-sized search.
- **R-E3**: JIT boundary: `population_eval = eqx.filter_jit(jax.vmap(single_eval))`. `single_eval` closes over `static`, `dataset`, `simulate_fn`, `state_to_output`, `solver`, `loss_fn`. Bucket dispatch loop unrolls inside the trace.
- **R-E4**: Flatten contract via `eqx.partition` + `jax.flatten_util.ravel_pytree`. `static` is closed over (callables don't pass through `vmap`).
- **R-E5**: Init modes: `"warm"` (default), `"uniform_box"`, `"lhs_box"` (Latin Hypercube via `scipy.stats.qmc`, host-side).
- **R-E6**: Best-ever individual tracked host-side via `jnp.argmin(fitnesses)` per generation.

#### Bound penalties

- **R-P1**: Bounds stay enforced by reparameterisation; the penalty is an *additional* term, never the feasibility mechanism. A physical bound violation remains unrepresentable.
- **R-P2**: `BoundScaler.to_latent` guards `logit` with a linear continuation (`soft_logit`), not `jnp.clip`. Rationale: a hard clip has exactly zero derivative outside the box, and sitting mid-graph that zero propagates to every upstream parameter, silently dropping state-derived sensitivities from the ODE adjoint. Inside `[logit_eps, 1-logit_eps]` the map is exactly the previous one.
- **R-P3**: Penalties hinge on the latent, never the physical output. `from_latent`'s derivative underflows to exactly `0.0` past `|z/T| ~ 15`, so a physical-space penalty vanishes exactly where saturation is worst.
- **R-P4**: The default penalty is evaluated at *points*: the measured points (the input vectors the loss actually sees at observed cells, gathered by `data_penalty_points` and following the length-mask prefix per phase) plus any user-supplied penalty-only points (`penalty_points`, positional per `BoundedPredictor` leaf in traversal order, physical units, no measurements needed). `box_grid(in_scaler, n_per_dim)` is the collocation-as-extension recipe: a deterministic tensor-product sweep of the input box, uniform in *warped* coordinates so log warps cover decades evenly. It requires no change to `simulate_fn`, `BoundedPredictor.__call__`, `predict_bucket`, or the `loss(pred_obs, bp)` contract, and is invariant to pytree nesting. When the penalty is enabled, every leaf must have at least one point source (measured, extras, or both), else the run raises pointing at the trajectory penalty for embedded predictors.
- **R-P5**: Penalty weight is opt-in, default zero. In the Optax path it is passed to the jitted penalty kernel as a traced 0-d array (like `length_mask_fraction`) so changing it never retraces; the Evosax path folds a closed-over Python float into the fitness (never traced, so it stays out of the population vmap).
- **R-P6**: `losses_history`, `restore_best`, early stopping, and the tournament score track the data term alone — plus any configured trajectory penalty (R-P8), which is charged inside `bucket_step`'s forward pass and therefore rides in the per-step loss; the **bound** penalty is the one reported separately via `TrainingUI.on_step_end(penalty=...)`. Evosax is the documented exception: it ranks by one scalar with no aux channel, so its `losses_history` (best-so-far) folds the bound penalty in when configured — same type, same position, different meaning from the Optax series, and the docstring says so.
- **R-P7**: `penalty_weight` is deliberately not an R-T2 phase-keyed field. Length 1 broadcasts across phases; any other length must equal `len(steps)`. R-T2's enumerated fields all lack a safe default, which is why they are mandatory; `penalty_weight` has an unambiguous off state. Convention: the weight is relative to the per-bucket-averaged data term — the penalty is a mean over its points, charged once per step onto the averaged data gradient — so the same weight means the same thing whatever the dataset or point-count size.
- **R-P8**: Trajectory-aware penalty, opt-in via `trajectory_penalty_fn(full_state, bp)` + scalar `trajectory_penalty_weight` on both configs, charged in the training step's single forward pass. For embedded predictors the penalty rides in extra ODE state components (whose time-integral is charged; helpers `attach_penalty_state` / `penalty_vector_field` / `strip_penalty_state` / `penalty_integral`); for parallel predictors `trajectory_saturation_penalty` charges output saturation over time. Probe conditions — scenarios to steer toward with no measurements — are all-False-mask experiments (channels carry `values=jnp.array([])`; the `ts` still defines the grid).

#### Trainability filter

- **R-F1**: A boolean PyTree mask matching the `predictors` pytree structure is the canonical filter. Same shape consumed by both Optax (`eqx.filter_value_and_grad(..., filter_spec=mask)`) and Evosax (`eqx.partition(predictors, mask)`).
- **R-F2**: Default predicate: `eqx.is_inexact_array` (all float arrays trainable).
- **R-F3**: Freezers are free functions that return a new mask: `freeze_paths`, `freeze_modules_of_type`, `freeze_where`. No `Predictor.set_trainable(...)` method.
- **R-F4**: Adding new trainability behaviour = adding a function, never a class.

#### RNG

- **R-R1**: Root key must be supplied by the user. Framework raises if missing; never silently defaults `jr.PRNGKey(0)`.
- **R-R2**: Internal subkeys derived via named folds: `jr.fold_in(root, _id("name"))` where `_id` is a stable hash. Names in use: `"tournament"`, `"tournament_attempt_{i}"`, `"seed_ensemble_tournament"`, `"bootstrap_{s}"`, `"bootstrap_seeds_{s}"`, `"bootstrap_sample_tournament"`, `"evosax_init"`, `"evosax_box_init"`, `"evosax_ask_{gen}"`, `"evosax_tell_{gen}"`.
- **R-R3**: Bucket visit order is fixed, not shuffled. Source package's per-step shuffle is dropped.

#### UI

- **R-U1**: Callback-based, two protocols: `TrainingUI` (Optax) and `EvosaxUI`. Concrete shipped UIs: `SilentUI`, `RichTrainingUI`, `RichEvosaxUI`.
- **R-U2**: `verbose: bool = True` on each training config picks Rich vs Silent. Explicit `ui=...` parameter overrides.
- **R-U3**: Compile-phase progress is a first-class lifecycle event (`on_compile_start/_progress/_done`).
- **R-U4**: Single `rich.live.Live` per training run; panels swap as run progresses.

#### JIT boundaries

- **R-J1**: Training and prediction have separate jit boundaries. `loss_and_grad`, `apply_update`, and `predict_bucket` are independently `eqx.filter_jit`'d.
- **R-J2**: The Python `for bp in bucket_payloads:` is the dispatch driver, not part of the jitted region.
- **R-J3**: One trace per bucket shape per jitted entry point. Acceptable for ≤ 20 buckets; out-of-scope optimisation otherwise.

#### Loss

- **R-L1**: Loss signature is `loss(pred_obs: [N, T, D], bp: BucketPayload) → scalar`. Pure function; framework wraps with simulate + state_to_output + jit.
- **R-L2**: Built-ins: `masked_mse`, `masked_mle`, `bal_mse`, `bal_mle`. Accept `channel_idx` and `channel_weights` kwargs.

#### Solver

- **R-S1**: `SolverConfig` is an `eqx.Module` whose fields are all `eqx.field(static=True)`: `solver` (a `diffrax.AbstractSolver` instance), `rtol`, `atol` (scalar or per-state tuple), `max_steps`, `dt0`.
- **R-S2**: Solver round-trip via a `SOLVER_REGISTRY` mapping name → class. Public `register_solver(name, cls)` for user solvers.

### 2.2 Out of scope for v1 (explicit non-requirements)

- Gaussian process regressors; entire Bayesian / variational-inference (`bayes/`) submodule.
- System embeddings (`EmbeddedMLP*`, `SystemConditionedRatePredictor`).
- Padded batched-experiments data interface.
- Time-varying *covariates* at the data layer (`Experiment.covariates` stays constant in time, whether scalar or vector). Time-varying *values* are first-class via the profile factories (R-D9) evaluated inside the user's vector field. See CONTEXT.md "Time profiles" and "Predictor inputs".
- A `Model` wrapper class.
- Temperature annealing in the bound scaler (`cosine_temperature_annealing`, `use_temp_annealing`, `initial_temperature`, `temp_cosine_fraction`, `temp_indices`). `BoundScaler.temperature` is a (typically frozen) parameter. This is distinct from R-T10's `annealing_schedule`, which anneals training hyperparameters only.
- `_build_filter_spec` per-class registry; per-step bucket shuffling.
- Builder registry for serialisation (deferred until friction is real).
- Live loss plots, ETA columns, notebook-specific UI layouts.
- Sub-batching the population in evosax; sub-batching within a bucket.
- Per-individual graceful error handling in evosax (one bad individual crashes the generation).
- Dropout, stochastic data augmentation, scheduled-sampling.
- Stochastic predictors of any kind in v1.

### 2.3 Deferred (post-v1)

- Builder registry for serialisable predictor reconstruction without templates.
- Sub-batching across population / within-bucket for memory-bound workloads.
- Time-varying covariate arrays pre-evaluated on the experiment grid and interpolated by the solver.
- Builder/loader registries for `state_to_output` / `simulate_fn` to enable Dataset round-trip.

---

## 3. Project layout

```
jax-hybridmodels/                      (repo)
├── pyproject.toml                      (uv-managed; PEP 621; hatchling backend)
├── README.md
├── LICENSE
├── CONTEXT.md                          (domain glossary)
├── SPEC.md                             (this file)
├── docs/
│   ├── api/                            (generated API reference)
│   ├── guide/                          (concepts, predictors, penalties, training, ...)
│   └── examples/                       (per-example walkthroughs)
├── src/
│   └── jaxhybridmodels/
│       ├── __init__.py                 (lazy public API re-exports)
│       ├── data.py                     (Experiment, ChannelObs, Dataset, BucketPayload, make_dataset, split_dataset)
│       ├── transforms.py                (BOUND_TRANSFORMS, WARPS, register_bound_transform, register_warp)
│       ├── penalties.py                (soft_logit, softclip, clip_ste, box_violation, box_grid, data_penalty_points, bound_penalty)
│       ├── solver.py                   (SolverConfig, SOLVER_REGISTRY, register_solver)
│       ├── losses.py                   (masked_mse, masked_mle, bal_mse, bal_mle)
│       ├── metrics.py                  (compute_metrics, print_metrics, ChannelMetrics)
│       ├── trainable.py                (default_trainable, trainable_mask, freeze_paths/_modules_of_type/_where)
│       ├── rng.py                      (named-fold helper)
│       ├── prediction.py               (predict_bucket, predict_dataset, predict_dense, ensemble_predictions)
│       ├── profiles.py                 (constant/step/ramp/piecewise_linear time profiles, R-D9)
│       ├── schedules.py                (annealing_schedule, R-T10)
│       ├── serialise.py                (save/load, last-shipped)
│       ├── penalties.py                (bound_penalty, box_grid, data_penalty_points, penalty_points, trajectory-penalty helpers, R-P8)
│       ├── predictors/
│       │   ├── __init__.py
│       │   ├── base.py                 (Predictor, BoundScaler, BoundedPredictor, reinitialize_with_key, reinitialize_pytree_with_key)
│       │   ├── mlp.py                  (MLPPredictor)
│       │   ├── kan.py                  (KANPredictor, uses jaxkan)
│       │   └── neural_npoly.py         (NeuralNPolynomial)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── kernels.py              (build_bucket_step, build_score_bucket, build_penalty_step, build_apply_update)
│       │   ├── optax.py                (train_with_optax, OptaxTrainingConfig, _shared_tournament, ensembles)
│       │   └── evosax.py               (train_with_evosax, EvosaxTrainingConfig, _build_strategy)
│       └── ui/
│           ├── __init__.py
│           ├── base.py                 (TrainingUI, EvosaxUI protocols, SilentUI)
│           ├── optax.py                (RichTrainingUI)
│           └── evosax.py               (RichEvosaxUI)
├── tests/
│   ├── conftest.py
│   ├── test_data_buckets.py
│   ├── test_predictors_serialise.py    (R-A5 enforcement)
│   ├── test_trainable_filters.py
│   ├── test_solver.py
│   ├── test_train_optax.py             (synthetic ODE: harmonic oscillator)
│   ├── test_train_evosax.py
│   ├── test_loss_functions.py
│   └── test_ui_callbacks.py
└── examples/
    ├── crystallisation/
    │   ├── train_kinetic.py            (MLP/KAN direct-rate fit; end-to-end script, verification target)
    │   ├── train_crystallisation_mechanistic.py  (CNT/parametric fit via evosax)
    │   └── notebook.py                 (walkthrough, plain script since the marimo removal)
    └── pendulum/
        └── train_harmonic.py           (proves domain-agnostic; known optimum)
```

### 3.1 `pyproject.toml` (sketch)

```toml
[project]
name = "jaxhybridmodels"
requires-python = ">=3.11"
dependencies = [
    "jax>=0.4.30",
    "jaxlib>=0.4.30",
    "equinox>=0.11",
    "diffrax>=0.6",
    "optax>=0.2",
    "evosax>=0.1.6",
    "jaxkan",                  # KAN backend
    "jaxtyping>=0.2",
    "rich>=13",
    "numpy",
    "scipy",                   # for LHS init in evosax
]

[project.optional-dependencies]
examples = ["openpyxl", "matplotlib"]
dev = ["pytest", "pytest-cov", "ruff", "ty"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3.2 Toolchain

The project is uv-managed. All commands use uv:

| Action | Command |
|---|---|
| Add a dependency | `uv add <pkg>` |
| Add a dev dependency | `uv add --dev <pkg>` |
| Install the project (editable) | `uv sync` |
| Run a test file | `uv run pytest tests/test_data_buckets.py` |
| Run an example | `uv run python examples/crystallisation/train_kinetic.py` |
| Run a one-off Python | `uv run python -c '...'` |
| Lint | `uv run ruff check src tests` |
| Typecheck | `uv run ty check src` |

Do not invoke `pip`, `python`, or `pytest` directly without `uv run`. The lockfile and environment are uv-owned.

---

## 4. Public API

All public names are re-exported from `jaxhybridmodels` via lazy `__getattr__` in `src/jaxhybridmodels/__init__.py`.

### 4.1 Data

```python
from jaxhybridmodels import (
    ChannelObs,        # NamedTuple-ish, eqx.Module
    Experiment,
    Dataset,
    BucketPayload,     # NamedTuple
    make_dataset,
    split_dataset,
)
```

### 4.2 The `simulate_fn` contract

```python
from typing import Callable
from jax import Array
from jaxtyping import Float

def simulate_fn(
    predictors,                                         # PyTree[eqx.Module], convention: tuple of BoundedPredictor leaves
    ts: Float[Array, "T"],                              # observation times, this experiment
    covariates: dict[str, Array],                       # named, constant-in-time scalars or vectors
    y0: Float[Array, "S"],                              # full initial state
    solver,                                             # SolverConfig instance
) -> Float[Array, "T S"]:                                # full state at each ts
    ...
```

The first argument is a pytree of `eqx.Module` leaves, runtime-permissive (any pytree shape works for autodiff). The fixed convention shown in CONTEXT.md and `examples/crystallisation/train_kinetic.py` is a tuple, with the single-predictor case written `(BP,)`. Examples may also use `dict[str, BoundedPredictor]` or a user-defined NamedTuple subclass; the framework never inspects the container type, only the leaves.

Inside the function, the user typically unpacks the tuple (`growth, nucleation = predictors`) and constructs per-call predictor input dicts (covariates ∪ state-derived ∪ exogenous time-dependent values; see CONTEXT.md "Predictor inputs"). The framework imposes no other constraints. The user constructs a `diffrax.ODETerm` and calls `diffrax.diffeqsolve` with `solver.solver`, `solver.rtol`, etc. Returning shape `[T, S]` is mandatory.

### 4.3 Solver

```python
from jaxhybridmodels import SolverConfig, SOLVER_REGISTRY, register_solver

solver = SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-4,
    atol=(1e-5,) * 6,
    max_steps=500_000,
    dt0=None,
)
```

### 4.4 Predictors

```python
from jaxhybridmodels.predictors import (
    Predictor,             # abstract base
    BoundScaler,
    BoundedPredictor,
    MLPPredictor,
    KANPredictor,
    reinitialize_with_key,           # per-Module re-init
    reinitialize_pytree_with_key,    # per-leaf re-init across the predictors pytree (tournament uses this)
)
```

### 4.5 Trainability filter

```python
from jaxhybridmodels.trainable import (
    default_trainable,           # leaf -> bool predicate (eqx.is_inexact_array)
    trainable_mask,              # (predictors, predicate=default_trainable) -> PyTree[bool]
    freeze_paths,                # (mask, paths: tuple[str, ...]) -> mask  (dot-joined segments, e.g. "0.inner.mlp.layers.0.weight"; raises if a path matches nothing)
    freeze_modules_of_type,      # (mask, predictors, cls) -> mask
    freeze_where,                # (mask, predictors, fn) -> mask
)
```

### 4.6 Losses

```python
from jaxhybridmodels.losses import masked_mse, masked_mle, bal_mse, bal_mle
```

### 4.7 Training

```python
from jaxhybridmodels.training import (
    OptaxTrainingConfig,
    EvosaxTrainingConfig,
    train_with_optax,
    train_with_evosax,
)
```

Function signatures:

```python
def train_with_optax(
    predictors,                      # PyTree[eqx.Module], convention: tuple
    dataset,
    config: OptaxTrainingConfig,
    *,
    simulate_fn,
    state_to_output,                 # [T, S] -> [T, D]; model-side, not on the Dataset
    solver: SolverConfig,            # required
    trainable=None,                  # PyTree[bool] | None, same shape as predictors; None uses default_trainable
    key,                             # required
    ui=None,                         # TrainingUI | None, None picks Rich/Silent from config.verbose
) -> tuple[list[float], "PyTree[eqx.Module]"]:  # history is RAW per-step data loss
    ...

def train_with_evosax(
    predictors,                      # PyTree[eqx.Module]
    dataset,
    config: EvosaxTrainingConfig,
    *,
    simulate_fn,
    state_to_output,                 # [T, S] -> [T, D]; model-side, not on the Dataset
    solver: SolverConfig,            # required
    trainable=None,
    key,
    ui=None,
) -> tuple[list[float], "PyTree[eqx.Module]"]:  # history is BEST-SO-FAR, monotone
    ...
```

### 4.8 Prediction

```python
from jaxhybridmodels import predict_bucket, predict_dataset
```

### 4.9 UI

```python
from jaxhybridmodels.ui import TrainingUI, EvosaxUI, SilentUI, RichTrainingUI, RichEvosaxUI
```

### 4.10 Serialisation (last-shipped)

```python
from jaxhybridmodels import save_predictors, load_predictors, save_run, load_run
```

---

## 5. Per-module specification

### 5.1 `data.py`

**Types:**

```python
class ChannelObs(eqx.Module):
    ts: Float[Array, "Tc"]
    values: Float[Array, "Tc"]
    variance: Float[Array, "Tc"]                   # always rank-1 post-init

class Experiment(eqx.Module):
    covariates: dict[str, float | Array]            # rank-0 or rank-1; constant in time
    y0: Float[Array, "S"]
    channels: dict[str, ChannelObs]
    exp_id: str = eqx.field(static=True)

class BucketPayload(NamedTuple):
    ts: Float[Array, "N T"]
    y_observed: Float[Array, "N T D"]
    yvar: Float[Array, "N T D"]
    mask: Bool[Array, "N T D"]
    covariates: dict[str, Array]
    y0: Float[Array, "N S"]
    n_obs: Int[Array, ""]                          # total observed (mask sum), for weighted reductions

class Dataset(eqx.Module):
    bucket_payloads: tuple[BucketPayload, ...]
    output_channel_names: tuple[str, ...] = eqx.field(static=True)
    covariate_names: tuple[str, ...] = eqx.field(static=True)
    _experiments: tuple[Experiment, ...] = ()       # private; enables split_dataset re-bucketing
```

**Functions:**

```python
def make_experiment(
    *,
    covariates: dict[str, float | Array],
    channels: dict[str, ChannelObs],
    y0_fn: Callable[[dict, dict[str, ChannelObs]], Array],
    exp_id: str = "",
) -> Experiment: ...

def make_dataset(
    experiments: Sequence[Experiment],
    *,
    output_channel_names: tuple[str, ...],
) -> Dataset:
    """Builds union timestamps, computes masks, groups by len(union_ts), stacks per bucket."""

def split_dataset(
    dataset: Dataset,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    key: Array,
) -> tuple[Dataset, Dataset, Dataset]:
    """Shuffles experiments, partitions, re-buckets each partition independently."""
```

### 5.2 `predictors/`

**`base.py`:**

```python
class Predictor(eqx.Module):
    """Abstract marker. Concrete subclasses are final per Equinox pattern.
    __call__: Float[Array, 'in'] -> Float[Array, 'out']."""

class BoundScaler(eqx.Module):
    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    transform: str = eqx.field(static=True)        # "sigmoid" only in v1
    temperature: float = 1.0                        # leaf, typically frozen by convention
    def to_latent(self, x: Array) -> Array: ...
    def from_latent(self, z: Array) -> Array: ...

class BoundedPredictor(eqx.Module):
    input_keys: tuple[str, ...] = eqx.field(static=True)   # named-input order; len matches in_scaler.bounds (>= 1)
    in_scaler: BoundScaler
    inner: Predictor
    out_scaler: BoundScaler

    def __init__(self, *, in_scaler, inner, out_scaler, input_keys=None) -> None:
        # input_keys=None auto-fills ("x1", ..., f"x{N}") so the static field is
        # always populated; saved predictors are self-describing regardless.
        ...

    def __call__(self, inputs: dict[str, Array] | Array) -> Array: ...
    # dict path: subset extraction in input_keys order (extra keys ignored, missing key -> KeyError).
    # Array path: rank-1, length len(input_keys); validated via eqx.error_if and passed through.
    # Inputs may include state-derived / exogenous time-dependent values mixed in by
    # the user inside their vector field; provenance is intentionally invisible here.

def reinitialize_with_key(predictor, key) -> Predictor:
    """Per-Module: re-init every inexact-float leaf to a fresh value matching its shape."""

def reinitialize_pytree_with_key(predictors, key) -> "PyTree[eqx.Module]":
    """Per-leaf across the predictors pytree.

    Splits ``key`` by traversal order over ``eqx.Module`` leaves
    (``jr.split(key, n_module_leaves)``) and applies ``reinitialize_with_key``
    independently to each leaf. Identical-shape sibling predictors get
    different re-init weights. R-T8 contract.
    """
```

**Why `input_keys` lives on `BoundedPredictor` (not a separate selector).** An earlier draft of this section split the named-input ordering into a standalone `CovariateSelector` `eqx.Module` composed inside `BoundedPredictor`. That class held no trainable leaves and one operation (`jnp.stack([d[k] for k in keys])`), a class wearing one logical line. Folding the keys onto `BoundedPredictor` deletes the empty Module while keeping the property that motivated a static field at all: a saved predictor self-describes its input contract. `__call__` then becomes polymorphic, taking either a `dict[str, Array]` (subset extraction in declared order, extra keys allowed) or a rank-1 `Array` (positional, passed through). The user mixes covariates / state-derived / exogenous time-dependent values into the input at the vector-field boundary as before; the predictor treats every key as a named scalar, provenance-blind.

`mlp.py`: `MLPPredictor` wrapping `eqx.nn.MLP`. Static fields: `width`, `depth`, `activation_name`. Trainable: weights/biases.

`kan.py`: `KANPredictor` wrapping a `jaxkan` model. Static fields: grid size, layer widths, basis kind. Trainable: spline coefficients.

`neural_npoly.py`: `NeuralNPolynomial`, a per-channel scalar polynomial in the (latented) input whose coefficients come from an inner trainable network. Its basis is `sum(inner_input)`, so it cannot separate "coefficients from condition A" from "basis in condition B" in a single predictor (SPEC §2.3, latent-vs-physical basis open question).

### 5.3 `solver.py`

```python
SOLVER_REGISTRY: dict[str, type[diffrax.AbstractSolver]] = {
    "Tsit5": diffrax.Tsit5,
    "Kvaerno3": diffrax.Kvaerno3,
    "Dopri5": diffrax.Dopri5,
    "Heun": diffrax.Heun,
    # ... extended via register_solver
}

ADJOINT_REGISTRY: dict[str, type[diffrax.AbstractAdjoint]] = {
    "RecursiveCheckpoint": diffrax.RecursiveCheckpointAdjoint,
    "Direct": diffrax.DirectAdjoint,
    "Backsolve": diffrax.BacksolveAdjoint,
    "ForwardMode": diffrax.ForwardMode,
    # ... extended via register_adjoint
}

def register_solver(name: str, cls: type[diffrax.AbstractSolver]) -> None: ...
def register_adjoint(name: str, cls: type[diffrax.AbstractAdjoint]) -> None: ...

class SolverConfig(eqx.Module):
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float | tuple[float, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True, default_factory=diffrax.DirectAdjoint)
    pcoeff: float = eqx.field(static=True, default=0.0)   # PID controller gains
    icoeff: float = eqx.field(static=True, default=1.0)
    dcoeff: float = eqx.field(static=True, default=0.0)

    def stepsize_controller(self) -> diffrax.PIDController: ...
    def diffeqsolve(self, term, ts, y0, args=None) -> diffrax.Solution: ...
    def to_dict(self) -> dict: ...                  # {"solver": "Tsit5", "rtol": ..., ...}
    @classmethod
    def from_dict(cls, d: dict) -> "SolverConfig": ...
```

`adjoint` selects the backward pass; `None` passed to the constructor
coerces to `DirectAdjoint`, the memory-cheap default (diffrax's own default
is `RecursiveCheckpointAdjoint` — pick it explicitly if you want it).
`diffeqsolve` forwards it, and hand-written vector fields that build their
own `diffrax.diffeqsolve` call should pass `solver.adjoint` the same way
`examples/crystallisation/train_kinetic.py` does.

### 5.4 `losses.py`

```python
def masked_mse(pred_obs, bp, *, channel_idx=None, channel_weights=None) -> Array: ...
def masked_mle(pred_obs, bp, *, channel_idx=None, channel_weights=None) -> Array: ...
def bal_mse(pred_obs, bp, *, channel_idx=None, channel_weights=None) -> Array: ...
def bal_mle(pred_obs, bp, *, channel_idx=None, channel_weights=None) -> Array: ...

LOSS_REGISTRY: dict[str, Callable] = {"mse": masked_mse, "mle": masked_mle, "bal_mse": bal_mse, "bal_mle": bal_mle}
```

### 5.5 `trainable.py`

```python
def default_trainable(leaf) -> bool:
    return eqx.is_inexact_array(leaf)

def trainable_mask(predictors, predicate=default_trainable) -> PyTree[bool]: ...
def freeze_paths(mask, paths: tuple[str, ...]) -> PyTree[bool]: ...   # dot-joined segments; raises on a path that matches no leaf
def freeze_modules_of_type(mask, predictors, cls) -> PyTree[bool]: ...
def freeze_where(mask, predictors, fn: Callable[[eqx.Module], bool]) -> PyTree[bool]: ...
```

### 5.6 `rng.py`

```python
def fold(root_key: Array, name: str) -> Array:
    """Stable: fold(root, 'tournament') always returns the same key for the same root."""
```

### 5.7 `training/optax.py`

```python
@dataclass(frozen=True)
class OptaxTrainingConfig:
    steps: tuple[int, ...]
    lr: tuple[float, ...]
    optimizer: tuple[OptimizerSpec, ...]             # name | factory(lr) | optax.GradientTransformation per phase
    reset_optimiser_state: tuple[bool, ...]
    length_schedule: tuple[float, ...] = (1.0,)
    penalty_weight: tuple[float, ...] = (0.0,)        # length-1 broadcasts across phases
    penalty_points: tuple[Array, ...] | None = None   # per-leaf penalty-only points
    penalty_fn: Callable | None = None                # replaces the bound penalty
    trajectory_penalty_fn: Callable | None = None     # R-P8: (full_state, bp) -> scalar
    trajectory_penalty_weight: float = 0.0
    loss: Callable | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    tournament_attempts: int = 1                     # tournament implicit when > 1
    tournament_steps: int = 0                        # AND > 0
    tournament_lr: float = 1e-4
    patience: int = 0                                # 0 disables
    restore_best: bool = True
    verbose: bool = True

def train_with_optax(predictors, dataset, config, *, simulate_fn, state_to_output, solver, trainable=None, key, ui=None) -> tuple[list[float], "PyTree[eqx.Module]"]: ...
```

### 5.8 `training/evosax.py`

```python
@dataclass(frozen=True)
class EvosaxTrainingConfig:
    algorithm: str = "CMA_ES"                        # key of ALGORITHM_REGISTRY
    population_size: int = 64
    num_generations: int = 100
    init: Literal["warm", "uniform_box", "lhs_box"] = "warm"
    init_box_extent: float = 2.0
    sigma_init: float = 0.1
    penalty_weight: float = 0.0
    penalty_points: tuple[Array, ...] | None = None
    penalty_fn: Callable | None = None                # replaces the bound penalty
    trajectory_penalty_fn: Callable | None = None     # R-P8: (full_state, bp) -> scalar
    trajectory_penalty_weight: float = 0.0
    penalty_fn: Callable | None = None
    loss: Callable | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    log_every: int = 1
    verbose: bool = True

def train_with_evosax(predictors, dataset, config, *, simulate_fn, state_to_output, solver, trainable=None, key, ui=None) -> tuple[list[float], "PyTree[eqx.Module]"]: ...
```

### 5.9 `ui/`

`base.py` defines the two protocols and `SilentUI` (all no-ops). `optax.py` and `evosax.py` ship `RichTrainingUI` / `RichEvosaxUI` using a single `rich.live.Live`. Lifecycle events: see R-U3 and CONTEXT.md.

### 5.10 `prediction.py`

```python
@eqx.filter_jit
def predict_bucket(predictors, bp, *, simulate_fn, state_to_output, solver) -> Float[Array, "N T D"]: ...

def predict_dataset(
    predictors, dataset, *, simulate_fn, state_to_output, solver
) -> tuple[Float[Array, "N T D"], ...]:
    """One stacked array per bucket; user concatenates if they want a flat list."""
```

### 5.11 `serialise.py`

```python
def save_predictors(path: str | Path, predictors) -> None: ...
def load_predictors(path: str | Path, template) -> "PyTree[eqx.Module]": ...
def save_run(directory: str | Path, *, predictors, solver, optax_config=None, evosax_config=None, loss_history=None, extras: dict | None = None) -> None: ...
def load_run(directory: str | Path, *, predictors_template, optax_cls=None, evosax_cls=None) -> dict: ...
```

---

## 6. Test plan (TDD-first)

Tests are the executable spec. Each module gets a test file written before the implementation:

| Test file | What it pins |
|---|---|
| `test_data_buckets.py` | bucketing groups by `len(union_ts)`; mask True iff channel observed at that timestamp; `make_dataset` is idempotent; `split_dataset` produces non-overlapping splits with valid buckets in each |
| `test_predictors_serialise.py` | **R-A5 enforcement.** Every concrete predictor round-trips through `eqx.tree_serialise_leaves` ↔ `eqx.tree_deserialise_leaves` with bit-exact recovery |
| `test_trainable_filters.py` | default mask trains all float arrays; `freeze_modules_of_type(mask, predictors, BoundScaler)` zeros the right leaves; mask shape matches the `predictors` pytree structure (tuple/dict/Module) |
| `test_solver.py` | `SolverConfig.to_dict`/`from_dict` round-trip via `SOLVER_REGISTRY` |
| `test_loss_functions.py` | masked losses respect mask; channel_weights apply correctly; balanced variants normalise per-experiment |
| `test_train_optax.py` | trains a synthetic harmonic-oscillator ODE to known parameters within tolerance; multi-phase config with `reset_optimiser_state=(False, True)` doesn't crash; tournament reduces variance across seeds |
| `test_train_evosax.py` | trains a 4-D synthetic kinetic problem to known minimum; `init="lhs_box"` gives wider population spread than `"warm"` |
| `test_ui_callbacks.py` | `SilentUI` produces no stdout; `RichTrainingUI` calls each lifecycle event the right number of times (recorded via a spy) |
| `test_rng.py` | `fold(root, "name")` is stable across reorderings; missing root key raises |

Verification target (not unit, integration): one `examples/crystallisation/train_kinetic.py` script that loads your existing thesis Excel data, trains an MLP-rate predictor, and reports a final loss within tolerance of the source-package result.

---

## 7. Migration / verification map

The source package's verification artifacts live in `hybridcrystals/thesis_training/` and `hybridcrystals/tests/`. The new package needs:

| Source artifact | New location | Notes |
|---|---|---|
| `hybridcrystals/data/irregular.py::IrregularDataset/Batch/_prestack_buckets` | `src/jaxhybridmodels/data.py` | Restructured: per-channel sparse Experiment; `state_to_output` passed to prediction/training |
| `hybridcrystals/regressor_models.py::BoundedRegressor` | `src/jaxhybridmodels/predictors/base.py::BoundedPredictor` | Composition, no inheritance hierarchy |
| `hybridcrystals/regressor_models.py::RateRegressorPair` | (deleted) | Multi-rate models compose by unpacking the `predictors` tuple in user vector field. R-A6 |
| `hybridcrystals/regressors/mlp.py` | `src/jaxhybridmodels/predictors/mlp.py` | Strip embedding-related code |
| `hybridcrystals/regressors/kan.py` + `regressor_kanx.py` | `src/jaxhybridmodels/predictors/kan.py` | Use `jaxkan` |
| `hybridcrystals/regressors/polynomial.py::NeuralNPolynomialRegressor` | `src/jaxhybridmodels/predictors/neural_npoly.py` | Now public: exported top level, tested, documented. The latent-vs-physical basis question is open (SPEC §2.3) |
| `hybridcrystals/mechanistic.py::vector_ode + simulate_ode + ODESimulationOptions` | `examples/crystallisation/ode.py` + `src/jaxhybridmodels/solver.py::SolverConfig` | The `vector_ode` is example code, not framework |
| `hybridcrystals/losses.py::irregular_*_from_batch` | `src/jaxhybridmodels/losses.py` | Adapt to `(pred_obs, bp)` signature |
| `hybridcrystals/training/irregular.py` | `src/jaxhybridmodels/training/optax.py` | Drop tournament modes "vmapped"/"serial"/"shared" → keep only shared semantics |
| `hybridcrystals/training_evosax.py` | `src/jaxhybridmodels/training/evosax.py` | Drop polishing-from-optax-config; new init modes |
| `hybridcrystals/regressor_registry.py::_build_filter_spec` | `src/jaxhybridmodels/trainable.py` | Replaced by composable freezer functions |
| `hybridcrystals/regressor_constants.py::COVARIATE_BOUNDS / CANONICAL_INPUT_*` | (deleted) | No canonical input order; bounds live with each `BoundedPredictor` instance |
| `hybridcrystals/thesis_training/rich_ui.py` | `src/jaxhybridmodels/ui/optax.py` | Generalised, single Live + panels |
| `hybridcrystals/thesis_training/sharedgrowth.py` | `examples/crystallisation/train_kinetic.py` | Verification script |
| `hybridcrystals/bayes/*`, `gaussian_process.py`, `_gp_init.py` | (deleted) | Out of scope |
| `hybridcrystals/regressors/embedded_mlp.py` | (deleted) | Out of scope |
| `hybridcrystals/data.py::UnscaledBatchedExperiments` | (deleted) | Bucketed-irregular only |

---

## 8. Build order (TDD)

Implement in this order; each step ships green tests before the next begins.

1. Project skeleton + tooling: `pyproject.toml`, `uv sync`, empty modules, `tests/` skeleton.
2. `solver.py` + `SOLVER_REGISTRY`: minimal, easiest to test.
3. `predictors/base.py`: `Predictor`, `BoundScaler`, `BoundedPredictor`. `test_predictors_serialise.py` is the gate.
4. `predictors/mlp.py`: concrete predictor; serialisation test extended.
5. `data.py`: `Experiment`, `Dataset`, `make_dataset`, `split_dataset`, bucketing.
6. `losses.py` + `prediction.py`: pure functions, easy.
7. `trainable.py`: freezer composition.
8. `rng.py`: named folds.
9. `ui/base.py`: protocols + `SilentUI`.
10. `training/optax.py`: `bucket_step`, `apply_update`, phase loop, shared tournament. Use `SilentUI` for tests.
11. `ui/optax.py`: `RichTrainingUI`. Visual smoke-test in an example.
12. `training/evosax.py`: single-eval, population vmap, init modes.
13. `ui/evosax.py`: `RichEvosaxUI`.
14. `predictors/kan.py`: KAN via `jaxkan`. Serialisation test extended.
15. `examples/crystallisation/`: port one thesis script end-to-end. This is the verification gate. The supersaturation-polynomial form is implemented as user vector-field code here, alongside the now-public `NeuralNPolynomial` framework class (see §7).
16. `serialise.py`: `save_predictors` / `load_predictors` / `save_run` / `load_run`. Last shipped per R-A5.
17. `examples/pendulum/`: final deliverable proving domain-agnostic.
