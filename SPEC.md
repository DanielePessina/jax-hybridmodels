# jax-hybridmodels — Architecture Specification

**Status:** v1 design — *to be implemented*. Locked through interactive grilling on 2026-05-03.

This spec is the architectural source of truth. It pairs with [`CONTEXT.md`](./CONTEXT.md) (the domain glossary) and the ADRs in [`docs/adr/`](./docs/adr/). Read CONTEXT.md first if any term here looks unfamiliar.

---

## 1. Purpose

A JAX/Equinox library for composing **trainable function approximators** (MLP, KAN, polynomial bases) with **user-written ODE dynamics**, and training the result against **irregular time-series experimental data**.

The package is a **strongly refactored port** of the existing `hybridcrystals` package, dropping its Bayesian/GP/embeddings machinery and consolidating its many regressor base classes into a flat composable design. Crystallisation kinetics is the *canonical example*, not the scope.

---

## 2. REQUIREMENTS

This section is the contract. Implementation is judged against these line by line.

### 2.1 Hard requirements (must hold in v1)

#### Architecture

- **R-A1** — There is **no `Model` wrapper class**. A "model" is the loose triple `(predictor, simulate_fn, solver_config)`. See [ADR-0001](./docs/adr/0001-no-model-wrapper-class.md).
- **R-A2** — `simulate_fn` is a **pure user-written function** with the mandatory signature in §4.2. The framework supplies vmap, jit, and gradient flow; the user supplies physics. The function is passed into training/prediction routines.
- **R-A3** — Composition over inheritance everywhere. Predictors follow Equinox's abstract/final pattern. No method overriding.
- **R-A4** — Bound-scaling is **decoupled from `Predictor`**. Implemented as composition (`BoundedPredictor` wraps a `Predictor` with input/output `BoundScaler`s).
- **R-A5** — Predictors must round-trip through `eqx.tree_serialise_leaves` ↔ `eqx.tree_deserialise_leaves` cleanly. This constraint applies *now*, regardless of when the save/load helpers are implemented.

#### Data

- **R-D1** — **Bucketed-irregular is the only data interface.** No padded `UnscaledBatchedExperiments` pathway.
- **R-D2** — `Experiment` accepts **per-channel sparse observations** (`ChannelObs` with its own `ts/values/variance`). The framework computes the per-experiment union timestamp axis and the resulting mask **automatically** at `make_dataset` time. Users never write mask code.
- **R-D3** — Bucketing groups by `len(union_ts)`. Within a bucket, individual experiments may have different `ts` values and different masks (mask is a per-experiment array).
- **R-D4** — `BucketPayload` is **not promoted to a class** — a `NamedTuple` of stacked `[N, T, ...]` arrays.
- **R-D5** — Covariates are passed as `dict[str, Array]`, **constant in time**, named (no canonical-order packed array).
- **R-D6** — `y0` is the **full model state**, constructed at data-import time via a user-supplied `y0_fn` hook and stored on `Experiment`.
- **R-D7** — `state_to_output` lives on `Dataset`; applied externally to `simulate_fn`'s full-state output, before loss.
- **R-D8** — `split_dataset(dataset, *, train, val, test, key)` is provided.

#### Training (Optax)

- **R-T1** — Training **step** = full pass over all buckets → accumulate gradients → **one** `optimizer.update`. Bucket ≠ step.
- **R-T2** — All phase-keyed config fields are **required tuples of equal length** (no scalar broadcast). Fields: `steps, lr, optimizer, reset_optimiser_state, length_schedule`.
- **R-T3** — `length_schedule` (per-phase fraction in `(0, 1]`) is implemented as a **runtime mask cutoff** to avoid JIT recompile across phase boundaries.
- **R-T4** — `reset_optimiser_state` per phase rebuilds the optimiser at that phase boundary. Default `(False,) * n_phases`.
- **R-T5** — `make_step(predictor, opt_state, bucket_payload)` is `eqx.filter_jit`-compiled per bucket shape and returns `(loss, grads)`. Optimiser update is in a **separate** jitted `apply_update`.
- **R-T6** — **Shared tournament only** ([ADR-0002](./docs/adr/0002-shared-tournament-only.md)). Implicitly enabled when `tournament_steps > 0 AND tournament_attempts > 1`. Reuses the main loop's compiled `make_step` and `apply_update`.
- **R-T7** — Tournament failure handling: on per-attempt failure (diffrax error, non-finite loss), drop and try a fresh RNG; if all fail, fall back to the original predictor with a `RuntimeWarning`.
- **R-T8** — `Predictor.initialized_with_key(key)` is a documented protocol used by the tournament, with a default free-function implementation `reinitialize_with_key`.

#### Training (Evosax)

- **R-E1** — Separate top-level entry point from Optax. **No polishing field on `OptaxTrainingConfig`**. Composition is by the user.
- **R-E2** — Targeted at **small-parameter (kinetic) predictors** (~4–10 dims). Not optimised for NN-sized search.
- **R-E3** — JIT boundary: `population_eval = eqx.filter_jit(jax.vmap(single_eval))`. `single_eval` closes over `static`, `dataset`, `simulate_fn`, `state_to_output`, `solver`, `loss_fn`. Bucket dispatch loop unrolls inside the trace.
- **R-E4** — Flatten contract via `eqx.partition` + `jax.flatten_util.ravel_pytree`. `static` is closed over (callables don't pass through `vmap`).
- **R-E5** — Init modes: `"warm"` (default), `"uniform_box"`, `"lhs_box"` (Latin Hypercube via `scipy.stats.qmc`, host-side).
- **R-E6** — Best-ever individual tracked **host-side** via `jnp.argmin(fitnesses)` per generation.

#### Trainability filter

- **R-F1** — A **boolean PyTree mask** matching predictor structure is the canonical filter. Same shape consumed by both Optax (`eqx.filter_value_and_grad(..., filter_spec=mask)`) and Evosax (`eqx.partition(predictor, mask)`). See [ADR-0003](./docs/adr/0003-trainability-filter-as-pytree.md).
- **R-F2** — Default predicate: `eqx.is_inexact_array` (all float arrays trainable).
- **R-F3** — Freezers are **free functions** that return a new mask: `freeze_paths`, `freeze_modules_of_type`, `freeze_where`. No `Predictor.set_trainable(...)` method.
- **R-F4** — Adding new trainability behaviour = adding a function, never a class.

#### RNG

- **R-R1** — Root key **must be supplied by the user**. Framework raises if missing; never silently defaults `jr.PRNGKey(0)`.
- **R-R2** — Internal subkeys derived via **named folds**: `jr.fold_in(root, _id("name"))` where `_id` is a stable hash. Names: `"init"`, `"tournament"`, `"phase_{i}"`, `"evosax_init"`, `"evosax_ask_{gen}"`.
- **R-R3** — Bucket visit order is **fixed**, not shuffled. Source package's per-step shuffle is dropped.

#### UI

- **R-U1** — Callback-based, two protocols: `TrainingUI` (Optax) and `EvosaxUI`. Concrete shipped UIs: `SilentUI`, `RichTrainingUI`, `RichEvosaxUI`.
- **R-U2** — `verbose: bool = True` on each training config picks Rich vs Silent. Explicit `ui=...` parameter overrides.
- **R-U3** — Compile-phase progress is a first-class lifecycle event (`on_compile_start/_progress/_done`).
- **R-U4** — Single `rich.live.Live` per training run; panels swap as run progresses.

#### JIT boundaries

- **R-J1** — Training and prediction have **separate** jit boundaries. `loss_and_grad`, `apply_update`, and `predict_bucket` are independently `eqx.filter_jit`'d.
- **R-J2** — The Python `for bp in bucket_payloads:` is the dispatch driver, **not** part of the jitted region.
- **R-J3** — One trace per bucket shape per jitted entry point. Acceptable for ≤ 20 buckets; out-of-scope optimisation otherwise.

#### Loss

- **R-L1** — Loss signature is `loss(pred_obs: [N, T, D], bp: BucketPayload) → scalar`. Pure function; framework wraps with simulate + state_to_output + jit.
- **R-L2** — Built-ins: `masked_mse`, `masked_mle`, `bal_mse`, `bal_mle`. Accept `channel_idx` and `channel_weights` kwargs.

#### Solver

- **R-S1** — `SolverConfig` is an `eqx.Module` whose fields are all `eqx.field(static=True)`: `solver` (a `diffrax.AbstractSolver` instance), `rtol`, `atol` (scalar or per-state tuple), `max_steps`, `dt0`.
- **R-S2** — Solver round-trip via a **`SOLVER_REGISTRY`** mapping name → class. Public `register_solver(name, cls)` for user solvers.

### 2.2 Out of scope for v1 (explicit non-requirements)

- Gaussian process regressors; entire Bayesian / variational-inference (`bayes/`) submodule.
- System embeddings (`EmbeddedMLP*`, `SystemConditionedRatePredictor`).
- Padded batched-experiments data interface.
- Time-varying covariates; ODE-internal NN inputs (state-derived inputs to the predictor).
- A `Model` wrapper class.
- Temperature annealing of any kind.
- `_build_filter_spec` per-class registry; bound-excursion penalty machinery; per-step bucket shuffling.
- Builder registry for serialisation (deferred until friction is real).
- Live loss plots, ETA columns, notebook-specific UI layouts.
- Sub-batching the population in evosax; sub-batching within a bucket.
- Per-individual graceful error handling in evosax (one bad individual crashes the generation).
- Dropout, stochastic data augmentation, scheduled-sampling.
- Stochastic predictors of any kind in v1.

### 2.3 Deferred (post-v1)

- Builder registry for serialisable predictor reconstruction without templates.
- Sub-batching across population / within-bucket for memory-bound workloads.
- Time-varying covariate hooks.
- `tanh` bound-scaling option.
- Non-crystallisation example (pendulum) — **last deliverable** of v1, gating "domain-agnostic" claim.
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
├── AGENTS.md                           (orientation for any AI/agent collaborator)
├── CLAUDE.md                           (Claude Code-specific entry point)
├── docs/
│   └── adr/
│       ├── 0001-no-model-wrapper-class.md
│       ├── 0002-shared-tournament-only.md
│       ├── 0003-trainability-filter-as-pytree.md
│       ├── 0004-bucketed-irregular-only.md
│       └── 0005-simulate-fn-mandatory-signature.md
├── src/
│   └── hybridmodels/
│       ├── __init__.py                 (lazy public API re-exports)
│       ├── data.py                     (Experiment, ChannelObs, Dataset, BucketPayload, make_dataset, split_dataset)
│       ├── solver.py                   (SolverConfig, SOLVER_REGISTRY, register_solver)
│       ├── losses.py                   (masked_mse, masked_mle, bal_mse, bal_mle)
│       ├── trainable.py                (default_trainable, trainable_mask, freeze_paths/_modules_of_type/_where)
│       ├── rng.py                      (named-fold helper)
│       ├── prediction.py               (predict_bucket, predict_dataset)
│       ├── serialise.py                (save/load — last-shipped)
│       ├── predictors/
│       │   ├── __init__.py
│       │   ├── base.py                 (Predictor, CovariateSelector, BoundScaler, BoundedPredictor, RatePair, reinitialize_with_key)
│       │   ├── mlp.py                  (MLPPredictor)
│       │   ├── kan.py                  (KANPredictor — uses jaxkan)
│       │   └── neural_npoly.py         (NeuralNPolynomial)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── optax.py                (train_with_optax, OptaxTrainingConfig, _shared_tournament)
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
    │   ├── loader_excel.py             (depends on openpyxl, isolated)
    │   ├── ode.py                      (the simulate_fn — moments + concentration)
    │   ├── kinetic_predictor.py        (CNT nucleation + power-law growth as BoundedPredictors)
    │   ├── mlp_predictor.py            (MLP-based RatePair)
    │   ├── train_optax.py              (end-to-end script — verification target)
    │   └── train_evosax_kinetic.py
    └── pendulum/                        (last deliverable — proves domain-agnostic)
        └── train.py
```

### 3.1 `pyproject.toml` (sketch)

```toml
[project]
name = "hybridmodels"
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
dev = ["pytest", "pytest-cov", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3.2 Toolchain

The project is **uv-managed**. All commands use uv:

| Action | Command |
|---|---|
| Add a dependency | `uv add <pkg>` |
| Add a dev dependency | `uv add --dev <pkg>` |
| Install the project (editable) | `uv sync` |
| Run a test file | `uv run pytest tests/test_data_buckets.py` |
| Run an example | `uv run python examples/crystallisation/train_optax.py` |
| Run a one-off Python | `uv run python -c '...'` |
| Lint | `uv run ruff check src tests` |
| Typecheck | `uv run mypy src` |

**Do not** invoke `pip`, `python`, or `pytest` directly without `uv run` — the lockfile and environment are uv-owned.

---

## 4. Public API surface

All public names are re-exported from `hybridmodels` via lazy `__getattr__` in `src/hybridmodels/__init__.py`.

### 4.1 Data

```python
from hybridmodels import (
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
    predictor,                                         # trainable eqx.Module (BoundedPredictor / RatePair / ...)
    ts: Float[Array, "T"],                              # observation times, this experiment
    covariates: dict[str, Array],                       # named, constant-in-time scalars
    y0: Float[Array, "S"],                              # full initial state
    solver,                                             # SolverConfig instance
) -> Float[Array, "T S"]:                                # full state at each ts
    ...
```

The framework imposes **no** other constraints inside this function. The user calls `predictor(...)` to evaluate the trainable component, constructs a `diffrax.ODETerm`, and calls `diffrax.diffeqsolve` with `solver.solver`, `solver.rtol`, etc. Returning shape `[T, S]` is mandatory.

### 4.3 Solver

```python
from hybridmodels import SolverConfig, SOLVER_REGISTRY, register_solver

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
from hybridmodels.predictors import (
    Predictor,             # abstract base
    CovariateSelector,
    BoundScaler,
    BoundedPredictor,
    RatePair,
    MLPPredictor,
    KANPredictor,
    NeuralNPolynomial,
    reinitialize_with_key,
)
```

### 4.5 Trainability filter

```python
from hybridmodels.trainable import (
    default_trainable,           # leaf -> bool predicate (eqx.is_inexact_array)
    trainable_mask,              # (predictor, predicate=default_trainable) -> PyTree[bool]
    freeze_paths,                # (mask, paths: tuple[str, ...]) -> mask
    freeze_modules_of_type,      # (mask, predictor, cls) -> mask
    freeze_where,                # (mask, predictor, fn) -> mask
)
```

### 4.6 Losses

```python
from hybridmodels.losses import masked_mse, masked_mle, bal_mse, bal_mle
```

### 4.7 Training

```python
from hybridmodels.training import (
    OptaxTrainingConfig,
    EvosaxTrainingConfig,
    train_with_optax,
    train_with_evosax,
)
```

Function signatures:

```python
def train_with_optax(
    predictor,
    dataset,
    config: OptaxTrainingConfig,
    *,
    simulate_fn,
    trainable=None,                  # PyTree[bool] | None — None uses default_trainable
    key,                             # required
    ui=None,                         # TrainingUI | None — None picks Rich/Silent from config.verbose
) -> tuple[list[float], Predictor]:
    ...

def train_with_evosax(
    predictor,
    dataset,
    config: EvosaxTrainingConfig,
    *,
    simulate_fn,
    trainable=None,
    key,
    ui=None,
) -> tuple[list[float], Predictor]:
    ...
```

### 4.8 Prediction

```python
from hybridmodels import predict_bucket, predict_dataset
```

### 4.9 UI

```python
from hybridmodels.ui import TrainingUI, EvosaxUI, SilentUI, RichTrainingUI, RichEvosaxUI
```

### 4.10 Serialisation (last-shipped)

```python
from hybridmodels import save_predictor, load_predictor, save_run, load_run
```

---

## 5. Per-module specification

### 5.1 `data.py`

**Types:**

```python
class ChannelObs(eqx.Module):
    ts: Float[Array, "Tc"]
    values: Float[Array, "Tc"]
    variance: Float[Array, "Tc"] | float = 1.0     # scalar broadcasts

class Experiment(eqx.Module):
    covariates: dict[str, float]
    y0: Float[Array, "S"]
    channels: dict[str, ChannelObs]
    exp_id: str = eqx.field(static=True)

class BucketPayload(NamedTuple):
    ts: Float[Array, "N T"]
    y_observed: Float[Array, "N T D"]
    yvar: Float[Array, "N T D"]
    mask: Bool[Array, "N T D"]
    covariates: dict[str, Float[Array, "N"]]
    y0: Float[Array, "N S"]
    n_obs: Int[Array, ""]                          # total observed (mask sum), for weighted reductions

class Dataset(eqx.Module):
    bucket_payloads: tuple[BucketPayload, ...]
    state_to_output: Callable = eqx.field(static=True)
    output_channel_names: tuple[str, ...] = eqx.field(static=True)
    covariate_names: tuple[str, ...] = eqx.field(static=True)
    _experiments: tuple[Experiment, ...] = ()       # private; enables split_dataset re-bucketing
```

**Functions:**

```python
def make_experiment(
    *,
    covariates: dict[str, float],
    channels: dict[str, ChannelObs],
    y0_fn: Callable[[dict, dict[str, ChannelObs]], Array],
    exp_id: str = "",
) -> Experiment: ...

def make_dataset(
    experiments: Sequence[Experiment],
    *,
    state_to_output: Callable,
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

class CovariateSelector(eqx.Module):
    keys: tuple[str, ...] = eqx.field(static=True)
    def __call__(self, covariates: dict[str, Array]) -> Array: ...

class BoundScaler(eqx.Module):
    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    transform: str = eqx.field(static=True)        # "sigmoid" only in v1
    temperature: float = 1.0                        # leaf — typically frozen by convention
    def to_latent(self, x: Array) -> Array: ...
    def from_latent(self, z: Array) -> Array: ...

class BoundedPredictor(eqx.Module):
    selector: CovariateSelector
    in_scaler: BoundScaler
    inner: Predictor
    out_scaler: BoundScaler
    def __call__(self, covariates: dict[str, Array]) -> Array: ...

class RatePair(eqx.Module):
    nucleation: BoundedPredictor
    growth: BoundedPredictor
    def __call__(self, covariates) -> Array: ...    # stacked [J, G]

def reinitialize_with_key(predictor, key) -> Predictor:
    """Default: re-init every inexact-float leaf to a fresh value matching its shape."""
```

**`mlp.py`** — `MLPPredictor` wrapping `eqx.nn.MLP`. Static fields: `width`, `depth`, `activation_name`. Trainable: weights/biases.

**`kan.py`** — `KANPredictor` wrapping a `jaxkan` model. Static fields: grid size, layer widths, basis kind. Trainable: spline coefficients.

**`neural_npoly.py`** — `NeuralNPolynomial(coeff_net: Predictor, exponents: tuple[float, ...])`. Composition only.

### 5.3 `solver.py`

```python
SOLVER_REGISTRY: dict[str, type[diffrax.AbstractSolver]] = {
    "Tsit5": diffrax.Tsit5,
    "Kvaerno3": diffrax.Kvaerno3,
    "Dopri5": diffrax.Dopri5,
    "Heun": diffrax.Heun,
    # ... extended via register_solver
}

def register_solver(name: str, cls: type[diffrax.AbstractSolver]) -> None: ...

class SolverConfig(eqx.Module):
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float | tuple[float, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def to_dict(self) -> dict: ...                  # {"solver": "Tsit5", "rtol": ..., ...}
    @classmethod
    def from_dict(cls, d: dict) -> "SolverConfig": ...
```

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

def trainable_mask(predictor, predicate=default_trainable) -> PyTree[bool]: ...
def freeze_paths(mask, paths: tuple[str, ...]) -> PyTree[bool]: ...
def freeze_modules_of_type(mask, predictor, cls) -> PyTree[bool]: ...
def freeze_where(mask, predictor, fn: Callable[[eqx.Module], bool]) -> PyTree[bool]: ...
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
    optimizer: tuple[str, ...]                       # "adamw" | "adabelief" per phase
    reset_optimiser_state: tuple[bool, ...]
    length_schedule: tuple[float, ...] = (1.0,)
    loss: Callable | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    tournament_attempts: int = 1                     # tournament implicit when > 1
    tournament_steps: int = 0                        # AND > 0
    tournament_lr: float = 1e-4
    log_every: int = 10
    patience: int = 0                                # 0 disables
    restore_best: bool = True
    verbose: bool = True

def train_with_optax(predictor, dataset, config, *, simulate_fn, trainable=None, key, ui=None) -> tuple[list[float], Predictor]: ...
```

### 5.8 `training/evosax.py`

```python
@dataclass(frozen=True)
class EvosaxTrainingConfig:
    algorithm: str = "CMA_ES"
    population_size: int = 64
    num_generations: int = 100
    init: Literal["warm", "uniform_box", "lhs_box"] = "warm"
    init_box_extent: float = 2.0
    sigma_init: float = 0.1
    loss: Callable | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    log_every: int = 1
    verbose: bool = True

def train_with_evosax(predictor, dataset, config, *, simulate_fn, trainable=None, key, ui=None) -> tuple[list[float], Predictor]: ...
```

### 5.9 `ui/`

`base.py` defines the two protocols and `SilentUI` (all no-ops). `optax.py` and `evosax.py` ship `RichTrainingUI` / `RichEvosaxUI` using a single `rich.live.Live`. Lifecycle events: see R-U3 and CONTEXT.md.

### 5.10 `prediction.py`

```python
@eqx.filter_jit
def predict_bucket(predictor, bp, *, simulate_fn, state_to_output, solver) -> Float[Array, "N T D"]: ...

def predict_dataset(predictor, dataset, *, simulate_fn, solver) -> tuple[Float[Array, "N T D"], ...]:
    """One stacked array per bucket; user concatenates if they want a flat list."""
```

### 5.11 `serialise.py`

```python
def save_predictor(path: str | Path, predictor: Predictor) -> None: ...
def load_predictor(path: str | Path, template: Predictor) -> Predictor: ...
def save_run(directory: str | Path, *, predictor, solver, optax_config=None, evosax_config=None, loss_history=None, extras: dict | None = None) -> None: ...
def load_run(directory: str | Path, *, predictor_template, optax_cls=None, evosax_cls=None) -> dict: ...
```

---

## 6. Test plan (TDD-first)

Tests are the executable spec. Each module gets a test file written **before** the implementation:

| Test file | What it pins |
|---|---|
| `test_data_buckets.py` | bucketing groups by `len(union_ts)`; mask True iff channel observed at that timestamp; `make_dataset` is idempotent; `split_dataset` produces non-overlapping splits with valid buckets in each |
| `test_predictors_serialise.py` | **R-A5 enforcement.** Every concrete predictor round-trips through `eqx.tree_serialise_leaves` ↔ `eqx.tree_deserialise_leaves` with bit-exact recovery |
| `test_trainable_filters.py` | default mask trains all float arrays; `freeze_modules_of_type(mask, predictor, BoundScaler)` zeros the right leaves; mask shape matches predictor structure |
| `test_solver.py` | `SolverConfig.to_dict`/`from_dict` round-trip via `SOLVER_REGISTRY` |
| `test_loss_functions.py` | masked losses respect mask; channel_weights apply correctly; balanced variants normalise per-experiment |
| `test_train_optax.py` | trains a synthetic harmonic-oscillator ODE to known parameters within tolerance; multi-phase config with `reset_optimiser_state=(False, True)` doesn't crash; tournament reduces variance across seeds |
| `test_train_evosax.py` | trains a 4-D synthetic kinetic problem to known minimum; `init="lhs_box"` gives wider population spread than `"warm"` |
| `test_ui_callbacks.py` | `SilentUI` produces no stdout; `RichTrainingUI` calls each lifecycle event the right number of times (recorded via a spy) |
| `test_rng.py` | `fold(root, "name")` is stable across reorderings; missing root key raises |

Verification target (not unit, integration): one `examples/crystallisation/train_optax.py` script that loads your existing thesis Excel data, trains an MLP-rate predictor, and reports a final loss within tolerance of the source-package result.

---

## 7. Migration / verification map

The source package's verification artifacts live in `hybridcrystals/thesis_training/` and `hybridcrystals/tests/`. The new package needs:

| Source artifact | New location | Notes |
|---|---|---|
| `hybridcrystals/data/irregular.py::IrregularDataset/Batch/_prestack_buckets` | `src/hybridmodels/data.py` | Restructured: per-channel sparse Experiment, Dataset owns `state_to_output` |
| `hybridcrystals/regressor_models.py::BoundedRegressor` | `src/hybridmodels/predictors/base.py::BoundedPredictor` | Composition, no inheritance hierarchy |
| `hybridcrystals/regressor_models.py::RateRegressorPair` | `src/hybridmodels/predictors/base.py::RatePair` | Pure composition |
| `hybridcrystals/regressors/mlp.py` | `src/hybridmodels/predictors/mlp.py` | Strip embedding-related code |
| `hybridcrystals/regressors/kan.py` + `regressor_kanx.py` | `src/hybridmodels/predictors/kan.py` | Use `jaxkan` |
| `hybridcrystals/regressors/polynomial.py::NeuralNPolynomialRegressor` | `src/hybridmodels/predictors/neural_npoly.py` | Composition `(coeff_net, exponents)` |
| `hybridcrystals/mechanistic.py::vector_ode + simulate_ode + ODESimulationOptions` | `examples/crystallisation/ode.py` + `src/hybridmodels/solver.py::SolverConfig` | The `vector_ode` is **example code**, not framework |
| `hybridcrystals/losses.py::irregular_*_from_batch` | `src/hybridmodels/losses.py` | Adapt to `(pred_obs, bp)` signature |
| `hybridcrystals/training/irregular.py` | `src/hybridmodels/training/optax.py` | Drop tournament modes "vmapped"/"serial"/"shared" → keep only shared semantics |
| `hybridcrystals/training_evosax.py` | `src/hybridmodels/training/evosax.py` | Drop polishing-from-optax-config; new init modes |
| `hybridcrystals/regressor_registry.py::_build_filter_spec` | `src/hybridmodels/trainable.py` | Replaced by composable freezer functions |
| `hybridcrystals/regressor_constants.py::COVARIATE_BOUNDS / CANONICAL_INPUT_*` | (deleted) | No canonical input order; bounds live with each `BoundedPredictor` instance |
| `hybridcrystals/thesis_training/rich_ui.py` | `src/hybridmodels/ui/optax.py` | Generalised, single Live + panels |
| `hybridcrystals/thesis_training/sharedgrowth.py` | `examples/crystallisation/train_optax.py` | Verification script |
| `hybridcrystals/bayes/*`, `gaussian_process.py`, `_gp_init.py` | (deleted) | Out of scope |
| `hybridcrystals/regressors/embedded_mlp.py` | (deleted) | Out of scope |
| `hybridcrystals/data.py::UnscaledBatchedExperiments` | (deleted) | Bucketed-irregular only |

---

## 8. Build order (TDD)

Implement in this order; each step ships green tests before the next begins.

1. **Project skeleton + tooling** — `pyproject.toml`, `uv sync`, empty modules, `tests/` skeleton.
2. **`solver.py` + `SOLVER_REGISTRY`** — minimal, easiest to test.
3. **`predictors/base.py`** — `Predictor`, `BoundScaler`, `CovariateSelector`, `BoundedPredictor`. **`test_predictors_serialise.py` is the gate.**
4. **`predictors/mlp.py`** — concrete predictor; serialisation test extended.
5. **`data.py`** — `Experiment`, `Dataset`, `make_dataset`, `split_dataset`, bucketing.
6. **`losses.py` + `prediction.py`** — pure functions, easy.
7. **`trainable.py`** — freezer composition.
8. **`rng.py`** — named folds.
9. **`ui/base.py`** — protocols + `SilentUI`.
10. **`training/optax.py`** — `make_step`, `apply_update`, phase loop, shared tournament. Use `SilentUI` for tests.
11. **`ui/optax.py`** — `RichTrainingUI`. Visual smoke-test in an example.
12. **`training/evosax.py`** — single-eval, population vmap, init modes.
13. **`ui/evosax.py`** — `RichEvosaxUI`.
14. **`predictors/kan.py`** — KAN via `jaxkan`. Serialisation test extended.
15. **`predictors/neural_npoly.py`** — composition example.
16. **`examples/crystallisation/`** — port one thesis script end-to-end. **This is the verification gate.**
17. **`serialise.py`** — `save_predictor` / `load_predictor` / `save_run` / `load_run`. Last shipped per R-A5.
18. **`examples/pendulum/`** — final deliverable proving domain-agnostic.
