# Training: Optax & Evosax Loops

Choose [`train_with_optax`](#train_with_optax) for gradient-based fitting or [`train_with_evosax`](#train_with_evosax) for population search over small parameter sets. Both take the same model pieces, return `(loss_history, trained_predictors)`, and require an explicit keyword-only `key`.

## Quick links

- [`OptaxTrainingConfig`](#optaxtrainingconfig)
- [`train_with_optax`](#train_with_optax)
- [`train_seed_ensemble`](#train_seed_ensemble)
- [`train_bootstrap_ensemble`](#train_bootstrap_ensemble)
- [`EvosaxTrainingConfig`](#evosaxtrainingconfig)
- [`train_with_evosax`](#train_with_evosax)
- [`register_algorithm`](#register_algorithm)

---

<a id="optaxtrainingconfig"></a>

### `OptaxTrainingConfig`

<small>`from hybridmodels.training import OptaxTrainingConfig` &nbsp;·&nbsp; also re-exported as `hybridmodels.OptaxTrainingConfig`</small>

```python
OptaxTrainingConfig(
    steps: 'tuple[int, ...]',
    lr: 'tuple[float, ...]',
    optimizer: 'tuple[OptimizerSpec, ...]',
    reset_optimiser_state: 'tuple[bool, ...]',
    length_schedule: 'tuple[float, ...]' = (1.0,),
    penalty_weight: 'tuple[float, ...]' = (0.0,),
    penalty_points: 'tuple[Array, ...] | None' = None,
    penalty_fn: 'Callable[..., Array] | None' = None,
    trajectory_penalty_fn: 'Callable[..., Array] | None' = None,
    trajectory_penalty_weight: 'float' = 0.0,
    loss: 'Callable[..., Array] | str' = 'mse',
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
    tournament_attempts: 'int' = 1,
    tournament_steps: 'int' = 0,
    tournament_lr: 'float' = 0.0001,
    patience: 'int' = 0,
    restore_best: 'bool' = True,
    verbose: 'bool' = True,
) -> None
```

Configuration for :func:`train_with_optax`.

The first five fields are **phase-keyed**: ``steps``, ``lr``,
``optimizer``, ``reset_optimiser_state`` and ``length_schedule`` carry
one entry per phase, must have the same length, and do not broadcast.
None has a defensible default, so a single-phase run spells out
one-element tuples::

    OptaxTrainingConfig(
        steps=(500,), lr=(1e-3,), optimizer=("adamw",),
        reset_optimiser_state=(False,),
    )

``penalty_weight`` is the exception. It has an unambiguous off state,
so a length-1 tuple broadcasts across every phase.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `steps` | `tuple[int, ...]` | Step budget per phase. Its length is the number of phases. |
| `lr` | `tuple[float, ...]` | Learning rate per phase. Applied to the live optimiser state unless that phase also resets it. |
| `optimizer` | `tuple[OptimizerSpec, ...]` | Optimiser per phase: a registered name (``"adamw"``, ``"adabelief"``), a factory taking ``learning_rate`` and returning an ``optax.GradientTransformation`` (e.g. ``optax.adamw``, or ``lambda learning_rate: optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))``), or a ready-made transformation instance. Names and factories are wrapped in ``optax.inject_hyperparams``, so a phase boundary can move the learning rate without a rebuild; a raw instance cannot be re-hyperparametrised, so a phase that changes ``lr`` with a raw instance must also set ``reset_optimiser_state``. A phase that changes the optimiser must set ``reset_optimiser_state``, because optimiser state belongs to the optimiser that built it. |
| `reset_optimiser_state` | `tuple[bool, ...]` | Per phase, rebuild the optimiser and discard its state at that boundary. Set it when switching optimiser, and when a length-schedule change has made the accumulated momentum wrong. |
| `length_schedule` | `tuple[float, ...]` | Fraction of each experiment's timeline the loss looks at, per phase, in ``(0, 1]``. It masks the **loss**, never the integration: the solver still runs the full trajectory, and only the first ``fraction`` of the observation times is scored, which stops a long-horizon divergence from drowning the gradient. Being a runtime mask rather than a shape change, a phase boundary costs no recompile. Default ``(1.0,)`` scores everything. |
| `penalty_weight` | `tuple[float, ...]` | Weight on the bound-saturation penalty. Length 1 broadcasts to every phase; any other length must match ``steps``. Entries must be non-negative. ``0.0`` disables the penalty. The weight is relative to the per-bucket-averaged data term: the penalty is a mean over its points, charged once per step onto the averaged data gradient, so the same weight means the same thing whatever the dataset or point-count size. |
| `penalty_points` | `tuple[Array, ...] | None` | User-supplied penalty-only points for the bound penalty: one ``[G, n_inputs]`` array of physical input vectors per ``BoundedPredictor`` leaf, in traversal order, matching each leaf's ``input_keys`` column order. No measurements are needed there; saturation is charged at these points regardless of the data. When ``None`` the penalty uses only the measured points gathered from the dataset (leaves whose inputs do not all resolve to dataset covariates must be covered by an entry here, or the run raises). ``hybridmodels.penalties.box_grid`` builds a warp-uniform box sweep for the "police the whole box" recipe. |
| `penalty_fn` | `Callable | None` | The regulariser added to the data objective, defaulting to :func:`hybridmodels.penalties.bound_penalty` when ``None``. A custom callable ``(predictors, points) -> scalar`` replaces the bound penalty with e.g. weight decay on inner weights or a monotonicity term; one that ignores points simply does not use them. |
| `trajectory_penalty_fn` | `Callable | None` | Trajectory-aware penalty for **embedded** hybrid models (the predictor runs inside the vector field). Called as ``(full_state, bp) -> scalar`` with the full state ``[N, T, S]`` *including* any penalty accumulators carried in the ODE state; add it to the data loss inside the same forward pass. ``None`` (the default) disables it. See the helpers in ``hybridmodels.penalties`` (``attach_penalty_state`` / ``penalty_vector_field`` / ``strip_penalty_state`` / ``penalty_integral``). |
| `trajectory_penalty_weight` | `float` | Scalar weight on ``trajectory_penalty_fn``. ``0.0`` disables it even if a function is set. Non-negative. |
| `loss` | `Callable | str` | A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``, ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``. |
| `channel_idx, channel_weights` | `tuple | None` | Forwarded into the resolved loss. See ``hybridmodels.losses``. |
| `tournament_attempts, tournament_steps` | `int` | The tournament runs only when ``tournament_steps > 0`` and ``tournament_attempts > 1``. Each attempt re-initialises the predictors, trains for ``tournament_steps`` steps, and is scored on the data term alone by a forward-only pass; the lowest wins. An attempt that raises a diffrax error or a non-finite loss is dropped and the next key tried. If all fail, the original predictors are used and a ``RuntimeWarning`` is raised. |
| `tournament_lr` | `float` | Learning rate for the tournament's short bursts, independent of ``lr``. |
| `patience` | `int` | Consecutive steps without a new best data loss before the current phase stops early. Counted within a phase and reset at every phase boundary, so a plateau at the end of one phase cannot kill the next before its new learning rate acts. ``0`` disables early stopping. |
| `restore_best` | `bool` | Return the predictors from the lowest-data-loss step instead of the last one. The running minimum resets whenever ``length_schedule`` changes, since losses over different horizons are not comparable and the shortest horizon would otherwise always own the minimum. |
| `verbose` | `bool` | Selects ``RichTrainingUI`` over ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L87)</small>

#### `OptaxTrainingConfig.penalty_weight_for_phase()`

```python
penalty_weight_for_phase(self, phase_idx: 'int') -> 'float'
```

Penalty weight for ``phase_idx``, honouring the length-1 broadcast.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L225)</small>

---

<a id="train_with_optax"></a>

### `train_with_optax()`

<small>`from hybridmodels.training import train_with_optax` &nbsp;·&nbsp; also re-exported as `hybridmodels.train_with_optax`</small>

```python
train_with_optax(
    predictors: 'Any',
    dataset: 'Dataset',
    config: 'OptaxTrainingConfig',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    trainable: 'Any' = None,
    key: 'Array',
    ui: 'TrainingUI | None' = None,
) -> tuple[list[float], Any]
```

Train ``predictors`` against ``dataset`` with Optax.

Runs the phases described by ``config``, optionally preceded by a
tournament. See the module docstring for what a step, a phase, and
the tournament are, and :class:`OptaxTrainingConfig` for the fields.

``predictors`` is a ``PyTree[eqx.Module]``, conventionally a tuple of
``BoundedPredictor`` leaves but any shape ``eqx.partition`` can walk.
``key`` is keyword-only and required, so reproducibility never rests on
an implicit default.

``state_to_output`` is the pure mapping ``[T, S] -> [T, D]`` from full
simulator state to observed channels. It is a property of the model,
passed here rather than stored on the ``Dataset`` (ADR-0008).

``trainable`` is a boolean mask matching ``predictors``. Omitting it
defaults to :func:`hybridmodels.trainable.trainable_mask`, which marks
every inexact-array leaf trainable. Pass a custom mask, usually from
the freezers in ``hybridmodels.trainable``, to hold leaves fixed;
freezing ``BoundScaler`` leaves is the common case.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[list[float], PyTree[eqx.Module]]` |  | ``(loss_history, trained_predictors)``.<br><br>``loss_history`` is the **raw per-step loss**, one entry per step, concatenated across phases. It can go up. It is the data term plus any configured trajectory penalty (charged inside the bucket forward pass); the bound penalty is excluded, so a ramping bound-penalty weight cannot move the series and runs with different bound weights stay comparable, and nothing is smoothed: these are the values the optimiser saw.<br><br>It differs from :func:`~hybridmodels.training.evosax.train_with_evosax`, whose history is best-so-far and therefore monotone. Same type, same position, different meaning: plotting both on one axis misleads.<br><br>``trained_predictors`` comes from the lowest-loss step when ``config.restore_best=True``, else the final step. That minimum resets whenever ``length_schedule`` changes, so the returned model always comes from the last horizon trained on. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L735)</small>

---

<a id="train_seed_ensemble"></a>

### `train_seed_ensemble()`

<small>`from hybridmodels.training import train_seed_ensemble` &nbsp;·&nbsp; also re-exported as `hybridmodels.train_seed_ensemble`</small>

```python
train_seed_ensemble(
    predictors: 'Any',
    dataset: 'Dataset',
    config: 'OptaxTrainingConfig',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    n_seeds: 'int',
    k_best: 'int | None' = None,
    trainable: 'Any' = None,
    key: 'Array',
    ui: 'TrainingUI | None' = None,
) -> list[tuple[float, Any]]
```

Train a seed ensemble: rank warm starts, fully train the best ``k_best``.

The **tournament** already ranks fresh initialisations cheaply (a few
warm-up steps each, scored forward-only). This reuses that ranking to
pick which seeds deserve a full training run, instead of fully training
every seed: with ``n_seeds`` tournament attempts it keeps the best
``k_best`` (default: all ``n_seeds``) and runs the full phase schedule
on each.

Returns the fully trained members as a list of ``(final_loss,
predictors)`` ranked ascending by final loss — the same container
shape as the single ``predictors`` you passed in, so the result is an
ensemble of model pytrees ready for :func:`ensemble_predictions`.
``final_loss`` labels the returned member: the best-step loss when
``config.restore_best`` is set (measured exactly at the returned
parameters), else the last step's reported loss (measured one update
before, as the loop reports it).

``config.tournament_steps`` controls how much warm-up each seed gets
before ranking; the default ``0`` ranks by the initialisation score
alone (still a valid, cheapest ranking). ``config`` is otherwise used
exactly as in :func:`train_with_optax`.

Each member is its own UI run: the warm-up compile and the tournament
land inside the first member's bracket, and ``on_run_start`` /
``on_run_end`` are fired once per member, so a live dashboard shows
each member's phases rather than freezing after the first.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L1061)</small>

---

<a id="train_bootstrap_ensemble"></a>

### `train_bootstrap_ensemble()`

<small>`from hybridmodels.training import train_bootstrap_ensemble` &nbsp;·&nbsp; also re-exported as `hybridmodels.train_bootstrap_ensemble`</small>

```python
train_bootstrap_ensemble(
    predictors: 'Any',
    dataset: 'Dataset',
    config: 'OptaxTrainingConfig',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    n_bootstraps: 'int',
    n_seeds: 'int' = 1,
    k_best: 'int | None' = None,
    trainable: 'Any' = None,
    key: 'Array',
    ui: 'TrainingUI | None' = None,
) -> list[tuple[float, Any]]
```

Train a bootstrap ensemble: one (or a seed-set of) model(s) per resample.

For each of ``n_bootstraps`` resampled datasets (via
:func:`hybridmodels.make_bootstrap_dataset`), trains a model. With
``n_seeds > 1`` each resample's member is itself seed-selected by
:func:`train_seed_ensemble` (so every member both sees different data
*and* is a good seed); with ``n_seeds == 1`` each resample contributes
one fresh re-initialisation trained via the same phase schedule as
:func:`train_with_optax`.

Returns the members as a list of ``(final_loss, predictors)`` ranked
ascending by final loss across **all** bootstrap samples. ``k_best``
trims the ensemble to its best members overall (default: keep
``n_bootstraps * max(n_seeds, 1)``). ``final_loss`` labels the
returned member: the best-step loss when ``config.restore_best`` is
set (measured exactly at the returned parameters), else the last
step's reported loss (measured one update before, as the loop
reports it).

Every resample and every training run is folded off the one ``key``, so
the whole ensemble is deterministic given it. Each member is its own
UI run, bracketed by ``on_run_start`` / ``on_run_end``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L1194)</small>

---

<a id="evosaxtrainingconfig"></a>

### `EvosaxTrainingConfig`

<small>`from hybridmodels.training import EvosaxTrainingConfig` &nbsp;·&nbsp; also re-exported as `hybridmodels.EvosaxTrainingConfig`</small>

```python
EvosaxTrainingConfig(
    algorithm: 'str' = 'CMA_ES',
    population_size: 'int' = 64,
    num_generations: 'int' = 100,
    init: "Literal['warm', 'uniform_box', 'lhs_box']" = 'warm',
    penalty_weight: 'float' = 0.0,
    penalty_points: 'tuple[Array, ...] | None' = None,
    penalty_fn: 'Callable[..., Array] | None' = None,
    trajectory_penalty_fn: 'Callable[..., Array] | None' = None,
    trajectory_penalty_weight: 'float' = 0.0,
    init_box_extent: 'float' = 2.0,
    sigma_init: 'float' = 0.1,
    loss: 'Callable[..., Array] | str' = 'mse',
    channel_idx: 'tuple[int, ...] | None' = None,
    channel_weights: 'tuple[float, ...] | None' = None,
    log_every: 'int' = 1,
    verbose: 'bool' = True,
) -> None
```

Configuration for :func:`train_with_evosax`.

Unlike :class:`~hybridmodels.training.optax.OptaxTrainingConfig` there
are no phase-keyed tuples: the run is a flat loop, so every field is a
scalar.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `algorithm` |  | Evosax strategy name, one of the keys of ``ALGORITHM_REGISTRY``: ``"CMA_ES"`` (default), ``"Sep_CMA_ES"``, ``"SimpleES"``. Add another strategy with :func:`register_algorithm`. |
| `population_size, num_generations` |  | Loop dimensions. The population is evaluated in parallel through ``vmap``; generations run in sequence. |
| `init` |  | Initial-population scheme, one of ``"warm"``, ``"uniform_box"``, ``"lhs_box"``. The module docstring explains the difference. |
| `penalty_weight` |  | Weight on the bound-saturation penalty, folded into each individual's fitness. ``0.0`` disables it. Scalar, not a tuple. Relative to the per-bucket-averaged data term, charged once per evaluation. |
| `penalty_points` |  | User-supplied penalty-only points for the bound penalty: one ``[G, n_inputs]`` array of physical input vectors per ``BoundedPredictor`` leaf, in traversal order, matching each leaf's ``input_keys`` column order. ``None`` (the default) uses only the measured points gathered from the dataset; leaves whose inputs do not all resolve to dataset covariates must be covered by an entry here, or the run raises. |
| `penalty_fn` |  | The regulariser added to each individual's fitness, defaulting to :func:`hybridmodels.penalties.bound_penalty` when ``None``. A custom callable ``(predictors, points) -> scalar`` replaces the bound penalty. |
| `trajectory_penalty_fn` |  | Trajectory-aware penalty for **embedded** hybrid models, called as ``(full_state, bp) -> scalar`` with the full state ``[N, T, S]`` *including* any penalty accumulators carried in the ODE state. Folded into each individual's fitness. ``None`` (the default) disables it. See ``hybridmodels.penalties`` (``attach_penalty_state`` / ``penalty_vector_field`` / ``strip_penalty_state`` / ``penalty_integral``). |
| `trajectory_penalty_weight` |  | Scalar weight on ``trajectory_penalty_fn``. ``0.0`` disables it even if a function is set. Non-negative. |
| `init_box_extent` |  | Half-width of the box for ``"uniform_box"`` and ``"lhs_box"``. Ignored by ``"warm"``. |
| `sigma_init` |  | Initial CMA-ES step size. Used directly by ``"warm"`` and as the prior step size for the box-init modes. CMA-ES adapts it after the first ``tell``. |
| `loss` |  | A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``, ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``. |
| `channel_idx, channel_weights` |  | Forwarded into the resolved loss. See ``hybridmodels.losses``. |
| `log_every` |  | UI heartbeat cadence. Honoured only by the Rich UIs; the silent and recording UIs see every generation. |
| `verbose` |  | Selects ``RichEvosaxUI`` over ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L116)</small>

---

<a id="train_with_evosax"></a>

### `train_with_evosax()`

<small>`from hybridmodels.training import train_with_evosax` &nbsp;·&nbsp; also re-exported as `hybridmodels.train_with_evosax`</small>

```python
train_with_evosax(
    predictors: 'Any',
    dataset: 'Dataset',
    config: 'EvosaxTrainingConfig',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    trainable: 'Any' = None,
    key: 'Array',
    ui: 'EvosaxUI | None' = None,
) -> tuple[list[float], Any]
```

Train ``predictors`` against ``dataset`` with an evolutionary strategy.

Runs ``config.num_generations`` generations of CMA-ES over a
population of ``config.population_size`` candidates, taking no
gradient. See the module docstring for the JIT boundary and the
initial-population modes.

``predictors`` is a ``PyTree[eqx.Module]`` in any shape. ``key`` is
keyword-only and required. ``state_to_output`` is the pure mapping
``[T, S] -> [T, D]`` from full simulator state to observed channels; a
property of the model, passed here rather than stored on the ``Dataset``
(ADR-0008). ``trainable`` defaults to
:func:`hybridmodels.trainable.trainable_mask`, and must select at
least one scalar.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `history` | `list[float]` | **Best loss so far** at the end of each generation, so the series is monotone non-increasing. Length ``config.num_generations``.<br><br>It differs from :func:`~hybridmodels.training.optax.train_with_optax`, whose history is the raw per-step loss and can go up. Same type, same position, different meaning: plotting both on one axis misleads.<br><br>When ``config.penalty_weight > 0`` the recorded value is the combined objective, since evosax ranks by one scalar. The optax history excludes its penalty. |
| `best_predictors` | `Any` | Predictors rebuilt from the lowest-loss flat vector seen in any generation, the warm-up evaluation of the input predictors included. Same container shape as the input. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L402)</small>

---

<a id="register_algorithm"></a>

### `register_algorithm()`

<small>`from hybridmodels.training.evosax import register_algorithm` &nbsp;·&nbsp; also re-exported as `hybridmodels.register_algorithm`</small>

```python
register_algorithm(name: 'str', cls: 'type') -> 'None'
```

Register an evosax strategy class under ``name`` for ``algorithm=``.

After registration, ``EvosaxTrainingConfig(algorithm=name)`` builds that
strategy. Re-registering an existing name overwrites without warning.
Mirrors :func:`hybridmodels.solver.register_solver`.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L106)</small>
