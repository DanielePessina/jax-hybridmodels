# Training: Optax & Evosax Loops

Two training entry points share the same `(predictors, dataset, config, *, simulate_fn, solver, trainable, key, ui)` signature:

- [`train_with_optax`](#train_with_optax) — gradient-based, multi-phase   schedule, optional shared tournament for warm-up restarts.
- [`train_with_evosax`](#train_with_evosax) — population-based search   via evosax strategies; useful when the loss landscape is   non-differentiable or has many local minima.

Both return `(loss_history, trained_predictors)`. Both require a `key` keyword-only argument so reproducibility never relies on an implicit default.

## Quick links

- [`OptaxTrainingConfig`](#optaxtrainingconfig)
- [`train_with_optax`](#train_with_optax)
- [`EvosaxTrainingConfig`](#evosaxtrainingconfig)
- [`train_with_evosax`](#train_with_evosax)

---

<a id="optaxtrainingconfig"></a>

### `OptaxTrainingConfig`

<small>`from hybridmodels.training import OptaxTrainingConfig` &nbsp;·&nbsp; also re-exported as `hybridmodels.OptaxTrainingConfig`</small>

```python
OptaxTrainingConfig(
    steps: 'tuple[int, ...]',
    lr: 'tuple[float, ...]',
    optimizer: 'tuple[str, ...]',
    reset_optimiser_state: 'tuple[bool, ...]',
    length_schedule: 'tuple[float, ...]' = (1.0,),
    penalty_weight: 'tuple[float, ...]' = (0.0,),
    penalty_grid_points: 'int' = 5,
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
| `optimizer` | `tuple[str, ...]` | Optimiser name per phase, ``"adamw"`` or ``"adabelief"``. A phase that changes the name must also set ``reset_optimiser_state``, because optimiser state belongs to the optimiser that built it. |
| `reset_optimiser_state` | `tuple[bool, ...]` | Per phase, rebuild the optimiser and discard its state at that boundary. Set it when switching optimiser, and when a length-schedule change has made the accumulated momentum wrong. |
| `length_schedule` | `tuple[float, ...]` | Fraction of each experiment's timeline the loss looks at, per phase, in ``(0, 1]``. It masks the **loss**, never the integration: the solver still runs the full trajectory, and only the first ``fraction`` of the observation times is scored, which stops a long-horizon divergence from drowning the gradient. Being a runtime mask rather than a shape change, a phase boundary costs no recompile. Default ``(1.0,)`` scores everything. |
| `penalty_weight` | `tuple[float, ...]` | Weight on the bound-saturation penalty. Length 1 broadcasts to every phase; any other length must match ``steps``. Entries must be non-negative. ``0.0`` disables the penalty. |
| `penalty_grid_points` | `int` | Points per input dimension in the collocation grid the penalty is evaluated on. At least 2 (one per box edge). |
| `loss` | `Callable | str` | A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``, ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``. |
| `channel_idx, channel_weights` | `tuple | None` | Forwarded into the resolved loss. See ``hybridmodels.losses``. |
| `tournament_attempts, tournament_steps` | `int` | The tournament runs only when ``tournament_steps > 0`` and ``tournament_attempts > 1``. Each attempt re-initialises the predictors, trains for ``tournament_steps`` steps, and is scored on the data term alone by a forward-only pass; the lowest wins. An attempt that raises a diffrax error or a non-finite loss is dropped and the next key tried. If all fail, the original predictors are used and a ``RuntimeWarning`` is raised. |
| `tournament_lr` | `float` | Learning rate for the tournament's short bursts, independent of ``lr``. |
| `patience` | `int` | Consecutive steps without a new best data loss before the current phase stops early. Counted within a phase and reset at every phase boundary, so a plateau at the end of one phase cannot kill the next before its new learning rate acts. ``0`` disables early stopping. |
| `restore_best` | `bool` | Return the predictors from the lowest-data-loss step instead of the last one. The running minimum resets whenever ``length_schedule`` changes, since losses over different horizons are not comparable and the shortest horizon would otherwise always own the minimum. |
| `verbose` | `bool` | Selects ``RichTrainingUI`` over ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L65)</small>

#### `OptaxTrainingConfig.penalty_weight_for_phase()`

```python
penalty_weight_for_phase(self, phase_idx: 'int') -> 'float'
```

Penalty weight for ``phase_idx``, honouring the length-1 broadcast.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L161)</small>

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

``trainable`` is a boolean mask matching ``predictors``. Omitting it
defaults to :func:`hybridmodels.trainable.trainable_mask`, which marks
every inexact-array leaf trainable. Pass a custom mask, usually from
the freezers in ``hybridmodels.trainable``, to hold leaves fixed;
freezing ``BoundScaler`` leaves is the common case.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[list[float], PyTree[eqx.Module]]` |  | ``(loss_history, trained_predictors)``.<br><br>``loss_history`` is the **raw per-step data loss**, one entry per step, concatenated across phases. It can go up. The bound penalty is excluded, so a ramping penalty weight cannot move the series and runs with different weights stay comparable, and nothing is smoothed: these are the values the optimiser saw.<br><br>It differs from :func:`~hybridmodels.training.evosax.train_with_evosax`, whose history is best-so-far and therefore monotone. Same type, same position, different meaning: plotting both on one axis misleads.<br><br>``trained_predictors`` comes from the lowest-loss step when ``config.restore_best=True``, else the final step. That minimum resets whenever ``length_schedule`` changes, so the returned model always comes from the last horizon trained on. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L699)</small>

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
    penalty_grid_points: 'int' = 5,
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
| `algorithm` |  | Evosax strategy name. Only ``"CMA_ES"`` is wired in. The field exists so another strategy can slot in without an API break. |
| `population_size, num_generations` |  | Loop dimensions. The population is evaluated in parallel through ``vmap``; generations run in sequence. |
| `init` |  | Initial-population scheme, one of ``"warm"``, ``"uniform_box"``, ``"lhs_box"``. The module docstring explains the difference. |
| `penalty_weight` |  | Weight on the bound-saturation penalty, folded into each individual's fitness. ``0.0`` disables it. Scalar, not a tuple. |
| `penalty_grid_points` |  | Points per input dimension in the collocation grid the penalty is evaluated on. |
| `init_box_extent` |  | Half-width of the box for ``"uniform_box"`` and ``"lhs_box"``. Ignored by ``"warm"``. |
| `sigma_init` |  | Initial CMA-ES step size. Used directly by ``"warm"`` and as the prior step size for the box-init modes. CMA-ES adapts it after the first ``tell``. |
| `loss` |  | A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``, ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``. |
| `channel_idx, channel_weights` |  | Forwarded into the resolved loss. See ``hybridmodels.losses``. |
| `log_every` |  | UI heartbeat cadence. Honoured only by the Rich UIs; the silent and recording UIs see every generation. |
| `verbose` |  | Selects ``RichEvosaxUI`` over ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L85)</small>

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
keyword-only and required. ``trainable`` defaults to
:func:`hybridmodels.trainable.trainable_mask`, and must select at
least one scalar.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `history` | `list[float]` | **Best loss so far** at the end of each generation, so the series is monotone non-increasing. Length ``config.num_generations``.<br><br>It differs from :func:`~hybridmodels.training.optax.train_with_optax`, whose history is the raw per-step loss and can go up. Same type, same position, different meaning: plotting both on one axis misleads.<br><br>When ``config.penalty_weight > 0`` the recorded value is the combined objective, since evosax ranks by one scalar. The optax history excludes its penalty. |
| `best_predictors` | `Any` | Predictors rebuilt from the lowest-loss flat vector seen in any generation, the warm-up evaluation of the input predictors included. Same container shape as the input. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L282)</small>
