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

OptaxTrainingConfig(steps: 'tuple[int, ...]', lr: 'tuple[float, ...]', optimizer: 'tuple[str, ...]', reset_optimiser_state: 'tuple[bool, ...]', length_schedule: 'tuple[float, ...]' = (1.0,), penalty_weight: 'tuple[float, ...]' = (0.0,), penalty_grid_points: 'int' = 5, loss: 'Callable[..., Array] | str' = 'mse', channel_idx: 'tuple[int, ...] | None' = None, channel_weights: 'tuple[float, ...] | None' = None, tournament_attempts: 'int' = 1, tournament_steps: 'int' = 0, tournament_lr: 'float' = 0.0001, patience: 'int' = 0, restore_best: 'bool' = True, verbose: 'bool' = True)

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L56)</small>

#### `OptaxTrainingConfig.penalty_weight_for_phase()`

```python
penalty_weight_for_phase(self, phase_idx: 'int') -> 'float'
```

Penalty weight for ``phase_idx``, honouring the length-1 broadcast.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L75)</small>

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

``predictors`` is a ``PyTree[eqx.Module]``: by convention a tuple of
``BoundedPredictor`` leaves, but any pytree shape is accepted (dict,
NamedTuple, single Module — ``eqx.partition`` walks them uniformly).
``key`` is required keyword-only — calling without it raises
``TypeError`` before any compilation, so reproducibility never
relies on an implicit default.

The ``trainable`` argument is a boolean PyTree mask matching
``predictors``'s structure. When omitted, it defaults to
:func:`hybridmodels.trainable.trainable_mask` over the supplied
pytree, which marks every inexact-array leaf as trainable; pass a
custom mask (typically built with the freezers in
``hybridmodels.trainable``) to hold specific leaves fixed during
training.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[list[float], PyTree[eqx.Module]]` |  | ``(loss_history, trained_predictors)``. ``loss_history`` is the training loss recorded once per step across every phase; ``trained_predictors`` is the predictors corresponding to the best-loss step seen so far when ``config.restore_best=True``, or to the final step otherwise. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/optax.py#L350)</small>

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

Configuration for ``train_with_evosax``.

Unlike the Optax loop, there are no per-phase tuples here: evosax is
a flat outer loop of ``num_generations`` over a population of
``population_size`` individuals.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `algorithm` |  | Evosax strategy name. Only ``"CMA_ES"`` is currently supported; the field exists so additional strategies can slot in without an API break. |
| `population_size, num_generations` |  | Outer-loop dimensions. Population is evaluated in parallel via ``vmap``; generations are sequential. |
| `init` |  | Initial-population scheme — see the module docstring for the difference between ``"warm"``, ``"uniform_box"``, and ``"lhs_box"``. |
| `init_box_extent` |  | Half-width of the box for ``"uniform_box"`` and ``"lhs_box"``. Ignored for ``"warm"``. |
| `sigma_init` |  | Initial CMA-ES step size. Used in ``"warm"`` and as the strategy's prior step size for the box-init modes (CMA-ES adapts it after the first ``tell``). |
| `loss` |  | Either a ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``, ``"bal_mle"``) or a callable matching the ``loss(pred_obs, bp) -> scalar`` contract. |
| `channel_idx, channel_weights` |  | Forwarded into the resolved loss; see ``hybridmodels.losses``. |
| `log_every` |  | UI heartbeat cadence (currently honoured only by Rich UIs; the silent and recording UIs see every generation). |
| `verbose` |  | Selects ``RichEvosaxUI`` vs ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L84)</small>

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

``predictors`` is a ``PyTree[eqx.Module]``; the convention is a
tuple of ``BoundedPredictor`` leaves but any pytree shape works.
``key`` is required keyword-only — calling without it raises
``TypeError`` before any work happens. ``trainable`` defaults to
:func:`hybridmodels.trainable.trainable_mask` over the supplied
pytree (every inexact-array leaf).

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `history` | `list[float]` | Best-loss-so-far per generation (length ``config.num_generations``). |
| `best_predictors` | `Any` | The reconstructed predictors pytree whose flat-parameter vector minimised the loss across every generation. Same container shape as the input ``predictors``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/evosax.py#L349)</small>
