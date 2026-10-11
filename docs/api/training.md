# Training: Optax & Evosax Loops

`HybridModel.fit` and `HybridModel.fit_ensemble` are the entry points. [`Objective`](#objective) says what is minimised; [`OptaxTrainingConfig`](#optaxtrainingconfig) or [`EvosaxTrainingConfig`](#evosaxtrainingconfig) says how to search. [`train_with_optax`](#train_with_optax) and [`train_with_evosax`](#train_with_evosax) are the functions `fit` dispatches to: they take the model first, then `fit`'s arguments, and return a `FitResult`.

## Quick links

- [`Objective`](#objective)
- [`OptaxTrainingConfig`](#optaxtrainingconfig)
- [`TrainingPhase`](#trainingphase)
- [`train_with_optax`](#train_with_optax)
- [`EvosaxTrainingConfig`](#evosaxtrainingconfig)
- [`train_with_evosax`](#train_with_evosax)
- [`register_algorithm`](#register_algorithm)

---

<a id="objective"></a>

### `Objective`

<small>`from jaxhybridmodels.objective import Objective` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.Objective`</small>

```python
Objective(
    loss: 'Callable[..., Array] | str' = 'mse',
    channel_weights: 'Mapping[str, float] | Sequence[float] | None' = None,
    penalty_weight: 'float' = 0.0,
    penalty_points: 'Mapping[str, ArrayLike] | None' = None,
    penalty_fn: 'Callable[..., Array] | None' = None,
    trajectory_penalty_fn: 'Callable[..., Array] | None' = None,
    trajectory_penalty_weight: 'float' = 0.0,
) -> None
```

The training objective: data loss plus optional penalties.

Keyword-only. ``Objective()`` is plain MSE over every channel with no
penalty.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `loss` | `Callable | str` | A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``, ``"bal_mle"``) or a callable ``loss(pred_obs, bp) -> scalar``. |
| `channel_weights` | `Mapping[str, float] | Sequence[float] | None` | Per-channel multipliers on the loss. A mapping keyed by channel name scores only the channels it names (``{"conc": 1.0}`` fits one channel of several); a sequence gives one weight per output channel, in ``output_channel_names`` order, and keeps every channel's cells in the loss's denominator. ``None`` weights every channel 1. Names are checked against the dataset's channels when training starts. A named channel with weight ``0.0`` contributes nothing but its cells still count in the denominator, so ``{"a": 1.0, "b": 0.0}`` gives half the loss of ``{"a": 1.0}``; leave a name out to drop it. |
| `penalty_weight` | `float` | Weight on the bound-saturation penalty, relative to the per-bucket-averaged data term. ``0.0`` disables it. ``TrainingPhase.penalty_weight`` overrides it for one Optax phase. |
| `penalty_points` | `Mapping[str, ArrayLike] | None` | Extra points where the bound penalty is charged, keyed by the path of a ``BoundedPredictor`` in the predictors pytree. The path is the dotted string ``fit(freeze=...)`` takes, so for a dict of predictors it is the key: ``{"rate": grid}``. Each value is a ``[G, n_inputs]`` array of physical input vectors in that predictor's ``input_keys`` order. Predictors left out are charged at the measured points only. ``jaxhybridmodels.penalties.box_grid`` builds a sweep of an input box. |
| `penalty_fn` | `Callable | None` | Replaces [`bound_penalty`](/api/penalties#bound_penalty). Called as ``(predictors, points) -> scalar`` where ``points`` is one ``[G, n_inputs]`` array per ``BoundedPredictor`` in traversal order; a custom term that needs no points ignores them. |
| `trajectory_penalty_fn` | `Callable | None` | Trajectory-aware penalty for embedded hybrid models, called as ``(full_state, bp) -> scalar`` on the full simulator state ``[N, T, S]``, penalty accumulators included. See the ``penalty_*`` helpers in ``jaxhybridmodels.penalties``. |
| `trajectory_penalty_weight` | `float` | Weight on ``trajectory_penalty_fn``. ``0.0`` disables it even when a function is set; a non-zero weight without a function raises. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/objective.py#L42)</small>

<a id="objectiveextra_penalty_points"></a>

#### `Objective.extra_penalty_points()`

```python
extra_penalty_points(self, predictors: 'Any') -> 'tuple[Array, ...]'
```

``penalty_points`` as one array per ``BoundedPredictor``, in traversal order.

Predictors with no entry get an empty ``[0, n_inputs]`` array. An
unknown path raises, naming the valid ones.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/objective.py#L146)</small>

<a id="objectiveloss_fn"></a>

#### `Objective.loss_fn()`

```python
loss_fn(
    self,
    output_channel_names: 'Sequence[str]',
) -> Callable[[Array, BucketPayload], Array]
```

The resolved ``(pred_obs, bp) -> scalar`` for a dataset with these channels.

Channel names in ``channel_weights`` are matched here, so an unknown
name raises before anything compiles.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/objective.py#L128)</small>

---

<a id="optaxtrainingconfig"></a>

### `OptaxTrainingConfig`

<small>`from jaxhybridmodels.training import OptaxTrainingConfig` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.OptaxTrainingConfig`</small>

```python
OptaxTrainingConfig(
    steps: 'InitVar[int | None]' = None,
    lr: 'InitVar[float | optax.Schedule | None]' = None,
    optimizer: 'InitVar[OptimizerSpec | None]' = None,
    phases: 'tuple[TrainingPhase, ...]' = (),
    tournament_attempts: 'int' = 1,
    tournament_steps: 'int' = 0,
    tournament_lr: 'float' = 0.0001,
    patience: 'int' = 0,
    restore_best: 'bool' = True,
    verbose: 'bool' = True,
) -> None
```

Search settings for [`train_with_optax`](/api/training#train_with_optax) and Optax ``HybridModel.fit``.

What is minimised (loss, channel weights, penalties) lives in
[`Objective`](/api/training#objective), passed separately as
``objective=``. This config says how to minimise it.

A single phase is spelled with ``steps=`` (plus optional ``lr=`` and
``optimizer=``). A schedule is spelled with ``phases=``, a tuple of
[`TrainingPhase`](/api/training#trainingphase). The two forms do not mix; either way the config
stores ``phases``::

    OptaxTrainingConfig(steps=500, lr=1e-3)
    OptaxTrainingConfig(steps=500, lr=optax.cosine_decay_schedule(1e-2, 500))
    OptaxTrainingConfig(
        phases=(
            TrainingPhase(steps=200, lr=3e-2, horizon=0.5),
            TrainingPhase(steps=200, lr=5e-3, reset_optimizer=True),
        )
    )

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `phases` | `tuple[TrainingPhase, ...]` | The schedule, run in order. See [`TrainingPhase`](/api/training#trainingphase) for the per-phase settings. |
| `tournament_attempts, tournament_steps` | `int` | The tournament runs only when ``tournament_steps > 0`` and ``tournament_attempts > 1``. Each attempt re-initialises the predictors, trains for ``tournament_steps`` steps, and is scored on the data term plus any configured trajectory penalty by a forward-only pass; the bound penalty remains excluded. The lowest score wins. An attempt that raises a diffrax error or a non-finite loss is dropped and the next key tried. If all fail, the original predictors are used and a ``RuntimeWarning`` is raised. |
| `tournament_lr` | `float` | Learning rate for the tournament's short bursts, independent of the phase learning rates. |
| `patience` | `int` | Consecutive steps without a new best tracked loss before the current phase stops early. The tracked loss is the data term plus any trajectory penalty, excluding the fixed-point bound penalty. Counted within a phase and reset at every phase boundary, so a plateau at the end of one phase cannot kill the next before its new learning rate acts. ``0`` disables early stopping. |
| `restore_best` | `bool` | Return the predictors from the lowest tracked-loss step instead of the last one. The running minimum resets whenever the horizon changes, since tracked losses over different horizons are not comparable and the shortest horizon would otherwise always own the minimum. |
| `verbose` | `bool` | Selects ``RichTrainingUI`` over ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/training/optax.py#L177)</small>

---

<a id="trainingphase"></a>

### `TrainingPhase`

<small>`from jaxhybridmodels.training import TrainingPhase` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.TrainingPhase`</small>

```python
TrainingPhase(
    steps: 'int',
    lr: 'float | optax.Schedule' = 0.001,
    optimizer: 'OptimizerSpec' = 'adamw',
    horizon: 'float' = 1.0,
    reset_optimizer: 'bool' = False,
    penalty_weight: 'float | None' = None,
) -> None
```

One contiguous block of Optax steps with its own schedule settings.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `steps` | `int` | Step budget for the phase. At least 1. |
| `lr` | `float | optax.Schedule` | Learning rate: a finite non-negative float, or an optax schedule ``step -> lr`` whose step count starts at 0 at the start of this phase, whatever ran before it. Chain several curves inside one phase with ``optax.join_schedules``. Any callable works: one JAX cannot trace, such as ``lambda s: 0.1 if s < 2 else 0.01``, is evaluated step by step on the host. A float is applied to the live optimiser state unless the phase also sets ``reset_optimizer``. A schedule needs an optimiser name or factory; a ready-made ``optax.GradientTransformation`` owns its learning rate, so pairing one with a schedule raises. |
| `optimizer` | `OptimizerSpec` | A registered name (``"adamw"``, ``"adabelief"``), a factory taking ``learning_rate`` and returning an ``optax.GradientTransformation`` (e.g. ``optax.adamw``, or ``lambda learning_rate: optax.chain( optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))``), or a ready-made transformation instance. Names and factories are wrapped in ``optax.inject_hyperparams``, so a phase boundary or a schedule can move the learning rate without a rebuild. A raw instance cannot be re-hyperparametrised, so a phase that changes ``lr`` with a raw instance must set ``reset_optimizer``. A phase that changes the optimiser must set it too, because optimiser state belongs to the optimiser that built it. |
| `horizon` | `float` | Fraction of each experiment's timeline the loss looks at, in ``(0, 1]``. It masks the **loss**, never the integration: the solver still runs the full trajectory, and only the first ``horizon`` of the observation times is scored, which stops a long-horizon divergence from drowning the gradient. Being a runtime mask rather than a shape change, a phase boundary costs no recompile. |
| `reset_optimizer` | `bool` | Rebuild the optimiser and discard its state at the start of this phase. Set it when switching optimiser, and when a horizon change has made the accumulated momentum wrong. |
| `penalty_weight` | `float | None` | Override of ``Objective.penalty_weight`` for this phase. ``None`` (the default) uses the objective's weight; ``0.0`` switches the bound penalty off for the phase. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/training/optax.py#L93)</small>

---

<a id="train_with_optax"></a>

### `train_with_optax()`

<small>`from jaxhybridmodels.training import train_with_optax`</small>

```python
train_with_optax(
    model: 'HybridModel',
    data: 'Data',
    config: 'OptaxTrainingConfig',
    key: 'Array',
    objective: 'Objective | None' = None,
    freeze: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
    trainable: 'Any' = None,
    ui: 'TrainingUI | None' = None,
    validation: 'Data | None' = None,
) -> FitResult
```

Train ``model``'s predictors with Optax; what ``model.fit`` runs for an Optax config.

Runs the phases described by ``config``, optionally preceded by a
tournament. See the module docstring for what a step, a phase, and the
tournament are, and [`OptaxTrainingConfig`](/api/training#optaxtrainingconfig) for the fields. Every
keyword means what it does for [`HybridModel.fit>`](/api/model#hybridmodelfit), and ``data`` and ``validation`` may
be a ``Dataset`` or a list of experiments.

``history.loss`` is the raw per-step loss, measured before that step's
update, concatenated across phases. It can go up. It is the data term
plus any trajectory penalty; the bound penalty is in
``history.penalty``, so a ramping bound weight does not move the loss
series. Evosax records best-so-far instead, so the two series mean
different things.

The returned model comes from the lowest-loss step when
``config.restore_best=True``, else the final step. That minimum resets
whenever the phase horizon changes, so the returned model always comes
from the last horizon trained on.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/training/optax.py#L859)</small>

---

<a id="evosaxtrainingconfig"></a>

### `EvosaxTrainingConfig`

<small>`from jaxhybridmodels.training import EvosaxTrainingConfig` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.EvosaxTrainingConfig`</small>

```python
EvosaxTrainingConfig(
    algorithm: 'str' = 'CMA_ES',
    population_size: 'int' = 64,
    steps: 'int' = 100,
    init: "Literal['warm', 'uniform_box', 'lhs_box']" = 'warm',
    init_box_extent: 'float' = 2.0,
    sigma_init: 'float' = 0.1,
    log_every: 'int' = 1,
    verbose: 'bool' = True,
) -> None
```

Search settings for [`train_with_evosax`](/api/training#train_with_evosax) and Evosax ``HybridModel.fit``.

What is minimised lives in [`Objective`](/api/training#objective), passed
separately as ``objective=``. Unlike
[`OptaxTrainingConfig`](/api/training#optaxtrainingconfig) there are
no phases: the run is a flat loop, so every field is a scalar.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `algorithm` |  | Evosax strategy name, one of the keys of ``ALGORITHM_REGISTRY``: ``"CMA_ES"`` (default), ``"Sep_CMA_ES"``, ``"SimpleES"``. Add another strategy with [`register_algorithm`](/api/training#register_algorithm). |
| `population_size` |  | Individuals per generation, evaluated in parallel through ``vmap``. |
| `steps` |  | Number of generations, run in sequence. Named like the Optax config's ``steps``: one generation is one Evosax step. |
| `init` |  | Initial-population scheme, one of ``"warm"``, ``"uniform_box"``, ``"lhs_box"``. The module docstring explains the difference. |
| `init_box_extent` |  | Half-width of the box for ``"uniform_box"`` and ``"lhs_box"``. Ignored by ``"warm"``. |
| `sigma_init` |  | Initial CMA-ES step size. Used directly by ``"warm"`` and as the prior step size for the box-init modes. CMA-ES adapts it after the first ``tell``. |
| `log_every` |  | UI heartbeat cadence. Honoured only by the Rich UIs; the silent and recording UIs see every generation. |
| `verbose` |  | Selects ``RichEvosaxUI`` over ``SilentUI`` when ``ui=None``. An explicit ``ui=...`` argument always wins. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/training/evosax.py#L122)</small>

---

<a id="train_with_evosax"></a>

### `train_with_evosax()`

<small>`from jaxhybridmodels.training import train_with_evosax`</small>

```python
train_with_evosax(
    model: 'HybridModel',
    data: 'Data',
    config: 'EvosaxTrainingConfig',
    key: 'Array',
    objective: 'Objective | None' = None,
    freeze: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
    trainable: 'Any' = None,
    ui: 'EvosaxUI | None' = None,
    validation: 'Data | None' = None,
) -> FitResult
```

Train ``model``'s predictors with an evolutionary strategy; ``fit`` for an Evosax config.

Runs ``config.steps`` generations over a population of
``config.population_size`` candidates, taking no gradient. See the
module docstring for the JIT boundary and the initial-population modes.
Every keyword means what it does for [`HybridModel.fit>`](/api/model#hybridmodelfit), and ``data`` and ``validation`` may
be a ``Dataset`` or a list of experiments.

Each individual's fitness is the bucket-averaged data loss plus any
trajectory penalty plus ``objective.penalty_weight`` times the bound
penalty, since evosax ranks by one scalar.

``history.loss`` is the best fitness so far after each generation, so
it is monotone non-increasing; Optax records the raw per-step loss
instead, so the two series mean different things. Without
``validation`` the returned model is rebuilt from the lowest-loss
candidate seen in any generation, the warm-up evaluation of the input
included. With it, each generation's best is scored on the validation
set and the returned model is the one that scored lowest there.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/training/evosax.py#L354)</small>

---

<a id="register_algorithm"></a>

### `register_algorithm()`

<small>`from jaxhybridmodels.training.evosax import register_algorithm`</small>

```python
register_algorithm(name: 'str', cls: 'type') -> 'None'
```

Register an evosax strategy class under ``name`` for ``algorithm=``.

After registration, ``EvosaxTrainingConfig(algorithm=name)`` builds that
strategy. Re-registering an existing name overwrites without warning.
Mirrors [`register_solver`](/api/solver#register_solver).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/training/evosax.py#L112)</small>
