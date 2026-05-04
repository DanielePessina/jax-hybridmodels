# UI: Training Dashboards

Training UIs are protocols implemented by `RichTrainingUI` / `RichEvosaxUI` (live dashboards) and `SilentUI` (no-op). Pass via the `ui=` keyword on `train_with_optax` / `train_with_evosax`; the trainer calls lifecycle hooks (`on_phase_start`, `on_step_end`, `on_run_end`, ...) at the right points. Custom UIs implement the matching protocol — useful for piping training metrics into your own logger.

## Quick links

- [`TrainingUI`](#trainingui)
- [`EvosaxUI`](#evosaxui)
- [`SilentUI`](#silentui)
- [`RichTrainingUI`](#richtrainingui)
- [`RichEvosaxUI`](#richevosaxui)

---

<a id="trainingui"></a>

### `TrainingUI`

<small>`from hybridmodels.ui import TrainingUI` &nbsp;·&nbsp; also re-exported as `hybridmodels.TrainingUI`</small>

```python
TrainingUI(*args, **kwargs)
```

Callback protocol for ``train_with_optax``.

Implementations are duck-typed (``@runtime_checkable``) so users can
define their own UI without inheriting from this class. The default
Rich and Silent implementations live in ``ui/optax.py`` and
``ui/base.py`` respectively.

Event order during a typical run::

    on_run_start
    for each bucket shape encountered first time:
        on_compile_start  -> on_compile_progress*  -> on_compile_done
    for phase in phases:
        on_phase_start
            on_step_end (one per training step)
        on_phase_end
    on_run_end

``on_message`` may fire at any point for log lines (e.g. tournament
fallback warnings).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L30)</small>

#### `TrainingUI.on_compile_done()`

```python
on_compile_done(self, *, bucket_idx: int) -> None
```

The bucket at ``bucket_idx`` finished compiling.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L66)</small>

#### `TrainingUI.on_compile_progress()`

```python
on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None
```

Periodic heartbeat during long compiles (best-effort, may not fire).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L62)</small>

#### `TrainingUI.on_compile_start()`

```python
on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None
```

A bucket of shape ``bucket_shape`` is about to be JIT-compiled for the first time.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L58)</small>

#### `TrainingUI.on_message()`

```python
on_message(self, *, level: str, text: str) -> None
```

Free-form log line. ``level`` is one of ``"info"``, ``"warning"``, ``"error"``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L88)</small>

#### `TrainingUI.on_phase_end()`

```python
on_phase_end(self, *, phase_idx: int) -> None
```

Fires after the last step of a phase, before any optimiser reset.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L76)</small>

#### `TrainingUI.on_phase_start()`

```python
on_phase_start(
    self,
    phase_idx: int,
    phase_steps: int,
    lr: float,
    optimizer: str,
) -> None
```

Fires at the start of each phase; ``phase_steps`` is the per-phase step budget.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L70)</small>

#### `TrainingUI.on_run_end()`

```python
on_run_end(self, *, final_loss: float) -> None
```

Fires once after every phase has completed (or training was aborted gracefully).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L84)</small>

#### `TrainingUI.on_run_start()`

```python
on_run_start(self, *, total_steps: int, num_phases: int) -> None
```

Fires once before the first phase. ``total_steps`` is the sum across phases.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L54)</small>

#### `TrainingUI.on_step_end()`

```python
on_step_end(self, *, step_idx: int, phase_idx: int, loss: float) -> None
```

Fires after each training step. ``step_idx`` is global; ``phase_idx`` localises it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L80)</small>

---

<a id="evosaxui"></a>

### `EvosaxUI`

<small>`from hybridmodels.ui import EvosaxUI` &nbsp;·&nbsp; also re-exported as `hybridmodels.EvosaxUI`</small>

```python
EvosaxUI(*args, **kwargs)
```

Callback protocol for ``train_with_evosax``.

Differs from ``TrainingUI`` because evosax does not have phases or
per-step gradient losses; instead each generation reports best/mean
fitness across the population.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L93)</small>

#### `EvosaxUI.on_compile_done()`

```python
on_compile_done(self, *, bucket_idx: int) -> None
```

The bucket finished compiling.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L114)</small>

#### `EvosaxUI.on_compile_progress()`

```python
on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None
```

Periodic compile-time heartbeat (best-effort).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L110)</small>

#### `EvosaxUI.on_compile_start()`

```python
on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None
```

A bucket of shape ``bucket_shape`` is about to be JIT-compiled.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L106)</small>

#### `EvosaxUI.on_generation_end()`

```python
on_generation_end(
    self,
    gen_idx: int,
    best_fitness: float,
    mean_fitness: float,
) -> None
```

Fires once per generation with population statistics.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L118)</small>

#### `EvosaxUI.on_message()`

```python
on_message(self, *, level: str, text: str) -> None
```

Free-form log line; same level set as ``TrainingUI.on_message``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L126)</small>

#### `EvosaxUI.on_run_end()`

```python
on_run_end(self, *, best_fitness: float) -> None
```

Fires once after the last generation.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L122)</small>

#### `EvosaxUI.on_run_start()`

```python
on_run_start(self, *, num_generations: int, population_size: int) -> None
```

Fires once before the first generation.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L102)</small>

---

<a id="silentui"></a>

### `SilentUI`

<small>`from hybridmodels.ui import SilentUI` &nbsp;·&nbsp; also re-exported as `hybridmodels.SilentUI`</small>

```python
SilentUI()
```

No-op UI satisfying both ``TrainingUI`` and ``EvosaxUI``.

Selected when ``config.verbose=False`` and used in tests where stdout
output would pollute captured logs. Every method accepts ``**kwargs``
and returns ``None``, so it tolerates protocol drift without raising.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/base.py#L131)</small>

---

<a id="richtrainingui"></a>

### `RichTrainingUI`

<small>`from hybridmodels.ui import RichTrainingUI` &nbsp;·&nbsp; also re-exported as `hybridmodels.RichTrainingUI`</small>

```python
RichTrainingUI(
    console: 'Console | None' = None,
    recent_messages: 'int' = 5,
) -> None
```

Live Rich dashboard satisfying ``hybridmodels.ui.base.TrainingUI``.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `console:` |  | Optional :class:`rich.console.Console`. ``None`` constructs a default ``Console()``. Tests typically pass a recording console (``Console(record=True, force_terminal=False, ...)``) so the rendered final state can be asserted on. |
| `recent_messages:` |  | Maximum number of lines the message-log panel holds. |

**Notes**

The instance carries one :class:`rich.live.Live` between
``on_run_start`` and ``on_run_end``. Any state mutation outside that
window updates the model only; the next ``on_run_start`` rebuilds Live
afresh, so a single ``RichTrainingUI`` instance can be reused for
sequential runs (used in tests).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/optax.py#L53)</small>

---

<a id="richevosaxui"></a>

### `RichEvosaxUI`

<small>`from hybridmodels.ui import RichEvosaxUI` &nbsp;·&nbsp; also re-exported as `hybridmodels.RichEvosaxUI`</small>

```python
RichEvosaxUI(
    console: 'Console | None' = None,
    log_every: 'int | None' = None,
    recent_generations: 'int' = 5,
    recent_messages: 'int' = 5,
) -> None
```

Live Rich dashboard satisfying ``hybridmodels.ui.base.EvosaxUI``.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `console:` |  | Optional :class:`rich.console.Console`. ``None`` constructs a default ``Console()``. Tests typically pass a recording console (``Console(record=True, force_terminal=False, ...)``) so the rendered final state can be asserted on. |
| `log_every:` |  | Throttle for the recent-generation table. ``log_every=k`` records generations where ``gen_idx % k == 0``. ``None`` (the default) defers the choice to :meth:`on_run_start`, which sets it to ``max(1, num_generations // 5)`` so the table accumulates to exactly five rows over the run instead of sliding past a constantly-changing last-five window. An explicit integer always overrides the auto-scale. |
| `recent_generations:` |  | Maximum number of rows the recent-generation table holds. |
| `recent_messages:` |  | Maximum number of lines the message-log panel holds. |

**Notes**

The instance carries one :class:`rich.live.Live` between
``on_run_start`` and ``on_run_end``. State is reset in
``on_run_start`` so a single instance can be reused across sequential
runs (used in tests).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/evosax.py#L68)</small>
