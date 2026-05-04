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
    log_every: 'int' = 1,
    recent_losses: 'int' = 5,
    recent_messages: 'int' = 5,
) -> None
```

Live Rich dashboard satisfying ``hybridmodels.ui.base.TrainingUI``.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `console:` |  | Optional :class:`rich.console.Console`. ``None`` constructs a default ``Console()``. Tests typically pass a recording console (``Console(record=True, force_terminal=False, ...)``) so the rendered final state can be asserted on. |
| `log_every:` |  | Step throttle for the recent-loss table. ``log_every=1`` records every step, ``log_every=k`` records steps where ``step_idx % k == 0``. |
| `recent_losses:` |  | Maximum number of rows the recent-loss table holds. |
| `recent_messages:` |  | Maximum number of lines the message-log panel holds. |

**Notes**

The instance carries one :class:`rich.live.Live` between
``on_run_start`` and ``on_run_end``. Any state mutation outside that
window updates the model only; the next ``on_run_start`` rebuilds Live
afresh, so a single ``RichTrainingUI`` instance can be reused for
sequential runs (used in tests).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/optax.py#L55)</small>

---

<a id="richevosaxui"></a>

### `RichEvosaxUI`

<small>`from hybridmodels.ui import RichEvosaxUI` &nbsp;·&nbsp; also re-exported as `hybridmodels.RichEvosaxUI`</small>

```python
RichEvosaxUI(
    console: 'Console | None' = None,
    log_every: 'int' = 1,
    recent_generations: 'int' = 5,
    recent_messages: 'int' = 5,
) -> None
```

Live Rich dashboard satisfying ``hybridmodels.ui.base.EvosaxUI``.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `console:` |  | Optional :class:`rich.console.Console`. ``None`` constructs a default ``Console()``. Tests typically pass a recording console (``Console(record=True, force_terminal=False, ...)``) so the rendered final state can be asserted on. |
| `log_every:` |  | Throttle for the recent-generation table. ``log_every=1`` records every generation, ``log_every=k`` records generations where ``gen_idx % k == 0``. |
| `recent_generations:` |  | Maximum number of rows the recent-generation table holds. |
| `recent_messages:` |  | Maximum number of lines the message-log panel holds. |

**Notes**

The instance carries one :class:`rich.live.Live` between
``on_run_start`` and ``on_run_end``. State is reset in
``on_run_start`` so a single instance can be reused across sequential
runs (used in tests).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/ui/evosax.py#L67)</small>
