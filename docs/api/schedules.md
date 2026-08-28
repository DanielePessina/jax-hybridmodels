# Schedules: Epoch-Scaled Annealing

## Quick links

- [`annealing_schedule`](#annealing_schedule)

---

<a id="annealing_schedule"></a>

### `annealing_schedule()`

<small>`from hybridmodels.schedules import annealing_schedule` &nbsp;·&nbsp; also re-exported as `hybridmodels.annealing_schedule`</small>

```python
annealing_schedule(
    kind: 'str' = 'cosine',
    total_epochs: 'int',
    init_value: 'float' = 1.0,
    end_value: 'float' = 0.0,
    warmup_epochs: 'int' = 0,
) -> optax.Schedule
```

Return ``schedule(step: int) -> float``, a multiplier over ``[0, total_epochs]``.

Built on optax's own schedule helpers, so the returned callable is a
plain pure function of the integer step count: ``jit``-safe, usable
inside a traced loop. Every kind lands exactly on ``end_value`` at
``step == total_epochs`` and stays there afterwards, so the run length
is genuinely baked in.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `kind` | `str` | One of ``"cosine"`` (default), ``"linear"``, ``"warmup_cosine"``, ``"exponential"``. The exponential kind decays geometrically with the per-epoch rate *derived* from ``(init_value, end_value, total_epochs)``, so its end point and run length are honoured like every other kind. |
| `total_epochs` | `int` | Run length in steps (one step == one epoch). Must be at least 1. |
| `init_value` | `float` | Value at ``step=0``, except ``"warmup_cosine"`` where it is the **peak** the schedule rises to after ``warmup_epochs``. Must be positive. |
| `end_value` | `float` | Value at ``step=total_epochs``, where every kind arrives. Must lie in ``[0, init_value]``. |
| `warmup_epochs` | `int` | ``"warmup_cosine"`` only: steps from 0 to ``init_value`` before the decay. Must lie in ``[0, total_epochs)``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `optax.Schedule` |  | ``schedule(step)`` in ``[end_value, init_value]``. Compose as a multiplier: ``lr = base_lr * schedule(step)``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/schedules.py#L36)</small>
