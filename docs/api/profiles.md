# Profiles: Time-Varying Inputs

## Quick links

- [`constant_profile`](#constant_profile)
- [`step_profile`](#step_profile)
- [`ramp_profile`](#ramp_profile)
- [`piecewise_linear_profile`](#piecewise_linear_profile)

---

<a id="constant_profile"></a>

### `constant_profile()`

<small>`from jaxhybridmodels.profiles import constant_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.constant_profile`</small>

```python
constant_profile(value: 'float') -> 'Callable[[Array], Array]'
```

Return ``t -> value``: a profile that never changes.

The degenerate case, included so a code path can treat every
quantity uniformly (a quantity is either a covariate or
``constant_profile(v)`` evaluated at ``t``).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L30)</small>

---

<a id="step_profile"></a>

### `step_profile()`

<small>`from jaxhybridmodels.profiles import step_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.step_profile`</small>

```python
step_profile(
    before: 'float',
    after: 'float',
    jump_at: 'float',
) -> Callable[[Array], Array]
```

Return ``t -> before if t < jump_at else after``: a step change.

For a reactor, the moment a feed valve opens or a heater switches.
Discontinuous at ``jump_at``; an adaptive solver sees a kink, so
place the jump at a known event time or accept a short transient.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L44)</small>

---

<a id="ramp_profile"></a>

### `ramp_profile()`

<small>`from jaxhybridmodels.profiles import ramp_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.ramp_profile`</small>

```python
ramp_profile(
    t0: 'float',
    t1: 'float',
    v0: 'float',
    v1: 'float',
) -> Callable[[Array], Array]
```

Return a flat-ramp-flat profile: ``v0`` until ``t0``, linear to ``v1`` by ``t1``, then flat.

The reactor heat-up shape: hold at the initial set point, ramp to the
final set point, hold there. Values before ``t0`` and after ``t1``
are exactly the edge values (the two flat profiles on either edge).

``t1 > t0`` is required; equal times would be a discontinuity. The
check runs when the times are host-side values; inside ``jit``/``vmap``
(per-experiment parameters from traced covariates) it is skipped, so
the factory stays trace-safe.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L58)</small>

---

<a id="piecewise_linear_profile"></a>

### `piecewise_linear_profile()`

<small>`from jaxhybridmodels.profiles import piecewise_linear_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.piecewise_linear_profile`</small>

```python
piecewise_linear_profile(
    knots: 'tuple[tuple[float, float], ...]',
) -> Callable[[Array], Array]
```

Return a piecewise-linear interpolation of ``(t, v)`` knots.

The general shape: any finite profile that is linear between its
knots. Constant on both edges (the first and last values extend
outward). Knots must be strictly increasing in ``t`` and contain at
least two entries.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L94)</small>
