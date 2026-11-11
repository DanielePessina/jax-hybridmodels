# Profiles: Time-Varying Inputs

## Quick links

- [`Profile`](#profile)
- [`constant_profile`](#constant_profile)
- [`step_profile`](#step_profile)
- [`ramp_profile`](#ramp_profile)
- [`piecewise_linear_profile`](#piecewise_linear_profile)

---

<a id="profile"></a>

### `Profile`

<small>`from jaxhybridmodels.profiles import Profile`</small>

```python
Profile(
    fn: 'Callable[[ArrayLike], Array]',
    jump_ts: 'ArrayLike | None' = None,
) -> None
```

A time profile: call it with ``t``, and read where it jumps or bends.

``jump_ts`` holds the times where the profile or its slope changes, empty
for a smooth one. Concatenate several profiles' ``jump_ts`` to pass them
to one solve.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L44)</small>

---

<a id="constant_profile"></a>

### `constant_profile()`

<small>`from jaxhybridmodels.profiles import constant_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.constant_profile`</small>

```python
constant_profile(value: 'ArrayLike') -> 'Profile'
```

Return ``t -> value``: a profile that never changes.

The degenerate case, included so a code path can treat every
quantity uniformly (a quantity is either a covariate or
``constant_profile(v)`` evaluated at ``t``).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L62)</small>

---

<a id="step_profile"></a>

### `step_profile()`

<small>`from jaxhybridmodels.profiles import step_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.step_profile`</small>

```python
step_profile(*, at: 'ArrayLike', before: 'ArrayLike', after: 'ArrayLike') -> 'Profile'
```

Return ``t -> before if t < at else after``: a step change.

For a reactor, the moment a feed valve opens or a heater switches.
Discontinuous at ``at``, which is its ``jump_ts``. To fit or infer
``at`` itself, pass ``jump_ts`` to the solve: the switch has no
derivative in ``at``, so without them the gradient with respect to it is
exactly 0. With them the solver ends a step on ``at`` and the gradient
comes through the step's end.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L76)</small>

---

<a id="ramp_profile"></a>

### `ramp_profile()`

<small>`from jaxhybridmodels.profiles import ramp_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.ramp_profile`</small>

```python
ramp_profile(
    *,
    start: 'ArrayLike',
    end: 'ArrayLike',
    before: 'ArrayLike',
    after: 'ArrayLike',
) -> Profile
```

Return a flat-ramp-flat profile: ``before`` until ``start``, linear to ``after`` by ``end``.

The reactor heat-up shape: hold at the initial set point, ramp to the
final set point, hold there. Values before ``start`` and after ``end``
are exactly ``before`` and ``after``, the names ``step_profile`` uses
for the same two levels.

``end > start`` is required; equal times would be a discontinuity. The
check runs when the times are host-side values; inside ``jit``/``vmap``
(per-experiment parameters from traced covariates) it is skipped, so
the factory stays trace-safe.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L93)</small>

---

<a id="piecewise_linear_profile"></a>

### `piecewise_linear_profile()`

<small>`from jaxhybridmodels.profiles import piecewise_linear_profile` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.piecewise_linear_profile`</small>

```python
piecewise_linear_profile(knots: 'tuple[tuple[ArrayLike, ArrayLike], ...]') -> 'Profile'
```

Return a piecewise-linear interpolation of ``(t, v)`` knots.

The general shape: any finite profile that is linear between its
knots. Constant on both edges (the first and last values extend
outward). Knots must be strictly increasing in ``t`` and contain at
least two entries. Its slope changes at every knot, so the knot times are
its ``jump_ts``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/profiles.py#L132)</small>
