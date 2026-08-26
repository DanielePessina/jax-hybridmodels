# Penalties: Gradient-Safe Bound Handling

## Quick links

- [`soft_inverse`](#soft_inverse)
- [`soft_logit`](#soft_logit)
- [`softclip`](#softclip)
- [`clip_ste`](#clip_ste)
- [`box_violation`](#box_violation)

---

<a id="soft_inverse"></a>

### `soft_inverse()`

<small>`from hybridmodels.penalties import soft_inverse` &nbsp;·&nbsp; also re-exported as `hybridmodels.soft_inverse`</small>

```python
soft_inverse(
    s: 'Array',
    inverse: 'Callable[[Array], Array]',
    inverse_slope: 'Callable[[Array], Array]',
    eps: 'float' = 0.001,
) -> Array
```

``inverse(s)``, extended linearly outside ``[eps, 1 - eps]``.

The generalisation of :func:`soft_logit` to any squash's inverse. Every
candidate inverse has a pole at each end of the unit interval, and the
reason a hard clip is unacceptable there does not depend on which
squash it is: a mid-graph zero derivative propagates to every upstream
parameter and drops state-derived sensitivities from the ODE adjoint
without raising (R-P2).

Exact in value and derivative inside the band, and C^1 across the
junction because the continuation uses the inverse's own slope at the
crossing. The ``stop_gradient`` on the clamp is load-bearing. Without
it the correction term picks up a contribution through the clip and the
interior derivative comes out wrong.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L90)</small>

---

<a id="soft_logit"></a>

### `soft_logit()`

<small>`from hybridmodels.penalties import soft_logit` &nbsp;·&nbsp; also re-exported as `hybridmodels.soft_logit`</small>

```python
soft_logit(s: 'Array', eps: 'float' = 0.001) -> 'Array'
```

``logit(s)``, extended linearly outside ``[eps, 1 - eps]``.

Exact in value and derivative for ``s`` inside the band, and C^1 across
the junction, since the continuation uses logit's own slope at the
crossing. Outside the band the result grows linearly instead of blowing
up at the pole, and the derivative is a finite constant instead of zero.

Replaces the ``logit(jnp.clip(s, eps, 1 - eps))`` idiom, whose
derivative outside the band is exactly zero. The module docstring
explains why that is a silent correctness bug.

:func:`softclip` cannot do this job. Its interior error is
``O(1 / beta)`` in the units of ``s``, and ``s`` is normalised to
``[0, 1]``, so any ``beta`` gentle enough to keep gradient far outside
the box also distorts the middle of it. Two regimes avoid the trade
entirely, with no interior distortion at any threshold.

``eps`` sets the continuation slope, ``1 / (eps * (1 - eps))``, roughly
``1 / eps``. It is the one tuning knob. At ``eps = 1e-6`` a 1% overshoot
maps to ``|z| ~ 1e4``, which saturates or overflows the inner network.
Larger ``eps`` shrinks the exact band. The default 1e-3 maps a 1%
overshoot to ``|z| ~ 10``, outside the sigmoid's linear region but still
a number a network can consume.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L115)</small>

---

<a id="softclip"></a>

### `softclip()`

<small>`from hybridmodels.penalties import softclip` &nbsp;·&nbsp; also re-exported as `hybridmodels.softclip`</small>

```python
softclip(
    x: 'Array',
    lo: 'float | Array',
    hi: 'float | Array',
    beta: 'float' = 20.0,
) -> Array
```

Smoothly clamp ``x`` into ``[lo, hi]`` with a derivative that never hits zero.

Built from two softplus shoulders, so the derivative is
``sigmoid(beta * (x - lo)) - sigmoid(beta * (x - hi))``, which lies in
``(0, 1)`` analytically. The interior is reproduced to ``O(1 / beta)``
and the output asymptotes to the bounds rather than meeting them.

Larger ``beta`` tracks a hard clip more closely but underflows sooner
outside the box. The default 20 holds interior error below about 0.05
box widths and keeps usable gradient roughly one width out.

This repairs the near field only. Several widths out the derivative
underflows just as a hard clip's does. Pair it with
:func:`box_violation`, which supplies the unbounded push-back.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L146)</small>

---

<a id="clip_ste"></a>

### `clip_ste()`

<small>`from hybridmodels.penalties import clip_ste` &nbsp;·&nbsp; also re-exported as `hybridmodels.clip_ste`</small>

```python
clip_ste(x: 'Array', lo: 'float | Array', hi: 'float | Array') -> 'Array'
```

Hard-clip on the forward pass, identity on the backward pass.

The straight-through estimator: use it when downstream code genuinely
requires a feasible number (a concentration that must not go negative
before a ``log``, say) but the task loss should keep flowing as though
the clip were not there.

The identity gradient is a deliberate fiction. It propagates whatever
the data loss asks for, including "go further out of bounds", forever.
A straight-through clip never pushes back on its own. Pair it with
:func:`box_violation` on the pre-clip value for the restoring force.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L166)</small>

---

<a id="box_violation"></a>

### `box_violation()`

<small>`from hybridmodels.penalties import box_violation` &nbsp;·&nbsp; also re-exported as `hybridmodels.box_violation`</small>

```python
box_violation(x: 'Array', lows: 'Array', highs: 'Array') -> 'Array'
```

Width-normalised squared hinge measuring how far ``x`` falls outside its box.

Returns a scalar. Zero in value and gradient strictly inside the box,
so it never perturbs the feasible interior. Outside it grows
quadratically, giving a restoring gradient linear in the overshoot,
which does not vanish the way a reparameterised bound's does.

Each component is normalised by its own width ``high - low`` so one
penalty weight works across channels. Bounds in this package run from
fractions of a unit to hundreds of kelvin, and an unnormalised hinge
would let the widest channel dominate on units alone.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `x` | `Array` | Values in physical units; broadcast against ``lows`` / ``highs``. |
| `lows, highs` | `Array` | Per-component box edges, as produced by ``BoundScaler._lows_highs``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | Scalar sum of squared fractional violations. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L182)</small>
