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

Generalises :func:`soft_logit` to the inverse of any squashing
function. Every such inverse has a pole at each end of the unit
interval, where a hard clip would zero the derivative and silently drop
state-derived sensitivities from the ODE adjoint (R-P2).

Exact in value and derivative inside the band, and C^1 across the
junction, since the continuation uses the inverse's own slope there.
The ``stop_gradient`` on the clamp is load-bearing: without it the
correction term picks up a contribution through the clip and the
interior derivative comes out wrong.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L79)</small>

---

<a id="soft_logit"></a>

### `soft_logit()`

<small>`from hybridmodels.penalties import soft_logit` &nbsp;·&nbsp; also re-exported as `hybridmodels.soft_logit`</small>

```python
soft_logit(s: 'Array', eps: 'float' = 0.001) -> 'Array'
```

``logit(s)``, extended linearly outside ``[eps, 1 - eps]``.

``s`` is a physical value already normalised into ``[0, 1]`` across its
declared box, so this is the step that turns a bounded quantity into an
unbounded latent. Outside the band the result grows linearly instead of
blowing up at the pole, and the derivative is a finite constant instead
of the exact zero ``logit(jnp.clip(s, eps, 1 - eps))`` would give.

:func:`softclip` cannot do this job: its interior error is
``O(1 / beta)`` and ``s`` spans only ``[0, 1]``, so any ``beta`` gentle
enough to keep gradient far outside the box also distorts the middle.

``eps`` sets the continuation slope, roughly ``1 / eps``, and is the one
tuning knob. The default 1e-3 maps a 1% overshoot to ``|z| ~ 10``, a
number a network can still consume; 1e-6 would map it to ``|z| ~ 1e4``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L102)</small>

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
``sigmoid(beta * (x - lo)) - sigmoid(beta * (x - hi))``, analytically in
``(0, 1)``. The interior is reproduced to ``O(1 / beta)``. Larger
``beta`` tracks a hard clip more closely but underflows sooner outside
the box; the default 20 holds interior error below about 0.05 box widths
and keeps usable gradient roughly one width out.

Repairs the near field only. Several widths out the derivative
underflows as a hard clip's does, so pair it with :func:`box_violation`
for unbounded push-back.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L125)</small>

---

<a id="clip_ste"></a>

### `clip_ste()`

<small>`from hybridmodels.penalties import clip_ste` &nbsp;·&nbsp; also re-exported as `hybridmodels.clip_ste`</small>

```python
clip_ste(x: 'Array', lo: 'float | Array', hi: 'float | Array') -> 'Array'
```

Hard-clip on the forward pass, identity on the backward pass.

The straight-through estimator. Use it when downstream code genuinely
requires a feasible number, say a concentration that must not go
negative before a ``log``, while the task loss keeps flowing as though
the clip were not there.

The identity gradient is a deliberate fiction: it propagates whatever
the data loss asks for, including "go further out of bounds", forever.
Pair it with :func:`box_violation` on the pre-clip value for the
restoring force.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L143)</small>

---

<a id="box_violation"></a>

### `box_violation()`

<small>`from hybridmodels.penalties import box_violation` &nbsp;·&nbsp; also re-exported as `hybridmodels.box_violation`</small>

```python
box_violation(x: 'Array', lows: 'Array', highs: 'Array') -> 'Array'
```

Width-normalised squared hinge measuring how far ``x`` falls outside its box.

Zero in value and gradient strictly inside the box, so it never
perturbs the feasible interior. Outside it grows quadratically, giving
a restoring gradient linear in the overshoot, which does not vanish the
way a reparameterised bound's does.

Each component is normalised by its own width ``high - low`` so one
penalty weight works across channels. Bounds here run from fractions of
a unit to hundreds of kelvin, and an unnormalised hinge would let the
widest channel dominate on units alone.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `x` | `Array` | Values in physical units; broadcast against ``lows`` / ``highs``. |
| `lows, highs` | `Array` | Per-component box edges, as produced by ``BoundScaler._lows_highs``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | Scalar sum of squared fractional violations. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L159)</small>
