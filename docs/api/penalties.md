# Penalties: Gradient-Safe Bound Handling

## Quick links

- [`soft_logit`](#soft_logit)
- [`softclip`](#softclip)
- [`clip_ste`](#clip_ste)
- [`box_violation`](#box_violation)

---

<a id="soft_logit"></a>

### `soft_logit()`

<small>`from hybridmodels.penalties import soft_logit` &nbsp;·&nbsp; also re-exported as `hybridmodels.soft_logit`</small>

```python
soft_logit(s: 'Array', eps: 'float' = 0.001) -> 'Array'
```

``logit(s)``, extended linearly outside ``[eps, 1 - eps]``.

Exact — value *and* derivative — for ``s`` inside the threshold band,
and ``C^1`` across the junction, because the continuation uses
``logit`` 's own slope at the crossing point. Outside the band the
result therefore grows linearly instead of blowing up at the pole, and
the derivative is a finite constant instead of zero.

This replaces the ``logit(jnp.clip(s, eps, 1 - eps))`` idiom, whose
derivative outside the band is exactly zero — see the module docstring
for why a mid-graph zero derivative is a silent correctness bug rather
than a numerical nuisance.

A plain :func:`softclip` cannot be used for this. Its error in the
interior is ``O(1 / beta)`` in the units of ``s``, and ``s`` here is
normalised to ``[0, 1]``, so any ``beta`` gentle enough to retain
gradient far outside the box also visibly distorts the middle of it.
The two-regime construction sidesteps the trade-off entirely: no
interior distortion at any threshold.

``eps`` sets the continuation slope, which is ``1 / (eps * (1 - eps))``
— roughly ``1 / eps``. That is the one real tuning knob: too small and
a modest excursion maps to an enormous latent (``eps = 1e-6`` sends a
1% overshoot to ``|z| ~ 1e4``, which then saturates or overflows the
inner network); too large and the exact-interior band shrinks. The
default ``1e-3`` maps a 1% overshoot to ``|z| ~ 10`` — firmly outside
the sigmoid's linear region, so the push-back is felt, but still a
number a network can consume.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L83)</small>

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

``beta`` trades interior fidelity against far-field gradient: larger
values track a hard clip more closely but decay to underflow sooner
outside the box. The default of ``20`` keeps the interior error below
roughly ``0.05`` box widths while retaining usable gradient about one
width out.

This is a *near-field* repair only. Several widths outside the box the
derivative underflows just as a hard clip's does, so ``softclip``
should be paired with :func:`box_violation` whenever the input can
stray far — the hinge is what supplies unbounded push-back.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L123)</small>

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
the data loss asks for, including "keep going further out of bounds",
indefinitely — a straight-through clip *never* pushes back on its own.
Pair it with :func:`box_violation` on the pre-clip value, which is the
term that actually supplies the restoring force.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L146)</small>

---

<a id="box_violation"></a>

### `box_violation()`

<small>`from hybridmodels.penalties import box_violation` &nbsp;·&nbsp; also re-exported as `hybridmodels.box_violation`</small>

```python
box_violation(x: 'Array', lows: 'Array', highs: 'Array') -> 'Array'
```

Width-normalised squared hinge measuring how far ``x`` falls outside its box.

Returns a scalar; exactly zero (value *and* gradient) strictly inside
the box, so the penalty never perturbs the feasible interior. Outside,
it grows quadratically, giving a restoring gradient that is linear in
the overshoot and therefore does not vanish the way a reparameterised
bound's does.

Normalising each component by its own width ``high - low`` is what
makes a single penalty weight portable: bounds in this package range
from fractions of a unit to hundreds of kelvin, and an unnormalised
hinge would let the widest channel dominate the term purely through
its units.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `x` | `Array` | Values in physical units; broadcast against ``lows`` / ``highs``. |
| `lows, highs` | `Array` | Per-component box edges, as produced by ``BoundScaler._lows_highs``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | Scalar sum of squared fractional violations. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L163)</small>
