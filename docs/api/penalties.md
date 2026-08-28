# Penalties: Gradient-Safe Bound Handling

## Quick links

- [`attach_penalty_state`](#attach_penalty_state)
- [`bound_penalty`](#bound_penalty)
- [`box_violation`](#box_violation)
- [`clip_ste`](#clip_ste)
- [`collocation_grids`](#collocation_grids)
- [`penalty_integral`](#penalty_integral)
- [`penalty_vector_field`](#penalty_vector_field)
- [`soft_inverse`](#soft_inverse)
- [`soft_logit`](#soft_logit)
- [`softclip`](#softclip)
- [`strip_penalty_state`](#strip_penalty_state)
- [`trajectory_saturation_penalty`](#trajectory_saturation_penalty)

---

<a id="attach_penalty_state"></a>

### `attach_penalty_state()`

<small>`from hybridmodels.penalties import attach_penalty_state` &nbsp;·&nbsp; also re-exported as `hybridmodels.attach_penalty_state`</small>

```python
attach_penalty_state(y0: 'Array', n: 'int' = 1) -> 'Array'
```

Append ``n`` zero-valued penalty accumulators to ``y0``.

The first step of the trajectory-penalty recipe: the ODE state widens
from ``[S]`` to ``[S + n]``, where the trailing components are
integrated penalty rates supplied by :func:`penalty_vector_field`.
``y0_fn`` should return ``attach_penalty_state(physics_y0, n)``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L307)</small>

---

<a id="bound_penalty"></a>

### `bound_penalty()`

<small>`from hybridmodels.penalties import bound_penalty` &nbsp;·&nbsp; also re-exported as `hybridmodels.bound_penalty`</small>

```python
bound_penalty(predictors: 'Any', grids: 'tuple[Array, ...]') -> 'Array'
```

Mean output-squash saturation over every ``BoundedPredictor`` in a pytree.

For each leaf, evaluates ``inner(in_scaler.to_latent(x))`` across that
leaf's collocation grid and charges
:meth:`~hybridmodels.predictors.BoundScaler.saturation` on the
resulting latents. Leaves are summed.

The default way to penalise bound behaviour here, and deliberately
trajectory-blind: saturation is a property of the predictor as a
function on its declared box, whatever any particular solve does. So it
needs no cooperation from ``simulate_fn``, ``predict_bucket`` or the
loss protocol, behaves the same inside a vector field or above one, and
handles arbitrary nesting by walking leaves.

That cuts both ways. It reports saturation anywhere in the declared box,
including regions no training trajectory visited, which catches
extrapolation failure early. It cannot answer "did this solve push an
input out of range", which needs the penalty computed where the state
actually went.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` | `PyTree[eqx.Module]` | Any pytree shape. Only ``BoundedPredictor`` leaves contribute. |
| `grids` | `tuple[Array, ...]` | Output of :func:`collocation_grids` for this same pytree. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | Non-negative scalar. Exactly zero when no leaf saturates. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L231)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L164)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L148)</small>

---

<a id="collocation_grids"></a>

### `collocation_grids()`

<small>`from hybridmodels.penalties import collocation_grids` &nbsp;·&nbsp; also re-exported as `hybridmodels.collocation_grids`</small>

```python
collocation_grids(predictors: 'Any', n_per_dim: 'int' = 5) -> 'tuple[Array, ...]'
```

Build one *collocation grid* per ``BoundedPredictor`` leaf of ``predictors``.

A collocation grid is a fixed set of input points at which a predictor
is evaluated for inspection, chosen up front rather than taken from any
trajectory. Each grid is a tensor product of ``n_per_dim`` evenly spaced
points along every dimension of that predictor's ``in_scaler.bounds``,
shape ``[n_per_dim ** n_inputs, n_inputs]``.

Call this once on the host before the training loop: the grids depend
only on static ``bounds``, so rebuilding them per step compiles work
that computes a constant. The returned tuple is positional and matches
the leaf order :func:`bound_penalty` walks.

The grid is deterministic rather than sampled because ``restore_best``
compares raw loss values across steps, and a resampled penalty could
pick a "best" that drew an easy sample.

Point count grows exponentially in input dimension. Predictors here take
two or three inputs, so ``n_per_dim=5`` is 25 to 125 forward passes and
negligible beside an ODE solve. Lower it if that stops holding.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Array, ...]` |  | One ``[G, n_inputs]`` grid per ``BoundedPredictor`` leaf, in traversal order. Empty if the pytree holds no such leaf. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L195)</small>

---

<a id="penalty_integral"></a>

### `penalty_integral()`

<small>`from hybridmodels.penalties import penalty_integral` &nbsp;·&nbsp; also re-exported as `hybridmodels.penalty_integral`</small>

```python
penalty_integral(state: 'Array', n: 'int' = 1) -> 'Array'
```

The accumulated (time-integrated) penalty values at the trajectory's end.

``state`` is the full state trajectory ``[..., T, S + n]`` as produced by
a :func:`penalty_vector_field` solve. Returns the trailing ``n``
components at the final time, ``[..., n]``. The training hook charges
these; divide by the time span to get the time-mean instead of the
integral.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L355)</small>

---

<a id="penalty_vector_field"></a>

### `penalty_vector_field()`

<small>`from hybridmodels.penalties import penalty_vector_field` &nbsp;·&nbsp; also re-exported as `hybridmodels.penalty_vector_field`</small>

```python
penalty_vector_field(
    base_rhs: 'Callable[[Array, Array, Any], Array]',
    penalty_rhs: 'Callable[[Array, Array, Any], Array]',
) -> Callable[[Array, Array, Any], Array]
```

Wrap a physics vector field with per-call penalty rates.

Returns ``(t, y, args) -> [physics_dot, penalty_rates]``. ``base_rhs``
is the user's vector field on the physical components; ``penalty_rhs``
returns the ``n`` penalty rates (e.g. ``[saturation(z),
input_violation(x)]``) at the current state. The solver integrates both,
so the trailing accumulators carry the *time-integral* of the penalty
along the trajectory.

The rates must read the physical components (typically ``y[:-n]``) and
are closed over the user's predictors — only the user knows the latent
``z`` or input ``x`` of an embedded predictor at call time.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L319)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L84)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L107)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L130)</small>

---

<a id="strip_penalty_state"></a>

### `strip_penalty_state()`

<small>`from hybridmodels.penalties import strip_penalty_state` &nbsp;·&nbsp; also re-exported as `hybridmodels.strip_penalty_state`</small>

```python
strip_penalty_state(state: 'Array', n: 'int' = 1) -> 'Array'
```

Drop the trailing ``n`` penalty accumulators from a full-state trajectory.

Use in ``state_to_output``: the loss and the observed channels should
see only the physics, not the integrated penalty components. ``state``
is ``[..., S + n]``; the result is ``[..., S]``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L345)</small>

---

<a id="trajectory_saturation_penalty"></a>

### `trajectory_saturation_penalty()`

<small>`from hybridmodels.penalties import trajectory_saturation_penalty` &nbsp;·&nbsp; also re-exported as `hybridmodels.trajectory_saturation_penalty`</small>

```python
trajectory_saturation_penalty(state: 'Array', out_scaler: 'Any') -> 'Array'
```

Sum over time of output saturation for a predictor whose output *is* the state.

For a **parallel** hybrid model — the predictor is outside the solver and
its output is a predicted channel — the full state already holds the
physical outputs. Invert them back to latents with the predictor's
``out_scaler`` and charge :meth:`BoundScaler.saturation` at every time
step, summed over time. This is the trajectory-aware counterpart of
``bound_penalty`` for the hoisted case: it fires only where the model
actually predicted, not across a synthetic grid.

``state`` is ``[..., T, D]`` (or ``[..., T, S]`` projected to the
predictor's channel); ``out_scaler`` is the ``BoundScaler`` whose
``from_latent`` produced those outputs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L367)</small>
