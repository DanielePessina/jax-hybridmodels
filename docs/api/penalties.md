# Penalties: Gradient-Safe Bound Handling

## Quick links

- [`PenaltyPointSource`](#penaltypointsource)
- [`attach_penalty_state`](#attach_penalty_state)
- [`bound_penalty`](#bound_penalty)
- [`box_grid`](#box_grid)
- [`box_violation`](#box_violation)
- [`clip_ste`](#clip_ste)
- [`data_penalty_points`](#data_penalty_points)
- [`length_mask_keep`](#length_mask_keep)
- [`penalty_integral`](#penalty_integral)
- [`penalty_vector_field`](#penalty_vector_field)
- [`select_penalty_points`](#select_penalty_points)
- [`soft_inverse`](#soft_inverse)
- [`soft_logit`](#soft_logit)
- [`softclip`](#softclip)
- [`strip_penalty_state`](#strip_penalty_state)
- [`trajectory_saturation_penalty`](#trajectory_saturation_penalty)
- [`validate_penalty_points`](#validate_penalty_points)

---

<a id="penaltypointsource"></a>

### `PenaltyPointSource`

<small>`from hybridmodels.penalties import PenaltyPointSource` &nbsp;·&nbsp; also re-exported as `hybridmodels.PenaltyPointSource`</small>

```python
PenaltyPointSource(
    points: ForwardRef('Array'),
    cell_ts: ForwardRef('Array'),
    cell_T: ForwardRef('Array'),
)
```

One ``BoundedPredictor`` leaf's gathered measured points.

Produced by :func:`data_penalty_points` for leaves whose ``input_keys``
all resolve to dataset covariates. One entry per *observed cell*: the
input vector the loss actually sees at that cell, in ``input_keys``
column order, in physical units.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `points` | `Float[Array, "G n_inputs"]` | The measured input vectors. |
| `cell_ts` | `Int[Array, " G"]` | Timestamp index of each cell inside its own experiment. |
| `cell_T` | `Int[Array, " G"]` | Length of that experiment's time grid. ``cell_ts`` and ``cell_T`` let :func:`length_mask_keep` apply the loss's prefix mask to the penalty, so the two never disagree about which points are live. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L212)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L561)</small>

---

<a id="bound_penalty"></a>

### `bound_penalty()`

<small>`from hybridmodels.penalties import bound_penalty` &nbsp;·&nbsp; also re-exported as `hybridmodels.bound_penalty`</small>

```python
bound_penalty(predictors: 'Any', points: 'tuple[Array, ...]') -> 'Array'
```

Mean output-squash saturation over every ``BoundedPredictor`` in a pytree.

For each leaf, evaluates ``inner(in_scaler.to_latent(x))`` across the
leaf's point set and charges
:meth:`~hybridmodels.predictors.BoundScaler.saturation` on the
resulting latents. Leaves are summed.

``points`` is positional, one ``[G, n_inputs]`` array per leaf in
traversal order: measured points from :func:`data_penalty_points`,
user-supplied penalty-only points, or a :func:`box_grid` sweep. An
empty per-leaf array contributes exactly zero, so a leaf the penalty
cannot reach does not NaN a run.

The default way to penalise bound behaviour here, and deliberately
evaluated at fixed input points rather than along a solve: saturation
is a property of the predictor as a function, whatever any particular
solve does. So it needs no cooperation from ``simulate_fn``,
``predict_bucket`` or the loss protocol, behaves the same inside a
vector field or above one, and handles arbitrary nesting by walking
leaves.

That cuts both ways. It reports saturation wherever the point set
reaches, including regions no training trajectory visited when the
points say so (a ``box_grid`` sweep, user extras); for "did *this*
solve push an input out of range" the trajectory-aware penalty
(ADR-0009) is the right instrument.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` | `PyTree[eqx.Module]` | Any pytree shape. Only ``BoundedPredictor`` leaves contribute. |
| `points` | `tuple[Array, ...]` | One ``[G, n_inputs]`` array per leaf, in traversal order: the output of :func:`data_penalty_points` (with any user extras concatenated), user penalty-only points, or a :func:`box_grid` sweep. An empty per-leaf array contributes zero. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | Non-negative scalar. Exactly zero when no leaf saturates. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L468)</small>

---

<a id="box_grid"></a>

### `box_grid()`

<small>`from hybridmodels.penalties import box_grid` &nbsp;·&nbsp; also re-exported as `hybridmodels.box_grid`</small>

```python
box_grid(in_scaler: 'Any', n_per_dim: 'int' = 5) -> 'Array'
```

Build a deterministic sweep of an input box, uniform in *warped* coordinates.

The collocation-as-extension recipe: a tensor product of ``n_per_dim``
evenly spaced points along every dimension of ``in_scaler``'s box, in
the box's *warped* coordinates, mapped back to physical units. A linear
warp therefore reproduces a plain physical ``linspace`` sweep
bit-for-bit, while ``log`` / ``log10`` warps cover each decade evenly
instead of starving the low end of the box.

``n_per_dim`` must be at least 2 so both edges of every input box are
represented. Deterministic: it depends only on the scaler's static
``bounds`` and ``warp``, so ``restore_best`` compares raw loss values
across steps with no resampling noise.

Point count grows exponentially in input dimension. Predictors here take
two or three inputs, so ``n_per_dim=5`` is 25 to 125 forward passes and
negligible beside an ODE solve. Lower it if that stops holding.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `in_scaler` | `BoundScaler` | The input scaler of the predictor the sweep is for (``leaf.in_scaler``). |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | ``[n_per_dim ** n_inputs, n_inputs]`` physical points, positional in the scaler's input dimension order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L237)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L181)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L165)</small>

---

<a id="data_penalty_points"></a>

### `data_penalty_points()`

<small>`from hybridmodels.penalties import data_penalty_points` &nbsp;·&nbsp; also re-exported as `hybridmodels.data_penalty_points`</small>

```python
data_penalty_points(
    predictors: 'Any',
    dataset: 'Dataset',
) -> tuple[PenaltyPointSource | None, ...]
```

Gather one :class:`PenaltyPointSource` per ``BoundedPredictor`` leaf.

A leaf is *resolvable* when every ``input_keys`` name is a dataset
covariate (in every bucket). Its measured points are then the input
vectors at the observed cells — the cells the loss actually charges —
with a ``None`` entry for an unresolvable leaf. A resolvable leaf whose
dataset holds no observed cells (all probes) yields a source with zero
points, which contributes nothing until extras are added.

Call this once on the host before the training loop: the points depend
only on the dataset and the leaf's static ``input_keys``.

Covariates must be scalar per experiment (constant in time, the v1
contract); a per-experiment covariate with extra dimensions has no
unambiguous column to feed an input key and raises.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L286)</small>

---

<a id="length_mask_keep"></a>

### `length_mask_keep()`

<small>`from hybridmodels.penalties import length_mask_keep` &nbsp;·&nbsp; also re-exported as `hybridmodels.length_mask_keep`</small>

```python
length_mask_keep(source: 'PenaltyPointSource', length_mask_fraction: 'float') -> 'Array'
```

Boolean keep-vector over a source's cells, matching the loss's prefix mask.

Applies the same prefix semantics :func:`~hybridmodels.training.kernels.apply_length_mask`
gives the loss: a cell is kept when its timestamp index is below
``ceil(T * fraction)``, clamped at 1 so a phase never scores nothing.
The penalty therefore follows the curriculum exactly, disagreeing with
the loss about which points are live only by mistake.

``length_mask_fraction`` must be a Python float, not a traced array:
the keep-vector is used to *index* the gathered points, and JAX rejects
boolean indexing with tracers. The penalty kernel retraces only when
the fraction changes (a phase boundary), never per step.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L352)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L619)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L574)</small>

---

<a id="select_penalty_points"></a>

### `select_penalty_points()`

<small>`from hybridmodels.penalties import select_penalty_points` &nbsp;·&nbsp; also re-exported as `hybridmodels.select_penalty_points`</small>

```python
select_penalty_points(
    sources: 'tuple[PenaltyPointSource | None, ...]',
    extras: 'tuple[Array, ...]',
    length_mask_fraction: 'float',
) -> tuple[Array, ...]
```

Per-leaf point arrays for one phase: length-mask-kept cells ∪ extras.

``sources`` is the per-leaf output of :func:`data_penalty_points`
(``None`` for unresolvable leaves); ``extras`` the user-supplied
penalty-only points, positional per leaf. Returns one ``[G, n_inputs]``
array per leaf in traversal order: the measured cells kept by
:func:`length_mask_keep` concatenated with that leaf's extras.

This is host-side selection: the keep-vector must be concrete to index
the gathered points, so call it with the phase's Python float outside
any jit (the stock trainers do, once per phase). The returned arrays
are what a ``penalty_step`` receives.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L434)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L101)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L124)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L147)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L608)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L632)</small>

---

<a id="validate_penalty_points"></a>

### `validate_penalty_points()`

<small>`from hybridmodels.penalties import validate_penalty_points` &nbsp;·&nbsp; also re-exported as `hybridmodels.validate_penalty_points`</small>

```python
validate_penalty_points(
    predictors: 'Any',
    sources: 'tuple[PenaltyPointSource | None, ...]',
    extras: 'tuple[Array, ...]',
    enabled: 'bool',
) -> None
```

Raise when an enabled penalty has a leaf no point set reaches.

``sources`` is the per-leaf output of :func:`data_penalty_points`
(``None`` for unresolvable leaves); ``extras`` the user-supplied
penalty-only points, positional per leaf. ``enabled`` is whether any
phase charges the penalty.

With the penalty on, every ``BoundedPredictor`` leaf must be covered by
measured points, extras, or both; an uncovered leaf would otherwise be
a silent no-penalty, the failure mode this penalty exists to prevent.
For embedded predictors (state-derived inputs) the trajectory penalty
(ADR-0009) is the instrument, and the error says so.

Point-shape mismatches are checked unconditionally, so a user cannot
carry a silently-wrong extras array into a run that later enables it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/penalties.py#L373)</small>
