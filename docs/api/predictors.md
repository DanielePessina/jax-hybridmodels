# Predictors: Trainable Components

A predictor is an Equinox module with an `Array → Array` call signature. [`BoundedPredictor`](#boundedpredictor) composes an input scaler, an inner predictor, and an output scaler. Any PyTree of predictor modules is accepted; the conventional single-predictor container is `(predictor,)`.

## Quick links

- [`Predictor`](#predictor)
- [`BoundScaler`](#boundscaler)
- [`BoundedPredictor`](#boundedpredictor)
- [`Parameter`](#parameter)
- [`Constant`](#constant)
- [`MLPPredictor`](#mlppredictor)
- [`KANPredictor`](#kanpredictor)
- [`NeuralNPolynomial`](#neuralnpolynomial)
- [`reinitialize_with_key`](#reinitialize_with_key)
- [`reinitialize_pytree_with_key`](#reinitialize_pytree_with_key)

---

<a id="predictor"></a>

### `Predictor`

<small>`from jaxhybridmodels.predictors import Predictor` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.Predictor`</small>

```python
Predictor() -> None
```

Abstract marker for a trainable ``Array -> Array`` module.

Carries no behaviour. It exists so the rest of the framework can say
"this leaf is a trainable function approximator" and so composition
wrappers have one type to accept. The base ``__call__`` raises.

To add your own family, subclass ``Predictor``, declare trainable
arrays as ordinary fields and hyperparameters as
``eqx.field(static=True)``, and implement
``__call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]``.
Dynamic leaves must be JAX float arrays and static fields must be
JSON-encodable, so the module round-trips through
``eqx.tree_serialise_leaves``. Optionally implement
``initialized_with_key(self, key) -> Self`` to control how the
tournament restarts your weights; without it,
[`reinitialize_with_key`](/api/predictors#reinitialize_with_key) replaces every float leaf with a
standard-normal sample, which skews any considered init scheme.

Do not subclass to add bound handling or named inputs.
``BoundedPredictor`` supplies both by composition.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L69)</small>

---

<a id="boundscaler"></a>

### `BoundScaler`

<small>`from jaxhybridmodels.predictors import BoundScaler` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.BoundScaler`</small>

```python
BoundScaler(
    bounds: 'Any',
    transform: 'str' = 'sigmoid',
    temperature: 'Any' = 1.0,
    warp: 'str' = 'linear',
    logit_eps: 'float' = 0.001,
    z_knee: 'float | None' = None,
) -> None
```

Two-way map between a physical range ``[low, high]`` and an unbounded latent.

**Why this exists**

Physical quantities have ranges, and an ODE solver handed a value
outside one either fails or returns nonsense. An optimiser knows
none of that; it proposes whatever number lowers the loss. Clipping
the proposal is not usable, because a clip has exactly zero
derivative outside the range, so a parameter that leaves the box has
no gradient to pull it back. This scaler reparameterises instead: the
inner predictor reads and writes a *latent* value, any real number,
and the scaler squashes it into the physical range. An out-of-range
physical value has no latent, so there is nothing to clip.

**Vocabulary**

latent
    The unbounded real number the inner predictor works in. Written
    ``z`` below.
physical
    The value in the units the simulator uses. Always inside
    ``[low, high]``.
warp
    Decides what "halfway between the bounds" means. ``"linear"``
    puts the midpoint of ``(1e-6, 1e2)`` at 50 and collapses the
    eight-decade low end into a sliver; ``"log10"`` puts it at
    ``1e-2``. Bounds are declared in physical units either way.
    Name-keyed registry in ``transforms.py``.
squash
    The map from latent onto ``(0, 1)``, applied before the affine
    rescale onto the box. ``"sigmoid"`` (default), ``"algebraic"``,
    ``"softsign"``. Same module, ``BOUND_TRANSFORMS``. Stored as the
    ``transform`` field.
temperature
    Divides the latent before the squash. A larger ``T`` spreads the
    same box over a wider latent range, so the squash saturates more
    slowly. ``T = 1.0`` is the plain squash.
knee
    The ``z_knee`` field. Latent magnitude past which
    `saturation` starts charging. Derived per transform from
    one physical criterion, the outer 5% of the box.

**The two maps**

`to_latent` warps the physical value, normalises it to
``[0, 1]`` against the warped bounds, applies the squash inverse, and
multiplies by ``T``. `from_latent` inverts that. With the
default ``"linear"`` warp and ``"sigmoid"`` squash they read
``z = logit((x - low) / (high - low)) * T`` and
``x = low + (high - low) * sigmoid(z / T)``.

The round trip is the identity strictly inside the open box. Outside
a narrow band at the endpoints ``to_latent`` continues linearly (see
its docstring), so it deviates there rather than hitting a pole.

**The cost**

Bounds hold by construction; the price is gradient. The squash
derivative decays as ``|z|`` grows, so a predictor pinned against a
bound has little signal left to pull it back. Sigmoid's decay is
exponential and dies at ``z = 16.8`` in float32; ``"algebraic"`` and
``"softsign"`` decay polynomially and buy far more runway (numbers in
``transforms.py``). Runway alone is not enough, since escape time
still grows fast with ``|z|``. `saturation` charges the output
end for sitting deep in the squash, `input_violation` charges
the input end for arriving outside its box. Both are pure queries;
``__call__`` invokes neither.

``temperature`` is a dynamic leaf, frozen by the default training mask.
An explicit custom mask can opt into training it. The scaler
defines the activation shape and the inner predictor learns inside
it; a trainable ``T`` moves gradient between the two and tends to
slow convergence.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `bounds` | `tuple[tuple[float, float], ...]` | Per-component ``(low, high)`` pairs in physical units. Length sets the input/output dimension and the pairs apply elementwise to the last axis. The constructor also takes one bare pair, ``BoundScaler((270.0, 350.0))``, as a one-component scaler, which acts elementwise on an array of any shape. Use None or signed infinity for a missing endpoint. Finite pairs must satisfy ``low < high``. Open pairs require a linear warp. |
| `transform` | `str` | Static. Squash name, a key of ``BOUND_TRANSFORMS``. Register your own with ``register_bound_transform``. |
| `temperature` | `Array` | Scalar (or per-component) latent sharpness. ``T = 1.0`` recovers the plain squash and its inverse. |
| `warp` | `str` | Static. Warp name, a key of ``WARPS``. Register your own with ``register_warp``. |
| `warped_bounds` | `tuple[tuple[float, float], ...]` | Static. ``bounds`` pushed through the warp once at construction. Resolved eagerly so no call has to warp the edges under a trace. |
| `logit_eps` | `float` | Static. Half-width of the band at each end of ``[0, 1]`` outside which ``to_latent`` continues linearly instead of running into the squash inverse's pole. Also sets the continuation slope (roughly ``1 / logit_eps``). |
| `z_knee` | `float` | Static. The knee. Defaults to the transform's own value, which for sigmoid is ``2.944`` (``logit(0.95)``), the outer 5% of the box. Do not share one number across transforms: 2.944 is 12.5% from the bound on softsign, which would charge 2.7x too hard. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L170)</small>

<a id="boundscalerfrom_latent"></a>

#### `BoundScaler.from_latent()`

```python
from_latent(self, z: 'Array') -> 'Array'
```

Map a latent value back into the physical box ``[low, high]``.

Squash ``z / temperature`` into ``(0, 1)``, affine rescale onto the
warped box, then unwarp. The result is finite and inside the box for
any finite ``z``, so this direction needs no guard.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L524)</small>

<a id="boundscalerinput_violation"></a>

#### `BoundScaler.input_violation()`

```python
input_violation(self, x: 'Array') -> 'Array'
```

Scalar squared hinge on how far ``x`` fell outside ``bounds``.

Zero in value *and* gradient strictly inside the box, so adding it
to a loss never perturbs the feasible interior. Outside, it grows
quadratically in the width-normalised overshoot.

The push-back half of a pair: `to_latent` keeps the forward
pass finite and differentiable near the box, this term supplies a
restoring force that keeps working far outside it. Pure, so the
caller decides whether and where to pay for it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L464)</small>

<a id="boundscalersaturation"></a>

#### `BoundScaler.saturation()`

```python
saturation(self, z: 'Array') -> 'Array'
```

Scalar squared overshoot of ``|z / temperature|`` past ``z_knee``.

Measures how hard the output squash is pinned against its bound.
For sigmoid the knee is ``2.944``, where ``sigmoid(2.944) = 0.95``,
the outer 5% of the physical box on each side.

The argument is the latent, never the physical value it maps to.
``from_latent``'s derivative carries a ``sigma'(z / T)`` factor that
underflows to 0.0 past ``|z / T| ~ 17``, so a penalty written
against the physical output dies exactly where saturation is worst.
Reading ``|z| / T`` gives a gradient linear in the overshoot.

Reduced with ``mean``, not ``sum``, so one weight means the same for
a one-output and a six-output predictor.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L495)</small>

<a id="boundscalerscale_inputs"></a>

#### `BoundScaler.scale_inputs()`

```python
scale_inputs(self, x: 'Array') -> 'Array'
```

Normalise network inputs: finite intervals map to latent, open ones pass through.

A one-sided input bound says where the input may go, not what scale
it has, so the network sees such an input unchanged.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L533)</small>

<a id="boundscalerto_latent"></a>

#### `BoundScaler.to_latent()`

```python
to_latent(self, x: 'Array') -> 'Array'
```

Map a physical value to its latent representative.

Warp ``x``, normalise it to ``[0, 1]`` against the warped bounds,
apply the squash inverse through
[`soft_inverse`](/api/penalties#soft_inverse), multiply by
``temperature``.

The squash inverse has a pole at each end of ``[0, 1]``, guarded by
a linear continuation rather than a hard ``jnp.clip``. A clip has
zero derivative outside the box, and since the guard sits mid-graph
that zero propagates to every upstream parameter. Predictor inputs
are often state-derived (supersaturation in the crystallisation
example), so a clipped input silently drops a real sensitivity from
the adjoint.

Inside ``[logit_eps, 1 - logit_eps]`` the map is exactly the plain
inverse in value and derivative. Outside it continues linearly at
the inverse's slope at the crossing: finite values, non-zero
constant gradient, and a C^1 join so an adaptive step controller
sees no kink.

The continuation reports direction, not magnitude. Pair it with
`input_violation` when an input can leave its box.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L423)</small>

---

<a id="boundedpredictor"></a>

### `BoundedPredictor`

<small>`from jaxhybridmodels.predictors import BoundedPredictor` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.BoundedPredictor`</small>

```python
BoundedPredictor(
    inner: 'Predictor',
    in_scaler: 'BoundScaler | None' = None,
    out_scaler: 'BoundScaler | None' = None,
    input_keys: 'tuple[str, ...] | None' = None,
    input_bounds: 'Any' = None,
    output_bounds: 'Any' = None,
) -> None
```

A network in physical units, built from an inner network that never sees a bound.

This is the thing a user's vector field calls. It takes named inputs
in physical units, returns an output in physical units, and keeps
the output inside its declared range.

**Three stages**

1. **Normalise the input.** ``in_scaler.scale_inputs`` maps each input
   from its physical range onto an unbounded latent, so the network
   receives numbers of comparable size whether the input was a
   temperature in the tens or a concentration in the thousandths.
   Inputs with an open bound pass through unchanged.
2. **Run the network.** The trainable ``inner`` ``Predictor`` maps
   latent to latent. It is handed no bound information at all.
3. **Squash the output.** ``out_scaler.from_latent`` maps the
   network's unbounded output into the physical output range.

The network is better off never seeing a bound. Given one it would
have to enforce the range itself, with either a clip (zero gradient
outside, so a parameter that leaves cannot come back) or a final
squash it would have to learn to aim. Moving the squash into
``out_scaler`` makes an out-of-range output unrepresentable, leaves
``inner`` free to be any ``Array -> Array`` function, and lets the
same network be reused under different bounds.

For a single trainable number or array with no inputs, use
[`Parameter`](/api/predictors#parameter). For a fixed value, use [`Constant`](/api/predictors#constant).

**Calling it**

The user builds predictor inputs in the vector field by mixing
constant covariates with state-derived or exogenous time-dependent
values (CONTEXT.md, "Predictor inputs"). ``__call__`` accepts:

- A mapping from name to value. Extra keys are allowed; only the subset named
  in ``self.input_keys`` is pulled, in declared order. A missing key
  raises ``KeyError``.
- An array, rank-1 of length ``len(input_keys)``, shape-checked.

It returns a scalar ``()`` when there is one output bound, else a
rank-1 ``[n_out]`` array. To evaluate over a grid of inputs, use
[`evaluate_predictor`](/api/prediction#evaluate_predictor).

**Construction**

Give the scalers either as ``in_scaler`` / ``out_scaler`` instances or
as ``input_bounds`` / ``output_bounds`` pairs (None for a missing
endpoint; one pair broadcasts to every component). Sizes not fixed by
the bounds come from ``input_keys`` or ``inner.in_size`` /
``inner.out_size``. At least one input is required: a zero-input
network has no training signal, and that case is a ``Parameter``.
Omitting ``input_keys`` auto-fills ``("x1", ..., "xN")``, so a saved
predictor always describes its own input contract.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `input_keys` | `tuple[str, ...]` | Static. Declared order of the physical-units inputs; one entry per ``in_scaler.bounds`` row. Drives subset extraction when ``__call__`` receives a dict. |
| `in_scaler` | `BoundScaler` | Maps physical-space inputs into the inner network's latent space. |
| `inner` | `Predictor` | Trainable Array -> Array module operating in latent space. |
| `out_scaler` | `BoundScaler` | Maps the inner network's latent output back to physical units. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L619)</small>

<a id="boundedpredictorinitialized_with_key"></a>

#### `BoundedPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'BoundedPredictor'
```

Re-initialise ``inner`` only, leaving both scalers untouched.

Without this method, ``reinitialize_with_key`` takes its generic
branch and replaces every inexact leaf, ``BoundScaler.temperature``
included, with a sample from ``N(0, 1)``. A temperature near zero
or negative inverts and blows up both ``to_latent`` and
``from_latent``.

Delegating to the free function also restores the inner predictor's
own init scheme rather than leaf-level normal sampling. The scalers
hold bound geometry, not learned state, so a restart has no reason
to touch them.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L808)</small>

---

<a id="parameter"></a>

### `Parameter`

<small>`from jaxhybridmodels.predictors import Parameter` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.Parameter`</small>

```python
Parameter(
    value: 'Any',
    bounds: 'Any' = None,
    transform: 'str' = 'sigmoid',
    warp: 'str' = 'linear',
) -> None
```

A trainable physical scalar or array, read through ``.value``.

The optimiser updates ``latent``. ``.value`` maps it through ``scaler``
into physical units, so the value respects ``bounds`` at every step
without clipping.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `value` |  | Initial physical value: a scalar or an array of any shape. Must be finite and strictly inside its bounds. |
| `bounds` |  | One ``(low, high)`` pair shared by every element, or one pair per element in flattened order. None for a missing endpoint. Omit for an unbounded value. Two finite endpoints use the ``transform`` squash. One endpoint uses softplus distance from it. |
| `transform, warp` |  | Passed to [`BoundScaler`](/api/predictors#boundscaler). ``warp="log10"`` suits rate constants that span decades and needs finite positive bounds. |

**Examples**

::

    rate = Parameter(0.1, bounds=(0.0, None))
    rate.value  # Array(0.1)

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L553)</small>

<a id="parameterinitialized_with_key"></a>

#### `Parameter.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'Parameter'
```

Draw a fresh standard-normal latent; the scaler is left untouched.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L613)</small>

---

<a id="constant"></a>

### `Constant`

<small>`from jaxhybridmodels.predictors import Constant` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.Constant`</small>

```python
Constant(value: 'Any') -> 'None'
```

Fixed physical values, read through ``.value``.

Lists become arrays; tuples keep their structure and numeric leaves
become JAX arrays. The default training mask and tournament restarts
leave constants unchanged.

**Examples**

::

    damping = Constant(0.0)
    damping.value  # Array(0.)

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L140)</small>

<a id="constantinitialized_with_key"></a>

#### `Constant.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'Constant'
```

Keep fixed values unchanged when restarting a model.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L165)</small>

---

<a id="mlppredictor"></a>

### `MLPPredictor`

<small>`from jaxhybridmodels.predictors import MLPPredictor` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.MLPPredictor`</small>

```python
MLPPredictor(
    in_size: 'int',
    out_size: 'int',
    width_size: 'int',
    depth: 'int',
    key: 'Array',
    activation_name: 'str' = 'tanh',
) -> None
```

Multi-layer perceptron ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

Thin wrapper around ``eqx.nn.MLP`` exposing the constructor hyperparameters
as static fields (so they survive serialisation and tournament re-init).
Only the inner ``mlp`` field carries trainable weights; the rest is metadata.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `mlp` | `eqx.nn.MLP` | The actual trainable network; weights and biases are inexact-array leaves picked up by ``default_trainable``. |
| `in_size, out_size` | `int` | Input / output dimensionality (static). |
| `width_size, depth` | `int` | Hidden width and number of hidden layers (static). |
| `activation_name` | `str` | Key into ``_ACTIVATION_MAP``; stored as a string rather than the callable so the module is JSON-serialisable. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/mlp.py#L47)</small>

<a id="mlppredictorinitialized_with_key"></a>

#### `MLPPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'MLPPredictor'
```

Return a fresh ``MLPPredictor`` with the same architecture, new weights.

The re-init protocol consumed by [`reinitialize_with_key`](/api/predictors#reinitialize_with_key) and
by the tournament when it restarts a stalled attempt.
Re-instantiating beats reinitialising leaves in place because
``eqx.nn.MLP`` owns its per-layer init (LeCun-uniform weights
scaled by fan-in, zero biases), which leaf-level normal sampling
would replace with a badly scaled scheme.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/mlp.py#L115)</small>

<a id="mlppredictorwith_zero_final_head"></a>

#### `MLPPredictor.with_zero_final_head()`

```python
with_zero_final_head(self) -> 'MLPPredictor'
```

Return a copy whose final ``Linear`` layer's weight and bias are zero.

Hidden layers keep their LeCun-uniform init, so the feature
transformation stays non-degenerate; only the readout is zeroed.
Inside a ``BoundedPredictor`` the resulting zero latent maps
through ``out_scaler.from_latent(0)`` to the exact midpoint of the
physical output box, a known-good start independent of the key.
That matters when rate bounds span decades, where an unlucky
readout draw can stall the solver on step one.

Structurally identical predictor; only the trailing
``eqx.nn.Linear``'s ``weight`` and ``bias`` change.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/mlp.py#L134)</small>

---

<a id="kanpredictor"></a>

### `KANPredictor`

<small>`from jaxhybridmodels.predictors import KANPredictor` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.KANPredictor`</small>

```python
KANPredictor(
    in_size: 'int',
    out_size: 'int',
    hidden_widths: 'tuple[int, ...]' = (8,),
    grid_size: 'int' = 5,
    basis: 'str' = 'spline',
    key: 'Array',
) -> None
```

Kolmogorov-Arnold network ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

A KAN learns a basis expansion on each edge instead of a weight matrix
per layer. Wraps ``jaxkan.models.KAN`` with a serialisation-clean
static/dynamic split (module docstring has the reasoning): the one
dynamic field is ``params``, everything else is static.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `in_size, out_size` | `int` | Input / output dimensionality. Static; pinned by ``layer_dims`` in the underlying KAN. |
| `hidden_widths` | `tuple[int, ...]` | Widths of the hidden KAN layers in order. Static; combined with ``in_size`` and ``out_size`` to form ``layer_dims = [in_size, *hidden_widths, out_size]``. |
| `grid_size` | `int` | Number of spline grid intervals (``G``). Static; passed as ``required_parameters['G']`` to jaxkan. |
| `basis` | `str` | Layer-type code, one of ``"spline"`` or ``"base"``. Static. |
| `seed` | `int` | Integer seed used to build the underlying jaxkan model and to recreate its rng-state on demand inside ``__call__``. Derived from the user-supplied ``key`` at construction; static thereafter. |
| `params` | `nnx.State` | The one dynamic field. The ``nnx.Param`` slice of the KAN's state, all float arrays. Trainable, and round-trips through ``eqx.tree_serialise_leaves``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/kan.py#L119)</small>

<a id="kanpredictorinitialized_with_key"></a>

#### `KANPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'KANPredictor'
```

Return a same-architecture KANPredictor with freshly initialised parameters.

The re-init protocol used by the tournament. Building a new
``KANPredictor`` rather than editing ``self.params`` keeps jaxkan's
per-layer init in charge (truncated-normal spline weights, ones
bias, identity residual), which leaf-level normal sampling would
replace with a badly scaled scheme.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/kan.py#L234)</small>

<a id="kanpredictorwith_zero_final_head"></a>

#### `KANPredictor.with_zero_final_head()`

```python
with_zero_final_head(self) -> 'KANPredictor'
```

Return a copy whose final KAN layer's parameters are all zero.

The final head is the readout layer at index ``len(hidden_widths)``.
Zeroing its four trainable arrays makes it produce
``zeros(out_size)`` for any input, which inside a
``BoundedPredictor`` maps to the midpoint of the physical box.

Earlier layers keep their jaxkan-default init, so the feature
transformation stays non-degenerate. Same intent as
[`MLPPredictor.with_zero_final_head`](/api/predictors#mlppredictorwith_zero_final_head) over a different
parameterisation.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/kan.py#L252)</small>

---

<a id="neuralnpolynomial"></a>

### `NeuralNPolynomial`

<small>`from jaxhybridmodels.predictors import NeuralNPolynomial` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.NeuralNPolynomial`</small>

```python
NeuralNPolynomial(
    coeff_net: 'Predictor',
    exponents: 'tuple[float, ...]',
    in_size: 'int',
    out_size: 'int',
) -> None
```

Polynomial-in-``sum(x)`` whose coefficients come from ``coeff_net``.

Pure composition wrapper. Only the inner ``coeff_net`` is trainable;
``exponents`` (non-empty), ``in_size`` and ``out_size`` are static
metadata, so only the inner network's float leaves reach the binary
checkpoint.

**Coefficient layout**

``coeff_net(x)`` must produce a flat ``[out_size * len(exponents)]``
vector, validated at construction. ``__call__`` reshapes it row-major
to ``[out_size, len(exponents)]``, so row ``o`` holds the coefficients
for output channel ``o``.

**Example**

A 3-term quadratic with two output channels backed by an MLP::

    coeff_net = MLPPredictor(in_size=3, out_size=6, width_size=8,
                             depth=2, activation_name="tanh", key=key)
    npoly = NeuralNPolynomial(coeff_net=coeff_net,
                              exponents=(0.0, 1.0, 2.0),
                              in_size=3, out_size=2)

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/neural_npoly.py#L48)</small>

<a id="neuralnpolynomialinitialized_with_key"></a>

#### `NeuralNPolynomial.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'NeuralNPolynomial'
```

Re-initialise the inner ``coeff_net``; keep the polynomial structure.

Calls straight through to [`reinitialize_with_key`](/api/predictors#reinitialize_with_key), which
prefers the inner predictor's own ``initialized_with_key`` and falls
back to elementwise sampling only when it offers no scheme.
``exponents``, ``in_size`` and ``out_size`` are static.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/neural_npoly.py#L135)</small>

---

<a id="reinitialize_with_key"></a>

### `reinitialize_with_key()`

<small>`from jaxhybridmodels.predictors import reinitialize_with_key`</small>

```python
reinitialize_with_key(predictor: 'eqx.Module', key: 'Array') -> 'eqx.Module'
```

Return a fresh copy of ``predictor`` with its float leaves re-initialised.

Single-Module helper. If ``predictor`` implements
``initialized_with_key`` (a ``self -> key -> self`` method for classes
that want their own re-init scheme, such as a KAN that has to rebuild
its grid), this delegates to it. Otherwise every inexact-array leaf is
replaced with a standard-normal sample of matching shape and dtype.

For a *pytree* of predictors, the convention at the ``simulate_fn``
boundary, use [`reinitialize_pytree_with_key`](/api/predictors#reinitialize_pytree_with_key).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L829)</small>

---

<a id="reinitialize_pytree_with_key"></a>

### `reinitialize_pytree_with_key()`

<small>`from jaxhybridmodels.predictors import reinitialize_pytree_with_key`</small>

```python
reinitialize_pytree_with_key(predictors: 'Any', key: 'Array') -> 'Any'
```

Per-``eqx.Module``-leaf re-initialisation across a ``predictors`` pytree.

Used by the training tournament to escape bad initial weights: when
an attempt diverges or stalls, the loop draws a fresh per-attempt key,
calls this, and restarts. Each ``eqx.Module`` leaf gets its own
independent subkey, so identical-shape siblings re-init differently.

Subkeys are handed out in traversal order, which is enough because the
pytree shape is fixed across re-inits within one run. Path-derived
subkeys would be path-stable at the cost of a hash per leaf.

Accepts any pytree shape ``jax.tree_util`` can walk, including a bare
``eqx.Module``. Returns a structurally identical pytree with fresh
weights on every ``eqx.Module`` leaf.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/predictors/base.py#L856)</small>
