# Predictors: Trainable Components

Predictors are the trainable components of a hybrid model — Equinox modules with a fixed `Array → Array` signature. The framework wraps them in a [`BoundedPredictor`](#boundedpredictor), which sigmoid-scales physical-unit inputs into a latent box, runs the inner predictor, and scales the output back to physical units. Inner predictors never see or enforce bounds.

**Predictors pytree convention:** any pytree of `eqx.Module` leaves is accepted (single module, tuple, dict, NamedTuple). The single-predictor case is conventionally written as `(predictor,)` so the surrounding code never branches on container type.

## Quick links

- [`Predictor`](#predictor)
- [`BoundScaler`](#boundscaler)
- [`BoundedPredictor`](#boundedpredictor)
- [`MLPPredictor`](#mlppredictor)
- [`KANPredictor`](#kanpredictor)
- [`reinitialize_with_key`](#reinitialize_with_key)
- [`reinitialize_pytree_with_key`](#reinitialize_pytree_with_key)

---

<a id="predictor"></a>

### `Predictor`

<small>`from hybridmodels.predictors import Predictor` &nbsp;·&nbsp; also re-exported as `hybridmodels.Predictor`</small>

```python
Predictor() -> None
```

Abstract marker for a trainable ``Array -> Array`` module.

``Predictor`` carries no behaviour. It exists so the rest of the
framework can say "this leaf is a trainable function approximator"
and so composition wrappers have one type to accept. The base
``__call__`` raises.

To add your own family, subclass ``Predictor``, declare your
trainable arrays as ordinary fields and your hyperparameters as
``eqx.field(static=True)``, and implement
``__call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]``.
Two rules apply to every predictor in this package. Dynamic leaves
must be JAX float arrays and static fields must be JSON-encodable,
so the module round-trips through ``eqx.tree_serialise_leaves``.
Optionally implement ``initialized_with_key(self, key) -> Self`` to
control how the training tournament restarts your weights; without
it, :func:`reinitialize_with_key` replaces every float leaf with a
standard-normal sample, which skews any considered init scheme.

Do not subclass to add bound handling or named inputs.
``BoundedPredictor`` holds a ``Predictor`` as a field and supplies
both, and it exposes the richer call signature
(``dict[str, Array] | Array -> Array``) without touching this class.

Multi-rate models need no framework wrapper either. Several
predictors compose as a tuple at the ``simulate_fn`` boundary and
the user unpacks them at the top of the vector field, naming each
one in their own code (``rate_growth, rate_nucleation = predictors``).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L70)</small>

---

<a id="boundscaler"></a>

### `BoundScaler`

<small>`from hybridmodels.predictors import BoundScaler` &nbsp;·&nbsp; also re-exported as `hybridmodels.BoundScaler`</small>

```python
BoundScaler(
    bounds: 'tuple[tuple[float, float], ...]',
    transform: 'str' = 'sigmoid',
    temperature: 'Any' = 1.0,
    warp: 'str' = 'linear',
    logit_eps: 'float' = 0.001,
    z_knee: 'float | None' = None,
) -> None
```

Two-way map between a physical range ``[low, high]`` and an unbounded latent.

**Why this exists**

Physical quantities have ranges. A rate constant is positive, a
solubility lies between known limits, and an ODE solver handed a
value outside the range either fails or returns nonsense. An
optimiser knows none of that. It proposes whatever number lowers the
loss.

Clipping the proposal looks like the fix and is not usable here. A
clip has exactly zero derivative outside the range, so the moment a
parameter leaves the box the gradient that would pull it back is
zero and it stays out. This scaler reparameterises instead. The
inner predictor reads and writes a *latent* value, any real number,
and the scaler squashes that latent into the physical range. No
latent maps to an out-of-range physical value, so a violation is
unrepresentable and there is nothing to clip.

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
    :meth:`saturation` starts charging. Derived per transform from
    one physical criterion, the outer 5% of the box.

**The two maps**

:meth:`to_latent` warps the physical value, normalises it to
``[0, 1]`` against the warped bounds, applies the squash inverse, and
multiplies by ``T``. :meth:`from_latent` squashes ``z / T`` into
``(0, 1)``, rescales onto the warped box, and unwarps. With the
default ``"linear"`` warp and ``"sigmoid"`` squash these read
``z = logit((x - low) / (high - low)) * T`` and
``x = low + (high - low) * sigmoid(z / T)``.

Composing inverse with forward is the identity strictly inside the
open box ``(low, high)``. Outside a narrow band at the endpoints
``to_latent`` switches to a linear continuation (see its docstring),
so the round trip deviates there rather than hitting a pole.

**The cost**

Bounds now hold by construction, and the price is gradient. The
squash derivative decays as ``|z|`` grows, so a predictor pinned
against a bound has little signal left to pull it back. Sigmoid's
decay is exponential and dies at ``z = 16.8`` in float32; the
``"algebraic"`` and ``"softsign"`` transforms decay polynomially and
buy far more runway (numbers in ``transforms.py``). Runway alone is
not enough, because escape time still grows fast with ``|z|``.
:meth:`saturation` charges the output end for sitting deep in the
squash and :meth:`input_violation` charges the input end for arriving
outside its box. Both are pure queries. ``__call__`` invokes neither,
and the caller decides whether to pay for them.

The temperature ``T`` is a dynamic leaf, so it could be trained. The
recommended convention is to freeze it, for example with
``freeze_modules_of_type(mask, predictor, BoundScaler)``. The scaler
defines the activation shape and the inner predictor learns inside
it; a trainable ``T`` moves gradient between the two and tends to
slow convergence.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `bounds` | `tuple[tuple[float, float], ...]` | Per-component ``(low, high)`` pairs in physical units. Length sets the input/output dimension and the pairs apply elementwise to the last axis. Must be finite and ordered ``low < high``. |
| `transform` | `str` | Static. Squash name, a key of ``BOUND_TRANSFORMS``. Register your own with ``register_bound_transform``. |
| `temperature` | `Array` | Scalar (or per-component) latent sharpness. ``T = 1.0`` recovers the plain squash and its inverse. |
| `warp` | `str` | Static. Warp name, a key of ``WARPS``. Register your own with ``register_warp``. |
| `warped_bounds` | `tuple[tuple[float, float], ...]` | Static. ``bounds`` pushed through the warp once at construction. Resolved eagerly so no call has to warp the edges under a trace. |
| `logit_eps` | `float` | Static. Half-width of the band at each end of ``[0, 1]`` outside which ``to_latent`` continues linearly instead of running into the squash inverse's pole. Also sets the continuation slope (roughly ``1 / logit_eps``). |
| `z_knee` | `float` | Static. The knee. Defaults to the transform's own value, which for sigmoid is ``2.944`` (``logit(0.95)``), the outer 5% of the box. Do not share one number across transforms: 2.944 is 12.5% from the bound on softsign, which would charge 2.7x too hard. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L105)</small>

#### `BoundScaler.from_latent()`

```python
from_latent(self, z: 'Array') -> 'Array'
```

Map a latent value back into the physical box ``[low, high]``.

Squash ``z / temperature`` into ``(0, 1)``, affine rescale onto the
warped box, then unwarp. The result is finite and inside the box for
any finite ``z``, so this direction needs no guard.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L350)</small>

#### `BoundScaler.input_violation()`

```python
input_violation(self, x: 'Array') -> 'Array'
```

Scalar squared hinge on how far ``x`` fell outside ``bounds``.

Zero in value *and* gradient strictly inside the box, so adding it
to a loss never perturbs the feasible interior. Outside, it grows
quadratically in the width-normalised overshoot.

This is the push-back half of a pair. :meth:`to_latent` keeps the
forward pass finite and differentiable near the box; this term
supplies a restoring force that keeps working far outside it.

Pure and side-effect free. Emitting a penalty is a separate query,
so the caller decides whether and where to pay for it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L311)</small>

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
falls to 4.5e-5 by ``|z / T| = 10`` and underflows to exactly 0.0
past roughly 15. A penalty written against the physical output
inherits that factor on the backward pass and dies exactly where
saturation is worst. Reading ``|z| / T`` gives a gradient linear in
the overshoot that never underflows.

Reduced with ``mean``, not ``sum``, so the term does not scale with
output width. One weight then means the same for a one-output and a
six-output predictor.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L328)</small>

#### `BoundScaler.to_latent()`

```python
to_latent(self, x: 'Array') -> 'Array'
```

Map a physical value to its latent representative.

Three steps. Warp ``x`` and normalise it to ``[0, 1]`` against the
warped bounds, apply the squash inverse through
:func:`~hybridmodels.penalties.soft_inverse`, multiply by
``temperature``.

The squash inverse has a pole at each end of ``[0, 1]``, and the
guard against it is a linear continuation rather than a hard
``jnp.clip``. A hard clip has exactly zero derivative outside the
box. The guard sits mid-graph, so that zero propagates to every
upstream parameter on the path. Predictor inputs are often
state-derived. Supersaturation in the crystallisation example is a
traced function of the ODE state, so a clipped input drops a real
sensitivity from the adjoint with nothing raised and nothing
logged.

Inside ``[logit_eps, 1 - logit_eps]`` the map is exactly the plain
inverse in both value and derivative, so a model trained before
this guard existed keeps its numerics wherever it behaved. Outside,
the map continues linearly at the inverse's slope at the crossing.
Values stay finite, the gradient stays a non-zero constant, and the
join is C^1 so an adaptive step controller sees no kink.

The continuation reports direction, not magnitude. It cannot tell a
small excursion from a catastrophic one in a way a loss can act on.
Pair it with :meth:`input_violation` when an input can leave its box.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L274)</small>

---

<a id="boundedpredictor"></a>

### `BoundedPredictor`

<small>`from hybridmodels.predictors import BoundedPredictor` &nbsp;·&nbsp; also re-exported as `hybridmodels.BoundedPredictor`</small>

```python
BoundedPredictor(
    in_scaler: 'BoundScaler',
    inner: 'Predictor',
    out_scaler: 'BoundScaler',
    input_keys: 'tuple[str, ...] | None' = None,
) -> None
```

A predictor in physical units, built from a network that never sees a bound.

This is the thing a user's vector field calls. It takes named inputs
in physical units, returns an output in physical units, and keeps
both inside their declared ranges.

**Three stages**

1. **Normalise the input.** ``in_scaler.to_latent`` maps each input
   from its physical range onto an unbounded latent, so the network
   receives numbers of comparable size whether the input was a
   temperature in the tens or a concentration in the thousandths.
2. **Run the network.** The trainable ``inner`` ``Predictor`` maps
   latent to latent. It is handed no bound information at all.
3. **Squash the output.** ``out_scaler.from_latent`` maps the
   network's unbounded output into the physical output box.

The network is better off never seeing a bound. Given one it would
have to enforce the range itself, and the only tools it has are a
clip (zero gradient outside the range, so a parameter that leaves
cannot come back) or a final squash it would have to learn to aim.
Moving the squash into ``out_scaler`` makes an out-of-range output
unrepresentable, leaves ``inner`` free to be any ``Array -> Array``
function, and lets the same network be reused under different bounds.

**Calling it**

The user builds predictor inputs in the vector field by mixing
constant covariates with state-derived or exogenous time-dependent
values (CONTEXT.md, "Predictor inputs"). ``__call__`` accepts that in
two forms:

- ``dict[str, Array]``. Extra keys are allowed. Only the subset named
  in ``self.input_keys`` is pulled, in declared order. A missing key
  raises ``KeyError``.
- ``Array``, rank-1 of length ``len(input_keys)``, passed through
  after a shape check. Use this when stacking positionally at the
  call site is more natural.

**Construction**

``input_keys`` must have the same length as ``in_scaler.bounds``, and
that length must be at least 1. A predictor with zero inputs has no
training signal. Omitting ``input_keys`` auto-fills
``("x1", "x2", ..., "xN")``, so the static field is always populated
and a saved predictor still describes its own input contract. A
positional ``Array`` call works either way.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `input_keys` | `tuple[str, ...]` | Static. Declared order of the physical-units inputs; one entry per ``in_scaler.bounds`` row. Drives subset extraction when ``__call__`` receives a dict. |
| `in_scaler` | `BoundScaler` | Maps physical-space inputs into the inner network's latent space. |
| `inner` | `Predictor` | Trainable Array -> Array module operating in latent space. |
| `out_scaler` | `BoundScaler` | Maps the inner network's latent output back to physical units. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L363)</small>

#### `BoundedPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'BoundedPredictor'
```

Re-initialise ``inner`` only, leaving both scalers untouched.

Without this method, ``reinitialize_with_key`` falls through to its
generic branch and replaces every inexact leaf, including
``BoundScaler.temperature``, with a sample from ``N(0, 1)``. A
temperature near zero (or negative) inverts and blows up both
``to_latent`` (which multiplies by ``T``) and ``from_latent`` (which
divides by it), so a tournament attempt on the documented
``(BoundedPredictor, ...)`` shape came back numerically wrecked.

Delegating to the free function also restores the inner predictor's
own scheme. ``MLPPredictor.initialized_with_key`` re-instantiates so
Equinox's LeCun-uniform init applies; leaf-level normal sampling
skews that distribution, which is the failure its docstring warns
about.

The scalers hold the bound geometry, not learned state, so a restart
has no reason to touch them.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L481)</small>

---

<a id="mlppredictor"></a>

### `MLPPredictor`

<small>`from hybridmodels.predictors import MLPPredictor` &nbsp;·&nbsp; also re-exported as `hybridmodels.MLPPredictor`</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/mlp.py#L47)</small>

#### `MLPPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'MLPPredictor'
```

Return a fresh ``MLPPredictor`` with the same architecture, new weights.

Implements the re-init protocol consumed by
:func:`reinitialize_with_key` and by the training tournament loop
when it restarts a stalled attempt. Re-instantiating the whole
module beats reinitialising leaves in place, because
``eqx.nn.MLP`` owns its per-layer init logic (LeCun-uniform
weights scaled by fan-in, zero biases). Leaf-level standard-normal
sampling would replace that scheme with a badly scaled one.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/mlp.py#L117)</small>

#### `MLPPredictor.with_zero_final_head()`

```python
with_zero_final_head(self) -> 'MLPPredictor'
```

Return a copy whose final ``Linear`` layer's weight and bias are zero.

Hidden layers keep their LeCun-uniform random init, so the input
feature transformation stays non-degenerate. Only the readout is
forced to zero. Composed inside a ``BoundedPredictor``, the zero
latent produced for every input maps through
``out_scaler.from_latent(0)`` to the *exact midpoint* of the
physical output box, a known-good starting value independent of
the key. That matters when the rate bounds span many decades: an
unlucky readout draw can place the initial output several decades
off midpoint, far enough that the ODE solver stalls or fails on
step one.

Returns a structurally identical predictor. Only the trailing
``eqx.nn.Linear``'s ``weight`` and ``bias`` arrays change.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/mlp.py#L137)</small>

---

<a id="kanpredictor"></a>

### `KANPredictor`

<small>`from hybridmodels.predictors import KANPredictor` &nbsp;·&nbsp; also re-exported as `hybridmodels.KANPredictor`</small>

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

A KAN learns a basis expansion on each edge instead of a weight
matrix per layer. Wraps ``jaxkan.models.KAN`` with a
serialisation-clean static/dynamic split; the module docstring has
the full reasoning. The trainable content is one ``params``
``nnx.State`` field whose leaves are float Param arrays. Everything
else (architecture, rng seed, basis kind) is static.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `in_size, out_size` | `int` | Input / output dimensionality. Static; pinned by ``layer_dims`` in the underlying KAN. |
| `hidden_widths` | `tuple[int, ...]` | Widths of the hidden KAN layers in order. Static; combined with ``in_size`` and ``out_size`` to form ``layer_dims = [in_size, *hidden_widths, out_size]``. |
| `grid_size` | `int` | Number of spline grid intervals (``G``). Static; passed as ``required_parameters['G']`` to jaxkan. |
| `basis` | `str` | Layer-type code, one of ``"spline"`` or ``"base"``. Static. |
| `seed` | `int` | Integer seed used to build the underlying jaxkan model and to recreate its rng-state on demand inside ``__call__``. Derived from the user-supplied ``key`` at construction; static thereafter. |
| `params` | `nnx.State` | The one dynamic field. The ``nnx.Param`` slice of the KAN's state, all float arrays. Trainable, and round-trips through ``eqx.tree_serialise_leaves``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/kan.py#L137)</small>

#### `KANPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'KANPredictor'
```

Return a same-architecture KANPredictor with freshly initialised parameters.

Implements the re-init protocol used by the training tournament
loop. Building a whole new ``KANPredictor``, rather than editing
``self.params`` in place, keeps jaxkan's per-layer init logic in
charge (truncated-normal spline weights, ones bias, identity
residual). Leaf-level standard-normal samples would replace that
scheme with a badly scaled one.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/kan.py#L262)</small>

#### `KANPredictor.with_zero_final_head()`

```python
with_zero_final_head(self) -> 'KANPredictor'
```

Return a copy whose final KAN layer's parameters are all zero.

For a KAN, the "final head" is the readout layer at index
``len(hidden_widths)`` in the underlying ``layer_dims`` list. Each
layer carries four trainable arrays (``c_basis``, ``c_spl``,
``c_res``, ``bias``); zeroing all of them makes the layer produce
``zeros(out_size)`` regardless of input, which composed inside a
``BoundedPredictor`` maps via ``out_scaler.from_latent(0)`` to the
midpoint of the physical bound box.

The earlier KAN layers keep their jaxkan-default init, so the
input feature transformation stays non-degenerate. Only the
readout is locked. Same intent as
:meth:`MLPPredictor.with_zero_final_head`, a seed-independent
initial physical output, over a different parameterisation.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/kan.py#L281)</small>

---

<a id="reinitialize_with_key"></a>

### `reinitialize_with_key()`

<small>`from hybridmodels.predictors import reinitialize_with_key` &nbsp;·&nbsp; also re-exported as `hybridmodels.reinitialize_with_key`</small>

```python
reinitialize_with_key(predictor: 'eqx.Module', key: 'Array') -> 'eqx.Module'
```

Return a fresh copy of ``predictor`` with its float leaves re-initialised.

Single-Module helper. If ``predictor`` implements the
``initialized_with_key`` protocol (a method ``self -> key -> self``
for classes that want their own re-init scheme, such as a KAN that
has to rebuild its grid), this delegates to it. Otherwise every
inexact-array leaf is replaced with a standard-normal sample of
matching shape and dtype, and non-inexact leaves and static fields
are left alone.

To re-initialise a *pytree* of predictors, the convention at the
``simulate_fn`` boundary, use :func:`reinitialize_pytree_with_key`.
It gives each ``eqx.Module`` leaf its own independently derived key.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L508)</small>

---

<a id="reinitialize_pytree_with_key"></a>

### `reinitialize_pytree_with_key()`

<small>`from hybridmodels.predictors import reinitialize_pytree_with_key` &nbsp;·&nbsp; also re-exported as `hybridmodels.reinitialize_pytree_with_key`</small>

```python
reinitialize_pytree_with_key(predictors: 'Any', key: 'Array') -> 'Any'
```

Per-``eqx.Module``-leaf re-initialisation across a ``predictors`` pytree.

Used by the training tournament to escape bad initial weights:
when an attempt diverges or stalls, the loop draws a fresh
per-attempt key, calls this function, and restarts. Each
``eqx.Module`` leaf gets its *own* independent subkey, so two
sibling predictors with identical shapes still re-init to
different random weights.

Subkeys are handed out in **traversal order**. Count the
``eqx.Module`` leaves with a Module-stopped traversal, call
``jr.split(key, n_module_leaves)`` once, and assign in that order.
The alternative, folding each leaf's path string, would give
path-stable subkeys at the cost of a hash per leaf and a less
obvious correspondence between subkey and pytree position.
Traversal order is enough because the pytree shape is fixed across
re-inits within one training run.

Accepts any pytree shape: the conventional
``tuple[BoundedPredictor, ...]``, a bare ``eqx.Module`` (a
one-leaf pytree, equivalent to calling
:func:`reinitialize_with_key` directly), ``dict[str, ...]``,
``NamedTuple`` subclasses, and any nested combinations
``jax.tree_util`` can walk.

Returns a structurally identical pytree with fresh weights on every
``eqx.Module`` leaf.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L538)</small>
