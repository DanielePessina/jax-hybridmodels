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

Abstract marker for trainable Array -> Array modules.

Concrete subclasses (MLPPredictor, KANPredictor, ...) implement `__call__`
with signature ``Float[Array, "in"] -> Float[Array, "out"]``. The
composition wrapper (`BoundedPredictor`) holds a `Predictor` as a field
and exposes a richer call signature (``dict[str, Array] | Array -> Array``)
without subclassing it.

Multi-rate models do not need a framework wrapper: multiple
predictors compose as a tuple at the ``simulate_fn`` boundary and
the user unpacks them at the top of the vector field, naming each
one in their own code (``rate_growth, rate_nucleation = predictors``).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L68)</small>

---

<a id="boundscaler"></a>

### `BoundScaler`

<small>`from hybridmodels.predictors import BoundScaler` &nbsp;·&nbsp; also re-exported as `hybridmodels.BoundScaler`</small>

```python
BoundScaler(
    bounds: 'tuple[tuple[float, float], ...]',
    transform: 'str' = 'sigmoid',
    temperature: 'Any' = 1.0,
    logit_eps: 'float' = 0.001,
    z_knee: 'float' = 3.0,
) -> None
```

Bidirectional sigmoid scaler between physical ``[low, high]`` and an unbounded latent.

The trainable inner predictor sees no bounds and outputs an
unbounded latent value; this scaler translates between that latent
and the physical box the simulator actually needs.

The forward map is
``physical -> latent = logit((x - low) / (high - low)) * T``; the
inverse is
``latent -> physical = low + (high - low) * sigmoid(z / T)``.
Composing inverse with forward is the identity strictly inside the
open box ``(low, high)``; outside a narrow band at the endpoints
``to_latent`` switches to a linear continuation (see method doc), so
the round trip deviates there rather than saturating.

Bounds are enforced by *construction* — the inner predictor emits an
unbounded latent and ``from_latent`` squashes it — so a physical
violation is unrepresentable and there is nothing to clip. What that
costs is gradient: the squash derivative decays exponentially, so a
predictor pinned against a bound has no signal left to pull it back.
:meth:`saturation` and :meth:`input_violation` are the optional
penalty queries that repair the two ends of that problem; both are
pure and neither is invoked by ``__call__``.

Sigmoid is the only transform supported here; alternative transforms
can be introduced by extending ``_SUPPORTED_TRANSFORMS`` and adding
matching forward/inverse maps.

The temperature ``T`` is a leaf, not a static field, so it could in
principle be trained. The recommended convention is to freeze it
(e.g. via ``freeze_modules_of_type(mask, predictor, BoundScaler)``)
because the scaler is meant to define the activation shape, not
learn it; leaving it trainable shifts the gradient signal between
the scaler and the inner predictor and tends to slow convergence.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `bounds` | `tuple[tuple[float, float], ...]` | Per-component ``(low, high)`` pairs. Length sets the I/O dimension; applies elementwise to the last axis of inputs. |
| `transform` | `str` | Name of the scaling transform; ``"sigmoid"`` is currently the only supported value. |
| `temperature` | `Array` | Scalar (or per-component) sharpness multiplier in latent space. ``T = 1.0`` recovers the standard logit/sigmoid pair. |
| `logit_eps` | `float` | Static. Half-width of the band at each end of ``[0, 1]`` outside which ``to_latent`` continues linearly instead of running into ``logit``'s pole. Sets the continuation slope (``~1 / logit_eps``). |
| `z_knee` | `float` | Static. Latent magnitude past which :meth:`saturation` starts charging. ``3.0`` is the outer 5% of the physical box. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L87)</small>

#### `BoundScaler.from_latent()`

```python
from_latent(self, z: 'Array') -> 'Array'
```

Map a latent value back into the physical box ``[low, high]``.

Apply ``sigmoid(z / temperature)`` to land in ``(0, 1)``, then affine
rescale to ``[low, high]``. The output is finite for any finite ``z``
(no clipping required on the inverse direction).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L248)</small>

#### `BoundScaler.input_violation()`

```python
input_violation(self, x: 'Array') -> 'Array'
```

Scalar squared hinge on how far ``x`` fell outside ``bounds``.

Zero in value *and* gradient strictly inside the box, so adding it
to a loss never perturbs the feasible interior. Outside, it grows
quadratically in the width-normalised overshoot.

This is the push-back half of the pair whose forward half is the
softclip in :meth:`to_latent`: the softclip keeps the forward pass
finite and differentiable near the box, this term supplies a
restoring force that keeps working arbitrarily far from it.

Pure and side-effect free — emitting a penalty is a separate query,
never a side effect of calling the scaler, which is what lets the
caller decide whether and where to pay for it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L205)</small>

#### `BoundScaler.saturation()`

```python
saturation(self, z: 'Array') -> 'Array'
```

Scalar squared overshoot of ``|z / temperature|`` past ``z_knee``.

Measures how hard the output squash is pinned against its bound.
``z_knee`` defaults to ``3.0``, i.e. ``sigmoid(3) ~ 0.953`` — the
outer 5% of the physical box on each side.

Deliberately a function of the *latent*, not of the physical value
it maps to. ``from_latent``'s derivative carries a ``sigma'(z / T)``
factor that decays to ``4.5e-5`` by ``|z / T| = 10`` and underflows
to exactly ``0.0`` past roughly ``15``; a penalty written against
the physical output inherits that factor on the backward pass and
so dies exactly where saturation is worst. Reading ``|z| / T``
directly gives a gradient linear in the overshoot that never
underflows.

Reduced with ``mean`` rather than ``sum`` so the term does not
scale with the number of output components — one penalty weight
then means the same thing for a one-output and a six-output
predictor.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L224)</small>

#### `BoundScaler.to_latent()`

```python
to_latent(self, x: 'Array') -> 'Array'
```

Map a physical-space value to its latent representative.

Steps: normalise to ``[0, 1]`` against ``bounds``, apply
:func:`~hybridmodels.penalties.soft_logit`, scale by ``temperature``.

The ``logit`` pole guard is a linear continuation, not a hard
``jnp.clip``. A hard clip has *exactly* zero derivative outside the
box, and because this guard sits mid-graph that zero propagates to
every upstream parameter on the path. Predictor inputs are
routinely state-derived — supersaturation in the crystallisation
example is a traced function of the ODE state — so a clipped input
silently drops a real sensitivity out of the ODE adjoint with
nothing raised and nothing logged.

Inside ``[logit_eps, 1 - logit_eps]`` the map is *exactly* the old
``logit``, value and derivative both, so models trained before this
change keep their numerics wherever they were behaving. Outside it,
the map continues linearly at ``logit``'s own slope at the
crossing: finite values, constant non-zero gradient, ``C^1`` across
the junction so an adaptive ODE controller sees no kink.

The continuation still only reports *direction*, not magnitude —
it cannot tell a small excursion from a catastrophic one in a way a
loss can act on. Pair it with :meth:`input_violation` when an input
can leave its declared box.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L174)</small>

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

Composition wrapper: ``in_scaler.to_latent -> inner -> out_scaler.from_latent``.

The full physical-units forward pass for a bound-scaled predictor.
The user constructs predictor inputs in the vector field by mixing
constant covariates with state-derived or exogenous time-dependent
values (CONTEXT.md "Predictor inputs"). ``__call__`` accepts that
construction in two equivalent forms:

- ``dict[str, Array]`` — the dict may carry extra keys; only the
  named subset listed in ``self.input_keys`` is pulled, in declared
  order. Missing keys raise ``KeyError``.
- ``Array`` (rank-1, length ``len(input_keys)``) — passed through
  after a shape check (``eqx.error_if``). Useful when the user
  prefers to stack positionally at the call site.

From there, ``in_scaler`` maps each value into the inner network's
latent input space, the trainable ``Predictor`` runs in unbounded
latent space, and ``out_scaler`` maps its output back into the
physical output box. ``inner`` therefore sees no bound information
and never has to clamp itself.

**Construction**

``input_keys`` is required to match ``len(in_scaler.bounds)`` and
that length must be at least 1 (a predictor with zero inputs has no
training signal). When ``input_keys`` is omitted (``None``), the
constructor auto-fills ``("x1", "x2", ..., "xN")`` so the static
field is always populated and the saved predictor remains
self-describing — the user can still call it with a positional
``Array`` even if they never wrote a names tuple.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `input_keys` | `tuple[str, ...]` | Static. Declared order of the physical-units inputs; one entry per ``in_scaler.bounds`` row. Drives subset extraction when ``__call__`` receives a dict. |
| `in_scaler` | `BoundScaler` | Maps physical-space inputs into the inner network's latent space. |
| `inner` | `Predictor` | Trainable Array -> Array module operating in latent space. |
| `out_scaler` | `BoundScaler` | Maps the inner network's latent output back to physical units. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L259)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/mlp.py#L45)</small>

#### `MLPPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'MLPPredictor'
```

Return a fresh ``MLPPredictor`` with the same architecture, new weights.

Implements the re-init protocol consumed by
:func:`reinitialize_with_key` and by the training tournament loop
when it restarts a stalled attempt. Re-instantiating the whole
module is cleaner than reinitialising leaves in place because
``eqx.nn.MLP`` owns its own per-layer init logic (Glorot/normal
scaling, zero biases); leaf-level standard-normal sampling would
skew the distribution and break that scheme.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/mlp.py#L115)</small>

#### `MLPPredictor.with_zero_final_head()`

```python
with_zero_final_head(self) -> 'MLPPredictor'
```

Return a copy whose final ``Linear`` layer's weight and bias are zero.

Hidden layers retain their LeCun-uniform random init, so the input
feature transformation is non-degenerate; only the readout layer is
forced to zero. Composed inside a ``BoundedPredictor``, the latent
zero produced for any input maps via ``out_scaler.from_latent(0)``
to the *exact midpoint* of the physical bound box — a known-good
starting output that is independent of the random key. This makes
training reproducible across seeds when the rate bounds span many
decades and an unlucky standard-normal readout draw could otherwise
place the initial output too far off midpoint for the downstream
ODE solver to handle.

Returns a structurally identical predictor; only the trailing
``eqx.nn.Linear``'s ``weight`` and ``bias`` arrays change.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/mlp.py#L135)</small>

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

Kolmogorov-Arnold network predictor: ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

Wraps ``jaxkan.models.KAN`` with a serialisation-clean static/dynamic split
(see module docstring for the full rationale). The trainable content is
exposed as a single ``params`` ``nnx.State`` field whose leaves are float
Param arrays; everything else (architecture, rng seed, basis kind) is
static.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `in_size, out_size` | `int` | Input / output dimensionality. Static; pinned by ``layer_dims`` in the underlying KAN. |
| `hidden_widths` | `tuple[int, ...]` | Widths of the hidden KAN layers in order. Static; combined with ``in_size`` and ``out_size`` to form ``layer_dims = [in_size, *hidden_widths, out_size]``. |
| `grid_size` | `int` | Number of spline grid intervals (``G``). Static; passed as ``required_parameters['G']`` to jaxkan. |
| `basis` | `str` | Layer-type code, one of ``"spline"`` or ``"base"``. Static. |
| `seed` | `int` | Integer seed used to build the underlying jaxkan model and to recreate its rng-state on demand inside ``__call__``. Derived from the user-supplied ``key`` at construction; static thereafter. |
| `params` | `nnx.State` | Dynamic field — the ``nnx.Param`` slice of the KAN's state, all float arrays. Trainable; serialised round-trip via ``eqx.tree_serialise_leaves``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/kan.py#L76)</small>

#### `KANPredictor.initialized_with_key()`

```python
initialized_with_key(self, key: 'Array') -> 'KANPredictor'
```

Return a same-architecture KANPredictor with freshly initialised parameters.

Implements the re-init protocol used by the training tournament
loop. Building a whole new ``KANPredictor`` (rather than
tweaking ``self.params`` in place) lets jaxkan's per-layer init
logic — truncated-normal spline weights, ones bias, identity
residual — drive the initialisation instead of replacing it with
leaf-level standard-normal samples that would skew the
distribution.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/kan.py#L192)</small>

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
input feature transformation is non-degenerate; only the readout
is locked. Mirrors :meth:`MLPPredictor.with_zero_final_head` —
same intent (seed-independent initial physical output), different
parameterisation.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/kan.py#L212)</small>

---

<a id="reinitialize_with_key"></a>

### `reinitialize_with_key()`

<small>`from hybridmodels.predictors import reinitialize_with_key` &nbsp;·&nbsp; also re-exported as `hybridmodels.reinitialize_with_key`</small>

```python
reinitialize_with_key(predictor: 'eqx.Module', key: 'Array') -> 'eqx.Module'
```

Return a fresh copy of `predictor` with inexact-float leaves re-initialised.

Single-Module helper. If `predictor` implements the
``initialized_with_key`` protocol (a method ``self -> key -> self``
used by predictor classes that want a custom re-init scheme — for
example a KAN that needs to rebuild its grid), it is delegated to.
Otherwise every inexact-array leaf in the pytree is replaced with a
standard-normal sample of matching shape and dtype; non-inexact
leaves and static fields are left untouched.

For re-initialising a *pytree* of predictors (the convention at the
``simulate_fn`` boundary — typically a tuple of ``BoundedPredictor``s),
use :func:`reinitialize_pytree_with_key` so each ``eqx.Module`` leaf
gets its own independently-derived key.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L363)</small>

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

The split is done by **traversal order**: we count the
``eqx.Module`` leaves with a Module-stopped traversal, call
``jr.split(key, n_module_leaves)`` once, and hand out subkeys in
that order. The alternative — folding the per-leaf path string —
would give path-stable subkeys but cost a hash per leaf and
produce a less obvious correspondence between subkeys and
pytree positions; traversal-order splitting is simpler and
sufficient because the pytree shape is fixed across re-inits
within a single training run.

Accepts any pytree shape: the conventional
``tuple[BoundedPredictor, ...]``, a bare ``eqx.Module`` (a
one-leaf pytree, equivalent to calling
:func:`reinitialize_with_key` directly), ``dict[str, ...]``,
``NamedTuple`` subclasses, and any nested combinations
``jax.tree_util`` can walk.

Returns a structurally identical pytree with fresh weights on every
``eqx.Module`` leaf.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/predictors/base.py#L394)</small>
