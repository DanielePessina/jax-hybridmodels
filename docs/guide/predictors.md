# Predictors and bounds

A predictor is a callable Equinox module. It maps an input array to an output
array. `MLPPredictor`, `KANPredictor`, and `NeuralNPolynomial` are included;
you can define another predictor by subclassing `Predictor`.

## Fixed values and trainable quantities

Use `.value` for stored quantities. A constant stays fixed under the default
training mask and tournament restarts:

```python
omega = hm.ConstantPredictor(value=1.0)
frequency = omega.value
constants = hm.ConstantPredictor(value=(1.0, [0.1, 0.2]))
scalar, vector = constants.value
```

Lists become arrays; tuples preserve their nested structure with array
leaves. Constants have no bounds or `trainable=True` option.

Use `BoundedPredictor(value=...)` for a trainable scalar or array:

```python
rate = hm.BoundedPredictor(value=0.1, bounds=(0.0, None))
offset = hm.BoundedPredictor(value=0.0)
kinetics = hm.BoundedPredictor(
    value=[0.1, 0.02], bounds=[(0.0, None), (0.0, 1.0)],
)
current_rate = rate.value
```

The initial value is in physical units and must lie strictly inside supplied
bounds. Scalars stay scalars; array parameters retain their shape. A single
bound pair broadcasts to all components; per-component pairs follow flattened
array order. Supply either `value=` or `inner=`, never both.

## Optional bounds

| Bounds | Output transform | Input scaling |
| --- | --- | --- |
| `(low, high)` | Existing finite-interval squash | Existing finite-interval scaling |
| `(low, None)` | `low + softplus(z / temperature)` | Identity |
| `(None, high)` | `high - softplus(z / temperature)` | Identity |
| `(None, None)` or omitted | Identity | Identity |

`-inf` and `inf` are equivalents to missing lower and upper endpoints.
Open bounds require a linear warp. Logarithmic warps remain available for
finite positive intervals. Unbounded outputs incur no saturation penalty;
one-sided outputs incur a penalty only toward their finite endpoint.
`box_grid` requires finite input intervals; supply explicit penalty points
for networks with open input bounds.

## A bounded predictor

`BoundedPredictor` composes three parts:

1. `in_scaler` maps physical inputs to latent inputs.
2. `inner` evaluates the predictor in latent space.
3. `out_scaler` maps the latent output to physical units.

```python
predictor = hm.BoundedPredictor(
    input_keys=("temperature_C", "supersaturation"),
    in_scaler=hm.BoundScaler(
        bounds=((0.0, 100.0), (0.0, 5.0)),
        warp="linear",
    ),
    inner=hm.MLPPredictor(
        in_size=2,
        out_size=1,
        width_size=16,
        depth=2,
        key=jr.PRNGKey(0),
    ),
    out_scaler=hm.BoundScaler(bounds=((0.0, 10.0),)),
)
```

The dictionary call uses `input_keys` to select and order inputs:

```python
rate = predictor({
    "temperature_C": temperature,
    "supersaturation": supersaturation,
})
```

The array call accepts a rank-1 array in the same order as `input_keys`.

## Choosing bounds

Finite bounds must be ordered. They should cover the values the solver may
encounter, with a small margin. Use `warp="log"` or `warp="log10"` when a
positive quantity spans several orders of magnitude.

The input warp must match the runtime domain. For example, a `log10` input
requires strictly positive state-derived values.

## Saturation

The output squash keeps physical values inside its declared box. Its gradient
gets small near the edges. The optional `penalty_weight` in an Optax or Evosax
config charges saturation at the *measured points* — the input vectors the
loss actually sees at observed cells — plus any user-supplied penalty-only
points (`penalty_points`, one array per leaf; `box_grid` builds a warp-uniform
box sweep if you want one).

This penalty is not a feasibility mechanism. Reparameterisation enforces the
physical bounds; the penalty keeps the predictor away from a saturated region
where optimization becomes slow.

## Initialization and freezing

Construct networks with an explicit JAX key. Value predictors need no random
key. The default trainability mask selects floating-point parameter leaves and
freezes `ConstantPredictor` and `BoundScaler` subtrees. Additional fixed
arrays in custom predictors still need a mask:

```python
mask = hm.trainable_mask(predictors)
mask = hm.freeze_modules_of_type(mask, predictors, hm.BoundScaler)
```

Every predictor must round-trip through Equinox serialization. A custom
predictor with a special initialization scheme can implement
`initialized_with_key(self, key)` for tournament restarts.
