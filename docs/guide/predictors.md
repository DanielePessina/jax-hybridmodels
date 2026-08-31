# Predictors and bounds

A predictor is a callable Equinox module. It maps an input array to an output
array. `MLPPredictor`, `KANPredictor`, and `NeuralNPolynomial` are included;
you can define another predictor by subclassing `Predictor`.

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

Bounds must be finite and ordered. They should cover the values the solver may
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

Construct predictors with an explicit JAX key. The default trainability mask
selects floating-point array leaves. Freeze known non-parameter leaves with a
mask:

```python
mask = hm.trainable_mask(predictors)
mask = hm.freeze_modules_of_type(mask, predictors, hm.BoundScaler)
```

Every predictor must round-trip through Equinox serialization. A custom
predictor with a special initialization scheme can implement
`initialized_with_key(self, key)` for tournament restarts.

