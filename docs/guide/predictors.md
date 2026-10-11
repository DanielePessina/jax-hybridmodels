# Predictors and bounds

A predictor is a callable Equinox module that maps an input array to an
output array. `MLPPredictor`, `KANPredictor` and `NeuralNPolynomial` ship
with the library. Subclass `Predictor` to add another family; see
[Custom predictors](/guide/custom-predictors).

Three leaf types cover most models:

| Leaf | Holds | Trained by default |
| --- | --- | --- |
| `Constant` | A fixed value | No |
| `Parameter` | A physical value with optional bounds | Yes |
| `BoundedPredictor` | A network from named inputs to a bounded physical output | Yes, except its scalers |

## Constants and parameters

Read both through `.value`:

```python
import jaxhybridmodels as hm

omega = hm.Constant(1.0)
omega.value                                   # Array(1.)

scalar, vector = hm.Constant((1.0, [0.1, 0.2])).value
```

Lists become arrays. Tuples keep their nested structure with array leaves.

```python
rate = hm.Parameter(0.1, bounds=(0.0, None))
offset = hm.Parameter(0.0)                    # unbounded
kinetics = hm.Parameter([0.1, 0.02], bounds=[(0.0, None), (0.0, 1.0)])
k_ref = hm.Parameter(1e-3, bounds=(1e-6, 1e2), warp="log10")
```

The initial value is in physical units and must lie strictly inside its
bounds. Every bounds argument in the library (`Parameter(bounds=)`,
`BoundedPredictor(input_bounds=, output_bounds=)`, `BoundScaler(bounds)`)
reads the same way: one `(low, high)` pair applies to every element, and a
sequence of pairs applies per element in flattened order. The optimiser updates an unbounded latent
and `.value` maps it back, so the value never leaves its bounds. Tournament
restarts draw a fresh latent.

## Bounds

| Bounds | Output | Input |
| --- | --- | --- |
| `(low, high)` | `transform` squash onto the interval | `warp`, then inverse squash |
| `(low, None)` | `low + softplus(z / temperature)` | identity |
| `(None, high)` | `high - softplus(z / temperature)` | identity |
| `(None, None)` or omitted | identity | identity |

`-inf` and `inf` mean the same as `None`. Finite bounds must be ordered.
Log warps need finite positive bounds; open bounds need the linear warp.

Cover the values the solver may meet, with a small margin. Use
`warp="log10"` when a positive quantity spans decades: across
`(1e-6, 1e2)` a linear warp puts the midpoint at 50, `log10` puts it at
`1e-2`. An input warp must also hold at runtime, so a `log10` input needs
strictly positive values even when they come from the ODE state.
[Recommendations](/guide/recommendations#bounds) covers choosing the box
and the squash.

## Bounded networks

`BoundedPredictor` composes three parts: `in_scaler` maps physical inputs to
latents, `inner` evaluates the network, and `out_scaler` maps its output
back to physical units. The short form builds both scalers from bound
pairs:

```python
import jax.numpy as jnp
import jax.random as jr

rate = hm.BoundedPredictor(
    inner=hm.MLPPredictor(in_size=2, out_size=1, width_size=16, depth=2, key=jr.key(0)),
    input_keys=("temperature_C", "supersaturation"),
    input_bounds=((0.0, 100.0), (0.0, 5.0)),
    output_bounds=(0.0, None),
)
```

Pass `in_scaler=hm.BoundScaler(...)` and `out_scaler=...` instead when you
need a non-default `transform` or `warp`.

Call it with a mapping or a rank-1 array:

```python
rate({"temperature_C": 20.0, "supersaturation": 1.5})   # selects and orders by input_keys
rate(jnp.array([20.0, 1.5]))                            # already in input_keys order
```

A mapping may carry extra keys; a missing key raises. The output is in
physical units: a scalar when there is one output bound, as here, else a
rank-1 array with one entry per output.

To evaluate over a grid, pass arrays to `evaluate_predictor`. Inputs are
broadcast against each other and the predictor is called once per point:

```python
temps = jnp.linspace(0.0, 100.0, 50)
hm.evaluate_predictor(rate, {"temperature_C": temps, "supersaturation": 1.5})   # [50]
hm.evaluate_predictor(
    rate, {"temperature_C": temps[:, None], "supersaturation": jnp.linspace(0.0, 5.0, 20)}
)                                                                                # [50, 20]
```

A network needs at least one input. For a trainable value with no inputs,
use `Parameter`.

## Saturation

Near the edge of a finite box the output squash flattens and its gradient
vanishes. `Objective(penalty_weight=...)` charges `BoundedPredictor` leaves
for sitting there. `Parameter` leaves are not charged. See
[Penalties](/guide/penalties).

## Training and saving

The default mask trains `Parameter` latents and network weights, and
freezes `Constant` values and every `BoundScaler`. `fit(freeze=...)`
freezes more; see [Freezing leaves](/guide/training#freezing-leaves), and
[Saving and loading](/guide/serialization) for checkpoints.
