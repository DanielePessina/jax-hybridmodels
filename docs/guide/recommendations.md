# Recommendations

Defaults that worked in the shipped examples. None are enforced.

## Bounds

### Centre the box on your guess

A fresh network outputs a latent near zero, which maps to the middle of the
output box. If the middle is physically absurd, the first solve can be so
stiff that it hits `max_steps` before training starts. A log10 nucleation
box of `(0, 15)` starts at `J ≈ 3e7`, about 17,000 times too large. Pick the
order of magnitude you would guess by hand and allow a few decades either
side, with `warp="log10"` if the box spans decades.

### Leave a margin on input boxes

Make input boxes slightly wider than the data, for example
`(13.0, 27.0)` for temperatures measured from 14 to 26 °C. The squash is
flattest at the edges, so a margin keeps the gradient usable across the
measured range.

### Zero the readout when the box is very wide

The last layer's random weights can still push the starting output far
from the middle of a wide box, such as the crystallisation example's
`LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)`. `with_zero_final_head()` on an
`MLPPredictor` or `KANPredictor` zeroes the last layer, so the initial
output is exactly the midpoint whatever the key. Hidden layers stay random.

```python
inner = hm.MLPPredictor(
    in_size=2, out_size=1, width_size=64, depth=1, activation_name="relu", key=key
).with_zero_final_head()
```

Use it when Diffrax reports `max_steps` on some seeds and not others.

### Pick the squash for how close to the edge you will sit

In float32, `transform="sigmoid"` loses its gradient at a latent of about
17, `"algebraic"` at about `3e3`, and `"softsign"` at about `1e7`.
`softsign` has a kink at the box midpoint, which solver steps cross often.
Use `sigmoid` by default and `algebraic` for a learned term inside a vector
field: those visit their bounds early in training, and a sigmoid can be
saturated before the rest of the model catches up.

## Solver

### Per-state `atol` when states span decades

Population-balance moments span about 18 decades, so one `atol` is wrong
for most of them. Pass one entry per state:

```python
solver = hm.SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-4,
    atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),   # about 9 decades below each state's scale
    max_steps=500_000,
    dt0=None,
)
```

`solver.diffeqsolve` and `solver.stepsize_controller()` convert the tuple
to the array Diffrax needs. A hand-built `PIDController` would not.

### Solver choice

Start with `diffrax.Tsit5()`. Switch to the implicit `diffrax.Kvaerno3()`
when `Tsit5` keeps hitting `max_steps`. `diffrax.Heun()` is cheap and fine
for checking the wiring. `dt0=None` lets the controller choose the first
step.

### Adjoint

The default `DirectAdjoint` keeps the whole forward pass in memory and its
compile time grows with `max_steps`. When a network runs inside the vector
field, memory usually runs out first; use
`adjoint=diffrax.RecursiveCheckpointAdjoint()`. `solver.diffeqsolve`
passes the adjoint on; a hand-written `diffrax.diffeqsolve` call must pass
`adjoint=solver.adjoint` itself.

### 64-bit floats for stiff problems

Enable x64 before anything else imports JAX:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

In float32, mass balance in the crystallisation model drifts within one
experiment, typically showing up as `mu0` going negative. The harmonic
oscillator does not need x64.

## Structuring predictors

### Direct rates or kinetic parameters

A rate can be a network of the conditions, emitting a bounded log-rate, or
a classical rate law with a few `Parameter`s. The network commits to no
mechanism. The rate law has a much smaller search space and fitted values
with physical meaning. The crystallisation examples show both:
[`train_kinetic.py`](/examples/crystallisation) trains networks with Optax,
and
[`train_crystallisation_mechanistic.py`](/examples/crystallisation-mechanistic)
fits four scalars with CMA-ES.

### One input dict per step

Build one dict of inputs in the vector field and pass it to every
predictor. Each one takes the keys in its `input_keys` and ignores the
rest:

```python
inputs = {
    "temperature_C": covariates["temperature_C"],
    "supersaturation": y[CONC_IDX] / covariates["c_sat"],
}
log10_G = predictors["growth"](inputs)[0]
log10_J = predictors["nucleation"](inputs)[0]
```

Index vector covariates into named entries (`composition[0]`), or pass a
rank-1 array straight to a predictor that takes the whole vector.

## Training

Start with one phase, `OptaxTrainingConfig(steps=1000, lr=1e-3)`. Add a
lower-rate phase when the loss plateaus, or a first phase with
`horizon=0.5` when the tail of each trajectory diverges early on.

Turn on the tournament (`tournament_attempts=8, tournament_steps=20`) when
`max_steps` errors appear on some seeds, or when several seeds all settle
at a poor loss.

Use Evosax for roughly 4 to 10 kinetic scalars, then refine with Optax.
See [Training](/guide/training#evosax-then-optax).

### Check the wiring with everything frozen

Freeze every network and run a few steps. The loss must be identical at
every step; if it moves, something you did not intend is training.

```python
mask = hm.freeze_modules_of_type(hm.trainable_mask(predictors), predictors, hm.BoundedPredictor)
```

`Parameter` leaves are not inside a `BoundedPredictor`; freeze them with
`freeze_modules_of_type(..., hm.Parameter)` as well. `model.fit` refuses a
mask with nothing left to train, so use `train_with_optax` for this check.

## Traps

### Guard the divisor, not the result

`jnp.where(d > eps, num / d, 0.0)` still computes `num / d` on the branch it
discards. That `inf` or `nan` reaches the gradient through `jnp.where`.
Replace the divisor first:

```python
safe_d = jnp.where(d > eps, d, 1.0)
result = jnp.where(d > eps, num / safe_d, 0.0)
```

The crystallisation example does this for `d43 = mu4 / mu3`. The same goes
for `log`: clip the argument, not the result.

### Missing channels

Every experiment must list every channel in `output_channel_names`. For a
channel that was not measured, pass an empty observation; it is masked
out of the loss. Do not pad with zeros, which would be fitted as data.

```python
channels["d43"] = hm.ChannelObs(jnp.array([]), jnp.array([]))
```

### One time unit everywhere

Observation times, the `ts` that `simulate_fn` receives, and the rate
constants in the vector field must share a unit. If you convert inside the
vector field, say so in a comment.
