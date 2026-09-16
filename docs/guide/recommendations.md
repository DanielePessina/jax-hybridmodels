# Recommendations

These are practical defaults from the shipped examples. The library does not
enforce them; choose different values when your problem requires it.

## Bounds

Every quantity a `BoundedPredictor` produces lives in a box you declare,
`(low, high)`, in physical units. A freshly initialised network emits
values centred roughly around latent zero, which maps to the midpoint of that
box. A random readout can still move the initial output away from the
midpoint. Use `with_zero_final_head()` when the exact midpoint is important,
for example because the initial ODE is stiff.

### Centre the box where you would guess the answer

The common failure is a box so wide that its midpoint is physically
absurd. The integrator then hits `max_steps` before training does
anything. For example:

> A log nucleation-rate box of `(0, 15)` puts the midpoint at
> `J ≈ 3e7`, about 17,000 times too large, and can make the moment ODE
> intractably stiff at initialisation.

Pick the order of magnitude you would guess by hand, then add a few
decades of slack on each side.

### Give input bounds a small margin over the data

For an input box, go slightly wider than the observed span:
`temperature_C: (13.0, 27.0)` when the data runs 14 to 26 °C. The
squash saturates at the edges, and a margin of one or two units keeps
the gradient well conditioned across the whole measured range.

### Use a log warp when the box spans decades

`(1e-6, 1e2)` under the default linear warp has a midpoint of 50, and
every value below `1e-2` is crushed into a sliver of the latent range.
`warp="log10"` moves the midpoint to `1e-2` and spreads the decades
evenly. Bounds stay in physical units either way. See
[Predictors and bounds](/guide/predictors).

### Pin the readout when the box is very wide

Even a well-centred wide box, such as
`LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)` at 26 decades, is exposed to an
unlucky readout draw. The final layer's random weights can push the
initial output far enough off midpoint to make the ODE intractably
stiff.

[`MLPPredictor.with_zero_final_head()`](/api/predictors#mlppredictor)
and the matching `KANPredictor.with_zero_final_head()` return a copy
whose last layer is zeroed, so the initial output is the exact physical
midpoint whatever the key. Hidden layers keep their random
initialisation, so the input transformation is still non-degenerate.

```python
inner = MLPPredictor(in_size=2, out_size=1, width_size=64, depth=1,
                     activation_name="relu", key=k_growth).with_zero_final_head()
```

Reach for it when `RuntimeWarning: max_steps exceeded` from Diffrax
shows up on some seeds and not others.

### Pick the squash to match how close to the bounds you expect to sit

`transform="sigmoid"` loses its gradient to float32 underflow at a
latent of 16.8. `"algebraic"` survives to about `3e3` and is smooth to
all orders. `"softsign"` survives to about `1.1e7` but is not twice
differentiable, with the kink at the box midpoint.

Default to `sigmoid`. Move to `algebraic` when a predictor is expected
to work near its bounds, which is the usual case for a learned term
inside a vector field. It visits the edges of its box during the early,
badly fitted part of training, and `sigmoid` can be dead by the time the
rest of the model catches up.

## Solver tolerances

### Use per-state atol when your state spans decades

Population-balance moments span roughly 18 decades during an
integration. A single `atol` either over-resolves the small components
or under-resolves the large ones. Pass a tuple with one entry per state
component:

```python
solver = SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-4,
    atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),  # about 9 decades below each natural magnitude
    max_steps=500_000,
    dt0=None,
)
```

Call `solver.stepsize_controller()` inside your `simulate_fn` rather
than constructing a `PIDController` by hand. Diffrax needs an array for
a per-component `atol`, and a Python tuple is not one.

### Match the solver to the stiffness

- `diffrax.Tsit5()`. Explicit Runge-Kutta. The default for non-stiff
  problems.
- `diffrax.Kvaerno3()`. Implicit. Use it when `Tsit5` keeps hitting
  `max_steps`.
- `diffrax.Heun()`. Cheap and low order. Fine for a feasibility check.

[`SOLVER_REGISTRY`](/api/solver#solver_registry) carries these by name.
[`register_solver`](/api/solver#register_solver) adds your own so the
JSON round-trip still works.

### Pick the adjoint by memory, not by habit

The default `DirectAdjoint` stores the whole forward tape. That is the
cheapest way to differentiate and the most memory-hungry. If a network
runs inside your vector field, it is usually the binding constraint, and
`diffrax.RecursiveCheckpointAdjoint()` is the right swap.

```python
solver = SolverConfig(..., adjoint=diffrax.RecursiveCheckpointAdjoint())
```

Then use `solver.adjoint` in your `diffeqsolve` call, so the setting
takes effect.

### Leave dt0 as None

The step controller picks a first step from `rtol`, `atol`, and the
early derivative. Set `dt0` only when you have a reason.

## Enable x64 for stiff problems

Set `jax_enable_x64` before anything imports JAX. Float32 mass balance
drifts visibly within a single crystallisation experiment, usually
showing up as `mu0` going negative.

```python
import jax
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402  # x64 must be set before diffrax imports JAX dtypes.
```

The crystallisation example needs it. The harmonic oscillator does not.

## Structuring predictors

### Direct rate or kinetic parameters

Any rate-law problem gives you two parameterisations.

**Direct rate.** One predictor per rate, consuming
`(temperature, supersaturation, ...)` and emitting a bounded log-rate.
The network learns whatever mapping the data implies. Pick this when you
do not want to commit to a mechanism.

**Kinetic parameters.** A few bounded scalars (`logA`, `gamma`, `Ag`,
`g`) feeding a classical rate law inside the vector field. Much smaller
search space, and the fitted numbers mean something.

Both ship as sibling scripts.
[`train_kinetic.py`](/examples/crystallisation) is the direct-rate form
trained with Optax.
[`train_crystallisation_mechanistic.py`](/examples/crystallisation-mechanistic)
is the four-scalar form trained with CMA-ES.

### One BoundedPredictor per rate

Two rates means two `BoundedPredictor`s, each with its own bounds. At
the top of the vector field you write `growth, nucleation = predictors`
and call each with its own inputs. There is no framework class bundling
a pair of rates, and there should not be.

### Build one input dict and pass it to everything

Construct one `inputs` dict per solver step and hand it to every
predictor. Each one takes only the keys named in its `input_keys`;
extras are ignored.

```python
inputs = {
    "temperature_C": covariates["temperature_C"],
    "loading": covariates["loading"],
    "supersaturation": y[CONC_IDX] / covariates["c_sat"],
}
log10_G = growth(inputs)
log10_J = nucleation(inputs)
```

Vector covariates stay as vectors in `simulate_fn`. Index them when a
predictor expects scalar named inputs, or pass a rank-1 array to a predictor
that accepts vector inputs:

```python
composition = covariates["feed_composition"]
inputs = {
    "temperature_C": covariates["temperature_C"],
    "component_a": composition[0],
    "component_b": composition[1],
}
```

## Training schedule

### Start with one phase

`steps=(1000,), lr=(1e-3,), optimizer=("adamw",)` is the right default.
Add a second phase when the loss plateaus and you want a lower rate, or
when you want a `length_schedule` warm-up that fits the first half of
each trajectory before unmasking the rest.

### When to turn on the tournament

Turn it on with `tournament_attempts=8, tournament_steps=20` if you see
either of:

- `RuntimeWarning: max_steps exceeded` from Diffrax;
- a stable but pathologically high final loss across several seeds.

It reuses the main loop's compiled step function, so it adds no
compilation cost.

### When to reach for evosax

Use `train_with_evosax` when the trainable set is small (roughly 4 to 10
parameters), kinetic-parameter shaped, or known to be multi-modal. Run
it first, then refine with `train_with_optax`. Both take the same
`trainable=` mask.

## Freezing

### Freeze BoundScaler.temperature

Every example freezes all `BoundScaler` leaves.

```python
from hybridmodels import BoundScaler, freeze_modules_of_type, trainable_mask

mask = trainable_mask(predictors)
mask = freeze_modules_of_type(mask, predictors, BoundScaler)
```

`temperature` is technically a trainable leaf. Training it moves signal
back and forth between the scaler and the inner network and slows
convergence. The box is the contract; the temperature is plumbing.

### Freeze everything for a wiring check

Freeze the entire model and run three steps. The loss must be
byte-identical at every step, because nothing moved. If it changes, some
part of your pipeline is not doing what you think.

```python
from hybridmodels import BoundedPredictor, freeze_where, trainable_mask

mask = freeze_where(trainable_mask(predictors), predictors,
                    lambda m: isinstance(m, BoundedPredictor))
```

Note how `freeze_where` reads. Its predicate is applied to every module
node in the tree, and any node that matches has its whole subtree
frozen. `BoundedPredictor` is the outermost node, so matching it freezes
everything below. A predicate like
`lambda m: not isinstance(m, BoundScaler)` also matches the outer
`BoundedPredictor` and therefore freezes the entire model, which is
almost never what someone writing it intends.

To freeze only the inner networks and leave the scalers free, name the
inner class:

```python
mask = freeze_where(trainable_mask(predictors), predictors,
                    lambda m: isinstance(m, MLPPredictor))
```

## Traps

### Guarded division must guard the divisor, not the result

A naive `jnp.where(divisor > eps, num / divisor, 0.0)` still evaluates
`num / divisor` on the discarded branch. That produces `inf` or `nan`,
whose gradient flows back through `jnp.where` and poisons the loss.
Replace the divisor before dividing:

```python
safe_divisor = jnp.where(divisor > eps, divisor, 1.0)
result = jnp.where(divisor > eps, num / safe_divisor, 0.0)
```

The discarded branch now evaluates `num / 1.0`, which is finite, so the
gradient is finite on both sides and the outer `where` still selects
correctly. The crystallisation example uses this in its
`state_to_output` when computing `d43 = mu4 / mu3`.

The same applies to `log`. Clip the argument inside the log, not the
result, or the gradient is `nan` even when a mask discards the value.

### Every experiment must define every channel

`make_dataset` requires each experiment to have all the channels named
in `output_channel_names`. If a sensor was offline, you cannot pad with
zeros. Drop the experiment.

```python
if "d43" not in channels:
    continue  # skip experiments missing the particle-size channel
```

### Time units must match end to end

`Experiment.channels[*].ts`, the `ts` argument to `simulate_fn`, and the
rate constants in your vector field all share one unit. The
crystallisation example stores time in minutes and converts to seconds
inside the vector field. If you do that, say so loudly in a comment.

### key= is keyword-only

`train_with_optax(predictors, dataset, config, key)` raises `TypeError`.
The barrier is deliberate. It makes the seed visible at every call site.
