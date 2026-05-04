# Recommendations

Practical guidance accumulated from the canonical examples. None of this is enforced by the framework; all of it will save you a debugging session.

## Bounds

### Pick bounds that put a *random* predictor in a sensible regime

`BoundedPredictor`'s out-scaler maps the latent zero (the sigmoid midpoint) to the geometric centre of the box. A random-init inner predictor lands near zero in latent space, so the *centre* of your output bounds is what the integrator sees on step 0.

A common failure mode: bounds that are too wide push the centre into a stiff regime and the integrator hits `max_steps` before training even starts. The crystallisation example documents this explicitly:

> Earlier draft used `(0, 15)`; that put the midpoint at `J ≈ 3e7` — \~17,000× too large, which made the moment ODE intractably stiff at random init even though the same ODE solves cleanly with sensible kinetic parameters.

**Rule of thumb:** centre your bounds on the order of magnitude you'd choose if you were guessing the answer by hand, then add a few decades of slack on either side.

### Slightly wider than the data span

For input bounds (e.g. `temperature_C: (13.0, 27.0)` when the data spans 14-26°C), give yourself a margin of 1-2 units on each side. Sigmoid saturates at the boundary; even a small margin keeps the gradient well-conditioned across the whole observed range.

### When the bounds span many decades, pin the readout

Wide rate bounds (e.g. `LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)` — a 26-decade range) are vulnerable to bad random readouts even after the bounds are well-centred: an unlucky standard-normal final-layer draw can place the initial output far enough off midpoint that the moment ODE is intractably stiff. [`MLPPredictor.with_zero_final_head()`](/api/predictors#mlppredictor) (and the matching `KANPredictor.with_zero_final_head()`) returns a copy with the trailing readout layer's weights zeroed, so the initial output sits at the *exact* physical midpoint regardless of the random key. The hidden layers keep their default init, so the input feature transformation is non-degenerate. Compose by chaining onto the constructor:

```python
inner = MLPPredictor(in_size=2, out_size=1, width_size=64, depth=1,
                     activation_name="relu", key=k_growth).with_zero_final_head()
```

Reach for it when you see `RuntimeWarning: max_steps exceeded` from `diffrax` only on certain seeds.

## Solver tolerances

### Per-state `atol` for stiff systems

If your state components span many orders of magnitude — population-balance moments are the canonical example — a uniform `atol` either over-resolves the small components or under-resolves the large ones. Pass a tuple matching `S`:

```python
solver = SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-4,
    atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),  # per-state floor, ~9 decades below natural magnitude
    max_steps=500_000,
    dt0=None,
)
```

### Pick a solver to match stiffness

- **`diffrax.Tsit5()`** — explicit RK, default for non-stiff problems.
- **`diffrax.Kvaerno3()`** — implicit, good for moderately stiff problems where `Tsit5` is hitting `max_steps`.
- **`diffrax.Heun()`** — cheap, low-order; reasonable for very fast feasibility tests.

[`SOLVER_REGISTRY`](/api/solver#solver_registry) carries these by name; [`register_solver`](/api/solver#register_solver) extends it for custom solvers (so the JSON round-trip still works).

### `dt0=None` lets the controller pick

Leave `dt0=None` unless you have a specific reason to seed an initial step size. The PID step controller picks something reasonable from `rtol`/`atol` and the early derivative.

## x64 by default for stiff problems

Set `jax_enable_x64` **before any JAX-touching import** in stiff or moments-style problems:

```python
import jax
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402  # x64 must be set before diffrax imports JAX dtypes.
```

float32 mass balance can drift visibly within a single experiment — most often as `mu0` going negative. The crystallisation example needs x64 to stay stable; the harmonic oscillator does not.

## Predictor structure

### Direct-rate vs kinetic-parameter

You have two parameterisations to choose from in any rate-law problem:

- **Direct-rate.** One predictor per rate, consuming `(temperature, loading, supersaturation, ...)` and emitting a bounded log-rate. The network learns whatever `(inputs) → rate` mapping the data implies. Best when you don't want to commit to a kinetic mechanism.
- **Kinetic-parameter.** One predictor consuming a smaller covariate set and emitting a few bounded scalars (`logA`, `gamma`, `Ag`, `g`). The vector field plugs these into classical CNT / power-law forms. Closer to a classical surrogate; smaller search space.

The crystallisation example ships both, with the kinetic-parameter path preserved as commented-out reference code so flipping between them takes one line.

### One BoundedPredictor per rate, not one per scalar

If your model has two rates (growth and nucleation), use **two** `BoundedPredictor`s. Inside the vector field you `growth, nucleation = predictors` and call each with its own bounds. There is no `RatePair` framework class — and there shouldn't be.

### Use the shared input dict

Construct one `inputs` dict per timestep inside the vector field and call every predictor with it. The framework subsets by `input_keys` automatically; extra keys are ignored:

```python
inputs = {
    "temperature_C": covariates["temperature_C"],
    "loading": covariates["loading"],
    "supersaturation": y[CONC_IDX] / covariates["c_sat"],
}
log10_G = growth(inputs)        # uses ("temperature_C", "loading", "supersaturation")
log10_J = nucleation(inputs)    # uses ("temperature_C", "loading", "supersaturation")
```

## Training schedule

### Start with one phase

A single-phase Optax run with `(steps=1000, lr=1e-3, optimizer="adamw")` is the right default. Add a second phase when:

- Your loss plateaus and you want a lower LR.
- You want a `length_schedule` warm-up: train on the first 50% of each trajectory to get the early dynamics right, then unmask the rest.

### When to reach for the tournament

Random init can be unlucky. If you see:

- `RuntimeWarning: max_steps exceeded` from `diffrax`, or
- A stable but pathologically high final loss across multiple seeds,

turn on the shared tournament with `tournament_attempts=8, tournament_steps=20`. It's cheap: shared with the main loop's compiled `make_step`, so no extra JIT cost.

### When to reach for evosax

If your trainable surface is small (~4-10 parameters), kinetic-parameter style, or has a known multi-modal landscape, `train_with_evosax` is the right tool. Run it first, then polish with `train_with_optax` — both accept the same `trainable=` mask.

## Freezing

### Always freeze BoundScaler.temperature

By convention, every example freezes all `BoundScaler` leaves:

```python
from hybridmodels import BoundScaler, freeze_modules_of_type, trainable_mask

mask = trainable_mask(predictors)
mask = freeze_modules_of_type(mask, predictors, BoundScaler)
```

The `temperature` field is technically a leaf, but training it confuses the optimiser; the box-bounds are the contract, the temperature is plumbing.

### Freeze the inner network for sanity-check runs

Freezing everything **except** the bound parameters of one predictor is a useful diagnostic — it lets you check the integrator/loss path is wired correctly without any actual learning happening:

```python
mask = trainable_mask(predictors)
mask = freeze_where(mask, predictors, lambda m: not isinstance(m, BoundScaler))
```

If the loss still drops, you have a wiring bug. (It usually does on the first run.)

## Pitfalls

### Autodiff-safe guarded division

A naive `jnp.where(divisor > eps, num / divisor, 0.0)` still computes `num / divisor` on the masked-out branch, producing `inf` / `nan` whose gradient flows back through `jnp.where` and poisons the loss. Wrap the divisor:

```python
safe_divisor = jnp.where(divisor > eps, divisor, 1.0)
result = jnp.where(divisor > eps, num / safe_divisor, 0.0)
```

The unused branch evaluates `num / 1.0 = num` (finite), so the gradient is finite on both sides and the outer `where` selects correctly. The crystallisation example uses this pattern in `_d43_from_moments`.

### `make_dataset` validation requires every channel

Every experiment must define every output channel listed in `output_channel_names`. If a sensor is offline for an experiment, you can't pad with zeros — drop the experiment, or skip it at load time:

```python
if "d43" not in channels:
    continue  # skip experiments missing the particle-size channel
```

### Time units must match across the pipeline

`Experiment.channels[*].ts`, the `ts` argument to `simulate_fn`, and the rate constants inside your vector field all share the same unit. The crystallisation example keeps time in **minutes** in the dataset and converts to seconds inside the vector field; if you do that, document it loudly.

### `key=` is keyword-only

Calling `train_with_optax(predictors, dataset, config, key)` raises `TypeError`. The keyword-only barrier is deliberate — it makes seed plumbing visible at every call site.

## What's next

- [Crystallisation walkthrough](/examples/crystallisation) — every recommendation here, applied to a real problem.
- [API Reference](/api/) — full surface, by module.
