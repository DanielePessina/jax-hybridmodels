---
name: jax-hybrid-models
description: Uses the jax-hybridmodels package (JAX/Equinox hybrid ODE + neural models): building experiments and datasets, writing simulate_fn and state_to_output, configuring BoundedPredictor bounds/warps, training with Optax or evosax, prediction, and serialisation. Use when working in this repo or writing code that imports hybridmodels, or when asked to build, train, evaluate, debug, or tune hybrid ODE-neural models with bounded trainable predictors on irregular time-series data.
---

# jax-hybridmodels

JAX/Equinox library: user-written ODE dynamics with trainable predictors inside, trained on irregular time-series experiments. Crystallisation kinetics is the canonical example, not the scope.

## What a "model" is (there is no `Model` class)

Five pieces passed separately to training/prediction:

| Piece | What it is |
|---|---|
| `predictors` | pytree of `eqx.Module` leaves; convention: `tuple` of `BoundedPredictor`, one per rate |
| `simulate_fn` | **your** pure function integrating ONE experiment → `[T, S]` |
| `state_to_output` | `[T, S]` → `[T, D]` (measured channels); belongs to the model, not the data |
| `solver` | `SolverConfig` (static diffrax settings) |
| `dataset` | experiments bucketed by `make_dataset`; pure data, no `state_to_output` |

The framework owns vmap/jit/grad over `simulate_fn`; the user owns physics.

## The `simulate_fn` contract (mandatory signature)

```python
def simulate_fn(predictors, ts, covariates, y0, solver) -> Float[Array, "T S"]:
```

- `predictors`: pytree of modules — call inside the vector field: `rate = predictors[0]({"temperature_C": ...})`
- `ts` `[T]`, `covariates` dict (constant in time), `y0` `[S]` (full state, may exceed measured channels)
- `solver`: `SolverConfig` — call `solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys`; use `solver.adjoint` and `solver.stepsize_controller()` inside
- Must be pure and JAX-transformable: no Python `if` on traced values, no side effects
- The framework vmaps it across the bucket, jits it, differentiates through it

## Quick start

```python
import diffrax, jax.numpy as jnp, jax.random as jr
import hybridmodels as hm

key = jr.PRNGKey(0)
k_init, k_train = jr.split(key, 2)

# 1. Predictor: named inputs -> bounded physical output.
inner = hm.MLPPredictor(in_size=1, out_size=1, width_size=64, depth=1,
                        activation_name="relu", key=k_init)
predictor = hm.BoundedPredictor(
    input_keys=("temperature_C",),
    in_scaler=hm.BoundScaler(bounds=((10.0, 40.0),), transform="sigmoid"),
    inner=inner,
    out_scaler=hm.BoundScaler(bounds=((1e-3, 1e3),), warp="log10", transform="algebraic"),
)
predictors = (predictor,)  # tuple even when there is only one

# 2. simulate_fn with the mandatory signature.
def simulate_fn(predictors, ts, covariates, y0, solver):
    rate = predictors[0](covariates)
    def vector_field(t, y, args):
        return -rate * y[0] + y[1]
    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys

# 3. state_to_output: drop unmeasured state components.
def state_to_output(state):
    return state[..., :1]

# 4. Data: one Experiment per physical run.
exps = [
    hm.make_experiment(
        covariates={"temperature_C": 25.0},
        channels={"conc": hm.ChannelObs(ts=ts, values=y, variance=v)},
        y0_fn=lambda c, ch: jnp.array([ch["conc"].values[0], 0.0]),
        exp_id=f"run_{i}",
    )
    for i, (ts, y, v) in enumerate(raw_measurements)
]
dataset = hm.make_dataset(exps, output_channel_names=("conc",))

# 5. Train (key= is keyword-only, no default).
solver = hm.SolverConfig(solver=diffrax.Tsit5(), rtol=1e-4, atol=1e-6, max_steps=50_000)
config = hm.OptaxTrainingConfig(steps=(1000,), lr=(1e-3,), optimizer=("adamw",),
                                reset_optimiser_state=(False,), loss="mse")
history, trained = hm.train_with_optax(predictors, dataset, config,
                                       simulate_fn=simulate_fn,
                                       state_to_output=state_to_output,
                                       solver=solver, key=k_train)
```

Full runnable version: `examples/pendulum/train_harmonic.py`; canonical end-to-end: `examples/crystallisation/train_kinetic.py`.

## Rules that agents get wrong (details in caveats.md)

1. **Freeze `BoundScaler.temperature`** — `mask = freeze_modules_of_type(trainable_mask(predictors), predictors, BoundScaler)`. Every example does this; training it slows convergence.
2. **Bounds**: finite, `low < high`, physical units. Centre the box at your best guess + slack; a huge box whose midpoint is physically absurd makes the ODE intractably stiff at init. Decade-spanning bounds → `warp="log10"` (log warps reject non-positive bounds).
3. **`transform="algebraic"`** for learned terms expected near their bounds (typical in a vector field): sigmoid's gradient dies at latent 16.8. Wide boxes + unlucky init → `MLPPredictor(...).with_zero_final_head()` to pin the initial output at the box midpoint.
4. **`key=` is keyword-only and required** on both trainers. Never `jr.PRNGKey(0)` defaults.
5. **Step = one full pass over all buckets** (one epoch). Bucket ≠ step. Phase fields (`steps`, `lr`, ...) are equal-length tuples, no scalar broadcast.
6. **Every experiment must define every output channel.** Sensor offline? Drop the experiment, never pad zeros.
7. **Guarded division guards the divisor**, not the result: `safe = jnp.where(d > eps, d, 1.0); r = jnp.where(d > eps, num / safe, 0.0)`. Same for `log` — clip the argument.
8. **Stiff / mass-balance problems: enable x64 first** (`jax.config.update("jax_enable_x64", True)` before importing jax/diffrax). Float32 drifts visibly, e.g. `mu0` going negative.
9. **Serialisation**: `load_run`/`load_predictors` need a template with the same container shape, module types, and static fields. Trainable configs with callables come back as metadata markers — rebind explicitly.
10. **Turn on the tournament** (`tournament_attempts=8, tournament_steps=20`) when you see `RuntimeWarning: max_steps exceeded` or stable-but-pathological loss. No extra compile cost.

## Navigation

- [usage.md](usage.md) — full API walkthrough with code: data, predictors, training configs, prediction, serialisation, losses
- [caveats.md](caveats.md) — traps and tuning advice with the reasoning
- Repo: `examples/` for working code, `docs/guide/*.md` for prose, `CONTEXT.md` for terminology, `SPEC.md` for the contract
- Toolchain in this repo: `uv run pytest`, `uv run ruff check .`, `uv run ty check src`. Never `pip`/`pytest` directly.