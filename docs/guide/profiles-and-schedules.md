# Profiles and schedules

A profile is a physical input that changes during one ODE solve. A
schedule is a hyperparameter that changes smoothly over training. Both are
pure JAX callables.

## Time-varying inputs

Covariates are constant during an experiment. Store a changing quantity's
parameters as covariates, build the profile inside `simulate_fn`, and
evaluate it at the solver's time `t`.

```python
import diffrax
import jaxhybridmodels as hm


def simulate_fn(predictors, ts, covariates, y0, solver):
    temperature = hm.ramp_profile(
        start=covariates["heat_start"],
        end=covariates["heat_end"],
        before=covariates["temperature_initial"],
        after=covariates["temperature_final"],
    )

    def vector_field(t, y, args):
        inputs = {
            "temperature": temperature(t),
            "concentration": y[0],
        }
        rate = predictors["rate"](inputs)
        return physics_rhs(t, y, rate)

    return solver.diffeqsolve(
        diffrax.ODETerm(vector_field), ts, y0, jump_ts=temperature.jump_ts
    ).ys
```

`jump_ts` lists the times where a profile jumps or its slope changes. An
adaptive solver assumes the vector field is smooth, so without them it rejects
steps around a step change, and it can step over a pulse shorter than a step
without seeing it. Passed to `diffeqsolve`, they end a step on each. Join
several profiles' times with `jnp.concatenate`.

The built-in factories are:

| Factory | Behaviour | `jump_ts` |
| --- | --- | --- |
| [`constant_profile`](/api/profiles#constant_profile) | Returns the same value for every `t`. | none |
| [`step_profile`](/api/profiles#step_profile) | Returns `before` until `at`, then `after`. | `at` |
| [`ramp_profile`](/api/profiles#ramp_profile) | Holds `before` until `start`, ramps linearly to `after` by `end`, then holds it. | `start`, `end` |
| [`piecewise_linear_profile`](/api/profiles#piecewise_linear_profile) | Interpolates knots and extends the first and last values outward. | every knot |

`ramp_profile` requires `end > start`; `piecewise_linear_profile` requires at
least two strictly increasing knots. Concrete arguments are checked when the
profile is built; traced covariates skip the check and stay JIT-safe.

## Smooth training schedules

`TrainingPhase.lr` takes an optax schedule, `step -> lr`, counted from the
first step of the phase. Any `optax` schedule works, and several chain into
one with `optax.join_schedules`; see
[Learning-rate schedules](/guide/training#learning-rate-schedules).

[`annealing_schedule`](/api/schedules#annealing_schedule), in the
`jaxhybridmodels.schedules` submodule, returns a multiplier with the run
length baked in. Scale it into a learning rate:

```python
schedule = hm.schedules.annealing_schedule(
    "warmup_cosine",
    total_steps=n_steps,
    init_value=1.0,
    end_value=0.1,
    warmup_steps=5,
)
phase = hm.TrainingPhase(steps=n_steps, lr=lambda step: base_lr * schedule(step))
```

`n_steps` is the phase's step budget and `base_lr` its peak learning rate.
One step is one pass over all buckets, so `total_steps` is the step budget.
`"warmup_cosine"` rises from 0 to `init_value` over `warmup_steps`, then
decays to `end_value`; the other kinds start at `init_value`. A custom loop
can call the same multiplier to ramp a penalty weight.

## Which boundary owns what?

| Quantity | Store it as | Evaluate it in |
| --- | --- | --- |
| Constant per-experiment condition | `Experiment.covariates` | `simulate_fn` |
| Changing physical quantity | Profile parameters in covariates plus a profile factory | The vector field at time `t` |
| Piecewise training strategy | `OptaxTrainingConfig(phases=...)` | `fit` |
| Smooth learning rate | An optax schedule or `annealing_schedule` as `TrainingPhase.lr` | `fit` |
| Other smooth hyperparameters | `annealing_schedule` | A custom loop |
