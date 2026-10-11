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
        t0=covariates["heat_start"],
        t1=covariates["heat_end"],
        v0=covariates["temperature_initial"],
        v1=covariates["temperature_final"],
    )

    def vector_field(t, y, args):
        inputs = {
            "temperature": temperature(t),
            "concentration": y[0],
        }
        rate = predictors["rate"](inputs)
        return physics_rhs(t, y, rate)

    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys
```

The built-in factories are:

| Factory | Behaviour |
| --- | --- |
| [`constant_profile`](/api/profiles#constant_profile) | Returns the same value for every `t`. |
| [`step_profile`](/api/profiles#step_profile) | Returns `before` before `jump_at`, then `after`. |
| [`ramp_profile`](/api/profiles#ramp_profile) | Holds `v0`, ramps linearly to `v1`, then holds `v1`. |
| [`piecewise_linear_profile`](/api/profiles#piecewise_linear_profile) | Interpolates knots and extends the first and last values outward. |

`ramp_profile` requires `t1 > t0`; `piecewise_linear_profile` requires at
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
    total_epochs=n_steps,
    init_value=1.0,
    end_value=0.1,
    warmup_epochs=5,
)
phase = hm.TrainingPhase(steps=n_steps, lr=lambda step: base_lr * schedule(step))
```

`n_steps` is the phase's step budget and `base_lr` its peak learning rate.
One step is one pass over all buckets, so `total_epochs` is the step budget.
`"warmup_cosine"` rises from 0 to `init_value` over `warmup_epochs`, then
decays to `end_value`; the other kinds start at `init_value`. A custom loop
can call the same multiplier to ramp a penalty weight.

## Which boundary owns what?

| Quantity | Store it as | Evaluate it in |
| --- | --- | --- |
| Constant per-experiment condition | `Experiment.covariates` | `simulate_fn` |
| Changing physical quantity | Profile parameters in covariates plus a profile factory | The vector field at time `t` |
| Piecewise training strategy | `OptaxTrainingConfig(phases=...)` | `fit` or `train_with_optax` |
| Smooth learning rate | An optax schedule or `annealing_schedule` as `TrainingPhase.lr` | `fit` or `train_with_optax` |
| Other smooth hyperparameters | `annealing_schedule` | A custom loop |
