# Profiles and schedules

Use a profile when a physical quantity changes during one ODE solve. Use an
annealing schedule when a custom training loop needs a smooth change in a
hyperparameter. Both are pure callables and compose with JAX transformations.

## Time-varying inputs

Dataset covariates are constant during an experiment. Store the parameters of
a changing quantity as ordinary covariates, build a profile inside
`simulate_fn`, and evaluate it at the solver's continuous time.

```python
import diffrax
import hybridmodels as hm


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
        rate = predictors[0](inputs)
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
least two strictly increasing knots. Host-side parameters are validated when
possible, while traced covariate parameters remain JIT-safe.

## Smooth training schedules

The stock trainers express changes as Optax phases. A custom loop can use
[`annealing_schedule`](/api/schedules#annealing_schedule) when a continuous
curve is more appropriate:

```python
schedule = hm.annealing_schedule(
    "warmup_cosine",
    total_epochs=n_steps,
    init_value=1.0,
    end_value=0.1,
    warmup_epochs=5,
)

for step in range(n_steps):
    lr = base_lr * schedule(step)
    # Build or update the custom loop using this lr.
```

The example assumes `n_steps` is the custom loop's budget and `base_lr` is
its initial learning rate. The schedule is an epoch-scaled multiplier. Here
one library step is one full pass over all buckets, so `total_epochs` is
normally the custom loop's step budget. The stock phase configuration does
not consume this helper.

## Which boundary owns what?

| Quantity | Store it as | Evaluate it in |
| --- | --- | --- |
| Constant per-experiment condition | `Experiment.covariates` | `simulate_fn` |
| Changing physical quantity | Profile parameters in covariates plus a profile factory | The vector field at time `t` |
| Piecewise training strategy | Optax phase tuples | `train_with_optax` |
| Smooth training strategy | `annealing_schedule` | A custom loop |
