# Extending

Each piece the trainers use is a callable or pytree you can replace. This
page lists the extension points and the interface each one expects.

## Predictors

Subclass `Predictor` and implement `__call__`. See
[Custom predictors](/guide/custom-predictors) and the
[worked example](/examples/custom-predictor).

## `simulate_fn` and `state_to_output`

`simulate_fn(predictors, ts, covariates, y0, solver) -> [T, S]` integrates
one experiment. It runs under `vmap` and `jit`, so it must be pure JAX.
`SolverConfig.diffeqsolve` keeps it short and forwards `solver.adjoint`:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    def vector_field(t, y, args):
        return -predictors["rate"].value * y

    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys
```

`state_to_output` maps `[T, S]` to the observed channels `[T, D]`. See
[Model interface](/guide/model-interface).

## Losses

A loss is `loss(pred_obs, bp) -> scalar`, passed as the config's `loss`:

```python
config = hm.OptaxTrainingConfig(steps=500, loss=my_huber_loss)
```

With `channel_idx`, a loss whose signature accepts `channel_idx` and
`channel_weights` receives them as keywords. A plain `(pred_obs, bp)` loss
receives only the selected channels, already sliced; combining it with
`channel_weights` raises, since the weights have to be applied inside the
loss. See `jaxhybridmodels.losses.resolve_loss_fn`.

## Optimisers

`optimizer` on a config or `TrainingPhase` accepts:

- a name, `"adamw"` or `"adabelief"`;
- a factory taking `learning_rate`, such as `optax.adamw` or
  `lambda learning_rate: optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))`;
- a ready-made `optax.GradientTransformation`.

Names and factories are wrapped in `optax.inject_hyperparams`, so a later
phase can change `lr` in place. A ready-made instance cannot, so a phase
that changes `lr` on one needs `reset_optimizer=True`.

## Regularisers

`penalty_fn(predictors, points) -> scalar` replaces the bound penalty on
both config types. It is charged once per step with weight
`penalty_weight`. See [Penalties](/guide/penalties#replacing-it).

## Trajectory-aware penalties

For a predictor inside the vector field, carry its saturation as extra ODE
state and charge the integral. The helpers in `jaxhybridmodels.penalties`
add, wrap and remove that state. A sketch for one embedded `BoundedPredictor`
whose input is state 0:

```python
from jaxhybridmodels.penalties import (
    attach_penalty_state, penalty_integral, penalty_vector_field, strip_penalty_state,
)

def simulate_fn(predictors, ts, covariates, y0, solver):
    rate = predictors["rate"]

    def physics(t, y, args):
        return -rate(y[:1]) * y

    def penalty_rates(t, y, args):
        z = rate.inner(rate.in_scaler.to_latent(y[:1]))
        return jnp.stack([
            jnp.sum(rate.out_scaler.saturation(z)),       # output pinned at a bound
            jnp.sum(rate.in_scaler.input_violation(y[:1])),  # input outside its box
        ])

    term = diffrax.ODETerm(penalty_vector_field(physics, penalty_rates))
    return solver.diffeqsolve(term, ts, attach_penalty_state(y0, n=2)).ys

def state_to_output(state):
    return strip_penalty_state(state, n=2)          # the loss never sees the accumulators

def trajectory_penalty(full_state, bp):
    return jnp.sum(penalty_integral(full_state, n=2))

config = hm.OptaxTrainingConfig(
    steps=500, trajectory_penalty_fn=trajectory_penalty, trajectory_penalty_weight=0.1
)
```

`trajectory_penalty_fn` gets the full state, accumulators included, and is
added to the data loss in the same forward pass. For a predictor whose
output is itself a channel, `trajectory_saturation_penalty(state,
out_scaler)` does the job without extra state.

## Training loops

The trainers are built from kernels in `jaxhybridmodels.training.kernels`:
`build_bucket_step` returns jitted `(loss, grads)` for one bucket,
`build_score_bucket` a forward-only loss, `build_penalty_step` the
per-step regulariser, and `build_apply_update` one optimiser update. One
step of the stock loop, without the optimiser setup and UI:

```python
step = build_bucket_step(
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, loss_fn=loss_fn, trainable=mask,
)
apply_update = build_apply_update(optimizer, mask)

buckets = dataset.bucket_payloads
grads = [step(predictors, bp, horizon)[1] for bp in buckets]
mean_grads = jax.tree.map(lambda *g: sum(g) / len(g), *grads)
predictors, opt_state = apply_update(predictors, mean_grads, opt_state)
```

The [custom loop example](/examples/custom-loop) is the full version, with
per-bucket weights and a custom regulariser.

The stock trainers change hyperparameters only at phase boundaries. For a
smooth schedule in your own loop, `annealing_schedule` returns a multiplier
over a fixed run length:

```python
from jaxhybridmodels.schedules import annealing_schedule

lr_scale = annealing_schedule("cosine", total_epochs=n_steps)
lr = base_lr * lr_scale(step)
```

## Several related systems

To share one model across related systems (a catalyst family, a set of
reactors), keep the system identity as a covariate and add a small
trainable embedding as another predictor leaf. The vector field feeds the
embedding's output to the rate network like any other input:

```python
predictors = {
    "embed": eqx.nn.Linear(2, 2, key=k1),
    "rate": hm.BoundedPredictor(inner=..., input_keys=("T", "z0", "z1"), ...),
}

def vector_field(t, y, args):
    z = predictors["embed"](covariates["system"])   # system: a 2-vector covariate
    k = predictors["rate"]({"T": covariates["T"], "z0": z[0], "z1": z[1]})[0]
    ...
```

Masks, the tournament and saving all walk the pytree, so the embedding
needs no special handling.

## Registries

Solvers, adjoints, transforms, warps and evosax strategies are looked up by
name, which is what lets configs round-trip through JSON:

| Register | Module |
| --- | --- |
| `register_solver`, `register_adjoint` | `jaxhybridmodels.solver` |
| `register_bound_transform`, `register_warp` | `jaxhybridmodels.transforms` |
| `register_algorithm` | `jaxhybridmodels.training.evosax` |

Register before loading a saved run that uses a custom entry.

## Progress UIs

`ui=` takes any object implementing `TrainingUI` (Optax) or `EvosaxUI`
(Evosax) from `jaxhybridmodels.ui`, for example to send metrics to a
logger. It overrides `verbose`.
