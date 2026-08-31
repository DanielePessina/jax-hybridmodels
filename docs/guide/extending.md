# Extending hybridmodels

The library is assembled from small public pieces. A predictor, loss,
optimizer, regulariser, or training loop is a callable or PyTree that you can
replace. Subclass `Predictor` only when you are defining a new predictor
family.

This page lists each extension point and its expected interface.

## Custom predictors

A predictor is an `eqx.Module` whose `__call__` maps an input to an
output. Subclass `Predictor` and implement `__call__`:

```python
class MyPredictor(Predictor):
    weight: Array

    def __call__(self, x):
        return self.weight * x
```

Trainable arrays are ordinary fields; hyperparameters are
`eqx.field(static=True)` so the module round-trips through
`eqx.tree_serialise_leaves`. Add `initialized_with_key(self, key)` to
control how the tournament re-initialises it. See
[Custom predictors](/guide/custom-predictors) and the
[worked example](/examples/custom-predictor).

## Custom `simulate_fn`

`simulate_fn` is the one you write yourself, with a fixed signature. It
integrates one experiment and returns the full state trajectory. The
library calls it inside vmap and jit, so it must be tracible (pure
JAX/diffrax). You own the physics; the framework owns batching,
compilation, and gradients. See [Concepts](/guide/concepts).

To keep the diffrax call thin, use `SolverConfig.diffeqsolve`:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    term = diffrax.ODETerm(my_vector_field(predictors, covariates))
    return jnp.asarray(solver.diffeqsolve(term, ts, y0).ys)
```

The helper forwards `solver.adjoint`, so your gradient method is honoured
wherever you run.

## Custom `state_to_output`

`state_to_output` maps the full state `[T, S]` to the observed channels
`[T, D]`. It is a plain callable, and it belongs to the model, not the
data: pass it to `predict_*` and `train_with_*`. (It used to live on the
`Dataset`.) Swap it to change what a run measures.

## Custom losses

A loss is a pure function `loss(pred_obs, bp) -> scalar`. Pass any
callable to a training config's `loss` field:

```python
config = OptaxTrainingConfig(..., loss=my_huber_loss)
```

Channel selection composes automatically: a loss that accepts
`channel_idx`/`channel_weights` gets them as keywords; a plain
`(pred_obs, bp)` loss is handed the selected channels pre-sliced. See
`hybridmodels.losses.resolve_loss_fn`.

## Custom optimisers

A training config's `optimizer` field accepts three kinds of value:

- a name: `"adamw"` or `"adabelief"`;
- a factory taking `learning_rate` and returning an
  `optax.GradientTransformation`, e.g.
  `lambda lr: optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))`;
- a ready-made transformation instance.

Names and factories are wrapped in `optax.inject_hyperparams`, so a phase
boundary can move the learning rate without a rebuild. A raw instance
cannot be re-hyperparametrised: a phase that changes `lr` on one must also
set `reset_optimiser_state`.

## Custom regularisers

A training config's `penalty_fn` replaces the default bound-saturation
penalty. It is called as `penalty_fn(predictors, points) -> scalar`,
once per step, outside the bucket loop, with the per-leaf point arrays the
default penalty would evaluate at. Use it for weight decay, a
monotonicity term, or anything else the bound penalty does not express.
Both `OptaxTrainingConfig` and `EvosaxTrainingConfig` take it.

## Trajectory-aware penalties (embedded hybrid models)

`penalty_fn` operates on the predictors alone — it cannot see where the
model actually ran. An **embedded** hybrid model — the predictor inside
the vector field, its output feeding the dynamics (a kinetic parameter, a
shape factor) — sometimes needs a penalty that fires along the
trajectories it actually simulates. The training configs' other penalty
hook does exactly that:

```python
config = OptaxTrainingConfig(
    ...,
    trajectory_penalty_fn=my_trajectory_penalty,   # (full_state, bp) -> scalar
    trajectory_penalty_weight=1e-1,
)
```

`trajectory_penalty_fn` receives the **full state** `[N, T, S]` —
including any extra ODE components — and is added to the data loss inside
the same forward pass. No second simulation, no `simulate_fn` or
`state_to_output` signature change.

For an embedded predictor, the penalty rides in the ODE state as extra
accumulators (their time-integrals are what you charge). The helpers in
`hybridmodels.penalties` are the user-side recipe:

```python
from hybridmodels.penalties import (
    attach_penalty_state, penalty_vector_field,
    strip_penalty_state, penalty_integral,
)

# y0_fn: widen the state with two zero accumulators.
def y0_fn(cov, ch):
    return attach_penalty_state(jnp.array([x0, v0]), n=2)

# simulate_fn: wrap the physics vector field with penalty rates.
def pen_rates(t, y, args):
    z = predictor.inner(predictor.in_scaler.to_latent(y[0]))   # the latent
    return jnp.stack([predictor.out_scaler.saturation(z),     # output saturation
                      predictor.in_scaler.input_violation(y[0])])  # input excursion
term = diffrax.ODETerm(penalty_vector_field(physics, pen_rates))

# state_to_output: drop the accumulators before the loss.
def state_to_output(state):
    return strip_penalty_state(state, n=2)

# trajectory_penalty_fn: charge the time-integrals.
def my_trajectory_penalty(full_state, bp):
    return jnp.sum(penalty_integral(full_state, n=2))
```

For a **parallel** hybrid model — the predictor's output *is* a predicted
channel — the hook is simpler: `trajectory_saturation_penalty(full_state,
out_scaler)` inverts the outputs back to latents and charges saturation
over time.

**Probe conditions.** "Conditions I will deploy at, but have no `y_true`
for" are just experiments with no observations: give each channel
`values=jnp.array([])` (its `ts` still defines the integration grid, and
the mask is all-False, so the data loss is zero and only the trajectory
penalty fires).

## Custom training loops

The kernels a trainer is built from are public, in
`hybridmodels.training.kernels`: `build_bucket_step` (the jitted
per-bucket `(loss, grads)` kernel), `build_score_bucket` (forward-only),
`build_penalty_step` (the per-step regulariser), and `build_apply_update`
(the single optimiser update). Write your own loop by composing them:

```python
step = build_bucket_step(
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, loss_fn=my_loss, trainable=mask,
)
for bp in dataset.bucket_payloads:      # accumulate over every bucket
    loss, grads = step(predictors, bp, fraction)
# one update per step
```

See the [custom training loop example](/examples/custom-loop) for the
full pattern, including per-bucket weighting and a custom regulariser.

**Annealing a hyperparameter.** A custom loop that wants a smooth curve
instead of phases — a cosine-decayed learning rate, a penalty weight that
ramps in — multiplies a schedule into the value each step. The library's
`annealing_schedule` returns an optax-style multiplier with the run
length baked in (one step is one epoch):

```python
from hybridmodels import annealing_schedule

lr_sched = annealing_schedule("warmup_cosine", total_epochs=n_steps, warmup_epochs=5)
w_sched = annealing_schedule("linear", total_epochs=n_steps, end_value=0.5)

for step in range(n_steps):
    lr = base_lr * lr_sched(step)          # warm up, then cosine down
    weight = w0 * w_sched(step)            # ramp the penalty weight in
    ...   # build the optimiser at `lr`, call penalty_step(predictors, weight)
```

The stock trainers do not take schedules: they express strategy changes
as phases, and a smooth schedule is what a custom loop is for.

## System-embedding pattern (user-space)

The source package had framework classes for conditioning a model on the
system it is running on (`EmbeddedMLP*`,
`SystemConditionedRatePredictor`). This library deliberately does not:
the pattern is a few lines of composition, and it is more flexible than
any class could be.

The idea is that one hybrid model serves several related systems — a
catalyst family, a batch of reactors, a set of plant conditions. Give
each system a **condition vector** that rides as ordinary covariates, and
add a small trainable embedding as an extra predictor leaf. The vector
field mixes the embedding's output into the predictor input dict, exactly
like a state-derived or time-varying input:

```python
embedding = eqx.nn.Linear(2, 4, key=k1)          # system_id -> 4-dim condition
rate = BoundedPredictor(input_keys=("T", "pH", "z0", "z1"), ...)

def vector_field(t, y, args):
    z = embedding(jnp.stack([cov["system_id"], cov["batch"]]))
    inputs = {"T": T_profile(t), "pH": cov["pH"], "z0": z[0], "z1": z[1]}
    k = rate(inputs)
    ...
```

`predictors = (embedding, rate)`. Everything the framework does — the
trainability mask, the tournament re-init, serialisation, the trajectory
penalty — walks the pytree and treats both leaves uniformly. The same
pattern covers a per-system *bias* (freeze the embedding, train only the
rate), or a shared trunk with per-system heads: it is just more leaves in
the tuple.

## Grouping the five pieces

`Getting Started` lists the five things that go into a working model:
the experiments (dataset), `simulate_fn`, `state_to_output`, the
predictors, and the solver. The library keeps them apart on purpose —
Nothing stops you from
bunching them up in your own container and passing that around:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class HybridModel:
    predictors: Any
    simulate_fn: Callable
    state_to_output: Callable
    solver: SolverConfig
    dataset: Dataset

def fit(model: HybridModel, config, key):
    return train_with_optax(
        model.predictors, model.dataset, config,
        simulate_fn=model.simulate_fn,
        state_to_output=model.state_to_output,
        solver=model.solver,
        key=key,
    )
```

This grouping remains in user code. Each field is still passed directly to the
library functions, so the public interfaces do not change. Serialization
continues to operate on the predictor PyTree, and prediction and ensembles
continue to use the same callables.

## Custom solvers, adjoints, transforms, warps

These are name-keyed registries. Register a class, then reference it by
name in a config:

- `register_solver(name, cls)` / `register_adjoint(name, cls)` —
  diffrax solvers and gradient methods.
- `register_bound_transform(name, transform)` /
  `register_warp(name, warp)` — squash shapes and axis warps.
- `register_algorithm(name, cls)` — evosax strategies.

Registration happens at import time; the config round-trips the name
through JSON, so a custom class must be registered before loading a saved
config that references it.

## Custom UIs

The training loops take a `ui=` object implementing `TrainingUI` (optax)
or `EvosaxUI` (evosax). Implement the protocol to pipe metrics into your
own logger — a TensorBoard writer, a file, a queue. An explicit `ui=`
always wins over the config's `verbose` flag.

## Summary

The stock trainers are built from the same public pieces available to user
code. Replace one piece at a time, or assemble a complete custom loop from
the training kernels.
