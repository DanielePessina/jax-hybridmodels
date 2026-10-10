# Custom training loop

`train_with_optax` is assembled from public pieces in
[`jaxhybridmodels.training.kernels`](/api/kernels). Read this example when
you need a loop the stock trainer cannot express, such as a different
accumulation rule or per-bucket weights: compose the same pieces instead of
forking the trainer.

## Run it

```bash
uv run python examples/custom_loop/train_custom_loop.py
```

Under a minute on CPU. Prints the loss, penalty and $\omega$ every 25
steps, then the recovered values and metrics; writes
`examples/custom_loop/figures/parity.png`. `--steps 5 --no-plot` is a quick
smoke run.

## The model

The same problem as the [harmonic oscillator](/examples/pendulum): recover
the frequency $\omega$ of $\ddot x = -\omega^2 x$ from noisy positions.
Four oscillators sit on two time grids, so the dataset forms two buckets of
three experiments and one.

The custom part is the accumulation. Each bucket's loss and gradients are
weighted by its experiment count, so the three-experiment bucket counts
three times the one-experiment bucket. The stock trainer has no option for
that. The loop also passes a masked Huber loss as `loss_fn` and an L2 term
on a second [`Parameter`](/api/predictors#parameter), `scale`, as
`penalty_fn`. `OptaxTrainingConfig` accepts both of those too (`loss=` and
`penalty_fn=`); they are here to show where each hook enters a
hand-written loop.

## Code

Three kernels:

| Kernel | Returns | Called |
| --- | --- | --- |
| [`build_bucket_step`](/api/kernels#build_bucket_step) | jitted `(predictors, bp, horizon) -> (loss, grads)` | once per bucket per step |
| [`build_penalty_step`](/api/kernels#build_penalty_step) | `(predictors, weight, points) -> (penalty, weighted_grads)` | once per step |
| [`build_apply_update`](/api/kernels#build_apply_update) | `(predictors, grads, opt_state) -> (predictors, opt_state)` | once per step |

Building them:

```python
bucket_step = build_bucket_step(
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    solver=solver,
    loss_fn=huber_loss,
    trainable=mask,
)
penalty_step = build_penalty_step(penalty_fn=weight_decay, trainable=mask)
optimizer = optax.adamw(learning_rate=1e-2)
apply_update = build_apply_update(optimizer, mask)
opt_state = optimizer.init(eqx.filter(predictors, mask))
```

One step of the loop:

```python
total_loss = jnp.asarray(0.0)
total_weight = 0.0
grads_sum = jax.tree.map(jnp.zeros_like, eqx.filter(predictors, mask))
for bp in ds.bucket_payloads:
    loss, grads = bucket_step(predictors, bp, horizon)
    w = bucket_weight(bp)
    total_loss = total_loss + w * loss
    total_weight += w
    grads_sum = jax.tree.map(lambda a, g, w=w: a + w * g, grads_sum, grads)
data_loss = total_loss / total_weight
grads = jax.tree.map(lambda g, t=total_weight: g / t, grads_sum)

# The regulariser is charged once per step, outside the bucket loop.
# It ignores penalty points, so none are passed.
penalty, penalty_grads = penalty_step(predictors, penalty_weight, ())
grads = jax.tree.map(jnp.add, grads, penalty_grads)

predictors, opt_state = apply_update(predictors, grads, opt_state)
```

## Results

Seed 0, 150 steps:

```
recovered omega = 1.0020  (truth 1.0)
final scale     = 0.070060  (regularised toward 0)
```

![Predicted against observed position for the custom-loop model](assets/custom-loop/parity.png)

The parity RMSE is 0.050, the noise standard deviation, so the fit has
reached the noise floor.

## Things to try

- Return `1.0` from `bucket_weight` to recover the stock trainer's equal
  weighting.
- Swap `huber_loss` for `jaxhybridmodels.losses.masked_mse`.
- Schedule `horizon` from 0.3 to 1.0 over the run, which is how the stock
  trainer's `TrainingPhase.horizon` works.
