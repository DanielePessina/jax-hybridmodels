# Getting started

Use `jaxhybridmodels` to fit user-written ODE dynamics with trainable physical
parameters or networks. You write a single-experiment simulator; the library
handles differentiation, batching irregular observations, and training.

## Install

```bash
uv add jax-hybridmodels==0.2.0b1
```

`HybridModel`, `Parameter` and `Constant` are in the repository's development
version. From a checkout, run `uv sync` to use them.

## Fit a physical parameter

This example learns an oscillator's frequency from positions. Velocity is an
unobserved state. `Parameter` holds a trainable physical quantity and
`Constant` a fixed one; both are read through `.value`.

```python
import diffrax
import jax.numpy as jnp
import jax.random as jr
import jaxhybridmodels as hm


def simulate_fn(predictors, ts, covariates, y0, solver):
    omega = predictors["omega"].value
    damping = predictors["damping"].value

    def rhs(t, y, args):
        return jnp.stack((y[1], -omega**2 * y[0] - damping * y[1]))

    return solver.diffeqsolve(diffrax.ODETerm(rhs), ts, y0).ys


ts = jnp.linspace(0.2, 5.0, 20)
experiment = hm.make_experiment(
    exp_id="oscillator", t0=0.0, y0=[1.0, 0.0],
    channels={"position": hm.ChannelObs(ts=ts, values=jnp.cos(ts))},
)
dataset = hm.make_dataset([experiment], output_channel_names=("position",))
model = hm.HybridModel(
    predictors={
        "omega": hm.Parameter(1.3, bounds=(0.5, 2.0)),
        "damping": hm.Constant(0.0),
    },
    simulate_fn=simulate_fn,
    state_to_output=lambda state: state[..., :1],
    output_channel_names=("position",),
    solver=hm.SolverConfig(
        solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8,
        dt0=0.05, max_steps=4096,
    ),
)
result = model.fit(dataset, hm.OptaxTrainingConfig(steps=200, lr=0.03), key=jr.key(0))
print(result.model.predictors["omega"].value)
print(result.stop_reason)

predictions = result.model.predict(dataset)
position = predictions["oscillator"].outputs["position"]
trajectory = result.model.simulate(
    ts=jnp.linspace(1.0, 8.0, 100), t0=0.0, y0=[0.5, 0.0],
)
```

`1.3` is the initial physical frequency, not a latent network weight.
Fitting returns a new model; the original model keeps its initial frequency.
The recovered frequency should be close to 1.0.

`t0=0.0` says where the initial state is defined even though the first
measurement is at 0.2. The dataset adds an unobserved integration point at t0.
When omitted, the initial state belongs to the first observation time.

Run the complete example from a checkout:

```bash
uv run python examples/pendulum/train_model_api.py
```

## Next steps

- [Model interface](/guide/model-interface): methods, simulation, and named predictions.
- [Predictors and bounds](/guide/predictors): fixed values, trainable values, and networks.
- [Training](/guide/training): phase configurations, validation, and loss histories.
- [Data and buckets](/guide/data): observations sampled at different times.

The functional layer underneath takes the predictors, simulator, observation
map and solver as separate arguments. Use `train_with_optax`,
`train_with_evosax` or `predict_dataset` directly when writing a custom
workflow.
