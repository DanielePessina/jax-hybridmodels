# Getting started

You write the ODE solve for one experiment. `jaxhybridmodels` batches it over
your data, differentiates through it, and fits the parameters or networks
inside it.

## Install

```bash
uv add jax-hybridmodels==0.2.0b1
```

The 0.2.0b1 release predates `HybridModel`, `Parameter` and `Constant`, which
this page uses. Until the next release, run `uv sync` in a checkout.

## Fit a physical parameter

This example learns an oscillator's frequency from its position, `cos(t)`, so
the true frequency is 1.0. Velocity is an unobserved state. `Parameter` holds
a trainable physical quantity and `Constant` a fixed one; both are read
through `.value`.

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

`1.3` is the starting guess in physical units, and `bounds=(0.5, 2.0)` keeps
every iterate inside that range. `fit` returns a new model in
`result.model`; `model` still holds 1.3. The fitted frequency prints as
1.0, and `stop_reason` is `"budget_exhausted"` because all 200 steps ran.

`t0=0.0` places the initial state before the first measurement at 0.2. The
dataset adds an unobserved point at t0, so `predictions["oscillator"].times`
has 21 entries for 20 measurements. Without `t0`, the initial state belongs to
the first observation time.

Run the complete example from a checkout:

```bash
uv run python examples/pendulum/train_model_api.py
```

## Next steps

- [Model interface](/guide/model-interface): `fit`, `predict`, `simulate` and their results.
- [Predictors and bounds](/guide/predictors): fixed values, trainable values and networks.
- [Training](/guide/training): multi-phase schedules, validation and loss histories.
- [Data and buckets](/guide/data): channels sampled at different times.

`HybridModel` is a thin layer over functions that take the predictors,
simulator, observation map and solver as separate arguments:
`train_with_optax`, `train_with_evosax` and `predict_dataset`. Call them
directly when you need something the methods do not cover, such as ensembles
or a custom training loop.
