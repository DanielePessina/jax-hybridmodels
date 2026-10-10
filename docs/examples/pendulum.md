# Harmonic oscillator

One unknown with a known answer: the angular frequency $\omega$ of a
harmonic oscillator, true value 1. Run this first, and whenever you want
to check that data, ODE, loss and training are wired correctly before you
ask whether a model is any good.

Two scripts share the problem:

| Script | What it shows |
| --- | --- |
| `examples/pendulum/train_harmonic.py` | Six noisy oscillators, metrics, parity and trajectory figures. |
| `examples/pendulum/train_model_api.py` | The shortest `HybridModel` round trip: fit, predict by name, simulate new conditions. |

```bash
uv run python examples/pendulum/train_harmonic.py
uv run python examples/pendulum/train_model_api.py
```

Both generate their data on every run, so there is no file to manage.

## The model

$$
\frac{dx}{dt} = v, \qquad \frac{dv}{dt} = -\omega^2\, x
$$

Each experiment is one oscillator with its own initial state
$(x_0, v_0)$. Only position is measured. Velocity is part of the state the
solver tracks, and `state_to_output` drops it:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    omega = predictors["omega"].value

    def vector_field(t, y, args):
        return jnp.stack([y[1], -(omega**2) * y[0]])

    return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)


def state_to_output(state):
    return state[..., :1]
```

`omega` is a [`Parameter`](/api/predictors#parameter): a trainable value
in physical units. The optimiser works on an unbounded latent, and
`.value` maps it into the bounds, so $\omega$ stays inside $[0.5, 2]$ at
every step without clipping.

## Data

Positions come from the closed form
$x(t) = x_0 \cos(\omega t) + (v_0 / \omega) \sin(\omega t)$ on twelve
points over $[0, 5]$, plus Gaussian noise with standard deviation 0.02.
Five time units is about 0.8 of a period, enough to pin $\omega$.

```python
hm.make_experiment(
    channels={"position": hm.ChannelObs(ts=ts, values=noisy, variance=...)},
    y0=[x0, v0],
    exp_id=f"osc_{i}",
)
```

`y0=` gives the initial state directly. Use `y0_fn=` instead when the
state has to be computed from covariates or the first observation.

## Fit and read out $\omega$

```python
model = hm.HybridModel(
    predictors={"omega": hm.Parameter(1.4, bounds=(0.5, 2.0))},
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    output_channel_names=("position",),
    solver=hm.SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05),
)
config = hm.OptaxTrainingConfig(steps=300, lr=5e-2, loss="mse", verbose=False)
result = model.fit(dataset, config, key=k_train)
result.model.predictors["omega"].value
```

`fit` returns a [`FitResult`](/api/model#fitresult). `result.model` is a
new model holding the fitted predictors; the original is unchanged.
Output from `train_harmonic.py`:

```
recovered omega = 0.9998 (target 1.0, best step 214)

  channel           n           MSE          RMSE           MAE      R^2
  position         72    3.5578e-04    1.8862e-02    1.5647e-02   0.9993
```

The MSE matches the noise variance, $0.02^2 = 4 \times 10^{-4}$, so the
fit has reached the noise floor.

## Predict and simulate

`train_model_api.py` fits a single oscillator from $\omega = 1.3$ with a
two-phase schedule, then uses the fitted model:

```python
config = hm.OptaxTrainingConfig(
    phases=(
        hm.TrainingPhase(steps=100, lr=0.03),
        hm.TrainingPhase(steps=100, lr=0.005),
    ),
    verbose=False,
)
result = model.fit(dataset, config, key=jr.key(0))

prediction = result.model.predict(dataset)["oscillator"]
prediction.outputs["position"]          # indexed by experiment, then channel

trajectory = result.model.simulate(ts=jnp.linspace(1.0, 8.0, 100), t0=0.0, y0=[0.5, 0.0])
trajectory.outputs["position"]
```

`predict` runs on each experiment's measured times. `simulate` needs no
dataset: give it times, an initial state and, if the model uses any,
covariates. This script also holds the damping at zero with
`hm.Constant(0.0)`, which the default training mask never updates.

If the recovered $\omega$ is more than about 1% from 1.0 after a change
to your own code, the problem is in the wiring, usually
`state_to_output` or the initial states.
