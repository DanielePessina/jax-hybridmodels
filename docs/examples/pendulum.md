# Harmonic oscillator

The quickstart. One unknown with a known answer, the angular frequency
$\omega$ of a harmonic oscillator, run through the whole `HybridModel`
workflow: build a dataset, fit, read the result, predict, simulate new
conditions. Read it first, and come back to it when you want to check that
your own data, ODE and training path are wired correctly.

## Run it

```bash
uv run python examples/pendulum/train_harmonic.py
uv run python examples/pendulum/train_harmonic.py --no-plot
```

About 15 s on CPU. The data is generated on every run. Writes
`figures/parity.png` and `figures/trajectories.png` next to the script
unless `--no-plot` is given.

## The model

$$
\frac{dx}{dt} = v, \qquad \frac{dv}{dt} = -\omega^2 x - c\, v
$$

$\omega$ is learned, with true value 1, starting from 1.4. The damping $c$
is known to be zero and stays fixed. Six oscillators start from different
$(x_0, v_0)$ at $t = 0$. Only position is measured, at twelve points from
$t = 0.2$ to $5$, with Gaussian noise of standard deviation 0.02. Five
time units is about 0.8 of a period, enough to pin $\omega$ without a
second minimum at a multiple of it.

## Code

`simulate_fn` integrates one experiment and returns the full state. The
predictors arrive as the dict given to the model, and both kinds of value
are read through `.value`. `state_to_output` keeps the measured column:

```python
def simulate_fn(
    predictors: dict[str, hm.Parameter | hm.Constant],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: hm.SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate one oscillator and return the full state ``[x, v]``."""
    omega = predictors["omega"].value
    damping = predictors["damping"].value

    def vector_field(t: ArrayLike, y: Float[Array, " 2"], args: object) -> Array:
        return jnp.stack([y[1], -(omega**2) * y[0] - damping * y[1]])

    return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)


def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Only position is observed."""
    return state[..., :1]
```

Each experiment gives its initial state at `t0`, before the first
measurement. Use `y0_fn=` instead of `y0=` when the state has to be
computed from covariates or from the first observation:

```python
hm.make_experiment(
    channels={"position": hm.ChannelObs(ts=TS, values=noisy, variance=NOISE_STD**2)},
    # The state is known at t0 = 0; the first measurement is at 0.2.
    t0=0.0,
    y0=[x0, v0],
    exp_id=f"osc_{i}",
)
```

The model binds the pieces together. A [`Parameter`](/api/predictors#parameter)
trains on an unbounded latent and maps it into its bounds, so $\omega$
stays inside $(0.5, 2)$ at every step without clipping. A
[`Constant`](/api/predictors#constant) is never trained:

```python
model = hm.HybridModel(
    predictors={
        # Trained, and kept inside (0.5, 2.0) at every step.
        "omega": hm.Parameter(OMEGA_INIT, bounds=(0.5, 2.0)),
        # Known and fixed: the default training mask never updates it.
        "damping": hm.Constant(0.0),
    },
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    output_channel_names=CHANNELS,
    solver=hm.SolverConfig(
        solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=4096, dt0=0.05
    ),
)
```

Training runs two phases, a fast one to get close and a slower one to
settle. `fit` returns a [`FitResult`](/api/model#fitresult);
`result.model` is a new model with the fitted values and `model` is
unchanged:

```python
config = hm.OptaxTrainingConfig(
    phases=(
        hm.TrainingPhase(steps=200, lr=0.05),
        hm.TrainingPhase(steps=100, lr=0.01),
    ),
    verbose=False,
)
result = model.fit(dataset, config, key=k_train)
```

`predict` runs the fitted model on each experiment's times and is indexed
by experiment, then channel. Its `times` start at `t0`, where nothing was
measured. `simulate` needs no dataset, only times, an initial state and
any covariates the model reads:

```python
predictions = result.model.predict(dataset)
first = predictions["osc_0"]
first.outputs["position"]

new = result.model.simulate(ts=jnp.linspace(0.5, 10.0, 50), t0=0.0, y0=[0.5, 0.0])
new.outputs["position"]
```

## Results

```
[fit]
omega 1.4 -> 0.9997 (true 1.0)
best step 186, stop reason budget_exhausted

  channel           n           MSE          RMSE           MAE      R^2
  position         72    3.5542e-04    1.8853e-02    1.5612e-02   0.9993
```

The MSE sits at the noise variance, $0.02^2 = 4 \times 10^{-4}$, so the
fit has reached the noise floor. `best_step` is the step whose parameters
were returned, and `budget_exhausted` means training ran all its steps
without stopping early.

![Predicted against observed position](assets/pendulum/parity.png)

![Fitted trajectories for the first three oscillators](assets/pendulum/trajectories.png)

## Things to try

- Make the damping trainable, `hm.Parameter(0.1, bounds=(0.0, 1.0))`, and
  start $\omega$ from 1.9. With two unknowns and a far start, the same 300
  steps reach $\omega \approx 1.03$ instead of 1.000, and `best_step` is
  the last step: the run needs more steps, not a different model.
- If $\omega$ comes out more than about 1% from 1.0 after you adapt this
  script, look first at `state_to_output` and the initial states.
