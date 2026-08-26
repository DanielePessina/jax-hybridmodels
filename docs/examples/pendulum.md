# Harmonic Oscillator

A synthetic counterpart to the [crystallisation example](/examples/crystallisation),
with one unknown: the angular frequency $\omega$ of a one-dimensional
harmonic oscillator. Because the correct answer is $\omega = 1$, this is
the example to run when you want to know whether your pipeline is wired
correctly rather than whether your model is any good.

The full script lives at `examples/pendulum/train_harmonic.py`. Run it with:

```bash
uv run python examples/pendulum/train_harmonic.py
```

The data is generated on every run from the closed form
$x(t) = x_0 \cos(\omega t) + (v_0 / \omega) \sin(\omega t)$, so there is
no file to manage.

## What we're modelling

Textbook second-order ODE:

$$
\frac{dx}{dt} = v, \qquad \frac{dv}{dt} = -\omega^2\, x
$$

Each **experiment** is one oscillator: a true $\omega = 1.0$ and its own
initial state $(x_0, v_0)$. Only position is measured, with light
Gaussian noise. Velocity is part of the state the solver tracks and no
instrument sees. The trainer has to recover $\omega$ from positions
alone.

## Step 1: a one-scalar custom predictor

A **predictor** is any trainable module taking an array and returning an
array. Here it is one scalar. Wrapping it in a `BoundedPredictor`
confines that scalar to `[0.5, 2.0]`. This is the pattern for writing
your own: subclass [`Predictor`](/api/predictors#predictor) directly, no
network needed.

```python
from hybridmodels.predictors.base import Predictor

class OmegaPredictor(Predictor):
    """One trainable scalar; ignores its input."""
    omega_lat: Array
    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)
    def __call__(self, x: Array) -> Float[Array, " 1"]:
        return self.omega_lat[None]
    def initialized_with_key(self, key):
        return OmegaPredictor(jr.normal(key))
```

`initialized_with_key` is what the
[tournament](/guide/training#the-shared-tournament) calls to draw a
fresh starting point on each attempt. You can omit it. The default
`reinitialize_with_key` then resamples every floating-point leaf from
`jr.normal`, which is worse for a module that owns its own
initialisation scheme.

Wrap it in a `BoundedPredictor`:

```python
from hybridmodels import BoundedPredictor, BoundScaler

predictor = BoundedPredictor(
    input_keys=("dummy",),                                   # at least one slot is required
    in_scaler=BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid"),
    inner=OmegaPredictor(jr.normal(key)),
    out_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),  # search omega in [0.5, 2.0]
)
predictors = (predictor,)
```

The `"dummy"` covariate exists only because `BoundedPredictor` requires
at least one input; a predictor with none has no training signal.
`OmegaPredictor` ignores the value. Every experiment shares the same
true $\omega$, so there is nothing to condition on.

## Step 2: generate the experiments

```python
INITIAL_STATES = ((1.0, 0.0), (0.0, 1.0), (0.5, -0.5),
                  (1.0, 1.0), (-0.7, 0.4), (0.3, 0.9))
T_MAX, N_TIMESTEPS, NOISE_STD = 5.0, 12, 0.02

ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
experiments = []
for i, (x0, v0) in enumerate(INITIAL_STATES):
    clean = x0 * jnp.cos(ts) + v0 * jnp.sin(ts)
    noisy = clean + NOISE_STD * jr.normal(jr.fold_in(noise_key, i), ts.shape)
    experiments.append(make_experiment(
        covariates={"dummy": 0.0},
        channels={"position": ChannelObs(ts=ts, values=noisy,
                                          variance=jnp.full(ts.shape, NOISE_STD**2))},
        y0_fn=(lambda c, ch, _y0=jnp.array([x0, v0], dtype=jnp.float32): _y0),
        exp_id=f"osc_{i}_x0={x0}_v0={v0}",
    ))
```

**`y0_fn`** builds one experiment's full initial state. It runs once,
here, and never during training. The `_y0` default argument is Python's
standard trick for binding a loop variable early. In a real workflow
`y0_fn` would derive the state from covariates or from the first
observation.

`T_MAX = 5.0` covers roughly 0.8 of one period, enough phase coverage to
fit $\omega$ without aliasing into the wrong basin.

## Step 3: state_to_output and simulate_fn

```python
def _state_to_output(state):
    """[T, 2] -> [T, 1]. Only position is observed."""
    return state[..., :1]

def _simulate_fn(predictor, ts, covariates, y0, solver):
    omega = predictor(covariates).reshape(())
    omega_sq = omega * omega
    def vector_field(t, y, args):
        return jnp.stack([y[1], -omega_sq * y[0]])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field), solver.solver,
        t0=ts[0], t1=ts[-1], dt0=solver.dt0 or 0.05, y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=solver.stepsize_controller(),
        max_steps=solver.max_steps,
        adjoint=solver.adjoint,
    )
    return jnp.asarray(sol.ys)
```

`predictor(covariates)` returns shape `[1]`, and `.reshape(())` makes it
a scalar so the multiplication broadcasts cleanly. `BoundedPredictor`
always returns an array for uniformity, so scalar problems reshape at
the call site.

Note also that `predictor` is called above `diffeqsolve`, not inside the
vector field. Its inputs are all covariates, so its value cannot change
during the trajectory, and hoisting it keeps it off the solver tape.

## Step 4: train and read out $\omega$

```python
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

dataset = make_dataset(experiments, state_to_output=_state_to_output,
                       output_channel_names=("position",))

solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=None)

config = OptaxTrainingConfig(
    steps=(300,), lr=(5e-2,), optimizer=("adamw",),
    reset_optimiser_state=(False,), length_schedule=(1.0,),
    loss="mse", verbose=True,
)

history, trained = train_with_optax(
    predictors, dataset, config,
    simulate_fn=_simulate_fn, solver=solver, key=jr.PRNGKey(1),
)

# Read out the trained omega.
recovered = float(trained[0]({"dummy": jnp.asarray(0.0)}).reshape(()))
print(f"recovered omega: {recovered:.4f}  (target: 1.0000, final loss: {history[-1]:.6f})")
```

Expected output (default seed):

```
recovered omega: 0.9986  (target: 1.0000, final loss: 0.000186)
```

## What this example exercises

The same set of pieces as the crystallisation example, on a problem with
a known answer.

- [`ChannelObs`](/api/data#channelobs), [`Experiment`](/api/data#experiment), [`make_experiment`](/api/data#make_experiment), [`make_dataset`](/api/data#make_dataset)
- A custom [`Predictor`](/api/predictors#predictor) subclass wrapped in a [`BoundedPredictor`](/api/predictors#boundedpredictor)
- A user-written `simulate_fn` matching the [mandatory signature](/guide/concepts#simulate-fn)
- [`SolverConfig`](/api/solver#solverconfig) with `diffrax.Tsit5`
- [`OptaxTrainingConfig`](/api/training#optaxtrainingconfig) and [`train_with_optax`](/api/training#train_with_optax)
- The [tournament re-init hook](/guide/training#the-shared-tournament),
  through `OmegaPredictor.initialized_with_key`

To check a refactor of your own physics, run this and confirm the
recovered $\omega$ lands within about 1% of `OMEGA_TRUE`. If it does not,
the integrator and loss path has a wiring bug, usually in
`state_to_output` or in `y0_fn`.

## What's next

- [Crystallisation walkthrough](/examples/crystallisation). The real
  problem.
- [Concepts](/guide/concepts). Pytree conventions, predictor inputs
  versus covariates, bound scaling.
- [API reference](/api/). Every public symbol, by module.
