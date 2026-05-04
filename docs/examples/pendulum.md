# Harmonic Oscillator

A self-contained synthetic counterpart to the [crystallisation example](/examples/crystallisation): the "hidden physics" is a single trainable scalar — the angular frequency $\omega$ of a 1-D harmonic oscillator. The known optimum ($\omega = 1$) makes this a good sanity check that everything is wired correctly.

The full script lives at `examples/pendulum/train_harmonic.py`. Run it with:

```bash
uv run python examples/pendulum/train_harmonic.py
```

The dataset is synthesised on every run from the closed form $x(t) = x_0 \cos(\omega t) + (v_0 / \omega) \sin(\omega t)$, so there's no external file to manage.

## What we're modelling

Textbook second-order ODE:

$$
\frac{dx}{dt} = v, \qquad \frac{dv}{dt} = -\omega^2\, x
$$

Each experiment is one oscillator with a known ground-truth $\omega = 1.0$ and a different initial state $(x_0, v_0)$. **Only the position channel is observed** with light Gaussian noise; velocity is part of the latent state. The trainer is invited to recover $\omega \approx 1.0$ from positions alone.

## Step 1 — a one-leaf custom predictor

The "predictor" here is trivial: a single scalar wrapped in `BoundedPredictor` so we can sigmoid-bound it into `[0.5, 2.0]`. This shows the pattern for writing your own predictor — subclass [`Predictor`](/api/predictors#predictor) directly, no MLP needed.

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

`initialized_with_key` is the [shared-tournament protocol](/guide/training#the-shared-tournament): if you turn on the tournament, this is what gets called per attempt to draw a fresh starting point. You can omit it; the framework's default `reinitialize_with_key` walks every inexact-float leaf with `jr.normal`.

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

The `"dummy"` covariate is a single constant required by `BoundedPredictor` (cardinality of `in_scaler.bounds` must be ≥ 1), but the `OmegaPredictor` itself ignores its input — every experiment shares the same ground-truth $\omega$, so there is nothing to condition on.

## Step 2 — synthesise experiments

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

The `y0_fn` closure captures the ground-truth $(x_0, v_0)$ at synthesis time; the `_y0` default-argument trick is just Python's standard "early bind to the loop variable" pattern. In a real workflow, `y0_fn` would derive the initial state from raw covariates or from the first observation.

`T_MAX = 5.0` covers roughly $0.8$ of one period — enough phase coverage to fit $\omega$ without aliasing into the wrong basin of attraction.

## Step 3 — `state_to_output` and `simulate_fn`

```python
def _state_to_output(state):
    """[T, 2] -> [T, 1] — only position is observed."""
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
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)
```

Note the unpacking: `predictor(covariates)` returns shape `[1]`; `.reshape(())` makes it a scalar so the multiplication broadcasts cleanly. This is a common pattern — `BoundedPredictor` is array-shaped for uniformity, scalar problems just `.reshape(())` at the call site.

## Step 4 — train and read out $\omega$

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

The same surface as the crystallisation example, on a problem with a known optimum:

- [`ChannelObs`](/api/data#channelobs), [`Experiment`](/api/data#experiment), [`make_experiment`](/api/data#make_experiment), [`make_dataset`](/api/data#make_dataset)
- A custom [`Predictor`](/api/predictors#predictor) subclass plus [`BoundedPredictor`](/api/predictors#boundedpredictor) wrapping
- A user-written `simulate_fn` matching the [mandatory signature](/guide/concepts#simulate-fn)
- [`SolverConfig`](/api/solver#solverconfig) with `diffrax.Tsit5`
- [`OptaxTrainingConfig`](/api/training#optaxtrainingconfig) and [`train_with_optax`](/api/training#train_with_optax)
- The [shared-tournament re-init protocol](/guide/training#the-shared-tournament) via `OmegaPredictor.initialized_with_key`

If you want to verify a refactor of your physics, drop the `OmegaPredictor` into a fresh repo, train it, and check that the recovered $\omega$ is within ~1% of `OMEGA_TRUE`. If it isn't, the integrator/loss path has a wiring bug — usually in the `state_to_output` projector or the `y0_fn` hook.

## What's next

- [Crystallisation walkthrough](/examples/crystallisation) — the canonical real-world problem.
- [Concepts](/guide/concepts) — pytree conventions, predictor inputs vs covariates, and the shared-tournament pattern.
- [API Reference](/api/) — full surface, by module.
