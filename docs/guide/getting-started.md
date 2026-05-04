# Getting Started

`hybridmodels` lets you compose a **trainable predictor pytree** with a **user-written ODE** and train the whole thing on irregular time-series experiments. The framework owns the JAX plumbing — `vmap` over experiments, `jit` per bucket shape, `grad` through the integrator. You own the physics.

This page shows the smallest end-to-end run. For a real-world walkthrough see [Crystallisation](/examples/crystallisation); for a self-contained synthetic problem with a known optimum see [Harmonic Oscillator](/examples/pendulum).

## Installation

The project is `uv`-managed. From the repository root:

```bash
uv sync
```

This installs `hybridmodels` in editable mode together with `jax`, `equinox`, `diffrax`, `optax`, `evosax`, `jaxkan`, and the small CLI/UI dependencies.

## The five things you write

A working hybrid model is five pieces of code, all hand-written, glued together by the framework:

1. **`Experiment` records** — one per real run, holding covariates, initial state, and per-channel sparse observations.
2. **A `simulate_fn`** with the [mandatory signature](/guide/concepts#simulate-fn) — your vector field, integrated with `diffrax`.
3. **A `state_to_output` projector** — pure `[T, S] → [T, D]` mapping the full state to observed channels.
4. **A predictors pytree** — typically a tuple of `BoundedPredictor` leaves wrapping `MLPPredictor` / `KANPredictor` / your own.
5. **A `SolverConfig`** — diffrax solver instance plus tolerances.

Then you call `train_with_optax` (or `train_with_evosax`) and get back `(loss_history, trained_predictors)`.

## A 60-line minimal example

A scalar harmonic oscillator with a single trainable parameter `omega`:

```python
import diffrax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    SolverConfig,
    make_dataset,
    make_experiment,
)
from hybridmodels.predictors.base import Predictor
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax


# 1. Predictor — single scalar omega.
class OmegaPredictor(Predictor):
    omega_lat: Array
    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)
    def __call__(self, x: Array) -> Float[Array, " 1"]:
        return self.omega_lat[None]


key = jr.PRNGKey(0)
predictor = BoundedPredictor(
    input_keys=("dummy",),
    in_scaler=BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid"),
    inner=OmegaPredictor(jr.normal(key)),
    out_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
)
predictors = (predictor,)


# 2. simulate_fn — second-order linear ODE.
def simulate_fn(predictors, ts, covariates, y0, solver):
    omega = predictors[0](covariates).reshape(())
    def vector_field(t, y, args):
        return jnp.stack([y[1], -omega * omega * y[0]])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field), solver.solver,
        t0=ts[0], t1=ts[-1], dt0=0.05, y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps, adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


# 3. state_to_output — only position is observed.
def state_to_output(state):
    return state[..., :1]


# 4. Synthesise three experiments with omega=1.0, different initial states.
ts = jnp.linspace(0.0, 5.0, 12)
experiments = []
for i, (x0, v0) in enumerate([(1.0, 0.0), (0.0, 1.0), (0.5, -0.5)]):
    clean = x0 * jnp.cos(ts) + v0 * jnp.sin(ts)
    experiments.append(make_experiment(
        covariates={"dummy": 0.0},
        channels={"position": ChannelObs(ts=ts, values=clean, variance=jnp.full(ts.shape, 1e-4))},
        y0_fn=lambda c, ch, _y0=jnp.array([x0, v0], dtype=jnp.float32): _y0,
        exp_id=f"osc_{i}",
    ))

dataset = make_dataset(experiments, state_to_output=state_to_output, output_channel_names=("position",))


# 5. SolverConfig + train.
solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=None)
config = OptaxTrainingConfig(
    steps=(300,), lr=(5e-2,), optimizer=("adamw",),
    reset_optimiser_state=(False,), length_schedule=(1.0,),
    loss="mse", verbose=False,
)
history, trained = train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, solver=solver, key=jr.PRNGKey(1),
)

print(f"final loss: {history[-1]:.6f}")
print(f"recovered omega: {float(trained[0]({'dummy': jnp.asarray(0.0)})):.4f}  (target: 1.0000)")
```

Expected output (≈300 Adam steps, default seed):

```
final loss: 0.000186
recovered omega: 0.9986  (target: 1.0000)
```

The same surface scales up: replace the `OmegaPredictor` with an `MLPPredictor`, add real covariates and channels, and you have the [crystallisation walkthrough](/examples/crystallisation).

## What to read next

- **[Crystallisation walkthrough](/examples/crystallisation)** — the canonical hybrid-modelling problem, end-to-end. Read this first; the rest of the docs are easier in its context.
- **[Concepts](/guide/concepts)** — the vocabulary of the package: `Predictor`, `BoundScaler`, `BoundedPredictor`, `Experiment`, `Dataset`, `BucketPayload`, `simulate_fn`, `state_to_output`, predictor inputs vs covariates.
- **[Training](/guide/training)** — multi-phase Optax schedules, the shared tournament, when to reach for `train_with_evosax`, and how RNG discipline keeps runs reproducible.
- **[Recommendations](/guide/recommendations)** — bounds choice, solver tolerances, freezing, common pitfalls.
- **[API Reference](/api/)** — every public symbol with signature, parameters, and source link.
