# Getting Started

`hybridmodels` is a JAX library for fitting **hybrid ODE models**: a user-supplied vector field whose unknown rate terms are produced by trainable predictors (MLP, KAN, a small bounded-parameter module, or a custom subclass). The framework owns the JAX plumbing — vectorisation across experiments, JIT compilation per bucket shape, and gradient propagation through the integrator. The user owns the physics: the vector field, the projection from full state to observed channels, and the choice of predictors.

This page covers installation, the public API surface a typical model uses, and a minimal end-to-end example. For a full walkthrough on a real dataset see the [Crystallisation example](/examples/crystallisation); for a self-contained synthetic problem with a known optimum see the [Harmonic Oscillator](/examples/pendulum).

## Installation

The package targets Python ≥ 3.11 and is managed with [`uv`](https://docs.astral.sh/uv/). It is not yet on PyPI; install directly from the GitHub repository:

```bash
uv add git+https://github.com/DanielePessina/jax-hybridmodels
```

Or, if you have cloned the repository for development:

```bash
uv sync
```

Either path installs `hybridmodels` together with its core dependencies (`jax`, `equinox`, `diffrax`, `optax`, `evosax`, `jaxkan`) and a small set of plotting and CLI utilities.

## The pieces of a hybrid model

A working model brings together five components. The first three are user-written; the last two are framework-provided types the user instantiates.

1. **Experiments** ([`Experiment`](/api/data#experiment), built via [`make_experiment`](/api/data#make_experiment)). One record per real run, holding constant-in-time covariates, an initial-state hook, and per-channel sparse observations ([`ChannelObs`](/api/data#channelobs)).
2. **A `simulate_fn`** with the [mandatory signature](/guide/concepts#simulate-fn) `(predictors, ts, covariates, y0, solver) -> [T, S]`. Inside, the user constructs the vector field and calls `diffrax.diffeqsolve` with `adjoint=diffrax.DirectAdjoint()`.
3. **A `state_to_output` projector**. A pure function `[T, S] -> [T, D]` that maps the full simulator state to the observed channels in a fixed order.
4. **A predictors PyTree**. Conventionally a tuple of [`BoundedPredictor`](/api/predictors#boundedpredictor) leaves wrapping `MLPPredictor`, `KANPredictor`, or a custom [`Predictor`](/api/predictors#predictor) subclass. Models with no covariate dependence (a small set of global kinetic constants, for example) can use a minimal `eqx.Module` directly.
5. **A [`SolverConfig`](/api/solver#solverconfig)**. A frozen container holding a `diffrax` solver instance and tolerances.

[`make_dataset`](/api/data#make_dataset) packages experiments together with the projector into a [`Dataset`](/api/data#dataset), automatically aligning per-channel timestamps onto per-experiment union grids and grouping experiments by grid length into JIT-friendly buckets.

[`train_with_optax`](/api/training#train_with_optax) and [`train_with_evosax`](/api/training#train_with_evosax) both accept `(predictors, dataset, config)` together with `simulate_fn`, `solver`, and a JAX random key, and both return `(loss_history, trained_predictors)`.

## A minimal example

A scalar harmonic oscillator with a single trainable parameter `omega`. The dataset is synthesised at runtime from the closed-form solution; the trainer is asked to recover `omega ≈ 1.0` from noisy position observations alone.

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


# 1. Custom predictor: a single trainable scalar.
class OmegaPredictor(Predictor):
    omega_lat: Array

    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)

    def __call__(self, x: Array) -> Float[Array, " 1"]:
        return self.omega_lat[None]


# 2. Wrap the predictor with sigmoid-scaled output bounds [0.5, 2.0].
key = jr.PRNGKey(0)
k_init, k_noise, k_train = jr.split(key, 3)

predictor = BoundedPredictor(
    input_keys=("dummy",),
    in_scaler=BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid"),
    inner=OmegaPredictor(jr.normal(k_init)),
    out_scaler=BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
)
predictors = (predictor,)


# 3. simulate_fn: the second-order linear ODE for one experiment.
def simulate_fn(predictors, ts, covariates, y0, solver):
    omega = predictors[0](covariates).reshape(())

    def vector_field(t, y, args):
        return jnp.stack([y[1], -omega * omega * y[0]])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=solver.dt0 if solver.dt0 is not None else 0.05,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


# 4. state_to_output: only the position channel is observed.
def state_to_output(state):
    return state[..., :1]


# 5. Synthesise three experiments with omega=1.0 and different initial states.
NOISE_STD = 0.02
ts = jnp.linspace(0.0, 5.0, 12)
experiments = []
for i, (x0, v0) in enumerate([(1.0, 0.0), (0.0, 1.0), (0.5, -0.5)]):
    clean = x0 * jnp.cos(ts) + v0 * jnp.sin(ts)
    noisy = clean + NOISE_STD * jr.normal(jr.fold_in(k_noise, i), ts.shape)
    experiments.append(
        make_experiment(
            covariates={"dummy": 0.0},
            channels={
                "position": ChannelObs(
                    ts=ts,
                    values=noisy,
                    variance=jnp.full(ts.shape, NOISE_STD**2),
                ),
            },
            y0_fn=lambda c, ch, _y0=jnp.array([x0, v0], dtype=jnp.float32): _y0,
            exp_id=f"osc_{i}",
        )
    )

dataset = make_dataset(
    experiments,
    state_to_output=state_to_output,
    output_channel_names=("position",),
)


# 6. Solver and training configuration.
solver = SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-6,
    atol=1e-8,
    max_steps=4096,
    dt0=None,
)

config = OptaxTrainingConfig(
    steps=(300,),
    lr=(5e-2,),
    optimizer=("adamw",),
    reset_optimiser_state=(False,),
    length_schedule=(1.0,),
    loss="mse",
    verbose=False,
)

history, trained = train_with_optax(
    predictors,
    dataset,
    config,
    simulate_fn=simulate_fn,
    solver=solver,
    key=k_train,
)

recovered = float(trained[0]({"dummy": jnp.asarray(0.0)}).reshape(()))
print(f"final loss: {history[-1]:.6f}")
print(f"recovered omega: {recovered:.4f} (target: 1.0000)")
```

Running the script with the seed above produces a final loss around `2e-4` and an `omega` estimate within roughly 1% of the target.

The same surface scales up. Replacing `OmegaPredictor` with an `MLPPredictor` or a `KANPredictor`, adding real covariates and channels, and writing a population-balance vector field gives the [Crystallisation walkthrough](/examples/crystallisation). For models whose trainable component is a handful of global parameters rather than a function approximator — for example, four kinetic constants feeding a Classical Nucleation Theory rate law — the [mechanistic crystallisation example](/examples/crystallisation-mechanistic) shows the same training entry points used with [`train_with_evosax`](/api/training#train_with_evosax) and CMA-ES.

## Next steps

- [Crystallisation walkthrough](/examples/crystallisation) — a full end-to-end example on a real dataset, with two `BoundedPredictor` branches predicting growth and nucleation rates inside a method-of-moments ODE. Read this first; the rest of the documentation is easier with it as context.
- [Concepts](/guide/concepts) — the package's vocabulary: `Predictor`, `BoundScaler`, `BoundedPredictor`, `Experiment`, `Dataset`, `BucketPayload`, `simulate_fn`, `state_to_output`, predictor inputs versus covariates.
- [Training](/guide/training) — multi-phase Optax schedules, the shared-tournament restart loop, when to reach for `train_with_evosax`, and the named-fold RNG discipline that keeps runs reproducible.
- [Recommendations](/guide/recommendations) — choosing bounds, solver tolerances, freezing patterns, and the autodiff-safe guards needed to keep gradients finite under JAX tracing.
- [API Reference](/api/) — every public symbol with signature, parameters, and source link.
