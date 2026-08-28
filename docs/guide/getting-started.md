# Getting started

## What problem this solves

You have measurements of something changing over time. You believe an
ordinary differential equation (ODE) governs it. You can write down
part of that ODE from first principles, and part of it you cannot: a
rate that depends on temperature in some unknown way, a correction term
you know is missing, a growth law nobody has derived.

`hybridmodels` fits the part you cannot write down while keeping the
part you can. This is a **hybrid** modelling package.
[Diffrax](https://docs.kidger.site/diffrax/) already lets you put a
neural network in a vector field and differentiate through the solver;
this library is built on it and does not re-teach it. What it adds is
the machinery around that: physical ranges that hold by construction,
ragged per-channel measurements handled without padding, and a training
loop shaped for both.

Concretely, you supply four things:

1. Your measurements, grouped into **experiments** (one experiment is
   one run of the real thing, with its own conditions and its own
   observations).
2. One Python function that integrates your ODE for a single experiment.
3. A statement of which quantities the network predicts, and the
   physical range each one lives in.
4. A training budget.

Training runs the integrator forward, compares the result to your
measurements, and sends gradients back through the integrator into the
network.

## What you need to know

You need Python. You do not need to know the stack below, and these
one-liners are enough to read the rest of this page.

| Library | What it does here |
|---|---|
| [JAX](https://docs.jax.dev/) | NumPy-style arrays that can be differentiated and compiled. `jax.numpy` is imported as `jnp` and behaves like NumPy. |
| [Equinox](https://docs.kidger.site/equinox/) | Neural networks written as plain Python classes JAX can differentiate. Written `eqx`. |
| [Diffrax](https://docs.kidger.site/diffrax/) | ODE solvers you can differentiate through. You call `diffrax.diffeqsolve` yourself. |
| [Optax](https://optax.readthedocs.io/) | Gradient optimisers (Adam and friends). |
| [evosax](https://github.com/RobertTLange/evosax) | Population search (CMA-ES), for when the fit has many local minima. |

One JAX word recurs throughout these docs. A **pytree** is any nesting
of tuples, lists, dicts, and Equinox modules with arrays at the bottom.
JAX walks that nesting and applies an operation to every array it finds,
so you can hand it a tuple of two networks and it differentiates both.
When the docs say "a pytree of predictors", read "your networks, in
whatever container you like".

## Installation

The package needs Python 3.11 or newer and is managed with
[`uv`](https://docs.astral.sh/uv/). It is not on PyPI yet, so install
from GitHub:

```bash
uv add git+https://github.com/DanielePessina/jax-hybridmodels
```

If you cloned the repository to work on it:

```bash
uv sync
```

Either path installs `hybridmodels` with `jax`, `equinox`, `diffrax`,
`optax`, `evosax`, and `jaxkan`.

## The pieces of a hybrid model

Five things go into a working model. You write the first three. The last
two are library types you fill in.

**1. Experiments.** One [`Experiment`](/api/data#experiment) per run of
the real thing, built with
[`make_experiment`](/api/data#make_experiment). It holds:

- **covariates**, the conditions that stay fixed for the whole run
  (temperature, pH, initial loading), given as a dict of named scalars;
- **channels**, one per measured quantity. Each channel is a
  [`ChannelObs`](/api/data#channelobs) carrying its own timestamps, its
  own values, and a variance. Two channels in one experiment can be
  measured at completely different times;
- a **`y0_fn`** hook, which builds the ODE's full initial state from the
  covariates and the channels. The state usually has components nobody
  measured, and this is where you supply their starting values.

**2. A `simulate_fn`.** Your function, with a
[signature the library fixes](/guide/concepts#simulate-fn):
`(predictors, ts, covariates, y0, solver) -> [T, S]`. It integrates one
experiment and returns the full state at every requested time. Inside,
you write the vector field (the right-hand side of your ODE) and call
`diffrax.diffeqsolve`.

**3. A `state_to_output`.** A function mapping the full state
trajectory `[T, S]` to only the quantities you actually measured
`[T, D]`, in a fixed order. The integrator tracks state your
instruments never see, and this drops or combines it.

**4. Predictors.** A **predictor** is a trainable network: array in,
array out, and nothing else. Wrap each one in a
[`BoundedPredictor`](/api/predictors#boundedpredictor), which names its
inputs and declares a low and a high value for every input and output.
The inner network works in an unbounded space; the wrapper squashes its
output into the declared range. Ship both
[`MLPPredictor`](/api/predictors#mlppredictor) (a standard multi-layer
network) and [`KANPredictor`](/api/predictors#kanpredictor) (a
Kolmogorov-Arnold network), or write your own by subclassing
[`Predictor`](/api/predictors#predictor). By convention you put them in
a tuple, even when there is only one.

**5. A [`SolverConfig`](/api/solver#solverconfig).** The Diffrax solver
instance plus its tolerances, step budget, and adjoint. The **adjoint**
is the strategy Diffrax uses to get gradients back out of the
integration; see [Concepts](/guide/concepts#solverconfig).

[`make_dataset`](/api/data#make_dataset) turns your experiments into a
[`Dataset`](/api/data#dataset). It merges each experiment's per-channel
timestamps into one axis, records which cells are real observations, and
groups experiments by axis length. `state_to_output` is passed to
training and prediction, not to the dataset.

[`train_with_optax`](/api/training#train_with_optax) and
[`train_with_evosax`](/api/training#train_with_evosax) both take
`(predictors, dataset, config)` plus `simulate_fn`, `state_to_output`,
`solver`, and a random `key`, and both return
`(loss_history, trained_predictors)`.

## A runnable example

A harmonic oscillator with one unknown: the angular frequency `omega`.
The data is generated at runtime from the closed-form solution, with
noise. The trainer must recover `omega = 1.0` from noisy positions
alone, never seeing velocity.

```python
import diffrax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

import hybridmodels as hm

# 1. A predictor holding one trainable scalar. It ignores its input,
#    because every experiment shares the same omega.
class OmegaPredictor(hm.Predictor):
    omega_lat: Array

    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)

    def __call__(self, x: Array) -> Float[Array, " 1"]:
        return self.omega_lat[None]


# 2. Wrap it so its output is confined to [0.5, 2.0]. BoundScaler is the
#    map between physical units and the unbounded space the inner
#    predictor works in; "sigmoid" is how it saturates near the edges.
key = jr.PRNGKey(0)
k_init, k_noise, k_train = jr.split(key, 3)

predictor = hm.BoundedPredictor(
    input_keys=("dummy",),
    in_scaler=hm.BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid"),
    inner=OmegaPredictor(jr.normal(k_init)),
    out_scaler=hm.BoundScaler(bounds=((0.5, 2.0),), transform="sigmoid"),
)
predictors = (predictor,)


# 3. simulate_fn: integrate one experiment. The signature is fixed by
#    the library; everything inside it is yours.
def simulate_fn(predictors, ts, covariates, y0, solver):
    omega = predictors[0](covariates).reshape(())

    def vector_field(t, y, args):
        return jnp.stack([y[1], -omega * omega * y[0]])

    return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)


# 4. state_to_output: the state is (position, velocity); only position
#    is measured. It belongs to the model, and is passed to training.
def state_to_output(state):
    return state[..., :1]


# 5. Three experiments, omega=1.0, different initial states.
NOISE_STD = 0.02
ts = jnp.linspace(0.0, 5.0, 12)
experiments = []
for i, (x0, v0) in enumerate([(1.0, 0.0), (0.0, 1.0), (0.5, -0.5)]):
    clean = x0 * jnp.cos(ts) + v0 * jnp.sin(ts)
    noisy = clean + NOISE_STD * jr.normal(jr.fold_in(k_noise, i), ts.shape)
    experiments.append(
        hm.make_experiment(
            covariates={"dummy": 0.0},
            channels={
                "position": hm.ChannelObs(
                    ts=ts,
                    values=noisy,
                    variance=jnp.full(ts.shape, NOISE_STD**2),
                ),
            },
            y0_fn=lambda c, ch, _y0=jnp.array([x0, v0], dtype=jnp.float32): _y0,
            exp_id=f"osc_{i}",
        )
    )

dataset = hm.make_dataset(experiments, output_channel_names=("position",))


# 6. Solver settings and training budget.
solver = hm.SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-6,
    atol=1e-8,
    max_steps=4096,
    dt0=None,
)

config = hm.OptaxTrainingConfig(
    steps=(300,),
    lr=(5e-2,),
    optimizer=("adamw",),
    reset_optimiser_state=(False,),
    length_schedule=(1.0,),
    loss="mse",
    verbose=False,
)

history, trained = hm.train_with_optax(
    predictors,
    dataset,
    config,
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    solver=solver,
    key=k_train,
)

recovered = hm.evaluate_predictor(trained[0], {"dummy": 0.0})
print(f"final loss: {history[-1]:.6f}")
print(f"recovered omega: {recovered:.4f} (target: 1.0000)")
```

With the seed above this prints a final loss near `0.00025` and
`recovered omega: 1.0013`.

Two details in that script recur everywhere.

The **`"dummy"` covariate** exists because a `BoundedPredictor` must
declare at least one input. A predictor with no inputs has no training
signal, so the constructor refuses one. `OmegaPredictor` ignores the
value it receives.

**`key=` is keyword-only** on both trainers, and has no default. The
library never falls back to `jr.PRNGKey(0)` behind your back, so every
run states its own seed.

## Scaling this up

Replace `OmegaPredictor` with an `MLPPredictor`, add real covariates and
channels, and write a real vector field, and you have the
the [crystallisation walkthrough](/examples/crystallisation).

Some models have no network at all. Their trainable part is a handful of
kinetic constants feeding a classical rate law. That works the same way,
and is usually better fitted by population search than by gradients. See
the [mechanistic crystallisation example](/examples/crystallisation-mechanistic).

## Next steps

- [Concepts](/guide/concepts). The vocabulary, and why each design
  choice is the way it is. Read this second.
- [Training](/guide/training). Multi-phase schedules, restart tournaments,
  population search, and freezing.
- [Custom predictors](/guide/custom-predictors). Writing a predictor
  family of your own, once `MLPPredictor` is the wrong prior.
- [Recommendations](/guide/recommendations). Choosing bounds, solver
  tolerances, and the traps that cost a debugging session.
- [API reference](/api/). Every public symbol.
