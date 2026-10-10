---
layout: home

hero:
  name: jaxhybridmodels
  text: Train functions inside ODE models
  tagline: A JAX library for combining user-written differential equations with trainable predictors, bounded physical quantities, and irregular time-series data.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Browse examples
      link: /examples/
    - theme: alt
      text: API reference
      link: /api/

features:
  - title: User-owned dynamics
    details: Write one simulate_fn for a single experiment. The library applies batching, JIT compilation, and differentiation around it.
  - title: Irregular observations
    details: Give each channel its own timestamps. The data layer builds masks and groups experiments by timestamp-axis length.
  - title: Physical bounds
    details: Declare a range for a parameter or a network's inputs and output. Training works on an unbounded latent and can never leave the range.
  - title: Two training loops
    details: Use Optax for gradient-based fitting or Evosax for population-based search over small parameter sets.
  - title: PyTree composition
    details: Predictors are any PyTree of Equinox modules, usually a dict keyed by role. HybridModel binds them to the simulator; the functional layer takes them directly.
  - title: Explicit extension points
    details: Replace the predictor, loss, regulariser, solver, UI, or training loop with a callable or PyTree of your own.
---

## Start here

From a repository checkout:

```bash
uv sync --extra examples
uv run python examples/pendulum/train_model_api.py
```

It fits the frequency of a harmonic oscillator and predicts with the fitted
model. [Getting started](/guide/getting-started) walks through the same
code, and [Examples](/examples/) lists the larger hybrid models.

## In a nutshell

You write the ODE solve for one experiment. The library batches it over
irregular data, differentiates through it, and trains the predictors.

| Part | Responsibility |
| --- | --- |
| `predictors` | A PyTree of trainable values and networks. |
| `simulate_fn` | Integrates one experiment and returns the full state. |
| `state_to_output` | Selects the observed channels from the full state. |
| `SolverConfig` | Stores the Diffrax solver and its settings. |
| `Dataset` | Stores bucketed observations and masks. |

`HybridModel` binds the first four; the dataset is passed per call:

```python
import jaxhybridmodels as hm

dataset = hm.make_dataset(experiments, output_channel_names=("conc",))
model = hm.HybridModel(
    predictors=predictors,
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    output_channel_names=("conc",),
    solver=solver,
)
result = model.fit(dataset, hm.OptaxTrainingConfig(steps=500, lr=1e-2), key=key)
predictions = result.model.predict(dataset)
```

[Getting started](/guide/getting-started) defines every variable here.
[Concepts](/guide/concepts) explains the data and model interfaces.

## Find your path

| If you want to... | Start here | Main API |
| --- | --- | --- |
| Build irregular experiments and datasets | [Data and buckets](/guide/data) | [`make_experiment`](/api/data#make_experiment), [`make_dataset`](/api/data#make_dataset) |
| Write the ODE and observation map | [Model interface](/guide/model-interface) | [`HybridModel`](/api/model#hybridmodel) |
| Choose parameters, networks and bounds | [Predictors and bounds](/guide/predictors) | [`Parameter`](/api/predictors#parameter), [`BoundedPredictor`](/api/predictors#boundedpredictor) |
| Train with gradients or population search | [Training](/guide/training) | [`OptaxTrainingConfig`](/api/training#optaxtrainingconfig), [`EvosaxTrainingConfig`](/api/training#evosaxtrainingconfig) |
| Add time-varying inputs or schedules | [Profiles and schedules](/guide/profiles-and-schedules) | [`ramp_profile`](/api/profiles#ramp_profile), [`annealing_schedule`](/api/schedules#annealing_schedule) |
| Save, evaluate, or ensemble models | [Saving and loading](/guide/serialization), [Ensembles](/guide/ensembles) | [`save_run`](/api/serialise#save_run), [`compute_metrics`](/api/metrics#compute_metrics) |

## Install

Python 3.11 or newer:

```bash
uv add jax-hybridmodels==0.2.0b1
```

The 0.2.0b1 release predates `HybridModel`, `Parameter` and `Constant`. Until
the next release, use a checkout and `uv sync --extra examples`.
