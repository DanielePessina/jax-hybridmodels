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
  - title: Bounded predictors
    details: Declare physical input and output ranges. The predictor works in latent space and maps its output back into those ranges.
  - title: Two training loops
    details: Use Optax for gradient-based fitting or Evosax for population-based search over small parameter sets.
  - title: PyTree composition
    details: Pass one predictor or a nested tuple, dictionary, or NamedTuple. There is no required model wrapper class.
  - title: Explicit extension points
    details: Replace the predictor, loss, regulariser, solver, UI, or training loop with a callable or PyTree of your own.
---

## Start here

From a repository checkout, run the known-answer sanity check first:

```bash
uv sync --extra examples
uv run python examples/pendulum/train_harmonic.py --no-plot
```

It recovers the frequency of a synthetic harmonic oscillator from noisy
position measurements. The [Getting started](/guide/getting-started) page
then builds the same pipeline step by step, and the [Examples](/examples/)
page shows where to go for a real hybrid ODE.

## In a nutshell

`jaxhybridmodels` keeps the ODE simulation function in user code. The Python
package is imported as `jaxhybridmodels`. The library
adds the surrounding data, predictor, and training machinery.

| Part | Responsibility |
| --- | --- |
| `predictors` | An Equinox PyTree of trainable functions. |
| `simulate_fn` | Integrates one experiment and returns the full state. |
| `state_to_output` | Selects the observed channels from the full state. |
| `SolverConfig` | Stores the Diffrax solver and its settings. |
| `Dataset` | Stores bucketed observations and masks. |

The core training call is:

```python
import jaxhybridmodels as hm

dataset = hm.make_dataset(experiments, output_channel_names=("value",))
history, trained = hm.train_with_optax(
    predictors,
    dataset,
    config,
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    solver=solver,
    key=key,
)
```

The variables in this short call are defined in the complete
[Getting started](/guide/getting-started) example. Read [Concepts](/guide/concepts)
for the data and model interfaces.

## Find your path

| If you want to... | Start here | Main API |
| --- | --- | --- |
| Build irregular experiments and datasets | [Data and buckets](/guide/data) | [`make_experiment`](/api/data#make_experiment), [`make_dataset`](/api/data#make_dataset) |
| Write the ODE and observation map | [Model interface](/guide/model-interface) | `simulate_fn`, `state_to_output` |
| Choose a bounded predictor | [Predictors and bounds](/guide/predictors) | [`BoundedPredictor`](/api/predictors#boundedpredictor), [`BoundScaler`](/api/predictors#boundscaler) |
| Train with gradients or population search | [Training](/guide/training) | [`train_with_optax`](/api/training#train_with_optax), [`train_with_evosax`](/api/training#train_with_evosax) |
| Add time-varying inputs or schedules | [Profiles and schedules](/guide/profiles-and-schedules) | [`ramp_profile`](/api/profiles#ramp_profile), [`annealing_schedule`](/api/schedules#annealing_schedule) |
| Save, evaluate, or ensemble models | [Saving and loading](/guide/serialization), [Ensembles](/guide/ensembles) | [`save_run`](/api/serialise#save_run), [`predict_dataset`](/api/prediction#predict_dataset) |

## Install

The package requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/):

```bash
uv add jaxhybridmodels==0.2.0b1
```

For a checkout of the repository:

```bash
uv sync --extra examples
```
