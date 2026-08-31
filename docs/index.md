---
layout: home

hero:
  name: hybridmodels
  text: Train functions inside ODE models
  tagline: A JAX library for combining user-written differential equations with trainable predictors, bounded physical quantities, and irregular time-series data.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Browse examples
      link: /examples/pendulum
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

## In a nutshell

`hybridmodels` keeps the ODE simulation function in user code. The library
adds the surrounding data, predictor, and training machinery.

| Part | Responsibility |
| --- | --- |
| `predictors` | An Equinox PyTree of trainable functions. |
| `simulate_fn` | Integrates one experiment and returns the full state. |
| `state_to_output` | Selects the observed channels from the full state. |
| `SolverConfig` | Stores the Diffrax solver and its settings. |
| `Dataset` | Stores bucketed observations and masks. |

The shortest useful example is:

```python
import hybridmodels as hm

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

Read [Getting started](/guide/getting-started) for a complete example.
Read [Concepts](/guide/concepts) for the data and model interfaces.

## Install

The package requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/DanielePessina/jax-hybridmodels
```

For a checkout of the repository:

```bash
uv sync
```

## Next steps

- [Getting started](/guide/getting-started) — fit a harmonic oscillator.
- [Concepts](/guide/concepts) — learn the core terminology.
- [Training](/guide/training) — choose Optax or Evosax.
- [Examples](/examples/pendulum) — run complete scripts.
- [API reference](/api/) — inspect the public API.
