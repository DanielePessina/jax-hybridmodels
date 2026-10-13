# jaxhybridmodels

A JAX/Equinox library for fitting user-written ODE models whose parameters or
rate laws are learned from data. Learned quantities stay inside declared
physical bounds, experiments may be sampled at irregular times, and training
uses Optax or Evosax.

## Documentation

<https://danielepessina.github.io/jax-hybridmodels/>. Start with
[Getting started](https://danielepessina.github.io/jax-hybridmodels/guide/getting-started),
pick a modelling pattern from the [Examples](https://danielepessina.github.io/jax-hybridmodels/examples/),
and look up exact signatures in the [API reference](https://danielepessina.github.io/jax-hybridmodels/api/).

| Need | Start here |
| --- | --- |
| Understand the model contract | [Model interface](https://danielepessina.github.io/jax-hybridmodels/guide/model-interface) |
| Build irregular data | [Data and buckets](https://danielepessina.github.io/jax-hybridmodels/guide/data) |
| Choose predictors and bounds | [Predictors and bounds](https://danielepessina.github.io/jax-hybridmodels/guide/predictors) |
| Train or freeze parameters | [Training](https://danielepessina.github.io/jax-hybridmodels/guide/training) |
| Extend the library | [Custom predictors](https://danielepessina.github.io/jax-hybridmodels/guide/custom-predictors) |

## First run

The examples are plain scripts. From a checkout:

```bash
uv sync --extra examples
uv run python examples/pendulum/train_harmonic.py
```

It fits the frequency of a harmonic oscillator to noisy position data with
`HybridModel.fit`, then predicts the data and simulates a new initial
condition. The Getting started page walks through a one-oscillator version
of the same code.

## Installation

### From PyPI

```bash
uv add jax-hybridmodels==0.2.0b1
```

The distribution is `jax-hybridmodels`; the import is `jaxhybridmodels`. The
0.2.0b1 release predates `HybridModel`, `Parameter` and `Constant`; use a
checkout for those until the next release.

### Development

```bash
uv sync --extra examples
```

This installs the package in editable mode plus the examples' extra
dependencies. Plain `uv sync` installs the library alone.

## Contributing

Run the verification checks before pushing:

```bash
uv run ruff check .                            # lint
uv run ruff format --check .                   # formatting
uv run ty check src tests examples scripts     # typecheck (needs --extra examples)
uv run pytest -q                               # test suite, about two minutes
uv run pytest -q -m slow                       # slow example tests
npm --prefix docs run docs:check               # API sync + site build
```

The first three also run as pre-commit hooks. Install them once with
`uvx pre-commit install`.

See [`docs/README.md`](./docs/README.md) for local preview, API generation,
and Pages deployment details.

## License

BSD-3-Clause. See [LICENSE](./LICENSE).
