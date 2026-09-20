# jaxhybridmodels

A JAX/Equinox library for combining user-written ODE dynamics with trainable
predictors. It supports bounded physical quantities, regular and irregular
time-series experiments, and Optax or Evosax training. Crystallisation
kinetics is the canonical example, not the scope.

## Documentation

The documentation site is built with VitePress and deployed to GitHub Pages at
**<https://danielepessina.github.io/jax-hybridmodels/>**.

Start with the [Getting started](https://danielepessina.github.io/jax-hybridmodels/guide/getting-started)
page, use the [Examples](https://danielepessina.github.io/jax-hybridmodels/examples/)
chooser to find a modelling pattern, and use the generated [API reference](https://danielepessina.github.io/jax-hybridmodels/api/)
for exact signatures.

| Need | Start here |
| --- | --- |
| Understand the model contract | [Model interface](https://danielepessina.github.io/jax-hybridmodels/guide/model-interface) |
| Build irregular data | [Data and buckets](https://danielepessina.github.io/jax-hybridmodels/guide/data) |
| Choose predictors and bounds | [Predictors and bounds](https://danielepessina.github.io/jax-hybridmodels/guide/predictors) |
| Train or freeze parameters | [Training](https://danielepessina.github.io/jax-hybridmodels/guide/training) |
| Extend the library | [Custom predictors](https://danielepessina.github.io/jax-hybridmodels/guide/custom-predictors) |

The site source lives under [`docs/`](./docs).

## First run

The examples are plain scripts. From a checkout:

```bash
uv sync --extra examples
uv run python examples/pendulum/train_harmonic.py --no-plot
```

The harmonic oscillator has a known optimum, so it is a useful wiring check.
See the [examples overview](https://danielepessina.github.io/jax-hybridmodels/examples/)
for the other workflows.

## Installation

### From PyPI

```bash
uv add jax-hybridmodels==0.2.0b1
```

The PyPI distribution is named `jax-hybridmodels`; import it in Python as
`jaxhybridmodels`.

### Development

```bash
uv sync --extra examples
```

This installs `jaxhybridmodels` in editable mode together with `jax`,
`equinox`, `diffrax`, `optax`, `evosax`, `jaxkan`, and the small CLI/UI
dependencies, plus the optional dependencies used by the examples.

For the library only, use `uv sync`.

## Contributing

Run the verification checks before pushing:

```bash
uv run ruff check .        # lint
uv run ty check src        # typecheck
uv run pytest -q           # test suite
npm --prefix docs run docs:check   # API sync + site build
```

See [`docs/README.md`](./docs/README.md) for local preview, API generation,
and Pages deployment details.

## License

BSD-3-Clause. See [LICENSE](./LICENSE).
