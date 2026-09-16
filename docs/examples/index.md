---
aside: false
outline: false
---

# Examples

Each example is a plain script that can be run from a repository checkout.
The pages explain the modelling choice, show the important code, and link to
the script that produces the result.

Install the optional example dependencies once:

```bash
uv sync --extra examples
```

## Choose an example

| Example | What it shows | Run it |
| --- | --- | --- |
| [Harmonic oscillator](/examples/pendulum) | A known one-parameter optimum for checking that the data, ODE, loss, and training path are wired correctly. | `uv run python examples/pendulum/train_harmonic.py --no-plot` |
| [Crystallisation](/examples/crystallisation) | The canonical end-to-end model with irregular channels, two bounded neural rates, and a population-balance ODE. | `uv run python examples/crystallisation/train_kinetic.py` |
| [Hybrid ODE](/examples/hybrid-ode) | Two predictors in different positions: one outside the solve and one inside the vector field. | `uv run python examples/hybrid_ode/train_hybrid_ode.py` |
| [Batch reactor](/examples/batch-reactor) | Evosax followed by Optax: a mechanistic Arrhenius trunk plus a learned residual. | `uv run python examples/batch_reactor/train_hybrid.py` |
| [Batch reactor RL](/examples/batch-reactor-rl) | A bounded controller trained with PPO around a frozen hybrid model. | `uv run python examples/batch_reactor/train_rl_deactivation.py` |
| [SBML hybrid kinetics](/examples/sbml-hybrid) | An external SBML mechanism with one unknown rate law supplied by a predictor. | `uv run python examples/sbml_hybrid/train_sbml_hybrid.py` |
| [Neural polynomial kinetics](/examples/supersaturation-poly) | A structured `NeuralNPolynomial` rate law instead of a black-box MLP. | `uv run python examples/supersaturation_poly/train_supersaturation_poly.py` |
| [Custom predictor](/examples/custom-predictor) | A random Fourier feature predictor with fixed and trainable leaves. | `uv run python examples/custom_predictor/train_custom_predictor.py` |
| [Custom training loop](/examples/custom-loop) | A hand-written loop assembled from the public gradient kernels. | `uv run python examples/custom_loop/train_custom_loop.py` |
| [Mechanistic crystallisation](/examples/crystallisation-mechanistic) | Four fitted kinetic constants and CMA-ES, with no neural rate predictors. | `uv run python examples/crystallisation/train_crystallisation_mechanistic.py` |

## Suggested order

Start with the [harmonic oscillator](/examples/pendulum), then read the
[crystallisation walkthrough](/examples/crystallisation). Use [Hybrid ODE](/examples/hybrid-ode)
when predictor placement and irregular data are the questions. Move to the
[custom predictor](/examples/custom-predictor) or [custom loop](/examples/custom-loop)
pages when the stock components no longer express your model.
