---
aside: false
outline: false
---

# Examples

Each example is a script you run from a repository checkout. Install the
example dependencies once:

```bash
uv sync --extra examples
```

| Example | What it teaches | Run it |
| --- | --- | --- |
| [Harmonic oscillator](/examples/pendulum) | `HybridModel` fit, predict and simulate on a one-`Parameter` problem with a known answer. Start here. | `uv run python examples/pendulum/train_harmonic.py --no-plot` |
| [Crystallisation](/examples/crystallisation) | Irregular multi-channel data, two bounded neural rates, and a population-balance ODE. | `uv run python examples/crystallisation/train_kinetic.py` |
| [Hybrid ODE](/examples/hybrid-ode) | One network above the solver and one inside the vector field, log warps, and the saturation penalty. | `uv run python examples/hybrid_ode/train_hybrid_ode.py` |
| [Batch reactor](/examples/batch-reactor) | Evosax search, then Optax refinement, on an Arrhenius trunk with a learned residual. | `uv run python examples/batch_reactor/train_hybrid.py` |
| [Batch reactor RL](/examples/batch-reactor-rl) | A bounded controller trained with PPO around a frozen hybrid model. | `uv run python examples/batch_reactor/train_rl_deactivation.py` |
| [SBML hybrid kinetics](/examples/sbml-hybrid) | An external SBML mechanism with one rate law learned by a predictor. | `uv run python examples/sbml_hybrid/train_sbml_hybrid.py` |
| [Neural polynomial kinetics](/examples/supersaturation-poly) | A structured `NeuralNPolynomial` rate law in place of a black-box MLP. | `uv run python examples/supersaturation_poly/train_supersaturation_poly.py` |
| [Custom predictor](/examples/custom-predictor) | Writing a predictor family, freezing some of its arrays, and saving it. | `uv run python examples/custom_predictor/train_custom_predictor.py` |
| [Custom training loop](/examples/custom-loop) | A hand-written training loop built from the public gradient kernels. | `uv run python examples/custom_loop/train_custom_loop.py` |
| [Mechanistic crystallisation](/examples/crystallisation-mechanistic) | Four kinetic constants fitted with CMA-ES, no neural predictors. | `uv run python examples/crystallisation/train_crystallisation_mechanistic.py` |

Read the [harmonic oscillator](/examples/pendulum) first, then
[crystallisation](/examples/crystallisation) for a full model on real
data. [Hybrid ODE](/examples/hybrid-ode) covers where a network sits
relative to the solver. The [custom predictor](/examples/custom-predictor)
and [custom loop](/examples/custom-loop) pages are for when the stock
components no longer fit your model.
