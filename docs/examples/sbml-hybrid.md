# SBML hybrid kinetics

This example loads a published SBML model, the EGF/ERK cascade of Fujita
et al. (*Sci. Signal.* 2010, 9 species, 11 reactions), and replaces one of
its rate constants with a neural network.

`sbml_loader.py` is a short converter that lives with the example.
python-libsbml reads each kinetic law as MathML, sympy turns it into an
expression, and `sympy.lambdify(..., "jax")` makes it a JAX function. The
converter also reads species, parameters, stoichiometry, initial
assignments and the state-dependent assignment rules. Those rules are the
paper's scaled readouts `pEGFR_tot`, `pAkt_tot` and `pS6_tot`, and the
example uses them as its `state_to_output`.

Reaction `v1`, EGF binding to its receptor, is reversible mass action:

```
v1 = Cell * (EGF * EGFR * k1 - EGF_EGFR * k2)
```

The forward constant `k1` becomes `Vmax(dose)`, a `BoundedPredictor` of
the EGF dose. The reverse constant `k2` stays as published. The data are
simulated with `Vmax(dose) = 2 * dose` at four doses, with 2% noise on each
readout, and the fit has to recover that law.

```bash
uv sync --extra examples   # python-libsbml and sympy
uv run python examples/sbml_hybrid/train_sbml_hybrid.py
```

## The hybrid model

`replace_flux` swaps `k1` for a symbol named `Vmax`, and `simulate_fn`
fills it from the network before integrating the SBML vector field:

```python
sbml.replace_flux(
    "v1_reaction_1",
    expr=sp.Symbol("Cell") * (sp.Symbol("Vmax") * egf * egfr - sp.Symbol("reaction_1_k2") * egf_egfr),
    extra_params=("Vmax",),
)


def simulate_fn(predictors, ts, covariates, y0, solver):
    dose = covariates["dose"]
    vmax = predictors["Vmax"]({"dose": dose})[0]
    params = {**sbml.params, "EGF": dose, "Vmax": vmax}
    return sbml.integrate(solver, ts, y0, params)
```

The network and model:

```python
vmax = hm.BoundedPredictor(
    inner=hm.MLPPredictor(in_size=1, out_size=1, width_size=16, depth=2,
                          activation_name="tanh", key=k_model),
    input_keys=("dose",),
    input_bounds=(0.03, 0.5),
    output_bounds=(0.05, 1.0),
)
model = hm.HybridModel(
    predictors={"Vmax": vmax},
    simulate_fn=simulate_fn,
    state_to_output=state_to_output,
    output_channel_names=CHANNELS,
    solver=SOLVER,
)
result = model.fit(dataset, hm.OptaxTrainingConfig(steps=120, lr=1e-2, penalty_weight=1e-2), key=k_train)
```

## Numerical choices

- **Keep the reverse term.** An irreversible uptake law drives EGFR to
  exactly zero, and the solver stalls on the kink, taking over 50k steps
  at some parameter values. The reversible law has a smooth equilibrium.
- **Use Kvaerno3.** Kvaerno5's Newton solve diverges at isolated parameter
  values. Kvaerno3 integrates the whole range in 50 to 100 steps.

## Results

Seed 0, 120 steps. All three readouts fit with `R^2 >= 0.995`, and the
learned `Vmax` is within 5% of `2 * dose` at every training dose.

![Predicted against observed SBML readouts](assets/sbml-hybrid/parity.png)

![Learned association constant against the truth](assets/sbml-hybrid/recovered_vmax.png)
