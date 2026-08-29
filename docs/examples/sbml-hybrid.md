# SBML hybrid kinetics

A published, curated SBML model — Fujita et al., *Sci. Signal.* 2010, the
EGF/ERK cascade (9 species, 11 reactions) — loaded from an actual `.xml`
file and plugged into a `hybridmodels` fit, with one of its rate parameters
supplied by a neural `BoundedPredictor`.

The file is parsed with a small hand-rolled converter (`sbml_loader.py`):
`python-libsbml` gives the MathML AST for every kinetic law, `sympy`
converts it into an expression, and `sympy.lambdify(..., "jax")` turns each
reaction's rate law into a callable JAX function. The converter is ~300
lines, example-local, and reads the SBML's own structure: species, global
and local parameters, stoichiometry, initial assignments, and
state-dependent assignment rules (which the example uses as its measurement
model — the paper's scaled "total" readouts `pEGFR_tot`, `pAkt_tot`,
`pS6_tot`).

The hybrid twist is the crystallisation pattern applied to a systems
biology model: the SBML supplies the known structure (stoichiometry and
published kinetics), and an MLP learns the one *unknown rate law* — how
the EGF-EGFR association constant depends on the EGF dose. Reaction `v1`
is reversible mass-action,

```
v1 = Cell * (EGF * EGFR * k1 - EGF_EGFR * k2)
```

so the example declares the forward constant `k1` unknown and lets the
network supply it as `Vmax(dose)` (the reverse constant `k2` stays from the
file). The truth is `Vmax(dose) = 2 * dose`. Data is simulated from the
SBML at four EGF doses with noisy readouts; the fit must recover the rate
law from the dose-response time series alone.

Two numerical lessons are documented in the example's docstring:

- **Keep the reverse term.** An irreversible uptake law drives EGFR to
  exactly zero, and the solver grinds on the resulting kink (some parameter
  values take >50k steps and fail). The reversible law has a well-defined
  equilibrium everywhere.
- **Kvaerno3, not Kvaerno5.** Kvaerno5's Newton solve diverges at specific
  parameter values (sharp threshold behaviour); Kvaerno3 integrates the
  whole parameter space in under ~100 steps.

```bash
uv run python examples/sbml_hybrid/train_sbml_hybrid.py
```

Dependencies (`python-libsbml`, `sympy`) are in the `examples` extra:

```bash
uv sync --extra examples
```