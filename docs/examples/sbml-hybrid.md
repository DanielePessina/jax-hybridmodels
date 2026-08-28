# SBML hybrid kinetics

An *external* mechanistic model — the kind you would load from an SBML
file — plugged into a `hybridmodels` fit, with one of its rate parameters
supplied by a neural `BoundedPredictor`.

The mechanistic core is a Michaelis-Menten system. The true maximum
velocity `Vmax` depends on the enzyme level `E` via `Vmax(E) = 2E`. We
keep the known physics (Michaelis-Menten with known `Km`) and let an MLP
*learn* `Vmax` as a function of `E` from noisy S/P time-series across
experiments at different enzyme levels — the crystallisation pattern
(known physics for structure, a network for the unknown rate law) applied
to a standard systems-biology model.

The example documents how to swap the hand-written core for a real SBML
file via `jaxkineticmodel`:

```python
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
kmodel = SBMLModel("model.xml").get_kinetic_model()  # kmodel(ts, y0, params) -> ys
```

The external model replaces the whole integration block — it carries its
own diffrax solver, tolerances, and adjoint — so `simulate_fn` injects the
neural rate into its params and calls it directly.

```bash
uv run python examples/sbml_hybrid/train_sbml_hybrid.py
```

Note: `jaxkineticmodel` currently pins `jax==0.4.35` / `optax==0.2.3`,
which conflict with `hybridmodels`' `jax>=0.10` / `optax>=0.2.8`, so it
must run in a separate environment. The recipe — not the dependency — is
the point.