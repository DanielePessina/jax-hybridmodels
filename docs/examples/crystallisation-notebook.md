---
aside: false
outline: false
---

# Crystallisation: runnable notebook

Two models of the same batch crystallisation, fitted to the same four
experiments and compared side by side. The first learns the growth and
nucleation rate laws with a pair of bounded neural networks. The second
keeps Classical Nucleation Theory and fits its four constants with
CMA-ES. The dataset, the population-balance ODE, the solver, and the
diagnostics are shared, so the only thing that differs is the trainable
part.

Exported from `examples/crystallisation/notebook.py`. Run it yourself
with `uv run marimo edit examples/crystallisation/notebook.py`.

<iframe
  src="/jax-hybridmodels/notebooks/crystallisation.html"
  style="width: 100%; height: calc(100vh - 80px); border: 1px solid var(--vp-c-divider); border-radius: 8px;"
  title="Crystallisation notebook"
></iframe>
