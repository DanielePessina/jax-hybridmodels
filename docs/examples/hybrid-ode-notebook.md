---
aside: false
outline: false
---

# Hybrid ODE, runnable notebook

A guided version of `examples/hybrid_ode/train_hybrid_ode.py`, written for
a reader who has not used this library before. It walks through getting
irregularly sampled data into the library, bounding physical quantities so
they cannot leave their range, placing one network above the solver and
another inside it, and checking afterwards whether the fitted rate law is
the real one.

Exported from `examples/hybrid_ode/notebook.py`. Run it interactively with
`uv run marimo edit examples/hybrid_ode/notebook.py`, or as a plain script
with `uv run python examples/hybrid_ode/notebook.py`.

<iframe
  src="/jax-hybridmodels/notebooks/hybrid_ode.html"
  style="width: 100%; height: calc(100vh - 80px); border: 1px solid var(--vp-c-divider); border-radius: 8px;"
  title="Hybrid ODE notebook"
></iframe>
