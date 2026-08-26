---
aside: false
outline: false
---

# Batch reactor: runnable notebook

A first-order reaction $A \to B$ whose rate constant depends on
temperature through Arrhenius, which we know, and on pH through a curve
with no first-principles form, which we do not. The classical fit gives
you one set of Arrhenius constants per pH and no way to predict between
them. The hybrid model keeps the Arrhenius trunk and adds a small
bounded residual network, so one model covers every pH at once and
interpolates. Fitted in two stages: CMA-ES for the parametric trunk,
then gradients for the residual.

Exported from `examples/batch_reactor/notebook.py`. Run it yourself with
`uv run marimo edit examples/batch_reactor/notebook.py`.

<iframe
  src="/jax-hybridmodels/notebooks/batch_reactor.html"
  style="width: 100%; height: calc(100vh - 80px); border: 1px solid var(--vp-c-divider); border-radius: 8px;"
  title="Batch reactor notebook"
></iframe>
