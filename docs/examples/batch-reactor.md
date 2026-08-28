---
aside: false
outline: false
---

# Batch reactor

A first-order reaction $A \to B$ whose rate constant depends on
temperature through Arrhenius, which we know, and on pH through a curve
with no first-principles form, which we do not. The classical fit gives
you one set of Arrhenius constants per pH and no way to predict between
them. The hybrid model keeps the Arrhenius trunk and adds a small
bounded residual network, so one model covers every pH at once and
interpolates. Fitted in two stages: CMA-ES for the parametric trunk,
then gradients for the residual.

Run it yourself with:

```bash
uv run python examples/batch_reactor/notebook.py
```