---
aside: false
outline: false
---

# Batch reactor RL, runnable notebook

A guided version of `examples/batch_reactor/train_rl_deactivation.py`. The
catalyst calibrated on the [batch reactor page](/examples/batch-reactor) fouls
as the batch runs, and a PPO policy recovers the missing activity. The policy
is an ordinary `BoundedPredictor` sampled in its latent space, so the bound is
structural and no log-determinant correction appears in the log-probability,
and PPO never differentiates the ODE.

Exported from `examples/batch_reactor/notebook_rl.py`. Run it interactively
with `uv run marimo edit examples/batch_reactor/notebook_rl.py`, or as a plain
script with `uv run python examples/batch_reactor/notebook_rl.py`.

<iframe
  src="/jax-hybridmodels/notebooks/batch_reactor_rl.html"
  style="width: 100%; height: calc(100vh - 80px); border: 1px solid var(--vp-c-divider); border-radius: 8px;"
  title="Batch reactor RL notebook"
></iframe>
