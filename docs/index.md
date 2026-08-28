---
layout: home

hero:
  name: hybridmodels
  text: Fit the unknown half of an ODE
  tagline: Keep the physics you trust in closed form. Learn the part you cannot write down with a neural network. Training runs on irregular time-series data, stays inside the ranges you declare, and compiles to a tight, correct JAX kernel.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Write your own training loop
      link: /examples/custom-loop
    - theme: alt
      text: API reference
      link: /api/

features:
  - title: You write the physics
    details: simulate_fn is one function that integrates your ODE for a single experiment. You write it; the library owns vmap, jit, and autodiff. There is no base class to subclass and no model object to configure.
  - title: Regular and irregular data
    details: Every measured quantity carries its own timestamps. make_dataset merges them per experiment, records which cells are real, and groups experiments by grid length, so the solver always sees rectangular arrays. No padding, no interpolation.
  - title: Physical quantities stay in range
    details: A BoundedPredictor declares a low and a high value for every input and output. The network inside works in an unbounded space and a smooth squash maps it into range, so an out-of-range value cannot be produced and nothing is clipped.
  - title: Correct JAX, by construction
    details: One compiled kernel per bucket shape, masks instead of branches, a gradient that is correct back through the solver. An integration test pins gradient agreement across every diffrax adjoint, and a compiled-kernel review tool is part of the workflow.
  - title: Hackable at every seam
    details: Custom losses, custom regularisers, custom predictors, custom optimisers, and even a custom training loop all compose against the public API. The kernels a trainer is built from are public, so you assemble your own loop instead of forking one.
  - title: A trained run is one folder
    details: save_predictors and load_predictors round-trip the trained weights. save_run and load_run add solver settings, training settings, and loss history.
---

## In a nutshell

```python
import hybridmodels as hm

# A predictor: one trainable scalar, wrapped so its output stays in range.
predictor = hm.BoundedPredictor(
    input_keys=("T",),
    in_scaler=hm.BoundScaler(bounds=((0.0, 100.0),)),
    inner=hm.MLPPredictor(in_size=1, out_size=1, width_size=16, depth=2, key=key),
    out_scaler=hm.BoundScaler(bounds=((1e-4, 1e2),), warp="log"),
)

# You write the physics; the library owns vmap/jit/grad.
def simulate_fn(predictors, ts, covariates, y0, solver):
    k = predictors[0]({"T": covariates["temperature_C"]}).reshape(())
    return solver.diffeqsolve(diffrax.ODETerm(my_vector_field(k)), ts, y0).ys

# Train on irregular, per-channel-timestamp data.
ds = hm.make_dataset(experiments, output_channel_names=("conc", "d43"))
history, trained = hm.train_with_optax(
    predictors, ds, config, simulate_fn=simulate_fn,
    state_to_output=state_to_output, solver=solver, key=key,
)
```

This is a hybrid modelling library, not a neural ODE library: it is built
on [Diffrax](https://docs.kidger.site/diffrax/) and
[Equinox](https://docs.kidger.site/equinox/), and adds what a hybrid
pipeline needs on top:

- **Physical ranges that hold by construction** — a reparameterisation, not
  a clip, so the gradient that recovers a parameter is never destroyed.
- **Ragged measurements handled directly** — per-channel timestamps, an
  automatic mask, and grouping by axis length so JAX compiles once per
  distinct length.
- **A training loop shaped for this** — multi-phase schedules, a growing
  horizon, freezing by name or type, a restart tournament, gradients or
  population search behind one signature.
- **Extensibility without a framework.** Everything you can plug in — a
  custom predictor, loss, optimiser, regulariser, or training loop — is a
  plain callable or PyTree. There's no registry you must extend to fit in,
  and no magic behind the scenes.

Read [Getting started](/guide/getting-started) for a runnable example,
[Concepts](/guide/concepts) for the vocabulary, and
[Extending hybridmodels](/guide/extending) to see how far you can take it.