---
layout: home

hero:
  name: hybridmodels
  text: Hybrid models in JAX
  tagline: Compose trainable function approximators with hand-written ODE dynamics, and fit them to irregular time-series experiments. Built on JAX, Equinox, and Diffrax.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Crystallisation Walkthrough
      link: /examples/crystallisation-notebook
    - theme: alt
      text: API Reference
      link: /api/

features:
  - title: Composition over inheritance
    details: A model is a triple — a predictors PyTree, a user-supplied simulate_fn, and a SolverConfig. The framework runs the JAX plumbing (vectorisation across experiments, JIT, and gradients through the integrator); the user supplies the physics. There is no Model base class to subclass.
  - title: Bounded predictors
    details: BoundedPredictor wraps any inner predictor (MLPPredictor, KANPredictor, or a custom subclass) with sigmoid-scaled physical-units bounds on inputs and outputs. The inner network operates in unbounded latent space, so it never has to clamp its own values; bounds are owned at the wrapper level and are part of the saved model.
  - title: Irregular time-series
    details: Each observation channel carries its own timestamps. make_dataset forms per-experiment union grids, derives observation masks automatically, and groups experiments by grid length into buckets. Each bucket is a JIT cache key — one compilation per bucket shape, then full JAX speed across every step.
  - title: Two trainers, one signature
    details: train_with_optax (gradient-based, multi-phase, optional shared-tournament restart loop) and train_with_evosax (population-based, CMA-ES and others) accept the same arguments and return the same (loss_history, trained_predictors) tuple. Switching is a one-line change; chaining the two is straightforward.
  - title: Composable trainability masks
    details: Trainability is a PyTree of booleans matching the predictors structure. Freezers select leaves by attribute path, by module type, or by an arbitrary predicate, and they compose. Use them for staged training, freezing scaler temperature, or freezing every leaf except one for diagnostic runs.
  - title: Reproducible run archives
    details: save_predictors and load_predictors round-trip any predictors PyTree via Equinox leaf serialisation. save_run and load_run bundle predictors, solver configuration, training configuration, and loss history into a single directory, so a trained run is one folder.
---

The package is JAX-first: every public entry point is `jit`-friendly, gradients flow through the integrator via `diffrax.DirectAdjoint`, and the same code runs unchanged on CPU, GPU, and TPU. The vocabulary, PyTree conventions, and architectural decisions are documented under [Guide → Concepts](/guide/concepts).
