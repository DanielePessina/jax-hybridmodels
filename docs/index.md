---
layout: home

hero:
  name: hybridmodels
  text: Hybrid models in JAX
  tagline: Compose trainable function approximators (MLP, KAN, ...) with user-written ODE dynamics, and train them on irregular time-series experiments. Crystallisation kinetics is the canonical example, not the scope.
  actions:
    - theme: brand
      text: Crystallisation Walkthrough
      link: /examples/crystallisation
    - theme: alt
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/

features:
  - title: Composition, not inheritance
    details: A "model" is a loose triple — predictors pytree, your simulate_fn, a SolverConfig. No Model wrapper class to subclass; the framework owns vmap / jit / grad, you own the physics.
  - title: Bounded, self-describing predictors
    details: BoundedPredictor wraps any inner predictor with sigmoid-scaled input/output bounds. Inner networks never have to clamp themselves; they always run in unbounded latent space.
  - title: Bucketed irregular time-series
    details: Each channel carries its own timestamps. Experiments are grouped by union-axis length and stacked into BucketPayloads — one JIT compilation per bucket shape.
  - title: Two trainers, one signature
    details: train_with_optax (gradient-based, multi-phase, shared tournament) and train_with_evosax (population-based) accept the same arguments and return the same (history, predictors) tuple.
  - title: Boolean PyTree freezing
    details: Trainability is a mask matching the predictors pytree. Freezers compose by path, by module type, or by arbitrary predicate — no per-class registry.
  - title: Round-trip serialisation
    details: save_predictors / load_predictors round-trip any predictors pytree via Equinox; save_run / load_run bundle predictors + solver config + training config + loss history into one directory.
---

The package is JAX-first: every public entry point is jit-friendly, gradients flow through the integrator via `diffrax.DirectAdjoint`, and the same pipeline runs unchanged on CPU, GPU, and TPU. Vocabulary, pytree conventions, and the surrounding architectural decisions are documented under [Guide → Concepts](/guide/concepts) and the ADRs in `docs/adr/` of the repo.
