---
layout: home

hero:
  name: hybridmodels
  text: Fit the unknown half of an ODE
  tagline: You have time-series measurements. You believe an ODE governs them. Part of that ODE is known physics and part is not. This library fits the unknown part with a neural network, and keeps every physical quantity inside a range you declare.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Crystallisation walkthrough
      link: /examples/crystallisation-notebook
    - theme: alt
      text: API reference
      link: /api/

features:
  - title: You write the physics
    details: You write one function that integrates your ODE for a single experiment. The library runs it across every experiment at once, compiles it, and pushes gradients back through the integrator into the neural network. There is no base class to subclass and no model object to configure.
  - title: Physical quantities stay in range
    details: A BoundedPredictor declares a low and a high value for every input and every output. The network inside works in an unbounded space and a smooth squash maps its output into the declared range, so an out-of-range value cannot be produced and nothing has to be clipped.
  - title: Ragged measurements, no padding
    details: Each measured quantity carries its own timestamps. Two instruments sampling one experiment at different times is the normal case. make_dataset merges the timestamps per experiment, records which cells are real, and groups experiments by grid length so the solver still gets rectangular arrays.
  - title: Gradients or population search
    details: train_with_optax follows gradients and runs a multi-phase schedule. train_with_evosax runs CMA-ES over a small parameter vector, for problems with many local minima. Both take the same arguments and return the same pair, so switching or chaining them is one line.
  - title: Freeze what you already know
    details: Trainability is a tree of booleans matching your model. Freezers select parts of it by name, by module type, or by an arbitrary test, and they compose. Use them to hold a calibrated parameter fixed while the rest of the model trains.
  - title: A trained run is one folder
    details: save_predictors and load_predictors round-trip the trained weights. save_run and load_run add the solver settings, the training settings, and the loss history, so a finished run is a directory you can reload later.
---

## What this is for

You measured something over time. You have a differential equation that
explains part of what you measured, and a term in it you cannot write
down: a reaction rate that depends on pH in some unknown way, a growth
law nobody has derived, a correction you know is missing. Fitting the
whole thing with a neural network throws away the physics you have.
Fitting a guessed functional form commits you to a mechanism you cannot
justify.

This library is for the model in between. You keep the structure you
know in closed form and learn only the parts you do not. It is a
**hybrid** modelling package, not a neural ODE package.
[Diffrax](https://docs.kidger.site/diffrax/) and
[Equinox](https://docs.kidger.site/equinox/) already do neural ODEs, and
this library is built on them.

What it adds on top:

- **Physical ranges that hold by construction.** Declare `(low, high)`
  for every quantity a network produces. A reparameterisation keeps it
  inside, with no clipping anywhere, so the gradient that would recover
  a parameter never gets destroyed. You choose separately what "halfway
  between the bounds" means and how the map saturates at the edges.
- **Ragged measurements handled directly.** Per-channel timestamps, a
  merged axis per experiment, an automatic mask, and grouping by axis
  length so JAX compiles once per distinct length. No padding, no
  interpolation, no mask code from you.
- **A training loop shaped for this.** Multi-phase schedules, a
  horizon that grows during the run, freezing by name or by type, a
  restart tournament for unlucky initialisations, gradients or
  population search behind one signature, and a finished run saved as
  one folder.

Read [Getting started](/guide/getting-started) for a runnable example,
then [Concepts](/guide/concepts) for the vocabulary.
