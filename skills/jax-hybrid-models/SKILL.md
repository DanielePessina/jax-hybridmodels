---
name: jax-hybrid-models
description: "Uses the jax-hybridmodels package (JAX/Equinox hybrid ODE + neural models): building experiments and datasets, writing simulate_fn and state_to_output, configuring BoundedPredictor bounds/warps, training with Optax or evosax, prediction, and serialisation. Use when working in this repo or writing code that imports jaxhybridmodels, or when asked to build, train, evaluate, debug, or tune hybrid ODE-neural models with bounded trainable predictors on irregular time-series data."
---

# jax-hybridmodels

Use this skill for work that builds, trains, evaluates, debugs, or documents
`jaxhybridmodels`: a JAX/Equinox library for user-written ODE dynamics with
trainable predictors and bucketed-irregular observations. Crystallisation is
the canonical example, not the scope.

## Source of truth

For repository work, read these before making a design or implementation
change:

1. `AGENTS.md` for local workflow and invariants.
2. `SPEC.md` for the architectural contract and build order.
3. `CONTEXT.md` for domain terms and shape conventions.
4. The relevant tests and source module for the behaviour being changed.

Resolve conflicts in that order: tests/source and explicit SPEC decisions beat
prose docs or this skill. Do not invent a wrapper, registry, inheritance
hierarchy, or new data pathway to make a task more convenient.

Use the repository's `uv` commands. Never call `pip`, bare `python`, or bare
`pytest`:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ty check src
```

## Route the request

| Task | Read first | Main surface |
| --- | --- | --- |
| Build experiments or handle sparse observations | `docs/guide/data.md`, `src/jaxhybridmodels/data.py` | `make_experiment`, `make_dataset`, `split_dataset` |
| Write or debug the physics boundary | `docs/guide/model-interface.md`, `CONTEXT.md` | `simulate_fn`, `state_to_output`, `SolverConfig` |
| Add time-varying inputs or smooth schedules | `docs/guide/profiles-and-schedules.md` | profile factories, `annealing_schedule` |
| Choose or extend a predictor | `docs/guide/predictors.md`, `docs/guide/custom-predictors.md` | `Predictor`, `BoundedPredictor`, `BoundScaler` |
| Configure or debug training | `docs/guide/training.md`, `skills/jax-hybrid-models/caveats.md` | Optax/Evosax configs and trainers |
| Evaluate or persist a result | `docs/guide/serialization.md`, `src/jaxhybridmodels/prediction.py` | prediction, metrics, ensembles, save/load |
| Change public API documentation | `README.md`, `docs/README.md`, source docstrings | `scripts/gen_api_docs.py` then `docs/api/` |

## Model contract

There is no `Model` wrapper. A model is the pieces passed separately:

| Piece | Contract |
| --- | --- |
| `predictors` | Any PyTree of `eqx.Module` leaves; conventionally `(BoundedPredictor, ...)`. |
| `simulate_fn` | User-written pure function `(predictors, ts, covariates, y0, solver) -> [T, S]`. |
| `state_to_output` | User-written pure function `[T, S] -> [T, D]`; the framework vmaps it, so it receives one trajectory. |
| `solver` | `SolverConfig` containing static Diffrax settings. |
| `dataset` | `Dataset` made of bucket payloads; it never stores `state_to_output`. |

The framework owns `vmap`, JIT, and gradient plumbing around one experiment.
The user owns the vector field and all physics.

## Non-negotiable invariants

- Bucketed-irregular is the only data interface. `make_dataset` builds union
  timestamps and masks; users do not write mask code or padded data pathways.
- A training step visits every bucket, accumulates gradients, and applies one
  optimizer update. A bucket is not a step.
- Optax phase fields are equal-length tuples. `key=` is keyword-only and
  required on both trainers; internal randomness uses named folds.
- Trainability is a boolean PyTree mask. Use `trainable_mask` and the
  `freeze_*` functions; there is no mutable per-predictor trainability API.
- Every predictor must round-trip through
  `eqx.tree_serialise_leaves`. Keep dynamic leaves as arrays and static fields
  as JSON-compatible primitives or tuples; rebuild user callables yourself.
- Use composition for bounds: `input_keys -> in_scaler -> inner -> out_scaler`.
  Reparameterisation enforces physical bounds; penalties only discourage
  saturation.
- The shared tournament is the only tournament mode. It retries Diffrax or
  non-finite failures, excludes the fixed-point bound penalty from ranking,
  and includes any configured trajectory penalty.

## Canonical physics boundary

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    def vector_field(t, y, args):
        inputs = {"temperature": covariates["temperature"], "state": y[0]}
        rate = predictors[0](inputs)
        return physics_rhs(t, y, rate)

    return solver.diffeqsolve(
        diffrax.ODETerm(vector_field), ts, y0
    ).ys
```

Callbacks run inside JAX transformations: return fixed shapes, avoid Python
control flow on traced values, and guard divisions/logarithms before the
invalid arithmetic. See `usage.md` for a complete pipeline.

## Training and extension guidance

- Freeze `BoundScaler` leaves by convention when the scaler temperature is
  part of the predictor tree: `freeze_modules_of_type(mask, predictors,
  BoundScaler)`.
- Use Optax for differentiable medium/large parameter sets. Use Evosax for
  small, kinetic-parameter-shaped searches; it has no per-individual failure
  handling in v1.
- For a custom loop, compose the public kernels in
  `jaxhybridmodels.training.kernels` over `dataset.bucket_payloads` in Python.
- For code changes, follow TDD and update `SPEC.md` or `CONTEXT.md` only when
  a decision or domain term genuinely changes.
- For documentation changes, edit guide Markdown or source docstrings, never
  generated `docs/api/*.md` directly. Run `npm --prefix docs run docs:gen`,
  `uv run python scripts/gen_api_docs.py --check`, and
  `npm --prefix docs run docs:build`.

## References

- [usage.md](usage.md) — exact API patterns, signatures, and return shapes.
- [caveats.md](caveats.md) — JAX, numerical, solver, training, serialization,
  and RNG traps.
- Repository examples in `examples/` — executable behaviour references.
- Human documentation in `docs/guide/` and generated API reference in
  `docs/api/` — reader-oriented explanations and signatures.
