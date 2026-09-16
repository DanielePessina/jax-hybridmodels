# Agent orientation — jax-hybridmodels

You are working on a **JAX/Equinox library for hybrid (ODE + neural) models**. The v1 build order in SPEC.md §8 is complete and the suite is green; work now is refinement, correctness, and release readiness rather than first implementation. The design is locked in the sense that SPEC.md's recorded decisions are not up for casual revision — but SPEC.md does grow, deliberately, when a decision genuinely needs revisiting (the penalty redesign is a worked example of reversing a documented choice).

## Read these first, in order

1. **[`SPEC.md`](./SPEC.md)** — architectural specification. The `REQUIREMENTS` section is the contract. The `Build order (TDD)` section tells you what to implement next.
2. **[`CONTEXT.md`](./CONTEXT.md)** — domain glossary. Every term used in SPEC and code is defined here.

## Core invariants (do not break)

- **No `Model` wrapper class.** A "model" is the loose triple `(predictor, simulate_fn, solver_config)`.
- **`simulate_fn` is user-written.** It has a mandatory signature (SPEC §4.2). The framework owns vmap/jit/grad; the user owns physics.
- **Bucketed-irregular is the only data interface.** No padded `UnscaledBatchedExperiments` pathway.
- **Composition over inheritance** for predictors, per Equinox's abstract/final pattern. No method overriding.
- **All predictors must round-trip** through `eqx.tree_serialise_leaves`. This is enforced by `tests/test_predictors_serialise.py` and applies to every new predictor.
- **Trainability is a boolean PyTree mask**, not a per-class registry.
- **Training step = full pass over all buckets → accumulate gradients → one optimizer update.** Bucket ≠ step.
- **Shared-tournament-only.** No vmapped or serial tournament modes.
- **RNG: root key is user-supplied, never defaulted.** Internal subkeys via `rng.fold(root, "name")` (named folds, not chained splits).

## Toolchain — uv only

This project is uv-managed. Do not invoke `pip`, `python`, or `pytest` directly.

| Action | Command |
|---|---|
| Add a dependency | `uv add <pkg>` |
| Add a dev dependency | `uv add --dev <pkg>` |
| Sync the env (after pulling) | `uv sync` |
| Run tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_data_buckets.py` |
| Run an example | `uv run python examples/crystallisation/train_optax.py` |
| Lint | `uv run ruff check .` |
| Typecheck | `uv run ty check src` |

If you need a one-off Python invocation, use `uv run python -c '...'`.

## Workflow expectations

- **TDD.** Write the test for a module before its implementation. The test file is the per-module spec; SPEC.md is the cross-module spec.
- **Build order is in `SPEC.md` §8.** Don't skip ahead. Each step has tests that must be green before the next begins.
- **Do not invent abstractions.** If you're tempted to add a class hierarchy, a registry, a wrapper, or a callback layer that isn't in SPEC.md, surface it as a question instead. The spec is intentionally minimal; the user has explicitly rejected over-architecting (Karpathy guidelines were invoked during design).
- **Surface assumptions.** If the spec is ambiguous on a concrete decision, ask before implementing. Don't fill in defaults silently.
- **Update `CONTEXT.md` if a term changes meaning.** It's an inline glossary, not a frozen artifact.
- **Record decisions in SPEC.md.** If a decision is hard to reverse, would surprise a future reader, or carries a real tradeoff, say so explicitly in SPEC.md — there are no separate decision documents.

## Cross-package context

The source-of-truth implementation being ported lives at:
`/Users/danielepessina/Documents/Local Uni/hybridcrystals/hybridcrystals`

The latest verification script there is `thesis_training/sharedgrowth.py` (and the `Thesis_sharedgrowth_*` test runners). Reference these for *behaviour*, not for *structure* — the new package is a strong refactor, not a transcription. SPEC.md §7 is the migration map.

## What's explicitly not your job

- Designing new abstractions beyond SPEC.md.
- Porting Bayesian/GP/embeddings code (out of scope).
- Adding plotting beyond what an example script needs.
- Optimising for >20 buckets or >100-dim evosax searches.

## Package skill

For code, debugging, or documentation work that uses `hybridmodels`, also read
[`skills/jax-hybrid-models/SKILL.md`](./skills/jax-hybrid-models/SKILL.md).
Its supporting references contain the package API patterns and known JAX,
Diffrax, training, and serialisation traps.
