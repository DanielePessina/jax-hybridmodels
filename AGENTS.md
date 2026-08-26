# Agent orientation — jax-hybridmodels

You are working on a **JAX/Equinox library for hybrid (ODE + neural) models**. The v1 build order in SPEC.md §8 is complete and the suite is green; work now is refinement, correctness, and release readiness rather than first implementation. The design is locked in the sense that ADRs are not up for casual revision — but SPEC.md does grow, deliberately, when a decision genuinely needs revisiting (see ADR-0007 for a worked example of reversing a documented non-requirement).

## Read these first, in order

1. **[`SPEC.md`](./SPEC.md)** — architectural specification. The `REQUIREMENTS` section is the contract. The `Build order (TDD)` section tells you what to implement next.
2. **[`CONTEXT.md`](./CONTEXT.md)** — domain glossary. Every term used in SPEC and code is defined here.
3. **[`docs/adr/`](./docs/adr/)** — architectural decision records. Read these before proposing structural changes; the decisions captured there are not up for casual revision.

## Core invariants (do not break)

- **No `Model` wrapper class.** A "model" is the loose triple `(predictor, simulate_fn, solver_config)`. ([ADR-0001](./docs/adr/0001-no-model-wrapper-class.md))
- **`simulate_fn` is user-written.** It has a mandatory signature (SPEC §4.2). The framework owns vmap/jit/grad; the user owns physics. ([ADR-0005](./docs/adr/0005-simulate-fn-mandatory-signature.md))
- **Bucketed-irregular is the only data interface.** No padded `UnscaledBatchedExperiments` pathway. ([ADR-0004](./docs/adr/0004-bucketed-irregular-only.md))
- **Composition over inheritance** for predictors, per Equinox's abstract/final pattern. No method overriding.
- **All predictors must round-trip** through `eqx.tree_serialise_leaves`. This is enforced by `tests/test_predictors_serialise.py` and applies to every new predictor.
- **Trainability is a boolean PyTree mask**, not a per-class registry. ([ADR-0003](./docs/adr/0003-trainability-filter-as-pytree.md))
- **Training step = full pass over all buckets → accumulate gradients → one optimizer update.** Bucket ≠ step.
- **Shared-tournament-only.** No vmapped or serial tournament modes. ([ADR-0002](./docs/adr/0002-shared-tournament-only.md))
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
- **Add an ADR** only when (a) the decision is hard to reverse, (b) it would surprise a future reader, (c) there's a real tradeoff. Format in `docs/adr/` next to the existing ones.

## Cross-package context

The source-of-truth implementation being ported lives at:
`/Users/danielepessina/Documents/Local Uni/hybridcrystals/hybridcrystals`

The latest verification script there is `thesis_training/sharedgrowth.py` (and the `Thesis_sharedgrowth_*` test runners). Reference these for *behaviour*, not for *structure* — the new package is a strong refactor, not a transcription. SPEC.md §7 is the migration map.

## What's explicitly not your job

- Designing new abstractions beyond SPEC.md.
- Porting Bayesian/GP/embeddings code (out of scope).
- Adding plotting beyond what an example script needs.
- Optimising for >20 buckets or >100-dim evosax searches.
