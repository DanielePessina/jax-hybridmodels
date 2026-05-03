# Claude Code — entry point for jax-hybridmodels

This file is your starting point. The substantive content lives elsewhere.

## Load these files into context before doing any work

1. **[`AGENTS.md`](./AGENTS.md)** — orientation, invariants, toolchain. Read fully.
2. **[`SPEC.md`](./SPEC.md)** — the architectural specification. The `REQUIREMENTS` and `Build order (TDD)` sections are the contract.
3. **[`CONTEXT.md`](./CONTEXT.md)** — domain glossary. Reference as terms come up.
4. **[`docs/adr/`](./docs/adr/)** — architectural decision records. Read before proposing structural changes.

## House rules (Claude-specific reminders)

- **uv-managed project.** Run everything via `uv run …` (`uv run pytest`, `uv run python …`, `uv run ruff check …`). Add dependencies with `uv add` / `uv add --dev`. Never `pip install`. Never call `python` or `pytest` bare.
- **TDD.** Write the test, see it red, implement, see it green. Build order is in SPEC.md §8 — don't skip ahead.
- **Karpathy-aligned discipline** (the user invoked `/karpathy-guidelines` during design):
  - Surgical changes only. No drive-by refactors.
  - No premature abstraction. If three lines repeat, three lines repeat — don't extract a helper until there's a fourth case.
  - Make assumptions explicit. If SPEC.md is silent on a decision, ask — don't pick.
  - Keep classes minimal. The framework deliberately has very few classes; do not add more without grounding in SPEC.
- **Composition over inheritance** for any new predictor or scaler. Equinox's abstract/final pattern applies.
- **Write informative comments.** Module docstrings explain the file's role and what invariants it owns. Function docstrings explain intent, contract, and shape conventions for non-trivial functions. Inline comments mark non-obvious *why* — hidden invariants, JAX tracing subtleties, deliberate references to spec/ADR sections, or workarounds. Avoid redundant comments that restate the next line of code (`i += 1  # increment i`).
- **No emojis** in code, comments, or docs unless explicitly requested.
- **Update `CONTEXT.md` inline** when a term's meaning changes. It's not a frozen artifact.

## Common tasks

| Task | Where to look |
|---|---|
| What does X mean? | `CONTEXT.md` |
| Should I do X this way? | `SPEC.md` REQUIREMENTS, then ADRs |
| What's the next thing to build? | `SPEC.md` §8 (Build order) |
| Why is X structured this way? | ADR with matching topic, then `SPEC.md` §2.2 (out-of-scope) |
| Where does the source-package equivalent live? | `SPEC.md` §7 (migration map) |
| How do I run a test? | `uv run pytest tests/test_<module>.py` |

## When something is unclear

Ask the user. The spec is intentionally tight; the design was locked through a long grilling session. If a question can't be answered from `SPEC.md` + `CONTEXT.md` + ADRs, it probably means the spec needs to grow — which is fine, but should be a deliberate update, not an inferred default.

## Agent skills

### Issue tracker

GitHub Issues on `DanielePessina/jax-hybridmodels`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles, default strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
