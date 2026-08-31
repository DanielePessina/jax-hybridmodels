# Build plan — `hybridmodels` v1

Companion to [`SPEC.md`](../../SPEC.md), [`CONTEXT.md`](../../CONTEXT.md), and the umbrella PRD at GitHub issue [#1](https://github.com/DanielePessina/jax-hybridmodels/issues/1).

This document is the **fine-grained, orchestrator-driven execution plan** for shipping `hybridmodels` v1 against the locked spec. Read alongside SPEC.md §8 (build order — coarse) and §6 (test plan).

The plan reorders SPEC §8 step 5 (Data) ahead of step 2 (Solver). Rationale: `data.py` is the deepest module with the most design surface; getting `BucketPayload` and `make_dataset` right early de-risks every subsequent training/prediction phase. Solver is shallow and lifts trivially after data lands.

---

## 1. Orchestration model

Two roles. Both are Claude. The orchestrator does not write source or test code directly — that work is delegated to subagents with rich, file-anchored prompts.

### Orchestrator (this Claude session)

- Holds the task list, mirrors progress to the GitHub issue.
- Runs all `uv`, `git`, `gh`, `rg`, and verification commands.
- Spawns subagents one phase at a time, hands back failure output for re-spawn if the first pass falls short.
- Verifies subagent output: tests green, ruff clean, ty clean, behaviour parity vs source `hybridcrystals` where the phase calls for it.
- Shapes commits and pushes after each phase.
- Gates progression: a phase is not "done" until its commit lands on `main` and the next phase is queued.

### Subagent (one Claude per module-cycle)

- Receives a prompt that names the spec sections and source-package files it must read first.
- Performs **one red→green TDD cycle for one module**: write the test file, run it red, implement, run it green.
- Does **not** commit. Hands the orchestrator a list of changed files and a one-line summary fit for a commit body.
- Does **not** push, edit `MEMORY.md`, or modify CLAUDE.md / SPEC.md.

### Two ways to spawn a subagent

The orchestrator picks per task; both are first-class.

**(A) External subprocess via `claude --dangerously-skip-permissions -p`.** Default for all module-implementation phases.

```bash
claude --dangerously-skip-permissions -p "$(cat <<'EOF'
<full prompt body — see template in §4>
EOF
)" 2>&1 | tee /tmp/subagent-phase-<n>.log
```

- `--dangerously-skip-permissions` is required because the subagent will run `uv` commands, edit files, and run tests without an interactive permission prompt. The subagent is sandboxed by repo cwd and inherits the same shell environment.
- `-p "<prompt>"` runs Claude in non-interactive (print) mode, exits when the response completes.
- `tee` captures the full transcript for orchestrator review.
- For long phases (Phase 9 optax, Phase 11 evosax) run with `&` and tail the log; the orchestrator polls completion with `wait` or by watching the log tail.
- Always cwd `/Users/danielepessina/code/jax-hybridmodels` before invocation. The subagent inherits cwd and uses it for relative paths in the prompt.
- The subagent gets a **fresh context** — no memory of prior phases. Its prompt must be self-contained: list every file it should read, every constraint it must obey, every artifact it must produce.

**(B) In-process Agent tool.** Reserved for narrow research/exploration sub-tasks (e.g. "find every place in the source package that touches mask construction") where the orchestrator wants an answer back into its own context window. Not used for module implementation, because the in-process budget would be consumed by back-and-forth file reads.

```
Agent({
  description: "Find mask handling in source package",
  subagent_type: "Explore",
  prompt: "..."
})
```

### When to use which

| Situation | Tool |
|---|---|
| TDD cycle for a module (most phases) | External `claude -p` |
| "Where in the source does X live?" | In-process Agent (`Explore`) |
| Long-running implementation that may exceed in-process tool budget | External `claude -p` |
| Quick read-only lookup in source package | In-process Agent (`Explore`) or direct `rg` |
| Behaviour parity numerical check (orchestrator runs both sides) | Orchestrator direct |

### Subagent failure handling

Default: **one re-spawn** with the failure transcript pasted into the next prompt. If the second attempt also fails, the orchestrator stops and surfaces the failure to the user. Re-spawn prompt prefix:

```
The previous run did not satisfy the definition of done. Failure transcript:

<paste failing pytest / ruff / ty output>

Diagnose the root cause and fix. Same constraints apply.
```

---

## 2. Defaults locked at kickoff

| # | Decision | Default |
|---|---|---|
| 1 | Phase 0 bootstrap | `uv init --lib`, then patch `pyproject.toml` to match SPEC §3.1 |
| 2 | Subagent failure | One re-spawn with transcript; if still red, bail to user |
| 3 | Source-package execution | `cd /Users/danielepessina/Documents/Local\ Uni/hybridcrystals/hybridcrystals && uv run …` (assumed uv-managed) |
| 4 | Commit identity | User's git identity, standard `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer; push after every phase |
| 5 | Tournament-variance assertion (Phase 9) | 5 seeds, `std(with_tournament) < std(without_tournament)` |

---

## 3. Phase plan

Each phase ends in **one commit** and **one push**. Verification gates between phases are explicit.

### Phase 0 — Project skeleton + tooling (orchestrator-direct)

SPEC §8 step 1.

| # | Task | Who | Verify |
|---|---|---|---|
| 0.1 | `uv init --lib` to bootstrap `pyproject.toml` + `src/hybridmodels/` | O | `pyproject.toml` exists |
| 0.2 | Edit `pyproject.toml`: name=`hybridmodels`, requires-python=`>=3.11`, hatchling backend, project metadata per SPEC §3.1 | O | `cat pyproject.toml` |
| 0.3 | `uv add jax jaxlib equinox diffrax optax evosax jaxkan jaxtyping rich numpy scipy` | O | `uv sync` |
| 0.4 | `uv add --dev pytest pytest-cov ruff ty` | O | `uv run pytest --version` |
| 0.5 | Add `[project.optional-dependencies] examples = ["openpyxl", "matplotlib"]` (hand-edit; uv `--optional` flag varies by version) | O | `cat pyproject.toml` |
| 0.6 | Create empty module files per SPEC §3 (`data.py`, `solver.py`, `losses.py`, `trainable.py`, `rng.py`, `prediction.py`, `serialise.py`, `predictors/{__init__,base,mlp,kan}.py`, `training/{__init__,optax,evosax}.py`, `ui/{__init__,base,optax,evosax}.py`) | O | `find src/hybridmodels -name '*.py'` |
| 0.7 | Write `src/hybridmodels/__init__.py` with **lazy `__getattr__`** stub (dispatch skeleton, empty re-export map) | O | `uv run python -c "import hybridmodels"` |
| 0.8 | Create `tests/conftest.py` + empty placeholder test files per SPEC §6 | O | `uv run pytest -q` collects 0 tests, exits 0 |
| 0.9 | Configure ruff in `pyproject.toml` (line-length 100, target-version 311, permissive ruleset to start) | O | `uv run ruff check src tests` |
| 0.10 | Create `ty.toml` at the repo root; relax `unresolved-attribute`, `invalid-method-override`, `call-non-callable` to tolerate equinox / optax dynamic-attribute patterns | O | `uv run ty check src` passes |
| 0.11 | `git add -A && git commit -m "scaffold: empty package skeleton + uv environment"` | O | `git log -1` |
| 0.12 | `git push origin main` | O | branch up to date |

---

### Phase 1 — Data (subagent — deep module, moved up from SPEC §8 step 5)

The deepest module. Build first to lock `BucketPayload` shape and the `simulate_fn` data contract.

| # | Task | Who | Verify |
|---|---|---|---|
| 1.1 | Subagent: write `tests/test_data_buckets.py` red — bucketing groups by `len(union_ts)`; mask True iff channel observed at that timestamp; `make_dataset` idempotent; per-experiment `y0` is the full state (built via `y0_fn` hook); covariates dict-shape preserved | S | red |
| 1.2 | Subagent: write `tests/test_data_split.py` red — `split_dataset(train=0.8, val=0.1, test=0.1, key)` non-overlapping splits, each split valid-bucketed | S | red |
| 1.3 | Subagent: implement `src/hybridmodels/data.py` — `ChannelObs`, `Experiment`, `BucketPayload`, `Dataset`, `make_experiment`, `make_dataset`, `split_dataset` | S | green |
| 1.4 | Smoke check: orchestrator constructs a tiny synthetic 3-experiment dataset and confirms produced `BucketPayload` shapes match what `simulate_fn` later expects | O | `uv run python -c "..."` |
| 1.5 | ruff + ty | O | clean |
| 1.6 | Commit `feat(data): bucketed-irregular dataset + split` | O | `git log -1` |
| 1.7 | Push | O | up to date |

---

### Phase 2 — Solver (subagent — small)

SPEC §8 step 2. Quick win after the deep data phase.

| # | Task | Who | Verify |
|---|---|---|---|
| 2.1 | Subagent: write `tests/test_solver.py` red — (a) `SolverConfig` static-field shape, (b) `to_dict`/`from_dict` round-trip via `SOLVER_REGISTRY`, (c) `register_solver` extends the registry | S | red |
| 2.2 | Same subagent: implement `src/hybridmodels/solver.py` to green | S | green |
| 2.3 | Add `SolverConfig`, `SOLVER_REGISTRY`, `register_solver` to `__init__.py` lazy map | O | `uv run python -c "from hybridmodels import SolverConfig"` |
| 2.4 | ruff + ty | O | clean |
| 2.5 | Commit `feat(solver): SolverConfig + registry + register_solver` + push | O | clean |

---

### Phase 3 — Predictors base + serialisation invariant (subagent — keystone)

SPEC §8 step 3. **R-A5 round-trip is the gate.** Every predictor in later phases extends `tests/test_predictors_serialise.py`'s parametrise list.

| # | Task | Who | Verify |
|---|---|---|---|
| 3.1 | Subagent: write `tests/test_predictors_serialise.py` red — parametrised over a placeholder `BoundedPredictor(<stub Predictor>)` fixture; assert `eqx.tree_serialise_leaves` round-trip is bit-exact | S | red |
| 3.2 | Subagent: write `tests/test_predictors_base.py` red — `BoundedPredictor.input_keys` ordering and dict/Array polymorphism (`TestBoundedPredictorInputKeys`), `BoundScaler` sigmoid bidirection (`from_latent(to_latent(x)) ≈ x` within bounds), `BoundedPredictor` composition, `reinitialize_with_key` re-inits inexact-float leaves of one Module, `reinitialize_pytree_with_key` splits an attempt key by traversal order across a tuple/dict pytree (R-T8) | S | red |
| 3.3 | Subagent: implement `src/hybridmodels/predictors/base.py` to green | S | green |
| 3.4 | `__init__.py` exports | O | import smoke |
| 3.5 | ruff + ty | O | clean |
| 3.6 | Commit `feat(predictors): base — Predictor, BoundedPredictor, BoundScaler, pytree re-init` + push | O | clean |

---

### Phase 4 — MLP predictor (subagent)

SPEC §8 step 4.

| # | Task | Who | Verify |
|---|---|---|---|
| 4.1 | Subagent: extend `tests/test_predictors_serialise.py` parametrise list with `MLPPredictor`; add `tests/test_predictors_mlp.py` for shape/forward-pass invariants | S | red |
| 4.2 | Subagent: implement `src/hybridmodels/predictors/mlp.py` wrapping `eqx.nn.MLP`, all hyperparams in `eqx.field(static=True)` | S | green |
| 4.3 | Cross-check vs source `hybridcrystals/regressors/mlp.py` — strip embedding code; orchestrator confirms no embedding leakage | O | manual |
| 4.4 | ruff + ty + commit + push | O | clean |

---

### Phase 5 — Losses + prediction (subagent)

SPEC §8 step 6.

| # | Task | Who | Verify |
|---|---|---|---|
| 5.1 | Subagent: write `tests/test_loss_functions.py` red — masked losses respect mask; `channel_idx` restriction; `channel_weights` weighting; `bal_*` per-experiment normalisation | S | red |
| 5.2 | Subagent: implement `src/hybridmodels/losses.py` (four pure functions + `LOSS_REGISTRY`) | S | green |
| 5.3 | Subagent: implement `src/hybridmodels/prediction.py` (`predict_bucket`, `predict_dataset`) — thin vmap+jit wrappers, no dedicated test (covered indirectly by `test_train_optax.py` later) | S | smoke |
| 5.4 | **Behaviour parity gate**: orchestrator runs source-package `hybridcrystals/losses.py::irregular_*_from_batch` on a fixture, runs new losses on the same fixture, asserts agreement within `rtol=1e-4` | O | numerical |
| 5.5 | ruff + ty + commit + push | O | clean |

---

### Phase 6 — Trainability filter (subagent)

SPEC §8 step 7.

| # | Task | Who | Verify |
|---|---|---|---|
| 6.1 | Subagent: write `tests/test_trainable_filters.py` red — default mask trains all float arrays; `freeze_modules_of_type(mask, predictor, BoundScaler)` zeros the right leaves; `freeze_paths` and `freeze_where` compose; mask shape matches predictor structure | S | red |
| 6.2 | Subagent: implement `src/hybridmodels/trainable.py` | S | green |
| 6.3 | ruff + ty + commit + push | O | clean |

---

### Phase 7 — RNG named-fold helper (subagent — tiny)

SPEC §8 step 8.

| # | Task | Who | Verify |
|---|---|---|---|
| 7.1 | Subagent: `tests/test_rng.py` red — `fold(root, "name")` is stable; same root + same name = same key; different name = different key; missing root key raises in training entry points | S | red |
| 7.2 | Subagent: implement `src/hybridmodels/rng.py` | S | green |
| 7.3 | ruff + ty + commit + push | O | clean |

---

### Phase 8 — UI base + SilentUI + RecordingUI fixture (subagent)

SPEC §8 step 9. Pull the spy fixture forward so optax tests can use it.

| # | Task | Who | Verify |
|---|---|---|---|
| 8.1 | Subagent: write `tests/test_ui_callbacks.py` red — `SilentUI` produces no stdout (verified via `capsys`); `RecordingUI` spy records lifecycle event fires; protocol shape pinned | S | red |
| 8.2 | Subagent: implement `src/hybridmodels/ui/base.py` (`TrainingUI` + `EvosaxUI` Protocols, `SilentUI`, `RecordingUI` exported under `hybridmodels.ui.testing`) | S | green |
| 8.3 | ruff + ty + commit + push | O | clean |

---

### Phase 9 — Optax training (subagent — deepest module so far)

SPEC §8 step 10. **Fat subagent prompt expected. Re-spawn likely.**

| # | Task | Who | Verify |
|---|---|---|---|
| 9.1 | Subagent: write `tests/test_train_optax.py` red — (a) trains a synthetic harmonic oscillator ODE to known parameters within `rtol=1e-2`; (b) multi-phase config with `reset_optimiser_state=(False, True)` does not crash; (c) `length_schedule=(0.5, 1.0)` does not trigger recompile (verify via JAX trace counter or by side-effect counter on a wrapped function); (d) tournament reduces variance across 5 seeds (`std(with) < std(without)`) | S | red |
| 9.2 | Subagent: implement `src/hybridmodels/training/optax.py` — `OptaxTrainingConfig`, `bucket_step`, `apply_update`, phase loop, length-schedule mask cutoff, shared tournament with diffrax-error/non-finite drop + fresh-RNG retry, missing-key raise, `RecordingUI` event fires | S | green |
| 9.3 | **Behaviour parity gate**: orchestrator runs `hybridcrystals/thesis_training/sharedgrowth.py` for ~50 steps on a tiny dataset, captures final loss; runs the new optax trainer on the same dataset/predictor for 50 steps; asserts within `rtol=1e-2` | O | numerical |
| 9.4 | ruff + ty + commit + push | O | clean |

---

### Phase 10 — RichTrainingUI (subagent — visual)

SPEC §8 step 11. **No new tests beyond the spy** — Rich rendering is verified by eye in Phase 14.

| # | Task | Who | Verify |
|---|---|---|---|
| 10.1 | Subagent: implement `src/hybridmodels/ui/optax.py::RichTrainingUI` — single `rich.live.Live`, panels swap on phase transition, compile-progress panel | S | manual |
| 10.2 | Smoke: orchestrator runs Phase 9's harmonic oscillator with `verbose=True`, eyeballs Rich panel rendering and lifecycle event firing | O | visual |
| 10.3 | ruff + ty + commit + push | O | clean |

---

### Phase 11 — Evosax training (subagent)

SPEC §8 step 12.

| # | Task | Who | Verify |
|---|---|---|---|
| 11.1 | Subagent: write `tests/test_train_evosax.py` red — (a) trains a 4-D synthetic kinetic problem to known minimum within `rtol=1e-2`; (b) `init="lhs_box"` gives wider population spread (max-pairwise-distance) than `"warm"`; (c) flatten/unflatten round-trip via `eqx.partition` + `ravel_pytree` is exact | S | red |
| 11.2 | Subagent: implement `src/hybridmodels/training/evosax.py` — `EvosaxTrainingConfig`, `_build_strategy`, `single_eval`, `population_eval`, three init modes (LHS via `scipy.stats.qmc` host-side), host-side best tracking | S | green |
| 11.3 | ruff + ty + commit + push | O | clean |

---

### Phase 12 — RichEvosaxUI (subagent — visual)

SPEC §8 step 13.

| # | Task | Who | Verify |
|---|---|---|---|
| 12.1 | Subagent: implement `src/hybridmodels/ui/evosax.py::RichEvosaxUI` | S | manual |
| 12.2 | Smoke + commit + push | O | visual |

---

### Phase 13 — KAN predictor (subagent)

SPEC §8 step 14.

| # | Task | Who | Verify |
|---|---|---|---|
| 13.1 | Subagent: extend `tests/test_predictors_serialise.py` parametrise list with `KANPredictor`; add minimal forward-pass test | S | red |
| 13.2 | Subagent: implement `src/hybridmodels/predictors/kan.py` wrapping `jaxkan`, all hyperparams static | S | green |
| 13.3 | Cross-check vs source `hybridcrystals/regressors/kan.py` and `regressor_kanx.py` | O | manual |
| 13.4 | ruff + ty + commit + push | O | clean |

---

### Phase 14 — Crystallisation example (subagent — verification gate)

SPEC §8 step 15. **The integration gate before serialisation ships.** (Phase numbering shifts down by one: NeuralNPolynomial is deferred to post-v1 per SPEC §2.3, so its prior Phase-14 slot is removed.)

| # | Task | Who | Verify |
|---|---|---|---|
| 14.1 | Subagent: port `examples/crystallisation/loader_excel.py` from source — isolated, `openpyxl` extras | S | smoke load |
| 14.2 | Subagent: write `examples/crystallisation/ode.py` — moments + concentration `simulate_fn` (mandatory signature, predictors-pytree first arg). The supersaturation-polynomial form lives here, expanded as user code (~3 lines) — see SPEC §2.3. | S | smoke run |
| 14.3 | Subagent: write `examples/crystallisation/kinetic_predictor.py` — CNT nucleation + power-law growth as `BoundedPredictor`s wrapped into a tuple `(growth_BP, nucleation_BP)` per the convention | S | shapes |
| 14.4 | Subagent: write `examples/crystallisation/mlp_predictor.py` — predictors tuple `(growth_BP, nucleation_BP)` where each is `BoundedPredictor(MLPPredictor(...))` | S | shapes |
| 14.5 | Subagent: write `examples/crystallisation/train_optax.py` — end-to-end script reproducing `thesis_training/sharedgrowth.py`, using the predictors-tuple convention | S | runs |
| 14.6 | **Verification gate**: orchestrator runs both new and source scripts on the same Excel data; asserts final loss within `rtol=1e-2` | O | numerical |
| 14.7 | Subagent: write `examples/crystallisation/train_evosax_kinetic.py` — same data, evosax search over the kinetic predictors tuple | S | runs |
| 14.8 | Commit + push | O | clean |

---

### Phase 15 — Serialisation (subagent — last shipped per R-A5)

SPEC §8 step 16.

| # | Task | Who | Verify |
|---|---|---|---|
| 15.1 | Subagent: write `tests/test_serialise.py` red — `save_predictor`/`load_predictor` round-trip on a tuple-of-BoundedPredictor pytree, `save_run`/`load_run` directory-shape contract per CONTEXT.md | S | red |
| 15.2 | Subagent: implement `src/hybridmodels/serialise.py` — four functions, `metadata.json` shape | S | green |
| 15.3 | **Verify against Phase 14**: save a trained predictors pytree, reload, re-run prediction, assert bit-exact agreement | O | numerical |
| 15.4 | ruff + ty + commit + push | O | clean |

---

### Phase 16 — Pendulum example (subagent — domain-agnostic gate)

SPEC §8 step 17. **Final v1 deliverable.**

| # | Task | Who | Verify |
|---|---|---|---|
| 16.1 | Subagent: write `examples/pendulum/train.py` — pendulum `simulate_fn` (closed-form `θ̈ = -(g/L)sin(θ)`), single-MLP `predictors = (BoundedPredictor(MLPPredictor(...)),)` for unknown `g/L`, train via `train_with_optax` to recover the constant | S | runs |
| 16.2 | **Domain-agnostic gate**: orchestrator runs `rg -i "crystal\|moment\|mu0\|d43" src/hybridmodels/`; asserts no matches | O | grep clean |
| 16.3 | Final commit + tag `v0.1.0` + push tags | O | tag pushed |

---

## 4. Subagent prompt template

Used for every external `claude -p` invocation. Phase-specific bits in `<angle brackets>` get filled by the orchestrator before the run.

```
You are working on jax-hybridmodels, a JAX/Equinox library being TDD-built per a locked spec.

## Required reading (in order, read them all before writing anything)

1. /Users/danielepessina/code/jax-hybridmodels/CLAUDE.md
2. /Users/danielepessina/code/jax-hybridmodels/AGENTS.md
3. /Users/danielepessina/code/jax-hybridmodels/SPEC.md  (focus: §<relevant section> + REQUIREMENTS R-<relevant ids>)
4. /Users/danielepessina/code/jax-hybridmodels/CONTEXT.md  (terms used: <term1>, <term2>, ...)
5. /Users/danielepessina/code/jax-hybridmodels/docs/agents/build-plan.md  (this phase: §<phase number>)

## Source-package reference

For behaviour parity, you may consult (read-only):
/Users/danielepessina/Documents/Local\ Uni/hybridcrystals/hybridcrystals/<source file>

The source package is *behaviour reference*, not structure reference. The new package is a strong refactor.

## Your task this run

<phase-specific task — TDD red→green for module X. State the test file path, the impl file path, and the specific behaviours each test must pin.>

## Constraints (all enforced; orchestrator will reject non-compliant work)

- TDD: write the test first, see it red, implement, see it green. No horizontal slicing.
- uv-managed: every command via `uv run …`. No bare `pip`/`python`/`pytest`.
- Composition over inheritance. No method overriding.
- No comments unless the *why* is non-obvious. No emojis anywhere.
- All `eqx.Module` configuration in `eqx.field(static=True)`; only float arrays as dynamic leaves.
- Predictors must round-trip through `eqx.tree_serialise_leaves`. If you add a predictor, also extend `tests/test_predictors_serialise.py`'s parametrise list.
- No new abstractions beyond what the spec names. If tempted, stop and write a question instead.
- Do NOT commit. Do NOT push. Do NOT touch CLAUDE.md / SPEC.md / MEMORY.md.

## Definition of done for this run

1. Test file exists at <path> and was red on first run (paste failing pytest output).
2. Implementation file at <path> makes the test green (paste passing pytest output).
3. `uv run ruff check src tests` clean.
4. `uv run ty check src` clean.
5. Hand back: bullet list of files changed + a one-line summary fit for a commit message body.
```

### Re-spawn prompt prefix

When the first run fails verification:

```
The previous run did not satisfy the definition of done. Failure transcript:

<paste failing pytest / ruff / ty output verbatim>

Diagnose the root cause and fix. Same task, same constraints, same definition of done.
```

---

## 5. Verification + commit cadence

### Per-phase verification

Every phase ends with the orchestrator running, in order:

1. `uv run pytest -q` — full suite green (not just the new test file)
2. `uv run ruff check src tests` — clean
3. `uv run ty check src` — clean
4. Phase-specific behaviour parity check (Phases 5, 9, 15, 16) against source `hybridcrystals`

A phase that fails any of these does **not** commit — orchestrator re-spawns the subagent or escalates to user.

### Commit shape

- Subject: `feat(<area>): <one-liner>` for new features. `test:`, `chore:`, `fix:` as appropriate.
- Body: links the build-step number from SPEC §8 and references this phase number.
- Trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

Heredoc form:

```bash
git commit -m "$(cat <<'EOF'
feat(data): bucketed-irregular dataset + split

Phase 1 of docs/agents/build-plan.md (SPEC §8 step 5, moved up).
Implements ChannelObs / Experiment / BucketPayload / Dataset and
make_dataset / split_dataset. Tests in tests/test_data_buckets.py
and tests/test_data_split.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Push cadence

`git push origin main` after every phase commit. Keeps the GitHub issue (#1) and the codebase in lockstep.

### Numerical tolerances (parity gates)

| Phase | Gate | Tolerance |
|---|---|---|
| 5 | Loss function values vs source-package losses | `rtol=1e-4` |
| 9 | Final loss after 50 steps vs source `sharedgrowth.py` | `rtol=1e-2` |
| 14 | Final crystallisation training loss vs source | `rtol=1e-2` |
| 15 | Reloaded predictors pytree predictions vs in-memory | bit-exact |
| 16 | Recovered `g/L` vs ground truth | `rtol=1e-2` |

### Domain-agnostic gate (Phase 16)

```bash
rg -i "crystal|moment|mu0|d43" src/hybridmodels/ && echo "FAIL" || echo "PASS"
```

Must print `PASS`.

---

## 6. Issue-tracker integration

GitHub Issues on `DanielePessina/jax-hybridmodels`. Per [`docs/agents/issue-tracker.md`](./issue-tracker.md).

- Umbrella PRD: [issue #1](https://github.com/DanielePessina/jax-hybridmodels/issues/1), labelled `needs-triage` at creation.
- After each phase commit, orchestrator posts a comment on issue #1: `Phase <n> complete. Commit: <sha>. Tests: <count> passing.`
- If a subagent's second-attempt failure surfaces a spec ambiguity, orchestrator opens a separate issue labelled `needs-info` and pauses the build.
