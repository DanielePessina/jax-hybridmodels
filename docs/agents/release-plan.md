# Release plan — `jaxhybridmodels` v1 → publication

Companion to [`SPEC.md`](../../SPEC.md), [`CONTEXT.md`](../../CONTEXT.md), and [`build-plan.md`](./build-plan.md). This document is the **release-readiness and reframing plan**: it covers JAX-correctness hardening, the extensibility seams (public training kernels, injectable optimizer, penalty hook, evosax algorithms), folding example patterns into the library, the docs rewrite, the marimo→scripts migration, new examples, and SPEC reconciliation. Each workstream ends with a verifiable exit criterion. Nothing here reopens recorded SPEC decisions except where noted and gated behind an explicit SPEC amendment.

## Locked decisions (from the grilling session)

| # | Decision |
|---|---|
| Q1 | SBML integration: use `jaxkineticmodel` (AbeelLab, PyPI `jaxkineticmodel`, JAX/Diffrax-native) as the JAX-native partner. RoadRunner only as a non-differentiable reference if needed. Example may use published dataset (Van Heerden glucose pulse) or synthetic. **Not** a flagship example. |
| Q2 | Extensibility = public `jaxhybridmodels.training.kernels` + config hooks (optimizer injectable, penalty hook). Not a full engine abstraction. |
| Q3 | Fold only the 4 strong pure-library patterns: `metrics`, `diffrax_solve`, `describe_buckets`+`count_trainable_params`, `evaluate`+`default_trainable`. Defer plotting/synthetic-data/symlog/bespoke plots. |
| Q4 | Optimizer: `(name | factory(learning_rate)->GradientTransformation | raw GradientTransformation)`. Names and factories are wrapped in `inject_hyperparams`; a raw instance is returned as-is and an lr change on one requires a reset. |
| Q5 | `state_to_output` moves **off the Dataset** onto the model/prediction call. Dataset becomes pure data. A recorded SPEC decision (R-D7). |
| Q6 | Evosax: add an algorithm registry + register more strategies (keep CMA-ES default). |
| Q7 | Examples reordered: code-first (custom training loop) then worked examples. jaxkineticmodel joins the others, not flagship. |
| Q8 | Convert the 4 marimo notebooks to plain scripts; delete marimo dependency, HTML embeds, orphaned html, ruff/ty exemptions, skills. Single-source scripts into VitePress docs. |

## Workstreams

### WS1 — JAX correctness hardening (research-backed)

Exit criterion: `tests/test_adjoint_consistency.py` green; a review-checklist doc exists and is applied; no correctness regressions.

1. **Adjoint-consistency test** — `tests/test_adjoint_consistency.py`: on a benign non-stiff problem, assert gradients from `SolverConfig.adjoint` in `Direct` / `RecursiveCheckpoint` / `Backsolve` agree within tolerance. Guards the recompute-vs-cache tradeoff the solver exposes.
2. **HLO-inspection review habit** — add a doc (`docs/guide/recommendations.md` or a dev note) and a smoke script: dump `jit.lower(...).compile().as_text()` for `predict_bucket`/`bucket_step` and grep for unexpected `convert`/`copy`/`retile` ops. Apply once per review.
3. **Checklist as a gate** — distill the 16-point correctness checklist (from the scaling-book research) into a review checklist. Use it when reviewing every WS2/WS3/WS5 diff.
4. No rework of the already-correct patterns (masked-not-branched, static-vs-traced, coarse jit boundaries, host-side dtype resolution, serialise-time registries). Confirmed present.

### WS2 — Core extensibility (public kernels + config hooks)

Exit criterion: a user can write a custom training loop against public `jaxhybridmodels.training.kernels` without touching privates; `optimizer=` accepts names, callables, and raw transformations; a custom `penalty_fn` composes.

1. **`jaxhybridmodels/training/kernels.py`** — promote `_build_bucket_step`, `_apply_length_mask`, `_build_score_bucket`, `_build_apply_update`, `_build_penalty_step` (and the `_predict_bucket_obs` body) into a public module with docstrings. Keep `_training_step`/`_shared_tournament`/`_begin_phase` where they are. Update `tests/test_prediction.py:244` to import from the public path. `_resolve_loss_fn` becomes public (`losses.resolve_loss_fn`) — it is already part of the extension contract.
2. **Injectable optimizer** — `_build_optimizer` accepts `str | Callable[[float], optax.GradientTransformation] | optax.GradientTransformation`. Names and factories are wrapped in `optax.inject_hyperparams` so per-phase LR stays injectable; a raw instance is returned as-is, and the `_begin_phase` guard raises if a raw instance is given a per-phase `lr` change without `reset_optimiser_state=True`.
3. **`penalty_fn` hook** — a config field / callable param defaulting to the current `bound_penalty` closure. Lets users add weight decay, monotonicity, or custom regularisers. Fixes the `_bounded_leaves` shape-coupling for free.
4. **Evosax algorithm registry** — `register_algorithm` + `_SUPPORTED_ALGORITHMS` widened. Ship CMA-ES (default) + at least one more (e.g. `Sep_CMA_ES` or `SimpleES`). Serialisation story already exists to copy from `register_solver`.
5. **Custom-loss + channel weighting** — fix the `TypeError` when a user callable loss meets `channel_idx`/`channel_weights` (losses.py:259-264): either document loudly that callables ignore channel config, or apply channel selection in the trainer before the user loss. Tests for both.
6. **Export `bound_penalty`/`box_grid`/`data_penalty_points`** at top level (or document `jaxhybridmodels.penalties` as the public path).
7. **Correctness/hygiene batch** — unique fold name for `_box_population` (evosax.py:390 collides with 373); refresh `rng.py` docstring name list (`init`/`phase_{i}` don't exist, `tournament_attempt_{i}`/`evosax_tell_{gen}` undocumented); fix `trainable.py:7-9` mechanism description; share the vmap-per-experiment core across `prediction.py`/`optax.py`/`evosax.py`; derive `__init__.py` `__all__`/`TYPE_CHECKING` from `_EXPORTS` with a drift test; delete dead `ui/_rich.py` or wire the UIs to its widgets; wire `EvosaxTrainingConfig.log_every` or delete it.

### WS3 — Fold example patterns into the library

Exit criterion: each folded helper is in `src/`, has tests, and is used by the migrated examples (proving it's not dead).

1. **`jaxhybridmodels.metrics`** — masked per-channel MSE/RMSE/MAE/R². Today 4 implementations: `_shared/_diagnostics.py:49-106`, `crystallisation/notebook.py:1002-1036`, `batch_reactor/notebook.py:1228-1252`, `hybrid_ode/notebook.py:1203-1214`. API: `compute_metrics(predictions, bp_or_dataset) -> per-channel metrics dataclass`. Plots stay out of the library.
2. **`diffrax_solve`** — a public helper that wraps `ODETerm`/`SaveAt(ts=...)`/`stepsize_controller()`/`max_steps`/`adjoint`/`jnp.asarray(sol.ys)`. Name is chosen for greppability: `diffrax_solve`. Lives beside `prediction.py` or on `SolverConfig` as a method. Keeps `simulate_fn` user-written — only the invocation is folded. Used by all migrated examples; kills ~18 boilerplate copies.
3. **`describe_buckets(dataset) -> str`** in `data.py` — ~8 sites, two already named `describe_buckets`.
4. **`count_trainable_params(predictors, mask)`** next to `trainable.py` — 2 identical sites.
5. **`evaluate(predictor, covariates) -> float`** — predictor→float readout, ~6 sites.
6. **`default_trainable(predictors, freeze=(...))`** — the `trainable_mask` + `freeze_modules_of_type(..., BoundScaler)` one-liner (~12 sites).
7. Deferred (not folded): plotting, synthetic-data generators, symlog warp, batch-reactor bespoke plots. Move `_shared` to an importable `jaxhybridmodels.examples` extra (or documented sys.path pattern) so notebooks/scripts can share without the hack.

### WS4 — Docs rewrite (reframe + technical-writing pass)

Exit criterion: VitePress site builds clean; landing reframed; every stale/wrong doc fixed; API pages regenerated.

1. **Reframe the landing** around the actual selling points: (a) regular & irregular data via bucketing, (b) embedding neural networks inside an ODE solver, (c) correct jit/vmap/autodiff, (d) extensibility/modularity where relevant. Tone per the equinox/diffrax/pysr research: "in a nutshell" landing, anti-magic rhetoric, "switch things out in the obvious way", stated expectations, "you own the physics".
2. **Structure** — adopt the Kidger template: Landing → Getting started → Usage (data & bucketing, writing a simulate_fn, predictors & state_to_output, custom losses, custom training loops, extending jaxhybridmodels) → Examples → API (Basic/Advanced split) → FAQ/Tricks/Citation. Add an **"Extending jaxhybridmodels"** page (the extension-points bullet page; "It's completely possible to hack your own training setup").
3. **Fix the stale/wrong content**:
   - `docs/guide/training.md:101-111` — tournament keeps the **best** candidate, not the first survivor (code + tests pin best-scoring-wins).
   - RNG fold names — document the real set (`tournament_attempt_{i}`, `evosax_tell_{gen}`), drop `init`/`phase_{i}`.
   - SPEC R-T4/R-T2 "mandatory" prose vs actual `length_schedule=(1.0,)` default + required `reset_optimiser_state`.
   - SPEC §3 stale example files (`train_evosax_kinetic.py`, `pendulum/train.py`), §5.x config sketches (penalty fields, adjoint, BoundScaler extras).
   - SPEC "ten pages" → 12 in `gen_api_docs.py` `render_index()`.
   - `docs/README.md` stale layout.
   - `with_zero_final_head`, `NeuralNPolynomial` (built-but-deferred) reconciliation.
4. **Technical-writing pass** — run the `technical-writing` skill over the guide; short sentences, explicit definitions, step-by-step structure. Cite CONTEXT.md terms consistently.

### WS5 — Examples → scripts + docs (marimo removal)

Exit criterion: no `marimo` anywhere (deps, HTML embeds, skills, ruff/ty exemptions, `__marimo__/`); all 4 notebooks converted to scripts; docs embed them; CI runs them as smoke tests.

1. **Convert the 4 marimo notebooks** (`examples/{crystallisation,hybrid_ode,batch_reactor}/notebook.py`, `examples/batch_reactor/notebook_rl.py`) to plain scripts: strip `import marimo`/`app = marimo.App`/`@app.cell`/`mo.md`, unwrap cell functions to module scope, add `main()` + explicit prints/`plt.show()`, relocate prose to docstrings/docs. Keep content.
2. **Delete marimo**: `pyproject.toml` dep, `[tool.ruff.lint.per-file-ignores]` notebook entries, `ty.toml` exclusion, `docs/public/notebooks/*.html`, the 4 iframe `.md` pages, orphaned `examples/batch_reactor/notebook.html`, `examples/**/__marimo__/`, `.agents/skills/marimo-notebook/` + `wasm-compatibility/` (and their symlinks/skills-lock entries), update `docs/.vitepress/config.ts` nav/sidebar, `docs/index.md` hero, cross-referencing links (getting-started:293, recommendations:310, training:226, hybrid-ode:9, batch-reactor-rl:11).
3. **Single-source examples** — keep real files in `examples/`, embed them in VitePress pages (code blocks + static figures + stated expectations), run in CI as smoke tests. `_shared` becomes an importable extra (see WS3.7).
4. Regenerate `uv.lock`; verify `uv sync` clean.

### WS6 — New examples + external tools

Exit criterion: each new example runs end-to-end and is documented.

1. **Custom training loop example** (code-first, flagship for WS2) — a user writes their own loop against `jaxhybridmodels.training.kernels`: custom accumulation, custom schedule, per-bucket weighting, custom regulariser via the penalty hook. Demonstrates "you should be able to hack".
2. **SBML→JAX example** (`jaxkineticmodel`) — load an SBML-derived kinetic model in JAX/Diffrax, embed a hybrid correction (neural term on the mechanistic rates), train. Optionally compare trajectories against a RoadRunner reference (non-differentiable). Data: published (Van Heerden glucose-pulse) or synthetic. Sits with the other examples, not flagship.
3. Existing four examples (RL, hybrid_ode, crystallization, batch reactor) remain canonical after WS5 conversion.

### WS7 — Release readiness

Exit criterion: spec and code reconciled, test gaps closed, README updated, package publishes to PyPI.

1. **SPEC reconciliation** — reconcile `NeuralNPolynomial` built-but-deferred; update §3 layout; config sketches; R-T2/R-T4 defaults; record the WS2/WS3/WS5 decisions that are hard to reverse in SPEC (state_to_output off the Dataset is R-D7; optimizer-injection + kernels-public are softer, document in CONTEXT).
2. **Test-gap closure** — `trainable=` end-to-end through a trainer; custom-loss callable through a trainer; dict/tuple-shaped predictor pytrees through `train_with_*`/`predict_*`; standalone custom-`Predictor` serialisation round-trip in pytest (not just the example).
3. **README** — update the docs/commands sections to reflect the marimo removal and new structure.
4. **Version + publish** — `jaxhybridmodels` is available on PyPI; set version, build, publish.
5. Optional: add a validation-dataset / eval-callback pathway (the `TrainingUI.on_step_end` gets loss only; `split_dataset` produces val splits nobody can use inside a run). Lower priority, additive.

## Suggested order

WS1 (correctness gate) → WS2 (extensibility core) → WS3 (fold patterns) → WS5 (examples/scripts) → WS4 (docs rewrite, last since it documents the new API) → WS6 (new examples) → WS7 (release). WS1 and WS2 can partially overlap (WS2's hygiene batch touches the same files as WS1's checklist).
