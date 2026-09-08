# Performance hunt — learnings log

Working log for the "hunt for performance wins" pass (Sep 2026). Method,
measurements, and decisions are recorded here so the next optimization pass
starts from evidence instead of folklore. All timings below are from this
machine: a CPU-only MacBook Pro, JAX 0.10.0 / Equinox 0.13.7 / diffrax
latest / optax 0.2.8, single bucket of N=12 experiments, an MLP
(1→16→16→1) inside a Tsit5 solve fed by a `BoundedPredictor`. **CPU
timings will not match a GPU machine; the *shape* of the findings (one-time
recompiles, host arithmetic) transfers, and recompile costs are typically
far worse on GPU.**

Benchmark harness: `scripts/bench_hotpath.py` (production kernel
construction + steady-state per-step timing).
Session: live in tmux `perf-hunt` (`tmux attach -t perf-hunt`).

---

## 2026-09-06 — Session 1: mid-run kernel retrace + per-step host arithmetic

### Symptom

`_run_phases` (one training step = all buckets + penalty + one optimiser
update) measured **~123–150 ms/step** while the three steady-state device
kernels (`bucket_step` + `penalty_step` + `apply_update`) summed to
**~1.2 ms/step**.

### Root cause A — the first optimizer update retraces every kernel once

Per-call wall-time bisection (each kernel call timed; a JAX recompile
shows as ~1–2 s, a cache hit as ~1 ms) showed the **first real step of the
first `_run_phases` recompiled `bucket_step` (~2.3 s), `penalty_step`
(~27 ms) and `apply_update` (~84 ms)**; the second step recompiled
`apply_update` once more; everything after that ran at steady state.

Controlled 2×2 experiment (`apply_update` in between vs plain
`eqx.apply_updates`, × `jax_log_compiles` on/off):

| sequence | next `bucket_step` call |
|---|---|
| `bucket(P)` … `bucket(P)` | hit |
| `AU(P, g, os0)` → `bucket(P2)` with P2 from AU | **recompile (~2.1 s)** |
| `eqx.apply_updates(P, g)` → `bucket(P2')` (no jit in between) | hit |
| `AU(P, …)` → `bucket(P)` (original, untouched) | hit |
| `AU(P2, …)` → `bucket(P2)` | **recompile** |

The trigger was then bisected to a single structural feature:

| predictor structure | post-update `bucket_step` |
|---|---|
| 1-d array leaves only (any count, 1..128) | hit |
| a **0-d array leaf** (e.g. `BoundScaler.temperature = jnp.asarray(1.0)`) | **recompile** |
| same 0-d leaf but **frozen** (not trainable) | hit |

And the reason: **JAX's compiled cache keys 0-d array leaves by value and
weak type.** `jnp.asarray(1.0)` is weak-typed (`~float32[]` in
`jax.typeof`); an optimiser-updated leaf is strong-typed (`float32[]`).
The first update flips the scalar, every kernel that reads the predictor
misses once, re-traces and recompiles — then the strong variant is cached
and everything runs at steady state forever.

`tests/_harness.py`'s `OmegaPredictor` already documents this exact rule
("strong-type the leaf … or force a retrace") — the package authors knew,
but the default `trainable_mask` (which trains `BoundScaler.temperature`)
couples it into every default-configured run.

Notes:
- Not per-step and not value-hashing of general weights: plain `jax.jit`
  doesn't value-cache (minimal repro), and after ONE recompile every
  further (differently valued) call hits.
- Survives with a trivial `simulate_fn` (no ODE): pure jit-cache
  behaviour in the kernel stack, not an ODE artifact.
- **cProfile flips it on** (cProfile run showed 19 XLA compiles in 10
  steps). Never profile a JAX loop with cProfile across a jit boundary;
  use per-call wall time or `jax_log_compiles`.
- Residual, unexplained, accepted: **one ~80–120 ms recompile of the
  fused/apply kernel at run step 1** that no settle round absorbs (tried
  1/2/3 rounds). Everything else is stable.

### Root cause B — per-step Python tree arithmetic (~7 ms/step on CPU)

Instrumented loop per step: `zero_grads` rebuild 0.39 ms, bucket-step
1.17 ms (device!), **tree add 3.91 ms**, **tree div 3.16 ms**, penalty
0.26 ms, merge 0.08 ms, apply 0.3 ms + phantom, floats 0.49 ms. Two of
the three eager per-leaf tree walks cost more than double the ODE solve
itself; each `jax.tree.map(jnp.add, …)` dispatches one compiled kernel
per parameter leaf.

### Fixes (in `training/optax.py`) — final state

**1. Conditional settle cycle in `_warmup_compile`.** After the per-shape
warm-up, and only when the run *can* hit the weak→strong flip
(`_has_weak_scalar_trainable`: a weak-typed 0-d leaf inside the trainable
partition), the compile bracket runs: `penalty_step` on the original
predictors (variant 0), then two rounds of update → `bucket_step` on
every shape → `penalty_step` → (score, if a tournament is enabled) on
the updated tree. Strong-typed or frozen scalars (e.g.
`freeze_modules_of_type(mask, predictors, BoundScaler)`, or the harness's
`OmegaPredictor`) skip it entirely, keeping the "one trace per bucket
shape" contract; `test_length_schedule_does_not_recompile` stays green.

**2. Fused per-step tail (`_build_step_update`).** `_training_step` now
returns the raw gradient sum (seeding the accumulator from the first
bucket instead of a `zeros_like` tree), and one jitted
`step_update(predictors, acc, penalty_grads, n_buckets, opt_state)`
averages + merges the penalty grads + runs the optimiser in a single
launch, in the same op order (bit-identical gradients). `n_buckets` is a
traced argument so bootstrap ensembles re-bucket correctly. The public
`apply_update` kernel is unchanged (the tournament still uses it).

Measured on the production path (`_build_training_kernels` →
`_run_phases`, 20 steps, single bucket, MLP-in-Tsit5, CPU):

| | total | per step |
|---|---|---|
| before anything | 2454 ms | 122.7 ms |
| settle cycle only | 347–474 ms | 17.4–23.7 ms |
| settle + fused step | 181–191 ms | **9.0–9.5 ms** |

Steady state after both: ~3 ms/step (bucket 1.2 + penalty 0.2 + fused
0.35 + floats 0.5 + Python) plus the one-time phantom.

### Honest cost/benefit

- The settle **relocates** the second compile per kernel from mid-run
  into the "compiling" bracket — wall-clock neutral on CPU, but first
  steps run at steady state immediately and no progress bar appears
  hung. On GPU the relocated compile is in the 10–60 s class a user
  already expects during compilation.
- The fused step is a **pure wall-clock win**: ~7 ms/step of Python tree
  arithmetic becomes a few microseconds of extra kernel code. It scales
  with parameter count (more leaves = more per-tree-op dispatch).

### Things that did **not** help / are not the fix

- `jax_log_compiles` — diagnostic only.
- More settle rounds beyond 2 — no change (the one-time phantom is not
  settleable).
- Warming each kernel once (the old behaviour) — the post-update variant
  is a second cache entry the one-call warm never creates.
- **Persistent on-disk compilation cache** (`jax_compilation_cache_dir`):
  diffrax embeds host callbacks in every solve (`error_if` on the
  solver-ok check in `_integrate.py`, `pure_callback` in step clipping),
  and JAX refuses to persist kernels containing host callbacks
  ("Not writing persistent cache entry … because it uses host
  callbacks"). Repeat runs of the same config always recompile. Upstream
  limitation; nothing to do in this package (the error semantics are part
  of the R-T7 tournament contract).

### Implications for the rest of the library

- **Tournament / ensembles**: reuse the same kernels, so they inherit the
  settle automatically (once per kernel construction).
- **Evosax**: `population_eval = jit(vmap(single_eval))` is only ever fed
  eager arrays (`strategy.ask` / `_box_population` host-side), so no
  post-update round trip. Measured (6 gens × pop 8, MLP hour-glass,
  CPU): `single_eval` compiles exactly once; evosax's internal
  `ask`/`tell` jits each recompile once (same family, ~0.05–0.1 s each,
  not per-generation), then run at steady state. Nothing to fix in the
  library.
- **`predict_bucket` / prediction**: eager inputs only; unaffected.

## Next candidates (documented, not yet done)

- **Per-step host syncs**: `_run_phases` does `float(avg_data)` +
  `float(avg_penalty)` per step (2 device→host round trips/step).
  Measured ~0.02–0.06 ms each on CPU (negligible); on GPU they serialize
  the pipeline (~10–50 µs each) but the update is already dispatched
  before the syncs, so the accelerator is not idle. Low priority.
- **Fusing the whole step (all buckets + penalty + update) into one jit
  region** would remove n_buckets+2 kernel launches per step — deliberate
  spec/design (R-J1/R-J2, one trace per bucket shape) and not touched.
- **`BoundScaler.to_latent`/`from_latent` rebuild their edge arrays and do
  registry lookups on every call** — trace-time constants only; measured
  impact below noise inside an ODE solve. Not worth changing.
- **x64 on GPU**: the crystallisation example enables `jax_enable_x64`;
  on a GPU backend float64 runs at a large penalty (and on some backends
  is unavailable). A float32 variant is a per-example choice, not a
  framework change.
- **Weak-typed 0-d leaves are a general JAX gotcha**: any model built
  with `jnp.asarray(1.0)`-style scalars that are later trained pays the
  one-time retrace. The framework's settle absorbs it for the stock
  trainers; custom loops should either strong-type scalars at init or
  freeze them.
---

## 2026-09-06 — Session 2: root-cause pass (strong-typing), simplification, verification

Follow-up review prompted "are these robust or band-aids?" and "simplify,
remove dead code". Findings:

### The settle was a band-aid for a framework-generated problem — so the root cause got fixed

The retrace trigger (weak-typed 0-d trainable leaf) was **generated by the
framework itself**: `BoundScaler.temperature = jnp.asarray(1.0)` is weak.
`BoundScaler.__init__` now strong-types its own scalar at construction
(`jnp.asarray(t, dtype=t.dtype)`), so default models have no weak trainable
leaf at all: the 2.3 s bucket retrace, the phantom apply recompile, and the
settle itself all disappear for framework models — verified per-call (no
retrace anywhere, step 1 included). The settle remains only as a gated,
one-round safety net for *user* trees with weak scalars
(`_has_weak_scalar_trainable`), with a dedicated test.

### Simplification after the root fix

- `_warmup_compile` now simply compiles every kernel once in the compile
  bracket (bucket per shape + penalty + fused step + score if tournament);
  the settle block is one round, not two, and only on `settle=True`.
- Dead `else: apply_update(...)` branch and the `apply_update` parameter
  removed from `_warmup_compile`; `apply_update` unthreaded from
  `_run_phases` and `_begin_phase` (the loop only calls the fused kernel;
  the tournament still owns its own `apply_update`).
- `_training_step`'s defensive empty-dataset branch removed (all callers
  validate); the first-bucket-seeds-accumulator lazy init stays.

### Final numbers (single bucket, MLP-in-Tsit5, CPU)

| phase | per-step |
|---|---|
| before anything | 122.7 ms |
| first pass (settle-only) | 17.4–23.7 ms |
| root fix + warm-all + fused step | **3.1–3.3 ms** |

Kernel build + warm-up: ~3.1 s (each kernel compiled exactly once).
Bit-exactness of the fused step verified with `jnp.array_equal` on both
the post-update predictors and the optimizer state vs the old Python path.

### Whole-library verification

- Full suite: 616 passed (incl. example smoke tests and optax
  history-pinning tests).
- All 10 example trainings run end-to-end on short budgets (hybrid_ode
  MLP+KAN, pendulum, custom_predictor, custom_loop, supersaturation_poly,
  kinetic, batch_reactor, sbml, mechanistic-evosax).
- Test sweep: `test_import.py` was a pure re-assertion of `__all__`
  (hand-edited on every export change); replaced with a compact
  "core entry points resolve and are callable" smoke over the stable
  user-facing surface. Added `test_weak_zero_d_scalar_trains_via_settle_path`
  so the settle safety-net branch is exercised.
