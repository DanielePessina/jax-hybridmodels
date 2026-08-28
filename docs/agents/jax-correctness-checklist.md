# JAX correctness review checklist

Distilled from the JAX scalability book (`https://jax-ml.github.io/scaling-book/`) research and this
codebase's own invariants. Use it to review every diff that touches compiled code, gradient machinery,
data shaping, or extension points.

## The mental model

`jax.jit` traces your function to StableHLO, then HLO, then machine code. Compilation cost and
cache-keying live at that boundary. Everything that changes *shape, dtype, or static value* across calls
forces a fresh trace → fresh compile. Python-layer logic (loops, dispatch, error handling) belongs
*outside* the trace.

## Checklist

**Boundaries and caches**
- [ ] Python-layer loops (bucket dispatch, tournament) stay outside the jit boundary. `predict_dataset`
      loops in Python; `predict_bucket` is the compiled kernel.
- [ ] The jit cache is keyed deliberately by shape/dtype/static value: one kernel per bucket shape,
      `SolverConfig` fully static, `length_mask_fraction` traced. Every value's static-vs-traced side is
      a deliberate, commented choice.
- [ ] No host object is passed into a jitted function unless it is *meant* to be a static key
      (`_BestTracker` warns against exactly this — it would retrace per instance).
- [ ] Training and prediction do not share a jit cache (R-J1, pinned by a test).

**Vmap and ragged data**
- [ ] `jax.vmap` `in_axes` are explicit; cross-experiment state (predictors, solver) is never on a mapped
      axis. A mapped parameter axis silently zeroes gradients.
- [ ] Irregular/ragged data is handled by **mask, never branch**: buckets + `mask` + `jnp.where`, with
      safe denominators (`jnp.maximum(count, 1)`).
- [ ] No data-dependent Python conditionals inside traced code; no per-element branching in vmapped code.
- [ ] No materialised cross-product intermediate (e.g. an `[S, D, F]` routing tensor).

**Autodiff**
- [ ] `filter_value_and_grad`/`filter_grad` are used with an explicit `filter_spec` (never grads everything).
- [ ] The backward pass is tested, not assumed — including grad-through-the-solve consistency across
      adjoint strategies (`tests/test_adjoint_consistency.py`).
- [ ] Adjoint choice is surfaced to the user (`SolverConfig.adjoint`), with the recompute-vs-memory
      tradeoff documented (Direct / RecursiveCheckpoint / Backsolve).
  - Caveat: the field is live only where a `simulate_fn` forwards it. Most shipped examples still
    hardcode `diffrax.DirectAdjoint()` and ignore `solver.adjoint`; `examples/hybrid_ode/` and
    `tests/test_adjoint_consistency.py` forward it. `diffrax_solve` (WS3) is the fix that makes the
    field live everywhere.
- [ ] NaN/Inf handling: non-finite loss is caught at the *Python layer* (the tournament), never wrapped
      in `lax.cond` inside the trace. Mask-safe denominators everywhere a loss divides by a count.

**Dtypes and coercion**
- [ ] Dtype resolution happens on the host at data-construction time (`np.result_type`), never inside a
      traced function.
- [ ] `jnp.asarray` appears inside traced code only to pin a dtype or convert a Python tuple/float
      (e.g. `stepsize_controller()`'s tuple-atol coercion) — never as an arbitrary hot-path cast.

**Memory**
- [ ] Optimiser state is built once and reused across steps, rebuilt only at an explicit reset/phase
      boundary (no per-step `optimizer.init`).
- [ ] Per-bucket-independent terms (the bound penalty) are evaluated once per step, outside the bucket
      loop, not recomputed per bucket.
- [ ] Gradients accumulate at the tree level (`jax.tree.map(jnp.add, ...)`) over pre-partitioned leaves,
      so the shape is invariant to tree structure.
- [ ] `block_until_ready` is placed at the right boundaries (e.g. after warmup compile), so timing is
      real, and host syncs are batched rather than interleaved with Python bookkeeping.

**Extensibility without breaking jit/vmap**
- [ ] User-extensible components (solvers, adjoints, transforms, warps) are registered via **host-side
      name tables**, resolved at serialisation time, never at trace time.
- [ ] New components are added without touching or re-tracing any existing kernel.
- [ ] A custom `simulate_fn` composes because it is closed over the compiled region and is tracible
      (pure JAX/diffrax). Backsolve has a known constraint: it cannot differentiate through values
      closed over in the vector field — it needs predictors threaded through `args` (see
      `tests/test_adjoint_consistency.py`).

**Review habit**
- [ ] Dump the compiled HLO and look for unexpected `copy`/`retile`/`transpose` ops:
      `uv run python scripts/smoke_hlo.py`. Dtype `convert`/`bitcast-convert` are expected inside
      diffrax's adaptive controller and are not a smell.
