# Shared tournament only (no vmapped / serial modes)

The Optax training loop ships exactly one tournament mode, shared. Up to `tournament_attempts` candidates are tried serially, each re-initialised from a fresh RNG, each trained for `tournament_steps` warm-up steps using the same JIT-compiled `bucket_step` and `apply_update` as the main loop (so no extra compile cost), then scored on the data term with a forward-only pass. The lowest-scoring candidate is returned. On per-attempt failure (diffrax error / non-finite loss), drop and try the next RNG. If all fail, fall back to the original predictor with a `RuntimeWarning`. Enabled implicitly when `tournament_steps > 0 AND tournament_attempts > 1`.

## Why this is non-obvious

The source package shipped three modes: `vmapped` (parallel via `eqx.filter_vmap`), `serial` (sequential with its own jit), and `shared` (sequential, reusing the main jit). A future reader may assume the parallel mode is "obviously better" and try to add it back. It isn't. Vmapped tournament parallelism crashes on the first solver failure across the population, and ships a second jit cache that doubles compile time. Shared mode is the only one robust to per-attempt diffrax failures that also amortises compilation. The other two were experimental dead ends.

## Considered alternatives

- Vmapped tournament. Rejected for fragility under per-attempt failures and duplicated jit cache.
- Serial tournament with its own jitted `bucket_step`. Rejected because it duplicates the main loop's compile work.
