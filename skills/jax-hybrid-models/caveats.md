# Caveats and traps

## Contents

- JAX tracing traps
- Numerical traps (divisions, x64, tolerances)
- Stiffness and initialisation traps
- Data traps
- Training-config traps
- Serialisation traps
- evosax traps
- Compilation and JIT traps
- RNG discipline

## JAX tracing traps

- **`TracerBoolConversionError`**: no Python `if`/`else` on array values inside `simulate_fn` or the vector field. Use `jnp.where`, `jax.lax.cond`, or a profile factory.
- **`NonConcreteBooleanIndexError`**: no boolean indexing that changes shape under a trace. Use masked arithmetic (`jnp.where`) with fixed shapes.
- The user callbacks run inside JAX transformations: they must return fixed-shape arrays and must not depend on Python side effects.
- A predictor with no inputs is refused at construction (`input_keys` ≥ 1) — it would have no training signal. Feed a dummy key you ignore.

## Numerical traps

- **Guarded division guards the divisor, not the result.** `jnp.where(d > eps, num / d, 0.0)` still evaluates `num / d` on the discarded branch → `inf`/`nan` gradient flows back through the `where`. Write `safe = jnp.where(d > eps, d, 1.0); jnp.where(d > eps, num / safe, 0.0)`.
- Same for `log`: clip the argument, not the result.
- **x64**: stiff/mass-balance problems drift in float32 (e.g. `mu0` going negative). Set `jax.config.update("jax_enable_x64", True)` **before anything imports JAX**. The oscillator example doesn't need it; crystallisation does.
- **Per-state `atol`** when the state spans decades: pass a tuple of per-component `atol` (one entry per state component) via `solver.stepsize_controller()`, not a hand-built `PIDController` (diffrax needs an array, not a tuple).
- Gradient-death table (float32, latent where the squash's derivative underflows to 0): sigmoid `z=16.8`, algebraic `z~3e3`, softsign `z~1.1e7`. Use `transform="algebraic"` when a learned term is expected to work near its bounds; `softsign`'s kink sits at the box midpoint and solver steps straddle it.
- `to_latent` uses a linear continuation outside the box (never a hard clip) — a hard clip zeroes the derivative exactly where inputs stray, silently dropping sensitivity from the ODE adjoint.

## Stiffness and initialisation traps

- **A box whose midpoint is physically absurd makes the ODE intractably stiff at init.** Fresh random predictors are centred roughly around latent zero, which maps to the box midpoint, but their readout can still move away from it. Recorded failure: log nucleation rate boxed `(0, 15)` → midpoint `J ≈ 3e7` (~17,000x too large) → `max_steps exceeded` everywhere. Centre the box at your hand-guessed order of magnitude + a few decades of slack.
- **`RuntimeWarning: max_steps exceeded` on some seeds but not others** → unlucky final-layer draw pushing the initial output off midpoint. Fix with `inner.with_zero_final_head()` (initial output = exact midpoint; hidden layers keep random init). Or switch `Tsit5` → `Kvaerno3` for genuinely stiff problems.
- Leave `dt0=None`; let the step controller pick the first step.
- **Adjoint by memory, not habit**: default `DirectAdjoint` stores the whole forward tape (cheapest gradients, most memory). With a network inside the vector field, `diffrax.RecursiveCheckpointAdjoint()` is usually the right swap.
- Input boxes: go slightly wider than the observed data span (e.g. `(13, 27)` for data in 14–26) — the squash saturates at edges, and margin keeps gradients well conditioned.

## Data traps

- **Every experiment must define every output channel.** A sensor being offline is not pad-with-zeros; drop the experiment. `make_dataset` raises naming the offending `exp_id`.
- **Time units must match end to end**: `ChannelObs.ts`, the `ts` passed to `simulate_fn`, and the vector field's rate constants share one unit. If you convert (e.g. minutes → seconds inside the vector field), say so loudly in a comment.
- `ts` must be finite and strictly increasing (the union axis is sorted for you, but the ODE grid needs a proper order), with ≥ 2 entries. `y0` must be `[S]`, `simulate_fn` returns `[T, S]`, `state_to_output` `[T, D]`.
- Covariates are constant-in-time by design. Time-varying values come from profile factories evaluated in the vector field, never from the data layer.
- A covariate key must have one shape across the dataset.

## Training-config traps

- Phase fields are **tuples, no scalar broadcast** — `OptaxTrainingConfig(steps=1000, ...)` is an error; spell out `(1000,)`.
- Changing `lr` across phases works for name/factory optimisers (wrapped in `inject_hyperparams`); a raw transformation instance needs `reset_optimiser_state=True` on the phase that changes `lr`.
- A phase that changes `length_schedule` changes what the loss measures → `restore_best` and `patience` reset at every phase boundary. Without the reset, "best" lands in the shortest-horizon phase (least-trained model).
- **Bound penalty is excluded from "best"**: `losses_history`, `restore_best`, early stopping, and tournament score track data + trajectory penalty only. The bound penalty is reported via UI `on_step_end(penalty=...)`. (evosax is the exception — one scalar, folds it in.)
- `penalty_weight` is relative to the per-bucket-averaged data term — the same weight means the same thing whatever the dataset size.
- A leaf with no point set while the bound penalty is enabled is a hard error (`validate_penalty_points`). Embedded predictors need explicit penalty points (`box_grid` or `data_penalty_points`); a silent no-penalty is the failure this penalty exists to prevent.
- Tournament is enabled only when `tournament_steps > 0 AND tournament_attempts > 1`. It retries diffrax failures/non-finite scores only — shape and user-code errors surface directly.

## Serialisation traps

- `load_run`/`load_predictors` need a **template** with identical container shape, module types, and static fields (bounds, warps, activation names, widths, depth). Rebuild the skeleton from the original construction code.
- Configs holding executable callables come back as metadata markers — rebind `trajectory_penalty_fn` and friends explicitly.
- Custom solvers/adjoints must be registered (`register_solver`, `register_adjoint`) before a `SolverConfig` that references them round-trips.
- All static fields must be JSON-encodable primitives/tuples; all dynamic leaves JAX arrays. No closures in non-static positions — this is what makes every predictor serialisable.
- `simulate_fn`, `state_to_output`, Dataset, and the trainable mask are never stored — they are code/data you own.

## evosax traps

- **No per-individual error handling**: a single diffrax failure or non-finite fitness crashes the generation. Mitigate with conservative `sigma_init` and `init_box_extent`, not code.
- Bounds are not enforced during search; the `BoundedPredictor` output squash keeps outputs in range.
- Not optimised for NN-sized search (~4–10 dims is the target). Use Optax for big predictors.
- Init modes: `"warm"` (start at current params — the default), `"uniform_box"`/`"lhs_box"` (latent ± `init_box_extent`).

## Compilation and JIT traps

- One compiled kernel per **bucket shape**; the Python bucket loop is the dispatch driver and must stay outside your own JIT boundaries.
- Don't mutate static predictor fields, solver settings, or callback identities mid-run — each change is a recompile.
- Prediction (`predict_bucket`) is jitted separately from training: forward-only, no backward pass.

## RNG discipline

- `key=` is **keyword-only and required** on both trainers; there is no default, ever. Passing positionally raises `TypeError` (deliberate).
- Internal randomness uses **named folds** of your root key (`hm.fold(root, "name")`), never chained splits — inserting or reordering consumers never shifts other keys.
- A run is a deterministic function of its root key (plus fixed bucket visit order — no shuffling).
- Tournament re-init: identical-shape sibling predictors get different re-init weights (per-leaf split of the attempt key).
