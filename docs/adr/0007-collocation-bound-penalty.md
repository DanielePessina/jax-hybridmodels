# Bound penalties are computed at the top level over a collocation grid

The penalty that discourages a `BoundedPredictor` from saturating its output squash is evaluated **outside** the simulator, by sweeping a deterministic grid across each predictor's declared input box and charging `BoundScaler.saturation` on the resulting latents. It is summed over every `BoundedPredictor` leaf of the `predictors` pytree, weighted, and added to the training objective.

Nothing about the simulation path changes. `simulate_fn` still returns `[T, S]`, `BoundedPredictor.__call__` still returns a bare `Array`, `predict_bucket` still returns `[N, T, D]`, and losses are still `loss(pred_obs, bp) -> scalar`.

## Why this is non-obvious

The obvious way to implement a penalty produced by a component deep inside a computation is to have that component *return* it — `(output, penalty)` pairs threaded outward, merged at each level, consumed by `jax.value_and_grad(..., has_aux=True)`. That is what the source package did (`call_with_penalty`), and it is the first design anyone reaches for.

It is the wrong default here, for a reason specific to this framework: `predictors` is *any pytree*, nested arbitrarily (ADR-0006), and `simulate_fn` is *user-written* with a mandatory signature (ADR-0005). Threading an aux value means every call site in every user vector field must cooperate — unpack a pair, merge a child penalty into its own, return a pair upward. The calling convention would depend on whether anything downstream happened to emit a penalty. For a framework whose selling point is that predictors compose freely, making composition conditional on a side-channel is a real tax, and one paid by every user whether or not they use penalties.

The observation that dissolves the problem: **saturation does not depend on the trajectory.** It is a property of the predictor as a function on the input box it declares. There is nothing to thread outward, because there is nothing produced inside. The top of the loss already holds the whole `predictors` pytree, so the penalty can simply be evaluated there.

That buys three properties the aux-threading design cannot:

- **Zero signature changes.** No user code changes, and no existing run changes meaning (the weight defaults to zero and is bit-for-bit inert; a test pins this).
- **Nesting invariance.** Leaves are found with the same `is_leaf`-stopped traversal `reinitialize_pytree_with_key` and `freeze_modules_of_type` already use, so a predictor five levels deep inside a dict inside a NamedTuple is found identically.
- **Call-site blindness.** Equally correct whether the predictor is evaluated inside a vector field at every solver stage or hoisted above the solve — a distinction both existing examples make differently.

### Why the penalty reads the latent, not the physical output

This is the load-bearing detail, and getting it backwards produces a penalty that silently does nothing.

`from_latent`'s derivative carries a factor of `sigma'(z/T)`, which decays exponentially: `4.5e-2` at `z=3`, `4.5e-4` at `z=10`, and it underflows to **exactly `0.0`** past roughly `|z/T| = 15`. A penalty written against the physical output inherits that factor on the backward pass, so it vanishes precisely when saturation is worst. Measured on a `(0, 10)` box at `z = 40`: `d(saturation)/dz = 74.0` while `d(physical)/dz = 0.000e+00`.

A penalty that reads as satisfied when the predictor is dead is worse than no penalty, because nothing distinguishes healthy from dead. Hinging on `|z| / T` gives push-back linear in the overshoot that never underflows.

### What this deliberately does not cover

Collocation is trajectory-blind. It reports saturation anywhere in the declared box, including regions the training trajectories never visited. For catching extrapolation failure before deployment that is a feature. For "did *this* solve push an input outside its range" it is the wrong instrument — that question is genuinely trajectory-dependent and needs the penalty computed where the state actually goes.

Answering it means widening `simulate_fn` to return `(states, penalty)`, which is an ADR-0005 change. Deferred (SPEC §2.3) until a case demands it, and gated behind an explicit opt-in flag when it lands — never auto-detected by inspecting what `simulate_fn` returned, which would make the contract depend on runtime shape.

Note also that carrying a penalty as an extra integrated ODE state does **not** by itself solve this. It gets the quantity to the end of the solve, but `state_to_output` then projects to `[T, D]` and the loss sees only that, so the penalty is dropped. Accumulating it and extracting it are separate problems, and only the second is a contract change.

## Considered alternatives

- **`(output, aux)` pairs threaded through `simulate_fn`** — rejected as the *default* for the viral-signature reason above. Retained as the intended escape hatch for trajectory-dependent penalties, opt-in.
- **Penalty as an extra ODE state.** Correct for accumulating a time-integral inside a solve, and the only correct way to do that (a vector field is called at RK stage points an unpredictable number of times, including on rejected steps, so ad-hoc accumulation across calls is meaningless). But it does not reach the loss on its own, and it is only needed when the predictor is genuinely state-dependent. Documented as a user-side recipe rather than framework machinery.
- **`eqx.nn.State`** — rejected outright. It is functionally-threaded state, not mutable state, so under `grad` it is just another output and buys nothing over aux pairs; and it does not work inside a diffrax solve at all. It exists for BatchNorm-style statistics whose update is *not* differentiated, which is the opposite of a penalty's purpose.
- **Random sampling instead of a fixed grid** — rejected because `restore_best` compares raw loss values across steps, and a resampled penalty would make that comparison noisy enough to pick a "best" that merely drew an easy sample.
- **Log barrier** — rejected. It distorts the interior by `O(mu)`, requires annealing, and is undefined at infeasible points, so one bad sample under `vmap` gives `nan` for the whole batch with no per-sample rejection available.
- **Charging the penalty inside `BoundedPredictor.__call__`** — rejected because it makes penalty emission a side effect of evaluation. `input_violation` and `saturation` are pure queries instead, so the caller decides whether and where to pay for them.
