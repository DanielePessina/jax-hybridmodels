# Bound penalties are charged at measured points; the box sweep is user-supplied extras

The penalty that discourages a `BoundedPredictor` from saturating its output squash is evaluated at *points*, never on a framework-built grid: the **measured points** — the input vectors the loss actually sees at observed cells, gathered from the dataset by `data_penalty_points`, following the same length-mask prefix as the loss per phase — plus any **user-supplied penalty-only points** (`penalty_points`, positional per leaf, no measurements needed). `box_grid(in_scaler, n_per_dim)` builds a deterministic warp-uniform sweep of an input box for the "police the whole box" recipe, as user extras. The penalty is a mean over its points, charged once per step onto the per-bucket-averaged data gradient.

Nothing about the simulation path changes. `simulate_fn` still returns `[T, S]`, `BoundedPredictor.__call__` still returns a bare `Array`, `predict_bucket` still returns `[N, T, D]`, and losses are still `loss(pred_obs, bp) -> scalar`.

## Why this is non-obvious

The obvious way to implement a penalty produced by a component deep inside a computation is to have that component *return* it, as `(output, penalty)` pairs threaded outward, merged at each level, consumed by `jax.value_and_grad(..., has_aux=True)`. That is what the source package did (`call_with_penalty`), and it is the first design anyone reaches for.

It is the wrong default here, for a reason specific to this framework. `predictors` is *any pytree*, nested arbitrarily (ADR-0006), and `simulate_fn` is *user-written* with a mandatory signature (ADR-0005). Threading an aux value means every call site in every user vector field must cooperate: unpack a pair, merge a child penalty into its own, return a pair upward. The calling convention would depend on whether anything downstream happened to emit a penalty. For a framework whose selling point is that predictors compose freely, making composition conditional on a side-channel is a real tax, and one paid by every user whether or not they use penalties.

The observation that dissolves the problem is that saturation does not depend on the trajectory. It is a property of the predictor as a function of its inputs, whatever any particular solve does. There is nothing to thread outward, because there is nothing produced inside. The top of the loss already holds the whole `predictors` pytree, so the penalty can simply be evaluated there.

That buys three properties the aux-threading design cannot:

- Zero signature changes. No user code changes, and no existing run changes meaning (the weight defaults to zero; a test pins that an off penalty is bit-for-bit inert).
- Nesting invariance. A predictor five levels deep inside a dict inside a NamedTuple is found the same as a top-level one.

  Two kinds of nesting matter here, and the first implementation only handled one. Walking the pytree with an `is_leaf` predicate that matches `BoundedPredictor`, the way `reinitialize_pytree_with_key` and `freeze_modules_of_type` do, stops *at* the first match. That covers nesting in containers but silently skips a `BoundedPredictor` sitting in another's `inner` field, so the inner box declared a range that was never penalised. `_bounded_leaves` recurses into each match instead. `BoundedPredictor` subclasses `Predictor`, so predictor-in-predictor nesting is a supported shape and the walk has to keep going.
- Call-site blindness. Equally correct whether the predictor is evaluated inside a vector field at every solver stage or hoisted above the solve, a distinction both existing examples make differently.

## Why measured points, not a box grid

This ADR's first version made the framework sweep each predictor's declared input box (`collocation_grids`). The grilling that produced this revision overturned that choice on two grounds:

1. **The points the penalty polices should be the points the model actually serves.** The loss is charged at the observed cells; the penalty should be charged at the same cells' inputs, or the two objectives police different regions. A box sweep charges regions no trajectory visits — useful as an extrapolation net, but it is the *user's* call where that net matters, not a framework default over the whole box.
2. **A grid is the wrong unit of configuration.** Point density, sampling convention (linear vs log-uniform under a warp), and placement are all user decisions that the framework should not make silently. The warp case is the sharp one: a physical-space `linspace` under `warp="log10"` parks ~4/5 of its points in the top decade, so low-decade saturation is never charged. A warp-uniform sweep fixes that, but only the user knows where their box actually needs policing.

So the framework's default is the measured points, and the box sweep survives as the collocation-as-extension recipe: `box_grid(in_scaler, n_per_dim)` returns a deterministic tensor-product sweep, uniform in *warped* coordinates (identical to a plain physical linspace sweep for the linear warp), which the user passes as `penalty_points` when they want box-wide coverage — a deployment region, a future operating point, or the whole box.

### Why this is hard to reverse (and why we reversed it anyway)

The original collocation design was itself a reversal of a documented non-requirement (§2.2's "no bound penalty"), and this revision reverses it in turn. Both reversals were grilling outcomes, not drift. The invariants that survive from the first version are the load-bearing ones:

- **The penalty reads the latent, not the physical output.** `from_latent`'s derivative carries a factor of `sigma'(z/T)`, which decays exponentially: `4.5e-2` at `z=3`, `4.5e-4` at `z=10`, and it underflows to exactly `0.0` past roughly `|z/T| = 15`. A penalty written against the physical output inherits that factor on the backward pass, so it vanishes precisely when saturation is worst. Measured on a `(0, 10)` box at `z = 40`: `d(saturation)/dz = 74.0` while `d(physical)/dz = 0.000e+00`.

  A penalty that reads as satisfied when the predictor is dead is worse than no penalty, because nothing distinguishes healthy from dead. Hinging on `|z| / T` gives push-back linear in the overshoot that never underflows.
- **Determinism.** The measured points are fixed by the dataset and the extras are fixed by the user, so `restore_best` compares raw loss values across steps with no resampling noise. (A resampled penalty would pick a "best" that drew an easy sample.)
- **Mean aggregation, once per step, onto the averaged data gradient.** The penalty is a mean over its points — scale-invariant to dataset and point-count size — and it is charged once per step, not once per bucket, onto the per-bucket-averaged data gradient. `penalty_weight` is therefore relative to the per-bucket-averaged data term, and the same weight means the same thing whatever the dataset size. A sum-aggregated penalty would grow with dataset size while the data term stays a mean, making the weight dataset-dependent.
- **Coverage validation.** With the penalty enabled, every leaf must have at least one point source (measured, extras, or both). An embedded predictor — whose inputs are state-derived, so the dataset can resolve no measured points for it — must be covered by extras or the trajectory penalty (ADR-0009); a silent no-penalty is the failure mode this penalty exists to prevent, and the error message says so.

### What this deliberately does not cover

Whether a *particular solve* pushed an input out of range. That question is genuinely trajectory-dependent and needs the penalty computed where the state actually goes.

Answering it lands via the opt-in trajectory-aware penalty (ADR-0009): the penalty rides in extra ODE state components whose time-integral is charged by `trajectory_penalty_fn`, without widening `simulate_fn`'s signature. `simulate_fn` still returns the full state; the accumulated penalty components are stripped in `state_to_output`.

Note also that carrying a penalty as an extra integrated ODE state does not by itself solve this. It gets the quantity to the end of the solve, but `state_to_output` then projects to `[T, D]` and the loss sees only that, so the penalty is dropped. Accumulating it and extracting it are separate problems, and only the second is a contract change.

## Considered alternatives

- `(output, aux)` pairs threaded through `simulate_fn`. Rejected as the *default* for the viral-signature reason above. Retained as the intended escape hatch for trajectory-dependent penalties, opt-in.
- Penalty as an extra ODE state. Correct for accumulating a time-integral inside a solve, and the only correct way to do that (a vector field is called at RK stage points an unpredictable number of times, including on rejected steps, so ad-hoc accumulation across calls is meaningless). But it does not reach the loss on its own, and it is only needed when the predictor is genuinely state-dependent. Documented as a user-side recipe rather than framework machinery.
- `eqx.nn.State`. Rejected outright. It is functionally-threaded rather than mutable state, so under `grad` it is just another output and buys nothing over aux pairs; and it does not work inside a diffrax solve at all. It exists for BatchNorm-style statistics whose update is *not* differentiated, which is the opposite of a penalty's purpose.
- Random sampling instead of fixed points. Rejected because `restore_best` compares raw loss values across steps, and a resampled penalty would make that comparison noisy enough to pick a "best" that merely drew an easy sample.
- Log barrier. Rejected. It distorts the interior by `O(mu)`, requires annealing, and is undefined at infeasible points, so one bad sample under `vmap` gives `nan` for the whole batch with no per-sample rejection available.
- Charging the penalty inside `BoundedPredictor.__call__`. Rejected because it makes penalty emission a side effect of evaluation. `input_violation` and `saturation` are pure queries instead, so the caller decides whether and where to pay for them.
- A framework-built box sweep as the default (this ADR's first version). Reversed: sampling convention and placement are user decisions; the framework's default is the measured points, and the sweep survives as the `box_grid` recipe for user extras.