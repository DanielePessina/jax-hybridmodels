# Trajectory-aware penalties ride in the ODE state; probe conditions are unobserved experiments

Training configs gain an opt-in `trajectory_penalty_fn(full_state, bp) -> scalar` hook (plus a scalar `trajectory_penalty_weight`), charged inside the training step's single forward pass. For an **embedded** hybrid model — the predictor runs inside the user's vector field — the penalty is accumulated in the ODE state as extra components whose derivatives are the per-call penalty rates; the charge is their time-integral. Probe conditions — scenarios to steer the fit toward, with no measurements — are expressed as experiments whose channels carry `values=jnp.array([])`, so the mask is all-False.

## Why this is non-obvious

ADR-0007 deliberately made the collocation penalty **trajectory-blind**: it sweeps a synthetic grid of each predictor's *declared input box* and charges output saturation there. That was the right default at the time — it needs no cooperation from `simulate_fn`, `state_to_output`, or the loss protocol, and it is an extrapolation safety net (it fires in regions no trajectory visits). But it cannot answer "did the model saturate *where it actually ran*", and for an embedded hybrid model — crystallisation's `G(S)` and `J(S)` feeding the moment equations, a shape factor learned from data — that is the question that matters. The predictor's output feeds the dynamics; whether it hugs a boundary on the trajectories the model actually simulates is a property of those trajectories, not of a box sweep.

The obvious way to collect a per-call penalty is to sum it up in Python at each call. That is wrong: a vector field is evaluated by the solver at RK stage points a data-dependent number of times, including rejected steps, so any hand-rolled accumulation is meaningless. The only sound accumulator in a solve is an extra ODE state component — the *time-integral* of the rate. This mirrors ADR-0007's own note ("carrying a penalty as an extra integrated ODE state ... is the only correct way to do that"), which previously failed only because `state_to_output` projected it away; here the recipe strips the accumulators in `state_to_output` and the new hook reads them *before* the projection.

A second, cheaper design was considered — a loss-stage penalty on `pred_obs` (per-trajectory saturation of the predictor's *output* channel) that needs no state change. It is offered as the parallel-hybrid path (`trajectory_saturation_penalty`), where the predictor's output is the measured channel. It cannot cover the embedded case, because the framework never sees the predictor's individual calls inside the user's vector field — no second forward pass helps, since a second call still does not expose them. The integrated-state recipe is the general answer; the loss-stage one is the shortcut when it applies.

## Why this is hard to reverse

It is an opt-in additive feature: with `trajectory_penalty_fn=None` (the default) the gradient kernels are byte-identical to before, and nothing in `simulate_fn` / `state_to_output` / the loss protocol changes. It reverses only ADR-0007's "what this deliberately does not cover" paragraph — which always said it would land "gated behind an explicit opt-in flag when it lands, never auto-detected by inspecting what `simulate_fn` returned". This is exactly that opt-in landing.

## Considered alternatives

- Manual accumulation in `simulate_fn`. Rejected: meaningless under adaptive error control (rejected steps, variable call counts).
- A second forward pass that "checks" the bounds. Rejected: the predictor calls live inside user code; a second pass cannot see them any better than the first, and it doubles the solve cost.
- Loss-stage penalty on `pred_obs` (no state change). Adopted for the parallel case (`trajectory_saturation_penalty`); insufficient for embedded predictors.
- Probe conditions as a synthetic-grid extension. Rejected: the grid is exactly the collocation weakness. Probing the *actual* condition (covariates, initial state, time grid) with no observations is strictly more targeted and costs nothing extra.

## Consequences

- `OptaxTrainingConfig` and `EvosaxTrainingConfig` gain `trajectory_penalty_fn` / `trajectory_penalty_weight` (validated: non-negative weight; weight > 0 requires a function).
- `build_bucket_step` and the evosax single-eval (both public) accept and honour the hook; with it unset they compile identically to the pre-hook kernels.
- `hybridmodels.penalties` gains the recipe helpers: `attach_penalty_state`, `penalty_vector_field`, `strip_penalty_state`, `penalty_integral`, `trajectory_saturation_penalty`.
- The data layer treats a channel with empty `values` as a probe: its `ts` still defines the union axis (so `simulate_fn` gets a grid) but every mask cell is False, so the data loss is exactly zero and only trajectory penalties fire.
- `trajectory_penalty_weight` is scalar, not per-phase; a per-phase schedule is a follow-up if it earns one.
