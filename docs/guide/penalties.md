# Penalties and bounds

A bounded output cannot leave its box: the scaler maps the box onto an
unbounded latent, so no latent produces an out-of-box value. The cost is
gradient. Near the edges the squash flattens, and in float32 the sigmoid's
derivative underflows to exactly zero once the latent is large enough. A
predictor pinned against a bound then gets no signal back.

Penalties supply that signal. They charge a predictor for saturating its
output squash, reading the latent rather than the physical value, so the
charge stays alive where the squash gradient is dead.

| | Bound penalty | Trajectory penalty |
|---|---|---|
| Question | Is the predictor saturated at these input points? | Did this solve drive it into saturation? |
| Evaluated at | Measured points plus any you add | The trajectories of each solve |
| Your code | Nothing extra | Cooperation from `simulate_fn` for embedded predictors |
| Switch | `penalty_weight > 0` | `trajectory_penalty_fn` and `trajectory_penalty_weight` |

The bound penalty covers `BoundedPredictor` leaves only. A `Parameter` is
never charged, so a bounded `Parameter` that drifts to an edge stays there
without warning. Check its `.value` after training if that matters.

## The bound penalty

The bound penalty is the mean output saturation of each `BoundedPredictor`
leaf over a set of input points. It needs no simulation. It is added once
per training step to the bucket-averaged data gradient, so a given
`penalty_weight` means the same thing at any dataset size or point count.

```python
config = hm.OptaxTrainingConfig(steps=400, lr=1e-2, penalty_weight=1e-3)
```

With phases, set `penalty_weight` on each `TrainingPhase`.

### Measured points

By default the points are the inputs the loss sees: for each observed cell,
the leaf's `input_keys` looked up in the experiment's covariates. When a
phase's `horizon` shortens the scored prefix, the penalty uses the same
prefix.

### Penalty-only points

`penalty_points` adds points where no measurement exists, typically a
region the process will run at but the data does not cover. Give one
`[G, n_inputs]` array per `BoundedPredictor` leaf, in physical units, with
columns in `input_keys` order. The penalty uses measured and added points
together.

```python
deploy = jnp.asarray([[20.0, 1.5], [20.0, 2.0], [25.0, 1.5], [25.0, 2.0]])

config = hm.OptaxTrainingConfig(
    steps=400,
    lr=1e-2,
    penalty_weight=1e-4,
    penalty_points=(deploy, deploy),   # growth, then nucleation
)
```

Leaves are taken in pytree traversal order. In a dict that means sorted
keys, so `{"nucleation": ..., "growth": ...}` still puts `growth` first.

`box_grid` builds a sweep of a whole input box. It samples evenly in warped
coordinates, so under `warp="log10"` five points across `(1e-6, 1e2)` land
at `1e-6, 1e-4, 1e-2, 1, 1e2`:

```python
from jaxhybridmodels.penalties import box_grid

sweep = box_grid(rate.in_scaler, n_per_dim=7)   # [49, 2] for a 2-D box
config = hm.OptaxTrainingConfig(steps=400, penalty_weight=1e-3, penalty_points=(sweep,))
```

It needs finite input bounds.

### Every leaf needs points

With `penalty_weight > 0`, a leaf with neither measured nor added points
raises rather than training silently without its penalty:

```text
ValueError: penalty: leaf 1 (input_keys=('temperature_C', 'supersaturation')) has
no penalty points: its inputs do not all resolve to dataset covariates and no
entry was given in penalty_points for it. ...
```

This happens to **embedded** predictors, whose inputs come from the ODE
state rather than covariates. Give them `penalty_points` or use the
trajectory penalty.

### Replacing it

`penalty_fn` replaces the bound penalty with any `(predictors, points) ->
scalar`, charged the same way. Coverage is not checked for a custom
function, and one that has no use for `points` ignores them:

```python
def weight_decay(predictors, points):
    return jnp.sum(predictors["rate"].inner.mlp.layers[-1].weight ** 2)

config = hm.OptaxTrainingConfig(steps=400, penalty_fn=weight_decay, penalty_weight=0.1)
```

## The trajectory penalty

The trajectory penalty charges saturation along the solves themselves. How
you wire it depends on where the predictor sits.

- **Embedded** predictors run inside the vector field, where only your code
  sees them. Carry the penalty as extra ODE state whose derivative is the
  per-call saturation; the solver integrates it and
  `trajectory_penalty_fn` charges the integral. The helpers are
  `attach_penalty_state`, `penalty_vector_field`, `strip_penalty_state`
  and `penalty_integral` in `jaxhybridmodels.penalties`;
  [Extending](/guide/extending#trajectory-aware-penalties) shows them
  wired together.
- **Parallel** predictors produce an output channel directly, so no state
  change is needed: `trajectory_saturation_penalty(state, out_scaler)`
  charges saturation of that channel over time.

`trajectory_penalty_fn` receives the full state `[N, T, S]` and the
`BucketPayload`, and is added to the data loss in the same forward pass.

**Probe conditions** are experiments with no measurements: each channel
has its `ts` but `values=jnp.array([])`. The data loss on them is zero, but
they still simulate, so the trajectory penalty steers the fit at
conditions you have no data for.

## Which to use

Use the bound penalty to keep the predictor unsaturated over a known input
region. Use the trajectory penalty when the risk is in the dynamics: a
state-derived input wandering out of range, or a probe condition
saturating along its path. They can be combined.
