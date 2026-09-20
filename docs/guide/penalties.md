# Penalties and bounds

Bounds keep a predictor's outputs inside a declared physical box. The box is
enforced by *reparameterisation*, not by clipping: the scaler maps the box onto
an unbounded latent space, so an out-of-box value has no latent that represents
it. Feasibility comes for free.

That leaves one problem. The gradient of the map back to physical units dies
near the edges — the squash derivative underflows to exactly zero past
`|z / T| = 15` — so a predictor pinned against a bound has nothing left to
pull it back. Penalties are the restoring force. They charge the predictor for
saturating its output squash, and the charge is read from the latent, never
the physical value, so it never dies where saturation is worst.

Two penalty families cover two different questions:

| | Bound penalty | Trajectory penalty |
|---|---|---|
| Answers | Is the predictor saturating at the points we evaluate? | Did a particular solve push the model into saturation? |
| Evaluated at | Fixed input points (measured + user-supplied) | The actual trajectories of a solve |
| Needs | Nothing from your physics code | Cooperation from your `simulate_fn` (embedded case) |
| Default | On when `penalty_weight > 0` | Opt-in via `trajectory_penalty_fn` |

## The bound penalty

The bound penalty charges each `BoundedPredictor` leaf for saturating its
output squash. It is a property of the predictor as a function: evaluating it
needs no simulation, so it works the same whether the predictor sits inside a
vector field or above one.

The penalty is a **mean** of saturation over a set of input points. The mean
keeps one `penalty_weight` meaningful at any dataset size: doubling the number
of points does not change the size of the charge. The charge is added **once
per training step** to the per-bucket-averaged data gradient, so the weight is
relative to the averaged data term. A weight tuned on a small dataset means
the same thing on a large one.

### Where the points come from

The default point set is the **measured points**: the input vectors the loss
actually sees at observed cells, gathered from the dataset by
`data_penalty_points`. Each leaf resolves its `input_keys` against the dataset
covariates and collects one input vector per observed cell. When a phase's
`length_schedule` masks the loss to a prefix of each trajectory, the penalty
follows the same prefix, so both objectives always look at the same points.

For a leaf whose inputs are covariates, this needs nothing from you:

```python
config = hm.OptaxTrainingConfig(
    steps=(400,),
    lr=(1e-2,),
    optimizer=("adamw",),
    reset_optimiser_state=(False,),
    loss="mse",
    penalty_weight=(1e-3,),   # length-1 tuple broadcasts across phases
)
```

### Penalty-only points

You can add **penalty-only points**: input vectors where saturation is charged
even though no measurement exists there. Pass them with `penalty_points`, one
`[G, n_inputs]` array per `BoundedPredictor` leaf, in traversal order, in
physical units matching the leaf's `input_keys`.

A deployment region is the typical use. The crystallisation model's rate laws
`G(S)` and `J(S)` share the inputs `("temperature_C", "supersaturation")`; the
training data spans one supersaturation range, but the process will run at
another:

```python
growth_pts = jnp.asarray([[20.0, 1.5], [20.0, 2.0], [25.0, 1.5], [25.0, 2.0]])
nucleation_pts = jnp.asarray([[20.0, 1.5], [20.0, 2.0], [25.0, 1.5], [25.0, 2.0]])

config = hm.OptaxTrainingConfig(
    ...,
    penalty_weight=(1e-4,),
    # Leaf order: growth_bp first, nucleation_bp second.
    penalty_points=(growth_pts, nucleation_pts),
)
```

Measured points and penalty-only points are combined: the penalty evaluates at
the union.

### box_grid: the whole-box sweep

If you want to police the entire declared box — the "collocation" idea — build
the sweep yourself with `box_grid`, and pass it as penalty-only points:

```python
sweep = hm.box_grid(in_scaler, n_per_dim=7)   # [49, 2] for a 2-D box
config = hm.OptaxTrainingConfig(..., penalty_points=(sweep,))
```

`box_grid` samples uniformly in **warped** coordinates, then maps back to
physical units. For the linear warp that is a plain evenly spaced sweep. For
`warp="log10"` it is uniform in log space: five points across `(1e-6, 1e2)`
land at `1e-6, 1e-4, 1e-2, 1, 1e2` instead of piling up in the top decade.
You decide the sampling, not the framework.

### Coverage: every leaf must have points

When `penalty_weight > 0`, every `BoundedPredictor` leaf must have at least
one point source: measured points, penalty-only points, or both. Otherwise the
run raises:

```
ValueError: penalty: leaf 1 (input_keys=('temperature_C', 'supersaturation')) has
no penalty points: its inputs do not all resolve to dataset covariates and no
entry was given in penalty_points for it. Add penalty_points for this leaf, or
use trajectory_penalty_fn for embedded predictors.
```

The first clause covers **embedded** predictors — leaves whose inputs are
state-derived (supersaturation from the ODE state, not a dataset covariate).
The dataset cannot resolve measured points for them, so they need either
penalty-only points or the trajectory penalty below. A silent no-penalty would
be worse than an error: a model trained without the charge it was configured
for looks healthy and is not.

A custom `penalty_fn` replaces the bound penalty entirely. Coverage is the
bound penalty's contract, so a custom hook is not checked.

## The trajectory penalty

The bound penalty cannot answer "did the model saturate where it actually
ran". That question needs the solve, and it has two flavours depending on
where the predictor sits:

- **Embedded** predictors run inside your vector field, so only your code
  sees their calls. The penalty rides in the ODE state as accumulators whose
  derivatives are the per-call penalty rates; the solver integrates them, and
  `trajectory_penalty_fn` charges the time-integral. Helpers:
  `attach_penalty_state`, `penalty_vector_field`, `strip_penalty_state`,
  `penalty_integral`.
- **Parallel** predictors are evaluated above the solve, so no state change is
  needed: `trajectory_saturation_penalty` charges saturation of the predicted
  output channel over time.

**Probe conditions** are experiments with no measurements at all: channels
carry `values=jnp.array([])`, so the mask is all-False and the data loss is
exactly zero. Their trajectories still simulate, and only trajectory penalties
fire on them. They steer the fit toward conditions you have no data for.

## Choosing

Use the bound penalty when the concern is the predictor as a function: it
stays unsaturated over the inputs it serves, including regions you declare in
advance. Use the trajectory penalty when the concern is the dynamics: a
particular solve pushed a state-derived input out of range, or a probe
condition saturated along its trajectory. The two compose: bound points for
the declared box, trajectory rates for the actual paths.

## Custom penalty functions

The `penalty_fn` config field replaces the bound penalty with any regulariser
that takes `(predictors, points)` and returns a scalar — weight decay on a
leaf, a monotonicity term, anything the bound penalty does not express. It is
charged once per step, exactly like the default. A hook that does not need
points ignores them:

```python
def weight_decay(predictors, points):
    return predictors[0].scale**2

config = hm.OptaxTrainingConfig(..., penalty_fn=weight_decay, penalty_weight=(0.1,))
```
