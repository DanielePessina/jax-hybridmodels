# Batch reactor, part two: design record

Design notes for `train_rl_deactivation.py` and `notebook_rl.py`. The
walkthrough and measured results are in `docs/examples/batch-reactor-rl.md`;
this file keeps the decisions and their reasons. Reference: Mowbray, Wu,
Rogers, Del Rio-Chanona and Zhang, "A reinforcement learning-based hybrid
modeling framework for bioprocess kinetics identification", Biotechnol
Bioeng 120:154, 2023.

## Purpose

Show a `BoundedPredictor` trained as a reinforcement learning policy that
emits a bounded kinetic parameter at each interval, with no gradient taken
through the ODE solve.

- **Bounds by reparameterisation.** The policy distribution lives in the
  latent space and `BoundScaler` maps the latent into the physical box as
  part of the dynamics. No `tanh` squash, no log-det-Jacobian correction,
  no clip, and `BoundScaler.saturation` becomes a reward term.
- **No adjoint.** PPO differentiates only the policy's log-probability and
  the value head. `SolverConfig.adjoint` is irrelevant, and the solver
  could be stiff or nonsmooth. This is the honest reason to use RL here.

Not a reproduction of the paper's numbers, not a claim that RL beats
gradient descent, and not a library feature: nothing in
`src/jaxhybridmodels` changes.

## Decisions

| Decision | Choice |
|---|---|
| Rollout | Free-running on the model's own state, as in the paper's Eq 4c and 4e. |
| RL stack | `rlax` for GAE and the clipped surrogate; actor and critic written here in equinox. |
| Algorithm | PPO. The MDP is deterministic, the horizon is 11 and rollouts are cheap, so SAC's replay buys little for about twice the code. |
| Observation | `(Ca, temperature_C, pH)`. |
| Action | Catalyst activity, multiplying the frozen hybrid rate from part one. |
| Data | The same 9 LHS points and 2 validation points as part one, aged catalyst, independent noise. |
| Baselines | Frozen trunk alone; trunk plus a fitted exponential decay; trunk plus the same policy trained by Optax through the solve. |
| Sharing | `_model.py` only. Truth and data generation are duplicated, since the two scripts build different datasets. |
| Dependency | `rlax` in the `examples` extra and the dev group, with a smoke test in `tests/test_example_batch_reactor_rl.py`. |

The observation includes the covariates because only `Ca` is measured and
`Ca0` is fixed: a policy on conversion alone returns the same action for a
hot acidic batch and a cool neutral one, and cannot fit the data.

## Truth

```
k_aged(T, pH, t) = a_true(t, pH) k_true(T, pH)
a_true(t, pH) = 1 / (1 + (t / tau(pH))^3),   tau(pH) = 1.2 (pH / 6)^2
```

`k_true` is part one's. Acid shortens `tau`, so low pH ages the catalyst
sooner. An exponential `a(t) = exp(-k_d t)` would be solvable in closed form
and one extra scalar would fit it exactly; the cubic Hill curve has a
plateau then a sharp fall, which no single decay scalar represents.

`TAU_REF` started at 2.6, which left the frozen trunk at validation R^2
0.895: too good a starting model to be worth correcting. At 1.2 the batch
stalls at roughly 55% conversion while the static model predicts nearly
complete conversion. The truth checkpoint asserts that consequence (an
absolute gap of at least 0.25 in final `Ca`), not particular activity
values.

`a_true(0, pH) = 1` exactly, which sets the action box below.

## Policy

```python
policy = BoundedPredictor(
    input_keys=("Ca", "temperature_C", "pH"),
    in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 50.0), (3.0, 9.0)), transform="sigmoid"),
    inner=MLPPredictor(in_size=3, out_size=1, width_size=32, depth=2,
                       activation_name="tanh", key=key),
    out_scaler=BoundScaler(bounds=((0.0, 1.05),), transform="sigmoid"),
)
```

- The upper bound is 1.05, not 1.0. A sigmoid only approaches its edge, so
  with a ceiling of 1 the policy would have to saturate to represent fresh
  catalyst and the saturation penalty would fight the fit.
- `tanh` activations, because a piecewise-linear activity curve reads badly
  against a smooth truth.
- Sampling splits the predictor at `out_scaler`: the mean is
  `inner(in_scaler.to_latent(obs))`, the spread is a state-independent
  `log_std`, the action is the latent `z`, and `out_scaler.from_latent(z)`
  runs inside the rollout. Recombined, the halves are exactly
  `BoundedPredictor.__call__`, so the trained policy drops into
  `predict_dataset`. A test pins this identity.
- The critic is an unbounded `MLPPredictor` on the same scaled observation.
- The agent is partitioned into `params` and `static` before the PPO scans,
  because `MLPPredictor` holds its activation as a callable leaf and a
  `lax.scan` carry must be arrays.

## Episodes and reward

One episode is one experiment: 12 observation times, 11 zero-order-hold
intervals, each a separate `diffeqsolve` inside a `lax.scan`.

```
r_t = exp(-(Ca_model - Ca_data)^2 / variance) - w * saturation(z_t)
```

- The variance comes from `ChannelObs.variance`, the dataset's own noise
  model. The penalty reads the latent, because `from_latent`'s derivative
  vanishes exactly where saturation is worst.
- Undiscounted (`discount = 1`, zero at the terminal step), GAE
  `lambda = 0.95`. The return is a fit criterion over a finite horizon.
- **The return has no ceiling of 11.** With residuals `N(0, sigma^2)` at the
  true model, `E[r] = 1/sqrt(3)`, so a perfect model scores about 6.35, and
  a model can score above the true law by fitting noise.
- **The exact hold target is the interval mean.** For
  `dCa/dt = -k a(t) Ca`, only the integral of `a` over an interval enters
  the solution, so holding activity at its interval mean reproduces the
  continuous truth exactly; the left endpoint overestimates on a falling
  curve. The reward checkpoint scores the truth this way, and every model
  on the page uses the same hold.
- 9 experiments times 64 noise samples gives 576 episodes per update.
  AdamW at `3e-4` with gradient clipping at 0.5, advantages normalised per
  batch. The Gaussian entropy is computed in closed form, since
  `rlax.entropy_loss` is categorical.

## Reporting

Five rows on the aged training and validation sets: the three baselines and
two PPO agents. Training return overshoots the true law (noise fitting), so
`train_ppo` keeps both the agent with the best training return and the
agent with the best validation return, and the page reports both. The
gradient-trained policy is expected to edge out PPO on a problem this
smooth, and the page says so.

Figures in `figures_rl/`: learning curves against the reference returns,
validation and training trajectories, recovered activity against the truth,
the latent distribution against the squash knee, and policy parity.

## Out of scope

- Library changes: no `train_with_rl`, config class or UI.
- SAC, model structure identification (the paper's Scenario 1 case 2), and
  matching the paper's numbers.
