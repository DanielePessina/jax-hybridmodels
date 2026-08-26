# Batch reactor, part two: RL-trained catalyst deactivation

**Status:** spec, not yet implemented. Locked through grilling on 2026-08-26.

Design contract for `examples/batch_reactor/train_rl_deactivation.py`. Every
decision below was settled in the grilling session and should not be
re-litigated without surfacing the issue. Reference: Mowbray, Wu, Rogers, Del
Rio-Chanona and Zhang, "A reinforcement learning-based hybrid modeling
framework for bioprocess kinetics identification", Biotechnol Bioeng 120:154,
2023.

---

## 1. Purpose

Show that a `BoundedPredictor` can be trained as a **reinforcement learning
policy** that emits a bounded kinetic parameter at each time interval, with
**no gradient taken through the ODE solve**.

The paper reframes parameter estimation as control: kinetic parameters become
actions, a policy maps the current model state to those actions, and reward is
the negative fit error at the next measurement. This example carries that
reframing onto the batch reactor from `SPEC.md` and uses it to demonstrate one
specific claim.

**The claim.** Bounds are enforced by reparameterisation, not by clipping and
not by the `tanh` squash that continuous-control RL normally bolts onto its
actor. The policy distribution lives in the latent space; `BoundScaler`
carries the latent into the physical box. No log-det-Jacobian correction is
needed, the bound is structural, and `BoundScaler.saturation` becomes a reward
term that keeps the policy off the dead flat region of the squash.

**The secondary claim, which is the honest reason to reach for RL here.** PPO
differentiates only the policy's log-probability. The ODE rollout produces
rewards and is never differentiated. `SolverConfig.adjoint` is irrelevant to
this training loop, and the solver could be stiff or nonsmooth without
consequence. That is what RL buys, and it is the same argument the evosax
trainer makes by a different route.

**What this is not.** It is not a reproduction of the paper's numbers, not a
claim that RL beats gradient descent on this problem, and not a new library
feature. See section 8.

---

## 2. Decisions locked

| Decision | Choice |
|---|---|
| Scope | Example only. No `src/hybridmodels` changes, no SPEC.md growth, no new public class. |
| Thesis | Showcase the bounded controller under RL. |
| Rollout | Free-running on model state, as in the paper's Eq 4c and 4e. |
| RL stack | `rlax` for the PPO math, equinox actor and critic written here. |
| Algorithm | PPO, on-policy. |
| System | The batch reactor from `SPEC.md`, extended with catalyst deactivation. |
| Observation | `(Ca, temperature_C, pH)`. |
| Action | Catalyst activity, multiplying a frozen hybrid trunk. |
| Trunk | Trained on fresh-catalyst runs, frozen, applied to aged runs. |
| Truth | Sigmoidal in time with pH-dependent onset. |
| Dataset | The same 9 LHS points and 2 validation points as `SPEC.md`. |
| Reward | `exp` of the negative variance-weighted squared error. |
| Baselines | Frozen trunk alone; trunk plus a fitted exponential decay scalar; trunk plus the same `BoundedPredictor` trained by optax backprop. |
| Sharing | Extract `_model.py` only. Duplicate the data generation. |
| Deliverables | Script first. Notebook and docs page in a second pass. |
| Dependency | `rlax` in the `examples` extra and the dev group, with one smoke test. |

### 2.1 Why the observation includes covariates

Mowbray's policy sees the model state alone, which was complete for his
systems because they had no covariates. Here only `Ca` is observed and `Ca0`
is fixed at 1.0, so the model state is one scalar: conversion. The true rate
depends on `(T, pH)`, which vary across the 9 experiments. A policy on
conversion alone returns the same action for a hot acidic batch and a cool
neutral batch at equal conversion, and cannot fit the data even in principle.
Feeding the known experiment covariates to the policy is correct modelling.

### 2.2 Why deactivation is not exponential

For first-order `A -> B` with `a(t) = exp(-k_d t)`, the ODE
`dCa/dt = -k a(t) Ca` has a closed form and one extra trainable scalar fits it
exactly. A time-varying policy would then be unnecessary. The truth has to be
a shape no small fixed form absorbs.

---

## 3. The true data-generating model

Extends section 3 of `SPEC.md`. The fresh-catalyst truth is unchanged:

```
k_true(T, pH) = k_sat(pH) * exp(-Ea_TRUE / R_GAS * (1/T_K - 1/T_REF))
k_sat(pH)     = 0.14 + 1.05 / (1 + (pH / 5.85)^5)
```

Aged runs multiply it by an activity factor:

```
k_aged(T, pH, t) = a_true(t, pH) * k_true(T, pH)
a_true(t, pH)    = 1 / (1 + (t / tau(pH))^3)
tau(pH)          = TAU_REF * (pH / PH_REF)^2
```

with `TAU_REF = 2.6` (time units, against a batch length of 5) and
`PH_REF = 6.0`. Acid shortens the induction period, so a run at pH 4.5 loses
activity roughly twice as early as one at pH 7.5. The cubic exponent gives a
flat plateau followed by a sharp fall, which an exponential cannot represent.

`a_true(0, pH) = 1.0` exactly, which matters for the bound choice in 5.2.

Helpers `_activity_true(t, pH)` and `_k_aged_true(T, pH, t)` are private to
the new script and are used only for data generation and truth overlays. The
predictor code path never sees them.

---

## 4. Datasets

Two datasets, both built with the same `_lhs_design(seed)` from `SPEC.md`
section 4.1 so activity is the only difference between them.

**Fresh runs.** Identical to the existing example: 9 LHS training experiments
plus 2 off-grid validation points, `t` in `[0, 5]`, 12 evenly spaced
observations, `Ca` only, heteroscedastic noise `sigma = 0.03 * max(|Ca|,
0.02)`, `ChannelObs.variance` populated as `sigma^2`. These are the
commissioning runs on fresh catalyst, where activity is 1 by definition.

**Aged runs.** Same 9 plus 2 `(T, pH)` points, same time grid, same noise
model, but integrated with `k_aged`. A separate noise seed offset so the two
datasets are not correlated realisations.

Both produce one bucket of 9 and one bucket of 2, so each kernel compiles
once.

The aged-run generator is written fresh in `train_rl_deactivation.py` rather
than shared. It duplicates roughly 40 lines of noise and LHS helper code from
`train_hybrid.py`, which is a deliberate accepted cost of the "extract
`_model.py` only" decision.

---

## 5. Model

### 5.1 The frozen trunk

`train_hybrid.py` gains a `--save-predictors PATH` flag and writes its trained
`(ArrheniusKinetics, BoundedPredictor)` tuple with `save_predictors`. The RL
script loads it with `load_predictors` against a template built by the same
`_build_predictors`. `load_predictors` requires the template to carry
identical static configuration on every leaf, which is exactly why
`ArrheniusKinetics` has to be a shared class rather than a second definition.

The trunk is frozen. It is never an argument to any gradient transformation in
this script.

```
log10 k_hybrid(T, pH) = log10 k_param(T) + delta_log10(T, pH)
```

### 5.2 The policy

```python
ACTIVITY_BOUNDS = ((0.0, 1.05),)
OBS_BOUNDS = ((0.0, 1.0), (0.0, 50.0), (3.0, 9.0))   # Ca, temperature_C, pH

policy = BoundedPredictor(
    input_keys=("Ca", "temperature_C", "pH"),
    in_scaler=BoundScaler(bounds=OBS_BOUNDS, transform="sigmoid"),
    inner=MLPPredictor(in_size=3, out_size=1, width_size=32, depth=2,
                       activation_name="tanh", key=k_policy),
    out_scaler=BoundScaler(bounds=ACTIVITY_BOUNDS, transform="sigmoid"),
)
```

The upper bound is 1.05 rather than 1.0 because `a_true(0, pH) = 1.0` exactly
and a sigmoid only approaches its edge asymptotically. With a hard 1.0 ceiling
the policy would have to saturate to represent a fresh catalyst, and the
saturation term in the reward would fight the fit. The 5% headroom puts
`a = 1` comfortably inside the box. `tanh` rather than `relu` on the inner MLP
because a piecewise-linear activity curve reads badly against a smooth truth.

### 5.3 Splitting the policy for sampling

The stochastic policy is defined **in the latent space**, which is what
removes the Jacobian correction.

- Mean: `mu_z = policy.inner(policy.in_scaler.to_latent(obs))`, shape `(1,)`.
- Spread: a state-independent trainable `log_std` of shape `(1,)`, the
  standard continuous-control choice and more stable than a state-dependent
  head on a one-dimensional action.
- Sample: `z ~ Normal(mu_z, exp(log_std))`. Log-probability is a plain
  diagonal Gaussian, with no squash correction, because `z` is the action.
- Physical parameter: `a_t = policy.out_scaler.from_latent(z)`, applied inside
  the rollout.

At deployment the pieces recombine: `policy(obs)` is
`out_scaler.from_latent(inner(in_scaler.to_latent(obs)))`, the ordinary
`BoundedPredictor.__call__`. The trained artefact is therefore a plain
`BoundedPredictor` that drops straight into `predict_dataset`, which is the
point of splitting at `out_scaler` and nowhere else.

### 5.4 The critic

`MLPPredictor(in_size=3, out_size=1, width_size=32, depth=2)` on the same
scaled observation, unbounded output. Not a `BoundedPredictor`: a value
function has no physical box.

### 5.5 The vector field

```
dCa/dt = -a_t * k_hybrid(T, pH) * Ca
dCb/dt = +a_t * k_hybrid(T, pH) * Ca
```

`a_t` is held constant across each interval, zero-order hold. The truth is
continuous, so the hold is an approximation, and the docs must say so rather
than imply the recovered step function is the truth.

---

## 6. RL formulation

**Episode.** One experiment. 12 observation times give 11 intervals, so the
horizon is 11 steps. State at step `t` is the model's own `Ca`, not the
measurement, per Eq 4c.

**Transition.** One `diffeqsolve` from `t_i` to `t_{i+1}` with `a_t` fixed,
driven by `lax.scan` over intervals. The scan carries `(Ca, Cb)`.

**Reward.**

```
r_t = exp(-(Ca_model(t+1) - Ca_data(t+1))^2 / variance(t+1))
      - PENALTY_WEIGHT * out_scaler.saturation(z_t)
```

The first term is bounded in `[0, 1]`, so the undiscounted return over 11
intervals has a known ceiling of 11 and results can be reported as a fraction
of it. The variance comes from `ChannelObs.variance`, already on the dataset,
so the weighting is the framework's noise model rather than an arbitrary
constant. The second term is the saturation penalty evaluated on the sampled
latent, not on the physical activity, for the reason given in
`BoundScaler.saturation`'s docstring.

**Discount.** `gamma = 1.0`. The return is a fit criterion over a finite
horizon, not a control return, and discounting it would down-weight late
measurements for no modelling reason. GAE `lambda = 0.95`.

**Batching.** 9 training experiments vmapped as parallel environments, times
`N_ROLLOUTS = 64` independent policy noise samples each, giving 576 episodes
of 11 steps per update. Rollouts are cheap because nothing is differentiated.

**Losses, all from `rlax`.**

- `rlax.truncated_generalized_advantage_estimation` for advantages.
- `rlax.clipped_surrogate_pg_loss` with `epsilon = 0.2`.
- `rlax.entropy_loss` for the exploration bonus.
- Value loss as plain MSE against the GAE returns.

Optimiser `optax.adamw`, learning rate `3e-4`, gradient clipping at global
norm 0.5. Advantages normalised per batch.

Note that `rlax` operates on arrays, not on any network framework, and imposes
no environment API. That is the reason it was chosen over rejax, Stoix and
purejaxrl, all of which are flax-based and expect a gymnax-style
`env.step(key, state, action, params)`.

---

## 7. Baselines and reporting

Four rows, all scored on the aged validation set, reporting R^2 on `Ca` and
RMSE:

1. **Frozen trunk, no deactivation.** The floor. Motivates the page.
2. **Trunk plus a fitted exponential decay scalar.** One extra trainable
   scalar `k_d`, fitted with `train_with_optax`. The strongest fixed-form
   competitor, and it should visibly miss the induction plateau.
3. **Trunk plus the policy `BoundedPredictor`, trained by optax backprop.**
   Identical model, gradient training through the solve. The apples-to-apples
   row.
4. **Trunk plus the policy `BoundedPredictor`, trained by PPO.**

Row 3 may well beat row 4 on this smooth problem. If it does, the docs say so
plainly and the page's conclusion becomes the accurate one: RL matches a
gradient fit here while never touching the adjoint, which is what makes it
worth having when the adjoint is unavailable.

Plots:

- Recovered activity against `a_true(t, pH)`, one panel per validation point.
- Validation `Ca` trajectories, all four models overlaid on the data.
- PPO return against update, with the ceiling of 11 marked.
- Sampled latent `z` distribution against the saturation knee, to show the
  policy is operating inside the responsive region of the squash.

---

## 8. Out of scope

- Any change to `src/hybridmodels`. No `train_with_rl`, no config dataclass,
  no UI, no SPEC.md requirement.
- SAC. The paper uses it; PPO is chosen because the MDP is deterministic,
  episodes are 11 steps, and rollouts are nearly free, so off-policy replay
  buys little against roughly twice the code and considerably more tuning
  risk. The docs state the deviation.
- Model structure identification, which is the paper's Scenario 1 case 2.
  Interesting, and a different page.
- Matching the paper's reported numbers.

---

## 9. Build order

1. `_model.py` extraction. Move `ArrheniusKinetics`, `simulate_fn`, `y0_fn`
   and `state_to_output` out of `train_hybrid.py` and import them back. Rerun
   `train_hybrid.py` and check the reported numbers are byte-identical to the
   committed figures. No behaviour change.
2. `--save-predictors` flag on `train_hybrid.py`, and a committed trunk
   artefact so the RL script runs standalone.
3. `uv add --optional examples rlax` and `uv add --dev rlax`. Confirm the
   `tfp-nightly` transitive pin resolves in the lockfile. Fall back to
   vendoring the three `rlax` functions under Apache-2.0 attribution if it
   does not.
4. Aged-run data generation, plus a truth-helper check that
   `a_true(0, pH) == 1.0` and that activity at `t = 5` spans a useful range
   across the pH design.
5. Rollout and reward, with no learning: a `lax.scan` of `diffeqsolve` calls
   scored against the data. Verify the return under `a_t = a_true` approaches
   the ceiling of 11 and the return under `a_t = 1` does not.
6. PPO update step, driven by `rlax`. Smoke test: a handful of updates on a
   fixed key moves the return upward.
7. Baselines 1 to 3.
8. Plots, diagnostics and the results table.
9. Notebook and docs page, once the numbers stop moving.

Step 5 is the real checkpoint. If the return under the true activity is not
close to the ceiling, the reward scaling or the zero-order hold is wrong, and
no amount of PPO tuning will fix it.

---

## 10. Risks

- **PPO does not converge.** Most likely cause is credit assignment over 11
  steps with a free-running rollout, where an early bad action poisons the
  episode. Mitigation, in order: raise `N_ROLLOUTS`, then shorten the horizon
  by training on a prefix first (the framework's `length_schedule` idea,
  applied by hand), then fall back to teacher forcing as a warm start.
- **`tfp-nightly` breaks the lockfile.** Mitigation is the vendoring fallback
  in step 3. `rlax` imports `distrax` only in `distributions.py` and
  `policy_targets.py`, neither of which PPO needs.
- **Row 3 beats row 4 decisively.** Not a failure. Report it, and lead the
  page with the no-adjoint property rather than with accuracy.
- **The extraction in step 1 changes behaviour.** Mitigated by the
  byte-identical check before anything else is built.
