# Batch reactor — hybrid (evosax → optax) example specification

**Status:** spec, not yet implemented. Locked through grilling on 2026-05-04.

This is the design contract for `examples/batch_reactor/train_hybrid.py` (a
single-file end-to-end script) and the matching documentation page at
`docs/examples/batch-reactor-hybrid.md`. All decisions below are firm; the
implementer should not re-litigate them without surfacing the issue.

---

## 1. Purpose

Demonstrate the canonical **hybrid grey-box pattern** end-to-end on a deliberately
simple physical system: a **batch reactor** running a first-order reaction
`A → B`, where the rate constant `k(T, pH)` has a known temperature law
(Arrhenius) and an unknown pH dependence. The example walks the reader through:

1. A **mechanistic parametric trunk** (Arrhenius only, two scalars), trained
   globally with **evosax/CMA-ES** to escape local minima and converge from a
   reasonable LHS prior.
2. A **residual MLP** (`(T, pH) → Δlog₁₀(k)`, 16-neuron) added log-additively on
   top of the frozen trunk, trained with **optax/AdamW** to capture the pH
   dependence the parametric cannot represent.

The two-phase pipeline shows off three under-demonstrated framework features:
- the trainability mask flipping between phases via `freeze_modules_of_type`,
- the multi-leaf `predictors` pytree convention (R-A2),
- the global-search-then-gradient-polish composition pattern documented in
  CONTEXT.md "Evosax training".

Pedagogically, this is the **textbook hybrid story**: parametric where physics
is well-understood (Arrhenius), neural where it is not (pH).

This example is **not** a CSTR. It is a closed batch reactor with no flow
terms. The naming reflects this everywhere; "CSTR" must not appear in code,
docstrings, file names, or docs.

---

## 2. Architecture summary

The model is a single `simulate_fn` integrating

```
dCa/dt = -k(T, pH) · Ca
dCb/dt =  k(T, pH) · Ca
```

with `k` decomposed log-additively from two predictors:

```
log₁₀(k(T, pH)) = log₁₀(k_param(T)) + Δlog₁₀(T, pH)
```

where:

- `k_param(T) = exp(log_k_ref - Ea/R · (1/T_K - 1/T_REF))` — the parametric
  trunk, **centred at `T_REF`**. Function of `T` only; pH-blind. Two trainable
  scalars `(log_k_ref, Ea)`. The centred form keeps `log_k_ref` directly
  interpretable as `ln(k(T_REF, _))` with tight, physical bounds (`[-3, 2]`),
  rather than the ~12 a non-centred `logA` would need.
- `Δlog₁₀(T, pH)` — the residual, output of a 16-neuron MLP wrapped in a
  `BoundedPredictor`. Bounded symmetric `[-2.0, +2.0]` decades. Function of
  both `T` and `pH`.

Predictors pytree shape: **`(parametric_trunk, residual_bp)`** — a tuple of two
`eqx.Module` leaves, mirroring `(growth_BP, nucleation_BP)` in
`train_kinetic.py`. Vector field unpacks `parametric, residual = predictors`.

---

## 3. The "true" data-generating model

The truth is private to the data generator; the predictors never see it.

```
k_true(T, pH) = k_sat(pH) · exp(-Ea_TRUE / R_GAS · (1/T_K - 1/T_REF))
```

where:

| Symbol      | Value                                              | Notes |
|-------------|----------------------------------------------------|-------|
| `k_sat(pH)` | `0.14 + 1.05 / (1 + (pH / 5.85)^5)`                | Slides' `saturation_k_from_ph`, ported verbatim. |
| `Ea_TRUE`   | `30.0` kJ/mol                                      | Mid-range activation energy; rate doubles ≈ per 10 °C. |
| `T_REF`     | `298.15` K (25 °C)                                 | Centring temperature so `k(T_REF, pH) = k_sat(pH)`. |
| `R_GAS`     | `8.314e-3` kJ/(mol·K)                              | Use kJ/(mol·K) consistently with `Ea` in kJ/mol. |

Helper: define a private `_k_true(temperature_C, pH) -> float` near the top of
the script. Used **only** for synthetic data generation and the truth overlay
in plots. Never imported by the predictor code path.

---

## 4. Synthetic dataset

### 4.1 Design of experiments

- **Training set: 9 LHS samples** in `(T, pH) ∈ [15, 35] °C × [4.5, 7.5]`,
  generated with `scipy.stats.qmc.LatinHypercube(d=2, seed=...)` (deterministic
  seed; a script flag `--doe-seed` exposes it). LHS gives near-uniform marginal
  coverage in both axes with only 9 points, which is why the framework already
  uses it in `init="lhs_box"` (R-E5).
- **Validation set: 2 hardcoded off-grid points** — `(T=20.0, pH=5.3)` and
  `(T=30.0, pH=6.8)`. Off any LHS sample; sit deliberately near the
  `pH = 5.85 = pH50` saturation knee where the parametric's pH-blindness shows
  most. These are stored in a separate `Dataset`, not in the training one.

### 4.2 Per-experiment observations

- `Ca₀ = 1.0` for every experiment. Fixed, not a covariate. Varying it would
  add a state-IC dimension without any rate-modulating role.
- `t ∈ [0, 5]`, `N_TIMESTEPS = 12` evenly-spaced observations.
- **Only `Ca` is observed.** `Cb = Ca₀ - Ca` for first-order `A → B`, so a `Cb`
  channel adds no information.
- Heteroscedastic Gaussian noise: `σ = 0.03 · max(|Ca|, 0.02)` — slides'
  `add_heteroscedastic_noise` ported verbatim. Negative samples clipped to 0.
  Per-experiment noise seed `= 11 + 75 · i` (where `i` is the experiment
  index), matching the slides' convention so the noise pattern is reproducible
  and visually similar to the slides figures.
- `ChannelObs.variance` populated as `σ²` per timestamp (variance, not std).

### 4.3 Bucket structure

All experiments share `len(union_ts) = 12`. `make_dataset` produces **one
bucket of size 9** (training) and **one bucket of size 2** (validation). One
bucket = one JIT compile per kernel = fastest possible.

### 4.4 Covariates

- `temperature_C: float` — physical units, °C, per-experiment scalar.
- `pH: float` — per-experiment scalar.

That's it. No `Ca0` covariate (fixed at 1.0). No "loading" or other
batch-reactor-irrelevant scalars. The covariates dict on each `Experiment` has
exactly two keys.

---

## 5. Predictors

### 5.1 `ArrheniusKinetics` — the parametric trunk

A small `eqx.Module` modelled on `KineticParameters` from
`train_crystallisation_mechanistic.py`. **Not** a `BoundedPredictor` — it has
no covariate inputs (it returns the parameters themselves; the vector field
combines them with `T`).

```python
class ArrheniusKinetics(eqx.Module):
    """Two trainable scalars (logA, Ea) with sigmoid-bounded latent."""

    latent: Float[Array, " 2"]
    out_scaler: BoundScaler  # bounds = (LOGA_BOUNDS, EA_BOUNDS)

    def __init__(self, *, key: Array) -> None:
        self.latent = jr.normal(key, (2,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOGA_BOUNDS, EA_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self) -> Float[Array, " 2"]:
        return self.out_scaler.from_latent(self.latent)
```

- `LOGA_BOUNDS = (-2.0, 8.0)` — `log10(A)` for prefactor `A` in `1/time-unit`.
  Wide enough to include the truth's effective `logA` regardless of `T_REF`
  centring. `logA = ln(A)` not `log10(A)` for Arrhenius — careful: `A` here is
  **physical** prefactor (not log-scale); we let the optimizer search a base-e
  exponent. Naming convention: in this module `logA` means the natural-log
  prefactor, consistent with `train_crystallisation_mechanistic.py::KineticParameters`.
- `EA_BOUNDS = (0.0, 80.0)` — kJ/mol. Truth sits at 30 kJ/mol so search has
  headroom on both sides.

### 5.2 `residual_bp` — the residual MLP

```python
INPUT_KEYS = ("temperature_C", "pH")
TEMPERATURE_BOUNDS = (0.0, 50.0)
PH_BOUNDS = (3.0, 9.0)
RES_LOG10_BOUNDS = (-2.0, 2.0)

residual_bp = BoundedPredictor(
    input_keys=INPUT_KEYS,
    in_scaler=BoundScaler(bounds=(TEMPERATURE_BOUNDS, PH_BOUNDS), transform="sigmoid"),
    inner=MLPPredictor(
        in_size=2,
        out_size=1,
        width_size=16,
        depth=1,
        activation_name="relu",
        key=k_residual,
    ),
    out_scaler=BoundScaler(bounds=(RES_LOG10_BOUNDS,), transform="sigmoid"),
)
```

- Output is `Δlog₁₀(k)` in decades.
- The output bound is **symmetric around 0**. A freshly-initialised MLP whose
  latent output is near zero gets mapped through the sigmoid to the midpoint
  of `[-2, +2]`, which is `0` decades = **identity residual**. Phase 2
  therefore starts at the exact phase-1 fit without any explicit zeroing.
- Input scaler bounds widen the data span (data: 15–35 °C, 4.5–7.5 pH) — same
  convention as `train_kinetic.py`'s `TEMPERATURE_BOUNDS`.

### 5.3 Predictors pytree

```python
predictors = (parametric_trunk, residual_bp)
```

Tuple of two `eqx.Module` leaves. Vector field unpacks
`parametric, residual = predictors`.

---

## 6. `simulate_fn`

```python
def simulate_fn(
    predictors: tuple[ArrheniusKinetics, BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    parametric, residual = predictors
    logA, Ea = parametric()              # [2] in physical units

    T_C = covariates["temperature_C"]
    pH = covariates["pH"]
    T_K = T_C + 273.15

    # log10(k_param) = (logA - Ea/(R·T_K)) / ln(10)
    log10_k_param = (logA - Ea / (R_GAS * T_K)) / jnp.log(10.0)

    inputs = {"temperature_C": T_C, "pH": pH}
    delta_log10_k = jnp.squeeze(residual(inputs))

    log10_k = log10_k_param + delta_log10_k
    k = jnp.power(10.0, log10_k)

    def vector_field(t, y, args):
        Ca, _Cb = y[0], y[1]
        rate = k * jnp.maximum(Ca, 0.0)  # guard against negative integrator excursions
        return jnp.stack([-rate, rate])
    # ... diffrax.diffeqsolve as in pendulum/train_harmonic.py
```

Two states `[Ca, Cb]`; `state_to_output` projects to `[Ca]` only (1 channel).
`y0_fn` returns `[Ca0, 0.0]` from the first observation:
`y0 = jnp.array([channels["Ca"].values[0], 0.0])`.

`Ca` clipping in the vector field guards against rare negative excursions of
the integrator near the asymptote — a one-line safety, not a workaround for a
deeper issue. Mass conservation is exact analytically; the clip stops a bad
solver step from driving `dCb/dt` negative.

Time units are arbitrary (slides' convention): `t ∈ [0, 5]` with
`k ∈ [0.14, 1.19]` gives ~one e-folding per `t = 1/k`. No second/minute
conversion (unlike crystallisation).

---

## 7. Training pipeline

### 7.1 Phase 1 — evosax (parametric)

```python
# Mask: parametric trainable, residual frozen wholesale.
mask_p1 = trainable_mask(predictors)
mask_p1 = freeze_modules_of_type(mask_p1, predictors, BoundedPredictor)
mask_p1 = freeze_modules_of_type(mask_p1, predictors, BoundScaler)

config_p1 = EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=32,
    num_generations=60,
    init="lhs_box",
    init_box_extent=2.0,
    sigma_init=0.5,
    loss="mse",
    verbose=True,
)
history_p1, predictors_p1 = train_with_evosax(
    predictors, dataset_train, config_p1,
    simulate_fn=simulate_fn, solver=solver,
    trainable=mask_p1, key=k_p1,
)
```

Why `freeze_modules_of_type(BoundedPredictor)` instead of `freeze_paths`:
`freeze_paths` requires explicit dotted leaf paths (`"1.inner.layers.0.weight"`,
…), which is brittle to MLP layer-naming changes and overly verbose for
"freeze the whole subtree". `freeze_modules_of_type` walks the pytree and
zeroes the entire `BoundedPredictor` submask in one call. Same effect, robust
to MLP internals. The `BoundScaler` freeze is the standard convention from
CONTEXT.md ("frozen by convention via `freeze_modules_of_type` … recommended in
every example") — applies to the parametric's `out_scaler` here.

The trainable parameter count after masking is **exactly 2** (the `latent`
vector of `ArrheniusKinetics`). Evosax is sized for this regime (R-E2: 4–10
dim; 2 is comfortably under).

### 7.2 Phase 2 — optax (residual)

```python
# Mask: parametric frozen, residual MLP trainable, BoundScalers frozen.
mask_p2 = trainable_mask(predictors_p1)
mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, ArrheniusKinetics)
mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, BoundScaler)

config_p2 = OptaxTrainingConfig(
    steps=(800,),
    lr=(3e-3,),
    optimizer=("adamw",),
    reset_optimiser_state=(False,),
    length_schedule=(1.0,),
    loss="mse",
    verbose=True,
)
history_p2, predictors_p2 = train_with_optax(
    predictors_p1, dataset_train, config_p2,
    simulate_fn=simulate_fn, solver=solver,
    trainable=mask_p2, key=k_p2,
)
```

Phase 2 starts from `predictors_p1` (the evosax-trained tuple). Because the
residual was frozen at its initial latent during phase 1 and its output bounds
are symmetric around 0, the phase-1 final loss equals the phase-2 step-0 loss
to within numerical noise — see verification §9.5.

### 7.3 Solver

```python
solver = SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-5,
    atol=1e-7,
    max_steps=10_000,
    dt0=0.05,
)
```

Same defaults as `pendulum/train_harmonic.py`. The dynamics are exp-decay-tame.

---

## 8. Plots

Seven plots total, four after phase 1, three after phase 2 + the loss curve.
Written to `examples/batch_reactor/figures/`. Use `apply_default_style()`
upstream, exactly as the existing examples do.

### 8.1 After phase 1 (evosax-fitted parametric)

1. **Trajectory grid** — 3×3 panels for the 9 training experiments (one per
   panel, sorted by pH then T). Each panel: noiseless truth (solid black),
   noisy observations (scatter, coloured by pH), parametric prediction
   (dashed, same colour as observations). Same panels at the same `T` but
   different `pH` will show the parametric collapsing to nearly-identical
   curves while the observations clearly fan apart — the headline failure
   mode. Reuse `_shared.trajectory_plot` if it accepts a 3×3 layout for one
   channel; otherwise a one-off helper inline in the script.

2. **Parity plot (training + validation overlay)** —
   `_shared.parity_plot(diagnostics_p1)`. Training markers and validation
   markers distinguished by edge colour or shape; validation set runs through
   the same `predict_dataset` call.

3. **k(pH) reveal at fixed T** — slides' `make_plot_6` analogue: plot
   `log10(k(pH))` over `pH ∈ [4, 8]` at three fixed `T` values
   (15, 25, 35 °C — the data's T-range endpoints + midpoint), one
   coloured curve per T. Solid for truth, dashed for parametric.
   The parametric's curves are flat in pH (constant per T) — visually
   obvious miss. LHS sample points and validation points overlaid as
   markers. Inline helper, not in `_shared`.

### 8.2 After phase 2 (hybrid)

4. **Trajectory grid** — same layout as plot 1, with the hybrid prediction
   replacing the parametric. Panels should now visibly fit truth at every
   `(T, pH)` combination.

5. **Parity plot (training + validation overlay)** — same shape as plot 2,
   tight cluster on the diagonal.

6. **k(pH) reveal at fixed T** — same axes as plot 3, with the hybrid
   prediction overlaid (dotted) alongside truth (solid) and parametric
   (dashed, carried over for direct visual comparison). Hybrid and truth
   should overlap; parametric stays flat.

### 8.3 Concatenated loss curve

7. **Loss curve** — single panel, x-axis is "training progress" (gen 0 → 60
   → step 0 → 800 concatenated), y-axis is `log10(loss)`. Vertical separator
   line at the phase boundary. Phase-1 trace is `history_p1` (best-loss-per-gen
   from evosax), phase-2 trace is `history_p2` (loss-per-step from optax). Sets
   a small precedent for combined-pipeline plots.

---

## 9. Verification checkpoints

Each checkpoint is a `print(...)` block plus an inline `assert ...` (or a
`warnings.warn` if the bound is too tight to make a hard assertion). The
implementer runs the script after each checkpoint comes online and confirms
the output makes sense before moving on. The numerical bounds below are
deliberately loose — they catch wiring errors, not parameter-sensitivity
drift.

### 9.1 Truth helper

After defining `_k_true`:

- `print` `_k_true(T_C=15.0, pH=5.85)`, `_k_true(T_C=25.0, pH=5.85)`, and
  `_k_true(T_C=35.0, pH=5.85)`. Expect roughly `(0.41, 0.67, 1.06)` (Arrhenius
  doubling per 10 °C around `pH50`).
- `assert 0.6 < _k_true(25.0, 5.85) < 0.75` (slides' `k_sat(5.85) = 0.665`,
  exact at `T_REF = 298.15 K`).

### 9.2 LHS dataset generation

After building `dataset_train`:

- `print` the 9 sampled `(T, pH)` pairs and 2 validation pairs.
- `print` `len(dataset_train.bucket_payloads)` — must be `1`.
- `print` `dataset_train.bucket_payloads[0].ts.shape` — must be `(9, 12)`.
- `assert int(dataset_train.bucket_payloads[0].n_obs) == 9 * 12`.
- `assert dataset_val.bucket_payloads[0].ts.shape == (2, 12)`.

### 9.3 Predictors at init

After building `predictors`:

- `parametric()` — print `(logA, Ea)` in physical units. Both should sit near
  the bound midpoints (`logA ≈ 3.0`, `Ea ≈ 40.0`) given the `latent ~ N(0,
  0.01)` init.
- `residual_bp({"temperature_C": 25.0, "pH": 5.85})` — print the scalar.
  Should sit near `0` decades (output bound midpoint, by design — see §5.2).
- `assert abs(float(jnp.squeeze(residual_bp(...)))) < 0.5` — fresh MLP cannot
  saturate the bound at random init.

### 9.4 Sanity-simulate at init

Pick the first training experiment, run `simulate_fn` once (no training
yet):

- `print` the `[12, 2]` returned trajectory's `Ca` column. Should be a
  monotone-decreasing exp-decay-shaped curve (because random-init `k > 0`
  almost always).
- `assert (sim_Ca[1:] <= sim_Ca[:-1] + 1e-6).all()` — non-increasing,
  modulo solver tolerance.
- `assert abs(float(sim_Ca[0]) - 1.0) < 1e-4` — `Ca` starts at `Ca0 = 1.0`.
- `assert float(sim_Ca[-1]) < 1.0` — something decayed.

### 9.5 Phase 1 transition

After phase 1 returns `history_p1, predictors_p1`:

- `print` final-generation best loss `history_p1[-1]`.
- `parametric_p1, residual_p1 = predictors_p1`. `print parametric_p1()`
  recovered `(logA, Ea)`. Expected ballpark: `Ea` somewhere in `[15, 50]`
  kJ/mol (truth is 30, but evosax fits an *aggregate* over 9 mixed-pH
  experiments, so it lands near the data's average effective Ea, not the
  truth's exact value).
- `assert history_p1[-1] < history_p1[0]` — evosax made progress.
- **Crucial frozen-residual check:** verify `residual_p1` weights are
  bit-exact equal to `residual_bp` at init. Pick one MLP weight leaf
  (e.g., `residual_p1.inner.layers[0].weight`) and `assert
  jnp.array_equal(...)` against the pre-phase-1 reference. If this fails,
  the trainability mask is wrong.
- **Phase-2-step-0 loss check:** compute the loss on `predictors_p1` (i.e.
  evaluate the model at the phase-1 endpoint). Print it. Should equal
  `history_p1[-1]` to within `1e-6` (same predictors, same data, same
  loss). If they differ, something between evosax internals and the
  fitness reporting is off.

### 9.6 Phase 2 transition

After phase 2 returns `history_p2, predictors_p2`:

- `print` final-step loss `history_p2[-1]` and `history_p2[0]` (the latter
  is the phase-1 transition loss).
- `assert history_p2[0] - history_p1[-1] < 1e-3` — phase 1 endpoint and
  phase 2 start agree (the residual was zero-init).
- `assert history_p2[-1] < 0.5 * history_p2[0]` — optax materially reduced
  the loss. The truth has a real pH shape that the parametric cannot
  capture; the residual MLP must do measurable work.
- **Crucial frozen-parametric check:** verify `parametric_p2.latent` is
  bit-exact equal to `parametric_p1.latent`. If different, mask flip
  failed.

### 9.7 Validation diagnostics

`predict_dataset(predictors_p2, dataset_val, ...)` followed by
`compute_diagnostics`. Print per-channel `R²` and `RMSE` for both training
and validation. Hybrid validation `R²` should comfortably exceed parametric
validation `R²` from §9.5; the spread between the two is the headline
quantitative result of the example.

---

## 10. File deliverables

```
examples/batch_reactor/
├── SPEC.md            (this file)
├── train_hybrid.py    (single-file script — all logic, no imports from siblings)
└── figures/           (created on first run; gitignored implicitly via .gitignore patterns)
```

Documentation page:

```
docs/examples/batch-reactor-hybrid.md  (registered in docs/.vitepress/config.ts)
```

The docs page mirrors the structure of `docs/examples/crystallisation.md` —
prose introduction, the truth, generating data, phase-1 walkthrough with
embedded plots 1–3, phase-2 walkthrough with plots 4–6, the loss curve, and
take-aways. Plots are referenced from `examples/batch_reactor/figures/` (or a
copy under `docs/public/`, matching whatever convention the existing pages
use).

---

## 11. Out of scope (do not creep)

- A CSTR variant. This is a batch reactor, end of story.
- Observing `Cb` as well as `Ca`. Adds no information for first-order kinetics.
- Varying `Ca₀` across experiments. Would add a state-IC dimension that the
  rate doesn't depend on; either the MLP ignores it (boring) or learns a
  spurious dependence (worse).
- A "red herring" covariate fed to the MLP that doesn't affect the rate.
  Pedagogically distinct lesson; if needed, a follow-up example.
- Time-varying covariates. State-derived inputs (e.g. supersaturation in
  crystallisation) are first-class via dict-mixing inside the vector field
  (CONTEXT.md "Predictor inputs"); this example doesn't need any.
- Sequential-pipeline alternatives where the MLP and parametric are blended
  via a learned mixing weight (Q5 option III). Settled: log-additive only.
- Tournament restarts in either phase (`tournament_attempts`, `tournament_steps`).
  Not needed for a 2-D evosax search and an 800-step optax polish on a
  9-experiment dataset. Add only if convergence is unreliable in practice.
- Multi-phase optax (phase 2 is a single phase). Adding phases would muddy
  the "evosax → optax" narrative.
- Updates to `SPEC.md` or `CONTEXT.md`. This example exercises only
  existing requirements (R-A2, R-F1, R-F3, R-E5) and introduces no new
  domain language.

---

## 12. Implementation order

The implementer should land the script in this order, running it at each
checkpoint to confirm:

1. Constants, `_k_true`, the heteroscedastic noise helper.
2. LHS DOE and dataset construction → run §9.2.
3. `simulate_fn`, `state_to_output`, `y0_fn`, the solver.
4. `ArrheniusKinetics` + `residual_bp` → run §9.3.
5. One sanity simulation at init → run §9.4.
6. Phase 1 mask + evosax call → run §9.5, generate plots 1–3.
7. Phase 2 mask + optax call → run §9.6, generate plots 4–6.
8. Loss-curve plot → §9.7 + plot 7.
9. Documentation page (after script is verified end-to-end).
