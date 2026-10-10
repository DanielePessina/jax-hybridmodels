# Batch reactor hybrid: design record

Design notes for `train_hybrid.py` and `_model.py`. The walkthrough is
`docs/examples/batch-reactor.md`; this file keeps the decisions and their
reasons. Part two is in `SPEC_RL.md`.

## Purpose

The textbook hybrid story on a small system: parametric where the physics
is known (Arrhenius in temperature), neural where it is not (pH). It shows
a global search followed by gradient polish on one `HybridModel`, with a
`trainable` mask switching which predictor trains.

The system is a closed batch reactor running `A -> B`, not a CSTR. There
are no flow terms.

## Model

```
dCa/dt = -k Ca,   dCb/dt = k Ca
log10 k(T, pH) = log10 k_param(T) + Δlog10(T, pH)
k_param(T) = exp(log_k_ref - Ea/R (1/T_K - 1/T_REF))
```

- `ArrheniusKinetics` (trunk): two scalars in a latent vector, mapped into
  `LOG_KREF_BOUNDS = (-3, 2)` and `EA_BOUNDS = (0, 80)` kJ/mol by a sigmoid
  `BoundScaler`. Centring at `T_REF = 298.15 K` keeps `log_k_ref` equal to
  `ln k(T_REF)`, which needs a far tighter box than an uncentred
  prefactor. It takes no covariates.
- Residual: `BoundedPredictor` over `("temperature_C", "pH")`, a 16-neuron
  ReLU MLP, output box `(-2, 2)` decades. The box is symmetric, so a fresh
  network outputs about zero and phase 2 starts exactly where phase 1
  ended, with no explicit zeroing.
- Predictors are the tuple `(trunk, residual)`.
- Only `Ca` is observed; `Cb = Ca0 - Ca` adds nothing. `y0` reads the first
  `Ca` observation. The vector field clips `Ca` at zero against rare
  negative solver steps.
- Temperature follows a flat-ramp-flat profile per run. Its four parameters
  are covariates and `ramp_profile` evaluates `T(t)` inside the vector
  field, so each run sweeps `T_C ± 3 °C`.

The trunk, vector field, hooks and predictor builder live in `_model.py`
because `train_rl_deactivation.py` loads the saved predictors against a
template, which needs identical classes and static configuration.

## Truth and data

```
k_true(T, pH) = k_sat(pH) exp(-Ea_TRUE/R (1/T_K - 1/T_REF))
k_sat(pH) = 0.14 + 1.05 / (1 + (pH/5.85)^5),   Ea_TRUE = 30 kJ/mol
```

- Training: 9 Latin hypercube points of ramp-midpoint `T` in `[15, 35] °C`
  and pH in `[4.5, 7.5]` (`--doe-seed`).
- Validation: `(20 °C, pH 5.3)` and `(30 °C, pH 6.8)`, off the design and
  near the saturation knee, where the trunk's pH-blindness shows most.
- 12 evenly spaced samples on `t in [0, 5]`, `Ca0 = 1`. Noise
  `sigma = 0.03 max(|Ca|, 0.02)`, clipped at zero, with `sigma^2` stored as
  the channel variance.
- One shared time grid, so each dataset is one bucket and each kernel
  compiles once.

## Training

1. CMA-ES, 60 generations, population 32, `init="lhs_box"`. The mask is
   `trainable_mask` (which already fixes every `BoundScaler`) with the
   residual frozen by `freeze_modules_of_type(..., BoundedPredictor)`,
   leaving exactly two scalars. `freeze_modules_of_type` beats
   `freeze_paths` here because it does not depend on the MLP's leaf names.
2. AdamW, 800 steps, `lr = 3e-3`, with the trunk frozen by
   `freeze_modules_of_type(..., ArrheniusKinetics)`.

Solver: Tsit5, `rtol = 1e-5`, `atol = 1e-7`, `dt0 = 0.05`.

## Checkpoints

The script asserts, with loose bounds that catch wiring errors rather than
tuning drift:

- `k_true(25 °C, pH 5.85)` is about 0.665 and `k_true` rises with `T`.
- Each dataset is one bucket of the expected shape.
- The fresh residual outputs about zero; an untrained simulation decays
  monotonically from the first observation.
- Phase 1 trains exactly 2 scalars, lowers the loss, and leaves the
  residual bit-identical.
- Phase 2 leaves the trunk bit-identical. It warns, rather than fails, if
  its first loss is more than `1e-3` from phase 1's last or if it does not
  halve the loss, so short runs still finish.

## Figures

Written to `figures/`: trajectory grid, parity and `log10 k(pH)` reveal
after each phase (`01` to `06`), and the combined loss curve (`07`).

## Out of scope

- Observing `Cb`, varying `Ca0`, or a red-herring covariate. `notebook.py`
  covers `Ca0` as a red herring.
- Blending trunk and residual with a learned weight. Log-additive only.
- Tournament restarts or multi-phase Optax. Neither is needed here.
