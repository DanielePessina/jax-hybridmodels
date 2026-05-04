# Crystallisation (mechanistic, CNT + power-law)

A four-parameter parametric counterpart to the [hybrid MLP example](/examples/crystallisation). Same four hardcoded experiments, same method-of-moments ODE, same `state_to_output` projector — but the two MLP rate predictors are replaced by a single `KineticParameters` module holding four scalars `(logA, gamma, Ag, g)` that drive Classical Nucleation Theory and a power-law growth term inside the vector field.

The full script lives at `examples/crystallisation/train_crystallisation_mechanistic.py`. Run it with:

```bash
uv run python examples/crystallisation/train_crystallisation_mechanistic.py
```

This page focuses only on what *differs* from the hybrid MLP version. For dataset details, the moment ODE, the `state_to_output` projector, and bucketing, read [Crystallisation (hybrid MLP)](/examples/crystallisation) first.

## Why a separate example

Two reasons:

1. **It's the parametric baseline.** Knowing what a small mechanistic model achieves on this dataset is the obvious yardstick for any hybrid model trained on it. Four parameters versus an MLP each side per rate.
2. **It's the canonical evosax use case.** `train_with_evosax` is sized for handful-of-scalar problems where gradient-based optimisation is overkill and CMA-ES sidesteps the local minima the CNT exponential creates near the metastable limit.

Everything else — dataset loading, `y0_fn`, `state_to_output`, the moment ODE, the solver — is identical to the hybrid MLP example.

## The dataset

Same four hardcoded experiments as the hybrid MLP example:

| `exp_id`      | T (°C) | n conc | n d43 | t span (min) | terminal d43 (µm) |
|---------------|-------:|-------:|------:|--------------|------------------:|
| `LowData4_3`  | 17.0   | 9      | 1     | 0 → 270      | 9.2               |
| `LowData4_4`  | 17.0   | 9      | 1     | 0 → 270      | 7.7               |
| `LowData4_7`  | 21.0   | 7      | 1     | 0 → 360      | 10.5              |
| `LowData4_9`  | 21.0   | 7      | 1     | 0 → 375      | 11.9              |

Concentration variance is a single made-up scalar (`CONC_VAR = 0.1`) broadcast across every row; the d43 variances are the rounded thesis values.

## Step 1 — the four mechanistic rate laws

The vector field uses the same population-balance moment ODEs as the hybrid example, but $G$ and $J$ are now the parametric forms

$$
J = \exp(\log A) \cdot S \cdot \exp\!\left( -\frac{16\pi\, \gamma^3\, v^2}{3 (k_B T)^3 \ln^2 S} \right)
$$

$$
G = \frac{10^{A_g}}{60} \cdot \max(S - 1,\ 0)^{g}
$$

with four global trainable scalars:

| symbol  | name           | physical meaning                       | bounds          |
|---------|----------------|----------------------------------------|-----------------|
| `logA`  | `LOGA_BOUNDS`  | $\ln A$, CNT pre-exponential           | (20.0, 65.0)    |
| `gamma` | `GAMMA_BOUNDS` | interfacial energy [mJ/m²]             | (0.15, 1.0)     |
| `Ag`    | `AG_BOUNDS`    | $\log_{10}$ growth pre-factor [m/s]    | (-20.0, -5.0)   |
| `g`     | `G_BOUNDS`     | power-law growth exponent              | (1.0, 3.5)      |

These are reproduced from the thesis-package bounds in `hybridcrystals/regressor_constants.py` so the trained scalars sit in the same physical box as the source-package runs.

## Step 2 — the `KineticParameters` module

The framework's `BoundedPredictor` is built around a covariate-keyed `__call__(dict | Array) -> Array` and requires at least one input. Here the four parameters are *global* (no covariate dependence at all), so we sidestep `BoundedPredictor` and define a minimal `eqx.Module` whose only inexact-array leaf is the four-vector latent. The `BoundScaler` output mapping is the same primitive the MLP-based predictor uses, so the optimiser still operates in an unbounded latent space and the simulator still sees physical-units parameters inside the bound box.

```python
import equinox as eqx
from hybridmodels import BoundScaler

LOGA_BOUNDS = (20.0, 65.0)
GAMMA_BOUNDS = (0.15, 1.0)
AG_BOUNDS = (-20.0, -5.0)
G_BOUNDS = (1.0, 3.5)

class KineticParameters(eqx.Module):
    """Four global mechanistic kinetic constants [logA, gamma, Ag, g]."""

    latent: jax.Array  # shape [4]
    out_scaler: BoundScaler

    def __init__(self, *, key):
        # Small Gaussian init in latent space puts the physical parameters
        # near the centre of each bound at gen 0; CMA-ES expands from there.
        self.latent = jr.normal(key, (4,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOGA_BOUNDS, GAMMA_BOUNDS, AG_BOUNDS, G_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self):
        return self.out_scaler.from_latent(self.latent)
```

CMA-ES sees a four-dimensional unbounded search; the sigmoid in `out_scaler` keeps every candidate inside the physical box no matter how wide the search spreads.

## Step 3 — the vector field

The simulator evaluates `predictor()` once at the top of the call (the parameters are global, not state-dependent) and closes them over by the vector field. The `(S > 1 + eps)` mask gates both rates so the ODE stops moving below the metastable limit.

```python
M_V = 2.97e-26          # molecular volume [m^3]
K_B = 1.38064852e-23    # Boltzmann constant [J/K]
META_EPS = 1e-5

def simulate_fn(predictor, ts, covariates, y0, solver):
    params = predictor()  # [4] in physical units, sigmoid-bounded
    logA = params[0]
    gamma_J_m2 = params[1] * 1e-3   # bounds in [mJ/m^2]; CNT in [J/m^2]
    Ag = params[2]
    g_exp = params[3]

    temperature_C = covariates["temperature_C"]
    T_K = temperature_C + 273.15
    conc_sat = (0.3705 + 7.171e-2 * temperature_C
                - 1.924e-3 * temperature_C**2 + 17.97e-5 * temperature_C**3)

    def vector_field(t, y, args):
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat
        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

        # Clip S inside the log so the exponent stays finite when meta_mask
        # is zero — the mask wipes the contribution out anyway, but the
        # gradient through the clipped log must stay finite or autodiff
        # returns NaN at every ODE step on a sub-saturated trajectory.
        S_safe = jnp.clip(S, min=1.0 + 1e-12)
        logS = jnp.log(S_safe)
        cnt_exp = (-16.0 * jnp.pi * gamma_J_m2**3 * M_V**2
                   / (3.0 * (K_B * T_K)**3 * logS**2))
        J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exp)

        growth_drive = jnp.maximum(S - 1.0, 0.0)
        G = meta_mask * jnp.power(10.0, Ag) / 60.0 * jnp.power(growth_drive, g_exp)

        return jnp.stack([
            J, G * mu0, 2.0 * G * mu1, 3.0 * G * mu2, 4.0 * G * mu3,
            -3.0 * K_V * RHO_C * G * mu2,
        ])

    # diffeqsolve call identical to the hybrid example
    ...
```

The `S_safe` clip on the inside of the log is the only structural difference from the MLP vector field. Without it, `log(S)` is `-inf` whenever `S <= 1`, so the CNT exponent is `-inf`, `exp(...)` is zero, *but the gradient through `log` is still NaN* and propagates regardless of `meta_mask`.

## Step 4 — train with evosax (CMA-ES)

```python
from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax

predictor = KineticParameters(key=k_init)

config = EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=64,
    num_generations=80,
    init="lhs_box",          # broadest 4-D coverage at gen 0
    init_box_extent=2.0,
    sigma_init=0.5,
    loss="mse",
    verbose=True,
)
history, trained_predictor = train_with_evosax(
    predictor, dataset, config,
    simulate_fn=simulate_fn, solver=solver, key=k_train,
)
print(f"final best loss: {history[-1]:.6f}")
```

`init="lhs_box"` injects a Latin-hypercube-sampled population at generation 0, so every quadrant of the four-parameter box is touched on the first evaluation; CMA-ES then takes over with `sigma_init=0.5` and adapts the step size from there. Population evaluation is `vmap`-parallel — 64 individuals × 4 buckets × 4 experiments fuses into one JIT-compiled kernel after the first generation.

The Rich UI's recent-generations table is sized to show **five rows total** (one row every `num_generations // 5 = 16` generations by default) so the table grows over the run rather than sliding past a constantly-changing window.

## Step 5 — read the trained constants

```python
final_params = trained_predictor()
print(
    f"logA={float(final_params[0]):.2f}, "
    f"gamma={float(final_params[1]):.3f} mJ/m^2, "
    f"Ag={float(final_params[2]):.2f}, "
    f"g={float(final_params[3]):.2f}"
)
```

These four scalars *are* the trained model. Same `predict_dataset` + diagnostics + parity/trajectory plots as the hybrid example for evaluation.

## Comparing against the hybrid model

The two scripts share dataset, ODE, projector, solver, and diagnostics — only the trainable component and the loop change.

| Aspect              | hybrid MLP                          | mechanistic                         |
|---------------------|-------------------------------------|-------------------------------------|
| Trainable component | `(growth_bp, nucleation_bp)` MLPs   | one `KineticParameters` (4 scalars) |
| Inputs              | `(temperature_C, supersaturation)`  | none (global parameters)            |
| Rate laws           | learned $\log_{10} G$, $\log_{10} J$ | CNT-J + power-law-G (parametric)    |
| Trainer             | `train_with_optax` (Adam/AdamW)     | `train_with_evosax` (CMA-ES)        |
| Default budget      | 600 gradient steps                  | 80 generations × 64 individuals     |
| Param count         | ~thousands per branch               | 4                                   |

## What's next

- [Crystallisation (hybrid MLP)](/examples/crystallisation) — the canonical end-to-end walk-through.
- [Recommendations](/guide/recommendations) — bounds choice, tolerances, freezing, autodiff-safe guarded division.
- [Training](/guide/training) — Optax, evosax, and the shared tournament.
- [API: Training](/api/training) — full reference for `EvosaxTrainingConfig` and `train_with_evosax`.
