# Crystallisation (mechanistic, CNT + power-law)

The parametric counterpart to the [hybrid MLP example](/examples/crystallisation).
Same four experiments, same population-balance ODE, same
`state_to_output`. The two neural rate predictors are replaced by one
`KineticParameters` module holding four scalars `(logA, gamma, Ag, g)`
that drive Classical Nucleation Theory and a power-law growth term
inside the vector field.

**Classical Nucleation Theory** (CNT) is the standard closed-form
expression for how fast new crystals appear, given supersaturation and
temperature, with two free constants. Using it means committing to that
mechanism, which is what the hybrid model avoids and what makes this the
right baseline to measure it against.

The full script lives at `examples/crystallisation/train_crystallisation_mechanistic.py`. Run it with:

```bash
uv run python examples/crystallisation/train_crystallisation_mechanistic.py
```

This page covers only what differs from the hybrid version. Read
[Crystallisation (hybrid MLP)](/examples/crystallisation) first for the
dataset, the moment ODE, the projector, and bucketing.

## Why a separate example

It is the parametric baseline: what four fitted scalars achieve on this
dataset is the yardstick any hybrid model has to beat.

It is also the canonical use for `train_with_evosax`. Population search
is sized for handful-of-scalar problems where gradients are overkill, and
CMA-ES steps past the local minima the CNT exponential creates near the
metastable limit.

Everything else (dataset loading, `y0_fn`, `state_to_output`, the moment
ODE, the solver) is identical.

## The dataset

The same four experiments as the hybrid MLP example.

| `exp_id`      | T (°C) | n conc | n d43 | t span (min) | terminal d43 (µm) |
|---------------|-------:|-------:|------:|--------------|------------------:|
| `LowData4_3`  | 17.0   | 9      | 1     | 0 → 270      | 9.2               |
| `LowData4_4`  | 17.0   | 9      | 1     | 0 → 270      | 7.7               |
| `LowData4_7`  | 21.0   | 7      | 1     | 0 → 360      | 10.5              |
| `LowData4_9`  | 21.0   | 7      | 1     | 0 → 375      | 11.9              |

Concentration variance is a single made-up scalar (`CONC_VAR = 0.1`) broadcast across every row; the d43 variances are the rounded thesis values.

## Step 1: the four mechanistic rate laws

The vector field uses the same population-balance moment ODEs as the hybrid example, but $G$ and $J$ are now the parametric forms

$$
J = \exp(\log A) \cdot S \cdot \exp\!\left( -\frac{16\pi\, \gamma^3\, v^2}{3 (k_B T)^3 \ln^2 S} \right)
$$

$$
G = \frac{10^{A_g}}{60} \cdot \max(S - 1,\ 0)^{g}
$$

with four global trainable scalars:

| symbol  | constant       | physical meaning                       | bounds          |
|---------|----------------|----------------------------------------|-----------------|
| `logA`  | `LOGA_BOUNDS`  | $\ln A$, CNT pre-exponential           | (20.0, 65.0)    |
| `gamma` | `GAMMA_BOUNDS` | interfacial energy [mJ/m²]             | (0.15, 1.0)     |
| `Ag`    | `AG_BOUNDS`    | $\log_{10}$ growth pre-factor [m/s]    | (-20.0, -5.0)   |
| `g`     | `G_BOUNDS`     | power-law growth exponent              | (1.0, 3.5)      |

These bounds are reproduced from `hybridcrystals/regressor_constants.py`
so the fitted scalars sit in the same physical box as the source-package
runs.

## Step 2: the KineticParameters module

`BoundedPredictor` requires at least one input, since a predictor with
none has no training signal. These four parameters are global, so it is
skipped here in favour of a minimal `eqx.Module` whose only trainable
array is the four-vector latent.

It still uses `BoundScaler` for the output map, so the optimiser works in
an unbounded space while the simulator sees physical units inside the
declared box. The predictors pytree can be any container; the library
never inspects it.

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

CMA-ES sees an unconstrained four-dimensional search. The squash in
`out_scaler` keeps every candidate inside the physical box however wide
the search spreads, which is why nothing enforces bounds during the
search itself.

## Step 3: the vector field

The parameters are global, so `predictor()` is called once at the top
and closed over by the vector field, above `diffeqsolve`. It never lands
on the solver tape. The `(S > 1 + eps)` mask gates both rates so the ODE
stops moving below the metastable limit.

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
        # is zero. The mask wipes the contribution out anyway, but the
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

The `S_safe` clip inside the log is the only structural difference from
the hybrid vector field. Without it, `log(S)` is `-inf` whenever
`S <= 1`, so the CNT exponent is `-inf` and `exp(...)` is zero. The
value is fine. The gradient through `log` is still `nan`, and it
propagates regardless of `meta_mask`. Clip the argument, not the
result.

## Step 4: train with evosax (CMA-ES)

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

`init="lhs_box"` seeds generation 0 with a Latin hypercube, so every
corner of the four-parameter box is touched on the first evaluation, and
CMA-ES adapts the step size from `sigma_init=0.5`. The population is
evaluated in parallel: 64 individuals across 4 buckets and 4 experiments
fuse into one compiled kernel after the first generation.

## Step 5: read the trained constants

```python
final_params = trained_predictor()
print(
    f"logA={float(final_params[0]):.2f}, "
    f"gamma={float(final_params[1]):.3f} mJ/m^2, "
    f"Ag={float(final_params[2]):.2f}, "
    f"g={float(final_params[3]):.2f}"
)
```

These four scalars are the whole trained model. Evaluation is the same
`predict_dataset` call and the same plots as the hybrid example.

## Comparing against the hybrid model

The two scripts share dataset, ODE, projector, solver, and diagnostics.
Only the trainable component and the training loop change.

| Aspect              | hybrid MLP                          | mechanistic                         |
|---------------------|-------------------------------------|-------------------------------------|
| Trainable component | `(growth_bp, nucleation_bp)` MLPs   | one `KineticParameters` (4 scalars) |
| Inputs              | `(temperature_C, supersaturation)`  | none (global parameters)            |
| Rate laws           | learned $\log_{10} G$, $\log_{10} J$ | CNT-J + power-law-G (parametric)    |
| Trainer             | `train_with_optax` (Adam/AdamW)     | `train_with_evosax` (CMA-ES)        |
| Default budget      | 600 gradient steps                  | 80 generations × 64 individuals     |
| Param count         | ~thousands per branch               | 4                                   |

## What's next

- [Crystallisation (hybrid MLP)](/examples/crystallisation). The
  end-to-end walkthrough.
- [Recommendations](/guide/recommendations). Bounds, tolerances,
  freezing, guarded division.
- [Training](/guide/training). Optax, evosax, and the tournament.
- [API: Training](/api/training). Full reference for
  `EvosaxTrainingConfig` and `train_with_evosax`.
