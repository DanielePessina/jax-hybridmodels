# Crystallisation (mechanistic, CNT + power-law)

The parametric counterpart to the [hybrid example](/examples/crystallisation).
It uses the same four experiments, moment ODE and `state_to_output`, and
replaces the two rate networks with closed-form kinetics: classical
nucleation theory (CNT) for nucleation and a power law for growth. Four
fitted scalars on this dataset are the baseline a hybrid model has to
beat.

```bash
uv run python examples/crystallisation/train_crystallisation_mechanistic.py
```

The script imports the dataset, the moment equations and the observation
model from `train_kinetic.py`, so read the
[hybrid page](/examples/crystallisation) first. This page covers what
differs.

## Rate laws

$$
J = \exp(\log A) \cdot S \cdot \exp\!\left( -\frac{16\pi\, \gamma^3\, v^2}{3 (k_B T)^3 \ln^2 S} \right)
\qquad
G = \frac{10^{A_g}}{60} \cdot \max(S - 1,\ 0)^{g}
$$

| name    | meaning                                 | bounds        |
|---------|-----------------------------------------|---------------|
| `logA`  | $\ln A$, CNT pre-exponential factor     | (20.0, 65.0)  |
| `gamma` | interfacial energy [mJ/m²]              | (0.15, 1.0)   |
| `Ag`    | $\log_{10}$ growth pre-factor [m/min]   | (-20.0, -5.0) |
| `g`     | growth exponent                         | (1.0, 3.5)    |

The bounds come from the thesis package, so the fitted values are
comparable with its runs.

## Parameters

Each constant is a `Parameter` that starts at the midpoint of its bounds:

```python
PARAMETER_BOUNDS = {
    "logA": (20.0, 65.0),
    "gamma": (0.15, 1.0),
    "Ag": (-20.0, -5.0),
    "g": (1.0, 3.5),
}
predictors = {
    name: hm.Parameter((low + high) / 2, bounds=(low, high))
    for name, (low, high) in PARAMETER_BOUNDS.items()
}
```

A `Parameter` stores an unbounded latent and returns a bounded physical
value from `.value`. CMA-ES searches the latents, and every candidate it
proposes maps inside the box, so the search needs no constraint handling.

## Vector field

The constants are the same for every experiment, so `simulate_fn` reads
them once, outside the vector field:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    logA = predictors["logA"].value
    gamma = predictors["gamma"].value * 1e-3  # mJ/m^2 to J/m^2
    Ag = predictors["Ag"].value
    g = predictors["g"].value

    temperature_C = covariates["temperature_C"]
    T_K = temperature_C + 273.15
    conc_sat = saturation_conc(temperature_C)

    def vector_field(t, y, args):
        S = y[5] / conc_sat
        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

        S_safe = jnp.clip(S, min=1.0 + 1e-12)
        cnt_exponent = (-16.0 * jnp.pi * gamma**3 * M_V**2
                        / (3.0 * (K_B * T_K) ** 3 * jnp.log(S_safe) ** 2))
        J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exponent)
        G = meta_mask * 10.0**Ag / 60.0 * jnp.maximum(S - 1.0, 0.0) ** g
        return moment_rhs(y, G, J)

    return integrate_minutes(vector_field, ts, y0, solver)
```

`S_safe` is the one structural difference from the hybrid vector field.
At `S <= 1`, `ln S` is zero or negative and sits squared in a
denominator. `meta_mask` zeros the rate there, but the gradient through
the unclipped expression is still `nan`, and it reaches every step of a
sub-saturated trajectory. Clip the argument, not the result.

## Training with CMA-ES

```python
config = hm.EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=64,
    num_generations=80,
    init="lhs_box",
    init_box_extent=2.0,
    sigma_init=0.5,
)
result = model.fit(dataset, config, key=jr.key(0))
```

Passing an `EvosaxTrainingConfig` to `fit` selects the evolutionary
trainer. Four scalars suit a population search, and CMA-ES gets past the
local minima the CNT exponential creates near the metastable limit.
`init="lhs_box"` fills generation 0 with a Latin hypercube over the
latent box, so the first evaluation covers the whole box. Each generation
evaluates the population in one compiled, vmapped call.

`result.history.best_fitness` holds the best loss so far at each
generation, and `result.model.predictors` the fitted constants. Prediction
and plots work as in the hybrid example.

## Hybrid and mechanistic side by side

| Aspect     | hybrid                                     | mechanistic                         |
|------------|--------------------------------------------|-------------------------------------|
| Predictors | two `BoundedPredictor` networks            | four `Parameter` scalars            |
| Inputs     | `temperature_C`, `supersaturation`         | none                                |
| Rate laws  | learned $\log_{10} G$, $\log_{10} J$       | CNT nucleation, power-law growth    |
| Config     | `OptaxTrainingConfig`                      | `EvosaxTrainingConfig` (CMA-ES)     |
| Budget     | 600 gradient steps                         | 80 generations of 64 candidates     |
