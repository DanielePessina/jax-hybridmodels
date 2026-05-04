# Crystallisation (hybrid MLP)

This is the canonical end-to-end example for `hybridmodels`. We train a hybrid kinetic model on four crystallisation experiments — irregular concentration trajectories with a single terminal particle-size measurement — where two `BoundedPredictor`s emit log-rates feeding a method-of-moments ODE.

The full script lives at `examples/crystallisation/train_kinetic.py`. Run it with:

```bash
uv run python examples/crystallisation/train_kinetic.py
```

The four experiments are inlined in the script — no Excel, CSV, or external data dependency. This page walks through the script piece by piece; every step here corresponds directly to a section in the file.

## What we're modelling

Six-state population balance ODE:

$$
\begin{aligned}
\frac{d\mu_0}{dt} &= J(t) \\
\frac{d\mu_k}{dt} &= k\, G(t)\, \mu_{k-1}, \quad k = 1, 2, 3, 4 \\
\frac{d\,\text{conc}}{dt} &= -3\, K_v\, \rho_c\, G(t)\, \mu_2
\end{aligned}
$$

where $G$ (growth velocity, m/s) and $J$ (nucleation rate, #/m³·s) are the *unknown* rate functions. The hybrid part is: instead of committing to a CNT or power-law form, we let two MLPs learn $\log_{10} G$ and $\log_{10} J$ from `(temperature_C, supersaturation)`. The vector field exponentiates back to physical rates inside the integrator.

The two observed channels are concentration (dense) and the volume-weighted mean diameter $d_{43} = \mu_4 / \mu_3 \cdot 10^6$ µm (a single terminal observation per experiment in this cut).

For a parametric counterpart that fits four CNT + power-law scalars instead of two MLPs, see [Crystallisation (mechanistic)](/examples/crystallisation-mechanistic).

## The dataset

Four experiments reproduced from the thesis `Unseeded_LowData4` cut, rounded to 1 dp on values and 3 dp on variance. Concentration variance is a single made-up scalar (`CONC_VAR = 0.1`) broadcast across every row; the d43 variances are the rounded thesis values.

| `exp_id`      | T (°C) | n conc | n d43 | t span (min) | terminal d43 (µm) |
|---------------|-------:|-------:|------:|--------------|------------------:|
| `LowData4_3`  | 17.0   | 9      | 1     | 0 → 270      | 9.2               |
| `LowData4_4`  | 17.0   | 9      | 1     | 0 → 270      | 7.7               |
| `LowData4_7`  | 21.0   | 7      | 1     | 0 → 360      | 10.5              |
| `LowData4_9`  | 21.0   | 7      | 1     | 0 → 375      | 11.9              |

Two experiments at each of two temperatures, with different time grids. `make_dataset` will form a per-experiment union axis and bucket by length parity, so you'll see two buckets at training time.

## Step 1 — define the hardcoded experiments

```python
from hybridmodels import ChannelObs, Experiment, make_experiment

EXPERIMENTS_DATA = (
    {
        "exp_id": "LowData4_3",
        "temperature_C": 17.0,
        "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
        "conc": (14.7, 13.7, 7.7, 7.0, 5.7, 5.5, 5.4, 5.1, 5.2),
        "d43_time_min": 270.0,
        "d43": 9.2,
        "d43_var": 5.345,
    },
    # ... three more entries (LowData4_4, _7, _9), see the script
)
CONC_VAR = 0.1  # made-up uniform concentration variance, broadcast per row

def y0_fn(covariates, channels):
    """Initial state [mu0..mu4, conc]: moments at zero, conc at first observation."""
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])

experiments = []
for data in EXPERIMENTS_DATA:
    time_min = jnp.asarray(data["time_min"], dtype=float)
    conc = jnp.asarray(data["conc"], dtype=float)
    experiments.append(make_experiment(
        covariates={"temperature_C": float(data["temperature_C"])},
        channels={
            "conc": ChannelObs(
                ts=time_min, values=conc,
                variance=jnp.full_like(conc, CONC_VAR),
            ),
            "d43": ChannelObs(
                ts=jnp.asarray((data["d43_time_min"],), dtype=float),
                values=jnp.asarray((data["d43"],), dtype=float),
                variance=jnp.asarray((data["d43_var"],), dtype=float),
            ),
        },
        y0_fn=y0_fn,
        exp_id=str(data["exp_id"]),
    ))
```

The `y0_fn` hook runs *once per experiment* at dataset-build time. Five population-balance moments start at zero (suspension nominally clear at `t = 0`); concentration starts at the first observed value of the `conc` channel. Temperature is the only covariate — the thesis `Loading` column is uniformly zero across LowData4 and is intentionally dropped.

The two channels carry independent `ts` axes: `conc` has 7-9 points per experiment, `d43` has just one (the terminal measurement). `make_dataset` recovers per-channel sparsity from this without any user mask code.

## Step 2 — `state_to_output` projector

The integrator returns the full six-state trajectory; we observe only `(conc, d43)`. The projector converts moments to $d_{43}$ with an autodiff-safe guarded division — see [Recommendations → Pitfalls](/guide/recommendations#autodiff-safe-guarded-division) for why the `safe_mu3` trick is mandatory.

```python
D43_MU3_EPS = 1e-6
D43_MAX = 55.0

def state_to_output(state):
    """[T, 6] -> [T, 2] in OUTPUT_CHANNELS = ('conc', 'd43') order."""
    mu3, mu4, conc = state[..., 3], state[..., 4], state[..., 5]
    safe_mu3 = jnp.where(mu3 > D43_MU3_EPS, mu3, 1.0)
    ratio = jnp.where(mu3 > D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
    d43 = jnp.clip(jnp.where(jnp.isfinite(ratio) & (ratio > 0.0), ratio, 0.0), 0.0, D43_MAX)
    return jnp.stack([conc, d43], axis=-1)
```

## Step 3 — bucket via `make_dataset`

```python
from hybridmodels import make_dataset

dataset = make_dataset(
    experiments,
    state_to_output=state_to_output,
    output_channel_names=("conc", "d43"),
)
print(f"{len(dataset.bucket_payloads)} buckets")
for i, bp in enumerate(dataset.bucket_payloads):
    print(f"  bucket {i}: ts={tuple(bp.ts.shape)}, n_obs={int(bp.n_obs)}")
```

For LowData4 you'll see two buckets — one for the two T=17 experiments (length 9, since `d43_time_min=270` is already in the conc grid) and one for the two T=21 experiments (length 7, with `d43_time_min` in {360, 375} extending the union axis by one). Each bucket is a JIT cache key; the framework compiles `make_step` once per bucket shape, then reuses it across every step.

## Step 4 — predictors: two `BoundedPredictor`s

The `predictors` pytree convention is a **tuple of predictors** — `(growth_bp, nucleation_bp)`. Both consume the same 2-key dict; the `BoundedPredictor` subsets by its `input_keys` field.

::: info MLP and KAN side by side
The shipped script trains the same `(growth_bp, nucleation_bp)` topology twice — once with an `MLPPredictor` inner and once with a `KANPredictor` inner — and writes both sets of figures into `figures/mlp/` and `figures/kan/`. The walkthrough below shows the MLP path; the KAN swap is one line (replace `MLPPredictor(...)` with `KANPredictor(in_size=2, out_size=1, hidden_widths=(64,), grid_size=5, basis="spline", key=...)`). Bound choices, scalers, and the surrounding ODE are identical.
:::

```python
from hybridmodels import BoundedPredictor, BoundScaler, MLPPredictor

INPUT_KEYS = ("temperature_C", "supersaturation")
TEMPERATURE_BOUNDS = (13.0, 27.0)        # °C, slightly wider than data span
SUPERSATURATION_BOUNDS = (0.0, 12.0)     # S = conc / conc_sat
LOG10_GROWTH_BOUNDS = (-15.0, -5.0)      # log10(G [m/s])
LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)   # log10(J [#/(m^3·s)])

in_scaler = BoundScaler(
    bounds=(TEMPERATURE_BOUNDS, SUPERSATURATION_BOUNDS),
    transform="sigmoid",
)

k_growth, k_nucleation = jr.split(key, 2)

growth = BoundedPredictor(
    input_keys=INPUT_KEYS,
    in_scaler=in_scaler,
    inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                       depth=1, activation_name="relu", key=k_growth),
    out_scaler=BoundScaler(bounds=(LOG10_GROWTH_BOUNDS,), transform="sigmoid"),
)
nucleation = BoundedPredictor(
    input_keys=INPUT_KEYS,
    in_scaler=in_scaler,
    inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                       depth=1, activation_name="relu", key=k_nucleation),
    out_scaler=BoundScaler(bounds=(LOG10_NUCLEATION_BOUNDS,), transform="sigmoid"),
)

predictors = (growth, nucleation)
```

The bound choice matters. `LOG10_GROWTH_BOUNDS = (-15, -5)` puts the sigmoid midpoint at $G \approx 10^{-10}$ m/s — physically reasonable for early-time growth. An earlier `(-12, -3)` draft put the midpoint at $G \approx 3 \cdot 10^{-8}$ m/s, large enough that random-init weights produced ODE rates the moment-balance solver could not track within `max_steps`. See [Recommendations → Bounds](/guide/recommendations#bounds).

::: tip Optional: pin the readout to the bound midpoint
For very wide rate bounds — like the nucleation `(-6.5, 20.0)` decade range here — an unlucky standard-normal readout draw can still place the initial output far enough off midpoint that the moment ODE is intractably stiff. [`MLPPredictor.with_zero_final_head()`](/api/predictors#mlppredictor) (and its KAN counterpart) returns a copy of the predictor whose final readout layer is zeroed, so the initial output sits at the *exact* physical midpoint regardless of the random key. The hidden layers keep their default init, so the input feature transformation is non-degenerate. Drop in by chaining onto the constructor:

```python
inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                   depth=1, activation_name="relu", key=k_growth).with_zero_final_head()
```

The shipped script runs without it for the default seed; turn it on if you see `max_steps` exceeded on a different seed.
:::

## Step 5 — the vector field

This is where you write physics. The framework's `simulate_fn` signature is fixed; everything inside is yours.

```python
RHO_C = 1370.0    # crystal density [kg/m^3]
K_V = 0.81        # volumetric shape factor
META_EPS = 1e-5   # supersaturation must exceed 1 + eps for nucleation/growth

def simulate_fn(predictors, ts, covariates, y0, solver):
    growth_bp, nucleation_bp = predictors
    temperature_C = covariates["temperature_C"]
    # Empirical solubility polynomial in °C.
    conc_sat = (0.3705 + 7.171e-2 * temperature_C
                - 1.924e-3 * temperature_C**2
                + 17.97e-5 * temperature_C**3)

    def vector_field(t, y, args):
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat

        # Predictor inputs: temperature is a covariate (constant in time),
        # supersaturation is state-derived. The framework treats every key
        # as a named scalar regardless of provenance.
        inputs = {"temperature_C": temperature_C, "supersaturation": S}

        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)
        log10_G = jnp.squeeze(growth_bp(inputs))
        log10_J = jnp.squeeze(nucleation_bp(inputs))
        G = meta_mask * jnp.power(10.0, log10_G)
        J = meta_mask * jnp.power(10.0, log10_J)

        return jnp.stack([
            J,                                 # dmu0
            G * mu0,                           # dmu1
            2.0 * G * mu1,                     # dmu2
            3.0 * G * mu2,                     # dmu3
            4.0 * G * mu3,                     # dmu4
            -3.0 * K_V * RHO_C * G * mu2,      # dconc
        ])

    times_sec = ts * 60.0  # dataset stores minutes; rate constants are SI seconds.
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field), solver.solver,
        t0=times_sec[0], t1=times_sec[-1], dt0=solver.dt0, y0=y0,
        saveat=diffrax.SaveAt(ts=times_sec),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)
```

Two notes: (1) the metastable mask `(S > 1 + 1e-5)` zeros both rates below the metastable limit, so the ODE stops moving on the dissolved branch; (2) the time conversion is one-line — keeping the dataset in minutes makes plots readable.

## Step 6 — solver and training config

Per-state `atol` matched to natural moment magnitudes — population-balance moments span ~18 decades during integration, so a uniform `atol` over-resolves the small components and under-resolves the large ones.

```python
from hybridmodels import SolverConfig
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

solver = SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-4,
    atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),  # per-state floor at ~9 decades below natural magnitude
    max_steps=500_000,
    dt0=None,
)

config = OptaxTrainingConfig(
    steps=(600,),
    lr=(1e-3,),
    optimizer=("adamw",),
    reset_optimiser_state=(False,),
    length_schedule=(1.0,),
    loss="mse",
    verbose=True,
)
```

Single-phase to start. Once the loss flattens, add a second phase with a lower LR for fine-tuning.

## Step 7 — train

```python
history, trained_predictors = train_with_optax(
    predictors, dataset, config,
    simulate_fn=simulate_fn, solver=solver, key=jr.PRNGKey(0),
)
print(f"final loss: {history[-1]:.6f}")
```

600 steps × N buckets is what compiles. The first step pays the JIT cost (one trace per bucket shape); subsequent steps are at full JAX speed.

## Step 8 — predict and diagnose

```python
from hybridmodels import predict_dataset

predictions = predict_dataset(
    trained_predictors, dataset, simulate_fn=simulate_fn, solver=solver,
)
# predictions is a tuple — one [N, T, D] array per bucket, parallel to dataset.bucket_payloads.

# Read the trained log-rates at any (T, S) point you like.
sample_inputs = {
    "temperature_C": jnp.asarray(20.0),
    "supersaturation": jnp.asarray(1.5),
}
trained_growth, trained_nucleation = trained_predictors
log10_G = float(jnp.squeeze(trained_growth(sample_inputs)))
log10_J = float(jnp.squeeze(trained_nucleation(sample_inputs)))
print(f"G(20°C, S=1.5) = {10**log10_G:.2e} m/s, J = {10**log10_J:.2e} #/(m³·s)")
```

The shipped script also writes parity and trajectory plots under `examples/crystallisation/figures/` via the helpers in `examples/_shared/`.

## What's next

- [Crystallisation (mechanistic)](/examples/crystallisation-mechanistic) — same dataset and ODE backbone, but with four CNT + power-law scalars trained by CMA-ES.
- [Pendulum example](/examples/pendulum) — a 60-line synthetic counterpart with a known optimum.
- [Recommendations](/guide/recommendations) — bounds choice, tolerances, freezing, and the autodiff-safe `where` pattern used in `state_to_output`.
- [Training](/guide/training) — multi-phase Optax, the shared tournament, evosax.
- [API: Data](/api/data), [API: Predictors](/api/predictors), [API: Training](/api/training) — full reference.
