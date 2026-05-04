# Crystallisation

This is the canonical end-to-end example for `hybridmodels`. We train a hybrid kinetic model on the thesis crystallisation dataset — irregular concentration and particle-size observations across many experiments — where two `BoundedPredictor`s emit log-rates feeding a method-of-moments ODE.

The full script lives at `examples/crystallisation/train_kinetic.py`. Run it with:

```bash
uv run python examples/crystallisation/train_kinetic.py
```

The thesis Excel is bundled alongside, so no external paths are required. This page walks through the script piece by piece — every step here corresponds directly to a section in the file.

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

The two observed channels are concentration (dense) and the volume-weighted mean diameter $d_{43} = \mu_4 / \mu_3 \cdot 10^6$ µm (sparser).

`loading` is also stored on `Experiment.covariates` (the dataset records it for every experiment) but the *direct-rate* predictors don't condition on it — both MLPs see only temperature and supersaturation. This is a deliberate scope choice; loading-dependent kinetics is a follow-on extension.

## Step 1 — load and wrap as `Experiment`

Each experiment is a row group in the Excel sheet. Per-channel sparsity is recovered by filtering rows where `d43 == -1` (the missing-row sentinel).

```python
from hybridmodels import ChannelObs, Experiment, make_experiment

def _y0_fn(covariates, channels):
    """Build the initial state [mu0, mu1, mu2, mu3, mu4, conc] for one experiment."""
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])

experiments = []
for exp_id, df_exp in df.groupby("Exp_ID"):
    time_min = jnp.asarray(df_exp["Time"].to_numpy(dtype=float))
    conc = jnp.asarray(df_exp["Concentration"].to_numpy(dtype=float))

    channels = {"conc": ChannelObs(ts=time_min, values=conc, variance=conc_var)}

    # d43 is sparser; filter out -1 sentinels so its ts axis is independent.
    valid_mask = (d43_arr > 0.0) & jnp.isfinite(d43_arr)
    if bool(jnp.any(valid_mask)):
        valid_idx = jnp.where(valid_mask)[0]
        channels["d43"] = ChannelObs(
            ts=time_min[valid_idx], values=d43_arr[valid_idx], variance=d43_var,
        )

    experiments.append(make_experiment(
        covariates={"temperature_C": float(df_exp["Temperature"].iloc[0]),
                    "loading": float(df_exp["Loading"].iloc[0])},
        channels=channels,
        y0_fn=_y0_fn,
        exp_id=f"{sheet_name}_{int(exp_id)}",
    ))
```

The `_y0_fn` hook runs *once per experiment* at dataset-build time. Five population-balance moments start at zero (suspension nominally clear at `t = 0`); concentration starts at the first observed value of the `conc` channel.

## Step 2 — `state_to_output` projector

The integrator returns the full six-state trajectory; we observe only `(conc, d43)`. The projector converts moments to $d_{43}$ with an autodiff-safe guarded division — see [Recommendations → Pitfalls](/guide/recommendations#autodiff-safe-guarded-division) for why the `safe_mu3` trick is mandatory.

```python
def _d43_from_moments(mu3, mu4):
    safe_mu3 = jnp.where(mu3 > _D43_MU3_EPS, mu3, 1.0)
    ratio = jnp.where(mu3 > _D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
    return jnp.clip(ratio, 0.0, _D43_MAX)

def _state_to_output(state):
    """[T, 6] -> [T, 2] in OUTPUT_CHANNELS = ('conc', 'd43') order."""
    return jnp.stack([state[..., 5], _d43_from_moments(state[..., 3], state[..., 4])], axis=-1)
```

## Step 3 — bucket via `make_dataset`

```python
from hybridmodels import make_dataset

dataset = make_dataset(
    experiments,
    state_to_output=_state_to_output,
    output_channel_names=("conc", "d43"),
)
print(f"{len(dataset.bucket_payloads)} buckets")
for i, bp in enumerate(dataset.bucket_payloads):
    print(f"  bucket {i}: ts={tuple(bp.ts.shape)}, n_obs={int(bp.n_obs)}")
```

Experiments with the same union timestamp count are stacked into one [`BucketPayload`](/api/data#bucketpayload). Each bucket is a JIT cache key; the framework compiles `make_step` once per bucket shape, then reuses it across every step.

## Step 4 — predictors: two `BoundedPredictor`s

The `predictors` pytree convention is a **tuple of predictors** — `(growth_bp, nucleation_bp)`. Both consume the same 2-key dict; the `BoundedPredictor` subsets by its `input_keys` field.

```python
from hybridmodels import BoundedPredictor, BoundScaler, MLPPredictor

INPUT_KEYS = ("temperature_C", "supersaturation")

in_scaler = BoundScaler(
    bounds=((13.0, 27.0), (0.0, 12.0)),  # (T_C, S)
    transform="sigmoid",
)

k_growth, k_nucleation = jr.split(key, 2)

growth = BoundedPredictor(
    input_keys=INPUT_KEYS,
    in_scaler=in_scaler,
    inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                       depth=1, activation_name="relu", key=k_growth),
    out_scaler=BoundScaler(bounds=((-15.0, -5.0),), transform="sigmoid"),   # log10(G), m/s
)
nucleation = BoundedPredictor(
    input_keys=INPUT_KEYS,
    in_scaler=in_scaler,
    inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                       depth=1, activation_name="relu", key=k_nucleation),
    out_scaler=BoundScaler(bounds=((-6.5, 20.0),), transform="sigmoid"),    # log10(J), #/(m^3*s)
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
def _simulate_fn(predictors, ts, covariates, y0, solver):
    growth_bp, nucleation_bp = predictors
    temperature_C = covariates["temperature_C"]
    conc_sat = _conc_sat(temperature_C)  # empirical solubility polynomial

    def vector_field(t, y, args):
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat

        # Predictor inputs: temperature is a covariate (constant in time),
        # supersaturation is state-derived. The framework treats every key
        # as a named scalar regardless of provenance.
        inputs = {"temperature_C": temperature_C, "supersaturation": S}

        meta_mask = (S > 1.0 + 1e-5).astype(y.dtype)
        log10_G = jnp.squeeze(growth_bp(inputs))
        log10_J = jnp.squeeze(nucleation_bp(inputs))
        G = meta_mask * jnp.power(10.0, log10_G)
        J = meta_mask * jnp.power(10.0, log10_J)

        return jnp.stack([
            J,                                   # dmu0
            G * mu0,                             # dmu1
            2.0 * G * mu1,                       # dmu2
            3.0 * G * mu2,                       # dmu3
            4.0 * G * mu3,                       # dmu4
            -3.0 * _K_V * _RHO_C * G * mu2,      # dconc
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
    simulate_fn=_simulate_fn, solver=solver, key=jr.PRNGKey(1),
)
print(f"final loss: {history[-1]:.6f}")
```

600 steps × N buckets is what compiles. The first step pays the JIT cost (one trace per bucket shape); subsequent steps are at full JAX speed.

## Step 8 — predict and diagnose

```python
from hybridmodels import predict_dataset

predictions = predict_dataset(
    trained_predictors, dataset, simulate_fn=_simulate_fn, solver=solver,
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

- [Pendulum example](/examples/pendulum) — a 60-line synthetic counterpart with a known optimum.
- [Recommendations](/guide/recommendations) — bounds choice, tolerances, freezing, and the autodiff-safe `where` pattern used in `_d43_from_moments`.
- [Training](/guide/training) — multi-phase Optax, the shared tournament, evosax.
- [API: Data](/api/data), [API: Predictors](/api/predictors), [API: Training](/api/training) — full reference.
