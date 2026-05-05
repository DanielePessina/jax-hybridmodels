"""Batch reactor walkthrough — evosax → optax hybrid pipeline.

Walks through the same physics as ``examples/batch_reactor/train_hybrid.py``
in narrative form: a first-order ``A -> B`` batch reactor whose true rate
constant ``k(T, pH)`` has a known temperature law (Arrhenius) and an unknown
pH dependence (a saturation curve hidden from the predictor). Two-phase fit:

* **Phase 1 (evosax/CMA-ES)** — fits a deliberately too-simple parametric
  trunk (centred Arrhenius, pH-blind), 2 scalars.
* **Phase 2 (optax/AdamW)** — adds a small residual MLP that captures the pH
  shape the parametric cannot represent.

Run interactively: ``uv run marimo edit examples/batch_reactor/notebook.py``
Run as script:     ``uv run python examples/batch_reactor/notebook.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Batch reactor: an evosax → optax hybrid pipeline

    This notebook fits a hybrid grey-box model of a batch reactor in two
    phases, using one global-search optimiser followed by one
    gradient-based optimiser on the same model. The example is
    deliberately small: a single first-order reaction $A \to B$ in a
    closed batch reactor, with a rate constant $k(T,\mathrm{pH})$ that
    we never let the predictor see directly. The exercise is to recover
    that hidden $k$ from a handful of noisy concentration trajectories.

    The pipeline shows three things at once:

    1. **A parametric trunk fit by global search.** The trunk is a
       centred Arrhenius law in temperature with two scalars
       $(\log k_{\mathrm{ref}},\ E_a)$. It is pH-blind by construction —
       the role of phase 1 is to nail the temperature dependence and
       leave the pH residual to the next phase. Optimisation uses
       Covariance Matrix Adaptation Evolution Strategy (CMA-ES) through
       `train_with_evosax`, which is well suited to small bounded
       parameter searches.
    2. **A neural residual fit by gradient descent.** A 16-neuron MLP
       wrapped in a `BoundedPredictor` adds a log-additive correction
       $\Delta\log_{10}k(T,\mathrm{pH})$ on top of the frozen Arrhenius
       trunk. Optimisation uses Adam through `train_with_optax`.
    3. **A trainability mask flipping between phases.** The same
       predictors pytree drives both phases. A boolean mask selects
       which leaves are trainable: the parametric latent in phase 1
       (residual frozen at zero), the MLP weights in phase 2
       (parametric frozen at the phase-1 endpoint).

    The hidden truth is a centred-Arrhenius temperature law multiplied
    by a sigmoidal pH-saturation factor — defined precisely in the
    *True model* section below — and the predictor never sees that
    factor directly; it only sees noisy concentration trajectories.

    A note on framework conventions used throughout. The trainable
    component of any model in `hybridmodels` is a *pytree of
    `eqx.Module` leaves*; the convention is to wrap it in a tuple
    even for the single-leaf case. The integrator is a user-supplied
    `simulate_fn` with a fixed signature; the framework owns vmap,
    jit, and autodiff, the user owns the physics.
    """)
    return


@app.cell
def _imports():
    import diffrax
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import Array
    from jaxtyping import Float
    from scipy.stats import qmc

    import marimo as mo
    from hybridmodels import (
        BoundedPredictor,
        BoundScaler,
        ChannelObs,
        Experiment,
        MLPPredictor,
        SolverConfig,
        freeze_modules_of_type,
        make_dataset,
        make_experiment,
        predict_dataset,
        trainable_mask,
    )
    from hybridmodels.training import (
        EvosaxTrainingConfig,
        OptaxTrainingConfig,
        train_with_evosax,
        train_with_optax,
    )

    return (
        Array,
        BoundScaler,
        BoundedPredictor,
        ChannelObs,
        EvosaxTrainingConfig,
        Experiment,
        Float,
        MLPPredictor,
        OptaxTrainingConfig,
        SolverConfig,
        diffrax,
        eqx,
        freeze_modules_of_type,
        jax,
        jnp,
        jr,
        make_dataset,
        make_experiment,
        mo,
        np,
        plt,
        predict_dataset,
        qmc,
        train_with_evosax,
        train_with_optax,
        trainable_mask,
    )


@app.cell(hide_code=True)
def _truth_md(mo):
    mo.md(r"""
    ## True model

    The reactor runs a single first-order reaction $A \to B$ in a closed
    batch at constant volume. The mass balance is

    $$
    \frac{dC_A}{dt} = -k(T,\mathrm{pH})\,C_A,
    \qquad
    \frac{dC_B}{dt} = +k(T,\mathrm{pH})\,C_A,
    $$

    with $C_A(0) = C_{A,0}$ and $C_B(0) = 0$. Mass conservation gives
    $C_A(t) + C_B(t) = C_{A,0}$ for all $t$, so observing $C_A$ alone is
    sufficient and the closed-form solution is a pure exponential decay
    $C_A(t) = C_{A,0}\,e^{-k(T,\mathrm{pH})\,t}$.

    The hidden rate constant factorises as Arrhenius in temperature
    times a Hill-type saturation in pH:

    $$
    k(T, \mathrm{pH}) = k_{\mathrm{sat}}(\mathrm{pH}) \cdot
    \exp\!\Big(-\frac{E_a}{R}\Big(\frac{1}{T_K} - \frac{1}{T_{\mathrm{ref}}}\Big)\Big),
    \qquad
    k_{\mathrm{sat}}(\mathrm{pH}) = b + \frac{a}{1 + (\mathrm{pH}/\mathrm{pH}_{50})^{n}}.
    $$

    Numerical values: baseline $b = 0.14$, amplitude $a = 1.05$,
    half-saturation $\mathrm{pH}_{50} = 5.85$, Hill coefficient $n = 5$,
    activation energy $E_a^{\mathrm{true}} = 30\,\mathrm{kJ/mol}$, and
    centring temperature $T_{\mathrm{ref}} = 298.15\,\mathrm{K}$. The
    Arrhenius factor is centred so that
    $k(T_{\mathrm{ref}}, \mathrm{pH}) = k_{\mathrm{sat}}(\mathrm{pH})$
    exactly. Together this gives a smooth roll-off from a maximum rate
    near $\mathrm{pH} = 4$ to a plateau near $\mathrm{pH} = 8$, with
    roughly a two-fold rate change per ten Kelvin around
    $T_{\mathrm{ref}}$.

    ```python
    T_REF = 298.15        # K (centring temperature)
    R_GAS = 8.314e-3      # kJ/(mol·K)
    EA_TRUE = 30.0        # kJ/mol

    def _k_sat_from_ph(pH):
        return 0.14 + 1.05 / (1.0 + jnp.maximum(pH / 5.85, 0.0) ** 5.0)

    def k_true(temperature_C, pH):
        T_K = jnp.asarray(temperature_C) + 273.15
        arrhenius = jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))
        return _k_sat_from_ph(pH) * arrhenius
    ```
    """)
    return


@app.cell
def _truth(jnp):
    # Truth and physics constants. Used only for synthetic data generation
    # and overlay curves in the plots; never imported into the predictor
    # code path.
    T_REF = 298.15  # K (25 °C); centring temperature for Arrhenius
    R_GAS = 8.314e-3  # kJ/(mol·K); pair with Ea in kJ/mol
    EA_TRUE = 30.0  # kJ/mol; rate doubles ~per 10 °C around T_REF

    K_SAT_BASELINE = 0.14
    K_SAT_AMPLITUDE = 1.05
    K_SAT_PH50 = 5.85
    K_SAT_HILL = 5.0

    def _k_sat_from_ph(pH):
        """Sigmoidal saturation curve. Hidden truth for the pH dependence."""
        pH_arr = jnp.asarray(pH)
        return K_SAT_BASELINE + K_SAT_AMPLITUDE / (
            1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
        )

    def k_true(temperature_C, pH):
        """Ground-truth rate constant: pH saturation × Arrhenius centred at T_REF."""
        T_K = jnp.asarray(temperature_C) + 273.15
        arrhenius = jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))
        return _k_sat_from_ph(pH) * arrhenius

    print(f"k_true(15°C, pH=5.85) = {float(k_true(15.0, K_SAT_PH50)):.4f}")
    print(
        f"k_true(25°C, pH=5.85) = {float(k_true(25.0, K_SAT_PH50)):.4f}  (== k_sat by construction)"
    )
    print(f"k_true(35°C, pH=5.85) = {float(k_true(35.0, K_SAT_PH50)):.4f}")
    return R_GAS, T_REF, k_true


@app.cell(hide_code=True)
def _proposed_md(mo):
    mo.md(r"""
    ## Proposed mechanistic model

    The proposed model integrates the same batch reactor mass balance,

    $$
    \frac{dC_A}{dt} = -\hat{k}(T,\mathrm{pH},C_{A,0})\,C_A,
    \qquad C_A(0) = C_{A,0},
    $$

    but the rate constant $\hat{k}$ is now a hybrid of a deliberately
    simple parametric trunk and a small neural residual, combined
    log-additively:

    $$
    \log_{10}\hat{k}(T,\mathrm{pH},C_{A,0})
    \;=\;
    \underbrace{\log_{10} k_{\mathrm{param}}(T)}_{\text{parametric trunk}}
    \;+\;
    \underbrace{\Delta\log_{10}(T,\mathrm{pH},C_{A,0})}_{\text{residual MLP}}.
    $$

    The parametric trunk is a centred Arrhenius law in temperature only,

    $$
    \log_{10} k_{\mathrm{param}}(T)
    = \frac{1}{\ln 10}\Big(
        \log k_{\mathrm{ref}} - \frac{E_a}{R}\Big(\frac{1}{T_K} - \frac{1}{T_{\mathrm{ref}}}\Big)
    \Big),
    $$

    with two trainable scalars $(\log k_{\mathrm{ref}},\,E_a)$ and
    *no pH input*. By construction it is pH-blind: at fixed $T$ it
    predicts the same rate regardless of $\mathrm{pH}$, so it cannot
    represent the saturation factor at all. The residual
    $\Delta\log_{10}$ is a 16-neuron one-hidden-layer MLP that takes
    $(T,\mathrm{pH},C_{A,0})$ and is responsible for whatever the trunk
    cannot represent — in this case, the hidden pH dependence.
    $C_{A,0}$ is fed in deliberately as a *red-herring covariate*:
    first-order kinetics depend only on $T$ and $\mathrm{pH}$, so a
    well-trained residual should learn to give it negligible influence.

    The two-phase fit nails the temperature dependence first
    (phase 1, evosax/CMA-ES on the two-scalar trunk) and then learns
    the pH residual on top of a frozen trunk
    (phase 2, optax/Adam on the MLP weights).
    """)
    return


@app.cell(hide_code=True)
def _doe_md(mo):
    mo.md(r"""
    ## Experimental design

    Eleven experiments are produced by Latin hypercube sampling over
    the box $T \in [15, 35]\,°\mathrm{C}$,
    $\mathrm{pH} \in [4.5, 7.5]$, and $C_{A,0} \in [0.75, 1.5]$. LHS
    gives near-uniform marginal coverage in all three axes from a
    small number of points, which is what lets the residual MLP
    interpolate cleanly between the sampled operating points. There
    is no hand-placed train/validation split — generalisation is
    instead verified by *leave-one-out cross-validation* further down,
    in which the entire two-phase pipeline is retrained eleven times,
    each time holding out a single experiment as the validation
    target.

    The initial concentration $C_{A,0}$ is treated as a *covariate*
    that the residual MLP receives as one of its three named inputs.
    Because the kinetics are first-order, the rate constant
    $k(T,\mathrm{pH})$ does not actually depend on $C_{A,0}$ — only
    the trajectory amplitude does, through the IC. This makes
    $C_{A,0}$ a *red-herring* covariate: the MLP sees it but should
    learn to assign it negligible influence on $\Delta\log_{10}k$.

    Each experiment runs for $t \in [0, 5]$ (dimensionless time units;
    one e-folding occurs around $t \approx 1/k$, so the trajectory is
    visibly decayed but not exhausted) with twelve evenly-spaced
    observations of $C_A$ only. $C_B = C_{A,0} - C_A$ for first-order
    $A \to B$, so observing $C_B$ would add no information.

    Heteroscedastic Gaussian noise with $\sigma = 0.03 \cdot
    \max(|C_A|, 0.02)$ is added to each observation. The variance is
    recorded on the `ChannelObs` so the framework's MLE-style losses
    could use it later if desired.

    ```python
    # 3-D LHS over (T, pH, Ca0). Ca0 is a covariate the truth doesn't depend on
    # (first-order kinetics); the residual MLP must learn to ignore it.
    sampler = qmc.LatinHypercube(d=3, seed=DOE_SEED)
    unit = sampler.random(n=N_EXPERIMENTS)                    # [11, 3] in [0, 1]
    lo = np.array([T_C_RANGE[0], PH_RANGE[0], CA0_RANGE[0]])
    hi = np.array([T_C_RANGE[1], PH_RANGE[1], CA0_RANGE[1]])
    lhs_design = [
        (float(t), float(ph), float(ca0)) for t, ph, ca0 in lo + (hi - lo) * unit
    ]

    def add_heteroscedastic_noise(values, key):
        scale = 0.03 * jnp.maximum(jnp.abs(values), 0.02)
        return jnp.clip(values + scale * jr.normal(key, values.shape), 0.0, None)
    ```
    """)
    return


@app.cell
def _doe(jnp, jr, np, qmc):
    T_C_RANGE = (15.0, 35.0)
    PH_RANGE = (4.5, 7.5)
    CA0_RANGE = (0.75, 1.5)
    N_EXPERIMENTS = 11

    T_MAX = 5.0
    N_TIMESTEPS = 12
    NOISE_REL = 0.03
    NOISE_FLOOR = 0.02

    DOE_SEED = 0
    NOISE_SEED = 1

    # 3-D LHS over (T, pH, Ca0). Ca0 has no rate effect — first-order kinetics
    # depend only on T and pH — so it acts as a "red-herring" covariate the
    # residual MLP must learn to ignore.
    sampler = qmc.LatinHypercube(d=3, seed=DOE_SEED)
    unit = sampler.random(n=N_EXPERIMENTS)
    lo = np.array([T_C_RANGE[0], PH_RANGE[0], CA0_RANGE[0]])
    hi = np.array([T_C_RANGE[1], PH_RANGE[1], CA0_RANGE[1]])
    lhs_design = [(float(t), float(ph), float(ca0)) for t, ph, ca0 in lo + (hi - lo) * unit]

    print(f"LHS samples ({N_EXPERIMENTS}):")
    for _i, (_T, _ph, _ca0) in enumerate(lhs_design):
        print(f"  [{_i:2d}] T = {_T:5.2f} °C, pH = {_ph:.3f}, Ca0 = {_ca0:.3f}")

    noise_root = jr.PRNGKey(NOISE_SEED)
    ts_global = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    return (
        N_EXPERIMENTS,
        NOISE_FLOOR,
        NOISE_REL,
        T_MAX,
        lhs_design,
        noise_root,
        ts_global,
    )


@app.cell
def _add_noise(NOISE_FLOOR, NOISE_REL, jnp, jr):
    def add_heteroscedastic_noise(values, key):
        """σ = NOISE_REL · max(|values|, NOISE_FLOOR), clipped at 0."""
        scale = NOISE_REL * jnp.maximum(jnp.abs(values), NOISE_FLOOR)
        noisy = values + scale * jr.normal(key, values.shape)
        return jnp.clip(noisy, 0.0, None)

    return (add_heteroscedastic_noise,)


@app.cell
def _true_trajectory(jnp, k_true):
    def true_ca_trajectory(ts, temperature_C, pH, ca0):
        """Closed-form Ca(t) = ca0 · exp(-k_true(T, pH) · t).

        Ca0 enters as the initial condition only; the rate constant
        depends only on (T, pH).
        """
        k = float(k_true(temperature_C, pH))
        return float(ca0) * jnp.exp(-k * jnp.asarray(ts))

    return (true_ca_trajectory,)


@app.cell(hide_code=True)
def _experiment_md(mo):
    mo.md(r"""
    ## Building the `Experiment` objects

    Each experiment is wrapped into a `hybridmodels.Experiment` that
    holds its noisy observations, the per-experiment covariates
    $(T, \mathrm{pH})$, and a function `y0_fn` that constructs the
    initial state of the ODE. The state of this system is two-dimensional,
    $y = [C_A, C_B]$, so the initial state is $y_0 = [C_{A,0},\,0]$.
    The framework computes the per-experiment union timestamp axis and
    the observation mask automatically; with a single channel observed
    at twelve evenly-spaced times, the union axis is just those twelve
    times.

    The state-to-output projector simply selects $C_A$ from the full
    state, since that is the only observed channel.

    ```python
    def y0_fn(covariates, channels):
        ca0 = jnp.asarray(channels["Ca"].values[0])
        return jnp.stack([ca0, jnp.zeros_like(ca0)])

    def state_to_output(state):
        return state[..., :1]                    # select Ca

    OUTPUT_CHANNELS = ("Ca",)

    # Build one experiment per (T, pH, Ca0) sample.
    make_experiment(
        covariates={
            "temperature_C": float(T_C),
            "pH": float(pH),
            "Ca0": float(ca0),
        },
        channels={"Ca": ChannelObs(ts=ts, values=noisy, variance=sigma**2)},
        y0_fn=y0_fn,
        exp_id=f"exp_{i:02d}_T{T_C:.1f}_pH{pH:.2f}_Ca0{ca0:.2f}",
    )

    # Stack experiments into buckets by len(union_ts); compute masks.
    dataset = make_dataset(
        experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    ```
    """)
    return


@app.cell
def _y0_fn(Array, ChannelObs, Float, jnp):
    def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 2"]:
        """[Ca, Cb] = [first observed Ca, 0]. Reads the (noisy) first observation."""
        ca0 = jnp.asarray(channels["Ca"].values[0])
        return jnp.stack([ca0, jnp.zeros_like(ca0)])

    return (y0_fn,)


@app.cell
def _state_to_output(Array, Float):
    def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
        """Project [Ca, Cb] to the observed channel [Ca]."""
        return state[..., :1]

    OUTPUT_CHANNELS = ("Ca",)
    return OUTPUT_CHANNELS, state_to_output


@app.cell
def _build_experiments(
    ChannelObs,
    Experiment,
    NOISE_FLOOR,
    NOISE_REL,
    add_heteroscedastic_noise,
    jnp,
    jr,
    lhs_design,
    make_experiment,
    noise_root,
    true_ca_trajectory,
    ts_global,
    y0_fn,
):
    def _build_one(temperature_C, pH, ca0, key, exp_id):
        clean = true_ca_trajectory(ts_global, temperature_C, pH, ca0)
        noisy = add_heteroscedastic_noise(clean, key)
        sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
        variance = sigma**2
        return make_experiment(
            covariates={
                "temperature_C": float(temperature_C),
                "pH": float(pH),
                "Ca0": float(ca0),
            },
            channels={"Ca": ChannelObs(ts=ts_global, values=noisy, variance=variance)},
            y0_fn=y0_fn,
            exp_id=exp_id,
        )

    experiments: list[Experiment] = []
    for _i, (_T, _ph, _ca0) in enumerate(lhs_design):
        _k = jr.fold_in(noise_root, _i)
        experiments.append(
            _build_one(
                _T,
                _ph,
                _ca0,
                _k,
                f"exp_{_i:02d}_T{_T:.1f}_pH{_ph:.2f}_Ca0{_ca0:.2f}",
            )
        )

    print(f"built {len(experiments)} experiments (LHS over T, pH, Ca0)")
    return (experiments,)


@app.cell
def _build_dataset(
    OUTPUT_CHANNELS,
    experiments: "list[Experiment]",
    make_dataset,
    state_to_output,
):
    dataset = make_dataset(
        experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"dataset: {len(dataset.bucket_payloads)} bucket(s)")
    for _i, _bp in enumerate(dataset.bucket_payloads):
        print(
            f"  bucket {_i}: ts={tuple(_bp.ts.shape)}, "
            f"y_observed={tuple(_bp.y_observed.shape)}, n_obs={int(_bp.n_obs)}"
        )
    return (dataset,)


@app.cell(hide_code=True)
def _raw_data_md(mo):
    mo.md(r"""
    ### Raw data

    Concentration trajectories for all eleven experiments, coloured
    by pH. The dispersion at any fixed time reflects the rate spread
    across the $(T, \mathrm{pH})$ design — at high pH the saturation
    curve flattens and the rate is small (slow decay), while at low
    pH and high temperature the decay is fast.
    """)
    return


@app.cell
def _raw_data_plot(
    experiments: "list[Experiment]",
    plt,
):
    fig_raw, ax_raw = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("viridis")
    _ph_min = min(float(e.covariates["pH"]) for e in experiments)
    _ph_max = max(float(e.covariates["pH"]) for e in experiments)

    def _color(ph):
        return cmap((ph - _ph_min) / (_ph_max - _ph_min + 1e-12))

    for _exp in experiments:
        _ts = _exp.channels["Ca"].ts
        _ca = _exp.channels["Ca"].values
        _ph = float(_exp.covariates["pH"])
        ax_raw.plot(_ts, _ca, "o-", color=_color(_ph), markersize=4, linewidth=1.0, alpha=0.85)
    ax_raw.set_xlabel("t")
    ax_raw.set_ylabel("Ca")
    ax_raw.set_title("Observed concentrations across the LHS design")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=_ph_min, vmax=_ph_max))
    plt.colorbar(sm, ax=ax_raw, label="pH")
    ax_raw.grid(alpha=0.3)
    fig_raw.tight_layout()
    fig_raw
    return


@app.cell(hide_code=True)
def _solver_md(mo):
    mo.md(r"""
    ## Solver configuration

    The dynamics are exp-decay-tame, so a stiff solver is unnecessary.
    Tsit5 with relative tolerance $10^{-5}$ and absolute tolerance
    $10^{-7}$ is more than sufficient for the first-order kinetics here;
    the same defaults are used in the harmonic-oscillator pendulum
    example. `dt0 = 0.05` gives the integrator a sensible first step
    on the unit-scale time axis.

    ```python
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.05,
    )
    ```
    """)
    return


@app.cell
def _solver(SolverConfig, diffrax):
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.05,
    )
    return (solver,)


@app.cell(hide_code=True)
def _predictor_md(mo):
    mo.md(r"""
    ## Predictors: parametric trunk + residual MLP

    The trainable component is a tuple
    `(parametric_trunk, residual_bp)` of two `eqx.Module` leaves. The
    framework's convention for the predictors pytree is "always wrap
    in a tuple"; multi-leaf tuples like this one are unpacked at the
    top of the user's vector field, exactly mirroring the
    `(growth_BP, nucleation_BP)` shape used in the crystallisation
    examples.

    ### `ArrheniusKinetics` — the parametric trunk

    A small `eqx.Module` holding two trainable scalars
    $(\log k_{\mathrm{ref}},\ E_a)$ in latent space, mapped onto
    physical bounds through a sigmoid `BoundScaler`. The trunk is
    *not* a `BoundedPredictor` — it has no covariate inputs (it
    returns the parameters; the vector field combines them with the
    per-experiment $T$). The centred form
    $k_{\mathrm{param}}(T) = \exp(\log k_{\mathrm{ref}} - (E_a/R)(1/T_K - 1/T_{\mathrm{ref}}))$
    keeps $\log k_{\mathrm{ref}}$ directly interpretable as
    $\ln k(T_{\mathrm{ref}}, \cdot)$, with tight bounds $[-3, 2]$
    (covering rate constants in $[0.05, 7.4]$); a non-centred
    Arrhenius would push $\log A$ to ${\sim}12$ and require a
    much wider, less interpretable bound.

    ### `residual_bp` — the residual MLP

    A 16-neuron one-hidden-layer MLP with ReLU activation, taking
    $(T, \mathrm{pH}, C_{A,0})$ as three named inputs and emitting a
    scalar $\Delta\log_{10} k$. Wrapped in a `BoundedPredictor` whose
    output bounds are *symmetric around zero*, $[-2, +2]$ decades.
    This matters: a freshly-initialised MLP sits near the sigmoid
    midpoint, which under symmetric output bounds is exactly $0$
    decades — i.e. the residual contributes nothing at init. Phase 2
    therefore starts at the phase-1 fit without any explicit zeroing.

    The third input slot is the red-herring axis: $C_{A,0}$ has no
    rate effect in first-order kinetics, so a well-trained residual
    will end phase 2 with a near-flat dependence on the $C_{A,0}$
    axis. The MLP's input scaler bounds it to $[0.75, 1.5]$ — the
    same range the LHS samples it over — so the network sees the
    full physical range mapped through a sigmoid into its latent
    feature space.

    ```python
    LOG_KREF_BOUNDS = (-3.0, 2.0)
    EA_BOUNDS = (0.0, 80.0)

    class ArrheniusKinetics(eqx.Module):
        latent: Float[Array, " 2"]
        out_scaler: BoundScaler

        def __init__(self, *, key):
            self.latent = jr.normal(key, (2,)) * 0.1
            self.out_scaler = BoundScaler(
                bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
                transform="sigmoid",
            )

        def __call__(self):
            return self.out_scaler.from_latent(self.latent)


    INPUT_KEYS = ("temperature_C", "pH", "Ca0")
    TEMPERATURE_BOUNDS = (0.0, 50.0)
    PH_BOUNDS = (3.0, 9.0)
    CA0_INPUT_BOUNDS = (0.75, 1.5)             # Ca0 is fed but doesn't affect the truth
    RES_LOG10_BOUNDS = (-2.0, 2.0)

    parametric_trunk = ArrheniusKinetics(key=k_param)
    residual_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=BoundScaler(
            bounds=(TEMPERATURE_BOUNDS, PH_BOUNDS, CA0_INPUT_BOUNDS),
            transform="sigmoid",
        ),
        inner=MLPPredictor(
            in_size=3, out_size=1, width_size=16, depth=1,
            activation_name="relu", key=k_residual,
        ),
        out_scaler=BoundScaler(bounds=(RES_LOG10_BOUNDS,), transform="sigmoid"),
    )
    predictors_init = (parametric_trunk, residual_bp)
    ```
    """)
    return


@app.cell
def _arrhenius_class(Array, BoundScaler, Float, eqx, jr):
    LOG_KREF_BOUNDS = (-3.0, 2.0)
    EA_BOUNDS = (0.0, 80.0)

    class ArrheniusKinetics(eqx.Module):
        """Centred-Arrhenius parametric trunk: two trainable scalars."""

        latent: Float[Array, " 2"]
        out_scaler: BoundScaler

        def __init__(self, *, key: Array) -> None:
            self.latent = jr.normal(key, (2,)) * 0.1
            self.out_scaler = BoundScaler(
                bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
                transform="sigmoid",
            )

        def __call__(self) -> Float[Array, " 2"]:
            """Return (log_k_ref, Ea) in physical units."""
            return self.out_scaler.from_latent(self.latent)

    return (ArrheniusKinetics,)


@app.cell
def _build_predictors(
    ArrheniusKinetics,
    BoundScaler,
    BoundedPredictor,
    MLPPredictor,
    jr,
):
    INPUT_KEYS = ("temperature_C", "pH", "Ca0")
    TEMPERATURE_BOUNDS = (0.0, 50.0)
    PH_BOUNDS = (3.0, 9.0)
    CA0_INPUT_BOUNDS = (0.75, 1.5)
    RES_LOG10_BOUNDS = (-2.0, 2.0)

    _root = jr.PRNGKey(0)
    _k_param, _k_residual = jr.split(_root, 2)

    parametric_trunk = ArrheniusKinetics(key=_k_param)
    residual_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=BoundScaler(
            bounds=(TEMPERATURE_BOUNDS, PH_BOUNDS, CA0_INPUT_BOUNDS),
            transform="sigmoid",
        ),
        inner=MLPPredictor(
            in_size=3,
            out_size=1,
            width_size=16,
            depth=1,
            activation_name="relu",
            key=_k_residual,
        ),
        out_scaler=BoundScaler(
            bounds=(RES_LOG10_BOUNDS,),
            transform="sigmoid",
        ),
    )
    predictors_init = (parametric_trunk, residual_bp)

    _log_k_ref0, _Ea0 = parametric_trunk()
    print(f"parametric init: log_k_ref={float(_log_k_ref0):.3f}, Ea={float(_Ea0):.2f} kJ/mol")
    return (predictors_init,)


@app.cell(hide_code=True)
def _simulate_md(mo):
    mo.md(r"""
    ## The user-supplied `simulate_fn`

    The framework requires a `simulate_fn` with a fixed signature
    that integrates one experiment to its observation timestamps and
    returns the full state trajectory. The function below implements
    the log-additive combination law

    $$
    \log_{10} k(T, \mathrm{pH}) = \log_{10} k_{\mathrm{param}}(T) + \Delta\log_{10}(T, \mathrm{pH})
    $$

    where $k_{\mathrm{param}}$ comes from the centred Arrhenius trunk
    and $\Delta\log_{10}$ from the residual MLP. The rate $k$ is
    evaluated once per simulator call (covariates are constant in
    time per the framework's per-experiment scalar contract) and
    closed over by the inner vector field. A clipping
    $C_A \mapsto \max(C_A, 0)$ in the rate term guards against rare
    negative excursions of the integrator near the asymptote; mass
    conservation is exact analytically, so this is purely numerical
    hygiene.

    ```python
    def simulate_fn(predictors, ts, covariates, y0, solver):
        parametric, residual = predictors
        log_k_ref, Ea = parametric()                         # [2] in physical units

        T_K = covariates["temperature_C"] + 273.15
        log10_k_param = (
            (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF))
            / jnp.log(10.0)
        )

        delta_log10_k = jnp.squeeze(
            residual({"temperature_C": covariates["temperature_C"],
                      "pH":            covariates["pH"],
                      "Ca0":           covariates["Ca0"]})
        )
        k = jnp.power(10.0, log10_k_param + delta_log10_k)   # log-additive combo

        def vector_field(t, y, args):
            Ca = jnp.maximum(y[0], 0.0)
            rate = k * Ca
            return jnp.stack([-rate, rate])

        return diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field), solver.solver,
            t0=ts[0], t1=ts[-1], dt0=solver.dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(
                rtol=solver.rtol, atol=solver.atol,
            ),
            max_steps=solver.max_steps,
            adjoint=diffrax.DirectAdjoint(),
        ).ys
    ```
    """)
    return


@app.cell
def _simulate_fn(
    Array,
    ArrheniusKinetics,
    BoundedPredictor,
    Float,
    R_GAS,
    SolverConfig,
    T_REF,
    diffrax,
    jnp,
):
    def simulate_fn(
        predictors: tuple[ArrheniusKinetics, BoundedPredictor],
        ts: Float[Array, " T"],
        covariates: dict[str, Array],
        y0: Float[Array, " 2"],
        solver: SolverConfig,
    ) -> Float[Array, "T 2"]:
        parametric, residual = predictors
        log_k_ref, Ea = parametric()

        T_C = covariates["temperature_C"]
        pH = covariates["pH"]
        Ca0 = covariates["Ca0"]
        T_K = T_C + 273.15

        log10_k_param = (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)

        inputs = {"temperature_C": T_C, "pH": pH, "Ca0": Ca0}
        delta_log10_k = jnp.squeeze(residual(inputs))

        log10_k = log10_k_param + delta_log10_k
        k = jnp.power(10.0, log10_k)

        def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
            Ca = jnp.maximum(y[0], 0.0)
            rate = k * Ca
            return jnp.stack([-rate, rate])

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0 if solver.dt0 is not None else 0.05,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
            max_steps=solver.max_steps,
            adjoint=diffrax.DirectAdjoint(),
        )
        return jnp.asarray(sol.ys)

    return (simulate_fn,)


@app.cell(hide_code=True)
def _phase1_md(mo):
    mo.md(r"""
    # Phase 1: evosax fits the parametric trunk

    The headline pipeline trains once on all eleven experiments — the
    purpose of that run is to expose the loss curve, the recovered
    Arrhenius parameters, and the post-fit function shape. Out-of-sample
    generalisation is verified separately by the leave-one-out
    cross-validation section further down.

    The trainability mask for phase 1 marks every leaf as trainable
    by default (`trainable_mask`), then freezes any leaf inside a
    `BoundedPredictor` (which removes the residual MLP from the
    search) and freezes any leaf inside a `BoundScaler` (the standard
    convention recommended by `CONTEXT.md` — the bound geometry
    should not drift during training). The result is a mask in which
    only the two-element `latent` of `ArrheniusKinetics` is True, so
    CMA-ES sees a two-dimensional search.

    The choice of `freeze_modules_of_type(BoundedPredictor)` over
    `freeze_paths(...)` is deliberate: the latter requires explicit
    dotted leaf paths (`"1.inner.layers.0.weight"`, …), which would be
    brittle to MLP layer naming. `freeze_modules_of_type` walks the
    pytree and zeroes the entire `BoundedPredictor` submask in one
    call.

    Initial population is drawn by Latin hypercube sampling in the
    latent box, with `init_box_extent=2.0` giving the population
    space-filling coverage and `sigma_init=0.5` setting the initial
    spread of the CMA-ES sampling distribution.

    ```python
    # Build the mask: trainable everywhere -> freeze the residual subtree
    # -> freeze every BoundScaler (the standard convention).
    mask_p1 = trainable_mask(predictors_init)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundedPredictor)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundScaler)
    # Result: only ArrheniusKinetics.latent is True (2 trainable scalars).

    config_p1 = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )
    history_p1, predictors_p1 = train_with_evosax(
        predictors_init, dataset, config_p1,
        simulate_fn=simulate_fn, solver=solver,
        trainable=mask_p1, key=jr.PRNGKey(0),
    )
    ```
    """)
    return


@app.cell
def _trainable_helper(eqx, jax):
    def count_trainable_params(predictors, mask):
        """Sum of array sizes for leaves whose mask is True."""
        pred_leaves = jax.tree_util.tree_leaves(predictors)
        mask_leaves = jax.tree_util.tree_leaves(mask)
        n = 0
        for _p, _m in zip(pred_leaves, mask_leaves, strict=True):
            if eqx.is_inexact_array(_p) and bool(_m):
                n += int(_p.size)
        return n

    return (count_trainable_params,)


@app.cell
def _phase1_mask(
    BoundScaler,
    BoundedPredictor,
    count_trainable_params,
    freeze_modules_of_type,
    predictors_init,
    trainable_mask,
):
    mask_p1 = trainable_mask(predictors_init)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundedPredictor)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundScaler)

    n_p1 = count_trainable_params(predictors_init, mask_p1)
    print(f"Phase 1 trainable scalars: {n_p1} (expected: 2 — the parametric latent)")
    return (mask_p1,)


@app.cell
def _phase1_train(
    EvosaxTrainingConfig,
    dataset,
    jr,
    mask_p1,
    predictors_init,
    simulate_fn,
    solver,
    train_with_evosax,
):
    config_p1 = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )
    history_p1, predictors_p1 = train_with_evosax(
        predictors_init,
        dataset,
        config_p1,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p1,
        key=jr.PRNGKey(0),
    )
    _log_k_ref, _Ea = predictors_p1[0]()
    print(f"phase 1: {len(history_p1)} generations, final best loss {history_p1[-1]:.6f}")
    print(f"  recovered log_k_ref = {float(_log_k_ref):.3f}")
    print(f"  recovered Ea         = {float(_Ea):.2f} kJ/mol")
    return history_p1, predictors_p1


@app.cell
def _phase1_predict(
    dataset,
    predict_dataset,
    predictors_p1,
    simulate_fn,
    solver,
):
    predictions_p1 = predict_dataset(
        predictors_p1,
        dataset,
        simulate_fn=simulate_fn,
        solver=solver,
    )
    return (predictions_p1,)


@app.cell
def _gather_diag(np):
    def gather_diagnostics(predictions, dataset):
        out: dict[str, dict] = {}
        for _d, _name in enumerate(dataset.output_channel_names):
            _obs_chunks: list = []
            _pred_chunks: list = []
            for _pred_b, _bp in zip(predictions, dataset.bucket_payloads, strict=True):
                _mask_d = np.asarray(_bp.mask[..., _d], dtype=bool)
                _obs_chunks.append(np.asarray(_bp.y_observed[..., _d])[_mask_d])
                _pred_chunks.append(np.asarray(_pred_b[..., _d])[_mask_d])
            _obs = np.concatenate(_obs_chunks)
            _pred = np.concatenate(_pred_chunks)
            _resid = _pred - _obs
            _mse = float(np.mean(_resid**2))
            _ss_tot = float(np.sum((_obs - _obs.mean()) ** 2))
            _r2 = 1.0 - float(np.sum(_resid**2)) / _ss_tot if _ss_tot > 0 else float("nan")
            out[_name] = {
                "n": int(_obs.shape[0]),
                "mse": _mse,
                "rmse": float(np.sqrt(_mse)),
                "mae": float(np.mean(np.abs(_resid))),
                "r2": _r2,
                "obs": _obs,
                "pred": _pred,
            }
        return out

    return (gather_diagnostics,)


@app.cell
def _phase1_diag(
    dataset,
    gather_diagnostics,
    predictions_p1,
):
    # In-sample fit diagnostics for the headline run. Out-of-sample
    # generalisation is reported in the LOO-CV section below.
    diag_p1 = gather_diagnostics(predictions_p1, dataset)
    print(f"  {'channel':<6} {'n':>4} {'MSE':>12} {'RMSE':>10} {'MAE':>10} {'R^2':>8}")
    for _name in dataset.output_channel_names:
        _s = diag_p1[_name]
        _r2 = "nan" if _s["r2"] != _s["r2"] else f"{_s['r2']:.4f}"
        print(
            f"  {_name:<6} {_s['n']:>4d} "
            f"{_s['mse']:>12.4e} {_s['rmse']:>10.4e} {_s['mae']:>10.4e} {_r2:>8}"
        )
    return (diag_p1,)


@app.cell
def _trajectory_grid_helper(T_MAX, jnp, np, plt, true_ca_trajectory):
    def trajectory_grid_plot(experiments, predictions_per_exp, title, predicted_label="predicted"):
        """N-panel grid; each panel shows truth, observations, and the prediction
        for one experiment. Used for both the in-sample headline run and the LOO-CV
        out-of-fold reveal — caller decides what to pass.
        """
        n = len(experiments)
        if n != len(predictions_per_exp):
            raise ValueError(
                f"experiments ({n}) and predictions ({len(predictions_per_exp)}) length mismatch"
            )
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(11, 3.0 * nrows), sharex=True, sharey=True, squeeze=False
        )
        ts_dense = np.linspace(0.0, T_MAX, 200)
        flat_axes = axes.flatten()
        for _ax, _exp, _pred in zip(flat_axes[:n], experiments, predictions_per_exp, strict=True):
            _T = float(_exp.covariates["temperature_C"])
            _ph = float(_exp.covariates["pH"])
            _ca0 = float(_exp.covariates["Ca0"])
            _clean = np.asarray(true_ca_trajectory(jnp.asarray(ts_dense), _T, _ph, _ca0))
            _ts_obs = np.asarray(_exp.channels["Ca"].ts)
            _ca_obs = np.asarray(_exp.channels["Ca"].values)
            _ax.plot(ts_dense, _clean, color="black", linewidth=1.4, label="truth")
            _ax.scatter(
                _ts_obs,
                _ca_obs,
                s=22,
                color="C0",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label="observed",
            )
            _ax.plot(
                _ts_obs,
                _pred[:, 0],
                color="C3",
                linestyle="--",
                linewidth=1.4,
                label=predicted_label,
            )
            _ax.set_title(f"T={_T:.1f}°C, pH={_ph:.2f}, Ca0={_ca0:.2f}", fontsize=9)
            _ax.set_ylim(-0.05, 1.6)
            _ax.grid(alpha=0.3)
        for _ax in flat_axes[n:]:
            _ax.set_visible(False)
        for _ax in axes[-1]:
            _ax.set_xlabel("t")
        for _ax in axes[:, 0]:
            _ax.set_ylabel("Ca")
        _handles, _labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            _handles, _labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99)
        )
        fig.suptitle(title, y=0.995, fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return fig

    return (trajectory_grid_plot,)


@app.cell
def _kreveal_helper(R_GAS, T_REF, jnp, k_true, np, plt):
    def k_reveal_plot(predictors_p1, predictors_p2, experiments, title):
        """Plot log10 k(pH) at three fixed temperatures: truth (solid), the
        parametric trunk's flat-in-pH prediction (dashed), and — if a hybrid
        model is supplied — its log10 k curve (dotted). LHS sample points are
        overlaid at their (pH, log10 k_true) location to anchor the reveal.
        """
        # pH grid spanning slightly beyond the data box for visual context.
        pH_grid = jnp.linspace(4.0, 8.0, 200)
        T_C_lines = (15.0, 25.0, 35.0)
        colors = ("C0", "C1", "C2")
        # The residual MLP takes Ca0 as a third input; the truth is independent of it,
        # so we evaluate the hybrid at the midpoint of the Ca0 input box for the reveal.
        ca0_eval = jnp.asarray(1.125)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        parametric_p1, _ = predictors_p1
        parametric_p2, residual_p2 = predictors_p2 if predictors_p2 is not None else (None, None)

        for _T_C, _color in zip(T_C_lines, colors, strict=True):
            _log10_truth = np.log10(np.asarray(k_true(_T_C, pH_grid)))
            ax.plot(
                pH_grid, _log10_truth, color=_color, linewidth=1.6, label=f"truth (T={_T_C:.0f}°C)"
            )

            _log_k_ref, _Ea = parametric_p1()
            _T_K = _T_C + 273.15
            _log10_param = float(
                (_log_k_ref - _Ea / R_GAS * (1.0 / _T_K - 1.0 / T_REF)) / jnp.log(10.0)
            )
            ax.plot(
                pH_grid,
                np.full_like(pH_grid, _log10_param),
                color=_color,
                linewidth=1.2,
                linestyle="--",
                label=("parametric" if _T_C == T_C_lines[1] else None),
            )

            if predictors_p2 is not None and parametric_p2 is not None and residual_p2 is not None:
                _log_k_ref2, _Ea2 = parametric_p2()
                _log10_param2 = float(
                    (_log_k_ref2 - _Ea2 / R_GAS * (1.0 / _T_K - 1.0 / T_REF)) / jnp.log(10.0)
                )
                _delta = np.array(
                    [
                        float(
                            jnp.squeeze(
                                residual_p2(
                                    {
                                        "temperature_C": jnp.asarray(_T_C),
                                        "pH": jnp.asarray(_ph),
                                        "Ca0": ca0_eval,
                                    }
                                )
                            )
                        )
                        for _ph in pH_grid
                    ]
                )
                _log10_hybrid = _log10_param2 + _delta
                ax.plot(
                    pH_grid,
                    _log10_hybrid,
                    color=_color,
                    linewidth=1.2,
                    linestyle=":",
                    label=("hybrid" if _T_C == T_C_lines[1] else None),
                )

        _exp_T = np.array([float(_e.covariates["temperature_C"]) for _e in experiments])
        _exp_pH = np.array([float(_e.covariates["pH"]) for _e in experiments])
        _exp_log10k = np.log10(np.asarray(k_true(_exp_T, _exp_pH)))
        ax.scatter(
            _exp_pH,
            _exp_log10k,
            s=44,
            color="black",
            marker="o",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label="LHS samples (truth)",
        )
        ax.set_xlabel("pH")
        ax.set_ylabel("log10 k")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
        fig.tight_layout()
        return fig

    return (k_reveal_plot,)


@app.cell(hide_code=True)
def _phase2_md(mo):
    mo.md(r"""
    # Phase 2: optax fits the residual MLP

    The trainability mask flips. `freeze_modules_of_type(predictors_p1, ArrheniusKinetics)`
    zeroes the entire trunk submask, so the parametric scalars stay
    fixed at their phase-1 endpoint. The residual MLP becomes the only
    trainable component. `BoundScaler` leaves stay frozen by
    convention (they define bound geometry, not learnable weights).

    Because the residual MLP's output bounds are symmetric around
    zero and a freshly-initialised MLP sits near the sigmoid
    midpoint, the MLP at phase-1 endpoint contributes ${\sim}0$
    decades of correction. The phase-1 final loss therefore equals
    the phase-2 step-0 loss to within numerical noise — the seam
    between the two phases is invisible in the loss curve.

    Optax uses AdamW with learning rate $3 \times 10^{-3}$ for 200
    steps. Training one step here means one full pass over every
    bucket (there is only one bucket of size 11), accumulating
    gradients and applying a single optimiser update.

    ```python
    # The mask flips: freeze ArrheniusKinetics, leave the residual MLP trainable.
    mask_p2 = trainable_mask(predictors_p1)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, ArrheniusKinetics)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, BoundScaler)
    # Result: 65 True leaves (the 16-neuron MLP weights and biases).

    config_p2 = OptaxTrainingConfig(
        steps=(200,),
        lr=(3e-3,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=False,
    )
    history_p2, predictors_p2 = train_with_optax(
        predictors_p1, dataset, config_p2,
        simulate_fn=simulate_fn, solver=solver,
        trainable=mask_p2, key=jr.PRNGKey(1),
    )
    ```
    """)
    return


@app.cell
def _phase2_mask(
    ArrheniusKinetics,
    BoundScaler,
    count_trainable_params,
    freeze_modules_of_type,
    predictors_p1,
    trainable_mask,
):
    mask_p2 = trainable_mask(predictors_p1)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, ArrheniusKinetics)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, BoundScaler)

    n_p2 = count_trainable_params(predictors_p1, mask_p2)
    print(f"Phase 2 trainable scalars: {n_p2}  (the 16-neuron MLP weights and biases)")
    return (mask_p2,)


@app.cell
def _phase2_train(
    OptaxTrainingConfig,
    dataset,
    jr,
    mask_p2,
    predictors_p1,
    simulate_fn,
    solver,
    train_with_optax,
):
    config_p2 = OptaxTrainingConfig(
        steps=(200,),
        lr=(3e-3,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=False,
    )
    history_p2, predictors_p2 = train_with_optax(
        predictors_p1,
        dataset,
        config_p2,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p2,
        key=jr.PRNGKey(1),
    )
    print(f"phase 2: {len(history_p2)} steps, final loss {history_p2[-1]:.6f}")
    return history_p2, predictors_p2


@app.cell
def _phase2_predict(
    dataset,
    predict_dataset,
    predictors_p2,
    simulate_fn,
    solver,
):
    predictions_p2 = predict_dataset(
        predictors_p2,
        dataset,
        simulate_fn=simulate_fn,
        solver=solver,
    )
    return (predictions_p2,)


@app.cell
def _phase2_diag(
    dataset,
    gather_diagnostics,
    predictions_p2,
):
    # In-sample fit diagnostics for the headline run. Out-of-sample
    # generalisation is reported in the LOO-CV section below.
    diag_p2 = gather_diagnostics(predictions_p2, dataset)
    print(f"  {'channel':<6} {'n':>4} {'MSE':>12} {'RMSE':>10} {'MAE':>10} {'R^2':>8}")
    for _name in dataset.output_channel_names:
        _s = diag_p2[_name]
        _r2 = "nan" if _s["r2"] != _s["r2"] else f"{_s['r2']:.4f}"
        print(
            f"  {_name:<6} {_s['n']:>4d} "
            f"{_s['mse']:>12.4e} {_s['rmse']:>10.4e} {_s['mae']:>10.4e} {_r2:>8}"
        )
    return (diag_p2,)


@app.cell(hide_code=True)
def _phase2_kreveal_md(mo):
    mo.md(r"""
    ### Post-fit $\log_{10} k$ reveal

    The headline diagnostic for the hybrid pipeline. Each coloured
    curve plots $\log_{10} k(\mathrm{pH})$ at a fixed temperature
    (15 °C, 25 °C, 35 °C): solid for the hidden truth, dashed for
    the parametric trunk's prediction (flat in pH by construction),
    and dotted for the full hybrid (parametric + residual MLP). The
    hybrid (dotted) tracks the truth (solid) closely, while the
    parametric (dashed) stays flat. The residual MLP has recovered
    the saturation shape of the hidden $k_{\mathrm{sat}}(\mathrm{pH})$
    curve from concentration data alone, without ever seeing the
    rate constant directly. The black markers show each LHS sample
    at its true $(\mathrm{pH}, \log_{10} k_{\mathrm{true}})$.
    """)
    return


@app.cell
def _phase2_kreveal_plot(
    experiments: "list[Experiment]",
    k_reveal_plot,
    predictors_p1,
    predictors_p2,
):
    fig_kreveal_p2 = k_reveal_plot(
        predictors_p1,
        predictors_p2,
        experiments,
        title="log10 k(pH) — truth vs parametric vs hybrid (Phase 2)",
    )
    fig_kreveal_p2
    return


@app.cell(hide_code=True)
def _loss_md(mo):
    mo.md(r"""
    ## Combined loss curve

    The two histories joined end-to-end on a logarithmic vertical
    axis. The phase boundary is marked by a vertical separator. The
    seamless transition (no jump in loss between the last evosax
    generation and the first optax step) is a property of the
    log-additive combination law combined with the symmetric residual
    output bound: a freshly-initialised residual contributes zero
    decades of correction at phase 2's first step.
    """)
    return


@app.cell
def _loss_curve(history_p1, history_p2, np, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(8.5, 4.0))
    n1 = len(history_p1)
    n2 = len(history_p2)
    x1 = np.arange(n1)
    x2 = np.arange(n1, n1 + n2)
    ax_loss.plot(
        x1,
        np.log10(np.asarray(history_p1)),
        color="C0",
        linewidth=1.4,
        label="phase 1 — evosax (best-of-gen)",
    )
    ax_loss.plot(
        x2,
        np.log10(np.asarray(history_p2)),
        color="C3",
        linewidth=1.4,
        label="phase 2 — optax (per-step)",
    )
    ax_loss.axvline(n1 - 0.5, color="gray", linestyle=":", linewidth=1.0)
    ax_loss.set_xlabel("training progress (generation, then step)")
    ax_loss.set_ylabel("log10 MSE loss")
    ax_loss.set_title("Loss curve across the evosax → optax pipeline")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best", frameon=False)
    fig_loss.tight_layout()
    fig_loss
    return


@app.cell(hide_code=True)
def _loocv_md(mo):
    mo.md(r"""
    # Leave-one-out cross-validation

    The headline run was trained on all eleven experiments. To verify
    the hybrid model actually generalises, we now retrain the entire
    two-phase pipeline eleven times — each time holding out a
    different experiment and predicting it from the model fit on the
    other ten. The aggregated out-of-fold predictions form the
    validation set: every experiment is predicted exactly once by a
    model that never saw it during training.

    This is heavy — eleven full evosax + optax cycles, each with the
    same population size and step budget as the headline run — but
    the resulting OOF parity, trajectory grid, and per-fold loss
    table are the most informative generalisation diagnostic this
    small example can produce. The configs are identical to the
    headline run; per-fold seeds are derived from the fold index so
    folds remain reproducible across reruns.
    """)
    return


@app.cell
def _loocv_run(
    EvosaxTrainingConfig,
    OUTPUT_CHANNELS,
    OptaxTrainingConfig,
    experiments,
    gather_diagnostics,
    jr,
    make_dataset,
    mask_p1,
    mask_p2,
    np,
    predict_dataset,
    predictors_init,
    simulate_fn,
    solver,
    state_to_output,
    train_with_evosax,
    train_with_optax,
):
    _cfg_p1 = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )
    _cfg_p2 = OptaxTrainingConfig(
        steps=(200,),
        lr=(3e-3,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=False,
    )

    _n_folds = len(experiments)
    fold_records: list[dict] = []
    print(f"LOO-CV: training {_n_folds} folds (phase 1 evosax + phase 2 optax each)")

    for _k in range(_n_folds):
        _train_exps = [_e for _i, _e in enumerate(experiments) if _i != _k]
        _train_ds = make_dataset(
            _train_exps,
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
        )
        _held_ds = make_dataset(
            [experiments[_k]],
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
        )

        # Phase 1 — refit the parametric trunk on the 10 in-fold experiments.
        # mask_p1 is structural so it is reused across folds without rebuilding.
        _hist_p1, _pred_p1 = train_with_evosax(
            predictors_init,
            _train_ds,
            _cfg_p1,
            simulate_fn=simulate_fn,
            solver=solver,
            trainable=mask_p1,
            key=jr.PRNGKey(_k),
        )
        # Phase 2 — refit the residual MLP on the same 10 experiments, frozen trunk.
        _hist_p2, _pred_p2 = train_with_optax(
            _pred_p1,
            _train_ds,
            _cfg_p2,
            simulate_fn=simulate_fn,
            solver=solver,
            trainable=mask_p2,
            key=jr.PRNGKey(1000 + _k),
        )

        # Out-of-fold prediction: simulate the held-out experiment under the fold's predictors.
        _oof_pred = predict_dataset(
            _pred_p2,
            _held_ds,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        _diag = gather_diagnostics(_oof_pred, _held_ds)
        _ch = next(iter(_diag))
        _oof_traj = np.asarray(_oof_pred[0][0])  # [T, n_channels] — for the trajectory grid

        _e = experiments[_k]
        fold_records.append(
            {
                "k": _k,
                "exp": _e,
                "history_p1_final": float(_hist_p1[-1]),
                "history_p2_final": float(_hist_p2[-1]),
                "oof_obs": _diag[_ch]["obs"],
                "oof_pred": _diag[_ch]["pred"],
                "oof_traj": _oof_traj,
                "oof_mse": _diag[_ch]["mse"],
            }
        )
        print(
            f"  fold {_k:2d}: T={float(_e.covariates['temperature_C']):5.2f}°C "
            f"pH={float(_e.covariates['pH']):.3f} "
            f"Ca0={float(_e.covariates['Ca0']):.3f} | "
            f"p1 final={float(_hist_p1[-1]):.3e}, "
            f"p2 final={float(_hist_p2[-1]):.3e}, "
            f"OOF MSE={_diag[_ch]['mse']:.3e}"
        )

    _all_mse = np.array([_r["oof_mse"] for _r in fold_records])
    print(
        f"  aggregate: mean OOF MSE={float(_all_mse.mean()):.3e}, "
        f"median={float(np.median(_all_mse)):.3e}, "
        f"max={float(_all_mse.max()):.3e}"
    )
    return (fold_records,)


@app.cell(hide_code=True)
def _loocv_parity_md(mo):
    mo.md(r"""
    ### LOO-CV parity

    Predicted vs observed $C_A$ for *every* observation across all
    eleven folds, where each prediction comes from a model that
    never saw the corresponding experiment during training. With
    one bucket of twelve timestamps per fold, this is 132 OOF
    points — a fair test of generalisation.
    """)
    return


@app.cell
def _loocv_parity_plot(fold_records, np, plt):
    _obs_all = np.concatenate([_r["oof_obs"] for _r in fold_records])
    _pred_all = np.concatenate([_r["oof_pred"] for _r in fold_records])
    _resid = _pred_all - _obs_all
    _mse = float(np.mean(_resid**2))
    _ss_tot = float(np.sum((_obs_all - _obs_all.mean()) ** 2))
    _r2 = 1.0 - float(np.sum(_resid**2)) / _ss_tot if _ss_tot > 0 else float("nan")

    fig_oof_par, ax_oof_par = plt.subplots(figsize=(5.5, 5.0))
    ax_oof_par.scatter(
        _obs_all,
        _pred_all,
        s=24,
        alpha=0.7,
        color="C0",
        label=f"OOF (n={len(_obs_all)})",
    )
    _lo = float(min(_obs_all.min(), _pred_all.min()))
    _hi = float(max(_obs_all.max(), _pred_all.max()))
    if _lo == _hi:
        _pad = 0.1 if _lo == 0.0 else abs(_lo) * 0.1
        _lo, _hi = _lo - _pad, _hi + _pad
    ax_oof_par.plot([_lo, _hi], [_lo, _hi], color="black", linestyle="--", linewidth=0.8)
    _r2_str = "nan" if _r2 != _r2 else f"{_r2:.4f}"
    ax_oof_par.set_title(f"LOO-CV parity (Ca)\nOOF R²={_r2_str}, MSE={_mse:.3e}")
    ax_oof_par.set_xlabel("observed")
    ax_oof_par.set_ylabel("predicted (out-of-fold)")
    ax_oof_par.legend(loc="best", fontsize=9)
    ax_oof_par.grid(alpha=0.3)
    fig_oof_par.tight_layout()
    fig_oof_par
    return


@app.cell(hide_code=True)
def _loocv_traj_md(mo):
    mo.md(r"""
    ### LOO-CV trajectories

    One panel per fold: the held-out experiment's noisy observations
    (cyan markers), the noiseless truth (solid black), and the model's
    out-of-fold prediction (dashed red). The dashed curves should
    track the truth at every $(T,\mathrm{pH},C_{A,0})$ — the fact
    that they do, at points the model never trained on, is the
    direct evidence that the residual MLP has learned the underlying
    pH dependence rather than memorising the LHS samples.
    """)
    return


@app.cell
def _loocv_traj_plot(fold_records, trajectory_grid_plot):
    _exps = [_r["exp"] for _r in fold_records]
    _per_exp = [_r["oof_traj"] for _r in fold_records]
    fig_oof_traj = trajectory_grid_plot(
        _exps,
        _per_exp,
        title="LOO-CV out-of-fold trajectories — held-out experiment per panel",
        predicted_label="OOF prediction",
    )
    fig_oof_traj
    return


@app.cell(hide_code=True)
def _outro(mo):
    mo.md(r"""
    ## Take-aways

    Three properties of the framework get a workout in this
    walkthrough that the existing examples don't fully exercise:

    1. **The trainability mask flips between phases.** The same
       predictors pytree drives both training calls. A boolean mask
       of the same pytree shape selects which subtree is trainable
       in each phase, built compositionally with `trainable_mask`
       and `freeze_modules_of_type`. Adding new freeze behaviour is
       a function, never a class.
    2. **The predictors pytree convention scales naturally to
       multi-leaf hybrid models.** A tuple `(parametric_trunk, residual_bp)`
       of two heterogeneous `eqx.Module` leaves works exactly like
       the `(growth_BP, nucleation_BP)` tuple in the crystallisation
       example — `eqx.partition`, `eqx.filter_value_and_grad`, and
       `eqx.tree_serialise_leaves` all walk the leaves uniformly,
       and the user's vector field unpacks the tuple at the top.
    3. **The global-search-then-gradient-polish pattern composes
       cleanly.** Phase 1 escapes the basin in two dimensions, phase
       2 polishes a sixty-five-dimensional residual on top, and the
       seam between the two phases is invisible in the loss curve.
       This is the textbook reason to compose evosax and optax —
       neither alone would do as well, and the framework lets the
       user write the composition explicitly without any
       polishing-from-config opaque magic.

    Generalisation is verified by the leave-one-out
    cross-validation section — the OOF parity tracks the diagonal
    and every held-out experiment is predicted to within
    observation-noise scale, including pH values the model never
    trained on.
    """)
    return


if __name__ == "__main__":
    app.run()
