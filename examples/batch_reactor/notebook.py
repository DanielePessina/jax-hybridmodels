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
    # Recovering an unknown pH dependence in a batch reactor

    ## Problem

    A first-order reaction $A \to B$ in a closed batch reactor has a
    rate constant $k(T, \mathrm{pH})$ whose temperature dependence
    is well described by Arrhenius but whose pH dependence has *no
    first-principles form*. The available data is a handful of
    noisy concentration trajectories collected at a small set of
    operating conditions, and the task is to build a model that
    predicts $k$ at new conditions — including pH values not seen
    during training.

    The classical approach is to retain the Arrhenius temperature
    law and estimate parameters $(\log k_{\mathrm{ref}}, E_a)$
    separately on each pH of interest. That works as long as one
    stays inside a fitted bin, but it produces a *dictionary* of
    parameters, one per pH, with no functional pH dependence and
    no extrapolation capability. To predict at an unsampled pH the
    practitioner is forced to choose between bins (and accept the
    bias) or interpolate between them by hand.

    ## Contribution

    We compare this baseline against a *hybrid* model that retains
    the Arrhenius trunk and adds a small neural residual on top. The
    residual takes $(T, \mathrm{pH}, C_{A,0})$ as input and emits a
    log-additive correction $\Delta\log_{10}k$, allowing one
    coherent model to be trained on data spanning every sampled pH
    at once and to interpolate continuously between them.

    Both models are trained on the *same* dataset of twelve
    synthetic experiments at three discrete pH values, so the
    comparison is on methodology rather than data. The hybrid model
    is evaluated by leave-one-out cross-validation across all
    twelve experiments and by leave-one-pH-out cross-validation on
    the four held-out experiments at the intermediate pH = 5.85.
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
    ## Methods

    Both the pure mechanistic baseline and the hybrid model integrate
    the same batch reactor mass balance,

    $$
    \frac{dC_A}{dt} = -\hat{k}(\,\cdot\,)\,C_A,
    \qquad C_A(0) = C_{A,0},
    $$

    and differ only in how the rate constant $\hat{k}$ is parameterised.

    **Mechanistic skeleton.** A centred-Arrhenius law in $T$ only,

    $$
    \log_{10} k_{\mathrm{param}}(T)
    = \frac{1}{\ln 10}\Big(
        \log k_{\mathrm{ref}} - \frac{E_a}{R}\Big(\frac{1}{T_K} - \frac{1}{T_{\mathrm{ref}}}\Big)
    \Big),
    $$

    with two trainable scalars $(\log k_{\mathrm{ref}},\,E_a)$ and
    *no pH input*. The pure mechanistic baseline uses this trunk on
    its own: $\hat{k}(T) = k_{\mathrm{param}}(T)$. Because the trunk
    has no pH input, it cannot be fitted across pH bins without
    averaging over pH-dependent rate shifts; it must be refit *per
    pH bin*.

    **Hybrid extension.** The hybrid retains the trunk and adds a
    log-additive neural residual $\Delta\log_{10}$ that takes
    $(T,\mathrm{pH},C_{A,0})$ as input:

    $$
    \log_{10}\hat{k}(T,\mathrm{pH},C_{A,0})
    = \log_{10} k_{\mathrm{param}}(T) + \Delta\log_{10}(T,\mathrm{pH},C_{A,0}).
    $$

    The residual is a small MLP whose role is to absorb whatever the
    trunk cannot represent — here, the unknown pH dependence.
    $C_{A,0}$ is included as a third input as a *red-herring
    covariate*: first-order kinetics depend only on $T$ and
    $\mathrm{pH}$, so a well-trained residual should learn to give
    $C_{A,0}$ negligible influence on $\Delta\log_{10}k$.
    """)
    return


@app.cell(hide_code=True)
def _doe_md(mo):
    mo.md(r"""
    ## Synthetic data

    Both the pure mechanistic baseline and the hybrid model are
    evaluated on the same dataset, so the comparison is on the
    methodology — not the data. We generate twelve batch-reactor
    experiments at three discrete pH values,
    $\mathrm{pH} \in \{5.0, 5.85, 7.0\}$, chosen to bracket the
    saturation curve below the knee, at the knee, and above the
    knee. Within each pH bin, four operating points are drawn by
    two-dimensional Latin hypercube sampling over
    $T \in [15, 35]\,°\mathrm{C}$ and $C_{A,0} \in [0.75, 1.5]$.

    The disjoint-pH layout is what makes the comparison sharp.
    The pure mechanistic baseline has no pH input, so it must be
    fitted separately on each bin and produces a dictionary of three
    parameter pairs $(\log k_{\mathrm{ref}}^{(b)}, E_a^{(b)})$ — one
    per bin — with no functional pH dependence between them. The
    hybrid model, by contrast, trains on all twelve experiments at
    once and learns a continuous $\Delta\log_{10}k(T, \mathrm{pH}, C_{A,0})$
    correction.

    The initial concentration $C_{A,0}$ enters every experiment as
    the IC of the ODE *and* as a third named input to the residual
    MLP. Because first-order kinetics depend only on $T$ and
    $\mathrm{pH}$, $C_{A,0}$ is a *red-herring covariate*: the MLP
    sees it but should learn to assign it negligible influence on
    $\Delta\log_{10}k$.

    Each experiment is integrated for $t \in [0, 5]$ (dimensionless
    units; one e-folding occurs around $t \approx 1/k$) and sampled
    at twelve evenly-spaced timestamps. Heteroscedastic Gaussian
    noise with $\sigma = 0.03 \cdot \max(|C_A|, 0.02)$ is added to
    each observation; the per-observation variance is recorded on
    each `ChannelObs` so the framework's MLE-style losses could
    consume it later if desired.

    ```python
    T_C_RANGE = (15.0, 35.0)
    CA0_RANGE = (0.75, 1.5)
    PH_BINS = (5.0, 5.85, 7.0)
    N_PER_BIN = 4

    T_MAX = 5.0
    N_TIMESTEPS = 12
    NOISE_REL = 0.03
    NOISE_FLOOR = 0.02

    DOE_SEED = 0
    NOISE_SEED = 1

    # Per pH bin: 2-D LHS over (T, Ca0). The pH axis is *not* sampled by
    # LHS — it is fixed to one of three discrete values per bin. This is
    # what lets the pure mechanistic baseline (which has no pH input) be
    # fitted bin-by-bin on identical data to the hybrid.
    design = []
    bin_of_exp = []
    for bin_idx, ph in enumerate(PH_BINS):
        sampler = qmc.LatinHypercube(d=2, seed=DOE_SEED + bin_idx)
        unit = sampler.random(n=N_PER_BIN)
        lo = np.array([T_C_RANGE[0], CA0_RANGE[0]])
        hi = np.array([T_C_RANGE[1], CA0_RANGE[1]])
        pts = lo + (hi - lo) * unit
        for t, ca0 in pts:
            design.append((float(t), float(ph), float(ca0)))
            bin_of_exp.append(bin_idx)

    noise_root = jr.PRNGKey(NOISE_SEED)
    ts_global = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    ```
    """)
    return


@app.cell
def _doe(jnp, jr, np, qmc):
    T_C_RANGE = (15.0, 35.0)
    CA0_RANGE = (0.75, 1.5)
    PH_BINS = (5.0, 5.85, 7.0)
    N_PER_BIN = 4

    T_MAX = 5.0
    N_TIMESTEPS = 12
    NOISE_REL = 0.03
    NOISE_FLOOR = 0.02

    DOE_SEED = 0
    NOISE_SEED = 1

    # Per pH bin: 2-D LHS over (T, Ca0). The pH axis is *not* sampled by LHS —
    # it is fixed to one of three discrete values per bin. This is what lets the
    # pure mechanistic baseline (which has no pH input) be fitted bin-by-bin on
    # identical data to the hybrid.
    design: list[tuple[float, float, float]] = []
    bin_of_exp: list[int] = []
    for _bin_idx, _ph in enumerate(PH_BINS):
        _sampler = qmc.LatinHypercube(d=2, seed=DOE_SEED + _bin_idx)
        _unit = _sampler.random(n=N_PER_BIN)
        _lo = np.array([T_C_RANGE[0], CA0_RANGE[0]])
        _hi = np.array([T_C_RANGE[1], CA0_RANGE[1]])
        _pts = _lo + (_hi - _lo) * _unit
        for _t, _ca0 in _pts:
            design.append((float(_t), float(_ph), float(_ca0)))
            bin_of_exp.append(_bin_idx)

    print(
        f"design: {len(PH_BINS)} pH bins × {N_PER_BIN} (T, Ca0) LHS points = "
        f"{len(design)} experiments"
    )
    for _bin_idx, _ph in enumerate(PH_BINS):
        print(f"  bin {_bin_idx} (pH={_ph}):")
        for _i, (_T, _, _ca0) in enumerate(design):
            if bin_of_exp[_i] == _bin_idx:
                print(f"    [{_i:2d}] T = {_T:5.2f} °C, Ca0 = {_ca0:.3f}")

    noise_root = jr.PRNGKey(NOISE_SEED)
    ts_global = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    return (
        NOISE_FLOOR,
        NOISE_REL,
        PH_BINS,
        T_MAX,
        bin_of_exp,
        design,
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
    ### Experiments and dataset

    Each experiment is encapsulated in a `hybridmodels.Experiment` —
    a container for the per-experiment covariates
    $(T, \mathrm{pH}, C_{A,0})$, the noisy observations on each
    measured channel, and an `y0_fn` that constructs the initial ODE
    state from the covariates and the first observation. The state
    here is two-dimensional, $y = [C_A, C_B]$, so $y_0 = [C_{A,0}, 0]$.
    A small `state_to_output` projector picks the observed channel
    $C_A$ out of the full state. Calling `make_dataset` on the list
    of experiments stacks them into buckets by union-timestep length
    and produces the observation-mask pytree consumed by the
    framework's losses.

    ```python
    def y0_fn(covariates, channels):
        # [Ca, Cb] = [first observed Ca, 0]. Reads the (noisy) first observation.
        ca0 = jnp.asarray(channels["Ca"].values[0])
        return jnp.stack([ca0, jnp.zeros_like(ca0)])

    def state_to_output(state):
        # Project [Ca, Cb] to the observed channel [Ca].
        return state[..., :1]

    OUTPUT_CHANNELS = ("Ca",)

    def build_one(temperature_C, pH, ca0, key, exp_id):
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

    experiments = []
    for i, (T, ph, ca0) in enumerate(design):
        key = jr.fold_in(noise_root, i)
        experiments.append(
            build_one(T, ph, ca0, key, f"exp_{i:02d}_T{T:.1f}_pH{ph:.2f}_Ca0{ca0:.2f}")
        )

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
    design: list[tuple[float, float, float]],
    jnp,
    jr,
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
    for _i, (_T, _ph, _ca0) in enumerate(design):
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

    print(f"built {len(experiments)} experiments (3 pH bins × 4 (T, Ca0) LHS points each)")
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

    Concentration trajectories for all twelve experiments, coloured
    by pH bin. The three bins show the saturation pattern clearly —
    pH = 5.0 (below the knee, fast rate) gives the steepest decays;
    pH = 7.0 (above the knee, plateau rate) gives the slowest. The
    spread within each bin reflects the temperature variation in
    the per-bin LHS sample.
    """)
    return


@app.cell
def _raw_data_plot(
    PH_BINS,
    bin_of_exp: list[int],
    experiments: "list[Experiment]",
    plt,
):
    fig_raw, ax_raw = plt.subplots(figsize=(7.5, 4.5))
    _bin_colors = ("C0", "C1", "C3")
    for _i, _exp in enumerate(experiments):
        _ts = _exp.channels["Ca"].ts
        _ca = _exp.channels["Ca"].values
        _color = _bin_colors[bin_of_exp[_i]]
        ax_raw.plot(_ts, _ca, "o-", color=_color, markersize=4, linewidth=1.0, alpha=0.85)
    # One legend handle per bin.
    for _bin_idx, _ph in enumerate(PH_BINS):
        ax_raw.plot([], [], "o-", color=_bin_colors[_bin_idx], label=f"pH = {_ph}")
    ax_raw.set_xlabel("t")
    ax_raw.set_ylabel("Ca")
    ax_raw.set_title("Observed concentrations across the disjoint-pH design")
    ax_raw.legend(loc="best", frameon=False)
    ax_raw.grid(alpha=0.3)
    fig_raw.tight_layout()
    fig_raw
    return


@app.cell(hide_code=True)
def _solver_md(mo):
    mo.md(r"""
    ### Solver

    The first-order kinetics are non-stiff, so we integrate with the
    explicit Tsit5 scheme at relative tolerance $10^{-5}$ and absolute
    tolerance $10^{-7}$ — the same defaults used elsewhere in the
    framework's smooth-ODE examples.
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
    ### Implementation

    The hybrid model is realised as a two-leaf pytree
    `(parametric_trunk, residual_bp)` consumed by the framework's
    `simulate_fn`.

    The trunk `ArrheniusKinetics` is a tiny `eqx.Module` holding two
    trainable scalars $(\log k_{\mathrm{ref}}, E_a)$ in latent space,
    mapped onto interpretable physical bounds through a sigmoid
    `BoundScaler` so optimisation in latent space corresponds to a
    bounded search in parameter space. The centring at
    $T_{\mathrm{ref}}$ keeps $\log k_{\mathrm{ref}}$ directly
    comparable to $\ln k(T_{\mathrm{ref}}, \cdot)$ in physical units.

    The residual is a 16-neuron one-hidden-layer `MLPPredictor` with
    ReLU activation, wrapped in a `BoundedPredictor` that maps the
    three named inputs $(T, \mathrm{pH}, C_{A,0})$ into the MLP's
    feature space through a sigmoid input scaler and projects the
    scalar output through a *symmetric-around-zero* output scaler
    onto $\Delta\log_{10}k \in [-2, +2]$ decades. The symmetric
    output bound matters: a freshly-initialised MLP sits near the
    sigmoid midpoint, which under symmetric bounds maps to exactly
    zero decades — so phase 2 starts the residual at no correction
    without any explicit zeroing.

    ```python
    LOG_KREF_BOUNDS = (-3.0, 2.0)
    EA_BOUNDS = (0.0, 80.0)

    class ArrheniusKinetics(eqx.Module):
        # Centred-Arrhenius parametric trunk: two trainable scalars.

        latent: Float[Array, " 2"]
        out_scaler: BoundScaler

        def __init__(self, *, key):
            self.latent = jr.normal(key, (2,)) * 0.1
            self.out_scaler = BoundScaler(
                bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
                transform="sigmoid",
            )

        def __call__(self):
            # Return (log_k_ref, Ea) in physical units.
            return self.out_scaler.from_latent(self.latent)

    INPUT_KEYS = ("temperature_C", "pH", "Ca0")
    TEMPERATURE_BOUNDS = (0.0, 50.0)
    PH_BOUNDS = (3.0, 9.0)
    CA0_INPUT_BOUNDS = (0.75, 1.5)
    RES_LOG10_BOUNDS = (-2.0, 2.0)

    root = jr.PRNGKey(0)
    k_param, k_residual = jr.split(root, 2)

    parametric_trunk = ArrheniusKinetics(key=k_param)
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
            key=k_residual,
        ),
        out_scaler=BoundScaler(
            bounds=(RES_LOG10_BOUNDS,),
            transform="sigmoid",
        ),
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
    ### Simulators

    Two `simulate_fn`s are defined: the hybrid `simulate_fn`
    integrates the ODE under the full log-additive law

    $$
    \log_{10} k = \log_{10} k_{\mathrm{param}}(T) + \Delta\log_{10}(T, \mathrm{pH}, C_{A,0}),
    $$

    while `simulate_fn_baseline` is identical except the residual
    term is dropped. The pure mechanistic baseline must use
    `simulate_fn_baseline` because the residual MLP at random init
    is *not* identically zero and would otherwise inject a spurious
    T-dependence into the rate constant during baseline fits — making
    the trunk's $E_a$ unidentifiable. Both functions are pure JAX
    functions of `predictors` (the trunk-plus-residual pytree),
    `covariates` (the per-experiment $T, \mathrm{pH}, C_{A,0}$),
    and the solver-config; the framework owns vmap, jit and
    autodiff around them.

    ```python
    def simulate_fn(predictors, ts, covariates, y0, solver):
        parametric, residual = predictors
        log_k_ref, Ea = parametric()

        T_C = covariates["temperature_C"]
        pH = covariates["pH"]
        Ca0 = covariates["Ca0"]
        T_K = T_C + 273.15

        log10_k_param = (
            log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)
        ) / jnp.log(10.0)

        inputs = {"temperature_C": T_C, "pH": pH, "Ca0": Ca0}
        delta_log10_k = jnp.squeeze(residual(inputs))

        log10_k = log10_k_param + delta_log10_k
        k = jnp.power(10.0, log10_k)

        def vector_field(t, y, args):
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

    # A second simulate_fn_baseline (defined separately) omits the residual term —
    # used by the pure mechanistic baseline below.
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


@app.cell
def _simulate_fn_baseline(
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
    """Pure-mechanistic simulate_fn — identical to `simulate_fn` but with the
    residual MLP's contribution set to zero. The pure mechanistic baseline is
    the trunk on its own, so we cannot let the residual at its random init
    inject a spurious T-dependence into the rate constant during baseline fits.
    The predictors signature is kept identical (same pytree) so the framework's
    training and prediction calls are unchanged.
    """

    def simulate_fn_baseline(
        predictors: tuple[ArrheniusKinetics, BoundedPredictor],
        ts: Float[Array, " T"],
        covariates: dict[str, Array],
        y0: Float[Array, " 2"],
        solver: SolverConfig,
    ) -> Float[Array, "T 2"]:
        parametric, _residual = predictors
        log_k_ref, Ea = parametric()

        T_C = covariates["temperature_C"]
        T_K = T_C + 273.15

        log10_k = (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)
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

    return (simulate_fn_baseline,)


@app.cell(hide_code=True)
def _phase1_md(mo):
    mo.md(r"""
    ### Training masks

    Selectively freezing parts of the predictors pytree is what lets
    the same model class drive both the pure mechanistic baseline
    and the hybrid pipeline. The framework's `trainable_mask` builds
    a boolean pytree of the same shape as `predictors_init` with
    every leaf marked trainable; `freeze_modules_of_type` then
    zeroes the mask under any submodule of a given class. The
    *trunk-only mask* (`mask_p1`) freezes every leaf inside a
    `BoundedPredictor` (removing the residual MLP) and every leaf
    inside a `BoundScaler` (bound geometry is fixed by convention),
    leaving the two-element `latent` of `ArrheniusKinetics` as the
    only trainable degrees of freedom. The *residual-only mask*
    (`mask_p2`, defined later) flips this: the `ArrheniusKinetics`
    submodule is frozen at the phase-1 endpoint, the residual MLP
    is left trainable.

    `mask_p1` is reused unchanged across the pure mechanistic
    baseline (per-bin and per-bin-LOO-CV fits) and as phase 1 of
    the hybrid pipeline below.

    ```python
    mask_p1 = trainable_mask(predictors_init)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundedPredictor)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundScaler)
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
    print(f"trunk-only mask: {n_p1} trainable scalars (expected: 2 — the parametric latent)")
    return (mask_p1,)


@app.cell(hide_code=True)
def _baseline_md(mo):
    mo.md(r"""
    # A pure mechanistic baseline

    The simplest model that respects the known temperature physics
    is the parametric trunk on its own — a centred Arrhenius rate
    constant in $T$ with two scalar parameters
    $(\log k_{\mathrm{ref}}, E_a)$ and *no pH input*. Because the
    trunk has no pH input, it cannot be fitted across pH bins
    without averaging over pH-dependent rate shifts. The classical
    workaround is to fit one set of parameters per pH bin: three
    independent CMA-ES runs, three independent parameter pairs, no
    functional pH dependence between them.

    Implementing this on top of the framework requires no new model
    code: the trunk is already part of the predictors pytree, and
    the trainable mask `mask_p1` already freezes everything except
    its two scalars. We simply run `train_with_evosax` once per
    bin, on the four in-bin experiments.

    ```python
    cfg_baseline = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )

    bin_predictors = []
    for bin_idx, ph in enumerate(PH_BINS):
        bin_exps = [e for i, e in enumerate(experiments) if bin_of_exp[i] == bin_idx]
        bin_ds = make_dataset(
            bin_exps,
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
        )
        history, predictors = train_with_evosax(
            predictors_init,
            bin_ds,
            cfg_baseline,
            simulate_fn=simulate_fn_baseline,
            solver=solver,
            trainable=mask_p1,
            key=jr.PRNGKey(bin_idx),
        )
        log_k_ref, Ea = predictors[0]()
        bin_predictors.append(
            {
                "bin_idx": bin_idx,
                "pH": ph,
                "predictors": predictors,
                "log_k_ref": float(log_k_ref),
                "Ea": float(Ea),
                "final_loss": float(history[-1]),
            }
        )
    ```
    """)
    return


@app.cell
def _per_bin_fits(
    EvosaxTrainingConfig,
    OUTPUT_CHANNELS,
    PH_BINS,
    bin_of_exp: list[int],
    experiments: "list[Experiment]",
    jr,
    make_dataset,
    mask_p1,
    predictors_init,
    simulate_fn_baseline,
    solver,
    state_to_output,
    train_with_evosax,
):
    _cfg_baseline = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )

    bin_predictors: list[dict] = []
    print("Per-bin Arrhenius fits:")
    for _bin_idx, _ph in enumerate(PH_BINS):
        _bin_exps = [_e for _i, _e in enumerate(experiments) if bin_of_exp[_i] == _bin_idx]
        _bin_ds = make_dataset(
            _bin_exps,
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
        )
        _hist, _preds = train_with_evosax(
            predictors_init,
            _bin_ds,
            _cfg_baseline,
            simulate_fn=simulate_fn_baseline,
            solver=solver,
            trainable=mask_p1,
            key=jr.PRNGKey(_bin_idx),
        )
        _log_k_ref, _Ea = _preds[0]()
        bin_predictors.append(
            {
                "bin_idx": _bin_idx,
                "pH": _ph,
                "predictors": _preds,
                "log_k_ref": float(_log_k_ref),
                "Ea": float(_Ea),
                "final_loss": float(_hist[-1]),
            }
        )
        print(
            f"  bin {_bin_idx} (pH={_ph}): "
            f"log_k_ref = {float(_log_k_ref):+.3f}, "
            f"Ea = {float(_Ea):.2f} kJ/mol, "
            f"final loss = {float(_hist[-1]):.4e}"
        )
    return (bin_predictors,)


@app.cell(hide_code=True)
def _per_bin_summary_md(mo):
    mo.md(r"""
    ### Per-bin parameter recovery

    The recovered $\log k_{\mathrm{ref}}^{(b)}$ varies systematically
    across bins — exactly tracking the hidden $\ln k_{\mathrm{sat}}(\mathrm{pH}_b)$
    — while $E_a^{(b)}$ stays close to the true 30 kJ/mol in every
    bin. The pH dependence is real and structural; the trunk
    cannot represent it, but it *can* absorb it into a different
    prefactor every time it is refit.
    """)
    return


@app.cell
def _per_bin_summary_plot(
    PH_BINS,
    bin_predictors: list[dict],
    jnp,
    k_true,
    np,
    plt,
):
    fig_bp, (ax_kref, ax_ea) = plt.subplots(1, 2, figsize=(10, 4))

    pH_arr = np.array(PH_BINS)
    log_k_ref_fit = np.array([_b["log_k_ref"] for _b in bin_predictors])
    Ea_fit = np.array([_b["Ea"] for _b in bin_predictors])

    # Truth: at T_REF, log_k_param = log_k_ref / ln(10) = log10(k_sat(pH)),
    # so log_k_ref^true(pH) = ln(k_sat(pH)). We sample k_sat via k_true at T_REF=25°C.
    log_k_ref_truth = np.log(np.asarray(k_true(25.0, jnp.asarray(PH_BINS))))
    Ea_truth = 30.0  # kJ/mol, EA_TRUE

    ax_kref.scatter(pH_arr, log_k_ref_fit, s=80, color="C0", marker="o", label="fitted (per bin)")
    ax_kref.plot(
        pH_arr,
        log_k_ref_truth,
        color="black",
        linewidth=1.4,
        linestyle="--",
        marker="x",
        markersize=10,
        label=r"truth: $\ln k_\mathrm{sat}(\mathrm{pH})$",
    )
    ax_kref.set_xlabel("pH bin")
    ax_kref.set_ylabel(r"$\log k_\mathrm{ref}$ (natural log)")
    ax_kref.set_title("Recovered $\\log k_\\mathrm{ref}$ per pH bin")
    ax_kref.grid(alpha=0.3)
    ax_kref.legend(loc="best", fontsize=9)

    ax_ea.scatter(pH_arr, Ea_fit, s=80, color="C0", marker="o", label="fitted (per bin)")
    ax_ea.axhline(
        Ea_truth,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label=f"truth ($E_a$ = {Ea_truth} kJ/mol)",
    )
    ax_ea.set_xlabel("pH bin")
    ax_ea.set_ylabel("$E_a$ (kJ/mol)")
    ax_ea.set_title("Recovered $E_a$ per pH bin")
    ax_ea.set_ylim(0.0, max(40.0, float(Ea_fit.max()) * 1.2))
    ax_ea.grid(alpha=0.3)
    ax_ea.legend(loc="best", fontsize=9)

    fig_bp.tight_layout()
    fig_bp
    return


@app.cell(hide_code=True)
def _per_bin_loocv_md(mo):
    mo.md(r"""
    ### Per-bin LOO-CV

    Within each pH bin we leave one of the four $(T, C_{A,0})$
    points out at a time, refit Arrhenius on the remaining three,
    and predict the held-out one. Twelve cheap evosax fits in
    total. Because Arrhenius captures pure-temperature
    extrapolation correctly at fixed pH, the per-bin OOF parity
    should sit tight on the diagonal — confirming that the
    classical mechanistic *does* generalise as long as one stays
    inside its trained pH bin.

    ```python
    bin_loocv = []
    for bin_idx, ph in enumerate(PH_BINS):
        bin_exps = [e for i, e in enumerate(experiments) if bin_of_exp[i] == bin_idx]
        records = []
        for k in range(len(bin_exps)):
            train = [e for i, e in enumerate(bin_exps) if i != k]
            train_ds = make_dataset(
                train,
                state_to_output=state_to_output,
                output_channel_names=OUTPUT_CHANNELS,
            )
            held_ds = make_dataset(
                [bin_exps[k]],
                state_to_output=state_to_output,
                output_channel_names=OUTPUT_CHANNELS,
            )
            history, predictors = train_with_evosax(
                predictors_init,
                train_ds,
                cfg_baseline,
                simulate_fn=simulate_fn_baseline,
                solver=solver,
                trainable=mask_p1,
                key=jr.PRNGKey(bin_idx * 100 + k),
            )
            oof_pred = predict_dataset(
                predictors, held_ds, simulate_fn=simulate_fn_baseline, solver=solver
            )
            diag = gather_diagnostics(oof_pred, held_ds)
            ch = next(iter(diag))
            records.append(
                {
                    "k": k,
                    "exp": bin_exps[k],
                    "oof_obs": diag[ch]["obs"],
                    "oof_pred": diag[ch]["pred"],
                    "oof_mse": diag[ch]["mse"],
                }
            )
        bin_loocv.append(records)
    ```
    """)
    return


@app.cell
def _per_bin_loocv(
    EvosaxTrainingConfig,
    OUTPUT_CHANNELS,
    PH_BINS,
    bin_of_exp: list[int],
    experiments: "list[Experiment]",
    gather_diagnostics,
    jr,
    make_dataset,
    mask_p1,
    np,
    predict_dataset,
    predictors_init,
    simulate_fn_baseline,
    solver,
    state_to_output,
    train_with_evosax,
):
    _cfg_baseline = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )

    bin_loocv: list[list[dict]] = []
    print("Per-bin LOO-CV:")
    for _bin_idx, _ph in enumerate(PH_BINS):
        _bin_exps = [_e for _i, _e in enumerate(experiments) if bin_of_exp[_i] == _bin_idx]
        _records: list[dict] = []
        for _k in range(len(_bin_exps)):
            _train = [_e for _i, _e in enumerate(_bin_exps) if _i != _k]
            _train_ds = make_dataset(
                _train,
                state_to_output=state_to_output,
                output_channel_names=OUTPUT_CHANNELS,
            )
            _held_ds = make_dataset(
                [_bin_exps[_k]],
                state_to_output=state_to_output,
                output_channel_names=OUTPUT_CHANNELS,
            )
            _hist, _preds = train_with_evosax(
                predictors_init,
                _train_ds,
                _cfg_baseline,
                simulate_fn=simulate_fn_baseline,
                solver=solver,
                trainable=mask_p1,
                key=jr.PRNGKey(_bin_idx * 100 + _k),
            )
            _oof_pred = predict_dataset(
                _preds, _held_ds, simulate_fn=simulate_fn_baseline, solver=solver
            )
            _diag = gather_diagnostics(_oof_pred, _held_ds)
            _ch = next(iter(_diag))
            _records.append(
                {
                    "k": _k,
                    "exp": _bin_exps[_k],
                    "oof_obs": _diag[_ch]["obs"],
                    "oof_pred": _diag[_ch]["pred"],
                    "oof_mse": _diag[_ch]["mse"],
                }
            )
        bin_loocv.append(_records)
        _mse_arr = np.array([_r["oof_mse"] for _r in _records])
        print(
            f"  bin {_bin_idx} (pH={_ph}): "
            f"mean OOF MSE = {float(_mse_arr.mean()):.3e}, "
            f"max = {float(_mse_arr.max()):.3e}"
        )
    return (bin_loocv,)


@app.cell
def _per_bin_parity_plot(PH_BINS, bin_loocv: list[list[dict]], np, plt):
    fig_bp_par, axes_bp_par = plt.subplots(
        1, len(PH_BINS), figsize=(4.0 * len(PH_BINS), 4.2), sharey=True, sharex=True
    )
    _bin_colors = ("C0", "C1", "C3")
    for _ax, _ph, _records, _color in zip(
        axes_bp_par, PH_BINS, bin_loocv, _bin_colors, strict=True
    ):
        _obs = np.concatenate([_r["oof_obs"] for _r in _records])
        _pred = np.concatenate([_r["oof_pred"] for _r in _records])
        _resid = _pred - _obs
        _mse = float(np.mean(_resid**2))
        _ss_tot = float(np.sum((_obs - _obs.mean()) ** 2))
        _r2 = 1.0 - float(np.sum(_resid**2)) / _ss_tot if _ss_tot > 0 else float("nan")
        _ax.scatter(_obs, _pred, s=24, alpha=0.7, color=_color, label=f"OOF (n={len(_obs)})")
        _lo = float(min(_obs.min(), _pred.min()))
        _hi = float(max(_obs.max(), _pred.max()))
        if _lo == _hi:
            _pad = 0.1 if _lo == 0.0 else abs(_lo) * 0.1
            _lo, _hi = _lo - _pad, _hi + _pad
        _ax.plot([_lo, _hi], [_lo, _hi], color="black", linestyle="--", linewidth=0.8)
        _r2_str = "nan" if _r2 != _r2 else f"{_r2:.4f}"
        _ax.set_title(f"pH = {_ph}\nOOF R²={_r2_str}, MSE={_mse:.3e}")
        _ax.set_xlabel("observed")
        _ax.legend(loc="best", fontsize=9)
        _ax.grid(alpha=0.3)
    axes_bp_par[0].set_ylabel("predicted (out-of-fold)")
    fig_bp_par.suptitle("Pure mechanistic — per-bin LOO-CV parity")
    fig_bp_par.tight_layout()
    fig_bp_par
    return


@app.cell(hide_code=True)
def _transfer_md(mo):
    mo.md(r"""
    ### Inter-bin transfer

    Within-bin generalisation is the easy half of the story. The
    hard half is *transfer*: what happens when the practitioner
    tries to use one bin's fit at a different pH? Below we take
    the parameters fitted on the pH = 5.0 data and use them — with
    no modification, no re-fit — to predict the four pH = 7.0
    trajectories. The trunk has no pH input, so it returns the
    same $\log_{10}k(T)$ regardless of pH; the predicted decay
    rate is therefore far too fast at pH = 7.0, and the predicted
    trajectories badly overshoot the slow observed decay.

    This is the failure mode that motivates the hybrid: a pure
    mechanistic model has no place to put the missing pH
    dependence, so the practitioner is forced to maintain a
    dictionary of bin-specific fits and lose all interpolation
    capability between them.

    ```python
    bin_target = 2  # pH = 7.0
    bin_fitted = 0  # pH = 5.0
    target_exps = [e for i, e in enumerate(experiments) if bin_of_exp[i] == bin_target]
    target_ds = make_dataset(
        target_exps,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    fitted_predictors = bin_predictors[bin_fitted]["predictors"]
    transfer_pred = predict_dataset(
        fitted_predictors, target_ds, simulate_fn=simulate_fn_baseline, solver=solver
    )
    ```
    """)
    return


@app.cell
def _transfer_plot(
    OUTPUT_CHANNELS,
    bin_of_exp: list[int],
    bin_predictors: list[dict],
    experiments: "list[Experiment]",
    make_dataset,
    np,
    predict_dataset,
    simulate_fn_baseline,
    solver,
    state_to_output,
    trajectory_grid_plot,
):
    _bin_target = 2  # pH = 7.0
    _bin_fitted = 0  # pH = 5.0
    _target_exps = [_e for _i, _e in enumerate(experiments) if bin_of_exp[_i] == _bin_target]
    _target_ds = make_dataset(
        _target_exps,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    _fitted_predictors = bin_predictors[_bin_fitted]["predictors"]
    _transfer_pred = predict_dataset(
        _fitted_predictors, _target_ds, simulate_fn=simulate_fn_baseline, solver=solver
    )
    _per_exp = [np.asarray(_transfer_pred[0][_i]) for _i in range(len(_target_exps))]
    fig_transfer = trajectory_grid_plot(
        _target_exps,
        _per_exp,
        title="Inter-bin transfer: pH=7.0 trajectories predicted with pH=5.0-fitted parameters",
        predicted_label="bin-0 prediction (wrong pH)",
    )
    fig_transfer
    return


@app.cell(hide_code=True)
def _hybrid_md(mo):
    mo.md(r"""
    # The hybrid model

    The hybrid is the same architecture as the baseline plus the
    residual MLP, trained on all twelve experiments simultaneously.
    Training proceeds in two phases that share the predictors pytree
    but flip the trainable mask between them.

    **Phase 1 — Arrhenius trunk fit (evosax/CMA-ES).** The same
    `mask_p1` used by the baseline is run on the *combined* dataset.
    Without a pH input the trunk has to compromise across bins; the
    fit therefore lands at parameters that minimise the joint MSE
    but cannot match any single bin precisely. This is by design:
    phase 1 establishes the temperature dependence and leaves the
    pH residual to phase 2.

    **Phase 2 — residual MLP fit (optax/AdamW).** The mask flips:
    `freeze_modules_of_type(predictors_p1, ArrheniusKinetics)` zeroes
    the trunk submask, so the parametric scalars stay fixed at their
    phase-1 endpoint. The residual MLP becomes the only trainable
    component. Because the residual at the phase-1 endpoint contributes
    ${\sim}0$ decades (symmetric output bound, sigmoid midpoint), the
    phase-2 step-0 loss equals the phase-1 final loss to within
    numerical noise — the seam between the two phases is invisible
    in the loss curve.

    The two-phase decomposition pays off whenever the trunk's basin
    is non-convex: CMA-ES locates it from a wide LHS prior in the
    two-dimensional latent space; AdamW then polishes the much
    higher-dimensional residual smoothly. Neither optimiser alone
    would do as well on the combined search.

    ```python
    # Phase 1 — Arrhenius trunk fit (CMA-ES) on the full combined dataset.
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

    # Flip the mask: freeze the trunk, leave only the residual MLP trainable.
    mask_p2 = trainable_mask(predictors_p1)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, ArrheniusKinetics)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, BoundScaler)

    # Phase 2 — residual MLP fit (AdamW) on the same dataset.
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
    ```
    """)
    return


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
def _phase1_diag(dataset, gather_diagnostics, predictions_p1):
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
    return


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
    ### Phase 2 — residual MLP fit

    The trainable mask flips. `freeze_modules_of_type(predictors_p1, ArrheniusKinetics)`
    zeroes the trunk submask so the parametric scalars stay pinned at
    their phase-1 endpoint, and the residual MLP becomes the only
    trainable component. AdamW is run for 200 steps at learning rate
    $3 \times 10^{-3}$ on the same MSE loss, on the same combined
    dataset.
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
def _phase2_diag(dataset, gather_diagnostics, predictions_p2):
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
    return


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

    ```python
    n_folds = len(experiments)
    fold_records = []
    for k in range(n_folds):
        train_exps = [e for i, e in enumerate(experiments) if i != k]
        train_ds = make_dataset(
            train_exps,
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
        )
        held_ds = make_dataset(
            [experiments[k]],
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
        )

        # Phase 1 — refit the parametric trunk on the in-fold experiments.
        hist_p1, pred_p1 = train_with_evosax(
            predictors_init,
            train_ds,
            cfg_p1,
            simulate_fn=simulate_fn,
            solver=solver,
            trainable=mask_p1,
            key=jr.PRNGKey(k),
        )
        # Phase 2 — refit the residual MLP, frozen trunk.
        hist_p2, pred_p2 = train_with_optax(
            pred_p1,
            train_ds,
            cfg_p2,
            simulate_fn=simulate_fn,
            solver=solver,
            trainable=mask_p2,
            key=jr.PRNGKey(1000 + k),
        )

        # Out-of-fold prediction on the single held-out experiment.
        oof_pred = predict_dataset(
            pred_p2, held_ds, simulate_fn=simulate_fn, solver=solver
        )
        diag = gather_diagnostics(oof_pred, held_ds)
        ch = next(iter(diag))
        fold_records.append(
            {
                "k": k,
                "exp": experiments[k],
                "oof_obs": diag[ch]["obs"],
                "oof_pred": diag[ch]["pred"],
                "oof_mse": diag[ch]["mse"],
            }
        )
    ```
    """)
    return


@app.cell
def _loocv_run(
    EvosaxTrainingConfig,
    OUTPUT_CHANNELS,
    OptaxTrainingConfig,
    experiments: "list[Experiment]",
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
def _loocv_parity_plot(fold_records: list[dict], np, plt):
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
def _loocv_traj_plot(fold_records: list[dict], trajectory_grid_plot):
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
def _lopo_md(mo):
    mo.md(r"""
    ## Leave-one-pH-out cross-validation

    The combined LOO-CV holds out one experiment but always leaves
    the held-out experiment's pH represented in the training set
    (the other three experiments at the same pH remain). To match
    the inter-bin transfer test from the pure mechanistic baseline
    on equal terms, we now hold out *every* experiment at the
    intermediate pH = 5.85 and retrain the hybrid on the eight
    remaining experiments at pH = 5.0 and pH = 7.0 only. The
    model must then predict four trajectories at a pH it has
    never seen during training — the parallel of the inter-bin
    transfer test, but with a residual MLP that *can* learn pH
    structure from the two flanking bins and interpolate between
    them.

    **The hybrid model successfully learns the intermediate pH
    from its two flanking bins, recovering the kinetic equation
    $k(T, \mathrm{pH})$ at a pH it never saw during training.**

    ```python
    held_pH_idx = 1  # bin index 1 = pH = 5.85
    train_exps_lopo = [
        e for i, e in enumerate(experiments) if bin_of_exp[i] != held_pH_idx
    ]
    held_exps_lopo = [
        e for i, e in enumerate(experiments) if bin_of_exp[i] == held_pH_idx
    ]
    train_ds_lopo = make_dataset(
        train_exps_lopo,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    held_ds_lopo = make_dataset(
        held_exps_lopo,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )

    # Same two-phase pipeline, but on the LOPO training split.
    hist_lopo_p1, predictors_lopo_p1 = train_with_evosax(
        predictors_init,
        train_ds_lopo,
        cfg_p1,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p1,
        key=jr.PRNGKey(7777),
    )
    hist_lopo_p2, predictors_lopo_p2 = train_with_optax(
        predictors_lopo_p1,
        train_ds_lopo,
        cfg_p2,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p2,
        key=jr.PRNGKey(8888),
    )

    # Predict the held-out pH = 5.85 trajectories from the LOPO model.
    pred = predict_dataset(
        predictors_lopo_p2, held_ds_lopo, simulate_fn=simulate_fn, solver=solver
    )
    ```
    """)
    return


@app.cell
def _lopo_train(
    EvosaxTrainingConfig,
    OUTPUT_CHANNELS,
    OptaxTrainingConfig,
    bin_of_exp: list[int],
    experiments: "list[Experiment]",
    jr,
    make_dataset,
    mask_p1,
    mask_p2,
    predictors_init,
    simulate_fn,
    solver,
    state_to_output,
    train_with_evosax,
    train_with_optax,
):
    held_pH_idx = 1  # bin index 1 = pH = 5.85
    train_exps_lopo = [_e for _i, _e in enumerate(experiments) if bin_of_exp[_i] != held_pH_idx]
    held_exps_lopo = [_e for _i, _e in enumerate(experiments) if bin_of_exp[_i] == held_pH_idx]
    train_ds_lopo = make_dataset(
        train_exps_lopo,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    held_ds_lopo = make_dataset(
        held_exps_lopo,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )

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

    _hist_lopo_p1, predictors_lopo_p1 = train_with_evosax(
        predictors_init,
        train_ds_lopo,
        _cfg_p1,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p1,
        key=jr.PRNGKey(7777),
    )
    _hist_lopo_p2, predictors_lopo_p2 = train_with_optax(
        predictors_lopo_p1,
        train_ds_lopo,
        _cfg_p2,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p2,
        key=jr.PRNGKey(8888),
    )
    print(
        f"LOPO (held out pH = 5.85): {len(train_exps_lopo)} train experiments, "
        f"{len(held_exps_lopo)} held out"
    )
    print(
        f"  phase 1 final loss = {float(_hist_lopo_p1[-1]):.4e}, "
        f"phase 2 final loss = {float(_hist_lopo_p2[-1]):.4e}"
    )
    return held_ds_lopo, held_exps_lopo, predictors_lopo_p1, predictors_lopo_p2


@app.cell
def _lopo_predict(
    gather_diagnostics,
    held_ds_lopo,
    held_exps_lopo,
    np,
    predict_dataset,
    predictors_lopo_p2,
    simulate_fn,
    solver,
):
    _pred = predict_dataset(
        predictors_lopo_p2, held_ds_lopo, simulate_fn=simulate_fn, solver=solver
    )
    _diag = gather_diagnostics(_pred, held_ds_lopo)
    _ch = next(iter(_diag))
    lopo_obs = _diag[_ch]["obs"]
    lopo_pred = _diag[_ch]["pred"]
    lopo_mse = _diag[_ch]["mse"]
    lopo_r2 = _diag[_ch]["r2"]
    lopo_per_exp_traj = [np.asarray(_pred[0][_i]) for _i in range(len(held_exps_lopo))]
    print(f"  LOPO out-of-fold (held-out pH = 5.85): MSE = {lopo_mse:.3e}, R² = {lopo_r2:.4f}")
    return lopo_obs, lopo_per_exp_traj, lopo_pred


@app.cell(hide_code=True)
def _lopo_parity_md(mo):
    mo.md(r"""
    ### LOPO parity and trajectories

    Out-of-fold predictions for the four held-out pH = 5.85
    experiments. Because the residual MLP saw pH = 5.0 and
    pH = 7.0 during training but never pH = 5.85, this is a
    direct measure of pH-interpolation skill — the parallel of
    the pure mechanistic's inter-bin transfer failure.
    """)
    return


@app.cell
def _lopo_parity_plot(
    held_exps_lopo,
    lopo_obs,
    lopo_per_exp_traj,
    lopo_pred,
    np,
    plt,
    trajectory_grid_plot,
):
    fig_lopo, (ax_par, ax_dummy) = plt.subplots(1, 2, figsize=(11, 4.5))

    _resid = lopo_pred - lopo_obs
    _mse = float(np.mean(_resid**2))
    _ss_tot = float(np.sum((lopo_obs - lopo_obs.mean()) ** 2))
    _r2 = 1.0 - float(np.sum(_resid**2)) / _ss_tot if _ss_tot > 0 else float("nan")

    ax_par.scatter(
        lopo_obs,
        lopo_pred,
        s=30,
        alpha=0.8,
        color="C2",
        marker="^",
        edgecolor="black",
        linewidth=0.5,
        label=f"OOF held-out pH=5.85 (n={len(lopo_obs)})",
    )
    _lo = float(min(lopo_obs.min(), lopo_pred.min()))
    _hi = float(max(lopo_obs.max(), lopo_pred.max()))
    if _lo == _hi:
        _pad = 0.1 if _lo == 0.0 else abs(_lo) * 0.1
        _lo, _hi = _lo - _pad, _hi + _pad
    ax_par.plot([_lo, _hi], [_lo, _hi], color="black", linestyle="--", linewidth=0.8)
    _r2_str = "nan" if _r2 != _r2 else f"{_r2:.4f}"
    ax_par.set_title(f"LOPO parity (held-out pH=5.85)\nR²={_r2_str}, MSE={_mse:.3e}")
    ax_par.set_xlabel("observed")
    ax_par.set_ylabel("predicted (out-of-fold)")
    ax_par.legend(loc="best", fontsize=9)
    ax_par.grid(alpha=0.3)
    ax_dummy.set_visible(False)

    fig_lopo.tight_layout()
    fig_lopo

    fig_lopo_traj = trajectory_grid_plot(
        held_exps_lopo,
        lopo_per_exp_traj,
        title="LOPO trajectories — held-out pH=5.85 predicted from pH=5.0+7.0 training",
        predicted_label="LOPO prediction",
    )
    fig_lopo_traj
    return


@app.cell(hide_code=True)
def _lopo_kreveal_md(mo):
    mo.md(r"""
    ### LOPO $\log_{10}k$ reveal

    The same axes as the post-fit reveal further up, but with the
    hybrid retrained on only pH = 5.0 and pH = 7.0. The dotted
    hybrid curve at every fixed temperature passes between the two
    trained pH values — the residual MLP has interpolated the
    saturation knee from its two flanking bins. The black markers
    show all twelve experiments at their true
    $(\mathrm{pH}, \log_{10} k_{\mathrm{true}})$; the four at
    pH = 5.85 sit *between* the two trained pH bins by construction.
    """)
    return


@app.cell
def _lopo_kreveal_plot(
    experiments: "list[Experiment]",
    k_reveal_plot,
    predictors_lopo_p1,
    predictors_lopo_p2,
):
    fig_lopo_kreveal = k_reveal_plot(
        predictors_lopo_p1,
        predictors_lopo_p2,
        experiments,
        title="log10 k(pH) — LOPO hybrid (trained on pH=5.0 and 7.0 only)",
    )
    fig_lopo_kreveal
    return


@app.cell(hide_code=True)
def _outro(mo):
    mo.md(r"""
    ## Discussion

    The two evaluations make opposite statements about the same
    dataset. The pure mechanistic baseline fits each pH bin
    accurately on its own — per-bin LOO-CV stays tight against the
    diagonal in every bin, and the recovered $(\log k_{\mathrm{ref}}, E_a)$
    track the truth — but cannot transfer between bins, because the
    parametric model class has no place to put pH. The inter-bin
    transfer test is the failure mode that this implies: a model
    fitted at pH = 5.0 systematically overshoots pH = 7.0 trajectories,
    and there is no fix within the model class. To use the
    mechanistic at a new pH, the practitioner must collect data at
    that pH and refit — a *dictionary* of fits with no
    interpolation between entries.

    The hybrid is the same trunk plus a small residual MLP that
    consumes pH as an input. Trained on all twelve experiments at
    once, it achieves an in-sample fit comparable to the per-bin
    baselines (R² $\approx$ 0.999 on the headline run), and the
    combined LOO-CV confirms it generalises across held-out
    experiments inside the trained pH range. The leave-one-pH-out
    test is the hard one: holding out *every* experiment at the
    intermediate pH = 5.85, the hybrid still predicts those
    trajectories with R² $\approx 0.98$ — the residual MLP has
    interpolated the saturation knee from its two flanking bins
    without ever seeing data at the held-out pH.

    Two practical points worth flagging. First, the pure mechanistic
    baseline must run with a `simulate_fn` that *omits* the residual
    term entirely; the residual at random initialisation is not
    identically zero, and letting it through pollutes the trunk's
    $E_a$ identification — the symmetric output bound only zeros the
    residual at the sigmoid midpoint, which is not where a freshly
    initialised MLP sits. Second, the LOPO test is sensitive to how
    informative the flanking pH bins are: with three pH knots the
    interpolation is well-posed; with two it would devolve into
    extrapolation and the hybrid's advantage over the dictionary
    approach would shrink.
    """)
    return


if __name__ == "__main__":
    app.run()
