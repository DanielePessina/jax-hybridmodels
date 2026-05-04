"""Crystallisation walkthrough — hybrid MLP and mechanistic side by side.

Runs the same dataset, ODE backbone, projector, solver, and diagnostics
through two different trainable components and compares them in one
place. Mirrors the prose in ``docs/examples/crystallisation.md`` and
``docs/examples/crystallisation-mechanistic.md``.

Run interactively: ``uv run marimo edit examples/crystallisation/notebook.py``
Run as script:     ``uv run python examples/crystallisation/notebook.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Crystallisation: a hybrid and a mechanistic model, side by side

    This notebook demonstrates how the `hybridmodels` library can be
    used to fit two different kinetic models of a batch crystallisation
    process to the same dataset. *Crystallisation* is the process by
    which a dissolved solute leaves solution and forms a solid
    crystalline phase; the two phenomena that drive it are *nucleation*
    (the appearance of new crystals) and *growth* (the enlargement of
    existing ones). Quantitative models of these two rates are central
    to the design and control of pharmaceutical and fine-chemical
    processes.

    The aim of *hybrid modelling* is to combine first-principles
    mechanistic structure (here, the conservation laws governing the
    crystal population) with data-driven components (here, neural
    networks that learn the rate laws from data). The framework allows
    the same physical backbone, numerical solver, and diagnostics to be
    reused while the trainable component is swapped. In what follows we
    consider a synthetic dataset of four batch experiments and fit two
    models:

    1. **Hybrid MLP model.** Two small multi-layer perceptrons emit the
       logarithms of the growth and nucleation rates as functions of
       temperature and supersaturation. Each network is wrapped in a
       `BoundedPredictor` so that its output is constrained to a
       physically plausible range. Training uses the Adam optimiser
       through `train_with_optax`.
    2. **Mechanistic model.** The two rate functions are replaced by
       Classical Nucleation Theory (CNT) for the nucleation rate and a
       power law for the growth rate. The model has only four scalar
       parameters and no explicit dependence on the input covariates;
       these parameters are estimated by Covariance Matrix Adaptation
       Evolution Strategy (CMA-ES) through `train_with_evosax`.

    Both models share the same dynamical backbone: a six-state
    *population balance* described by the *method of moments*. In a
    population balance, the distribution of crystal sizes is tracked by
    its first few statistical moments $\mu_k(t)$, where $\mu_k$ is the
    integral of $L^k n(L,t)$ over crystal size $L$, with $n$ the number
    density. Together with the solute concentration this yields the
    following system of ordinary differential equations:

    $$
    \begin{aligned}
    \frac{d\mu_0}{dt} &= J(t) \\
    \frac{d\mu_k}{dt} &= k\, G(t)\, \mu_{k-1}, \quad k = 1, 2, 3, 4 \\
    \frac{d\,\text{conc}}{dt} &= -3\, K_v\, \rho_c\, G(t)\, \mu_2
    \end{aligned}
    $$

    Here $G(t)$ is the linear growth velocity (m/s), $J(t)$ is the
    nucleation rate (number of new crystals per cubic metre per
    second), $K_v$ is a volumetric shape factor relating crystal volume
    to the cube of a characteristic length, and $\rho_c$ is the crystal
    density. The two rates $G$ and $J$ are the unknown functions to be
    learned (or specified mechanistically). The observable quantities
    are the solute concentration, sampled densely in time, and the
    *volume-weighted mean diameter* $d_{43} = (\mu_4 / \mu_3) \cdot
    10^{6}$ in micrometres, which characterises the average size of
    the crystals produced and is observed only at the end of each
    experiment.
    """)
    return


@app.cell
def _imports():
    import jax

    # x64 must be enabled before any other JAX-touching import. The
    # population-balance moments span ~18 decades during integration;
    # float32 mass balance drifts visibly within a single experiment.
    jax.config.update("jax_enable_x64", True)

    import diffrax
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import Array
    from jaxtyping import Float

    import marimo as mo
    from hybridmodels import (
        BoundedPredictor,
        BoundScaler,
        ChannelObs,
        Experiment,
        MLPPredictor,
        SolverConfig,
        make_dataset,
        make_experiment,
        predict_dataset,
    )
    from hybridmodels.training.evosax import (
        EvosaxTrainingConfig,
        train_with_evosax,
    )
    from hybridmodels.training.optax import (
        OptaxTrainingConfig,
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
        jnp,
        jr,
        make_dataset,
        make_experiment,
        mo,
        np,
        plt,
        predict_dataset,
        train_with_evosax,
        train_with_optax,
    )


@app.cell(hide_code=True)
def _dataset_md(mo):
    mo.md(r"""
    ## The dataset

    A synthetic dataset of four experiments is provided, consisting of
    two replicates at each of two operating temperatures (17 °C and
    21 °C). Each experiment records solute concentration at a sequence
    of sampling times together with a single volume-weighted mean
    diameter $d_{43}$ measurement at the end of the run. This sampling
    pattern reflects common laboratory practice, in which concentration
    is monitored continuously by a spectroscopic probe whereas particle
    size is measured offline only on the final slurry.

    The concentration channel is assigned a uniform observation
    variance of $0.1$ (broadcast across every time point), while the
    $d_{43}$ channel carries its own per-experiment variance reflecting
    the measurement uncertainty of the offline sizing instrument.

    | `exp_id` | T (°C) | n conc | n d43 | t span (min) | terminal d43 (µm) |
    |----------|-------:|-------:|------:|--------------|------------------:|
    | `E1`     | 17.0   | 9      | 1     | 0 → 270      | 9.2               |
    | `E2`     | 17.0   | 9      | 1     | 0 → 270      | 7.7               |
    | `E3`     | 21.0   | 7      | 1     | 0 → 360      | 10.5              |
    | `E4`     | 21.0   | 7      | 1     | 0 → 375      | 11.9              |

    The four experiments do not share a common time grid. The
    `make_dataset` helper handles this by forming a union of the
    sampling times for each experiment and grouping experiments of
    matching length into *buckets* that can be vectorised together at
    training time. With this dataset two buckets are produced, one
    containing the two experiments at 17 °C and one containing the two
    at 21 °C.
    """)
    return


@app.cell
def _experiments_data():
    EXPERIMENTS_DATA = (
        {
            "exp_id": "E1",
            "temperature_C": 17.0,
            "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
            "conc": (14.7, 13.7, 7.7, 7.0, 5.7, 5.5, 5.4, 5.1, 5.2),
            "d43_time_min": 270.0,
            "d43": 9.2,
            "d43_var": 5.345,
        },
        {
            "exp_id": "E2",
            "temperature_C": 17.0,
            "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
            "conc": (11.6, 10.8, 10.8, 10.5, 10.9, 9.5, 6.8, 6.1, 5.7),
            "d43_time_min": 270.0,
            "d43": 7.7,
            "d43_var": 3.734,
        },
        {
            "exp_id": "E3",
            "temperature_C": 21.0,
            "time_min": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            "conc": (16.8, 12.1, 8.6, 7.6, 7.2, 6.8, 6.4),
            "d43_time_min": 360.0,
            "d43": 10.5,
            "d43_var": 1.421,
        },
        {
            "exp_id": "E4",
            "temperature_C": 21.0,
            "time_min": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 375.0),
            "conc": (14.4, 14.2, 13.6, 10.1, 9.3, 7.6, 7.0),
            "d43_time_min": 375.0,
            "d43": 11.9,
            "d43_var": 0.267,
        },
    )
    CONC_VAR = 0.1  # made-up uniform concentration variance, broadcast per row
    return CONC_VAR, EXPERIMENTS_DATA


@app.cell
def _raw_data_plot(EXPERIMENTS_DATA, plt):
    fig_raw, ax_raw = plt.subplots(figsize=(7, 3.5))
    color_by_T = {17.0: "tab:blue", 21.0: "tab:orange"}
    for d in EXPERIMENTS_DATA:
        ax_raw.plot(
            d["time_min"],
            d["conc"],
            "o-",
            color=color_by_T[d["temperature_C"]],
            label=f"{d['exp_id']} (T={d['temperature_C']:.0f}°C, d43={d['d43']:.1f} µm)",
            alpha=0.85,
            linewidth=1.2,
            markersize=4,
        )
    ax_raw.set_xlabel("time (min)")
    ax_raw.set_ylabel("concentration")
    ax_raw.set_title("Concentration trajectories — two replicates per temperature")
    ax_raw.legend(fontsize=8, loc="upper right")
    ax_raw.grid(alpha=0.3)
    fig_raw.tight_layout()
    fig_raw
    return


@app.cell(hide_code=True)
def _y0_md(mo):
    mo.md(r"""
    ## Initial conditions and `Experiment` construction

    Each experiment is represented as an `Experiment` object that
    bundles together its observations, covariates, and a function
    `y0_fn` returning the initial state of the ODE system. The latter
    is evaluated once per experiment when the dataset is built. Here
    the five population-balance moments $\mu_0, \dots, \mu_4$ are
    initialised to zero, reflecting the assumption that the suspension
    contains no crystals at the start of the run; the concentration
    state is initialised from the first observed value of the `conc`
    channel. Temperature is the only experimental covariate.

    ```python
    def y0_fn(covariates, channels):
        # [mu0..mu4, conc]: moments at zero, conc at first observation.
        init_conc = jnp.asarray(channels["conc"].values[0])
        return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])

    experiments = []
    for data in EXPERIMENTS_DATA:
        time_min = jnp.asarray(data["time_min"], dtype=float)
        conc     = jnp.asarray(data["conc"], dtype=float)
        d43_ts   = jnp.asarray((data["d43_time_min"],), dtype=float)
        d43_vals = jnp.asarray((data["d43"],), dtype=float)
        d43_var  = jnp.asarray((data["d43_var"],), dtype=float)
        experiments.append(make_experiment(
            covariates={"temperature_C": float(data["temperature_C"])},
            channels={
                "conc": ChannelObs(ts=time_min, values=conc,
                                   variance=jnp.full_like(conc, CONC_VAR)),
                "d43":  ChannelObs(ts=d43_ts,   values=d43_vals, variance=d43_var),
            },
            y0_fn=y0_fn,
            exp_id=str(data["exp_id"]),
        ))
    ```
    """)
    return


@app.cell
def _y0_fn(Array, ChannelObs, Float, jnp):
    def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 6"]:
        """[mu0..mu4, conc]: moments at zero, conc at first observation."""
        init_conc = jnp.asarray(channels["conc"].values[0])
        return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])

    return (y0_fn,)


@app.cell
def _build_experiments(
    CONC_VAR,
    ChannelObs,
    EXPERIMENTS_DATA,
    Experiment,
    jnp,
    make_experiment,
    y0_fn,
):
    experiments: list[Experiment] = []
    for _data in EXPERIMENTS_DATA:
        _time_min = jnp.asarray(_data["time_min"], dtype=float)
        _conc = jnp.asarray(_data["conc"], dtype=float)
        _d43_ts = jnp.asarray((_data["d43_time_min"],), dtype=float)
        _d43_vals = jnp.asarray((_data["d43"],), dtype=float)
        _d43_var = jnp.asarray((_data["d43_var"],), dtype=float)
        experiments.append(
            make_experiment(
                covariates={"temperature_C": float(_data["temperature_C"])},
                channels={
                    "conc": ChannelObs(
                        ts=_time_min,
                        values=_conc,
                        variance=jnp.full_like(_conc, CONC_VAR),
                    ),
                    "d43": ChannelObs(ts=_d43_ts, values=_d43_vals, variance=_d43_var),
                },
                y0_fn=y0_fn,
                exp_id=str(_data["exp_id"]),
            )
        )

    print(f"built {len(experiments)} experiments")
    for _exp in experiments:
        print(
            f"  {_exp.exp_id}: T={float(_exp.covariates['temperature_C']):.1f}°C, "
            f"conc obs={_exp.channels['conc'].values.shape[0]}, "
            f"d43 obs={_exp.channels['d43'].values.shape[0]}"
        )
    return (experiments,)


@app.cell(hide_code=True)
def _projector_md(mo):
    mo.md(r"""
    ## Mapping the ODE state to observable quantities

    The integrator returns the full six-state trajectory containing
    the five moments and the concentration. The observable channels
    are concentration and the volume-weighted mean diameter
    $d_{43} = \mu_4 / \mu_3$. A *projector* function
    `state_to_output` is therefore required to map the integrator
    output to the observed channels.

    Computing $d_{43}$ requires care because $\mu_3$ vanishes whenever
    no crystals have yet nucleated, so an unguarded division would
    produce non-finite values. A naive guard of the form
    `jnp.where(mu3 > eps, mu4 / mu3, 0.0)` is not sufficient: although
    JAX selects the safe branch in the forward pass, the unsafe branch
    is still traced for the gradient, where the division by zero
    yields NaN values that propagate through reverse-mode
    differentiation and corrupt the loss. The standard remedy in JAX
    is a *double-`where*` pattern: the denominator is first replaced
    by a safe value (`safe_mu3`) before the division is performed, and
    a second `where` then selects between the safe ratio and a zero
    output. This ensures that both the value and its gradient are
    well-defined everywhere.

    ```python
    D43_MU3_EPS = 1e-6
    D43_MAX = 55.0

    def state_to_output(state):
        # Maps [T, 6] -> [T, 2] in OUTPUT_CHANNELS = ('conc', 'd43') order.
        mu3 = state[..., 3]
        mu4 = state[..., 4]
        conc = state[..., 5]
        safe_mu3 = jnp.where(mu3 > D43_MU3_EPS, mu3, 1.0)
        ratio = jnp.where(mu3 > D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
        d43 = jnp.clip(
            jnp.where(jnp.isfinite(ratio) & (ratio > 0.0), ratio, 0.0),
            0.0,
            D43_MAX,
        )
        return jnp.stack([conc, d43], axis=-1)
    ```
    """)
    return


@app.cell
def _state_to_output(Array, Float, jnp):
    D43_MU3_EPS = 1e-6
    D43_MAX = 55.0

    def state_to_output(state: Float[Array, "T 6"]) -> Float[Array, "T 2"]:
        """[T, 6] -> [T, 2] in OUTPUT_CHANNELS = ('conc', 'd43') order."""
        mu3 = state[..., 3]
        mu4 = state[..., 4]
        conc = state[..., 5]
        safe_mu3 = jnp.where(mu3 > D43_MU3_EPS, mu3, 1.0)
        ratio = jnp.where(mu3 > D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
        d43 = jnp.clip(
            jnp.where(jnp.isfinite(ratio) & (ratio > 0.0), ratio, 0.0),
            0.0,
            D43_MAX,
        )
        return jnp.stack([conc, d43], axis=-1)

    OUTPUT_CHANNELS = ("conc", "d43")
    return OUTPUT_CHANNELS, state_to_output


@app.cell(hide_code=True)
def _dataset_step_md(mo):
    mo.md(r"""
    ## Bucketing the experiments with `make_dataset`

    The call to `make_dataset` constructs a union time axis for each
    experiment and groups experiments of matching length into buckets
    that can be processed together as a single batched array. With
    this dataset, two buckets are produced. The first contains the two
    experiments at 17 °C, both of length 9 (the terminal $d_{43}$
    measurement at 270 minutes coincides with an existing concentration
    sample). The second contains the two experiments at 21 °C, of
    length 8 (a $d_{43}$ measurement at 360 or 375 minutes extends the
    seven-point concentration grid by one).

    The bucket shape is used as a key for just-in-time compilation:
    the training step is compiled once per bucket and then reused
    across every gradient update, which amortises the compilation cost
    across the optimisation loop.

    ```python
    OUTPUT_CHANNELS = ("conc", "d43")

    dataset = make_dataset(
        experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    ```
    """)
    return


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
    print(f"{len(dataset.bucket_payloads)} bucket(s)")
    for _i, _bp in enumerate(dataset.bucket_payloads):
        print(
            f"  bucket {_i}: ts={tuple(_bp.ts.shape)}, "
            f"y_observed={tuple(_bp.y_observed.shape)}, "
            f"mask={tuple(_bp.mask.shape)}, n_obs={int(_bp.n_obs)}"
        )
    return (dataset,)


@app.cell(hide_code=True)
def _solver_md(mo):
    mo.md(r"""
    ## Configuring the ODE solver

    The dynamical system is integrated with the explicit Runge-Kutta
    method `Tsit5` from the `diffrax` library, with adaptive step-size
    control governed by relative and absolute tolerances. A subtle
    issue here is that the population-balance moments $\mu_k$ have
    very different natural magnitudes: $\mu_0$ counts particles per
    unit volume and grows to large values, while higher-order moments
    can be many orders of magnitude smaller or larger depending on
    crystal size. Across a full integration the moments span roughly
    eighteen decades, so a single scalar absolute tolerance would
    either over-resolve the small components or under-resolve the
    large ones. To address this, `SolverConfig` accepts a tuple of
    per-state absolute tolerances matched to the natural magnitude of
    each state variable.

    `Tsit5` is suitable for both models in this notebook. If the CNT
    exponent stiffens the system substantially during training, an
    implicit method such as `Kvaerno3` may be used instead.

    ```python
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        # per-state floor at ~9 decades below natural magnitude
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )
    ```
    """)
    return


@app.cell
def _solver(SolverConfig, diffrax):
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        # per-state floor at ~9 decades below natural magnitude
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )
    return (solver,)


@app.cell(hide_code=True)
def _mlp_section_md(mo):
    mo.md(r"""
    # Part 1: the hybrid MLP model

    In the hybrid formulation, the two unknown rate functions $G$ and
    $J$ are represented by separate multi-layer perceptrons (MLPs)
    that take temperature and supersaturation as inputs and return
    the base-ten logarithm of the rate. *Supersaturation* $S$ is the
    ratio of the current solute concentration to its equilibrium
    solubility at the current temperature, $S = c / c_{\text{sat}}(T)$;
    it is the thermodynamic driving force of the process and is the
    natural argument for both rate laws.

    Two predictors are constructed and grouped into a tuple
    `(growth_bp, nucleation_bp)`. Each is a `BoundedPredictor`, which
    wraps an `MLPPredictor` together with two `BoundScaler` objects.
    The input scaler maps the physical input variables to the unit
    interval through a sigmoid transform, ensuring that the network
    operates on inputs of comparable magnitude. The output scaler maps
    the unconstrained network output through a sigmoid into a
    user-specified physical range. Both predictors expect a dictionary
    keyed by covariate name; the `input_keys` field selects the
    relevant subset for each predictor.

    The choice of output bounds has a significant effect on
    optimisation. The growth rate is bounded to $[10^{-15}, 10^{-5}]$
    m/s, which places the midpoint of the sigmoid (the value attained
    by an unconstrained zero output) near $10^{-10}$ m/s. This is
    physically plausible for early-stage crystal growth and yields ODE
    trajectories that the solver can integrate within its step budget
    from random initial weights. Looser bounds that place the midpoint
    several orders of magnitude higher tend to produce stiff ODEs at
    initialisation and prevent training from making progress.

    ```python
    INPUT_KEYS              = ("temperature_C", "supersaturation")
    TEMPERATURE_BOUNDS      = (13.0, 27.0)        # °C, slightly wider than data span
    SUPERSATURATION_BOUNDS  = (0.0, 12.0)         # S = conc / conc_sat
    LOG10_GROWTH_BOUNDS     = (-15.0, -5.0)       # log10(G [m/s])
    LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)        # log10(J [#/(m^3·s)])

    in_scaler = BoundScaler(
        bounds=(TEMPERATURE_BOUNDS, SUPERSATURATION_BOUNDS),
        transform="sigmoid",
    )
    k_growth, k_nucleation = jr.split(jr.PRNGKey(0), 2)

    growth_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=in_scaler,
        inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                           depth=1, activation_name="relu", key=k_growth),
        out_scaler=BoundScaler(bounds=(LOG10_GROWTH_BOUNDS,), transform="sigmoid"),
    )
    nucleation_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=in_scaler,
        inner=MLPPredictor(in_size=2, out_size=1, width_size=64,
                           depth=1, activation_name="relu", key=k_nucleation),
        out_scaler=BoundScaler(bounds=(LOG10_NUCLEATION_BOUNDS,), transform="sigmoid"),
    )
    mlp_predictors = (growth_bp, nucleation_bp)
    ```
    """)
    return


@app.cell
def _mlp_predictors(BoundScaler, BoundedPredictor, MLPPredictor, jr):
    INPUT_KEYS = ("temperature_C", "supersaturation")
    TEMPERATURE_BOUNDS = (13.0, 27.0)  # °C, slightly wider than data span
    SUPERSATURATION_BOUNDS = (0.0, 12.0)  # S = conc / conc_sat
    LOG10_GROWTH_BOUNDS = (-15.0, -5.0)  # log10(G [m/s])
    LOG10_NUCLEATION_BOUNDS = (-6.5, 20.0)  # log10(J [#/(m^3·s)])

    in_scaler = BoundScaler(
        bounds=(TEMPERATURE_BOUNDS, SUPERSATURATION_BOUNDS),
        transform="sigmoid",
    )

    k_growth, k_nucleation = jr.split(jr.PRNGKey(0), 2)
    growth_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=in_scaler,
        inner=MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=1,
            activation_name="relu",
            key=k_growth,
        ),
        out_scaler=BoundScaler(bounds=(LOG10_GROWTH_BOUNDS,), transform="sigmoid"),
    )
    nucleation_bp = BoundedPredictor(
        input_keys=INPUT_KEYS,
        in_scaler=in_scaler,
        inner=MLPPredictor(
            in_size=2,
            out_size=1,
            width_size=64,
            depth=1,
            activation_name="relu",
            key=k_nucleation,
        ),
        out_scaler=BoundScaler(bounds=(LOG10_NUCLEATION_BOUNDS,), transform="sigmoid"),
    )
    mlp_predictors = (growth_bp, nucleation_bp)
    print("predictor pair: (growth_bp, nucleation_bp)")
    print(f"  growth_bp.out bounds: {LOG10_GROWTH_BOUNDS}")
    print(f"  nucleation_bp.out bounds: {LOG10_NUCLEATION_BOUNDS}")
    return (mlp_predictors,)


@app.cell(hide_code=True)
def _mlp_vector_field_md(mo):
    mo.md(r"""
    ### Vector field for the hybrid MLP model

    The user supplies a `simulate_fn` that takes the trainable
    component, a time vector, the experimental covariates, the initial
    state, and a `SolverConfig`, and returns the integrated state
    trajectory. The signature is fixed by the library so that training
    and prediction utilities can call it generically; the body is
    free.

    Two implementation details are worth highlighting. First, both
    rate functions are gated by a *metastable mask*
    `(S > 1 + 1e-5)`. The *metastable limit* is the supersaturation
    boundary below which neither nucleation nor growth occurs; below
    this limit the solute is effectively in equilibrium or
    undersaturated, and the ODE state should remain stationary. Casting
    the mask to a float and multiplying both rates by it forces the
    derivatives to vanish in this regime, preventing spurious
    dissolution dynamics from being modelled by rates that are intended
    to describe nucleation and growth only.

    Second, the dataset records time in minutes whereas the rate
    constants are expressed in SI seconds. The conversion is performed
    at the boundary of the simulator by multiplying the time vector by
    sixty before passing it to the integrator, keeping the vector
    field itself free of unit conversions.

    ```python
    RHO_C    = 1370.0  # crystal density [kg/m^3]
    K_V      = 0.81    # volumetric shape factor
    META_EPS = 1e-5    # supersaturation must exceed 1 + eps for nucleation/growth

    def simulate_fn_mlp(predictors, ts, covariates, y0, solver):
        growth_bp, nucleation_bp = predictors
        temperature_C = covariates["temperature_C"]
        # Empirical solubility polynomial in °C.
        conc_sat = (0.3705
                    + 7.171e-2 * temperature_C
                    - 1.924e-3 * temperature_C**2
                    + 17.97e-5 * temperature_C**3)

        def vector_field(t, y, args):
            mu0, mu1, mu2, mu3, _mu4, conc = y
            S = conc / conc_sat
            meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

            inputs = {"temperature_C": temperature_C, "supersaturation": S}
            log10_G = jnp.squeeze(growth_bp(inputs))
            log10_J = jnp.squeeze(nucleation_bp(inputs))
            G = meta_mask * jnp.power(10.0, log10_G)
            J = meta_mask * jnp.power(10.0, log10_J)

            return jnp.stack([
                J,
                G * mu0,
                2.0 * G * mu1,
                3.0 * G * mu2,
                4.0 * G * mu3,
                -3.0 * K_V * RHO_C * G * mu2,
            ])

        times_sec = ts * 60.0  # dataset stores minutes; rate constants are in seconds
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver.solver,
            t0=times_sec[0], t1=times_sec[-1], dt0=solver.dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=times_sec),
            stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
            max_steps=solver.max_steps,
            adjoint=diffrax.DirectAdjoint(),
        )
        return sol.ys
    ```
    """)
    return


@app.cell
def _mlp_simulate_fn(
    Array,
    BoundedPredictor,
    Float,
    SolverConfig,
    diffrax,
    jnp,
):
    RHO_C = 1370.0  # crystal density [kg/m^3]
    K_V = 0.81  # volumetric shape factor
    META_EPS = 1e-5  # supersaturation must exceed 1 + eps for nucleation/growth

    def simulate_fn_mlp(
        predictors: tuple[BoundedPredictor, BoundedPredictor],
        ts: Float[Array, " T"],
        covariates: dict[str, Array],
        y0: Float[Array, " 6"],
        solver: SolverConfig,
    ) -> Float[Array, "T 6"]:
        growth_bp, nucleation_bp = predictors
        temperature_C = covariates["temperature_C"]
        # Empirical solubility polynomial in °C.
        conc_sat = (
            0.3705
            + 7.171e-2 * temperature_C
            - 1.924e-3 * temperature_C**2
            + 17.97e-5 * temperature_C**3
        )

        def vector_field(t: Array, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
            mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
            S = conc / conc_sat
            meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

            inputs = {"temperature_C": temperature_C, "supersaturation": S}
            log10_G = jnp.squeeze(growth_bp(inputs))
            log10_J = jnp.squeeze(nucleation_bp(inputs))
            G = meta_mask * jnp.power(10.0, log10_G)
            J = meta_mask * jnp.power(10.0, log10_J)

            dmu0 = J
            dmu1 = G * mu0
            dmu2 = 2.0 * G * mu1
            dmu3 = 3.0 * G * mu2
            dmu4 = 4.0 * G * mu3
            dconc = -3.0 * K_V * RHO_C * G * mu2
            return jnp.stack([dmu0, dmu1, dmu2, dmu3, dmu4, dconc])

        # Dataset stores time in minutes; rate constants are SI-second-based.
        times_sec = ts * 60.0
        if solver.dt0 is None:
            span = times_sec[-1] - times_sec[0]
            dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
        else:
            dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)

        atol = (
            jnp.asarray(solver.atol, dtype=times_sec.dtype)
            if isinstance(solver.atol, tuple)
            else solver.atol
        )

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver.solver,
            t0=times_sec[0],
            t1=times_sec[-1],
            dt0=dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=times_sec),
            stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=atol),
            max_steps=solver.max_steps,
            adjoint=diffrax.DirectAdjoint(),
        )
        return jnp.asarray(sol.ys)

    return (simulate_fn_mlp,)


@app.cell(hide_code=True)
def _mlp_train_md(mo):
    mo.md(r"""
    ### Training the hybrid MLP

    The MLP predictors are fitted by minimising the mean-squared error
    between the simulated and observed channels. Optimisation uses 300
    iterations of the Adam variant of stochastic gradient descent
    through the `train_with_optax` driver. The first iteration incurs
    a one-time just-in-time compilation cost per bucket shape;
    subsequent iterations run at the full speed of compiled JAX code.

    ```python
    config_mlp = OptaxTrainingConfig(
        steps=(300,),
        lr=(1e-3,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
    )
    history_mlp, trained_mlp = train_with_optax(
        mlp_predictors,
        dataset,
        config_mlp,
        simulate_fn=simulate_fn_mlp,
        solver=solver,
        key=jr.PRNGKey(0),
    )
    ```
    """)
    return


@app.cell
def _mlp_train(
    OptaxTrainingConfig,
    dataset,
    jr,
    mlp_predictors,
    simulate_fn_mlp,
    solver,
    train_with_optax,
):
    MLP_STEPS = 300
    config_mlp = OptaxTrainingConfig(
        steps=(MLP_STEPS,),
        lr=(1e-3,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=False,
    )
    history_mlp, trained_mlp = train_with_optax(
        mlp_predictors,
        dataset,
        config_mlp,
        simulate_fn=simulate_fn_mlp,
        solver=solver,
        key=jr.PRNGKey(0),
    )
    print(f"MLP final loss: {history_mlp[-1]:.6f}  ({len(history_mlp)} steps)")
    _sample_every = max(1, len(history_mlp) // 8)
    print("MLP loss trace: " + ", ".join(f"{loss:.4f}" for loss in history_mlp[::_sample_every]))
    return history_mlp, trained_mlp


@app.cell
def _mlp_loss_curve(history_mlp, plt):
    fig_loss_mlp, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(history_mlp, color="tab:blue", linewidth=1.2)
    _ax.set_xlabel("step")
    _ax.set_ylabel("MSE loss")
    _ax.set_yscale("log")
    _ax.set_title("Hybrid MLP — training loss")
    _ax.grid(alpha=0.3)
    fig_loss_mlp.tight_layout()
    fig_loss_mlp
    return


@app.cell
def _mlp_predict(
    dataset,
    predict_dataset,
    simulate_fn_mlp,
    solver,
    trained_mlp,
):
    predictions_mlp = predict_dataset(
        trained_mlp,
        dataset,
        simulate_fn=simulate_fn_mlp,
        solver=solver,
    )
    return (predictions_mlp,)


@app.cell(hide_code=True)
def _mech_section_md(mo):
    mo.md(r"""
    # Part 2: the mechanistic model (CNT and power-law growth)

    The same dataset, ODE backbone, projector, and solver are now
    paired with a fully mechanistic specification of the rate laws.
    Classical Nucleation Theory (CNT) provides a physically motivated
    expression for the nucleation rate $J$ as a function of
    supersaturation, temperature, and the interfacial energy of the
    solid-liquid interface. The growth rate $G$ is described by a
    power law in the supersaturation excess. Together these introduce
    only four scalar parameters and no explicit dependence on
    experimental covariates beyond temperature and supersaturation.

    The rate laws are

    $$
    J = \exp(\log A) \cdot S \cdot
    \exp\!\left(-\frac{16\pi\,\gamma^3 v^2}{3 (k_B T)^3 \ln^2 S}\right)
    $$

    $$
    G = \frac{10^{A_g}}{60} \cdot \max(S - 1,\ 0)^{g}
    $$

    where $v$ is the molecular volume of the solute, $k_B$ is the
    Boltzmann constant, and $T$ is the absolute temperature. The four
    fitted parameters and their physical bounds are summarised below.

    | symbol  | physical meaning                       | bounds          |
    |---------|----------------------------------------|-----------------|
    | `logA`  | $\ln A$, CNT pre-exponential           | (20.0, 65.0)    |
    | `gamma` | interfacial energy [mJ/m²]             | (0.15, 1.0)     |
    | `Ag`    | $\log_{10}$ growth pre-factor [m/s]    | (-20.0, -5.0)   |
    | `g`     | power-law growth exponent              | (1.0, 3.5)      |

    The `BoundedPredictor` class assumes a callable signature of the
    form `(dict | Array) -> Array`, intended for predictors that
    consume covariates. The four mechanistic parameters here are
    global, in the sense that they do not depend on any covariate, so
    a custom `eqx.Module` named `KineticParameters` is defined
    instead. It contains a four-element vector of unconstrained latent
    parameters and a `BoundScaler` that maps these onto the physical
    bounds of each parameter. Using the same `BoundScaler` primitive
    employed by the hybrid model ensures that the optimiser operates
    in an unbounded latent space while the simulator receives
    parameters in physical units.

    ```python
    LOGA_BOUNDS  = (20.0, 65.0)
    GAMMA_BOUNDS = (0.15, 1.0)
    AG_BOUNDS    = (-20.0, -5.0)
    G_BOUNDS     = (1.0, 3.5)

    class KineticParameters(eqx.Module):
        # Four global mechanistic kinetic constants [logA, gamma, Ag, g].

        latent: Float[Array, " 4"]
        out_scaler: BoundScaler

        def __init__(self, *, key):
            self.latent = jr.normal(key, (4,)) * 0.1
            self.out_scaler = BoundScaler(
                bounds=(LOGA_BOUNDS, GAMMA_BOUNDS, AG_BOUNDS, G_BOUNDS),
                transform="sigmoid",
            )

        def __call__(self):
            return self.out_scaler.from_latent(self.latent)

    mech_predictor = KineticParameters(key=jr.PRNGKey(0))
    ```
    """)
    return


@app.cell
def _kinetic_parameters(Array, BoundScaler, Float, eqx, jr):
    LOGA_BOUNDS = (20.0, 65.0)
    GAMMA_BOUNDS = (0.15, 1.0)
    AG_BOUNDS = (-20.0, -5.0)
    G_BOUNDS = (1.0, 3.5)

    class KineticParameters(eqx.Module):
        """Four global mechanistic kinetic constants [logA, gamma, Ag, g]."""

        latent: Float[Array, " 4"]
        out_scaler: BoundScaler

        def __init__(self, *, key: Array) -> None:
            # Small Gaussian init in latent space puts the physical
            # parameters near the centre of each bound at gen 0;
            # CMA-ES expands from there under ``sigma_init``.
            self.latent = jr.normal(key, (4,)) * 0.1
            self.out_scaler = BoundScaler(
                bounds=(LOGA_BOUNDS, GAMMA_BOUNDS, AG_BOUNDS, G_BOUNDS),
                transform="sigmoid",
            )

        def __call__(self) -> Float[Array, " 4"]:
            return self.out_scaler.from_latent(self.latent)

    mech_predictor = KineticParameters(key=jr.PRNGKey(0))
    _init = mech_predictor()
    print(
        "init params: "
        f"logA={float(_init[0]):.2f}, "
        f"gamma={float(_init[1]):.3f} mJ/m^2, "
        f"Ag={float(_init[2]):.2f}, "
        f"g={float(_init[3]):.2f}"
    )
    return KineticParameters, mech_predictor


@app.cell(hide_code=True)
def _mech_vector_field_md(mo):
    mo.md(r"""
    ### Vector field for the mechanistic model

    Because the four mechanistic parameters are global, the predictor
    is evaluated once at the top of the simulator and the resulting
    physical-units values are closed over by the inner vector field.
    The metastable mask `(S > 1 + 1e-5)` again zeroes both rates below
    the metastable limit.

    The mechanistic vector field differs from the hybrid one in one
    important respect, namely the protective clipping of the
    supersaturation before it enters the logarithm in the CNT
    expression. When $S \leq 1$ the unguarded $\log(S)$ is non-finite,
    and although the metastable mask would set the rates to zero in
    the forward pass, the gradient of $\log$ through this branch
    remains undefined and propagates as NaN values during reverse-mode
    differentiation. Replacing $S$ by $\max(S, 1 + 10^{-12})$ inside
    the logarithm ensures a well-defined gradient even though
    population-level training of this model uses CMA-ES rather than
    gradient descent. The same clipping practice is recommended
    whenever the same simulator is reused with a gradient-based
    optimiser.

    ```python
    M_V = 2.97e-26          # molecular volume [m^3]
    K_B = 1.38064852e-23    # Boltzmann constant [J/K]

    def simulate_fn_mech(predictor, ts, covariates, y0, solver):
        params = predictor()                   # [4] in physical units
        logA       = params[0]
        gamma_J_m2 = params[1] * 1e-3          # bounds [mJ/m^2]; CNT in [J/m^2]
        Ag         = params[2]
        g_exp      = params[3]

        T_K = covariates["temperature_C"] + 273.15
        # ... conc_sat polynomial as in the hybrid simulator ...

        def vector_field(t, y, args):
            mu0, mu1, mu2, mu3, _mu4, conc = y
            S = conc / conc_sat
            meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

            # Clip S inside log so the gradient stays finite when meta_mask
            # is zero. log(S) for S <= 1 still produces a NaN gradient
            # otherwise, which propagates regardless of the mask.
            S_safe = jnp.clip(S, min=1.0 + 1e-12)
            logS   = jnp.log(S_safe)
            cnt_exp = (-16.0 * jnp.pi * gamma_J_m2**3 * M_V**2
                       / (3.0 * (K_B * T_K)**3 * logS**2))
            J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exp)

            growth_drive = jnp.maximum(S - 1.0, 0.0)
            G = (meta_mask * jnp.power(10.0, Ag) / 60.0
                 * jnp.power(growth_drive, g_exp))

            return jnp.stack([
                J, G * mu0, 2.0 * G * mu1, 3.0 * G * mu2, 4.0 * G * mu3,
                -3.0 * K_V * RHO_C * G * mu2,
            ])
        # ... diffeqsolve call identical to the hybrid simulator ...
    ```
    """)
    return


@app.cell
def _mech_simulate_fn(
    Array,
    Float,
    KineticParameters,
    SolverConfig,
    diffrax,
    jnp,
):
    RHO_C_M = 1370.0  # crystal density [kg/m^3]
    K_V_M = 0.81  # volumetric shape factor
    M_V = 2.97e-26  # molecular volume [m^3]
    K_B = 1.38064852e-23  # Boltzmann constant [J/K]
    META_EPS_M = 1e-5

    def simulate_fn_mech(
        predictor: KineticParameters,
        ts: Float[Array, " T"],
        covariates: dict[str, Array],
        y0: Float[Array, " 6"],
        solver: SolverConfig,
    ) -> Float[Array, "T 6"]:
        params = predictor()  # [4] in physical units, sigmoid-bounded
        logA = params[0]
        gamma_J_m2 = params[1] * 1e-3  # bounds [mJ/m^2]; CNT in [J/m^2]
        Ag = params[2]
        g_exp = params[3]

        temperature_C = covariates["temperature_C"]
        T_K = temperature_C + 273.15
        conc_sat = (
            0.3705
            + 7.171e-2 * temperature_C
            - 1.924e-3 * temperature_C**2
            + 17.97e-5 * temperature_C**3
        )

        def vector_field(t: Array, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
            mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
            S = conc / conc_sat
            meta_mask = (S > 1.0 + META_EPS_M).astype(y.dtype)

            S_safe = jnp.clip(S, min=1.0 + 1e-12)
            logS = jnp.log(S_safe)
            cnt_exp = -16.0 * jnp.pi * gamma_J_m2**3 * M_V**2 / (3.0 * (K_B * T_K) ** 3 * logS**2)
            J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exp)

            growth_drive = jnp.maximum(S - 1.0, 0.0)
            G = meta_mask * jnp.power(10.0, Ag) / 60.0 * jnp.power(growth_drive, g_exp)

            dmu0 = J
            dmu1 = G * mu0
            dmu2 = 2.0 * G * mu1
            dmu3 = 3.0 * G * mu2
            dmu4 = 4.0 * G * mu3
            dconc = -3.0 * K_V_M * RHO_C_M * G * mu2
            return jnp.stack([dmu0, dmu1, dmu2, dmu3, dmu4, dconc])

        times_sec = ts * 60.0
        if solver.dt0 is None:
            span = times_sec[-1] - times_sec[0]
            dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
        else:
            dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)

        atol = (
            jnp.asarray(solver.atol, dtype=times_sec.dtype)
            if isinstance(solver.atol, tuple)
            else solver.atol
        )

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver.solver,
            t0=times_sec[0],
            t1=times_sec[-1],
            dt0=dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=times_sec),
            stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=atol),
            max_steps=solver.max_steps,
            adjoint=diffrax.DirectAdjoint(),
        )
        return jnp.asarray(sol.ys)

    return (simulate_fn_mech,)


@app.cell(hide_code=True)
def _mech_train_md(mo):
    mo.md(r"""
    ### Training the mechanistic model with CMA-ES

    The mechanistic model is fitted by Covariance Matrix Adaptation
    Evolution Strategy (CMA-ES), a derivative-free optimiser that
    maintains a population of candidate parameter vectors and
    iteratively updates a sampling distribution to favour
    well-performing candidates. CMA-ES is well suited to small,
    bounded, possibly non-smooth objective surfaces of the kind
    arising here, and it sidesteps the need for backpropagation
    through the ODE solver.

    The initial population of 32 individuals is drawn by Latin
    hypercube sampling across the full bound box, which provides
    space-filling coverage of the parameter space. Optimisation
    proceeds for 30 generations with an initial standard deviation of
    `sigma_init=0.5` in the latent space. Population evaluation is
    vectorised through `jax.vmap` so that every individual is
    simulated against every bucket within a single compiled kernel.

    ```python
    config_mech = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",          # space-filling Latin-hypercube init
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
    )
    history_mech, trained_mech = train_with_evosax(
        mech_predictor,
        dataset,
        config_mech,
        simulate_fn=simulate_fn_mech,
        solver=solver,
        key=jr.PRNGKey(0),
    )
    ```
    """)
    return


@app.cell
def _mech_train(
    EvosaxTrainingConfig,
    dataset,
    jr,
    mech_predictor,
    simulate_fn_mech,
    solver,
    train_with_evosax,
):
    MECH_POP = 32
    MECH_GENS = 30
    config_mech = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=MECH_POP,
        num_generations=MECH_GENS,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )
    history_mech, trained_mech = train_with_evosax(
        mech_predictor,
        dataset,
        config_mech,
        simulate_fn=simulate_fn_mech,
        solver=solver,
        key=jr.PRNGKey(0),
    )
    print(f"mechanistic final best loss: {history_mech[-1]:.6f}  ({len(history_mech)} generations)")
    _sample_every = max(1, len(history_mech) // 8)
    print(
        "mech best-loss trace: "
        + ", ".join(f"{loss:.4f}" for loss in history_mech[::_sample_every])
    )
    _final = trained_mech()
    print(
        f"trained -> logA={float(_final[0]):.2f}, "
        f"gamma={float(_final[1]):.3f} mJ/m^2, "
        f"Ag={float(_final[2]):.2f}, "
        f"g={float(_final[3]):.2f}"
    )
    return history_mech, trained_mech


@app.cell
def _mech_loss_curve(history_mech, plt):
    fig_loss_mech, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(history_mech, color="tab:purple", linewidth=1.2)
    _ax.set_xlabel("generation")
    _ax.set_ylabel("best MSE loss")
    _ax.set_yscale("log")
    _ax.set_title("Mechanistic — CMA-ES best-of-population loss")
    _ax.grid(alpha=0.3)
    fig_loss_mech.tight_layout()
    fig_loss_mech
    return


@app.cell
def _mech_predict(
    dataset,
    predict_dataset,
    simulate_fn_mech,
    solver,
    trained_mech,
):
    predictions_mech = predict_dataset(
        trained_mech,
        dataset,
        simulate_fn=simulate_fn_mech,
        solver=solver,
    )
    return (predictions_mech,)


@app.cell(hide_code=True)
def _compare_md(mo):
    mo.md(r"""
    # Part 3: side-by-side comparison

    Both models have now been fitted to the same dataset using the
    same ODE backbone, projector, solver, and diagnostics. Only the
    trainable component and the optimisation procedure differ between
    the two pipelines, allowing the contribution of the data-driven
    rate laws to be isolated from numerical and structural choices.
    The table below summarises the differences.

    | Aspect              | Hybrid MLP                          | Mechanistic                         |
    |---------------------|-------------------------------------|-------------------------------------|
    | Trainable component | `(growth_bp, nucleation_bp)` MLPs   | one `KineticParameters` (4 scalars) |
    | Inputs              | `(temperature_C, supersaturation)`  | none (global parameters)            |
    | Rate laws           | learned $\log_{10} G,\ \log_{10} J$ | CNT-J + power-law-G (parametric)    |
    | Trainer             | `train_with_optax` (Adam/AdamW)     | `train_with_evosax` (CMA-ES)        |
    | Param count         | ~thousands per branch               | 4                                   |
    """)
    return


@app.cell
def _diagnostics_both(dataset, np, predictions_mech, predictions_mlp):
    def channel_stats(obs, pred):
        n = int(obs.shape[0])
        if n == 0:
            return {
                "n": 0,
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
            }
        residuals = pred - obs
        mse = float(np.mean(residuals**2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else float("nan")
        return {
            "n": n,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(residuals))),
            "r2": r2,
        }

    def gather(predictions):
        out: dict[str, dict] = {}
        for _d, _name in enumerate(dataset.output_channel_names):
            _obs_chunks: list[np.ndarray] = []
            _pred_chunks: list[np.ndarray] = []
            for _pred_b, _bp in zip(predictions, dataset.bucket_payloads, strict=True):
                _mask_d = np.asarray(_bp.mask[..., _d], dtype=bool)
                _obs_chunks.append(np.asarray(_bp.y_observed[..., _d])[_mask_d])
                _pred_chunks.append(np.asarray(_pred_b[..., _d])[_mask_d])
            _obs = np.concatenate(_obs_chunks)
            _pred = np.concatenate(_pred_chunks)
            out[_name] = {**channel_stats(_obs, _pred), "obs": _obs, "pred": _pred}
        return out

    diag_mlp = gather(predictions_mlp)
    diag_mech = gather(predictions_mech)

    print(f"  {'channel':<6} {'model':<7} {'n':>4} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'R^2':>8}")
    for _name in dataset.output_channel_names:
        for _label, _diag in (("MLP", diag_mlp), ("mech", diag_mech)):
            _s = _diag[_name]
            _r2 = "nan" if _s["r2"] != _s["r2"] else f"{_s['r2']:.4f}"
            print(
                f"  {_name:<6} {_label:<7} {_s['n']:>4d} "
                f"{_s['mse']:>12.4e} {_s['rmse']:>12.4e} "
                f"{_s['mae']:>12.4e} {_r2:>8}"
            )
    return diag_mech, diag_mlp


@app.cell(hide_code=True)
def _compare_parity_md(mo):
    mo.md(r"""
    ### Parity plot

    The parity plot displays predicted against observed values for
    each output channel, with the identity line $y = x$ shown for
    reference; points lying on this line correspond to perfect
    agreement between model and data. The hybrid MLP places the
    concentration scatter closer to the identity line, reflecting the
    additional flexibility provided by the neural rate laws. The
    mechanistic model accepts a degree of additional misfit in
    exchange for a parsimonious description of the system in terms of
    four physically interpretable scalars rather than thousands of
    network weights.
    """)
    return


@app.cell
def _parity_overlay(diag_mech, diag_mlp, plt):
    _channels = list(diag_mlp.keys())
    fig_par, axes_par = plt.subplots(
        1, len(_channels), figsize=(4 * len(_channels), 4), squeeze=False
    )
    for _ax, _name in zip(axes_par.flatten(), _channels):
        _smlp = diag_mlp[_name]
        _smech = diag_mech[_name]
        _ax.scatter(
            _smlp["obs"],
            _smlp["pred"],
            s=22,
            alpha=0.75,
            color="tab:blue",
            label="hybrid MLP",
        )
        _ax.scatter(
            _smech["obs"],
            _smech["pred"],
            s=22,
            alpha=0.75,
            color="tab:purple",
            marker="^",
            label="mechanistic",
        )
        _all_obs = [_smlp["obs"], _smech["obs"], _smlp["pred"], _smech["pred"]]
        _lo = float(min(arr.min() for arr in _all_obs))
        _hi = float(max(arr.max() for arr in _all_obs))
        if _lo == _hi:
            _pad = 1.0 if _lo == 0.0 else abs(_lo) * 0.1
            _lo, _hi = _lo - _pad, _hi + _pad
        _ax.plot([_lo, _hi], [_lo, _hi], color="black", linestyle="--", linewidth=0.8)
        _ax.set_xlabel("observed")
        _ax.set_ylabel("predicted")
        _r2_mlp = "nan" if _smlp["r2"] != _smlp["r2"] else f"{_smlp['r2']:.3f}"
        _r2_mech = "nan" if _smech["r2"] != _smech["r2"] else f"{_smech['r2']:.3f}"
        _ax.set_title(f"{_name}\nMLP R²={_r2_mlp} | mech R²={_r2_mech}")
        _ax.grid(alpha=0.3)
        _ax.legend(loc="best", fontsize=8)
    fig_par.suptitle("Parity — hybrid MLP vs. mechanistic")
    fig_par.tight_layout()
    fig_par
    return


@app.cell(hide_code=True)
def _compare_traj_md(mo):
    mo.md(r"""
    ### Trajectory comparison

    The figure below shows one row per experiment and one column per
    output channel. The two trained models are re-simulated on a dense
    time grid so that the curvature of the trajectory between sparse
    measurements is visible, and the observed values are overlaid as
    points. This visualisation makes it possible to assess
    qualitatively the regimes in which each model captures the
    transient behaviour of the concentration and the terminal particle
    size, and where they diverge.
    """)
    return


@app.cell
def _trajectory_overlay(
    dataset,
    jnp,
    np,
    plt,
    predictions_mlp,
    simulate_fn_mech,
    simulate_fn_mlp,
    solver,
    trained_mech,
    trained_mlp,
):
    rows: list[dict] = []
    for _pred_mlp, _bp in zip(predictions_mlp, dataset.bucket_payloads, strict=True):
        _N = _bp.ts.shape[0]
        for _i in range(_N):
            _ts_obs = np.asarray(_bp.ts[_i])
            _t0, _t1 = float(_ts_obs[0]), float(_ts_obs[-1])
            _dense_grid = np.linspace(_t0, _t1, 200)
            _ts_dense = np.unique(np.concatenate([_ts_obs, _dense_grid]))
            _ts_jax = jnp.asarray(_ts_dense)
            _cov_i = {k: v[_i] for k, v in _bp.covariates.items()}

            _state_mlp = simulate_fn_mlp(trained_mlp, _ts_jax, _cov_i, _bp.y0[_i], solver)
            _y_mlp = np.asarray(dataset.state_to_output(_state_mlp))
            _state_mech = simulate_fn_mech(trained_mech, _ts_jax, _cov_i, _bp.y0[_i], solver)
            _y_mech = np.asarray(dataset.state_to_output(_state_mech))

            rows.append(
                {
                    "ts_obs": _ts_obs,
                    "y_obs": np.asarray(_bp.y_observed[_i]),
                    "mask": np.asarray(_bp.mask[_i], dtype=bool),
                    "ts_pred": _ts_dense,
                    "y_mlp": _y_mlp,
                    "y_mech": _y_mech,
                }
            )

    _channels = list(dataset.output_channel_names)
    fig_traj, axes_traj = plt.subplots(
        len(rows),
        len(_channels),
        figsize=(4 * len(_channels), 2.4 * max(len(rows), 1)),
        squeeze=False,
    )
    for _r, _row in enumerate(rows):
        for _c, _name in enumerate(_channels):
            _ax = axes_traj[_r][_c]
            _ax.plot(
                _row["ts_pred"],
                _row["y_mlp"][:, _c],
                label="hybrid MLP",
                linewidth=1.5,
                color="tab:blue",
            )
            _ax.plot(
                _row["ts_pred"],
                _row["y_mech"][:, _c],
                label="mechanistic",
                linewidth=1.5,
                color="tab:purple",
                linestyle="--",
            )
            _mc = _row["mask"][:, _c]
            if bool(_mc.any()):
                _ax.scatter(
                    _row["ts_obs"][_mc],
                    _row["y_obs"][_mc, _c],
                    s=22,
                    marker="o",
                    label="observed",
                    zorder=3,
                    color="black",
                )
            if _r == 0:
                _ax.set_title(_name)
            if _r == len(rows) - 1:
                _ax.set_xlabel("t (min)")
            if _c == 0:
                _ax.set_ylabel(f"E{_r + 1}")
            if _r == 0 and _c == 0:
                _ax.legend(loc="best", fontsize=8)
            _ax.grid(alpha=0.3)
    fig_traj.suptitle("Trajectories — hybrid MLP vs. mechanistic")
    fig_traj.tight_layout()
    fig_traj
    return


@app.cell(hide_code=True)
def _outro(mo):
    mo.md(r"""
    ## Inspecting the trained rates

    Once training has completed, both predictors can be queried
    directly. The hybrid MLP predictors are callables that accept a
    dictionary of covariates and return the bounded logarithm of the
    corresponding rate, so any operating point
    $(T, S)$ can be evaluated. The mechanistic predictor takes no
    arguments and, when called, returns the four fitted physical
    parameters. Comparing these readouts side by side at a
    representative operating point is a useful sanity check on the
    learned rate laws and provides a starting point for any subsequent
    physical interpretation or extrapolation.

    ```python
    # Hybrid MLP: query at any (T, S) operating point.
    trained_growth, trained_nucleation = trained_mlp
    sample_inputs = {
        "temperature_C": jnp.asarray(20.0),
        "supersaturation": jnp.asarray(1.5),
    }
    log10_G = float(jnp.squeeze(trained_growth(sample_inputs)))
    log10_J = float(jnp.squeeze(trained_nucleation(sample_inputs)))

    # Mechanistic: predictor takes no arguments.
    final_params = trained_mech()
    logA, gamma, Ag, g = final_params
    ```
    """)
    return


@app.cell
def _readout(jnp, trained_mech, trained_mlp):
    trained_growth, trained_nucleation = trained_mlp
    sample_inputs = {
        "temperature_C": jnp.asarray(20.0),
        "supersaturation": jnp.asarray(1.5),
    }
    log10_G = float(jnp.squeeze(trained_growth(sample_inputs)))
    log10_J = float(jnp.squeeze(trained_nucleation(sample_inputs)))
    print("Hybrid MLP at T=20°C, S=1.5:")
    print(f"  log10_G = {log10_G:>7.3f}   ->  G = {10**log10_G:.3e} m/s")
    print(f"  log10_J = {log10_J:>7.3f}   ->  J = {10**log10_J:.3e} #/(m³·s)")

    _final = trained_mech()
    print("Mechanistic constants (no covariate dependence):")
    print(f"  logA  = {float(_final[0]):.2f}")
    print(f"  gamma = {float(_final[1]):.3f} mJ/m²")
    print(f"  Ag    = {float(_final[2]):.2f}")
    print(f"  g     = {float(_final[3]):.2f}")
    return


if __name__ == "__main__":
    app.run()
