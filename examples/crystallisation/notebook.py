r"""Crystallisation walkthrough — hybrid MLP and mechanistic side by side.

Runs the same dataset, ODE backbone, projector, solver, and diagnostics
through two different trainable components and compares them in one
place. Mirrors the prose in ``docs/examples/crystallisation.md`` and
``docs/examples/crystallisation-mechanistic.md``.

Run: ``uv run python examples/crystallisation/notebook.py``

# Crystallisation: a hybrid and a mechanistic model, side by side

Two kinetic models of the same batch crystallisation, fitted to the
same four experiments. *Crystallisation* is a dissolved solute leaving
solution as a solid phase, driven by *nucleation*, new crystals
appearing, and *growth*, existing ones enlarging. Models of those two
rates are central to designing and controlling pharmaceutical and
fine-chemical processes.

*Hybrid modelling* combines mechanistic structure, here the conservation
laws governing the crystal population, with data-driven components, here
networks that learn the rate laws. The backbone, solver and diagnostics
stay fixed while the trainable component is swapped:

1. **Hybrid MLP model.** Two small networks emit the logarithms of the
   growth and nucleation rates from temperature and supersaturation,
   each in a `BoundedPredictor` that confines the output to a plausible
   range. Trained with Adam through `train_with_optax`.
2. **Mechanistic model.** Classical Nucleation Theory for nucleation and
   a power law for growth: four scalar parameters, no covariate
   dependence, fitted with CMA-ES through `train_with_evosax`.

Both share a six-state *population balance* in the *method of moments*,
which tracks the crystal size distribution by its first few moments
$\mu_k(t)$, the integral of $L^k n(L,t)$ over size $L$. With the solute
concentration that gives:

$$
\begin{aligned}
\frac{d\mu_0}{dt} &= J(t) \\
\frac{d\mu_k}{dt} &= k\, G(t)\, \mu_{k-1}, \quad k = 1, 2, 3, 4 \\
\frac{d\,\text{conc}}{dt} &= -3\, K_v\, \rho_c\, G(t)\, \mu_2
\end{aligned}
$$

$G(t)$ is the linear growth velocity in m/s, $J(t)$ the nucleation rate
in crystals per cubic metre per second, $K_v$ a volumetric shape factor
and $\rho_c$ the crystal density. $G$ and $J$ are the unknowns, learned
or specified mechanistically. Two quantities are observed: solute
concentration, sampled densely, and the volume-weighted mean diameter
$d_{43} = (\mu_4 / \mu_3) \cdot 10^{6}$ in micrometres, measured only at
the end of each experiment.
"""

# ruff: noqa: F722

from pathlib import Path

import jax

# x64 must be enabled before any other JAX-touching import. The
# population-balance moments span ~18 decades during integration;
# float32 mass balance drifts visibly within a single experiment.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jax import Array  # noqa: E402
from jaxtyping import Float  # noqa: E402

from hybridmodels import (  # noqa: E402
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
from hybridmodels.training.evosax import (  # noqa: E402
    EvosaxTrainingConfig,
    train_with_evosax,
)
from hybridmodels.training.optax import (  # noqa: E402
    OptaxTrainingConfig,
    train_with_optax,
)


def main() -> None:
    Path("examples/crystallisation/figures").mkdir(parents=True, exist_ok=True)

    # ## The dataset
    # ------------------------------------------------------------------
    # Four experiments, two replicates at each of 17 °C and 21 °C. Each
    # records solute concentration at a series of times plus one
    # volume-weighted mean diameter $d_{43}$ at the end of the run, the
    # usual laboratory pattern: concentration monitored continuously by
    # a spectroscopic probe, particle size measured offline on the final
    # slurry.
    #
    # The concentration channel gets a uniform observation variance of
    # $0.1$; the $d_{43}$ channel carries its own per-experiment variance
    # from the offline sizing instrument.
    #
    #     | `exp_id` | T (°C) | n conc | n d43 | t span (min) | terminal d43 (µm) |
    #     |----------|-------:|-------:|------:|--------------|------------------:|
    #     | `E1`     | 17.0   | 9      | 1     | 0 → 270      | 9.2               |
    #     | `E2`     | 17.0   | 9      | 1     | 0 → 270      | 7.7               |
    #     | `E3`     | 21.0   | 7      | 1     | 0 → 360      | 10.5              |
    #     | `E4`     | 21.0   | 7      | 1     | 0 → 375      | 11.9              |
    #
    # The four do not share a time grid. `make_dataset` forms a union of
    # each experiment's sampling times and groups matching lengths into
    # *buckets* that vectorise together at training time, here two: the
    # pair at 17 °C and the pair at 21 °C.

    # -- 1. Data -------------------------------------------------------
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
    fig_raw.savefig("examples/crystallisation/figures/raw_data.png")
    plt.close(fig_raw)

    # ## Initial conditions and `Experiment` construction
    # -------------------------------------------------------------
    # An `Experiment` bundles the observations, the covariates, and a
    # `y0_fn` returning the ODE's initial state, evaluated once per
    # experiment when the dataset is built. The five moments
    # $\mu_0, \dots, \mu_4$ start at zero, since the suspension holds no
    # crystals at the start, and the concentration starts at its first
    # observed value. Temperature is the only covariate.

    def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 6"]:
        """[mu0..mu4, conc]: moments at zero, conc at first observation."""
        init_conc = jnp.asarray(channels["conc"].values[0])
        return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])

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

    # ## Mapping the ODE state to observable quantities
    # -----------------------------------------------------------------
    # The integrator returns all six states, the five moments and the
    # concentration, while the observed channels are concentration and
    # the volume-weighted mean diameter $d_{43} = \mu_4 / \mu_3$. The
    # projector `state_to_output` maps one to the other.
    #
    # $d_{43}$ needs care, because $\mu_3$ vanishes before anything has
    # nucleated. A naive `jnp.where(mu3 > eps, mu4 / mu3, 0.0)` is not
    # enough: JAX takes the safe branch forwards but still traces the
    # unsafe one for the gradient, where the division by zero returns NaN
    # that propagates back into the loss. The remedy is the
    # *double-`where`*: replace the denominator with a safe value first,
    # divide, then select. Value and gradient are then both well-defined
    # everywhere.

    D43_MU3_EPS = 1e-6  # mu3 floor for the d43 ratio
    D43_MAX = 55.0  # upper guard on d43 [um]

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

    # ## Bucketing the experiments with `make_dataset`
    # -------------------------------------------------------------
    # The call to `make_dataset` constructs a union time axis for each
    # experiment and groups experiments of matching length into buckets
    # that can be processed together as a single batched array. With
    # this dataset, two buckets are produced. The first contains the two
    # experiments at 17 °C, both of length 9 (the terminal $d_{43}$
    # measurement at 270 minutes coincides with an existing concentration
    # sample). The second contains the two experiments at 21 °C, of
    # length 8 (a $d_{43}$ measurement at 360 or 375 minutes extends the
    # seven-point concentration grid by one).
    #
    # The bucket shape is used as a key for just-in-time compilation:
    # the training step is compiled once per bucket and then reused
    # across every gradient update, which amortises the compilation cost
    # across the optimisation loop.
    dataset = make_dataset(
        experiments,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"{len(dataset.bucket_payloads)} bucket(s)")
    for _i, _bp in enumerate(dataset.bucket_payloads):
        print(
            f"  bucket {_i}: ts={tuple(_bp.ts.shape)}, "
            f"y_observed={tuple(_bp.y_observed.shape)}, "
            f"mask={tuple(_bp.mask.shape)}, n_obs={int(_bp.n_obs)}"
        )

    # ## Configuring the ODE solver
    # -----------------------------
    # The dynamical system is integrated with the explicit Runge-Kutta
    # method `Tsit5` from the `diffrax` library, with adaptive step-size
    # control governed by relative and absolute tolerances. The
    # population-balance moments $\mu_k$ have very different natural
    # magnitudes: across a full integration they span roughly eighteen
    # decades, so a single scalar absolute tolerance would either
    # over-resolve the small components or under-resolve the large ones.
    # `SolverConfig` accepts a tuple of per-state absolute tolerances
    # matched to the natural magnitude of each state variable.
    #
    # `Tsit5` drives the hybrid-MLP phase. The mechanistic phase uses the
    # implicit `Kvaerno3` instead: the CNT exponential stiffens the system
    # for generic CMA-ES candidates, and the notebook's solver prose names
    # an implicit method as exactly this remedy. Tolerances and the
    # per-state `atol` tuple are identical for both phases.
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        # per-state floor at ~9 decades below natural magnitude
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )
    solver_mech = SolverConfig(
        solver=diffrax.Kvaerno3(),
        rtol=1e-4,
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )

    # # Part 1: the hybrid MLP model
    # ------------------------------
    # Here the two unknown rate functions $G$ and $J$ are separate MLPs
    # taking temperature and supersaturation and returning the base-ten
    # logarithm of the rate. *Supersaturation* $S = c / c_{\text{sat}}(T)$
    # is the ratio of solute concentration to its equilibrium solubility,
    # the thermodynamic driving force and the natural argument for both
    # laws.
    #
    # The two predictors travel as a tuple `(growth_bp, nucleation_bp)`.
    # Each is a `BoundedPredictor` wrapping an `MLPPredictor` between two
    # `BoundScaler`s: the input scaler maps the physical variables into
    # the unit interval so the network sees comparable magnitudes, the
    # output scaler squashes its output into a physical range. Both take
    # a dictionary keyed by covariate name, and `input_keys` selects the
    # subset each one reads.
    #
    # Output bounds matter a great deal to optimisation. Growth is
    # bounded to $[10^{-15}, 10^{-5}]$ m/s, putting the sigmoid
    # midpoint, the value a zero output gives, near $10^{-10}$ m/s. That
    # is plausible for early-stage growth and integrable within the
    # solver's step budget from random weights. Looser bounds put the
    # midpoint decades higher, making the ODE stiff at initialisation
    # and stalling training.
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

    # ### Vector field for the hybrid MLP model
    # ----------------------------------------
    # The user supplies a `simulate_fn` taking the trainable component, a
    # time vector, the covariates, the initial state and a `SolverConfig`,
    # and returning the state trajectory. The library fixes the signature
    # so training and prediction can call it generically; the body is
    # free.
    #
    # Two details. Both rates are gated by a *metastable mask*
    # `(S > 1 + 1e-5)`. Below the metastable limit the solute is at
    # equilibrium or undersaturated and neither nucleation nor growth
    # occurs, so multiplying by the mask makes the derivatives vanish
    # there rather than letting rate laws meant for growth model
    # dissolution.
    #
    # And the dataset records minutes while the rate constants are in SI
    # seconds. The conversion happens at the simulator boundary, so the
    # vector field itself carries no unit conversions.
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

        def vector_field(
            t: Array, y: Float[Array, " 6"], args: object
        ) -> Float[Array, " 6"]:
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

    # ### Training the hybrid MLP
    # ---------------------------
    # The MLP predictors are fitted by minimising the mean-squared error
    # between the simulated and observed channels. Optimisation uses 300
    # iterations of the Adam variant of stochastic gradient descent
    # through the `train_with_optax` driver. The first iteration incurs
    # a one-time just-in-time compilation cost per bucket shape;
    # subsequent iterations run at the full speed of compiled JAX code.
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
        state_to_output=state_to_output,
        solver=solver,
        key=jr.PRNGKey(0),
    )
    print(f"MLP final loss: {history_mlp[-1]:.6f}  ({len(history_mlp)} steps)")
    _sample_every = max(1, len(history_mlp) // 8)
    print(
        "MLP loss trace: "
        + ", ".join(f"{loss:.4f}" for loss in history_mlp[::_sample_every])
    )

    fig_loss_mlp, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(history_mlp, color="tab:blue", linewidth=1.2)
    _ax.set_xlabel("step")
    _ax.set_ylabel("MSE loss")
    _ax.set_yscale("log")
    _ax.set_title("Hybrid MLP — training loss")
    _ax.grid(alpha=0.3)
    fig_loss_mlp.tight_layout()
    fig_loss_mlp.savefig("examples/crystallisation/figures/mlp_loss_curve.png")
    plt.close(fig_loss_mlp)

    predictions_mlp = predict_dataset(
        trained_mlp,
        dataset,
        simulate_fn=simulate_fn_mlp,
        state_to_output=state_to_output,
        solver=solver,
    )

    # # Part 2: the mechanistic model (CNT and power-law growth)
    # ----------------------------------------------------------
    # The same dataset, backbone, projector and solver, now with fully
    # mechanistic rate laws. Classical Nucleation Theory gives the
    # nucleation rate $J$ from supersaturation, temperature and the
    # interfacial energy of the solid-liquid interface; growth follows a
    # power law in the supersaturation excess. Four scalar parameters in
    # total.
    #
    # The rate laws are
    #
    # $$ J = \exp(\log A) \cdot S \cdot
    # \exp\!\left(-\frac{16\pi\,\gamma^3 v^2}{3 (k_B T)^3 \ln^2 S}\right) $$
    #
    # $$ G = \frac{10^{A_g}}{60} \cdot \max(S - 1,\ 0)^{g} $$
    #
    # where $v$ is the molecular volume of the solute, $k_B$ is the
    # Boltzmann constant, and $T$ is the absolute temperature. The four
    # fitted parameters and their physical bounds are summarised below.
    #
    # | symbol  | physical meaning                    | bounds        |
    # |---------|-------------------------------------|---------------|
    # | `logA`  | $\ln A$, CNT pre-exponential        | (20.0, 65.0)  |
    # | `gamma` | interfacial energy [mJ/m²]          | (0.15, 1.0)   |
    # | `Ag`    | $\log_{10}$ growth pre-factor [m/s] | (-20.0, -5.0) |
    # | `g`     | power-law growth exponent           | (1.0, 3.5)    |
    #
    # `BoundedPredictor` is built for predictors that consume covariates,
    # with a `(dict | Array) -> Array` signature. These four parameters
    # are global, so a small `eqx.Module` called `KineticParameters`
    # takes its place: a four-element latent vector and a `BoundScaler`
    # mapping it onto each parameter's physical bounds. Reusing the same
    # scaler primitive keeps the optimiser in an unbounded space while
    # the simulator receives physical units.
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

    # ### Vector field for the mechanistic model
    # ------------------------------------------
    # The four parameters are global, so the predictor is evaluated once
    # at the top of the simulator and its physical-units values are
    # closed over by the vector field. The metastable mask
    # `(S > 1 + 1e-5)` again zeroes both rates below the metastable
    # limit.
    #
    # One difference from the hybrid field: the supersaturation is
    # clipped before it enters the CNT logarithm. At $S \leq 1$ an
    # unguarded $\log(S)$ is non-finite, and while the mask zeroes the
    # rates going forwards, the gradient through that branch is undefined
    # and propagates as NaN. Replacing $S$ by $\max(S, 1 + 10^{-12})$
    # inside the logarithm keeps it well-defined. CMA-ES takes no
    # gradients here, but the same clipping is what makes the simulator
    # safe to reuse with one that does.
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

        def vector_field(
            t: Array, y: Float[Array, " 6"], args: object
        ) -> Float[Array, " 6"]:
            mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
            S = conc / conc_sat
            meta_mask = (S > 1.0 + META_EPS_M).astype(y.dtype)

            S_safe = jnp.clip(S, min=1.0 + 1e-12)
            logS = jnp.log(S_safe)
            cnt_exp = (
                -16.0
                * jnp.pi
                * gamma_J_m2**3
                * M_V**2
                / (3.0 * (K_B * T_K) ** 3 * logS**2)
            )
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

    # ### Training the mechanistic model with CMA-ES
    # ----------------------------------------------
    # CMA-ES is derivative-free: it keeps a population of candidate
    # parameter vectors and moves its sampling distribution towards the
    # good ones. That suits a small, bounded, possibly non-smooth surface
    # like this one, and it sidesteps backpropagation through the solver
    # entirely.
    #
    # The initial population of 32 individuals is drawn by Latin
    # hypercube sampling across the full bound box, which provides
    # space-filling coverage of the parameter space. Optimisation
    # proceeds for 30 generations with an initial standard deviation of
    # `sigma_init=0.5` in the latent space. Population evaluation is
    # vectorised through `jax.vmap` so that every individual is
    # simulated against every bucket within a single compiled kernel.
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
        state_to_output=state_to_output,
        solver=solver_mech,
        key=jr.PRNGKey(0),
    )
    print(
        f"mechanistic final best loss: {history_mech[-1]:.6f}  "
        f"({len(history_mech)} generations)"
    )
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

    fig_loss_mech, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(history_mech, color="tab:purple", linewidth=1.2)
    _ax.set_xlabel("generation")
    _ax.set_ylabel("best MSE loss")
    _ax.set_yscale("log")
    _ax.set_title("Mechanistic — CMA-ES best-of-population loss")
    _ax.grid(alpha=0.3)
    fig_loss_mech.tight_layout()
    fig_loss_mech.savefig("examples/crystallisation/figures/mech_loss_curve.png")
    plt.close(fig_loss_mech)

    predictions_mech = predict_dataset(
        trained_mech,
        dataset,
        simulate_fn=simulate_fn_mech,
        state_to_output=state_to_output,
        solver=solver_mech,
    )

    # # Part 3: side-by-side comparison
    # --------------------------------
    # Both models have now been fitted to the same dataset using the
    # same ODE backbone, projector, solver, and diagnostics. Only the
    # trainable component and the optimisation procedure differ between
    # the two pipelines, allowing the contribution of the data-driven
    # rate laws to be isolated from numerical and structural choices.
    # The table below summarises the differences.
    #
    # | Aspect | Hybrid MLP | Mechanistic |
    # | Trainable component | `(growth_bp, nucleation_bp)` MLPs |
    # |                     | one `KineticParameters` (4 scalars) |
    # | Inputs | `(temperature_C, supersaturation)` | none (global parameters) |
    # | Rate laws | learned $\log_{10} G,\ \log_{10} J$ | CNT-J + power-law-G (parametric) |
    # | Trainer | `train_with_optax` (Adam/AdamW) | `train_with_evosax` (CMA-ES) |
    # | Param count | ~thousands per branch | 4 |
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
            _n = int(_obs.shape[0])
            if _n == 0:
                _stats = {
                    "n": 0,
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                }
            else:
                _residuals = _pred - _obs
                _mse = float(np.mean(_residuals**2))
                _ss_tot = float(np.sum((_obs - _obs.mean()) ** 2))
                _r2 = (
                    1.0 - float(np.sum(_residuals**2)) / _ss_tot if _ss_tot > 0 else float("nan")
                )
                _stats = {
                    "n": _n,
                    "mse": _mse,
                    "rmse": float(np.sqrt(_mse)),
                    "mae": float(np.mean(np.abs(_residuals))),
                    "r2": _r2,
                }
            out[_name] = {**_stats, "obs": _obs, "pred": _pred}
        return out

    diag_mlp = gather(predictions_mlp)
    diag_mech = gather(predictions_mech)

    print(
        f"  {'channel':<6} {'model':<7} {'n':>4} "
        f"{'MSE':>12} {'RMSE':>12} {'MAE':>12} {'R^2':>8}"
    )
    for _name in dataset.output_channel_names:
        for _label, _diag in (("MLP", diag_mlp), ("mech", diag_mech)):
            _s = _diag[_name]
            _r2 = "nan" if _s["r2"] != _s["r2"] else f"{_s['r2']:.4f}"
            print(
                f"  {_name:<6} {_label:<7} {_s['n']:>4d} "
                f"{_s['mse']:>12.4e} {_s['rmse']:>12.4e} "
                f"{_s['mae']:>12.4e} {_r2:>8}"
            )

    # ### Parity plot
    # ---------------
    # The parity plot displays predicted against observed values for
    # each output channel, with the identity line $y = x$ shown for
    # reference; points lying on this line correspond to perfect
    # agreement between model and data. The hybrid MLP places the
    # concentration scatter closer to the identity line, reflecting the
    # additional flexibility provided by the neural rate laws. The
    # mechanistic model accepts a degree of additional misfit in
    # exchange for a parsimonious description of the system in terms of
    # four physically interpretable scalars rather than thousands of
    # network weights.
    _channels = list(diag_mlp.keys())
    fig_par, axes_par = plt.subplots(
        1, len(_channels), figsize=(4 * len(_channels), 4), squeeze=False
    )
    for _ax, _name in zip(axes_par.flatten(), _channels, strict=True):
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
    fig_par.savefig("examples/crystallisation/figures/parity.png")
    plt.close(fig_par)

    # ### Trajectory comparison
    # -------------------------
    # The figure below shows one row per experiment and one column per
    # output channel. The two trained models are re-simulated on a dense
    # time grid so that the curvature of the trajectory between sparse
    # measurements is visible, and the observed values are overlaid as
    # points. This visualisation makes it possible to assess
    # qualitatively the regimes in which each model captures the
    # transient behaviour of the concentration and the terminal particle
    # size, and where they diverge.
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
            _y_mlp = np.asarray(state_to_output(_state_mlp))
            _state_mech = simulate_fn_mech(trained_mech, _ts_jax, _cov_i, _bp.y0[_i], solver_mech)
            _y_mech = np.asarray(state_to_output(_state_mech))

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
    fig_traj.savefig("examples/crystallisation/figures/trajectories.png")
    plt.close(fig_traj)

    # ## Inspecting the trained rates
    # -------------------------------
    # Both trained predictors can be queried directly. The MLP predictors
    # take a dictionary of covariates and return the bounded log-rate, so
    # any operating point $(T, S)$ can be evaluated; the mechanistic
    # predictor takes no arguments and returns its four fitted
    # parameters. Comparing the two at a representative operating point is
    # a quick sanity check and the starting point for any physical
    # interpretation.
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


if __name__ == "__main__":
    main()