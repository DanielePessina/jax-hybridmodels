"""Generate presentation plots for the batch reactor hybrid example.

Four figures under ``figures/presentation/``, each three pH-bin columns wide
with four temperature curves per column. Scatter is the noisy observations,
the line is the model on a dense time grid. No truth curve: the message is
model against data.

* ``01_per_bin_arrhenius.png`` — three independent Arrhenius fits, one per
  pH bin, the classical dictionary baseline.
* ``02_joint_arrhenius.png`` — one Arrhenius fit on all twelve experiments.
  The trunk has no pH input, so the same curves appear in every column while
  the data they meet changes. This is what the residual MLP has to correct.
* ``02b_linear_ph_arrhenius.png`` — a smarter parametric guess,
  ``log_k_ref(pH) = a + b·pH`` with one shared Ea. The truth's effective Ea
  varies strongly with pH, so the best single Ea is a compromise that misses
  both edge bins, with opposite sign in each.
* ``03_hybrid.png`` — the full hybrid, trunk plus residual MLP, on all twelve
  experiments. One model, tracking the data across pH.

Run with::

    uv run python examples/batch_reactor/presentation_plots.py

About a minute on CPU: four CMA-ES fits and one AdamW phase.
"""

# ruff: noqa: F722

from pathlib import Path

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jaxtyping import Float
from scipy.stats import qmc

from jaxhybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    MLPPredictor,
    SolverConfig,
    freeze_modules_of_type,
    make_dataset,
    make_experiment,
    trainable_mask,
)
from jaxhybridmodels.training import (
    EvosaxTrainingConfig,
    OptaxTrainingConfig,
    train_with_evosax,
    train_with_optax,
)

# Truth and physics constants — only used for synthetic data generation.
T_REF = 298.15  # K (25 °C); centring temperature for Arrhenius
R_GAS = 8.314e-3  # kJ/(mol·K); pair with Ea in kJ/mol
EA_TRUE = 30.0  # kJ/mol

K_SAT_BASELINE = 0.14
K_SAT_AMPLITUDE = 1.05
K_SAT_PH50 = 5.85
K_SAT_HILL = 5.0

# Hidden coupling: the saturation step has its own pH-dependent activation
# energy that adds to the main reaction's Ea. Equivalent (and physically
# motivated) to "k_sat depends on temperature in a pH-modulated way", but
# expressed as an Arrhenius prefactor + pH-dependent Ea so that ``k(T, pH)``
# at fixed pH is still an exact Arrhenius. This is the key design choice:
#   * per-bin Arrhenius: each bin sees its own effective Ea — perfect fit;
#     the bias from the truth's Ea is absorbed silently;
#   * linear-pH proposal: shared Ea, can capture log(k_ref) linear in pH but
#     CANNOT vary the slope with pH — fails noticeably at the bin extremes;
#   * hybrid: residual MLP can see (T, pH) jointly and fix the slope drift.
EA_PH_INTERCEPT = 5.85  # pH at which Ea_eff equals EA_TRUE
EA_PH_SLOPE = 30.0  # kJ/mol per pH-unit — Ea_eff = EA_TRUE + slope·(pH − intercept)


def _k_sat_from_ph(pH):
    """pH-dependent prefactor — sigmoidal saturation. T-independent."""
    pH_arr = jnp.asarray(pH)
    return K_SAT_BASELINE + K_SAT_AMPLITUDE / (
        1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
    )


def _ea_eff(pH):
    """Effective activation energy of the lumped reaction at pH (kJ/mol)."""
    return EA_TRUE + EA_PH_SLOPE * (jnp.asarray(pH) - EA_PH_INTERCEPT)


def k_true(temperature_C, pH):
    """Ground-truth rate constant: pH prefactor × Arrhenius with pH-dependent Ea."""
    T_K = jnp.asarray(temperature_C) + 273.15
    arrhenius = jnp.exp(-_ea_eff(pH) / R_GAS * (1.0 / T_K - 1.0 / T_REF))
    return _k_sat_from_ph(pH) * arrhenius


# Design of experiments — same disjoint-pH LHS as the notebook.
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


def _build_design():
    """Per-bin 2-D LHS over (T, Ca0); pH is fixed per bin (not LHS-sampled)."""
    design: list[tuple[float, float, float]] = []
    bin_of_exp: list[int] = []
    for bin_idx, ph in enumerate(PH_BINS):
        sampler = qmc.LatinHypercube(d=2, seed=DOE_SEED + bin_idx)
        unit = sampler.random(n=N_PER_BIN)
        lo = np.array([T_C_RANGE[0], CA0_RANGE[0]])
        hi = np.array([T_C_RANGE[1], CA0_RANGE[1]])
        pts = lo + (hi - lo) * unit
        for t, ca0 in pts:
            design.append((float(t), float(ph), float(ca0)))
            bin_of_exp.append(bin_idx)
    return design, bin_of_exp


def _add_heteroscedastic_noise(values, key):
    """σ = NOISE_REL · max(|values|, NOISE_FLOOR), clipped at 0."""
    scale = NOISE_REL * jnp.maximum(jnp.abs(values), NOISE_FLOOR)
    noisy = values + scale * jr.normal(key, values.shape)
    return jnp.clip(noisy, 0.0, None)


def _true_ca_trajectory(ts, temperature_C, pH, ca0):
    """Closed-form Ca(t) = ca0 · exp(-k_true(T, pH) · t)."""
    k = float(k_true(temperature_C, pH))
    return float(ca0) * jnp.exp(-k * jnp.asarray(ts))


def _y0_fn(covariates, channels):
    """Initial state [Ca, Cb] = [first observed Ca, 0]."""
    ca0 = jnp.asarray(channels["Ca"].values[0])
    return jnp.stack([ca0, jnp.zeros_like(ca0)])


def _state_to_output(state):
    """Project [Ca, Cb] onto the observed channel [Ca]."""
    return state[..., :1]


OUTPUT_CHANNELS = ("Ca",)


def _build_experiments():
    """Construct the twelve synthetic experiments and the per-experiment bin index."""
    design, bin_of_exp = _build_design()
    noise_root = jr.PRNGKey(NOISE_SEED)
    ts_global = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments = []
    for i, (T, ph, ca0) in enumerate(design):
        key = jr.fold_in(noise_root, i)
        clean = _true_ca_trajectory(ts_global, T, ph, ca0)
        noisy = _add_heteroscedastic_noise(clean, key)
        sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
        variance = sigma**2
        experiments.append(
            make_experiment(
                covariates={
                    "temperature_C": float(T),
                    "pH": float(ph),
                    "Ca0": float(ca0),
                },
                channels={"Ca": ChannelObs(ts=ts_global, values=noisy, variance=variance)},
                y0_fn=_y0_fn,
                exp_id=f"exp_{i:02d}_T{T:.1f}_pH{ph:.2f}_Ca0{ca0:.2f}",
            )
        )
    return experiments, bin_of_exp


# Predictor — Arrhenius trunk + residual MLP, identical layout to the notebook.
LOG_KREF_BOUNDS = (-3.0, 2.0)
EA_BOUNDS = (0.0, 120.0)


class ArrheniusKinetics(eqx.Module):
    """Centred-Arrhenius parametric trunk: two trainable scalars in latent space."""

    latent: Float[Array, " 2"]
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        self.latent = jr.normal(key, (2,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOG_KREF_BOUNDS, EA_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self) -> Float[Array, " 2"]:
        return self.out_scaler.from_latent(self.latent)


LINPH_INTERCEPT_BOUNDS = (-3.0, 3.0)
LINPH_SLOPE_BOUNDS = (-1.0, 1.0)


class LinearPHArrheniusKinetics(eqx.Module):
    """Hand-crafted "smarter" trunk: ``log_k_ref(pH) = a + b·pH`` with a single
    trainable Ea shared across all pH.

    Three trainable scalars (a, b, Ea). The proposer guesses that the prefactor
    varies linearly in pH and that activation energy is a single constant of
    the reaction. The truth's effective Ea actually varies strongly with pH
    (saturation step has its own pH-dependent Ea), so no choice of (a, b, Ea)
    can match the per-pH temperature drift simultaneously: the LSQ-best fit
    converges to a compromise Ea midway between the per-bin true values, which
    is wrong at every off-centre bin.
    """

    latent: Float[Array, " 3"]
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        self.latent = jr.normal(key, (3,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LINPH_INTERCEPT_BOUNDS, LINPH_SLOPE_BOUNDS, EA_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self) -> Float[Array, " 3"]:
        return self.out_scaler.from_latent(self.latent)


INPUT_KEYS = ("temperature_C", "pH", "Ca0")
TEMPERATURE_BOUNDS = (0.0, 50.0)
PH_BOUNDS = (3.0, 9.0)
CA0_INPUT_BOUNDS = (0.75, 1.5)
RES_LOG10_BOUNDS = (-2.0, 2.0)


# Both parametric trunks share a keyword-only ``key`` constructor. A bare
# ``type[eqx.Module]`` promises nothing about that constructor, so spelling
# the union out is what lets the ``key=`` call below type-check.
_TrunkFactory = type[ArrheniusKinetics] | type[LinearPHArrheniusKinetics]


def _build_predictors_init(trunk_cls: _TrunkFactory = ArrheniusKinetics):
    """Build the (parametric_trunk, residual_bp) two-leaf predictors pytree.

    ``trunk_cls`` selects which parametric trunk goes in the first leaf — the
    plain ``ArrheniusKinetics`` (2 params) for the per-bin and joint baselines
    plus the hybrid; ``LinearPHArrheniusKinetics`` (3 params) for the linear-pH
    parametric variant. The residual ``BoundedPredictor`` leaf is identical
    across all configurations so the trunk-only mask logic can be shared.
    """
    root = jr.PRNGKey(0)
    k_param, k_residual = jr.split(root, 2)
    parametric_trunk = trunk_cls(key=k_param)
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
    return (parametric_trunk, residual_bp)


# Simulators — full hybrid (trunk + residual) and the pure-mechanistic baseline
# (residual contribution dropped; safe to run with a freshly-initialised
# residual MLP that is not identically zero).
def simulate_fn(predictors, ts, covariates, y0, solver):
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


def simulate_fn_baseline(predictors, ts, covariates, y0, solver):
    parametric, _residual = predictors
    log_k_ref, Ea = parametric()

    T_C = covariates["temperature_C"]
    T_K = T_C + 273.15

    log10_k = (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)
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


def simulate_fn_linear_ph(predictors, ts, covariates, y0, solver):
    """Linear-in-pH analytic extension: ``log_k_ref(pH) = a + b·pH``.

    Same Arrhenius temperature law as the baseline, but the pre-exponential
    intercept is now a linear function of pH. Three trainable scalars in the
    trunk; the residual MLP leaf is present for pytree-shape compatibility but
    is not consulted (mirrors ``simulate_fn_baseline``).
    """
    parametric, _residual = predictors
    a, b_ph, Ea = parametric()

    T_C = covariates["temperature_C"]
    pH = covariates["pH"]
    T_K = T_C + 273.15

    log_k_ref = a + b_ph * pH
    log10_k = (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)
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


# Training configs and fit drivers.
def _cfg_baseline():
    return EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        loss="mse",
        verbose=False,
    )


# Plotting.
DARK2 = plt.get_cmap("Dark2")
# Dark2 indices used cold→warm. Dark2 is qualitative so this mapping is purely
# conventional; the cold→warm intuition comes from sorting experiments by T
# before assigning colours, not from the palette itself.
T_COLOR_INDICES = (0, 1, 2, 3)

DENSE_TS = jnp.linspace(0.0, T_MAX, 200)


def _three_panel_figure(
    experiments,
    bin_of_exp,
    predictors_for_bin,
    sim_fn,
    solver,
    out_path,
    figsize: tuple[float, float] = (12, 4),
    equation: str | None = None,
):
    """Render a 1×3 figure. ``predictors_for_bin[bin_idx]`` selects the model
    used in that column — different per bin for the per-bin baseline, the same
    pytree for every bin under the joint and hybrid models. ``figsize`` defaults
    to a 4×4-per-panel layout that fits comfortably on a slide. ``equation``,
    if given, is rendered as a suptitle (LaTeX maths supported via ``$...$``)
    so each figure carries its own model-form caption."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for bin_idx, (ax, ph) in enumerate(zip(axes, PH_BINS, strict=True)):
        bin_exps = [e for i, e in enumerate(experiments) if bin_of_exp[i] == bin_idx]
        bin_exps_sorted = sorted(bin_exps, key=lambda e: float(e.covariates["temperature_C"]))
        preds = predictors_for_bin[bin_idx]

        for k, exp in enumerate(bin_exps_sorted):
            color = DARK2(T_COLOR_INDICES[k])
            T_C = float(exp.covariates["temperature_C"])
            ts_obs = np.asarray(exp.channels["Ca"].ts)
            ca_obs = np.asarray(exp.channels["Ca"].values)
            # Integrate the ODE on a dense time grid for a smooth line.
            covariates = {
                "temperature_C": jnp.asarray(float(exp.covariates["temperature_C"])),
                "pH": jnp.asarray(float(exp.covariates["pH"])),
                "Ca0": jnp.asarray(float(exp.covariates["Ca0"])),
            }
            y0 = _y0_fn(covariates, exp.channels)
            traj = sim_fn(preds, DENSE_TS, covariates, y0, solver)
            ca_dense = np.asarray(traj[:, 0])
            ax.scatter(
                ts_obs,
                ca_obs,
                s=44,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=f"T = {T_C:.1f} °C",
            )
            ax.plot(np.asarray(DENSE_TS), ca_dense, color=color, linewidth=1.8, zorder=2)

        ax.set_title(f"pH = {ph:.2f}", fontsize=15)
        ax.set_xlabel("t", fontsize=12)
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.35)
        ax.grid(which="minor", alpha=0.15, linestyle=":")
        ax.legend(loc="upper right", fontsize=12, frameon=True)

    axes[0].set_ylabel("Ca", fontsize=12)
    if equation is not None:
        fig.suptitle(equation, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# Entry point.
def main():
    plt.rcParams.update({"font.size": 12})

    out_dir = Path(__file__).resolve().parent / "figures" / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    experiments, bin_of_exp = _build_experiments()
    dataset = make_dataset(
        experiments,
        output_channel_names=OUTPUT_CHANNELS,
    )

    predictors_init = _build_predictors_init()
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.05,
    )

    # Trunk-only mask: freeze the residual MLP and any BoundScaler leaves so
    # only the two-element ArrheniusKinetics latent is trainable. Reused
    # across the per-bin baseline and the joint phase-1 fit.
    mask_p1 = trainable_mask(predictors_init)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundedPredictor)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors_init, BoundScaler)

    print("Fitting per-bin Arrhenius...")
    # One Arrhenius fit per pH bin.
    bin_predictors: dict[int, tuple] = {}
    for bin_idx, ph in enumerate(PH_BINS):
        bin_exps = [e for i, e in enumerate(experiments) if bin_of_exp[i] == bin_idx]
        bin_ds = make_dataset(
            bin_exps,
            output_channel_names=OUTPUT_CHANNELS,
        )
        _hist, preds = train_with_evosax(
            predictors_init,
            bin_ds,
            _cfg_baseline(),
            simulate_fn=simulate_fn_baseline,
            state_to_output=_state_to_output,
            solver=solver,
            trainable=mask_p1,
            key=jr.PRNGKey(bin_idx),
        )
        bin_predictors[bin_idx] = preds
        log_k_ref, Ea = preds[0]()
        print(
            f"  bin {bin_idx} (pH={ph}): log_k_ref={float(log_k_ref):+.3f}, "
            f"Ea={float(Ea):.2f} kJ/mol"
        )

    print("Fitting joint Arrhenius (hybrid phase 1)...")
    # Single Arrhenius fit on all 12 experiments — phase 1 of the hybrid.
    _hist, preds = train_with_evosax(
        predictors_init,
        dataset,
        _cfg_baseline(),
        simulate_fn=simulate_fn_baseline,
        state_to_output=_state_to_output,
        solver=solver,
        trainable=mask_p1,
        key=jr.PRNGKey(0),
    )
    log_k_ref, Ea = preds[0]()
    print(f"  joint Arrhenius: log_k_ref={float(log_k_ref):+.3f}, Ea={float(Ea):.2f} kJ/mol")
    predictors_p1 = preds
    joint_for_each_bin = {b: predictors_p1 for b in range(len(PH_BINS))}

    # Linear-in-pH analytic variant — same trunk-only mask logic as the joint
    # baseline, just a different parametric family. The residual leaf is unused
    # at simulate time but is present so trainable_mask + freeze_modules_of_type
    # stay identical.
    print("Fitting linear-pH Arrhenius...")
    predictors_init_linph = _build_predictors_init(LinearPHArrheniusKinetics)
    mask_linph = trainable_mask(predictors_init_linph)
    mask_linph = freeze_modules_of_type(mask_linph, predictors_init_linph, BoundedPredictor)
    mask_linph = freeze_modules_of_type(mask_linph, predictors_init_linph, BoundScaler)
    # Single linear-pH Arrhenius fit on all 12 experiments. Three params.
    _hist, preds = train_with_evosax(
        predictors_init_linph,
        dataset,
        _cfg_baseline(),
        simulate_fn=simulate_fn_linear_ph,
        state_to_output=_state_to_output,
        solver=solver,
        trainable=mask_linph,
        key=jr.PRNGKey(0),
    )
    a, b_ph, Ea = preds[0]()
    print(
        f"  linear-pH Arrhenius: log_k_ref(pH) = {float(a):+.3f} "
        f"{float(b_ph):+.3f}·pH, Ea={float(Ea):.2f} kJ/mol"
    )
    predictors_linph = preds
    linph_for_each_bin = {b: predictors_linph for b in range(len(PH_BINS))}

    print("Fitting hybrid (phase 2)...")
    # Residual-only mask: freeze the trunk at its phase-1 endpoint and let the
    # MLP weights move. BoundScalers stay frozen by convention.
    mask_p2 = trainable_mask(predictors_p1)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, ArrheniusKinetics)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, BoundScaler)
    # Phase 2 — fit the residual MLP with the trunk pinned at its phase-1
    # endpoint.
    _hist, predictors_p2 = train_with_optax(
        predictors_p1,
        dataset,
        OptaxTrainingConfig(
            steps=(200,),
            lr=(3e-3,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            length_schedule=(1.0,),
            loss="mse",
            verbose=False,
        ),
        simulate_fn=simulate_fn,
        state_to_output=_state_to_output,
        solver=solver,
        trainable=mask_p2,
        key=jr.PRNGKey(1),
    )
    hybrid_for_each_bin = {b: predictors_p2 for b in range(len(PH_BINS))}

    # Equations rendered as suptitles. LaTeX maths via $...$. Coefficients are
    # rounded by hand from the trained models — kept inline in the strings so
    # the captions are self-explanatory on a slide.
    eq_per_bin = (
        r"$k(T) = k_\mathrm{ref}\,\exp[-E_a/R\,(1/T - 1/T_\mathrm{ref})]$"
        r"      one fit per pH bin:  $E_a \approx 3,\ 31,\ 65$ kJ/mol"
    )
    eq_joint = (
        r"$k(T) = k_\mathrm{ref}\,\exp[-E_a/R\,(1/T - 1/T_\mathrm{ref})]$"
        r"      single fit:  $k_\mathrm{ref} \approx 0.65$,  $E_a \approx 20$ kJ/mol"
    )
    eq_linph = (
        r"$k(T,\mathrm{pH}) = e^{a + b\,\mathrm{pH}}\,\exp[-E_a/R\,(1/T - 1/T_\mathrm{ref})]$"
        r"      $a \approx 1.5$,  $b \approx -0.33$,  $E_a \approx 30$ kJ/mol  (all trainable)"
    )
    eq_hybrid = (
        r"$k(T, \mathrm{pH}) = k_\mathrm{param}(T) + k_\mathrm{residual}(T, \mathrm{pH}, C_{a0})$"
    )

    print("Rendering figures...")
    _three_panel_figure(
        experiments,
        bin_of_exp,
        bin_predictors,
        sim_fn=simulate_fn_baseline,
        solver=solver,
        out_path=out_dir / "01_per_bin_arrhenius.png",
        equation=eq_per_bin,
    )
    _three_panel_figure(
        experiments,
        bin_of_exp,
        joint_for_each_bin,
        sim_fn=simulate_fn_baseline,
        solver=solver,
        out_path=out_dir / "02_joint_arrhenius.png",
        equation=eq_joint,
    )
    _three_panel_figure(
        experiments,
        bin_of_exp,
        linph_for_each_bin,
        sim_fn=simulate_fn_linear_ph,
        solver=solver,
        out_path=out_dir / "02b_linear_ph_arrhenius.png",
        equation=eq_linph,
    )
    _three_panel_figure(
        experiments,
        bin_of_exp,
        hybrid_for_each_bin,
        sim_fn=simulate_fn,
        solver=solver,
        out_path=out_dir / "03_hybrid.png",
        equation=eq_hybrid,
    )
    print("Done.")


if __name__ == "__main__":
    main()
