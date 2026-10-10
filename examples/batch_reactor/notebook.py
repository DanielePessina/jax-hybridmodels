"""Batch reactor walkthrough: per-pH Arrhenius fits against one hybrid model.

A first-order ``A -> B`` reaction in a closed batch reactor has a rate
constant ``k(T, pH)``. Arrhenius describes the temperature dependence; the
pH dependence has no first-principles form.

The classical answer fits ``(log k_ref, E_a)`` separately at each pH. That
works inside a fitted pH bin, but it yields a table of parameters with no
functional pH dependence, so predicting at an unsampled pH means picking a
bin or interpolating by hand. The hybrid keeps the Arrhenius trunk and adds
a small neural residual ``Δlog10 k(T, pH, C_A0)``, trained on every pH at
once, which interpolates between them.

Both models see the same twelve synthetic experiments at three pH values.
The baseline is scored by per-bin leave-one-out and by transfer between
bins; the hybrid by leave-one-out over all twelve and by leaving out every
experiment at the middle pH.

Unlike ``train_hybrid.py``, temperature is constant within each run and
``C_A0`` varies, entering the residual as a covariate the true rate ignores.

Library features shown: two ``HybridModel`` objects sharing one predictor
pytree, ``trainable`` masks to fit the trunk alone or the residual alone,
and repeated ``fit`` / ``predict`` calls for cross-validation.

About 40 fits in total (CMA-ES and AdamW), so expect several minutes on CPU.
Prints fit summaries and writes eleven ``notebook_*.png`` figures to
``figures/``.

Run: ``uv run python examples/batch_reactor/notebook.py``
"""

# ruff: noqa: F722

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float
from scipy.stats import qmc

from jaxhybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Dataset,
    Experiment,
    FitResult,
    HybridModel,
    MLPPredictor,
    SolverConfig,
    compute_metrics,
    count_trainable_params,
    freeze_modules_of_type,
    make_dataset,
    make_experiment,
    print_metrics,
    trainable_mask,
)
from jaxhybridmodels.training import EvosaxTrainingConfig, OptaxTrainingConfig

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Every fit on this page uses one of these two configs.
CMA_ES = EvosaxTrainingConfig(
    algorithm="CMA_ES",
    population_size=32,
    num_generations=30,
    init="lhs_box",
    init_box_extent=2.0,
    sigma_init=0.5,
    loss="mse",
    verbose=False,
)
ADAMW = OptaxTrainingConfig(steps=200, lr=3e-3, loss="mse", verbose=False)


def _pairs(buckets: tuple[Array, ...], dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Observed and predicted ``Ca`` at every observed point, as flat arrays."""
    obs, pred = [], []
    for values, bp in zip(buckets, dataset.bucket_payloads, strict=True):
        mask = np.asarray(bp.mask[..., 0], dtype=bool)
        obs.append(np.asarray(bp.y_observed[..., 0])[mask])
        pred.append(np.asarray(values[..., 0])[mask])
    return np.concatenate(obs), np.concatenate(pred)


def _score(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """``(MSE, R^2)``; R^2 is nan when the observations do not vary."""
    resid = pred - obs
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(np.mean(resid**2)), r2


def _parity_panel(ax, obs: np.ndarray, pred: np.ndarray, title: str, **style) -> None:
    """One out-of-fold parity scatter with the identity line and its scores in the title."""
    mse, r2 = _score(obs, pred)
    ax.scatter(obs, pred, label=f"OOF (n={len(obs)})", **style)
    lo, hi = float(min(obs.min(), pred.min())), float(max(obs.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"{title}\nR²={r2:.4f}, MSE={mse:.3e}")
    ax.set_xlabel("observed")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)


def main() -> None:
    """Run the whole walkthrough: synthetic data, baseline, hybrid, both CVs."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ## True model
    #
    # One reaction A -> B at constant volume:
    #
    #     dC_A/dt = -k(T, pH) C_A,      dC_B/dt = +k(T, pH) C_A
    #
    # with C_A(0) = C_A0 and C_B(0) = 0, so C_A(t) = C_A0 exp(-k t). The
    # hidden rate is a Hill-type saturation in pH times Arrhenius in T:
    #
    #     k(T, pH) = k_sat(pH) * exp(-E_a/R (1/T_K - 1/T_ref))
    #     k_sat(pH) = b + a / (1 + (pH/pH50)^n)
    #
    # Centring at T_ref makes k(T_ref, pH) = k_sat(pH). The rate falls from
    # a maximum near pH 4 to a plateau near pH 8, and roughly doubles per
    # ten kelvin.
    T_REF = 298.15  # K
    R_GAS = 8.314e-3  # kJ/(mol·K)
    EA_TRUE = 30.0  # kJ/mol

    K_SAT_BASELINE = 0.14
    K_SAT_AMPLITUDE = 1.05
    K_SAT_PH50 = 5.85
    K_SAT_HILL = 5.0

    def k_true(temperature_C, pH):
        """Ground-truth rate constant. Data generation and plot overlays only."""
        pH_arr = jnp.asarray(pH)
        k_sat = K_SAT_BASELINE + K_SAT_AMPLITUDE / (
            1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
        )
        T_K = jnp.asarray(temperature_C) + 273.15
        return k_sat * jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))

    for T_C in (15.0, 25.0, 35.0):
        print(f"k_true({T_C:.0f}°C, pH=5.85) = {float(k_true(T_C, K_SAT_PH50)):.4f}")

    # ## Synthetic data
    #
    # Twelve experiments at pH 5.0, 5.85 and 7.0: below the saturation
    # knee, at it, and above it. Each pH bin gets four (T, C_A0) points
    # from its own Latin hypercube over T in [15, 35] °C and C_A0 in
    # [0.75, 1.5]. pH is fixed per bin rather than sampled, which is what
    # lets the pH-blind baseline be fitted bin by bin on the same data the
    # hybrid sees.
    #
    # Each run covers t in [0, 5] with twelve evenly spaced samples and
    # heteroscedastic noise sigma = 0.03 * max(|C_A|, 0.02). The variance
    # is stored on each ChannelObs.
    T_C_RANGE = (15.0, 35.0)
    CA0_RANGE = (0.75, 1.5)
    PH_BINS = (5.0, 5.85, 7.0)
    N_PER_BIN = 4
    T_MAX = 5.0
    N_TIMESTEPS = 12
    NOISE_REL = 0.03
    NOISE_FLOOR = 0.02
    OUTPUT_CHANNELS = ("Ca",)

    design: list[tuple[float, float, float]] = []
    bin_of_exp: list[int] = []
    for bin_idx, ph in enumerate(PH_BINS):
        unit = qmc.LatinHypercube(d=2, seed=bin_idx).random(n=N_PER_BIN)
        lo = np.array([T_C_RANGE[0], CA0_RANGE[0]])
        hi = np.array([T_C_RANGE[1], CA0_RANGE[1]])
        for T_C, ca0 in lo + (hi - lo) * unit:
            design.append((float(T_C), ph, float(ca0)))
            bin_of_exp.append(bin_idx)

    def true_ca(ts, temperature_C, pH, ca0):
        """Closed-form C_A(t); C_A0 is only the initial condition."""
        return ca0 * jnp.exp(-float(k_true(temperature_C, pH)) * jnp.asarray(ts))

    # The state is y = [C_A, C_B], so y0 = [C_A0, 0], and state_to_output
    # picks C_A back out. y0 reads the noisy first observation.
    def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 2"]:
        ca0 = jnp.asarray(channels["Ca"].values[0])
        return jnp.stack([ca0, jnp.zeros_like(ca0)])

    def state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
        return state[..., :1]

    ts_global = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    noise_root = jr.PRNGKey(1)
    experiments: list[Experiment] = []
    for i, (T_C, ph, ca0) in enumerate(design):
        clean = true_ca(ts_global, T_C, ph, ca0)
        sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
        noisy = jnp.clip(clean + sigma * jr.normal(jr.fold_in(noise_root, i), clean.shape), 0.0)
        experiments.append(
            make_experiment(
                covariates={"temperature_C": T_C, "pH": ph, "Ca0": ca0},
                channels={"Ca": ChannelObs(ts=ts_global, values=noisy, variance=sigma**2)},
                y0_fn=y0_fn,
                exp_id=f"exp_{i:02d}_T{T_C:.1f}_pH{ph:.2f}_Ca0{ca0:.2f}",
            )
        )

    def dataset_of(exps: list[Experiment]) -> Dataset:
        return make_dataset(exps, output_channel_names=OUTPUT_CHANNELS)

    def in_bin(bin_idx: int) -> list[Experiment]:
        return [e for e, b in zip(experiments, bin_of_exp, strict=True) if b == bin_idx]

    dataset = dataset_of(experiments)
    print(f"{len(experiments)} experiments, {len(dataset.bucket_payloads)} bucket(s)")
    for e in experiments:
        print(f"  {e.exp_id}")

    # Raw data, coloured by pH bin. pH 5.0 decays fastest and pH 7.0
    # slowest; the spread inside a bin comes from temperature.
    bin_colors = ("C0", "C1", "C3")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for e, b in zip(experiments, bin_of_exp, strict=True):
        obs = e.channels["Ca"]
        ax.plot(obs.ts, obs.values, "o-", color=bin_colors[b], markersize=4, linewidth=1.0)
    for b, ph in enumerate(PH_BINS):
        ax.plot([], [], "o-", color=bin_colors[b], label=f"pH = {ph}")
    ax.set_xlabel("t")
    ax.set_ylabel("Ca")
    ax.set_title("Observed concentrations across the three pH bins")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "notebook_01_raw_data.png")
    plt.close(fig)

    # ## Model
    #
    # Both models integrate dC_A/dt = -k_hat C_A and differ only in k_hat.
    #
    # The trunk is centred Arrhenius in T, two trainable scalars, no pH:
    #
    #     log10 k_param(T) = (log k_ref - E_a/R (1/T_K - 1/T_ref)) / ln 10
    #
    # The hybrid adds a residual network:
    #
    #     log10 k_hat = log10 k_param(T) + Δlog10(T, pH, C_A0)
    #
    # C_A0 is a red herring: first-order kinetics do not depend on it, so a
    # well-trained residual should mostly ignore it.
    #
    # ArrheniusKinetics holds (log k_ref, E_a) as a latent vector mapped into
    # physical boxes by a sigmoid BoundScaler. The residual is a 16-neuron
    # ReLU MLP inside a BoundedPredictor whose output box [-2, 2] decades is
    # symmetric: a fresh network sits near the sigmoid midpoint, which is
    # zero decades, so the hybrid starts out equal to the trunk.
    LOG_KREF_BOUNDS = (-3.0, 2.0)
    EA_BOUNDS = (0.0, 80.0)

    class ArrheniusKinetics(eqx.Module):
        """Centred-Arrhenius trunk: two trainable scalars."""

        latent: Float[Array, " 2"]
        out_scaler: BoundScaler

        def __init__(self, *, key: Array) -> None:
            self.latent = jr.normal(key, (2,)) * 0.1
            self.out_scaler = BoundScaler(bounds=(LOG_KREF_BOUNDS, EA_BOUNDS), transform="sigmoid")

        def __call__(self) -> Float[Array, " 2"]:
            """Return (log_k_ref, Ea) in physical units."""
            return self.out_scaler.from_latent(self.latent)

    k_param, k_residual = jr.split(jr.PRNGKey(0), 2)
    predictors_init = (
        ArrheniusKinetics(key=k_param),
        BoundedPredictor(
            input_keys=("temperature_C", "pH", "Ca0"),
            in_scaler=BoundScaler(bounds=((0.0, 50.0), (3.0, 9.0), CA0_RANGE), transform="sigmoid"),
            inner=MLPPredictor(
                in_size=3,
                out_size=1,
                width_size=16,
                depth=1,
                activation_name="relu",
                key=k_residual,
            ),
            out_scaler=BoundScaler(bounds=((-2.0, 2.0),), transform="sigmoid"),
        ),
    )

    def log10_k_param(parametric: ArrheniusKinetics, temperature_C: ArrayLike) -> Array:
        log_k_ref, Ea = parametric()
        T_K = jnp.asarray(temperature_C) + 273.15
        return (log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)

    def decay(log10_k: Array, ts: Array, y0: Array, solver: SolverConfig) -> Array:
        """Integrate A -> B at the constant rate 10**log10_k."""
        k = jnp.power(10.0, log10_k)

        def vector_field(t: ArrayLike, y: Float[Array, " 2"], args: object) -> Array:
            # Clipping guards rare negative solver excursions near zero.
            rate = k * jnp.maximum(y[0], 0.0)
            return jnp.stack([-rate, rate])

        return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)

    def simulate_hybrid(predictors, ts, covariates, y0, solver):
        parametric, residual = predictors
        delta = jnp.squeeze(residual(covariates))
        return decay(log10_k_param(parametric, covariates["temperature_C"]) + delta, ts, y0, solver)

    # The baseline ignores the residual entirely. A fresh MLP is not exactly
    # zero, and letting it into the rate would add a spurious T-dependence
    # and make the trunk's E_a unidentifiable. Keeping the same predictor
    # pytree means one mask serves both models.
    def simulate_baseline(predictors, ts, covariates, y0, solver):
        return decay(log10_k_param(predictors[0], covariates["temperature_C"]), ts, y0, solver)

    solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05)
    hybrid = HybridModel(
        predictors=predictors_init,
        simulate_fn=simulate_hybrid,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
        solver=solver,
    )
    baseline = replace(hybrid, simulate_fn=simulate_baseline)

    # ## Training masks
    #
    # The default mask already holds BoundScaler leaves fixed.
    # freeze_modules_of_type then freezes one half of the pytree: mask_trunk
    # leaves only the trunk's two latent scalars, mask_residual only the MLP.
    mask_trunk = freeze_modules_of_type(
        trainable_mask(predictors_init), predictors_init, BoundedPredictor
    )
    mask_residual = freeze_modules_of_type(
        trainable_mask(predictors_init), predictors_init, ArrheniusKinetics
    )
    print(f"trunk mask trains {count_trainable_params(predictors_init, mask_trunk)} scalars")
    print(f"residual mask trains {count_trainable_params(predictors_init, mask_residual)} scalars")

    def fit_hybrid(train: Dataset, key_p1: Array, key_p2: Array) -> tuple[FitResult, FitResult]:
        """CMA-ES on the trunk, then AdamW on the residual with the trunk frozen."""
        phase1 = hybrid.fit(train, CMA_ES, key=key_p1, trainable=mask_trunk)
        phase2 = phase1.model.fit(train, ADAMW, key=key_p2, trainable=mask_residual)
        return phase1, phase2

    def trajectory_grid_plot(exps, predictions, title, predicted_label="predicted"):
        """One panel per experiment: truth, observations and a prediction."""
        n = len(exps)
        nrows = (n + 2) // 3
        fig, axes = plt.subplots(
            nrows, 3, figsize=(11, 3.0 * nrows), sharex=True, sharey=True, squeeze=False
        )
        ts_dense = np.linspace(0.0, T_MAX, 200)
        flat = axes.flatten()
        for ax, e in zip(flat[:n], exps, strict=True):
            T_C, ph, ca0 = (float(e.covariates[k]) for k in ("temperature_C", "pH", "Ca0"))
            pred = predictions[e.exp_id]
            ax.plot(ts_dense, true_ca(ts_dense, T_C, ph, ca0), color="black", lw=1.4, label="truth")
            ax.scatter(
                e.channels["Ca"].ts,
                e.channels["Ca"].values,
                s=22,
                color="C0",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label="observed",
            )
            ax.plot(
                pred.times, pred.outputs["Ca"], color="C3", ls="--", lw=1.4, label=predicted_label
            )
            ax.set_title(f"T={T_C:.1f}°C, pH={ph:.2f}, Ca0={ca0:.2f}", fontsize=9)
            ax.set_ylim(-0.05, 1.6)
            ax.grid(alpha=0.3)
        for ax in flat[n:]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("t")
        for ax in axes[:, 0]:
            ax.set_ylabel("Ca")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(title, y=0.995, fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return fig

    def k_reveal_plot(predictors_p1, predictors_p2, title):
        """log10 k(pH) at 15, 25 and 35 °C: truth solid, trunk dashed, hybrid dotted.

        The residual also takes C_A0, which the truth ignores, so the hybrid
        is drawn at the middle of the C_A0 range.
        """
        pH_grid = jnp.linspace(4.0, 8.0, 200)
        T_C_lines = (15.0, 25.0, 35.0)
        ca0_mid = jnp.full_like(pH_grid, 1.125)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for T_C, color in zip(T_C_lines, ("C0", "C1", "C2"), strict=True):
            middle = T_C == T_C_lines[1]
            ax.plot(
                pH_grid,
                np.log10(k_true(T_C, pH_grid)),
                color=color,
                lw=1.6,
                label=f"truth (T={T_C:.0f}°C)",
            )
            trunk = float(log10_k_param(predictors_p1[0], T_C))
            ax.plot(
                pH_grid,
                np.full_like(pH_grid, trunk),
                color=color,
                lw=1.2,
                ls="--",
                label="parametric" if middle else None,
            )
            if predictors_p2 is not None:
                parametric, residual = predictors_p2
                inputs = {
                    "temperature_C": jnp.full_like(pH_grid, T_C),
                    "pH": pH_grid,
                    "Ca0": ca0_mid,
                }
                delta = eqx.filter_vmap(residual)(inputs)[:, 0]
                hybrid_curve = log10_k_param(parametric, T_C) + delta
                ax.plot(
                    pH_grid,
                    hybrid_curve,
                    color=color,
                    lw=1.2,
                    ls=":",
                    label="hybrid" if middle else None,
                )
        T_obs = np.array([float(e.covariates["temperature_C"]) for e in experiments])
        pH_obs = np.array([float(e.covariates["pH"]) for e in experiments])
        ax.scatter(
            pH_obs,
            np.log10(k_true(T_obs, pH_obs)),
            s=44,
            color="black",
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

    # ## Pure mechanistic baseline
    #
    # With no pH input, the trunk cannot be fitted across bins without
    # averaging over the pH shifts, so the classical workaround is one fit
    # per bin: three CMA-ES runs and three parameter pairs with nothing
    # connecting them.
    per_bin = []
    print("Per-bin Arrhenius fits:")
    for b, ph in enumerate(PH_BINS):
        fit = baseline.fit(dataset_of(in_bin(b)), CMA_ES, key=jr.PRNGKey(b), trainable=mask_trunk)
        log_k_ref, Ea = fit.model.predictors[0]()
        per_bin.append(fit.model)
        print(
            f"  pH={ph}: log_k_ref = {float(log_k_ref):+.3f}, Ea = {float(Ea):.2f} kJ/mol, "
            f"final loss = {fit.history.best_fitness[-1]:.4e}"
        )

    # Per-bin parameter recovery. The fitted log k_ref follows ln k_sat(pH)
    # across bins, and E_a stays near 30 kJ/mol in each: the trunk absorbs
    # the pH dependence into a different prefactor every time it is refit.
    fitted = np.array([[float(v) for v in m.predictors[0]()] for m in per_bin])
    log_k_ref_truth = np.log(np.asarray(k_true(25.0, jnp.asarray(PH_BINS))))
    fig, (ax_kref, ax_ea) = plt.subplots(1, 2, figsize=(10, 4))
    ax_kref.scatter(PH_BINS, fitted[:, 0], s=80, color="C0", label="fitted (per bin)")
    ax_kref.plot(
        PH_BINS,
        log_k_ref_truth,
        color="black",
        ls="--",
        marker="x",
        markersize=10,
        label=r"truth: $\ln k_\mathrm{sat}(\mathrm{pH})$",
    )
    ax_kref.set_ylabel(r"$\log k_\mathrm{ref}$ (natural log)")
    ax_kref.set_title(r"Recovered $\log k_\mathrm{ref}$ per pH bin")
    ax_ea.scatter(PH_BINS, fitted[:, 1], s=80, color="C0", label="fitted (per bin)")
    ax_ea.axhline(EA_TRUE, color="black", ls="--", label=f"truth ($E_a$ = {EA_TRUE} kJ/mol)")
    ax_ea.set_ylabel("$E_a$ (kJ/mol)")
    ax_ea.set_title("Recovered $E_a$ per pH bin")
    ax_ea.set_ylim(0.0, max(40.0, float(fitted[:, 1].max()) * 1.2))
    for ax in (ax_kref, ax_ea):
        ax.set_xlabel("pH bin")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "notebook_02_per_bin_param_recovery.png")
    plt.close(fig)

    # Per-bin LOO-CV: refit on three of a bin's four points and predict the
    # fourth, twelve fits in all. Arrhenius extrapolates correctly in T at
    # fixed pH, so the out-of-fold points should sit on the diagonal. The
    # classical model generalises as long as you stay inside its bin.
    fig, axes = plt.subplots(1, len(PH_BINS), figsize=(12.0, 4.2), sharex=True, sharey=True)
    print("Per-bin LOO-CV:")
    for b, (ax, ph) in enumerate(zip(axes, PH_BINS, strict=True)):
        bin_exps = in_bin(b)
        obs, pred, mses = [], [], []
        for k, held in enumerate(bin_exps):
            train = dataset_of([e for e in bin_exps if e is not held])
            fit = baseline.fit(train, CMA_ES, key=jr.PRNGKey(b * 100 + k), trainable=mask_trunk)
            held_ds = dataset_of([held])
            o, p = _pairs(fit.model.predict(held_ds).buckets, held_ds)
            obs.append(o)
            pred.append(p)
            mses.append(_score(o, p)[0])
        print(f"  pH={ph}: mean OOF MSE = {np.mean(mses):.3e}, max = {np.max(mses):.3e}")
        _parity_panel(
            ax,
            np.concatenate(obs),
            np.concatenate(pred),
            f"pH = {ph}",
            s=24,
            alpha=0.7,
            color=bin_colors[b],
        )
    axes[0].set_ylabel("predicted (out-of-fold)")
    fig.suptitle("Pure mechanistic, per-bin LOO-CV parity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "notebook_03_per_bin_loocv_parity.png")
    plt.close(fig)

    # Inter-bin transfer, the hard half. The pH 5.0 fit predicts the four
    # pH 7.0 runs without refitting. The trunk has no pH input, so it
    # returns the same k(T) either way and the decay comes out far too
    # fast. This failure is what motivates the hybrid.
    target = dataset_of(in_bin(2))
    fig = trajectory_grid_plot(
        in_bin(2),
        per_bin[0].predict(target),
        title="pH 7.0 runs predicted with the pH 5.0 fit",
        predicted_label="pH 5.0 fit (wrong pH)",
    )
    fig.savefig(FIG_DIR / "notebook_04_interbin_transfer.png")
    plt.close(fig)

    # ## The hybrid model
    #
    # One model on all twelve experiments, in two phases on the same
    # predictor pytree. Phase 1 is CMA-ES on the trunk; without pH it lands
    # on a compromise across bins, which fixes the temperature dependence
    # and leaves pH to phase 2. Phase 2 is AdamW on the residual with the
    # trunk frozen. The residual starts at zero decades, so phase 2's first
    # loss matches phase 1's last and the seam does not show.
    phase1, phase2 = fit_hybrid(dataset, jr.PRNGKey(0), jr.PRNGKey(1))
    history_p1 = phase1.history.best_fitness
    history_p2 = phase2.history.loss
    log_k_ref, Ea = phase1.model.predictors[0]()
    print(f"phase 1: {len(history_p1)} generations, best loss {history_p1[-1]:.6f}")
    print(f"  recovered log_k_ref = {float(log_k_ref):.3f}, Ea = {float(Ea):.2f} kJ/mol")
    print_metrics(compute_metrics(phase1.model.predict(dataset).buckets, dataset))
    print(f"phase 2: {len(history_p2)} steps, final loss {history_p2[-1]:.6f}")
    print_metrics(compute_metrics(phase2.model.predict(dataset).buckets, dataset))

    # The headline plot. The trunk is flat in pH by construction; the
    # hybrid follows the truth, so the residual has recovered the shape of
    # k_sat(pH) from concentration data alone.
    fig = k_reveal_plot(
        phase1.model.predictors,
        phase2.model.predictors,
        "log10 k(pH): truth vs parametric vs hybrid",
    )
    fig.savefig(FIG_DIR / "notebook_05_k_reveal_phase2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    n1 = len(history_p1)
    ax.plot(
        np.arange(n1),
        np.log10(history_p1),
        color="C0",
        lw=1.4,
        label="phase 1, evosax (best so far)",
    )
    ax.plot(
        np.arange(n1, n1 + len(history_p2)),
        np.log10(history_p2),
        color="C3",
        lw=1.4,
        label="phase 2, optax (per step)",
    )
    ax.axvline(n1 - 0.5, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("training progress (generation, then step)")
    ax.set_ylabel("log10 MSE loss")
    ax.set_title("Loss across the evosax → optax pipeline")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "notebook_06_loss_curve.png")
    plt.close(fig)

    # ## Leave-one-out cross-validation
    #
    # Retrain both phases twelve times, each without one experiment, and
    # predict the one left out. Every experiment is then predicted once by
    # a model that never saw it. Seeds come from the fold index, so folds
    # reproduce across runs.
    obs, pred, oof_mses = [], [], []
    oof_predictions = {}
    print(f"LOO-CV: {len(experiments)} folds")
    for k, held in enumerate(experiments):
        train = dataset_of([e for e in experiments if e is not held])
        fold_p1, fold_p2 = fit_hybrid(train, jr.PRNGKey(k), jr.PRNGKey(1000 + k))
        held_ds = dataset_of([held])
        result = fold_p2.model.predict(held_ds)
        oof_predictions[held.exp_id] = result[held.exp_id]
        o, p = _pairs(result.buckets, held_ds)
        obs.append(o)
        pred.append(p)
        oof_mses.append(_score(o, p)[0])
        print(
            f"  fold {k:2d}: {held.exp_id} | p1 {fold_p1.history.best_fitness[-1]:.3e}, "
            f"p2 {fold_p2.history.loss[-1]:.3e}, OOF MSE {oof_mses[-1]:.3e}"
        )
    print(
        f"  mean OOF MSE {np.mean(oof_mses):.3e}, median {np.median(oof_mses):.3e}, "
        f"max {np.max(oof_mses):.3e}"
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _parity_panel(
        ax, np.concatenate(obs), np.concatenate(pred), "LOO-CV parity (Ca)", s=24, alpha=0.7
    )
    ax.set_ylabel("predicted (out-of-fold)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "notebook_07_loocv_parity.png")
    plt.close(fig)

    # Each panel is a held-out run. The dashed curve tracks the truth at
    # (T, pH, C_A0) points the model never trained on, which says the
    # residual learned the pH dependence rather than memorising samples.
    fig = trajectory_grid_plot(
        experiments,
        oof_predictions,
        title="LOO-CV out-of-fold trajectories, one held-out experiment per panel",
        predicted_label="OOF prediction",
    )
    fig.savefig(FIG_DIR / "notebook_08_loocv_trajectories.png")
    plt.close(fig)

    # ## Leave-one-pH-out
    #
    # LOO-CV always leaves the held-out run's pH in the training set. To
    # match the baseline's transfer test, hold out all four runs at pH 5.85
    # and train on pH 5.0 and 7.0 only. The hybrid then predicts a pH it has
    # never seen.
    held_exps = in_bin(1)
    held_ds = dataset_of(held_exps)
    lopo_p1, lopo_p2 = fit_hybrid(
        dataset_of([e for e, b in zip(experiments, bin_of_exp, strict=True) if b != 1]),
        jr.PRNGKey(7777),
        jr.PRNGKey(8888),
    )
    lopo_result = lopo_p2.model.predict(held_ds)
    lopo_obs, lopo_pred = _pairs(lopo_result.buckets, held_ds)
    lopo_mse, lopo_r2 = _score(lopo_obs, lopo_pred)
    print(
        f"LOPO (pH 5.85 held out): phase 1 loss {lopo_p1.history.best_fitness[-1]:.4e}, "
        f"phase 2 loss {lopo_p2.history.loss[-1]:.4e}, OOF MSE {lopo_mse:.3e}, R² {lopo_r2:.4f}"
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _parity_panel(
        ax,
        lopo_obs,
        lopo_pred,
        "LOPO parity (held-out pH 5.85)",
        s=30,
        alpha=0.8,
        color="C2",
        marker="^",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("predicted (out-of-fold)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "notebook_09_lopo_parity.png")
    plt.close(fig)

    fig = trajectory_grid_plot(
        held_exps,
        lopo_result,
        title="pH 5.85 runs predicted from a model trained on pH 5.0 and 7.0",
        predicted_label="LOPO prediction",
    )
    fig.savefig(FIG_DIR / "notebook_10_lopo_trajectories.png")
    plt.close(fig)

    # The same reveal as before, retrained without pH 5.85. The hybrid
    # curve passes between the two trained pH values: the residual has
    # interpolated the saturation knee from its two neighbours.
    fig = k_reveal_plot(
        lopo_p1.model.predictors,
        lopo_p2.model.predictors,
        "log10 k(pH): LOPO hybrid, trained on pH 5.0 and 7.0 only",
    )
    fig.savefig(FIG_DIR / "notebook_11_k_reveal_lopo.png")
    plt.close(fig)

    # ## Discussion
    #
    # The baseline fits each pH bin well on its own: per-bin LOO-CV sits on
    # the diagonal and the recovered parameters track the truth. It cannot
    # transfer between bins, because the model has nowhere to put pH. The
    # hybrid is the same trunk plus a residual that takes pH as an input,
    # trained on all twelve runs at once, and under leave-one-pH-out it
    # interpolates the saturation knee from the two neighbouring bins.


if __name__ == "__main__":
    main()
