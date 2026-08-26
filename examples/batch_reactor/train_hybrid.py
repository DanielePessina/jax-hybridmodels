"""Batch reactor — hybrid (evosax → optax) training example.

End-to-end script implementing the design in
``examples/batch_reactor/SPEC.md``. The state hooks, the parametric trunk
class, the vector field and the predictor builder live in ``_model.py``,
shared with ``train_rl_deactivation.py``. Everything else, data generation
included, is local to this file.

Pipeline
--------
A first-order ``A -> B`` batch reactor with hidden truth
``k(T, pH) = k_sat(pH) * exp(-Ea/R * (1/T - 1/T_ref))``. Two-phase fit:

* **Phase 1 (evosax/CMA-ES)** — fits a deliberately-too-simple parametric
  trunk: centred Arrhenius (pH-blind), two scalars ``(log_k_ref, Ea)``.
  Residual MLP is frozen at zero contribution (output bound is symmetric
  around 0, so a freshly-init MLP contributes 0 decades of correction).
* **Phase 2 (optax/AdamW)** — freezes the parametric, unfreezes a
  16-neuron MLP residual that adds ``Δlog10(k)(T, pH)`` log-additively
  on top of the Arrhenius trunk. The MLP picks up the pH dependence the
  parametric cannot represent.

Verification checkpoints (per SPEC §9) print at each transition; loose
asserts catch wiring errors.

Run: ``uv run python examples/batch_reactor/train_hybrid.py``
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float
from scipy.stats import qmc

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ``_model`` sits in this script's own directory, so it resolves without the
# sys.path insert above. See its docstring for why these definitions are shared
# with train_rl_deactivation.py rather than defined here.
from _model import (  # noqa: E402
    R_GAS,
    T_REF,
    ArrheniusKinetics,
    build_predictors,
    simulate_fn,
    state_to_output,
    y0_fn,
)
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
)

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Truth
EA_TRUE: float = 30.0  # kJ/mol; rate doubles ~per 10 °C around T_REF

# Saturation curve parameters (slides' k_saturation_from_ph)
K_SAT_BASELINE: float = 0.14
K_SAT_AMPLITUDE: float = 1.05
K_SAT_PH50: float = 5.85
K_SAT_HILL: float = 5.0

# DOE
T_C_RANGE: tuple[float, float] = (15.0, 35.0)
PH_RANGE: tuple[float, float] = (4.5, 7.5)
N_TRAIN_EXPERIMENTS: int = 9
VALIDATION_POINTS: tuple[tuple[float, float], ...] = ((20.0, 5.3), (30.0, 6.8))

# Per-experiment observations
T_MAX: float = 5.0
N_TIMESTEPS: int = 12
CA0: float = 1.0
NOISE_REL: float = 0.03  # heteroscedastic relative-std factor
NOISE_FLOOR: float = 0.02  # absolute floor so near-zero Ca still has finite noise

OUTPUT_CHANNELS: tuple[str, ...] = ("Ca",)


# --------------------------------------------------------------------------- #
# Truth helpers — used for synthetic data generation and plot overlays only.  #
# Never imported into the predictor code path.                                #
# --------------------------------------------------------------------------- #


def _k_sat_from_ph(pH: Array | float) -> Array:
    """Slides' saturation curve. Hidden truth for the pH dependence."""
    pH_arr = jnp.asarray(pH)
    return K_SAT_BASELINE + K_SAT_AMPLITUDE / (
        1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
    )


def _k_true(temperature_C: ArrayLike, pH: ArrayLike) -> Array:
    """Ground-truth rate constant: pH saturation × Arrhenius centred at T_REF."""
    T_K = jnp.asarray(temperature_C) + 273.15
    arrhenius = jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))
    return _k_sat_from_ph(pH) * arrhenius


def _true_ca_trajectory(
    ts: Float[Array, " T"], temperature_C: float, pH: float
) -> Float[Array, " T"]:
    """Closed-form ``Ca(t) = Ca0 * exp(-k * t)`` for the true rate."""
    k = float(_k_true(temperature_C, pH))
    return CA0 * jnp.exp(-k * jnp.asarray(ts))


def _add_heteroscedastic_noise(
    values: Array, *, key: Array, rel: float = NOISE_REL, floor: float = NOISE_FLOOR
) -> Array:
    """Slides' noise model: ``σ = rel · max(|values|, floor)``, clipped at 0."""
    scale = rel * jnp.maximum(jnp.abs(values), floor)
    noisy = values + scale * jr.normal(key, values.shape)
    return jnp.clip(noisy, 0.0, None)


# --------------------------------------------------------------------------- #
# DOE: 9 LHS samples in (T, pH) + 2 hardcoded validation points               #
# --------------------------------------------------------------------------- #


def _lhs_design(seed: int, n: int = N_TRAIN_EXPERIMENTS) -> list[tuple[float, float]]:
    """Latin hypercube samples in ``(T_C, pH)`` with the configured bounds.

    Uses ``scipy.stats.qmc.LatinHypercube`` (already a project dependency; the
    framework's evosax ``init="lhs_box"`` mode uses the same constructor).
    Returns a list of (T_C, pH) tuples in row order.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n)
    lo = np.array([T_C_RANGE[0], PH_RANGE[0]])
    hi = np.array([T_C_RANGE[1], PH_RANGE[1]])
    scaled = lo + (hi - lo) * unit
    return [(float(t), float(ph)) for t, ph in scaled]


# --------------------------------------------------------------------------- #
# Synthetic dataset construction                                              #
# --------------------------------------------------------------------------- #


def _make_experiment_for(
    *,
    temperature_C: float,
    pH: float,
    noise_key: Array,
    exp_id: str,
) -> Experiment:
    """Build one synthetic experiment at fixed ``(T, pH)``.

    Closed-form ``Ca(t) = Ca0 · exp(-k_true · t)`` for the dense truth, then
    ``add_heteroscedastic_noise`` on top to produce the observation series.
    """
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    clean = _true_ca_trajectory(ts, temperature_C=temperature_C, pH=pH)
    noisy = _add_heteroscedastic_noise(clean, key=noise_key)
    sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
    variance = sigma**2
    channels = {"Ca": ChannelObs(ts=ts, values=noisy, variance=variance)}
    return make_experiment(
        covariates={"temperature_C": float(temperature_C), "pH": float(pH)},
        channels=channels,
        y0_fn=y0_fn,
        exp_id=exp_id,
    )


def _build_datasets(
    *,
    doe_seed: int,
    noise_key: Array,
) -> tuple[list[Experiment], list[Experiment]]:
    """Synthesise the 9 training experiments (LHS) + 2 validation experiments."""
    train_design = _lhs_design(seed=doe_seed)
    train_experiments: list[Experiment] = []
    for i, (T_C, pH) in enumerate(train_design):
        # Per-experiment noise key derived from the global noise root via fold-in;
        # this matches the slides' "seed=11+75·i" convention spiritually but routes
        # through JAX's split-keys so the script stays JAX-deterministic.
        k_i = jr.fold_in(noise_key, i)
        train_experiments.append(
            _make_experiment_for(
                temperature_C=T_C,
                pH=pH,
                noise_key=k_i,
                exp_id=f"train_{i:02d}_T{T_C:.1f}_pH{pH:.2f}",
            )
        )

    val_experiments: list[Experiment] = []
    for j, (T_C, pH) in enumerate(VALIDATION_POINTS):
        k_j = jr.fold_in(noise_key, 1000 + j)
        val_experiments.append(
            _make_experiment_for(
                temperature_C=T_C,
                pH=pH,
                noise_key=k_j,
                exp_id=f"val_{j:02d}_T{T_C:.1f}_pH{pH:.2f}",
            )
        )

    return train_experiments, val_experiments


# --------------------------------------------------------------------------- #
# Custom plots                                                                #
# --------------------------------------------------------------------------- #


def _trajectory_grid_plot(
    experiments: list[Experiment],
    predictions_per_exp: list[np.ndarray],
    *,
    title: str,
    save_path: Path,
) -> None:
    """3×3 grid of ``Ca(t)``: truth (solid), observations (scatter), prediction (dashed).

    Panels ordered by experiment index. Each panel labelled with ``(T, pH)``.
    """
    n = len(experiments)
    if n != 9:
        raise ValueError(f"_trajectory_grid_plot expects 9 experiments, got {n}")

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0), sharex=True, sharey=True)
    ts_dense = np.linspace(0.0, T_MAX, 200)
    for ax, exp, pred in zip(axes.flatten(), experiments, predictions_per_exp, strict=True):
        T_C = float(exp.covariates["temperature_C"])
        pH = float(exp.covariates["pH"])
        clean = np.asarray(_true_ca_trajectory(jnp.asarray(ts_dense), T_C, pH))
        ts_obs = np.asarray(exp.channels["Ca"].ts)
        ca_obs = np.asarray(exp.channels["Ca"].values)
        ax.plot(ts_dense, clean, color="black", linewidth=1.4, label="truth")
        ax.scatter(
            ts_obs,
            ca_obs,
            s=20,
            color="C0",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="observed",
        )
        ax.plot(ts_obs, pred[:, 0], color="C3", linestyle="--", linewidth=1.4, label="predicted")
        ax.set_title(f"T={T_C:.1f}°C, pH={pH:.2f}", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("Ca")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99)
    )
    fig.suptitle(title, y=0.995, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(save_path)
    plt.close(fig)


def _k_vs_ph_reveal_plot(
    *,
    predictors_p1: tuple[ArrheniusKinetics, BoundedPredictor],
    predictors_p2: tuple[ArrheniusKinetics, BoundedPredictor] | None,
    train_experiments: list[Experiment],
    val_experiments: list[Experiment],
    title: str,
    save_path: Path,
) -> None:
    """``log10 k(pH)`` at three fixed ``T`` (15, 25, 35°C): truth, parametric, hybrid.

    Three coloured curves per panel-quantity. LHS sample points and validation
    points overlaid as scatter at their actual ``(T, pH)``.
    """
    pH_grid = jnp.linspace(PH_RANGE[0] - 0.4, PH_RANGE[1] + 0.4, 200)
    T_C_lines = (15.0, 25.0, 35.0)
    colors = ("C0", "C1", "C2")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    parametric_p1, residual_p1 = predictors_p1
    parametric_p2, residual_p2 = predictors_p2 if predictors_p2 is not None else (None, None)

    for T_C, color in zip(T_C_lines, colors, strict=True):
        log10_truth = np.log10(np.asarray(_k_true(T_C, pH_grid)))
        ax.plot(pH_grid, log10_truth, color=color, linewidth=1.6, label=f"truth (T={T_C:.0f}°C)")

        log_k_ref, Ea = parametric_p1()
        T_K = T_C + 273.15
        log10_k_param = float((log_k_ref - Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0))
        ax.plot(
            pH_grid,
            np.full_like(pH_grid, log10_k_param),
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"parametric (T={T_C:.0f}°C)" if T_C == T_C_lines[1] else None,
        )

        if predictors_p2 is not None and parametric_p2 is not None and residual_p2 is not None:
            log_k_ref2, Ea2 = parametric_p2()
            log10_k_param2 = float(
                (log_k_ref2 - Ea2 / R_GAS * (1.0 / T_K - 1.0 / T_REF)) / jnp.log(10.0)
            )
            delta_curve = np.array(
                [
                    float(
                        jnp.squeeze(
                            residual_p2({"temperature_C": jnp.asarray(T_C), "pH": jnp.asarray(ph_)})
                        )
                    )
                    for ph_ in pH_grid
                ]
            )
            log10_hybrid = log10_k_param2 + delta_curve
            ax.plot(
                pH_grid,
                log10_hybrid,
                color=color,
                linewidth=1.2,
                linestyle=":",
                label=f"hybrid (T={T_C_lines[1]:.0f}°C)" if T_C == T_C_lines[1] else None,
            )

    train_T = np.array([float(e.covariates["temperature_C"]) for e in train_experiments])
    train_pH = np.array([float(e.covariates["pH"]) for e in train_experiments])
    train_log10k = np.log10(np.asarray(_k_true(train_T, train_pH)))
    val_T = np.array([float(e.covariates["temperature_C"]) for e in val_experiments])
    val_pH = np.array([float(e.covariates["pH"]) for e in val_experiments])
    val_log10k = np.log10(np.asarray(_k_true(val_T, val_pH)))
    ax.scatter(
        train_pH,
        train_log10k,
        s=44,
        color="black",
        marker="o",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="LHS train (truth)",
    )
    ax.scatter(
        val_pH,
        val_log10k,
        s=60,
        color="black",
        marker="^",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="validation (truth)",
    )

    ax.set_xlabel("pH")
    ax.set_ylabel("log10 k")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _loss_curve_plot(
    *,
    history_p1: list[float],
    history_p2: list[float],
    save_path: Path,
) -> None:
    """Concatenated phase-1 (best-loss-per-gen) + phase-2 (loss-per-step) curve."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n1 = len(history_p1)
    n2 = len(history_p2)
    x1 = np.arange(n1)
    x2 = np.arange(n1, n1 + n2)
    ax.plot(
        x1,
        np.log10(np.asarray(history_p1)),
        color="C0",
        linewidth=1.4,
        label="phase 1 — evosax (best-of-gen)",
    )
    ax.plot(
        x2,
        np.log10(np.asarray(history_p2)),
        color="C3",
        linewidth=1.4,
        label="phase 2 — optax (per-step)",
    )
    ax.axvline(n1 - 0.5, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("training progress (generation, then step)")
    ax.set_ylabel("log10 MSE loss")
    ax.set_title("Loss curve across the evosax → optax pipeline")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Verification helpers (SPEC §9)                                              #
# --------------------------------------------------------------------------- #


def _verify_truth_helper() -> None:
    """SPEC §9.1 — ``_k_true`` returns sane values across (T, pH50)."""
    k15 = float(_k_true(15.0, K_SAT_PH50))
    k25 = float(_k_true(25.0, K_SAT_PH50))
    k35 = float(_k_true(35.0, K_SAT_PH50))
    print(f"  _k_true(T=15°C, pH=5.85) = {k15:.4f}")
    print(f"  _k_true(T=25°C, pH=5.85) = {k25:.4f}  (expect ≈ 0.665)")
    print(f"  _k_true(T=35°C, pH=5.85) = {k35:.4f}")
    assert 0.6 < k25 < 0.75, f"k_true at T_REF, pH50 should be ≈0.665, got {k25}"
    assert k15 < k25 < k35, "Arrhenius is monotone increasing in T"


def _verify_dataset_shapes(train_ds, val_ds, train_design, val_design) -> None:
    """SPEC §9.2 — bucket layout and DOE coverage."""
    print(f"  LHS training (T, pH) samples ({len(train_design)}):")
    for i, (T_C, pH) in enumerate(train_design):
        print(f"    [{i}] T={T_C:.2f}°C, pH={pH:.3f}")
    print(f"  Validation (T, pH) samples ({len(val_design)}):")
    for j, (T_C, pH) in enumerate(val_design):
        print(f"    [{j}] T={T_C:.2f}°C, pH={pH:.3f}")
    assert len(train_ds.bucket_payloads) == 1, "training dataset should have one bucket"
    bp_train = train_ds.bucket_payloads[0]
    print(f"  train bucket: ts={tuple(bp_train.ts.shape)}, n_obs={int(bp_train.n_obs)}")
    assert bp_train.ts.shape == (N_TRAIN_EXPERIMENTS, N_TIMESTEPS)
    assert int(bp_train.n_obs) == N_TRAIN_EXPERIMENTS * N_TIMESTEPS
    bp_val = val_ds.bucket_payloads[0]
    print(f"  val   bucket: ts={tuple(bp_val.ts.shape)}, n_obs={int(bp_val.n_obs)}")
    assert bp_val.ts.shape == (len(VALIDATION_POINTS), N_TIMESTEPS)


def _verify_predictors_at_init(
    predictors: tuple[ArrheniusKinetics, BoundedPredictor],
) -> None:
    """SPEC §9.3 — sigmoid-midpoint init values, residual contributes ≈ 0 decades."""
    parametric, residual = predictors
    log_k_ref, Ea = parametric()
    print(f"  parametric init: log_k_ref={float(log_k_ref):.3f}, Ea={float(Ea):.2f} kJ/mol")
    delta = float(
        jnp.squeeze(residual({"temperature_C": jnp.asarray(25.0), "pH": jnp.asarray(K_SAT_PH50)}))
    )
    print(f"  residual init at (T=25°C, pH=5.85): Δlog10(k)={delta:+.3f}  (expect ≈ 0)")
    assert abs(delta) < 0.5, f"fresh-init residual should be near 0, got {delta}"


def _verify_sanity_simulation(
    *,
    predictors: tuple[ArrheniusKinetics, BoundedPredictor],
    train_experiments: list[Experiment],
    solver: SolverConfig,
) -> None:
    """SPEC §9.4 — one simulate_fn call returns a monotone-decreasing Ca trace."""
    exp = train_experiments[0]
    ts = exp.channels["Ca"].ts
    cov = {k: jnp.asarray(v) for k, v in exp.covariates.items()}
    y0 = y0_fn(cov, exp.channels)
    sim = simulate_fn(predictors, ts, cov, y0, solver)
    sim_ca = sim[:, 0]
    print(
        f"  sim Ca[0]={float(sim_ca[0]):.3f}, Ca[-1]={float(sim_ca[-1]):.3f}, "
        f"min={float(jnp.min(sim_ca)):.3f}"
    )
    assert (sim_ca[1:] <= sim_ca[:-1] + 1e-6).all(), "Ca should be non-increasing"
    # ``y0_fn`` reads the (noisy) first observation, so ``sim_ca[0]`` matches
    # that observation — not the clean ``CA0``. Round-trip exactly to ``y0[0]``,
    # and stay loosely near ``CA0`` (the noise is bounded by NOISE_REL ≈ 3%).
    assert abs(float(sim_ca[0]) - float(y0[0])) < 1e-6, "diffrax round-trips y0"
    assert abs(float(sim_ca[0]) - CA0) < 5.0 * NOISE_REL, "Ca starts near CA0"
    assert float(sim_ca[-1]) < float(sim_ca[0]), "something must have decayed"


def _residual_weights_signature(residual: BoundedPredictor) -> Array:
    """Concatenated MLP weight leaves — used to assert phase-1 left them untouched."""
    leaves = jax.tree_util.tree_leaves(residual.inner)
    return jnp.concatenate([leaf.reshape(-1) for leaf in leaves if eqx.is_inexact_array(leaf)])


def _count_trainable_params(predictors: object, mask: object) -> int:
    """Count scalar parameters whose mask leaf is True.

    Walks ``predictors`` and ``mask`` leaves in lockstep. Mask leaves are
    plain ``bool`` (produced by ``tree_map(predicate, predictors)`` over
    the inexact-array predicate), not JAX arrays — so a filter on
    ``leaf.dtype == bool_`` would silently drop them all.
    """
    pred_leaves = jax.tree_util.tree_leaves(predictors)
    mask_leaves = jax.tree_util.tree_leaves(mask)
    n = 0
    for pred, m in zip(pred_leaves, mask_leaves, strict=True):
        if eqx.is_inexact_array(pred) and bool(m):
            n += int(pred.size)
    return n


# --------------------------------------------------------------------------- #
# Script body                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doe-seed", type=int, default=0, help="LHS sampler seed")
    parser.add_argument("--seed", type=int, default=0, help="JAX root key")
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=60)
    parser.add_argument("--sigma-init", type=float, default=0.5)
    parser.add_argument("--init-box-extent", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    root_key = jr.PRNGKey(args.seed)
    k_noise, k_init, k_p1, k_p2 = jr.split(root_key, 4)

    # ---- §9.1 truth helper ------------------------------------------------- #
    print("[verify §9.1] truth helper")
    _verify_truth_helper()

    # ---- §9.2 dataset ------------------------------------------------------ #
    print("\n[build] synthetic dataset")
    train_design = _lhs_design(seed=args.doe_seed)
    train_experiments, val_experiments = _build_datasets(doe_seed=args.doe_seed, noise_key=k_noise)
    train_dataset = make_dataset(
        train_experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    val_dataset = make_dataset(
        val_experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print("[verify §9.2] dataset shapes")
    _verify_dataset_shapes(train_dataset, val_dataset, train_design, list(VALIDATION_POINTS))

    # ---- Solver ------------------------------------------------------------ #
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.05,
    )

    # ---- Predictors -------------------------------------------------------- #
    print("\n[build] predictors")
    predictors = build_predictors(key=k_init)
    print("[verify §9.3] predictors at init")
    _verify_predictors_at_init(predictors)

    print("[verify §9.4] one sanity simulation")
    _verify_sanity_simulation(
        predictors=predictors,
        train_experiments=train_experiments,
        solver=solver,
    )
    # Snapshot the residual MLP weights so we can confirm phase 1 left them frozen.
    residual_weights_pre_p1 = _residual_weights_signature(predictors[1])

    # ---- Phase 1: evosax over the parametric trunk ------------------------- #
    print("\n[phase 1] evosax — parametric trunk only (residual MLP frozen)")
    mask_p1 = trainable_mask(predictors)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors, BoundedPredictor)
    mask_p1 = freeze_modules_of_type(mask_p1, predictors, BoundScaler)

    n_trainable_p1 = _count_trainable_params(predictors, mask_p1)
    print(f"  trainable scalars (Phase 1): {n_trainable_p1} (expect 2)")
    assert n_trainable_p1 == 2, (
        f"Phase 1 should train exactly the parametric latent (2 scalars), got {n_trainable_p1}"
    )

    config_p1 = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=args.population_size,
        num_generations=args.num_generations,
        init="lhs_box",
        init_box_extent=args.init_box_extent,
        sigma_init=args.sigma_init,
        loss="mse",
        verbose=True,
    )
    history_p1, predictors_p1 = train_with_evosax(
        predictors,
        train_dataset,
        config_p1,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p1,
        key=k_p1,
    )
    print(f"  {len(history_p1)} generations; final best loss {history_p1[-1]:.6f}")

    # §9.5 verification — phase 1 transition.
    print("\n[verify §9.5] phase 1 transition")
    parametric_p1, residual_p1 = predictors_p1
    log_k_ref_p1, Ea_p1 = parametric_p1()
    print(f"  recovered log_k_ref={float(log_k_ref_p1):.3f}, Ea={float(Ea_p1):.2f} kJ/mol")
    assert history_p1[-1] < history_p1[0], "evosax must reduce loss across the run"

    residual_weights_post_p1 = _residual_weights_signature(residual_p1)
    if not jnp.array_equal(residual_weights_pre_p1, residual_weights_post_p1):
        warnings.warn(
            "residual MLP weights differ between init and phase-1 endpoint — the "
            "trainability mask did not freeze the residual subtree.",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        print("  residual MLP weights bit-exact unchanged across phase 1 [OK]")

    # ---- Plots after phase 1 ---------------------------------------------- #
    if not args.no_plot:
        print("\n[plot] phase 1 figures")
        train_predictions_p1 = predict_dataset(
            predictors_p1,
            train_dataset,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        val_predictions_p1 = predict_dataset(
            predictors_p1,
            val_dataset,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        diag_train_p1 = compute_diagnostics(train_predictions_p1, train_dataset)
        diag_val_p1 = compute_diagnostics(val_predictions_p1, val_dataset)
        print("  training diagnostics (parametric only):")
        print_diagnostics(diag_train_p1)
        print("  validation diagnostics (parametric only):")
        print_diagnostics(diag_val_p1)

        train_pred_per_exp = [
            np.asarray(train_predictions_p1[0][i]) for i in range(N_TRAIN_EXPERIMENTS)
        ]
        _trajectory_grid_plot(
            train_experiments,
            train_pred_per_exp,
            title="Phase 1 (evosax) — Arrhenius parametric only",
            save_path=args.plot_dir / "01_trajectory_grid_phase1.png",
        )
        parity_plot(
            diag_train_p1,
            title="Phase 1 parity (training set)",
            save_path=args.plot_dir / "02_parity_phase1.png",
        )
        _k_vs_ph_reveal_plot(
            predictors_p1=predictors_p1,
            predictors_p2=None,
            train_experiments=train_experiments,
            val_experiments=val_experiments,
            title="log10 k(pH) — truth vs parametric (Phase 1)",
            save_path=args.plot_dir / "03_k_reveal_phase1.png",
        )

    # ---- Phase 2: optax over the residual MLP ------------------------------ #
    print("\n[phase 2] optax — residual MLP only (parametric frozen)")
    mask_p2 = trainable_mask(predictors_p1)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, ArrheniusKinetics)
    mask_p2 = freeze_modules_of_type(mask_p2, predictors_p1, BoundScaler)

    n_trainable_p2 = _count_trainable_params(predictors_p1, mask_p2)
    print(f"  trainable scalars (Phase 2): {n_trainable_p2} (16-neuron MLP weights+biases)")
    assert n_trainable_p2 > n_trainable_p1, (
        "Phase 2 should train strictly more scalars than Phase 1 (parametric is frozen, "
        "MLP is unfrozen)."
    )

    parametric_latent_pre_p2 = predictors_p1[0].latent

    config_p2 = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        verbose=True,
    )
    history_p2, predictors_p2 = train_with_optax(
        predictors_p1,
        train_dataset,
        config_p2,
        simulate_fn=simulate_fn,
        solver=solver,
        trainable=mask_p2,
        key=k_p2,
    )
    print(f"  {len(history_p2)} steps; final loss {history_p2[-1]:.6f}")

    # §9.6 verification — phase 2 transition.
    print("\n[verify §9.6] phase 2 transition")
    print(f"  history_p1[-1]={history_p1[-1]:.6f}, history_p2[0]={history_p2[0]:.6f}")
    p1_to_p2_gap = abs(history_p2[0] - history_p1[-1])
    if p1_to_p2_gap > 1e-3:
        warnings.warn(
            f"phase 1 endpoint loss ({history_p1[-1]:.6f}) and phase 2 start loss "
            f"({history_p2[0]:.6f}) differ by {p1_to_p2_gap:.4e} — residual contribution "
            "may not be exactly zero at init.",
            RuntimeWarning,
            stacklevel=1,
        )
    if not history_p2[-1] < 0.5 * history_p2[0]:
        warnings.warn(
            f"phase 2 final loss ({history_p2[-1]:.6f}) did not halve the start "
            f"loss ({history_p2[0]:.6f}); the MLP may be under-trained or stuck.",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        print(
            f"  phase 2 reduced loss from {history_p2[0]:.6f} to "
            f"{history_p2[-1]:.6f} ({(1 - history_p2[-1] / history_p2[0]) * 100:.1f}% drop)"
        )

    parametric_latent_post_p2 = predictors_p2[0].latent
    if not jnp.array_equal(parametric_latent_pre_p2, parametric_latent_post_p2):
        warnings.warn(
            "parametric latent differs between phase 1 endpoint and phase 2 endpoint — "
            "the trainability mask did not freeze the parametric subtree.",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        print("  parametric latent bit-exact unchanged across phase 2 [OK]")

    # ---- Plots after phase 2 ---------------------------------------------- #
    if not args.no_plot:
        print("\n[plot] phase 2 figures")
        train_predictions_p2 = predict_dataset(
            predictors_p2,
            train_dataset,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        val_predictions_p2 = predict_dataset(
            predictors_p2,
            val_dataset,
            simulate_fn=simulate_fn,
            solver=solver,
        )
        diag_train_p2 = compute_diagnostics(train_predictions_p2, train_dataset)
        diag_val_p2 = compute_diagnostics(val_predictions_p2, val_dataset)
        print("  training diagnostics (hybrid):")
        print_diagnostics(diag_train_p2)
        print("  validation diagnostics (hybrid):")
        print_diagnostics(diag_val_p2)

        train_pred_per_exp_p2 = [
            np.asarray(train_predictions_p2[0][i]) for i in range(N_TRAIN_EXPERIMENTS)
        ]
        _trajectory_grid_plot(
            train_experiments,
            train_pred_per_exp_p2,
            title="Phase 2 (optax) — Arrhenius + residual MLP",
            save_path=args.plot_dir / "04_trajectory_grid_phase2.png",
        )
        parity_plot(
            diag_train_p2,
            title="Phase 2 parity (training set)",
            save_path=args.plot_dir / "05_parity_phase2.png",
        )
        _k_vs_ph_reveal_plot(
            predictors_p1=predictors_p1,
            predictors_p2=predictors_p2,
            train_experiments=train_experiments,
            val_experiments=val_experiments,
            title="log10 k(pH) — truth vs parametric vs hybrid (Phase 2)",
            save_path=args.plot_dir / "06_k_reveal_phase2.png",
        )
        _loss_curve_plot(
            history_p1=history_p1,
            history_p2=history_p2,
            save_path=args.plot_dir / "07_loss_curve.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
