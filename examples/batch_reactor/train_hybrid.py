"""Batch reactor: fit a parametric trunk, then learn what it cannot represent.

A first-order ``A -> B`` reactor whose hidden rate is
``k(T, pH) = k_sat(pH) * exp(-Ea/R * (1/T - 1/T_ref))``. The model knows the
Arrhenius form in ``T`` and nothing about pH. It is fitted in two phases on
one ``HybridModel``:

* **Phase 1, CMA-ES.** Fits a two-scalar Arrhenius trunk that cannot see
  pH. The residual network is frozen with a ``trainable`` mask; its
  symmetric output bound means a fresh one adds nothing to the rate.
* **Phase 2, AdamW.** Freezes the trunk and trains a 16-neuron residual
  ``Δlog10 k(T, pH)``, which picks up the pH dependence.

Each experiment heats along a flat-ramp-flat temperature profile whose
parameters are ordinary covariates; ``simulate_fn`` in ``_model.py``
evaluates ``ramp_profile`` inside the vector field.

Library features shown: ``HybridModel.fit`` with an Evosax and an Optax
config, ``trainable_mask`` with ``freeze_modules_of_type`` to switch which
part trains, ``FitResult.history``, ``HybridModel.predict``,
``compute_metrics`` and ``save_predictors``.

The script prints checkpoints with loose asserts that catch wiring errors,
then writes seven figures to ``figures/``. ``--save-predictors PATH`` saves
the phase-2 predictors for ``train_rl_deactivation.py``.

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

from jaxhybridmodels import (
    BoundedPredictor,
    ChannelObs,
    Dataset,
    Experiment,
    HybridModel,
    PredictionResult,
    SolverConfig,
    compute_metrics,
    count_trainable_params,
    evaluate_predictor,
    freeze_modules_of_type,
    make_dataset,
    make_experiment,
    print_metrics,
    ramp_profile,
    save_predictors,
    trainable_mask,
)
from jaxhybridmodels.training import EvosaxTrainingConfig, OptaxTrainingConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ``_model`` sits in this script's own directory, so it resolves without the
# insert above. Its docstring says why those definitions are shared.
from _model import (  # noqa: E402
    R_GAS,
    T_REF,
    ArrheniusKinetics,
    build_predictors,
    log10_k,
    log10_k_trunk,
    simulate_fn,
    state_to_output,
    y0_fn,
)
from _shared import apply_default_style, parity_diagnostics, parity_plot  # noqa: E402

EA_TRUE: float = 30.0  # kJ/mol; rate doubles about every 10 °C near T_REF

# The hidden pH dependence: a Hill-type saturation curve.
K_SAT_BASELINE: float = 0.14
K_SAT_AMPLITUDE: float = 1.05
K_SAT_PH50: float = 5.85
K_SAT_HILL: float = 5.0

# Design of experiments.
T_C_RANGE: tuple[float, float] = (15.0, 35.0)
PH_RANGE: tuple[float, float] = (4.5, 7.5)
N_TRAIN_EXPERIMENTS: int = 9
VALIDATION_POINTS: tuple[tuple[float, float], ...] = ((20.0, 5.3), (30.0, 6.8))

# Heating ramp: the jacket goes from T_lo to T_hi between RAMP_T0 and
# RAMP_T1, flat before and after, so each run sweeps a band of temperatures.
RAMP_HALF_WIDTH: float = 3.0  # °C, each experiment sweeps T_C ± 3
RAMP_T0: float = 0.5
RAMP_T1: float = 4.5  # before T_MAX, so both flat edges are visible

# Observations.
T_MAX: float = 5.0
N_TIMESTEPS: int = 12
CA0: float = 1.0
NOISE_REL: float = 0.03  # noise std as a fraction of the value
NOISE_FLOOR: float = 0.02  # so near-zero Ca still has finite noise
OUTPUT_CHANNELS: tuple[str, ...] = ("Ca",)


# Truth helpers, used for data generation and plot overlays only.


def _k_true(temperature_C: ArrayLike, pH: ArrayLike) -> Array:
    """Ground-truth rate constant: pH saturation times Arrhenius centred at T_REF."""
    T_K = jnp.asarray(temperature_C) + 273.15
    arrhenius = jnp.exp(-EA_TRUE / R_GAS * (1.0 / T_K - 1.0 / T_REF))
    pH_arr = jnp.asarray(pH)
    k_sat = K_SAT_BASELINE + K_SAT_AMPLITUDE / (
        1.0 + jnp.maximum(pH_arr / K_SAT_PH50, 0.0) ** K_SAT_HILL
    )
    return k_sat * arrhenius


def _true_ca_trajectory(
    ts: Float[Array, " T"],
    *,
    ramp_t0: float,
    ramp_t1: float,
    T_lo: float,
    T_hi: float,
    pH: float,
) -> Float[Array, " T"]:
    """True ``Ca(t)`` under the heating ramp, ``dCa/dt = -k_true(T(t), pH) Ca``.

    No closed form once ``T`` varies, so it is integrated at tight
    tolerances. ``T(t)`` is the same ``ramp_profile`` the model evaluates,
    so model and truth differ only in the rate law.
    """
    T_profile = ramp_profile(t0=ramp_t0, t1=ramp_t1, v0=T_lo, v1=T_hi)

    def field(t: ArrayLike, y: Array, args: object) -> Array:
        return -_k_true(T_profile(t), pH) * y

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=0.01,
        y0=jnp.asarray(CA0, dtype=jnp.float32),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=10_000,
    )
    return jnp.asarray(sol.ys)


def _lhs_design(seed: int, n: int = N_TRAIN_EXPERIMENTS) -> list[tuple[float, float]]:
    """Latin hypercube samples of ``(T_C, pH)``, in row order."""
    unit = qmc.LatinHypercube(d=2, seed=seed).random(n=n)
    lo = np.array([T_C_RANGE[0], PH_RANGE[0]])
    hi = np.array([T_C_RANGE[1], PH_RANGE[1]])
    return [(float(t), float(ph)) for t, ph in lo + (hi - lo) * unit]


def _make_experiment_for(*, T_C: float, pH: float, noise_key: Array, exp_id: str) -> Experiment:
    """One synthetic run ramping through ``T_C ± RAMP_HALF_WIDTH`` at fixed pH.

    Noise is heteroscedastic, ``sigma = NOISE_REL * max(|Ca|, NOISE_FLOOR)``,
    clipped at zero, and ``sigma**2`` is stored as the channel variance.
    """
    ramp = {
        "ramp_t0": RAMP_T0,
        "ramp_t1": RAMP_T1,
        "T_lo": T_C - RAMP_HALF_WIDTH,
        "T_hi": T_C + RAMP_HALF_WIDTH,
    }
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    clean = _true_ca_trajectory(ts, pH=pH, **ramp)
    sigma = NOISE_REL * jnp.maximum(jnp.abs(clean), NOISE_FLOOR)
    noisy = jnp.clip(clean + sigma * jr.normal(noise_key, clean.shape), 0.0, None)
    return make_experiment(
        covariates={**ramp, "pH": pH},
        channels={"Ca": ChannelObs(ts=ts, values=noisy, variance=sigma**2)},
        y0_fn=y0_fn,
        exp_id=exp_id,
    )


def _trajectory_grid_plot(
    experiments: list[Experiment],
    predictions: PredictionResult,
    *,
    title: str,
    save_path: Path,
) -> None:
    """3x3 grid of ``Ca(t)``: truth solid, observations as dots, prediction dashed."""
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0), sharex=True, sharey=True)
    ts_dense = np.linspace(0.0, T_MAX, 200)
    for ax, exp in zip(axes.flatten(), experiments, strict=True):
        cov = {name: float(value) for name, value in exp.covariates.items()}
        clean = _true_ca_trajectory(
            jnp.asarray(ts_dense),
            ramp_t0=cov["ramp_t0"],
            ramp_t1=cov["ramp_t1"],
            T_lo=cov["T_lo"],
            T_hi=cov["T_hi"],
            pH=cov["pH"],
        )
        obs = exp.channels["Ca"]
        pred = predictions[exp.exp_id]
        ax.plot(ts_dense, np.asarray(clean), color="black", linewidth=1.4, label="truth")
        ax.scatter(
            np.asarray(obs.ts),
            np.asarray(obs.values),
            s=20,
            color="C0",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="observed",
        )
        ax.plot(
            np.asarray(pred.times),
            np.asarray(pred.outputs["Ca"]),
            color="C3",
            linestyle="--",
            linewidth=1.4,
            label="predicted",
        )
        ax.set_title(f"T {cov['T_lo']:.0f}→{cov['T_hi']:.0f}°C, pH={cov['pH']:.2f}", fontsize=9)
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
    """``log10 k(pH)`` at 15, 25 and 35 °C: truth solid, trunk dashed, hybrid dotted.

    Training and validation points are drawn at their ramp-midpoint
    temperature on the truth curve.
    """
    pH_grid = jnp.linspace(PH_RANGE[0] - 0.4, PH_RANGE[1] + 0.4, 200)
    T_C_lines = (15.0, 25.0, 35.0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for T_C, color in zip(T_C_lines, ("C0", "C1", "C2"), strict=True):
        middle = T_C == T_C_lines[1]
        ax.plot(
            pH_grid,
            np.log10(np.asarray(_k_true(T_C, pH_grid))),
            color=color,
            linewidth=1.6,
            label=f"truth (T={T_C:.0f}°C)",
        )
        trunk = float(log10_k_trunk(predictors_p1[0], T_C))
        ax.plot(
            pH_grid,
            np.full_like(pH_grid, trunk),
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"parametric (T={T_C:.0f}°C)" if middle else None,
        )
        if predictors_p2 is not None:
            hybrid = jax.vmap(lambda ph, T_C=T_C: log10_k(predictors_p2, T_C, ph))(pH_grid)
            ax.plot(
                pH_grid,
                np.asarray(hybrid),
                color=color,
                linewidth=1.2,
                linestyle=":",
                label=f"hybrid (T={T_C:.0f}°C)" if middle else None,
            )

    for experiments, marker, size, label in (
        (train_experiments, "o", 44, "LHS train (truth)"),
        (val_experiments, "^", 60, "validation (truth)"),
    ):
        T_mid = np.array(
            [(float(e.covariates["T_lo"]) + float(e.covariates["T_hi"])) / 2.0 for e in experiments]
        )
        pH = np.array([float(e.covariates["pH"]) for e in experiments])
        ax.scatter(
            pH,
            np.log10(np.asarray(_k_true(T_mid, pH))),
            s=size,
            color="black",
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label=label,
        )

    ax.set_xlabel("pH")
    ax.set_ylabel("log10 k")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _loss_curve_plot(history_p1: tuple[float, ...], history_p2: tuple[float, ...], path: Path):
    """Both phases end to end: CMA-ES best-so-far per generation, then AdamW per step."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n1 = len(history_p1)
    ax.plot(
        np.arange(n1),
        np.log10(history_p1),
        color="C0",
        linewidth=1.4,
        label="phase 1, evosax (best so far)",
    )
    ax.plot(
        np.arange(n1, n1 + len(history_p2)),
        np.log10(history_p2),
        color="C3",
        linewidth=1.4,
        label="phase 2, optax (per step)",
    )
    ax.axvline(n1 - 0.5, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("training progress (generation, then step)")
    ax.set_ylabel("log10 MSE loss")
    ax.set_title("Loss across the evosax → optax pipeline")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _report(
    model: HybridModel, train: Dataset, val: Dataset, label: str
) -> tuple[PredictionResult, PredictionResult]:
    """Print training and validation metrics and return both prediction sets."""
    train_pred = model.predict(train)
    val_pred = model.predict(val)
    print(f"  training metrics ({label}):")
    print_metrics(compute_metrics(train_pred.buckets, train))
    print(f"  validation metrics ({label}):")
    print_metrics(compute_metrics(val_pred.buckets, val))
    return train_pred, val_pred


def _inner_weights(residual: BoundedPredictor) -> Array:
    """The residual network's weights as one vector, to check a phase left them alone."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(residual.inner, eqx.is_inexact_array))
    return jnp.concatenate([leaf.reshape(-1) for leaf in leaves])


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
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--save-predictors",
        type=Path,
        default=None,
        help="Write the phase-2 predictors here; train_rl_deactivation.py loads them.",
    )
    args = parser.parse_args()

    apply_default_style()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    k_noise, k_init, k_p1, k_p2 = jr.split(jr.PRNGKey(args.seed), 4)

    print("[verify] truth helper")
    k15, k25, k35 = (float(_k_true(T, K_SAT_PH50)) for T in (15.0, 25.0, 35.0))
    print(f"  k_true at pH 5.85: {k15:.4f} (15°C), {k25:.4f} (25°C), {k35:.4f} (35°C)")
    assert 0.6 < k25 < 0.75, f"k_true at T_REF and pH50 should be about 0.665, got {k25}"
    assert k15 < k25 < k35, "Arrhenius is monotone increasing in T"

    print("\n[build] synthetic dataset")
    # The LHS samples the ramp midpoint; each run sweeps ±RAMP_HALF_WIDTH
    # around it. Noise keys are folded from one root so the data is
    # reproducible.
    train_design = _lhs_design(seed=args.doe_seed)
    train_experiments = [
        _make_experiment_for(
            T_C=T_C,
            pH=pH,
            noise_key=jr.fold_in(k_noise, i),
            exp_id=f"train_{i:02d}_T{T_C:.1f}_pH{pH:.2f}",
        )
        for i, (T_C, pH) in enumerate(train_design)
    ]
    val_experiments = [
        _make_experiment_for(
            T_C=T_C,
            pH=pH,
            noise_key=jr.fold_in(k_noise, 1000 + j),
            exp_id=f"val_{j:02d}_T{T_C:.1f}_pH{pH:.2f}",
        )
        for j, (T_C, pH) in enumerate(VALIDATION_POINTS)
    ]
    train_dataset = make_dataset(train_experiments, output_channel_names=OUTPUT_CHANNELS)
    val_dataset = make_dataset(val_experiments, output_channel_names=OUTPUT_CHANNELS)

    print("[verify] dataset shapes")
    for i, (T_C, pH) in enumerate(train_design):
        T_lo, T_hi = T_C - RAMP_HALF_WIDTH, T_C + RAMP_HALF_WIDTH
        print(f"  train [{i}] T {T_lo:.1f}→{T_hi:.1f}°C, pH={pH:.3f}")
    for j, (T_C, pH) in enumerate(VALIDATION_POINTS):
        print(f"  val   [{j}] T={T_C:.2f}°C, pH={pH:.3f}")
    # Every run shares one 12-point grid, so each dataset is a single bucket.
    assert len(train_dataset.bucket_payloads) == 1
    assert train_dataset.bucket_payloads[0].ts.shape == (N_TRAIN_EXPERIMENTS, N_TIMESTEPS)
    assert val_dataset.bucket_payloads[0].ts.shape == (len(VALIDATION_POINTS), N_TIMESTEPS)

    solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05)
    model = HybridModel(
        predictors=build_predictors(key=k_init),
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
        solver=solver,
    )

    print("\n[verify] predictors at init")
    parametric, residual = model.predictors
    log_k_ref, Ea = parametric()
    print(f"  trunk: log_k_ref={float(log_k_ref):.3f}, Ea={float(Ea):.2f} kJ/mol")
    delta = evaluate_predictor(residual, {"temperature_C": 25.0, "pH": K_SAT_PH50})
    print(f"  residual at (25°C, pH 5.85): Δlog10 k={delta:+.3f}  (expect about 0)")
    assert abs(delta) < 0.5, f"a fresh residual should sit near 0, got {delta}"

    # One simulation before training: Ca should start at the first
    # observation and decay monotonically.
    exp = train_experiments[0]
    sim_ca = np.asarray(
        model.simulate(
            ts=exp.channels["Ca"].ts,
            y0=y0_fn(exp.covariates, exp.channels),
            covariates=exp.covariates,
        ).outputs["Ca"]
    )
    print(f"  untrained Ca: {sim_ca[0]:.3f} → {sim_ca[-1]:.3f}")
    assert (sim_ca[1:] <= sim_ca[:-1] + 1e-6).all(), "Ca should be non-increasing"
    assert abs(sim_ca[0] - CA0) < 5.0 * NOISE_REL, "Ca starts near CA0"

    print("\n[phase 1] evosax on the trunk, residual frozen")
    # The default mask already holds BoundScaler leaves fixed; this freezes
    # the whole residual on top, leaving the trunk's two latent scalars.
    mask_p1 = freeze_modules_of_type(
        trainable_mask(model.predictors), model.predictors, BoundedPredictor
    )
    n_p1 = count_trainable_params(model.predictors, mask_p1)
    print(f"  trainable scalars: {n_p1}")
    assert n_p1 == 2, f"phase 1 should train the trunk's 2 scalars, got {n_p1}"
    residual_before = _inner_weights(residual)

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
    fit_p1 = model.fit(train_dataset, config_p1, key=k_p1, trainable=mask_p1)
    model_p1 = fit_p1.model
    history_p1 = fit_p1.history.best_fitness
    log_k_ref, Ea = model_p1.predictors[0]()
    print(f"  {len(history_p1)} generations, best loss {history_p1[-1]:.6f}")
    print(f"  recovered log_k_ref={float(log_k_ref):.3f}, Ea={float(Ea):.2f} kJ/mol")
    assert history_p1[-1] < history_p1[0], "evosax must reduce the loss"
    assert jnp.array_equal(residual_before, _inner_weights(model_p1.predictors[1])), (
        "the mask should have frozen the residual during phase 1"
    )

    if not args.no_plot:
        train_pred, _ = _report(model_p1, train_dataset, val_dataset, "trunk only")
        _trajectory_grid_plot(
            train_experiments,
            train_pred,
            title="Phase 1 (evosax): Arrhenius trunk only",
            save_path=args.plot_dir / "01_trajectory_grid_phase1.png",
        )
        parity_plot(
            parity_diagnostics(train_pred.buckets, train_dataset),
            title="Phase 1 parity (training set)",
            save_path=args.plot_dir / "02_parity_phase1.png",
        )
        _k_vs_ph_reveal_plot(
            predictors_p1=model_p1.predictors,
            predictors_p2=None,
            train_experiments=train_experiments,
            val_experiments=val_experiments,
            title="log10 k(pH): truth vs parametric (phase 1)",
            save_path=args.plot_dir / "03_k_reveal_phase1.png",
        )

    print("\n[phase 2] optax on the residual, trunk frozen")
    mask_p2 = freeze_modules_of_type(
        trainable_mask(model_p1.predictors), model_p1.predictors, ArrheniusKinetics
    )
    print(f"  trainable scalars: {count_trainable_params(model_p1.predictors, mask_p2)}")
    config_p2 = OptaxTrainingConfig(steps=args.steps, lr=args.lr, loss="mse", verbose=True)
    fit_p2 = model_p1.fit(train_dataset, config_p2, key=k_p2, trainable=mask_p2)
    model_p2 = fit_p2.model
    history_p2 = fit_p2.history.loss
    print(f"  {len(history_p2)} steps, loss {history_p2[0]:.6f} → {history_p2[-1]:.6f}")

    # The residual starts at zero decades, so phase 2 starts where phase 1
    # ended and the seam is invisible in the loss curve. Both checks warn
    # rather than fail, so short runs with --steps or --num-generations
    # still finish.
    gap = abs(history_p2[0] - history_p1[-1])
    if gap > 1e-3:
        warnings.warn(f"phase 2 started {gap:.2e} away from the phase-1 loss", stacklevel=1)
    if not history_p2[-1] < 0.5 * history_p2[0]:
        warnings.warn(
            "phase 2 did not halve the loss; the residual may be undertrained", stacklevel=1
        )
    assert jnp.array_equal(model_p1.predictors[0].latent, model_p2.predictors[0].latent), (
        "the mask should have frozen the trunk during phase 2"
    )

    if args.save_predictors is not None:
        args.save_predictors.parent.mkdir(parents=True, exist_ok=True)
        save_predictors(args.save_predictors, model_p2.predictors)
        print(f"\n[save] phase-2 predictors written to {args.save_predictors}")

    if not args.no_plot:
        train_pred, _ = _report(model_p2, train_dataset, val_dataset, "hybrid")
        _trajectory_grid_plot(
            train_experiments,
            train_pred,
            title="Phase 2 (optax): Arrhenius trunk plus residual",
            save_path=args.plot_dir / "04_trajectory_grid_phase2.png",
        )
        parity_plot(
            parity_diagnostics(train_pred.buckets, train_dataset),
            title="Phase 2 parity (training set)",
            save_path=args.plot_dir / "05_parity_phase2.png",
        )
        _k_vs_ph_reveal_plot(
            predictors_p1=model_p1.predictors,
            predictors_p2=model_p2.predictors,
            train_experiments=train_experiments,
            val_experiments=val_experiments,
            title="log10 k(pH): truth vs parametric vs hybrid (phase 2)",
            save_path=args.plot_dir / "06_k_reveal_phase2.png",
        )
        _loss_curve_plot(history_p1, history_p2, args.plot_dir / "07_loss_curve.png")
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
