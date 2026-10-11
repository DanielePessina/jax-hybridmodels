"""Batch reactor: an Arrhenius trunk fitted by CMA-ES, then a residual network by AdamW.

A first-order ``A -> B`` reaction whose rate constant depends on
temperature through Arrhenius, which the model knows, and on pH through a
saturation curve, which it does not. The model is a two-scalar Arrhenius
trunk plus a 16-neuron residual ``Δlog10 k(T, pH)``. Each run heats along a
temperature ramp, so ``T`` changes inside every solve.

Shows:

- one ``HybridModel`` fitted twice, with an ``EvosaxTrainingConfig`` and
  then an ``OptaxTrainingConfig``
- ``fit(freeze=...)`` with a predictor class, to switch which half of the
  predictor tuple trains
- a ``ramp_profile`` evaluated inside the vector field from scalar
  covariates
- ``FitResult.history``, ``predict``, ``simulate(experiment=...)``,
  ``compute_metrics`` and ``evaluate_predictor``
- ``save_predictors`` for the RL example, which loads this fitted model

Run::

    uv run python examples/batch_reactor/train_hybrid.py
    uv run python examples/batch_reactor/train_hybrid.py --no-plot \\
        --save-predictors examples/batch_reactor/artefacts/trunk_fresh.eqx

Prints recovered parameters and metrics after each phase, and writes seven
figures to ``figures/``. About 30 s on CPU.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

import jaxhybridmodels as hm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ``_model`` and ``_data`` sit in this script's own directory, so they
# resolve without the insert above, which is for ``_shared``.
from _data import (  # noqa: E402
    CA0,
    K_SAT_PH50,
    N_TIMESTEPS,
    N_TRAIN_EXPERIMENTS,
    NOISE_REL,
    OUTPUT_CHANNELS,
    PH_RANGE,
    RAMP_HALF_WIDTH,
    T_MAX,
    VALIDATION_POINTS,
    k_true,
    lhs_design,
    make_ramp_experiment,
    ramp_midpoint,
    true_ca,
)
from _model import (  # noqa: E402
    SOLVER,
    ArrheniusKinetics,
    build_predictors,
    log10_k,
    log10_k_trunk,
    simulate_fn,
    state_to_output,
)
from _shared import apply_default_style, parity_diagnostics, parity_plot  # noqa: E402


def _trajectory_grid_plot(
    experiments: list[hm.Experiment],
    predictions: hm.PredictionResult,
    *,
    title: str,
    save_path: Path,
) -> None:
    """3x3 grid of ``Ca(t)``: truth solid, observations as dots, prediction dashed."""
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0), sharex=True, sharey=True)
    ts_dense = np.linspace(0.0, T_MAX, 200)
    for ax, exp in zip(axes.flatten(), experiments, strict=True):
        pH = float(exp.covariates["pH"])
        clean = true_ca(jnp.asarray(ts_dense), T_C=ramp_midpoint(exp), pH=pH)
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
        T_lo, T_hi = float(exp.covariates["T_lo"]), float(exp.covariates["T_hi"])
        ax.set_title(f"T {T_lo:.0f}→{T_hi:.0f}°C, pH={pH:.2f}", fontsize=9)
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
    predictors_p1: tuple[ArrheniusKinetics, hm.BoundedPredictor],
    predictors_p2: tuple[ArrheniusKinetics, hm.BoundedPredictor] | None,
    train_experiments: list[hm.Experiment],
    val_experiments: list[hm.Experiment],
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
            np.log10(np.asarray(k_true(T_C, pH_grid))),
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
            hybrid = log10_k(predictors_p2, T_C, pH_grid)
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
        T_mid = np.array([ramp_midpoint(e) for e in experiments])
        pH = np.array([float(e.covariates["pH"]) for e in experiments])
        ax.scatter(
            pH,
            np.log10(np.asarray(k_true(T_mid, pH))),
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
    model: hm.HybridModel, train: hm.Dataset, val: hm.Dataset, label: str
) -> hm.PredictionResult:
    """Print training and validation metrics and return the training predictions."""
    train_pred = model.predict(train)
    print(f"  training metrics ({label}):")
    hm.print_metrics(train_pred)
    print(f"  validation metrics ({label}):")
    hm.print_metrics(model.predict(val))
    return train_pred


def _inner_weights(residual: hm.BoundedPredictor) -> Array:
    """The residual network's weights as one vector, to check a phase left them alone."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(residual.inner, eqx.is_inexact_array))
    return jnp.concatenate([leaf.reshape(-1) for leaf in leaves])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doe-seed", type=int, default=0, help="LHS sampler seed")
    parser.add_argument("--seed", type=int, default=0, help="JAX root key")
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=60, help="CMA-ES generations")
    parser.add_argument("--sigma-init", type=float, default=0.5)
    parser.add_argument("--init-box-extent", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=800, help="AdamW steps")
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
    k_noise, k_init, k_p1, k_p2 = jr.split(jr.PRNGKey(args.seed), 4)

    print("[data] synthetic ramp runs")
    k15, k25, k35 = (float(k_true(T, K_SAT_PH50)) for T in (15.0, 25.0, 35.0))
    print(f"  k_true at pH 5.85: {k15:.4f} (15°C), {k25:.4f} (25°C), {k35:.4f} (35°C)")
    assert 0.6 < k25 < 0.75, f"k_true at T_REF and pH50 should be about 0.665, got {k25}"
    assert k15 < k25 < k35, "Arrhenius is monotone increasing in T"

    # The LHS samples the ramp midpoint; each run sweeps ±RAMP_HALF_WIDTH
    # around it. Noise keys are folded from one root so the data is
    # reproducible.
    train_design = lhs_design(seed=args.doe_seed)
    train_experiments = [
        make_ramp_experiment(
            T_C=T_C,
            pH=pH,
            noise_key=jr.fold_in(k_noise, i),
            exp_id=f"train_{i:02d}_T{T_C:.1f}_pH{pH:.2f}",
        )
        for i, (T_C, pH) in enumerate(train_design)
    ]
    val_experiments = [
        make_ramp_experiment(
            T_C=T_C,
            pH=pH,
            noise_key=jr.fold_in(k_noise, 1000 + j),
            exp_id=f"val_{j:02d}_T{T_C:.1f}_pH{pH:.2f}",
        )
        for j, (T_C, pH) in enumerate(VALIDATION_POINTS)
    ]
    # Built once here so the bucket layout can be checked; fit and predict
    # would also take the experiment lists directly.
    train_dataset = hm.make_dataset(train_experiments, output_channel_names=OUTPUT_CHANNELS)
    val_dataset = hm.make_dataset(val_experiments, output_channel_names=OUTPUT_CHANNELS)
    for i, (T_C, pH) in enumerate(train_design):
        T_lo, T_hi = T_C - RAMP_HALF_WIDTH, T_C + RAMP_HALF_WIDTH
        print(f"  train [{i}] T {T_lo:.1f}→{T_hi:.1f}°C, pH={pH:.3f}")
    for j, (T_C, pH) in enumerate(VALIDATION_POINTS):
        print(f"  val   [{j}] T={T_C:.2f}°C, pH={pH:.3f}")
    # Every run shares one 12-point grid, so each dataset is a single bucket.
    assert len(train_dataset.bucket_payloads) == 1
    assert train_dataset.bucket_payloads[0].ts.shape == (N_TRAIN_EXPERIMENTS, N_TIMESTEPS)
    assert val_dataset.bucket_payloads[0].ts.shape == (len(VALIDATION_POINTS), N_TIMESTEPS)

    # region model
    model = hm.HybridModel(
        predictors=build_predictors(key=k_init),
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
        solver=SOLVER,
        # endregion model
    )

    # Before training: the residual's symmetric output box puts a fresh
    # network near zero decades, and one simulation should start at the
    # first observation and decay monotonically.
    parametric, residual = model.predictors
    delta = hm.evaluate_predictor(residual, {"temperature_C": 25.0, "pH": K_SAT_PH50})
    print(f"  residual at init, (25°C, pH 5.85): Δlog10 k={delta:+.3f}  (expect about 0)")
    assert abs(delta) < 0.5, f"a fresh residual should sit near 0, got {delta}"
    sim_ca = np.asarray(model.simulate(experiment=train_experiments[0]).outputs["Ca"])
    assert (sim_ca[1:] <= sim_ca[:-1] + 1e-6).all(), "Ca should be non-increasing"
    assert abs(sim_ca[0] - CA0) < 5.0 * NOISE_REL, "Ca starts near CA0"

    print("\n[fit] phase 1: CMA-ES on the trunk, residual frozen")
    # freeze_mask is the mask behind fit(freeze=...), used here only to count
    # what phase 1 will train.
    n_p1 = hm.count_trainable_params(
        model.predictors, hm.freeze_mask(model.predictors, hm.BoundedPredictor)
    )
    print(f"  trainable scalars: {n_p1}")
    assert n_p1 == 2, f"phase 1 should train the trunk's 2 scalars, got {n_p1}"
    residual_before = _inner_weights(residual)

    # region phase-1
    config_p1 = hm.EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=args.population_size,
        steps=args.generations,
        init="lhs_box",
        init_box_extent=args.init_box_extent,
        sigma_init=args.sigma_init,
    )
    # The default mask already holds BoundScaler leaves fixed. Freezing every
    # BoundedPredictor on top leaves the trunk's two latent scalars.
    fit_p1 = model.fit(train_dataset, config_p1, key=k_p1, freeze=hm.BoundedPredictor)
    model_p1 = fit_p1.model
    # endregion phase-1
    # Evosax records the best-so-far loss, one entry per generation.
    history_p1 = fit_p1.history.loss
    log_k_ref, Ea = model_p1.predictors[0]()
    print(f"  {len(history_p1)} generations, best loss {history_p1[-1]:.6f}")
    print(f"  recovered log_k_ref={float(log_k_ref):.3f}, Ea={float(Ea):.2f} kJ/mol")
    assert history_p1[-1] <= history_p1[0], "the best-so-far loss cannot rise"
    assert jnp.array_equal(residual_before, _inner_weights(model_p1.predictors[1])), (
        "freeze= should have held the residual fixed during phase 1"
    )

    print("\n[fit] phase 2: AdamW on the residual, trunk frozen")
    mask_p2 = hm.freeze_mask(model_p1.predictors, ArrheniusKinetics)
    print(f"  trainable scalars: {hm.count_trainable_params(model_p1.predictors, mask_p2)}")
    # region phase-2
    config_p2 = hm.OptaxTrainingConfig(steps=args.steps, lr=args.lr)
    fit_p2 = model_p1.fit(train_dataset, config_p2, key=k_p2, freeze=ArrheniusKinetics)
    # endregion phase-2
    model_p2 = fit_p2.model
    history_p2 = fit_p2.history.loss
    print(f"  {len(history_p2)} steps, loss {history_p2[0]:.6f} → {history_p2[-1]:.6f}")

    # The residual starts at zero decades, so phase 2 starts where phase 1
    # ended and the seam is invisible in the loss curve. Both checks warn
    # rather than fail, so short runs with --steps or --generations finish.
    gap = abs(history_p2[0] - history_p1[-1])
    if gap > 1e-3:
        warnings.warn(f"phase 2 started {gap:.2e} away from the phase-1 loss", stacklevel=1)
    if not history_p2[-1] < 0.5 * history_p2[0]:
        warnings.warn(
            "phase 2 did not halve the loss; the residual may be undertrained", stacklevel=1
        )
    assert jnp.array_equal(model_p1.predictors[0].latent, model_p2.predictors[0].latent), (
        "freeze= should have held the trunk fixed during phase 2"
    )

    print("\n[results]")
    train_pred_p1 = _report(model_p1, train_dataset, val_dataset, "trunk only")
    train_pred_p2 = _report(model_p2, train_dataset, val_dataset, "hybrid")

    if args.save_predictors is not None:
        args.save_predictors.parent.mkdir(parents=True, exist_ok=True)
        hm.save_predictors(args.save_predictors, model_p2.predictors)
        print(f"  phase-2 predictors written to {args.save_predictors}")

    if args.no_plot:
        return
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    for phase, (title, pred, models) in enumerate(
        (
            ("Phase 1 (CMA-ES): Arrhenius trunk only", train_pred_p1, (model_p1, None)),
            ("Phase 2 (AdamW): trunk plus residual", train_pred_p2, (model_p1, model_p2)),
        )
    ):
        offset = 3 * phase
        _trajectory_grid_plot(
            train_experiments,
            pred,
            title=title,
            save_path=args.plot_dir / f"{offset + 1:02d}_trajectory_grid_phase{phase + 1}.png",
        )
        parity_plot(
            parity_diagnostics(pred),
            title=f"Phase {phase + 1} parity (training set)",
            save_path=args.plot_dir / f"{offset + 2:02d}_parity_phase{phase + 1}.png",
        )
        _k_vs_ph_reveal_plot(
            predictors_p1=models[0].predictors,
            predictors_p2=None if models[1] is None else models[1].predictors,
            train_experiments=train_experiments,
            val_experiments=val_experiments,
            title=f"log10 k(pH) after phase {phase + 1}",
            save_path=args.plot_dir / f"{offset + 3:02d}_k_reveal_phase{phase + 1}.png",
        )
    _loss_curve_plot(history_p1, history_p2, args.plot_dir / "07_loss_curve.png")
    print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
