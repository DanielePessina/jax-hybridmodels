"""Batch reactor: per-pH Arrhenius fits against one hybrid model, scored out of sample.

The classical way to handle a rate that depends on pH with no known form is
to fit Arrhenius separately at each pH. That works inside each pH bin, but
it gives a table of constants with no way to predict between bins. The
hybrid model of ``train_hybrid.py`` keeps the Arrhenius trunk and adds a
residual network in ``(T, pH)``, trained on every bin at once. This script
fits both to twelve ramp runs at pH 5.0, 5.85 and 7.0 and scores each by
predicting runs it never trained on.

Shows:

- two ``HybridModel`` objects sharing one predictor tuple and differing
  only in ``simulate_fn``
- ``fit(freeze=...)`` to fit the trunk alone or the residual alone
- repeated ``fit`` and ``predict`` calls on lists of experiments for
  cross-validation, and ``ExperimentPrediction.observed`` /
  ``observation_mask`` for scoring

Run::

    uv run python examples/batch_reactor/compare_baselines.py
    uv run python examples/batch_reactor/compare_baselines.py --generations 5 --steps 20

Prints the per-bin parameters and a table of out-of-sample R^2 for six
tests, and writes three figures to ``figures/``. About 54 fits; a few
minutes on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from scipy.stats import qmc

import jaxhybridmodels as hm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import (  # noqa: E402
    EA_TRUE,
    OUTPUT_CHANNELS,
    T_C_RANGE,
    T_MAX,
    k_true,
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
    simulate_trunk_fn,
    state_to_output,
)
from _shared import apply_default_style  # noqa: E402

# Below the saturation knee, at it, and above it. pH is fixed per bin
# rather than sampled, which is what lets the pH-blind baseline be fitted
# bin by bin on the same data the hybrid sees.
PH_BINS: tuple[float, ...] = (5.0, 5.85, 7.0)
N_PER_BIN: int = 4
HELD_OUT_BIN: int = 1  # pH 5.85, the knee, for the leave-one-pH-out test
BIN_COLORS: tuple[str, ...] = ("C0", "C1", "C3")


def _build_runs(noise_key: Array) -> tuple[list[hm.Experiment], list[int]]:
    """Four ramp runs per pH bin, midpoints from a per-bin Latin hypercube in T."""
    experiments, bins = [], []
    for b, pH in enumerate(PH_BINS):
        unit = qmc.LatinHypercube(d=1, seed=b).random(n=N_PER_BIN)[:, 0]
        for j, T_C in enumerate(T_C_RANGE[0] + (T_C_RANGE[1] - T_C_RANGE[0]) * unit):
            i = len(experiments)
            experiments.append(
                make_ramp_experiment(
                    T_C=float(T_C),
                    pH=pH,
                    noise_key=jr.fold_in(noise_key, i),
                    exp_id=f"pH{pH:.2f}_{j}_T{float(T_C):.1f}",
                )
            )
            bins.append(b)
    return experiments, bins


def _pairs(result: hm.PredictionResult) -> tuple[np.ndarray, np.ndarray]:
    """Observed and predicted ``Ca`` at every measured cell, as flat arrays."""
    obs, pred = [], []
    for exp_pred in result.values():
        mask = np.asarray(exp_pred.observation_mask["Ca"], dtype=bool)
        obs.append(np.asarray(exp_pred.observed["Ca"])[mask])
        pred.append(np.asarray(exp_pred.outputs["Ca"])[mask])
    return np.concatenate(obs), np.concatenate(pred)


def _r2_rmse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    resid = pred - obs
    r2 = 1.0 - float(np.sum(resid**2)) / float(np.sum((obs - obs.mean()) ** 2))
    return r2, float(np.sqrt(np.mean(resid**2)))


def _k_reveal_plot(
    baseline: hm.HybridModel,
    hybrid: hm.HybridModel,
    experiments: list[hm.Experiment],
    bins: list[int],
    path: Path,
) -> None:
    """``log10 k(pH)`` at three temperatures from models trained without pH 5.85.

    The joint Arrhenius fit is flat in pH by construction. The hybrid has
    to bridge the held-out bin from its two neighbours.
    """
    pH_grid = jnp.linspace(4.5, 7.5, 200)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for T_C, color in zip((15.0, 25.0, 35.0), ("C0", "C1", "C2"), strict=True):
        middle = T_C == 25.0
        ax.plot(
            pH_grid,
            np.log10(np.asarray(k_true(T_C, pH_grid))),
            color=color,
            lw=1.6,
            label=f"truth, {T_C:.0f}°C",
        )
        trunk = float(log10_k_trunk(baseline.predictors[0], T_C))
        ax.axhline(
            trunk, color=color, lw=1.1, ls="--", label="Arrhenius, pH 5+7" if middle else None
        )
        ax.plot(
            pH_grid,
            log10_k(hybrid.predictors, T_C, pH_grid),
            color=color,
            lw=1.4,
            ls=":",
            label="hybrid, pH 5+7" if middle else None,
        )
    ax.axvspan(
        PH_BINS[HELD_OUT_BIN] - 0.05,
        PH_BINS[HELD_OUT_BIN] + 0.05,
        color="grey",
        alpha=0.2,
        label="held-out pH",
    )
    T_mid = np.array([ramp_midpoint(e) for e in experiments])
    pH_obs = np.array([PH_BINS[b] for b in bins])
    ax.scatter(
        pH_obs,
        np.log10(np.asarray(k_true(T_mid, pH_obs))),
        s=36,
        color="black",
        zorder=4,
        label="runs (truth at ramp midpoint)",
    )
    ax.set_xlabel("pH")
    ax.set_ylabel("log10 k")
    ax.set_title("Rate law learned without the pH 5.85 runs")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parity_plot(panels: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    """Out-of-sample parity, three panels per row, with R^2 in each title."""
    nrows = (len(panels) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(10.5, 3.6 * nrows), sharex=True, sharey=True)
    for ax, (title, obs, pred) in zip(axes.flat, panels, strict=True):
        r2, _ = _r2_rmse(obs, pred)
        ax.scatter(obs, pred, s=14, alpha=0.7)
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8)
        ax.set_title(f"{title}\nR² = {r2:.3f}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("observed Ca")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted Ca (never trained on)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _held_out_plot(
    experiments: list[hm.Experiment],
    predictions: dict[str, hm.PredictionResult],
    path: Path,
) -> None:
    """The held-out pH 5.85 runs against the baseline and the hybrid."""
    fig, axes = plt.subplots(
        1, len(experiments), figsize=(3.2 * len(experiments), 3.4), sharey=True
    )
    ts_dense = jnp.linspace(0.0, T_MAX, 200)
    for ax, exp in zip(axes, experiments, strict=True):
        pH = float(exp.covariates["pH"])
        ax.plot(
            ts_dense,
            true_ca(ts_dense, T_C=ramp_midpoint(exp), pH=pH),
            color="black",
            lw=1.4,
            label="truth",
        )
        ax.scatter(
            exp.channels["Ca"].ts, exp.channels["Ca"].values, s=16, zorder=3, label="observed"
        )
        for (name, result), style in zip(predictions.items(), ("--", ":"), strict=True):
            pred = result[exp.exp_id]
            ax.plot(pred.times, pred.outputs["Ca"], ls=style, lw=1.6, label=name)
        ax.set_title(f"T ramp around {ramp_midpoint(exp):.1f}°C", fontsize=9)
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Ca")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("pH 5.85 runs, predicted by models trained on pH 5.0 and 7.0 only", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=30, help="CMA-ES generations per fit")
    parser.add_argument("--steps", type=int, default=200, help="AdamW steps per residual fit")
    parser.add_argument("--seed", type=int, default=0, help="JAX root key")
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    apply_default_style()
    k_noise, k_init = jr.split(jr.PRNGKey(args.seed))

    print("[data] four ramp runs at each of three pH values")
    experiments, bins = _build_runs(k_noise)
    in_bin = [
        [e for e, b in zip(experiments, bins, strict=True) if b == k] for k in range(len(PH_BINS))
    ]
    for exp in experiments:
        print(f"  {exp.exp_id}")

    # One predictor tuple, two simulators. The baseline ignores the
    # residual, so the same freeze choices serve both models below.
    # region models
    predictors = build_predictors(key=k_init)
    hybrid, baseline = (
        hm.HybridModel(
            predictors=predictors,
            simulate_fn=simulate,
            state_to_output=state_to_output,
            output_channel_names=OUTPUT_CHANNELS,
            solver=SOLVER,
        )
        for simulate in (simulate_fn, simulate_trunk_fn)
    )
    # endregion models
    cma_es = hm.EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        steps=args.generations,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        verbose=False,
    )
    adamw = hm.OptaxTrainingConfig(steps=args.steps, lr=3e-3, verbose=False)

    def fit_baseline(runs: list[hm.Experiment], seed: int) -> hm.HybridModel:
        return baseline.fit(runs, cma_es, key=jr.PRNGKey(seed), freeze=hm.BoundedPredictor).model

    def fit_hybrid(runs: list[hm.Experiment], seed: int) -> hm.HybridModel:
        """CMA-ES on the trunk, then AdamW on the residual with the trunk frozen."""
        trunk = hybrid.fit(runs, cma_es, key=jr.PRNGKey(seed), freeze=hm.BoundedPredictor).model
        return trunk.fit(runs, adamw, key=jr.PRNGKey(seed + 1), freeze=ArrheniusKinetics).model

    print("\n[fit] Arrhenius, one fit per pH bin")
    # The trunk has nowhere to put pH, so each refit absorbs it into the
    # prefactor: log k_ref tracks ln k_sat(pH) while Ea stays near the truth.
    per_bin = [fit_baseline(runs, seed=b) for b, runs in enumerate(in_bin)]
    print(f"  {'pH':>5}{'log k_ref':>11}{'truth':>8}{'Ea':>8}{'truth':>8}")
    for pH, model in zip(PH_BINS, per_bin, strict=True):
        log_k_ref, Ea = (float(v) for v in model.predictors[0]())
        truth = float(jnp.log(k_true(25.0, pH)))
        print(f"  {pH:>5.2f}{log_k_ref:>11.3f}{truth:>8.3f}{Ea:>8.1f}{EA_TRUE:>8.1f}")

    print("\n[fit] cross-validation")
    print("  Arrhenius: leave one run out within each bin (12 fits)")
    obs, pred = [], []
    for b, runs in enumerate(in_bin):
        for k, held in enumerate(runs):
            model = fit_baseline([e for e in runs if e is not held], seed=100 * b + k)
            o, p = _pairs(model.predict([held]))
            obs.append(o)
            pred.append(p)
    baseline_loo = (np.concatenate(obs), np.concatenate(pred))

    print("  Arrhenius: one fit across every pH, leave one run out (12 fits)")
    obs, pred = [], []
    for k, held in enumerate(experiments):
        model = fit_baseline([e for e in experiments if e is not held], seed=500 + k)
        o, p = _pairs(model.predict([held]))
        obs.append(o)
        pred.append(p)
    baseline_pooled_loo = (np.concatenate(obs), np.concatenate(pred))

    # Held-out pH: train without the pH 5.85 runs and predict them. The
    # per-bin baseline has no model for that pH, so it borrows the nearest
    # bin's fit; the pooled baseline is refitted on the two outer bins.
    held_runs = in_bin[HELD_OUT_BIN]
    outer = [e for e, b in zip(experiments, bins, strict=True) if b != HELD_OUT_BIN]
    print("  Arrhenius: pH 5.0 fit (nearest bin) predicting pH 5.85")
    baseline_nearest = _pairs(per_bin[0].predict(held_runs))
    print("  Arrhenius: one fit on pH 5.0 and 7.0, predicting pH 5.85")
    baseline_joint = fit_baseline(outer, seed=7777)
    baseline_lopo_pred = baseline_joint.predict(held_runs)
    baseline_lopo = _pairs(baseline_lopo_pred)

    print("  hybrid: leave one run out over all twelve (24 fits)")
    obs, pred = [], []
    for k, held in enumerate(experiments):
        model = fit_hybrid([e for e in experiments if e is not held], seed=1000 + 2 * k)
        o, p = _pairs(model.predict([held]))
        obs.append(o)
        pred.append(p)
    hybrid_loo = (np.concatenate(obs), np.concatenate(pred))

    print("  hybrid: trained on pH 5.0 and 7.0, predicting pH 5.85")
    hybrid_lopo_model = fit_hybrid(outer, seed=8888)
    hybrid_lopo_pred = hybrid_lopo_model.predict(held_runs)
    hybrid_lopo = _pairs(hybrid_lopo_pred)

    tests = [
        ("Arrhenius per bin", "leave one run out, same pH", baseline_loo),
        ("Arrhenius, all pH", "leave one run out", baseline_pooled_loo),
        ("hybrid", "leave one run out", hybrid_loo),
        ("Arrhenius, pH 5.0", "held-out pH 5.85", baseline_nearest),
        ("Arrhenius, pH 5+7", "held-out pH 5.85", baseline_lopo),
        ("hybrid, pH 5+7", "held-out pH 5.85", hybrid_lopo),
    ]
    print("\n[results] out-of-sample Ca")
    print(f"  {'model':<20}{'test':<28}{'R^2':>8}{'RMSE':>9}")
    for name, target, (o, p) in tests:
        r2, rmse = _r2_rmse(o, p)
        print(f"  {name:<20}{target:<28}{r2:>8.3f}{rmse:>9.4f}")

    if args.no_plot:
        return
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    _parity_plot(
        [(f"{name}:\n{target}", o, p) for name, target, (o, p) in tests],
        args.plot_dir / "compare_parity.png",
    )
    _held_out_plot(
        held_runs,
        {"Arrhenius, pH 5+7": baseline_lopo_pred, "hybrid, pH 5+7": hybrid_lopo_pred},
        args.plot_dir / "compare_held_out_ph.png",
    )
    _k_reveal_plot(
        baseline_joint, hybrid_lopo_model, experiments, bins, args.plot_dir / "compare_k_reveal.png"
    )
    print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
