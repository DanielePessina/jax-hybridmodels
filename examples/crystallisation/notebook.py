r"""Crystallisation: a hybrid and a mechanistic model, side by side.

Fits two kinetic models of the same batch crystallisation to the same four
experiments and compares them. The model pieces come from the two scripts
next to this file; this notebook trains both with smaller budgets and
plots them together.

Run: ``uv run python examples/crystallisation/notebook.py``

Writes ``raw_data.png``, ``loss_curves.png``, ``parity.png`` and
``trajectories.png`` to ``examples/crystallisation/figures/``.

# Crystallisation: a hybrid and a mechanistic model

*Crystallisation* is a dissolved solute leaving solution as a solid. It is
driven by *nucleation*, new crystals appearing, and *growth*, existing
ones getting larger. Neither rate can be derived from first principles for
a given system.

Both models share a six-state population balance in the *method of
moments*. It tracks the crystal size distribution by its moments
$\mu_k(t)$, the integral of $L^k n(L,t)$ over size $L$, plus the solute
concentration:

$$
\begin{aligned}
\frac{d\mu_0}{dt} &= J(t) \\
\frac{d\mu_k}{dt} &= k\, G(t)\, \mu_{k-1}, \quad k = 1, 2, 3, 4 \\
\frac{d\,\text{conc}}{dt} &= -3\, K_v\, \rho_c\, G(t)\, \mu_2
\end{aligned}
$$

$G$ is the growth velocity in m/s and $J$ the nucleation rate in crystals
per m³ per second. They are the unknowns:

1. **Hybrid.** Two small MLPs map temperature and supersaturation to
   $\log_{10} G$ and $\log_{10} J$, each in a ``BoundedPredictor`` that
   keeps the output in a plausible range. Trained with Adam.
2. **Mechanistic.** Classical nucleation theory for $J$ and a power law
   for $G$: four bounded ``Parameter`` scalars, fitted with CMA-ES.

Two quantities are observed: solute concentration, sampled often, and the
volume-weighted mean diameter $d_{43} = (\mu_4 / \mu_3) \cdot 10^{6}$ µm,
measured once at the end of each experiment.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

# train_kinetic enables float64 on import, before any array is made.
import train_crystallisation_mechanistic as mech
import train_kinetic as kin

import jaxhybridmodels as hm

FIGURES = Path(__file__).resolve().parent / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ## The dataset
    # ------------------------------------------------------------------
    # Four experiments, two at each of 17 °C and 21 °C. Each records
    # concentration over time and one $d_{43}$ at the end: concentration
    # comes from an in-line probe, particle size from offline sizing of
    # the final slurry. Concentration gets a uniform variance of 0.1;
    # each $d_{43}$ carries the sizing instrument's own variance.
    #
    # The experiments do not share a time grid. `make_dataset` merges each
    # experiment's sampling times and groups experiments whose merged grid
    # has the same length into *buckets*. The terminal $d_{43}$ time is
    # already a concentration time in every experiment, so the 17 °C pair
    # forms a bucket of length 9 and the 21 °C pair one of length 7. Each
    # bucket compiles once and is reused for every training step.
    dataset = kin.make_dataset()
    print(hm.describe_buckets(dataset))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    color_by_T = {17.0: "tab:blue", 21.0: "tab:orange"}
    for d in kin.EXPERIMENTS_DATA:
        ax.plot(
            d["time_min"],
            d["conc"],
            "o-",
            color=color_by_T[d["temperature_C"]],
            label=f"{d['exp_id']} ({d['temperature_C']:.0f} °C, d43 = {d['d43']:.1f} µm)",
            lw=1.2,
            ms=4,
        )
    ax.set_xlabel("time (min)")
    ax.set_ylabel("concentration")
    ax.set_title("Concentration, two replicates per temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "raw_data.png")
    plt.close(fig)

    # ## The shared model pieces
    # ------------------------------------------------------------------
    # `y0_fn` starts each experiment with no crystals and the first
    # observed concentration. `state_to_output` maps the six states to the
    # two observed channels. Its $d_{43}$ division replaces $\mu_3$ by a
    # safe value *before* dividing: guarding only the result still
    # computes $\mu_4 / 0$ on the discarded branch, and that `nan` reaches
    # the gradient.
    #
    # `integrate_minutes` solves in seconds while the data are in minutes,
    # with one absolute tolerance per state. The moments span about 18
    # decades, so a single `atol` would over-resolve some states and
    # under-resolve others.

    # # Part 1: the hybrid model
    # ------------------------------------------------------------------
    # Each rate is a `BoundedPredictor`: it reads `temperature_C` and
    # `supersaturation` by name, scales them into the network's latent
    # space, and squashes the output into a log10 range. *Supersaturation*
    # $S = c / c_{\text{sat}}(T)$ is computed from the state at every
    # solver step.
    #
    # The output bounds set the starting point. An untrained network
    # outputs roughly the midpoint of its box, so growth bounded to
    # $10^{-15}$ to $10^{-5}$ m/s starts near $10^{-10}$ m/s, plausible for
    # early growth and easy to integrate. Looser bounds start decades
    # higher and make the ODE stiff from the first step.
    #
    # Both rates are multiplied by a metastable mask, `S > 1 + 1e-5`, so
    # nothing nucleates or grows in an undersaturated solution.
    def mlp(key):
        return hm.MLPPredictor(
            in_size=2, out_size=1, width_size=64, depth=1, activation_name="relu", key=key
        )

    hybrid = hm.HybridModel(
        predictors=kin.rate_predictors(mlp, jr.key(0)),
        simulate_fn=kin.simulate_fn,
        state_to_output=kin.state_to_output,
        output_channel_names=kin.OUTPUT_CHANNELS,
        solver=kin.SOLVER,
    )
    hybrid_fit = hybrid.fit(
        dataset, hm.OptaxTrainingConfig(steps=300, lr=1e-3, verbose=False), key=jr.key(1)
    )
    print(f"hybrid best loss: {min(hybrid_fit.history.loss):.6f}")

    # # Part 2: the mechanistic model
    # ------------------------------------------------------------------
    # Classical nucleation theory gives $J$ from supersaturation,
    # temperature and the interfacial energy $\gamma$; growth is a power
    # law in $S - 1$:
    #
    # $$ J = \exp(\log A) \cdot S \cdot
    # \exp\!\left(-\frac{16\pi\,\gamma^3 v^2}{3 (k_B T)^3 \ln^2 S}\right)
    # \qquad G = \frac{10^{A_g}}{60} \cdot \max(S - 1,\ 0)^{g} $$
    #
    # | name    | meaning                               | bounds        |
    # |---------|---------------------------------------|---------------|
    # | `logA`  | $\ln A$, CNT pre-exponential factor   | (20.0, 65.0)  |
    # | `gamma` | interfacial energy [mJ/m²]            | (0.15, 1.0)   |
    # | `Ag`    | $\log_{10}$ growth pre-factor [m/min] | (-20.0, -5.0) |
    # | `g`     | growth exponent                       | (1.0, 3.5)    |
    #
    # Each is a `Parameter` starting at the midpoint of its bounds. The
    # CNT logarithm uses $\max(S, 1 + 10^{-12})$: at $S \le 1$ the mask
    # zeros the rate, but an unclipped $\ln S$ would still put `nan` in
    # the gradient.
    #
    # CMA-ES needs no gradients. It keeps a population of candidates and
    # moves its sampling distribution towards the good ones, which suits
    # four bounded scalars on a surface with local minima. Generation 0 is
    # a Latin hypercube over the latent box, and each generation is one
    # vmapped, compiled evaluation of the whole population.
    mechanistic = hm.HybridModel(
        predictors={
            name: hm.Parameter((low + high) / 2, bounds=(low, high))
            for name, (low, high) in mech.PARAMETER_BOUNDS.items()
        },
        simulate_fn=mech.simulate_fn,
        state_to_output=kin.state_to_output,
        output_channel_names=kin.OUTPUT_CHANNELS,
        solver=mech.SOLVER,
    )
    mech_config = hm.EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=32,
        num_generations=30,
        init="lhs_box",
        init_box_extent=2.0,
        sigma_init=0.5,
        verbose=False,
    )
    mech_fit = mechanistic.fit(dataset, mech_config, key=jr.key(2))
    print(f"mechanistic best loss: {min(mech_fit.history.best_fitness):.6f}")
    mech.print_constants("fitted constants", mech_fit.model.predictors)

    fig, (ax_h, ax_m) = plt.subplots(1, 2, figsize=(9, 3))
    ax_h.plot(hybrid_fit.history.loss, color="tab:blue")
    ax_h.set(xlabel="step", ylabel="loss", yscale="log", title="Hybrid (Adam)")
    ax_m.plot(mech_fit.history.best_fitness, color="tab:purple")
    ax_m.set(xlabel="generation", ylabel="best loss", yscale="log", title="Mechanistic (CMA-ES)")
    fig.tight_layout()
    fig.savefig(FIGURES / "loss_curves.png")
    plt.close(fig)

    # # Part 3: comparison
    # ------------------------------------------------------------------
    # Same data, ODE, observation model and diagnostics; only the rate
    # laws and the optimiser differ.
    #
    # | Aspect     | Hybrid                             | Mechanistic                      |
    # |------------|------------------------------------|----------------------------------|
    # | Predictors | two `BoundedPredictor` MLPs        | four `Parameter` scalars         |
    # | Inputs     | `temperature_C`, `supersaturation` | none                             |
    # | Rate laws  | learned $\log_{10} G$, $\log_{10} J$ | CNT nucleation, power-law growth |
    # | Config     | `OptaxTrainingConfig`              | `EvosaxTrainingConfig`           |
    fits = {"hybrid": hybrid_fit.model, "mechanistic": mech_fit.model}
    predictions = {label: model.predict(dataset) for label, model in fits.items()}
    for label, preds in predictions.items():
        hm.print_metrics(hm.compute_metrics(preds.buckets, dataset), header=label)

    # ### Parity
    # Predicted against observed for each channel; points on the dashed
    # line are perfect. `model.predict` returns, per experiment, the
    # predicted `outputs` and the `observed` values (`nan` where a channel
    # was not measured), keyed by channel name.
    styles = {
        "hybrid": {"color": "tab:blue", "marker": "o"},
        "mechanistic": {"color": "tab:purple", "marker": "^"},
    }
    channels = dataset.output_channel_names
    fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 4))
    for ax, name in zip(axes, channels, strict=True):
        for label, preds in predictions.items():
            observed = np.concatenate([np.asarray(p.observed[name]) for p in preds.values()])
            predicted = np.concatenate([np.asarray(p.outputs[name]) for p in preds.values()])
            seen = np.isfinite(observed)
            ax.scatter(
                observed[seen], predicted[seen], s=22, alpha=0.75, label=label, **styles[label]
            )
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8)
        ax.set(xlabel="observed", ylabel="predicted", title=name)
        ax.legend(fontsize=8)
    fig.suptitle("Parity, hybrid vs mechanistic")
    fig.tight_layout()
    fig.savefig(FIGURES / "parity.png")
    plt.close(fig)

    # ### Trajectories
    # `model.simulate` runs a model at any conditions without a dataset.
    # Here it re-runs each experiment on a 200-point grid, so the curves
    # are smooth between the sparse measurements.
    data_by_id = {d["exp_id"]: d for d in kin.EXPERIMENTS_DATA}
    exp_ids = list(predictions["hybrid"])
    fig, axes = plt.subplots(
        len(exp_ids), len(channels), figsize=(4 * len(channels), 2.4 * len(exp_ids)), squeeze=False
    )
    for row, exp_id in enumerate(exp_ids):
        data = data_by_id[exp_id]
        observed = predictions["hybrid"][exp_id]
        y0 = jnp.concatenate([jnp.zeros(5), jnp.asarray([data["conc"][0]])])
        dense = jnp.linspace(0.0, float(observed.times[-1]), 200)
        for label, model in fits.items():
            curve = model.simulate(
                ts=dense, y0=y0, covariates={"temperature_C": data["temperature_C"]}
            )
            for col, name in enumerate(channels):
                axes[row][col].plot(
                    dense, curve.outputs[name], lw=1.5, label=label, color=styles[label]["color"]
                )
        for col, name in enumerate(channels):
            ax = axes[row][col]
            ax.scatter(observed.times, observed.observed[name], s=22, color="black", zorder=3)
            ax.set_title(f"{exp_id}: {name}", fontsize=9)
            if row == len(exp_ids) - 1:
                ax.set_xlabel("t (min)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Trajectories, hybrid vs mechanistic")
    fig.tight_layout()
    fig.savefig(FIGURES / "trajectories.png")
    plt.close(fig)

    # ## Reading the fitted rates
    # The hybrid rates can be evaluated at any $(T, S)$ inside their
    # bounds, including conditions no experiment visited. The
    # mechanistic model is its four constants.
    reference = {"temperature_C": 20.0, "supersaturation": 1.5}
    log10_G = hm.evaluate_predictor(hybrid_fit.model.predictors["growth"], reference)
    log10_J = hm.evaluate_predictor(hybrid_fit.model.predictors["nucleation"], reference)
    print("hybrid at T=20 °C, S=1.5:")
    print(f"  G = {10**log10_G:.3e} m/s")
    print(f"  J = {10**log10_J:.3e} #/(m³·s)")


if __name__ == "__main__":
    main()
