"""Learn one rate law inside a published SBML model (Fujita et al. 2010, EGF/ERK).

The SBML file supplies the mechanism: 9 species, 11 reactions, and the
paper's scaled readouts ``pEGFR_tot``, ``pAkt_tot`` and ``pS6_tot``.
``sbml_loader.py`` turns it into a JAX vector field. One rate law is
replaced by a neural ``BoundedPredictor`` and learned from data.

Reaction ``v1``, EGF binding to its receptor, is reversible mass action:

    v1 = Cell * (EGF * EGFR * k1 - EGF_EGFR * k2)

The forward constant ``k1`` becomes ``Vmax(dose)``, a network of the EGF
dose. The reverse constant ``k2`` stays as published. The synthetic truth
is ``Vmax(dose) = 2 * dose``; the data are the three readouts with 2%
noise at four doses.

Two numerical choices matter here:

- Keep the reverse term. An irreversible uptake law drives EGFR to exactly
  zero, and the solver stalls on the resulting kink (over 50k steps at
  some parameter values). The reversible law has a smooth equilibrium.
- Use Kvaerno3. Kvaerno5's Newton solve diverges at isolated parameter
  values (``Vmax = 0.5445`` fails, ``0.54450196`` succeeds); Kvaerno3
  integrates the whole range in 50 to 100 steps.

Run (needs the ``examples`` extra for python-libsbml and sympy):
    uv run python examples/sbml_hybrid/train_sbml_hybrid.py [--no-plot]

Prints the dataset buckets, per-channel fit metrics, and the learned
``Vmax`` next to the truth at each dose. Writes ``parity.png`` and
``recovered_vmax.png`` to ``figures/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from jax import Array
from sbml_loader import SBMLKineticModel

import jaxhybridmodels as hm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import apply_default_style, parity_diagnostics, parity_plot  # noqa: E402

SBML_PATH = Path(__file__).resolve().parent / "Fujita_SciSignal2010.xml"
CHANNELS = ("pEGFR_tot", "pAkt_tot", "pS6_tot")
DOSES = (0.05, 0.1, 0.2, 0.4)
N_TIMESTEPS = 40
T_MAX = 300.0  # pEGFR_tot peaks near t=100, pS6_tot only after t=250
NOISE_STD = 0.02  # fraction of each channel's peak

SOLVER = hm.SolverConfig(
    solver=diffrax.Kvaerno3(),
    rtol=1e-4,
    atol=1e-6,
    max_steps=50_000,
    dt0=1.0,  # the fast transient is short; the step controller shrinks dt0 if needed
)

# Swap v1's forward constant for a parameter named Vmax, which simulate_fn
# fills from the network.
sbml = SBMLKineticModel.from_file(str(SBML_PATH))
egf, egfr, egf_egfr = sp.symbols("EGF EGFR EGF_EGFR")
sbml.replace_flux(
    "v1_reaction_1",
    expr=sp.Symbol("Cell")
    * (sp.Symbol("Vmax") * egf * egfr - sp.Symbol("reaction_1_k2") * egf_egfr),
    extra_params=("Vmax",),
)


def simulate_fn(predictors, ts, covariates, y0, solver):
    dose = covariates["dose"]
    # The dose is constant per experiment, so Vmax is evaluated once per solve.
    vmax = predictors["Vmax"]({"dose": dose})[0]
    params = {**sbml.params, "EGF": dose, "Vmax": vmax}
    return sbml.integrate(solver, ts, y0, params)


def state_to_output(state: Array) -> Array:
    """The SBML's own readouts: its state-dependent assignment rules."""
    return sbml.output_fn(state, sbml.params)


def make_data(key: Array) -> hm.Dataset:
    """Noisy readouts simulated from the SBML with the true Vmax, one experiment per dose."""
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments = []
    for dose, k in zip(DOSES, jax.random.split(key, len(DOSES)), strict=True):
        params = {**sbml.params, "EGF": dose, "Vmax": 2.0 * dose}
        clean = state_to_output(sbml.integrate(SOLVER, ts, sbml.y0, params))
        scale = NOISE_STD * jnp.max(jnp.abs(clean), axis=0)
        observed = clean + scale * jax.random.normal(k, clean.shape)
        experiments.append(
            hm.make_experiment(
                covariates={"dose": dose},
                channels={
                    name: hm.ChannelObs(ts, observed[:, i]) for i, name in enumerate(CHANNELS)
                },
                y0=sbml.y0,
                exp_id=f"dose_{dose:g}",
            )
        )
    return hm.make_dataset(experiments, output_channel_names=CHANNELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    apply_default_style()

    # One root key drives the data noise, the initialisation and training.
    k_data, k_model, k_train = jax.random.split(jax.random.key(0), 3)
    dataset = make_data(k_data)
    print(hm.describe_buckets(dataset))

    vmax = hm.BoundedPredictor(
        inner=hm.MLPPredictor(
            in_size=1, out_size=1, width_size=16, depth=2, activation_name="tanh", key=k_model
        ),
        input_keys=("dose",),
        input_bounds=(0.03, 0.5),
        output_bounds=(0.05, 1.0),
    )
    model = hm.HybridModel(
        predictors={"Vmax": vmax},
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    # A small bound penalty keeps the output squash away from saturation.
    config = hm.OptaxTrainingConfig(steps=120, lr=1e-2, penalty_weight=1e-2)
    result = model.fit(dataset, config, key=k_train)

    predictions = result.model.predict(dataset)
    hm.print_metrics(
        hm.compute_metrics(predictions.buckets, dataset),
        header=f"fit (best loss {min(result.history.loss):.3e})",
    )

    learned_vmax = result.model.predictors["Vmax"]
    print("\nrecovered Vmax(dose) vs truth 2*dose:")
    for dose in DOSES:
        learned = hm.evaluate_predictor(learned_vmax, {"dose": dose})
        print(f"  dose={dose:4.2f}  learned Vmax={learned:6.3f}  truth={2.0 * dose:6.3f}")

    if args.no_plot:
        return
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(
        parity_diagnostics(predictions.buckets, dataset),
        title="SBML hybrid parity (trained model)",
        save_path=args.plot_dir / "parity.png",
    )

    doses = np.asarray(DOSES)
    dense = np.linspace(doses[0], doses[-1], 200)
    learned_dense = jax.vmap(lambda d: learned_vmax({"dose": d})[0])(jnp.asarray(dense))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(dense, 2.0 * dense, color="black", ls="--", lw=1.4, label="truth 2·dose")
    ax.plot(dense, learned_dense, color="tab:red", lw=1.4, label="learned Vmax(dose)")
    ax.plot(doses, 2.0 * doses, "ko", ms=4, label="training doses")
    ax.set_xlabel("EGF dose")
    ax.set_ylabel("association constant  Vmax")
    ax.set_title("Recovered rate law")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot_dir / "recovered_vmax.png")
    plt.close(fig)
    print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
