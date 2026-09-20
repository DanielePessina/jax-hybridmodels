"""Hybrid kinetics from a real SBML file: the Fujita 2010 EGF/ERK model.

Loads a published, curated SBML model (Fujita et al., Sci. Signal. 2010 —
the EGF/ERK cascade, 9 species, 11 reactions) with a small hand-rolled
``python-libsbml`` + ``sympy`` converter (``sbml_loader.py``), then applies
the crystallisation pattern: the SBML file supplies the *structure* (known
stoichiometry and kinetics), and an MLP ``BoundedPredictor`` learns one
*unknown rate law* — how the EGF-EGFR association constant depends on the
EGF dose.

Concretely, reaction ``v1`` in the file is reversible mass-action with a
fixed forward constant::

    v1 = Cell * (EGF * EGFR * k1 - EGF_EGFR * k2)

We declare the forward constant unknown and let the network supply it as
``Vmax(dose)``, keeping the reverse constant ``k2`` from the file. The
truth is ``Vmax(dose) = 2 * dose``. Data: simulate the SBML (with ``Vmax``
set to the truth) at four EGF doses, observe the paper's scaled readouts
``pEGFR_tot``, ``pAkt_tot``, ``pS6_tot`` (the SBML's own assignment rules),
add noise. The fit then has to recover ``Vmax(dose)`` from the dose-response
time series alone — the same shape as the crystallisation example (learn
``Vmax(E)``), with the mechanistic core coming from an actual SBML file
instead of a hand-written vector field.

Two numerical lessons from building this example:

- **Keep the reverse term.** An irreversible uptake law drives EGFR to
  exactly zero and the solver grinds on the resulting kink (the EGFR
  turnover rate fighting a zero binding flux) — some parameter values then
  take >50k steps and ``diffeqsolve`` fails. The reversible law has a
  well-defined equilibrium at every parameter value.
- **Kvaerno3, not Kvaerno5.** Both are implicit, but Kvaerno5's Newton
  solve diverges at specific parameter values (a sharp threshold: Vmax
  0.5445 fails, 0.54450196 succeeds), while Kvaerno3 integrates the whole
  parameter space in 50-100 steps. jaxkineticmodel's default is Kvaerno5;
  that choice does not survive contact with this model.

Run:
    uv run python examples/sbml_hybrid/train_sbml_hybrid.py
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
from jaxhybridmodels.data import ChannelObs, make_dataset, make_experiment
from jaxhybridmodels.solver import SolverConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
)

SBML_PATH = "examples/sbml_hybrid/Fujita_SciSignal2010.xml"

DOSES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)  # EGF dose per experiment
N_TIMESTEPS = 40
T_MAX = 300.0  # pEGFR_tot peaks ~t=100, pS6_tot only after ~t=250
NOISE_STD = 0.02  # relative to each channel's range


def _solver() -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Kvaerno3(),
        rtol=1e-4,
        atol=1e-6,
        max_steps=50_000,
        dt0=1.0,  # start big; the fast transient is short, PID adapts
    )


# ---------------------------------------------------------------------------
# The hybrid model: SBML core + a neural rate parameter.
# ---------------------------------------------------------------------------


def simulate_fn(predictors, ts, covariates, y0, solver):
    """Integrate the SBML core with ``Vmax`` supplied by the network."""

    def vector_field(t, y, args):
        vmax = predictors[0]({"dose": covariates["dose"]}).reshape(())
        params, bc, ar = args
        params = {**params, "Vmax": vmax}
        return model.vector_field(t, y, (params, bc, ar))

    params = {**model.params, "EGF": covariates["dose"]}
    sol = solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0, args=(params, None, None))
    return jnp.asarray(sol.ys)


# Load the SBML and swap v1's fixed association constant for a neural one.
# The published law is reversible mass-action::
#
#     v1 = Cell * (EGF * EGFR * k1 - EGF_EGFR * k2)
#
# We treat the forward constant ``k1`` as the unknown rate law and let the
# network supply it as ``Vmax(dose)``, keeping the reverse constant ``k2``
# from the file. Keeping the reverse term matters numerically: an
# irreversible uptake law drives EGFR to exactly zero, and the solver then
# grinds on the resulting kink (the EGFR turnover rate fights a zero
# binding flux); the reversible law has a well-defined equilibrium at
# every parameter value.
model = SBMLKineticModel.from_file(SBML_PATH)
egf, egfr, e_egfr = sp.Symbol("EGF"), sp.Symbol("EGFR"), sp.Symbol("EGF_EGFR")
model.replace_flux(
    "v1_reaction_1",
    expr=sp.Symbol("Cell") * (
        sp.Symbol("Vmax") * egf * egfr - sp.Symbol("reaction_1_k2") * e_egfr
    ),
    extra_params=("Vmax",),
)


def state_to_output(state: Array) -> Array:
    """The SBML's own readouts: the scaled 'total' assignment rules."""
    return model.output_fn(state, model.params)


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()

    # One experiment per EGF dose; the paper's scaled readouts, noisy. The
    # truth is ``Vmax(dose) = 2 * dose``; the network never sees the dose's
    # role beyond the covariate it is asked to map to Vmax.
    key = jax.random.PRNGKey(0)
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)

    clean = {}
    for d in DOSES:
        params = {**model.params, "EGF": float(d), "Vmax": 2.0 * d}
        ys = model.integrate(_solver(), ts, model.y0, params)
        clean[d] = model.output_fn(ys, model.params)

    noise_std = {k: NOISE_STD * float(jnp.max(jnp.abs(v))) for k, v in clean.items()}

    experiments = []
    for i, d in enumerate(DOSES):
        noise_key = jax.random.fold_in(key, i)
        noise_key, *noise_keys = jax.random.split(noise_key, 4)
        obs = clean[d] + jnp.stack(
            [noise_std[d] * jax.random.normal(k, ts.shape) for k in noise_keys],
            axis=-1,
        )
        experiments.append(
            make_experiment(
                covariates={"dose": float(d)},
                channels={
                    name: ChannelObs(ts=ts, values=obs[:, k])
                    for k, name in enumerate(("pEGFR_tot", "pAkt_tot", "pS6_tot"))
                },
                y0_fn=lambda c, ch: model.y0,
                exp_id=f"dose_{d:g}",
            )
        )
    ds = make_dataset(experiments, output_channel_names=("pEGFR_tot", "pAkt_tot", "pS6_tot"))
    print(hm.describe_buckets(ds))

    # Model init and training derive from the same root as the noise.
    key, k_model, k_train = jax.random.split(key, 3)

    # An MLP with one bounded output: Vmax in [0.05, 1.0].
    predictor = hm.BoundedPredictor(
        input_keys=("dose",),
        in_scaler=hm.BoundScaler(bounds=((0.03, 0.5),)),
        inner=hm.MLPPredictor(
            in_size=1, out_size=1, width_size=16, depth=2,
            activation_name="tanh", key=k_model,
        ),
        out_scaler=hm.BoundScaler(bounds=((0.05, 1.0),)),
    )
    predictors = (predictor,)
    mask = hm.trainable_mask(predictors)

    history, trained = hm.train_with_optax(
        predictors,
        ds,
        hm.OptaxTrainingConfig(
            steps=(120,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            penalty_weight=(1e-2,),
        ),
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=_solver(),
        trainable=mask,
        key=k_train,
    )

    # Diagnostics on the held-in data.
    preds = hm.predict_dataset(
        trained, ds, simulate_fn=simulate_fn,
        state_to_output=state_to_output, solver=_solver(),
    )
    hm.print_metrics(
        hm.compute_metrics(preds, ds), header=f"fit (final data loss {history[-1]:.3e})"
    )

    # The recovered rate law vs truth: Vmax(dose) should be ~2 * dose.
    print("\nrecovered Vmax(dose) vs truth 2*dose:")
    for d in DOSES:
        learned = hm.evaluate_predictor(trained[0], {"dose": float(d)})
        print(f"  dose={d:4.2f}  learned Vmax={learned:6.3f}  truth={2.0 * d:6.3f}")

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        # ``compute_metrics`` keeps only the summary stats; the scatter needs
        # the raw value pairs, so re-walk the mask here.
        parity_data = parity_diagnostics(preds, ds)
        parity_plot(
            parity_data,
            title="SBML hybrid parity (trained model)",
            save_path=args.plot_dir / "parity.png",
        )

        # The recovered association law against the truth it had to find.
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        doses = np.asarray(DOSES)
        learned = np.asarray(
            [float(hm.evaluate_predictor(trained[0], {"dose": float(d)})) for d in DOSES]
        )
        truth = 2.0 * doses
        ax.plot(doses, truth, color="black", ls="--", lw=1.4, label="truth 2·dose")
        ax.plot(doses, learned, color="tab:red", marker="o", lw=1.4, label="learned Vmax(dose)")
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