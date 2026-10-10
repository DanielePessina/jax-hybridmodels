"""Learn a nucleation rate law with a ``NeuralNPolynomial`` predictor.

Two moments of a crystal size distribution, ``d mu0/dt = J`` and
``d mu1/dt = G * mu0``, with growth ``G`` known and the nucleation rate
``J(S)`` learned. ``J`` is a polynomial in a latent coordinate of
``S - 1`` with exponents 2, 3 and 4, whose coefficients come from a small
MLP. Only ``mu0`` is measured. The synthetic truth is
``J = 0.02 + 0.15 (S - 1)^4`` at five supersaturations.

Shows:

- ``NeuralNPolynomial`` inside a ``BoundedPredictor``
- a ``"log"`` output warp for a rate spanning two decades
- a predictor evaluated once per solve from a constant covariate

Run::

    uv run python examples/supersaturation_poly/train_supersaturation_poly.py
    uv run python examples/supersaturation_poly/train_supersaturation_poly.py --steps 5 --no-plot

Prints the buckets, metrics and the learned rate at each supersaturation;
writes ``parity.png`` and ``recovered_rate.png`` to ``figures/``. About a
minute on CPU.
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
from jax import Array

import jaxhybridmodels as hm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import apply_default_style, parity_diagnostics, parity_plot  # noqa: E402

G_TRUE = 0.1
SUPERSATURATIONS = (0.3, 0.6, 0.9, 1.2, 1.5)  # S - 1
T_MAX = 8.0
N_TIMESTEPS = 40
NOISE_STD = 0.01
J_BOUNDS = (0.01, 2.0)
EXPONENTS = (2.0, 3.0, 4.0)

SOLVER = hm.SolverConfig(solver=diffrax.Tsit5(), rtol=1e-7, atol=1e-9, max_steps=4096, dt0=0.05)


def true_rate(s_minus_one):
    """The rate law the fit has to recover."""
    return 0.02 + 0.15 * s_minus_one**4


def moments(j_rate: Array | float, ts: Array, y0: Array, solver: hm.SolverConfig):
    """Integrate the two-moment ODE at a constant nucleation rate."""

    def vector_field(t, y, args):
        return jnp.stack([j_rate, G_TRUE * y[0]])

    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys


# region simulate-fn
def simulate_fn(predictors, ts, covariates, y0, solver):
    # S is constant within an experiment, so J is evaluated once per solve
    # instead of at every solver step.
    j_rate = predictors["J"]({"super_sat": covariates["super_sat"]})[0]
    return moments(j_rate, ts, y0, solver)


# endregion simulate-fn


def state_to_output(state: Array) -> Array:
    return state[..., :1]  # mu0


def make_data(key: Array) -> hm.Dataset:
    """Noisy mu0 series from the true rate law, one experiment per supersaturation."""
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    y0 = jnp.zeros(2)
    experiments = []
    for s_minus_one, k in zip(
        SUPERSATURATIONS, jax.random.split(key, len(SUPERSATURATIONS)), strict=True
    ):
        mu0 = moments(true_rate(s_minus_one), ts, y0, SOLVER)[:, 0]
        experiments.append(
            hm.make_experiment(
                covariates={"super_sat": s_minus_one},
                channels={
                    "mu0": hm.ChannelObs(ts, mu0 + NOISE_STD * jax.random.normal(k, ts.shape))
                },
                y0=y0,
                exp_id=f"S_{s_minus_one:g}",
            )
        )
    return hm.make_dataset(experiments, output_channel_names=("mu0",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    apply_default_style()

    # One root key drives the data noise, the initialisation and training.
    k_data, k_model, k_train = jax.random.split(jax.random.key(0), 3)
    dataset = make_data(k_data)
    print("[data]")
    print(hm.describe_buckets(dataset))

    # region predictor
    # The coefficient network outputs one coefficient per exponent.
    coeff_net = hm.MLPPredictor(
        in_size=1,
        out_size=len(EXPONENTS),
        width_size=16,
        depth=2,
        activation_name="tanh",
        key=k_model,
    )
    rate = hm.BoundedPredictor(
        inner=hm.NeuralNPolynomial(coeff_net=coeff_net, exponents=EXPONENTS, in_size=1, out_size=1),
        input_keys=("super_sat",),
        input_bounds=(-0.2, 2.0),
        # A log warp because J spans two decades inside its bounds.
        out_scaler=hm.BoundScaler(bounds=(J_BOUNDS,), warp="log"),
    )
    # endregion predictor
    model = hm.HybridModel(
        predictors={"J": rate},
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=("mu0",),
        solver=SOLVER,
    )
    print("\n[fit]")
    result = model.fit(dataset, hm.OptaxTrainingConfig(steps=args.steps, lr=1e-2), key=k_train)

    predictions = result.model.predict(dataset)
    print("\n[results]")
    hm.print_metrics(
        hm.compute_metrics(predictions.buckets, dataset),
        header=f"fit (best loss {min(result.history.loss):.3e})",
    )

    learned_rate = result.model.predictors["J"]
    print("\nrecovered J(S) vs truth:")
    for s_minus_one in SUPERSATURATIONS:
        learned = hm.evaluate_predictor(learned_rate, {"super_sat": s_minus_one})
        truth = true_rate(s_minus_one)
        print(f"  S-1={s_minus_one:4.1f}  J_learned={learned:.4f}  J_truth={truth:.4f}")

    if args.no_plot:
        return
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(
        parity_diagnostics(predictions.buckets, dataset),
        title="Neural polynomial kinetics parity (trained model)",
        save_path=args.plot_dir / "parity.png",
    )

    s = np.asarray(SUPERSATURATIONS)
    dense = np.linspace(s[0], s[-1], 200)
    learned_dense = jax.vmap(lambda v: learned_rate({"super_sat": v})[0])(jnp.asarray(dense))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(dense, true_rate(dense), color="black", ls="--", lw=1.4, label="truth")
    ax.plot(dense, learned_dense, color="tab:red", lw=1.4, label="learned J(S)")
    ax.plot(s, true_rate(s), "ko", ms=4, label="training conditions")
    ax.set_xlabel("supersaturation  S − 1")
    ax.set_ylabel("nucleation rate  J")
    ax.set_title("Recovered rate law")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot_dir / "recovered_rate.png")
    plt.close(fig)
    print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
