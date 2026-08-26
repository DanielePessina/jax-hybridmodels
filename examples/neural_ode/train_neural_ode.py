"""Pure neural ODE on rectangular data, the Equinox example ported.

What this is
------------
The Equinox gallery's ``neural_ode`` script fits an MLP vector field to a
cubic spiral sampled on a fixed time grid. This file fits the same MLP to
the same data through ``hybridmodels``, so the two can be read side by
side. Nothing mechanistic is kept. The whole right-hand side is learned::

    dy/dt = f_theta(y)

What changes going through the framework
----------------------------------------
Three things, and they are the reason this file exists.

The MLP is wrapped in a ``BoundedPredictor``. Upstream, the network reads
raw ``y`` and writes a raw derivative. Here ``in_scaler`` normalises ``y``
into the network's latent input space and ``out_scaler`` squashes its
output back into a declared derivative box. The network never sees a
bound and never has to clamp itself, and a trained model cannot emit a
derivative outside ``DERIVATIVE_BOUNDS`` no matter what it is fed. That
is what makes the fitted object safe to hand to someone else.

Data goes through ``make_dataset`` rather than staying a ``[N, T, 2]``
array. On a rectangular grid that produces exactly one bucket with an
all-``True`` mask, which is the same array with a mask attached. The
framework has no separate code path for regular data; regular data is the
degenerate case of bucketing. ``train_hybrid_ode.py`` is the same model
machinery on data that actually needs the general case.

The truncated fit is a training phase, not a loop. Upstream ramps the
trajectory length by hand across three ``for`` loops. Here it is
``length_schedule``, one entry per phase. The one difference worth
knowing: ``length_schedule`` masks the loss to the first fraction of each
trajectory but still integrates the whole thing, where upstream shortens
the integration too. Same curriculum, slightly more work per early step.

How to run
----------
``uv run python examples/neural_ode/train_neural_ode.py``

Add ``--steps`` to change the per-phase budget, ``--no-plot`` to skip the
figures.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    predict_dataset,
)
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# ``_shared`` lives one level up, ``_data`` alongside this file. Both are
# added to the path so the script runs with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import CHANNELS, describe_buckets, rectangular_spiral  # noqa: E402
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
    trajectory_plot,
)

STATE_BOUNDS: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (-2.0, 2.0))
"""Box the network's input is normalised against. Wider than the data so a
trajectory that wanders early is still inside the exact region of
``to_latent`` rather than on its linear continuation."""

DERIVATIVE_BOUNDS: tuple[tuple[float, float], ...] = ((-4.0, 4.0), (-4.0, 4.0))
"""Box the learned derivative is squashed into. The true field peaks near
1.4 on this data, so the model has room without being able to produce a
stiff field that the solver would then spend its step budget on."""


def _state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 2"]:
    """Both coordinates are observed, so the projection is the identity."""
    return state


def build_field(key: Array) -> BoundedPredictor:
    """The learned vector field: ``[y1, y2] -> [dy1, dy2]``, bounded at both ends.

    ``softsign`` rather than ``sigmoid`` on the output. A vector field is
    expected to visit the edges of its derivative box during the early,
    badly-fitted part of training, and sigmoid's gradient underflows to
    exactly zero past a latent of about 15. Softsign decays polynomially,
    so a component that gets pinned can still come back. See
    ``hybridmodels.transforms`` for the measured decay rates.
    """
    return BoundedPredictor(
        input_keys=("y1", "y2"),
        in_scaler=BoundScaler(bounds=STATE_BOUNDS, transform="sigmoid"),
        inner=MLPPredictor(
            in_size=2, out_size=2, width_size=64, depth=2, activation_name="softplus", key=key
        ),
        out_scaler=BoundScaler(bounds=DERIVATIVE_BOUNDS, transform="softsign"),
    )


def simulate_fn(
    predictors: tuple[BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate the learned field over one experiment's timestamps.

    The predictor is called *inside* the vector field, once per solver
    step. That is the defining property of a neural ODE and the reason
    the adjoint choice matters: every call is on the tape.
    ``train_hybrid_ode.py`` shows the other placement, where a network is
    evaluated once before the solve.
    """
    (field,) = predictors

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        # Positional array path of BoundedPredictor.__call__: rank-1 of
        # length len(input_keys). Equivalent to passing
        # {"y1": y[0], "y2": y[1]} and cheaper inside a hot loop.
        return field(y)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=solver.dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=solver.stepsize_controller(),
        max_steps=solver.max_steps,
        adjoint=solver.adjoint,
    )
    return jnp.asarray(sol.ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000, help="Steps in the final phase.")
    parser.add_argument("--experiments", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-dir", type=Path, default=Path(__file__).parent / "figures")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    k_data, k_init, k_train = jr.split(jr.PRNGKey(args.seed), 3)

    print("[data] rectangular spiral, Equinox neural_ode dataset")
    experiments = rectangular_spiral(n_experiments=args.experiments, key=k_data)
    dataset = make_dataset(
        experiments, state_to_output=_state_to_output, output_channel_names=CHANNELS
    )
    print(describe_buckets(dataset))

    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=1e-6,
        max_steps=4096,
        dt0=0.1,
        # A network inside the vector field makes the forward tape the
        # memory bottleneck. RecursiveCheckpoint trades recomputation for
        # O(log n) storage; Direct, the framework default, would keep the
        # whole thing.
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    predictors = (build_field(k_init),)

    print("\n[train] three phases, growing trajectory length")
    config = OptaxTrainingConfig(
        steps=(args.steps // 2, args.steps // 2, args.steps),
        lr=(3e-3, 2e-3, 5e-4),
        optimizer=("adamw", "adamw", "adamw"),
        reset_optimiser_state=(False, False, False),
        length_schedule=(0.2, 0.5, 1.0),
        # restore_best defaults to True and is safe here: the running
        # minimum resets whenever length_schedule changes (R-T9), so the
        # returned model is the best full-length one rather than the best
        # 20%-prefix one, which would be the least-trained point in the run.
        loss="mse",
        verbose=False,
    )
    history, trained = train_with_optax(
        predictors, dataset, config, simulate_fn=simulate_fn, solver=solver, key=k_train
    )
    boundaries = [args.steps // 2, args.steps]
    print(f"  loss at phase starts: {[f'{history[i]:.4f}' for i in [0, *boundaries]]}")
    print(f"  final loss: {history[-1]:.5f}")

    predictions = predict_dataset(trained, dataset, simulate_fn=simulate_fn, solver=solver)
    diag = compute_diagnostics(predictions, dataset)
    print()
    print_diagnostics(diag)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(diag, title="Neural ODE parity", save_path=args.plot_dir / "node_parity.png")
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained,
            simulate_fn=simulate_fn,
            solver=solver,
            max_experiments=4,
            title="Neural ODE trajectories",
            save_path=args.plot_dir / "node_trajectories.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
