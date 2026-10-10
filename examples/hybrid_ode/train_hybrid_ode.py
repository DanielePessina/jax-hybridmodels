"""Hybrid ODE: keep the physics you know, learn the two parts you do not.

A damped rotation, ``dy/dt = [[-k, w], [-w, -k]] y + C y^3``, with ``w``
known. The damping rate ``k`` follows an Arrhenius law in temperature that
the model is never told, and the cubic coupling ``C y^3`` is missing from
the model entirely. One network learns ``k(T)`` from a covariate, another
learns the missing term from trajectories alone.

Shows:

- a network above the solver (``rate_net``, once per experiment) and one
  inside the vector field (``residual_net``, once per solver step)
- ``BoundScaler`` warps: ``log10`` for a rate spanning decades, and a
  custom ``symlog`` warp registered with ``register_warp``
- ``softsign`` output squash for a network that visits its bounds
- irregular per-channel sampling (three buckets) against a rectangular grid
- the saturation penalty on a ``box_grid`` over each network's input box
- ``RecursiveCheckpointAdjoint`` for a network on the solver tape

Run::

    uv run python examples/hybrid_ode/train_hybrid_ode.py
    uv run python examples/hybrid_ode/train_hybrid_ode.py --data rectangular
    uv run python examples/hybrid_ode/train_hybrid_ode.py --mechanistic-only
    uv run python examples/hybrid_ode/train_hybrid_ode.py --inner kan

Prints the bucket layout, losses, per-network saturation penalty, recovered
``k(T)`` and residual error, and per-channel metrics. Writes parity and
trajectory figures to ``figures/`` unless ``--no-plot``. About a minute on
CPU.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jax.typing import ArrayLike
from jaxtyping import Array, Float

import jaxhybridmodels as hm
from jaxhybridmodels.penalties import bound_penalty, box_grid
from jaxhybridmodels.prediction import evaluate_predictor
from jaxhybridmodels.transforms import Warp, register_warp

# ``_data`` sits alongside this file, ``_shared`` one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import (  # noqa: E402
    CHANNELS,
    COUPLING,
    OMEGA_TRUE,
    TEMPERATURES,
    build_dataset,
    irregular_experiments,
    rectangular_experiments,
    state_to_output,
    true_k,
)
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
    trajectory_plot,
)

SYMLOG_EPS: float = 0.25  # half-width of the symlog warp's linear region

# Rate-network boxes. The input is wider than the covariate levels in the
# data, so ``to_latent`` stays in its exact region at the extremes. The
# output is three decades against true rates of 0.006 to 0.28: loose on
# purpose, which under ``log10`` costs almost nothing in resolution.
TEMPERATURE_BOUNDS: tuple[tuple[float, float], ...] = ((270.0, 350.0),)
K_BOUNDS: tuple[tuple[float, float], ...] = ((1e-3, 1.0),)

# Residual-network boxes. The true coupling reaches about 0.6 here, so most
# of the output box is headroom the symlog warp deliberately under-resolves.
STATE_BOUNDS: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (-2.0, 2.0))
RESIDUAL_BOUNDS: tuple[tuple[float, float], ...] = ((-3.0, 3.0), (-3.0, 3.0))


# A signed-logarithmic axis warp under the name ``symlog``:
# ``forward(x) = sign(x) log(1 + |x| / eps)``, inverted exactly. Monotone on
# the whole line, smooth through zero, and ``forward(0) = 0``, so the box
# midpoint stays at zero and a fresh residual network starts near no
# correction rather than at an arbitrary interior point. ``register_warp``
# overwrites without warning, so prefix custom names if a collision with a
# future package warp would matter to you.
# region symlog-warp
register_warp(
    "symlog",
    Warp(
        forward=lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x) / SYMLOG_EPS),
        inverse=lambda w: jnp.sign(w) * SYMLOG_EPS * jnp.expm1(jnp.abs(w)),
        requires_positive=False,
    ),
)
# endregion symlog-warp


def _inner(kind: str, *, in_size: int, out_size: int, width: int, key: Array) -> hm.Predictor:
    """Build the trainable core of a ``BoundedPredictor``.

    The only place that knows which architecture is in use; everything
    downstream sees a ``Predictor``.
    """
    if kind == "mlp":
        return hm.MLPPredictor(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=2,
            activation_name="softplus",
            key=key,
        )
    if kind == "kan":
        return hm.KANPredictor(
            in_size=in_size, out_size=out_size, hidden_widths=(width,), grid_size=5, key=key
        )
    raise ValueError(f"--inner must be 'mlp' or 'kan'; got {kind!r}")


# region simulate-fn
def simulate_fn(
    predictors: dict[str, hm.BoundedPredictor],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: hm.SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate the hybrid field for one experiment.

    The rate network runs here once and its output is closed over as a
    constant; the residual network runs inside ``vector_field`` once per
    step. Ordinary calls either way: the library is never told which is
    which. Without a ``"residual"`` entry the model is mechanistic only.
    """
    rate_net = predictors["rate"]
    residual_net = predictors.get("residual")

    k = rate_net(covariates).reshape(())
    rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]])

    def vector_field(t: ArrayLike, y: Float[Array, " 2"], args: object) -> Array:
        mechanistic = rotation @ y
        if residual_net is None:
            return mechanistic
        return mechanistic + residual_net(y)

    term = diffrax.ODETerm(vector_field)
    return jnp.asarray(solver.diffeqsolve(term, ts, y0).ys)
    # endregion simulate-fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--experiments", type=int, default=24)
    parser.add_argument(
        "--data",
        choices=("irregular", "rectangular"),
        default="irregular",
        help="Sampling layout. Same model either way; only the bucketing differs.",
    )
    parser.add_argument("--inner", choices=("mlp", "kan"), default="mlp")
    parser.add_argument(
        "--mechanistic-only",
        action="store_true",
        help="Drop the residual network, leaving the rate network alone.",
    )
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=1e-3,
        help="Saturation penalty weight; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-dir", type=Path, default=Path(__file__).parent / "figures")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    k_data, k_rate, k_res, k_train = jr.split(jr.key(args.seed), 4)

    print(f"[data] {args.data} sampling")
    if args.data == "rectangular":
        experiments = rectangular_experiments(n_experiments=args.experiments, key=k_data)
    else:
        experiments = irregular_experiments(n_experiments=args.experiments, key=k_data)
    dataset = build_dataset(experiments)
    print(hm.describe_buckets(dataset))

    # region solver
    solver = hm.SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=1e-6,
        max_steps=4096,
        dt0=0.1,
        # A network inside the vector field makes the forward tape the
        # memory bottleneck. RecursiveCheckpoint trades recomputation for
        # O(log n) storage; Direct, the default, keeps the lot.
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    # endregion solver

    # ``temperature -> k``, evaluated once per experiment above the solver.
    # ``warp="log10"`` is what makes this read well: a linear box over
    # ``(1e-3, 1)`` has midpoint ``0.5``, putting every rate in the data in
    # the bottom 3% of the box, reachable only through large negative
    # latents. Under ``log10`` the midpoint is ``0.032`` and the true range
    # covers the middle of the box.
    # region predictors
    rate_net = hm.BoundedPredictor(
        input_keys=("temperature",),
        in_scaler=hm.BoundScaler(bounds=TEMPERATURE_BOUNDS, transform="sigmoid"),
        inner=_inner(args.inner, in_size=1, out_size=1, width=16, key=k_rate),
        out_scaler=hm.BoundScaler(bounds=K_BOUNDS, transform="sigmoid", warp="log10"),
    )
    predictors = {"rate": rate_net}
    if not args.mechanistic_only:
        # ``[y1, y2] -> correction``, evaluated once per solver step inside
        # the vector field. softsign keeps a usable gradient far past the
        # point where a sigmoid's underflows, for a term that visits the
        # edge of its box early in training.
        predictors["residual"] = hm.BoundedPredictor(
            input_keys=("y1", "y2"),
            in_scaler=hm.BoundScaler(bounds=STATE_BOUNDS, transform="sigmoid"),
            inner=_inner(args.inner, in_size=2, out_size=2, width=32, key=k_res),
            out_scaler=hm.BoundScaler(bounds=RESIDUAL_BOUNDS, transform="softsign", warp="symlog"),
        )
    # endregion predictors
    model = hm.HybridModel(
        predictors=predictors,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=CHANNELS,
        solver=solver,
    )

    # region fit
    config = hm.OptaxTrainingConfig(
        # Fit the first 40% of each trajectory, then all of it. The
        # saturation penalty applies in both phases.
        phases=(
            hm.TrainingPhase(
                steps=args.steps // 2, lr=5e-3, horizon=0.4, penalty_weight=args.penalty_weight
            ),
            hm.TrainingPhase(steps=args.steps, lr=1e-3, penalty_weight=args.penalty_weight),
        ),
        # The residual network reads the ODE state ("y1", "y2"), so the
        # dataset can resolve no measured points for it. Both leaves get a
        # grid over their input box in warped space instead (``box_grid``).
        # One array per leaf, in pytree order (dict keys sorted).
        penalty_points=tuple(box_grid(leaf.in_scaler, n_per_dim=7) for leaf in predictors.values()),
        loss="mse",
        verbose=False,
    )
    print(f"\n[fit] {list(predictors)} with {args.inner!r} inner networks, two phases")
    result = model.fit(dataset, config, key=k_train)
    # endregion fit
    trained = result.model.predictors
    loss, phase = result.history.loss, result.history.phase
    starts = [f"{loss[phase.index(p)]:.4f}" for p in sorted(set(phase))]
    print(f"  loss at phase starts: {' -> '.join(starts)}")
    print(f"  final loss: {loss[-1]:.5f}; returned step {result.best_step}")

    # Zero means no predictor is pinned against a bound anywhere in its
    # declared input box, which is what you want to ship. Non-zero says a
    # network is leaning on its bound, which is probably in the wrong place.
    print("  end-of-run saturation penalty, by network:")
    for name, leaf in trained.items():
        print(f"    {name:9s} {float(bound_penalty((leaf,), (box_grid(leaf.in_scaler),))):.4e}")

    print("\n[results] recovered rate law k(T)")
    # The check that matters for the outside-the-solver network: it sees only
    # trajectories, so agreeing with the Arrhenius law here means the
    # covariate dependence was recovered, not memorised per experiment.
    print("  temperature   true k     fitted k    ratio")
    for temperature in TEMPERATURES:
        truth = float(true_k(temperature))
        fitted = evaluate_predictor(trained["rate"], {"temperature": temperature})
        print(f"    {temperature:6.1f}    {truth:.5f}    {fitted:.5f}    {fitted / truth:5.2f}")
    if "residual" in trained:
        print("\n[results] recovered cubic coupling")
        # Relative RMS over the grid rather than pointwise, since the residual
        # is only identifiable where trajectories actually went.
        axis = jnp.linspace(-0.8, 1.0, 9)
        grid = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        truth = (COUPLING @ (grid**3).T).T
        fitted = jnp.stack([trained["residual"](point) for point in grid])
        rms_error = float(jnp.sqrt(jnp.mean((fitted - truth) ** 2)))
        rms_truth = float(jnp.sqrt(jnp.mean(truth**2)))
        print(f"  residual RMS error {rms_error:.4f} against a true RMS of {rms_truth:.4f}")
        print(f"  relative {rms_error / rms_truth:.2%}")

    predictions = result.model.predict(dataset).buckets
    print()
    hm.print_metrics(hm.compute_metrics(predictions, dataset))

    if not args.no_plot:
        variant = "mechanistic" if args.mechanistic_only else args.inner
        suffix = f"{args.data}_{variant}"
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            parity_diagnostics(predictions, dataset),
            title=f"Hybrid ODE parity ({suffix})",
            save_path=args.plot_dir / f"parity_{suffix}.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=solver,
            max_experiments=4,
            title=f"Hybrid ODE trajectories ({suffix})",
            save_path=args.plot_dir / f"trajectories_{suffix}.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
