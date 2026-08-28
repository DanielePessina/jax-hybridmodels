"""Hybrid ODE fit: keep the physics you know, learn the parts you do not.

A mechanistic model with two gaps: a rate constant that varies with an
operating condition in a way you cannot write down, and a term missing
from the model entirely. The ground truth is::

    dy/dt = [[-k, w], [-w, -k]] y  +  C y^3        w = 1, k = k(temperature)

The model keeps the rotation and knows ``w``. One network learns ``k`` from
a temperature covariate, another learns ``C y^3``.

What it shows
-------------
**Two network placements.** ``rate_net`` reads a covariate that is constant
along a trajectory, so it runs once per experiment above ``diffeqsolve``
and never touches the solver tape. ``residual_net`` depends on the state,
so it runs inside the vector field, once per step (a network there is a
neural ODE, which diffrax and Equinox document). The two travel as a plain
tuple, unpacked at the top of ``simulate_fn``. The library never inspects
the container, so ``--mechanistic-only`` just shortens the tuple.

**Bounds that hold by construction.** ``BoundedPredictor`` means neither
network sees a bound or can leave one. ``k`` is bounded to ``(1e-3, 1)``
under ``warp="log10"``, since the true rates span 1.6 decades and a linear
box would put 99% of the latent range above 0.5. The residual uses
``softsign``, whose gradient decays polynomially, because a term that
visits the edge of its box needs one that does not underflow.

**Two data layouts, one model.** ``--data rectangular`` gives one bucket
with a full mask, ``--data irregular`` gives three about half full. The
model code is identical: regular data is the degenerate case, not a
separate path.

**A saturation penalty on the latent**, evaluated on a grid over each
predictor's declared input box rather than along the trajectories, so it
reports extrapolation trouble the training loss cannot see.

The residual's box straddles zero, so ``log10`` is unusable and a linear
box wastes resolution on corrections that should never happen. The script
registers a signed-logarithmic warp of its own with ``register_warp``.

Run::

    uv run python examples/hybrid_ode/train_hybrid_ode.py
    uv run python examples/hybrid_ode/train_hybrid_ode.py --data rectangular
    uv run python examples/hybrid_ode/train_hybrid_ode.py --mechanistic-only
    uv run python examples/hybrid_ode/train_hybrid_ode.py --inner kan
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    MLPPredictor,
    SolverConfig,
    Warp,
    compute_metrics,
    evaluate_predictor,
    predict_dataset,
    print_metrics,
    register_warp,
)
from hybridmodels.penalties import bound_penalty, collocation_grids
from hybridmodels.predictors.base import Predictor
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# ``_data`` sits alongside this file, ``_shared`` one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import (  # noqa: E402
    COUPLING,
    OMEGA_TRUE,
    TEMPERATURES,
    build_dataset,
    describe_buckets,
    irregular_experiments,
    rectangular_experiments,
    state_to_output,
    true_k,
)
from _shared import (  # noqa: E402
    apply_default_style,
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


def _register_symlog() -> None:
    """Register a signed-logarithmic axis warp under the name ``symlog``.

    ``forward(x) = sign(x) log(1 + |x| / eps)``, inverted exactly. Monotone
    on the whole line, smooth through zero, and ``forward(0) = 0``, so the
    box midpoint stays at zero and a fresh residual network starts near no
    correction rather than at an arbitrary interior point.

    ``register_warp`` overwrites without warning, so prefix custom names if
    a collision with a future package warp would matter to you.
    """
    register_warp(
        "symlog",
        Warp(
            forward=lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x) / SYMLOG_EPS),
            inverse=lambda w: jnp.sign(w) * SYMLOG_EPS * jnp.expm1(jnp.abs(w)),
            requires_positive=False,
        ),
    )


_register_symlog()


def _inner(kind: str, *, in_size: int, out_size: int, width: int, key: Array) -> Predictor:
    """Build the trainable core of a ``BoundedPredictor``.

    The only place that knows which architecture is in use; everything
    downstream sees a ``Predictor``.
    """
    if kind == "mlp":
        return MLPPredictor(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=2,
            activation_name="softplus",
            key=key,
        )
    if kind == "kan":
        from hybridmodels import KANPredictor

        return KANPredictor(
            in_size=in_size, out_size=out_size, hidden_widths=(width,), grid_size=5, key=key
        )
    raise ValueError(f"--inner must be 'mlp' or 'kan'; got {kind!r}")


def build_rate_net(key: Array, kind: str) -> BoundedPredictor:
    """``temperature -> k``, evaluated once per experiment above the solver.

    ``warp="log10"`` is what makes this read well. A linear box over
    ``(1e-3, 1)`` has midpoint ``0.5``, putting every rate in the data in
    the bottom 3% of the box, reachable only through large negative
    latents. Under ``log10`` the midpoint is ``0.032`` and the true range
    covers the middle of the box.
    """
    return BoundedPredictor(
        input_keys=("temperature",),
        in_scaler=BoundScaler(bounds=TEMPERATURE_BOUNDS, transform="sigmoid"),
        inner=_inner(kind, in_size=1, out_size=1, width=16, key=key),
        out_scaler=BoundScaler(bounds=K_BOUNDS, transform="sigmoid", warp="log10"),
    )


def build_residual_net(key: Array, kind: str) -> BoundedPredictor:
    """``[y1, y2] -> correction``, evaluated once per solver step inside the field."""
    return BoundedPredictor(
        input_keys=("y1", "y2"),
        in_scaler=BoundScaler(bounds=STATE_BOUNDS, transform="sigmoid"),
        inner=_inner(kind, in_size=2, out_size=2, width=32, key=key),
        out_scaler=BoundScaler(bounds=RESIDUAL_BOUNDS, transform="softsign", warp="symlog"),
    )


def simulate_fn(
    predictors: tuple[BoundedPredictor, ...],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate the hybrid field for one experiment.

    Both placements are visible in the first six lines: the rate network
    runs here once and its output is closed over as a constant, the
    residual network runs inside ``vector_field`` once per step. Ordinary
    calls either way; the library is never told which is which.
    """
    rate_net = predictors[0]
    residual_net = predictors[1] if len(predictors) > 1 else None

    k = rate_net(covariates).reshape(())
    rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]])

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        mechanistic = rotation @ y
        if residual_net is None:
            return mechanistic
        return mechanistic + residual_net(y)

    term = diffrax.ODETerm(vector_field)
    return jnp.asarray(solver.diffeqsolve(term, ts, y0).ys)


def report_rate_net(rate_net: BoundedPredictor) -> None:
    """Print recovered against true ``k`` at every temperature level in the data.

    The check that matters for the outside-the-solver network: it sees only
    trajectories, so agreeing with the Arrhenius law here means the
    covariate dependence was recovered, not memorised per experiment.
    """
    print("  temperature   true k     fitted k    ratio")
    for temperature in TEMPERATURES:
        truth = float(true_k(temperature))
        fitted = evaluate_predictor(rate_net, {"temperature": temperature})
        print(f"    {temperature:6.1f}    {truth:.5f}    {fitted:.5f}    {fitted / truth:5.2f}")


def _parity_diagnostics(predictions, dataset):
    """Masked obs/pred pairs per channel for ``parity_plot``.

    ``compute_metrics`` keeps only the summary stats; the scatter needs the
    raw value pairs, so re-walk the mask here.
    """
    metrics = compute_metrics(predictions, dataset)
    out: dict[str, SimpleNamespace] = {}
    for d, name in enumerate(dataset.output_channel_names):
        obs_chunks: list = []
        pred_chunks: list = []
        for pred, bp in zip(predictions, dataset.bucket_payloads, strict=True):
            mask = bp.mask[..., d]
            obs_chunks.append(bp.y_observed[..., d][mask])
            pred_chunks.append(pred[..., d][mask])
        obs = jnp.concatenate(obs_chunks) if obs_chunks else jnp.empty(0)
        pred = jnp.concatenate(pred_chunks) if pred_chunks else jnp.empty(0)
        m = metrics[name]
        out[name] = SimpleNamespace(
            name=name, n=m.n, obs=obs, pred=pred, r2=float(m.r2), rmse=float(m.rmse)
        )
    return out


def report_residual_net(residual_net: BoundedPredictor) -> None:
    """Compare the learned correction with ``C y^3`` on a grid inside the data range.

    Relative RMS over the grid rather than pointwise, since the residual is
    only identifiable where trajectories actually went.
    """
    axis = jnp.linspace(-0.8, 1.0, 9)
    grid = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    truth = (COUPLING @ (grid**3).T).T
    fitted = jnp.stack([residual_net(point) for point in grid])
    rms_error = float(jnp.sqrt(jnp.mean((fitted - truth) ** 2)))
    rms_truth = float(jnp.sqrt(jnp.mean(truth**2)))
    print(f"  residual RMS error {rms_error:.4f} against a true RMS of {rms_truth:.4f}")
    print(f"  relative {rms_error / rms_truth:.2%}")


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
        help="Drop the residual network. The predictors tuple becomes length one.",
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
    k_data, k_rate, k_res, k_train = jr.split(jr.PRNGKey(args.seed), 4)

    print(f"[data] {args.data} sampling")
    if args.data == "rectangular":
        experiments = rectangular_experiments(n_experiments=args.experiments, key=k_data)
    else:
        experiments = irregular_experiments(n_experiments=args.experiments, key=k_data)
    dataset = build_dataset(experiments)
    print(describe_buckets(dataset))

    solver = SolverConfig(
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

    predictors: tuple[BoundedPredictor, ...] = (build_rate_net(k_rate, args.inner),)
    if not args.mechanistic_only:
        predictors = (*predictors, build_residual_net(k_res, args.inner))
    print(f"\n[model] {len(predictors)} predictor(s), inner architecture {args.inner!r}")

    config = OptaxTrainingConfig(
        steps=(args.steps // 2, args.steps),
        lr=(5e-3, 1e-3),
        optimizer=("adamw", "adamw"),
        reset_optimiser_state=(False, False),
        length_schedule=(0.4, 1.0),
        # Length-1 tuple, so the weight broadcasts across both phases. A
        # two-element tuple would set it per phase.
        penalty_weight=(args.penalty_weight,),
        penalty_grid_points=7,
        loss="mse",
        verbose=False,
    )
    print("[train] two phases, saturation penalty on")
    history, trained = train_with_optax(
        predictors,
        dataset,
        config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        key=k_train,
    )
    print(f"  loss at phase starts: {history[0]:.4f} -> {history[args.steps // 2]:.4f}")
    print(f"  final loss: {history[-1]:.5f}")

    # Zero means no predictor is pinned against a bound anywhere in its
    # declared input box, which is what you want to ship. Non-zero says a
    # network is leaning on its bound, which is probably in the wrong place.
    print("  end-of-run saturation penalty, by leaf:")
    names = ("rate_net", "residual_net")
    for name, leaf in zip(names, trained, strict=False):
        print(f"    {name:13s} {float(bound_penalty((leaf,), collocation_grids((leaf,)))):.4e}")

    print("\n[rate network] recovered temperature dependence")
    report_rate_net(trained[0])
    if len(trained) > 1:
        print("\n[residual network] recovered cubic coupling")
        report_residual_net(trained[1])

    predictions = predict_dataset(
        trained, dataset, simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver
    )
    metrics = compute_metrics(predictions, dataset)
    print()
    print_metrics(metrics)

    if not args.no_plot:
        variant = "mechanistic" if args.mechanistic_only else args.inner
        suffix = f"{args.data}_{variant}"
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            _parity_diagnostics(predictions, dataset),
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
