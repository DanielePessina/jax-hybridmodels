"""Hybrid ODE on bucketed irregular data: two networks, two placements.

What this is
------------
The same spiral as ``train_neural_ode.py``, but the model keeps what is
known and learns only what is not, and the data has the shape the
framework was actually built for.

Ground truth::

    dy/dt = [[-k, w], [-w, -k]] y  +  C y^3        w = 1, k = k(temperature)

The model keeps the rotation and its frequency ``w``, which is known. It
learns the other two pieces with two separate networks, placed on
opposite sides of the solver:

    **outside the solve** -- ``rate_net``: ``temperature -> k``. One
    evaluation per experiment, hoisted above ``diffeqsolve`` because the
    covariate does not change during a trajectory. It is not on the
    solver tape at all, so its gradient path is short and its cost is
    independent of how many steps the solver takes. ``k`` is bounded to
    ``(1e-3, 1)`` with ``warp="log10"``, because the true rates span
    decades and a linear box would put 99% of the latent range above
    ``0.5``.

    **inside the solve** -- ``residual_net``: ``y -> correction``. One
    evaluation per solver step, on the tape, exactly like the pure neural
    ODE. Bounded to a small symmetric box so it can correct the
    mechanistic core without replacing it.

They travel together as a plain tuple, ``(rate_net, residual_net)``,
unpacked at the top of ``simulate_fn``. The framework never inspects the
container (ADR-0006); a dict or a NamedTuple would work identically.
``--mechanistic-only`` drops the second entry and the tuple becomes
length one, with no other change to the plumbing.

Data
----
``irregular_spiral`` gives every experiment its own end time, its own
sample times, and independently thinned channels. Union lengths then
differ across experiments, so ``make_dataset`` builds three buckets and
the mask runs about half full. Nothing is padded to a common length and
nothing is dropped. Compare the one dense bucket in
``train_neural_ode.py``.

Extensibility
-------------
Two things this file needs that the package does not ship, both added
without touching the package:

``register_warp("symlog", ...)`` -- the residual box straddles zero, so
``log10`` is unusable, but a linear box spends its resolution on large
corrections that should never happen. Symlog is linear near zero and
logarithmic in the tails, which is where a residual's prior belongs.

``penalty_weight`` -- the saturation penalty, charged on the latent
rather than the physical output, so it keeps pulling after the squash
gradient has died. See ``docs/adr/0007-collocation-bound-penalty.md``.

``--inner kan`` swaps both networks from MLPs to KANs. The rest of the
file is unchanged, which is the point of ``BoundedPredictor`` holding a
``Predictor`` rather than subclassing one.

How to run
----------
``uv run python examples/neural_ode/train_hybrid_ode.py``
``uv run python examples/neural_ode/train_hybrid_ode.py --mechanistic-only``
``uv run python examples/neural_ode/train_hybrid_ode.py --inner kan``
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
    Warp,
    make_dataset,
    predict_dataset,
    register_warp,
)
from hybridmodels.penalties import bound_penalty, collocation_grids
from hybridmodels.predictors.base import Predictor
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# ``_shared`` lives one level up, ``_data`` alongside this file. Both are
# added to the path so the script runs with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import (  # noqa: E402
    CHANNELS,
    COUPLING,
    OMEGA_TRUE,
    TEMPERATURES,
    describe_buckets,
    irregular_spiral,
    true_k,
)
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
    trajectory_plot,
)

SYMLOG_EPS: float = 0.25
"""Half-width of the linear region of the ``symlog`` warp, in the units of
whatever axis it is applied to."""

TEMPERATURE_BOUNDS: tuple[tuple[float, float], ...] = ((270.0, 350.0),)
"""Input box of the rate network. Slightly wider than the covariate levels
in the data, so ``to_latent`` stays in its exact region at the extremes."""

K_BOUNDS: tuple[tuple[float, float], ...] = ((1e-3, 1.0),)
"""Output box of the rate network, three decades wide. The true rates span
0.006 to 0.28, so the box is loose on purpose: the point of ``log10`` is
that a loose box costs almost nothing in resolution."""

STATE_BOUNDS: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (-2.0, 2.0))
"""Input box of the residual network."""

RESIDUAL_BOUNDS: tuple[tuple[float, float], ...] = ((-3.0, 3.0), (-3.0, 3.0))
"""Output box of the residual network. The true coupling reaches about 0.6
on this data, so most of the box is headroom the symlog warp deliberately
under-resolves."""


def _register_symlog() -> None:
    """Register a signed-logarithmic axis warp under the name ``symlog``.

    ``forward(x) = sign(x) log(1 + |x| / eps)``, inverted exactly. Monotone
    on the whole line, smooth through zero, and ``forward(0) = 0``, so the
    box midpoint stays at zero: a freshly initialised residual network
    sitting near latent zero produces a correction near zero rather than
    one at some arbitrary interior point.

    Registration is idempotent here because the module-level call runs
    once, but ``register_warp`` overwrites without warning, so a name
    collision with a future package warp would be silent. Prefix custom
    names if that matters to you.
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

    The only place in this file that knows which architecture is in use.
    Both branches return a ``Predictor``, and everything downstream sees
    only that.
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

    ``warp="log10"`` is the whole reason this reads well. With a linear
    box over ``(1e-3, 1)`` the latent midpoint is ``0.5``, and every rate
    in the data would sit in the bottom 3% of the box where the sigmoid is
    steepest and the network has to produce large negative latents to
    reach. Under ``log10`` the midpoint is ``0.032`` and the true range
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


def _state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 2"]:
    """Both coordinates are observed."""
    return state


def simulate_fn(
    predictors: tuple[BoundedPredictor, ...],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate the hybrid field for one experiment.

    The two network placements are visible in the first six lines. The
    rate network runs here, once, and its output is closed over as a
    constant. The residual network runs inside ``vector_field``, once per
    step. Both are ordinary calls on ordinary pytrees; the framework does
    not need to be told which is which.
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


def report_rate_net(rate_net: BoundedPredictor) -> None:
    """Print recovered against true ``k`` at every temperature level in the data.

    This is the check that matters for the outside-the-solver network: it
    never sees ``k``, only trajectories, so agreeing with the Arrhenius
    law here means the covariate dependence was recovered rather than
    memorised per experiment.
    """
    print("  temperature   true k     fitted k    ratio")
    for temperature in TEMPERATURES:
        truth = float(true_k(temperature))
        fitted = float(rate_net({"temperature": jnp.asarray(temperature)}).reshape(()))
        print(f"    {temperature:6.1f}    {truth:.5f}    {fitted:.5f}    {fitted / truth:5.2f}")


def report_residual_net(residual_net: BoundedPredictor) -> None:
    """Compare the learned correction with ``C y^3`` on a grid inside the data range.

    Reported as a relative RMS over the grid, not pointwise, because the
    residual is only identifiable where trajectories actually went.
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

    print("[data] irregular spiral, diffrax latent_ode sampling, thinned per channel")
    experiments = irregular_spiral(n_experiments=args.experiments, key=k_data)
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
        predictors, dataset, config, simulate_fn=simulate_fn, solver=solver, key=k_train
    )
    print(f"  loss at phase starts: {history[0]:.4f} -> {history[args.steps // 2]:.4f}")
    print(f"  final loss: {history[-1]:.5f}")

    # The penalty the optimiser was actually charged, read back at the end.
    # Zero means no predictor is pinned against a bound anywhere in its
    # declared input box, which is the state you want to ship in. A
    # non-zero value says the network is relying on its bound to represent
    # something, and the bound is probably in the wrong place.
    saturation = float(bound_penalty(trained, collocation_grids(trained)))
    print(f"  end-of-run saturation penalty: {saturation:.3e}")

    print("\n[rate network] recovered temperature dependence")
    report_rate_net(trained[0])
    if len(trained) > 1:
        print("\n[residual network] recovered cubic coupling")
        report_residual_net(trained[1])

    predictions = predict_dataset(trained, dataset, simulate_fn=simulate_fn, solver=solver)
    diag = compute_diagnostics(predictions, dataset)
    print()
    print_diagnostics(diag)

    if not args.no_plot:
        suffix = "mechanistic" if args.mechanistic_only else args.inner
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            diag,
            title=f"Hybrid ODE parity ({suffix})",
            save_path=args.plot_dir / f"hybrid_parity_{suffix}.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained,
            simulate_fn=simulate_fn,
            solver=solver,
            max_experiments=4,
            title=f"Hybrid ODE trajectories ({suffix})",
            save_path=args.plot_dir / f"hybrid_trajectories_{suffix}.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
