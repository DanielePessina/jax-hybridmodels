"""Harmonic-oscillator hybrid training (single-file end-to-end example).

Overview
--------
A minimal, pure-physics counterpart to the crystallisation example: the
"hidden physics" is a single trainable scalar — the angular frequency
``omega`` of a 1-D harmonic oscillator. The ODE is the textbook system::

    dx/dt = v
    dv/dt = -omega^2 * x

Each experiment in the synthetic dataset is one oscillator with a known
ground-truth ``omega = 1.0`` and a different initial state ``(x0, v0)``.
Only the position channel is observed (with light Gaussian noise on top of
the closed-form solution); velocity is part of the latent state.

A ``BoundedPredictor`` wraps a single-leaf ``OmegaPredictor`` and bounds
``omega`` into ``[0.5, 2.0]``. The trainer is invited to recover ``omega ≈ 1.0``
from positions alone.

Initial state convention
------------------------
``y0 = [x0, v0]`` is constructed by ``_y0_fn`` from a ground-truth tuple
synthesised at dataset-build time. Both components are needed because the
ODE is second-order and the predictor only owns ``omega``.

Phase status
------------
Same surface as ``train_kinetic.py``: phases 1-9. The synthetic dataset
exercises ``ChannelObs`` / ``Experiment`` / ``make_dataset`` / ``BoundedPredictor``
/ ``train_with_optax`` end-to-end on a problem with a known optimum, which
makes it useful both as documentation and as a quick sanity check that
training is wired correctly.

How to run
----------
``uv run python examples/pendulum/train_harmonic.py``

No external data — the dataset is synthesised on every run from the closed
form ``x(t) = x0 cos(omega t) + (v0 / omega) sin(omega t)``.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
    SolverConfig,
    make_dataset,
    make_experiment,
    predict_dataset,
)
from hybridmodels.predictors.base import Predictor
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# ``examples/_shared`` is a sibling of this scenario directory; add the
# parent of this file to sys.path so the helpers import as a top-level
# package without requiring any install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
    trajectory_plot,
)

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

OMEGA_TRUE: float = 1.0
"""Ground-truth angular frequency. Training should recover this from data."""

OMEGA_BOUNDS: tuple[float, float] = (0.5, 2.0)
"""Search box for the recovered ``omega``; ``BoundScaler`` keeps the latent
parameter inside this interval via the sigmoid transform."""

INITIAL_STATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, -0.5),
    (1.0, 1.0),
    (-0.7, 0.4),
    (0.3, 0.9),
)
"""Per-experiment ``(x0, v0)`` initial conditions."""

T_MAX: float = 5.0
"""Final observation time, in oscillator units (omega=1 -> period 2π ~ 6.28).

Five units cover roughly 0.8 of one period — enough phase coverage to fit
``omega`` without aliasing into the wrong basin of attraction."""

N_TIMESTEPS: int = 12
"""Per-experiment number of evenly-spaced observations on ``[0, T_MAX]``."""

NOISE_STD: float = 0.02
"""Gaussian noise standard deviation applied to observed positions."""

OUTPUT_CHANNELS: tuple[str, ...] = ("position",)
"""Only the first state component (``x``) is observable."""


# --------------------------------------------------------------------------- #
# Predictor — a single-leaf scalar bounded into [0.5, 2.0]                    #
# --------------------------------------------------------------------------- #


class OmegaPredictor(Predictor):
    """Trivial predictor whose only trainable parameter is a scalar ``omega``.

    Wrapped by ``BoundedPredictor`` in ``_build_predictor``; the wrapper's
    ``out_scaler`` maps the unbounded latent value to the physical box
    ``OMEGA_BOUNDS``. The predictor's ``__call__`` ignores the input — it
    returns the same scalar regardless of the covariate it receives, since
    every experiment shares the same ground-truth ``omega``.

    Implementing ``initialized_with_key`` lets ``reinitialize_with_key`` /
    the optax tournament re-init the leaf with a fresh standard-normal
    sample rather than re-running the abstract default.
    """

    omega_lat: Array

    def __init__(self, omega_lat: Array | float = 0.0) -> None:
        self.omega_lat = jnp.asarray(omega_lat, dtype=jnp.float32)

    def __call__(self, x: Array) -> Float[Array, " 1"]:
        return self.omega_lat[None]

    def initialized_with_key(self, key: Array) -> OmegaPredictor:
        return OmegaPredictor(jr.normal(key))


def _build_predictor(key: Array) -> BoundedPredictor:
    """Construct ``BoundedPredictor`` wrapping the ``OmegaPredictor``.

    Pipeline ``dict[str, Array] -> [omega]``::

        BoundedPredictor.input_keys : ("dummy",) -> [1]
        in_scaler                   : [1] physical -> [1] latent (no-op in practice
                                      because the inner predictor ignores its input)
        OmegaPredictor              : Array -> [1]   (returns the bounded latent omega)
        out_scaler                  : [1] latent -> [1] physical (sigmoid into OMEGA_BOUNDS)

    ``BoundedPredictor`` needs at least one input slot (cardinality of
    ``in_scaler.bounds``); we use a constant ``"dummy"`` covariate to
    satisfy the framework's named-input contract without leaking
    experiment-specific information.
    """
    in_scaler = BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid")
    inner = OmegaPredictor(jr.normal(key))
    out_scaler = BoundScaler(bounds=(OMEGA_BOUNDS,), transform="sigmoid")
    return BoundedPredictor(
        input_keys=("dummy",),
        in_scaler=in_scaler,
        inner=inner,
        out_scaler=out_scaler,
    )


# --------------------------------------------------------------------------- #
# Hooks: y0_fn, state_to_output                                               #
# --------------------------------------------------------------------------- #


def _y0_fn_factory(initial_state: Float[Array, " 2"]):
    """Per-experiment closure returning the closed-over ``[x0, v0]``.

    ``y0_fn`` must accept ``(covariates, channels)``; both are unused here
    because the ground-truth state is captured at synthesis time, not
    derived from the observations.
    """

    def _y0_fn(_cov: dict[str, Array], _channels: dict[str, ChannelObs]) -> Array:
        return initial_state

    return _y0_fn


def _state_to_output(state: Float[Array, "T 2"]) -> Float[Array, "T 1"]:
    """Project full state ``[x, v]`` to the observed channel ``[x]``.

    Channel order matches ``OUTPUT_CHANNELS = ("position",)``.
    """
    return state[..., :1]


# --------------------------------------------------------------------------- #
# simulate_fn — second-order linear ODE                                       #
# --------------------------------------------------------------------------- #


def _simulate_fn(
    predictor: BoundedPredictor,
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 2"],
    solver: SolverConfig,
) -> Float[Array, "T 2"]:
    """Integrate the harmonic oscillator for one experiment.

    Conforms to the framework's ``simulate_fn`` signature
    ``(predictor, ts, covariates, y0, solver) -> [T, S]`` — i.e. it
    returns the full simulator state for every timestamp in ``ts``.
    The user owns the physics (the vector field built below); the
    framework owns the surrounding ``vmap`` / ``jit`` / ``grad``
    plumbing.

    Pipeline
    --------
    1. Evaluate ``predictor(covariates) -> [omega]`` (bounded, physical units).
    2. Build the linear vector field ``f(t, [x, v]) = [v, -omega^2 x]``.
    3. ``diffrax.diffeqsolve`` over ``ts``.

    Shape conventions
    -----------------
    ``ts``: ``[T]`` — observation times.
    ``y0``: ``[2] = [x0, v0]`` from ``_y0_fn``.
    Returns ``[T, 2]`` aligned with ``ts``.
    """
    omega = predictor(covariates).reshape(())
    omega_sq = omega * omega

    def vector_field(t: Array, y: Float[Array, " 2"], args: object) -> Array:
        return jnp.stack([y[1], -omega_sq * y[0]])

    term = diffrax.ODETerm(vector_field)
    sol = diffrax.diffeqsolve(
        term,
        solver.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=solver.dt0 if solver.dt0 is not None else 0.05,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


# --------------------------------------------------------------------------- #
# Synthetic dataset                                                           #
# --------------------------------------------------------------------------- #


def _true_position(omega: float, t: Array, x0: float, v0: float) -> Array:
    """Closed-form solution ``x(t) = x0 cos(ωt) + (v0/ω) sin(ωt)``."""
    return x0 * jnp.cos(omega * t) + (v0 / omega) * jnp.sin(omega * t)


def _build_experiments(noise_key: Array) -> list[Experiment]:
    """Synthesise the harmonic-oscillator dataset.

    For each ``(x0, v0)`` in ``INITIAL_STATES`` we sample ``N_TIMESTEPS`` evenly
    on ``[0, T_MAX]``, evaluate the closed-form position, add Gaussian noise of
    standard deviation ``NOISE_STD``, and bundle the result into an
    ``Experiment`` whose ``y0_fn`` returns the ground-truth initial state.

    A single trivial covariate ``"dummy"`` is added so the predictor's
    ``input_keys`` tuple has a slot to pull on.
    """
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments: list[Experiment] = []
    rng = noise_key
    for i, (x0, v0) in enumerate(INITIAL_STATES):
        rng, k = jr.split(rng)
        clean = _true_position(OMEGA_TRUE, ts, x0, v0)
        noisy = clean + NOISE_STD * jr.normal(k, ts.shape)
        channels = {
            "position": ChannelObs(
                ts=ts,
                values=noisy,
                variance=jnp.full(ts.shape, NOISE_STD**2),
            )
        }
        y0 = jnp.asarray([x0, v0], dtype=jnp.float32)
        experiments.append(
            make_experiment(
                covariates={"dummy": 0.0},
                channels=channels,
                y0_fn=_y0_fn_factory(y0),
                exp_id=f"osc_{i}_x0={x0}_v0={v0}",
            )
        )
    return experiments


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _read_omega(predictor: BoundedPredictor) -> float:
    """Pull the bounded ``omega`` value out of the predictor for reporting."""
    out = predictor({"dummy": jnp.asarray(0.0)})
    return float(jnp.asarray(out).reshape(()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to write parity + trajectory PNGs into.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (still prints diagnostics).",
    )
    args = parser.parse_args()

    apply_default_style()

    root_key = jr.PRNGKey(args.seed)
    k_data, k_init, k_train = jr.split(root_key, 3)

    print("[build] synthetic harmonic-oscillator dataset")
    experiments = _build_experiments(k_data)
    dataset = make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(experiments)} experiments, {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(
            f"    bucket {i}: ts={tuple(bp.ts.shape)}, "
            f"y_observed={tuple(bp.y_observed.shape)}, n_obs={int(bp.n_obs)}"
        )

    print("\n[build] solver + predictor")
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=10_000,
        dt0=0.05,
    )
    predictor = _build_predictor(k_init)
    print(f"  initial omega = {_read_omega(predictor):.4f} (target {OMEGA_TRUE})")

    print("\n[train] optax (single phase, mse loss)")
    config = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        log_every=max(1, args.steps // 10),
        verbose=False,
    )
    history, trained = train_with_optax(
        predictor,
        dataset,
        config,
        simulate_fn=_simulate_fn,
        solver=solver,
        key=k_train,
    )
    print(f"  {len(history)} steps; final loss {history[-1]:.6f}")
    sample_every = max(1, len(history) // 10)
    sampled = [f"{loss:.4f}" for loss in history[::sample_every]]
    print(f"  loss every ~{sample_every} steps: {sampled}")

    final_omega = _read_omega(trained)
    print(f"\n  recovered omega = {final_omega:.4f} (target {OMEGA_TRUE})")
    print(f"  absolute error  = {abs(final_omega - OMEGA_TRUE):.4f}")

    # Diagnostics + default plots: predict_dataset returns one [N, T, D]
    # array per bucket; the helpers walk it in lockstep with the dataset.
    print("\n[diagnostics] per-channel parity stats over the training set")
    predictions = predict_dataset(trained, dataset, simulate_fn=_simulate_fn, solver=solver)
    diag = compute_diagnostics(predictions, dataset)
    print_diagnostics(diag)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            diag,
            title="Pendulum parity (trained model)",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            max_experiments=len(experiments),
            title="Pendulum trajectories (trained model)",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
