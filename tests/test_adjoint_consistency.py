"""End-to-end consistency of gradients taken back through an ODE solve.

``SolverConfig.adjoint`` picks how diffrax differentiates through the
solve: ``Direct`` stores the whole forward tape, ``RecursiveCheckpoint``
stores O(log n) checkpoints and recomputes the rest, ``Backsolve``
re-integrates the adjoint ODE backwards in constant memory. All must
return the *same* gradient on a benign problem; only the memory and the
error accumulation differ.

These are integration tests over the public surface (``predict_bucket``,
``masked_mse``), not unit tests of inner kernels. The ``simulate_fn``
they drive forwards ``solver.adjoint`` into the diffrax call — the
documented contract (the user owns physics, the framework owns
jit/vmap/grad).

One real constraint is pinned here because it is easy to trip over.
Diffrax's ``BacksolveAdjoint`` differentiates by re-integrating the
adjoint ODE backwards, and its VJP rule can only differentiate with
respect to values the vector field receives through ``args``. It cannot
differentiate through values *closed over* in the vector field, which is
the canonical shape of this library's ``simulate_fn`` (predictors are
closed over). So Backsolve needs the args-threading recipe below; Direct
and RecursiveCheckpoint work with the plain closed-over shape.

The finite-difference anchor keeps the test non-vacuous: it asserts the
gradient through the solve is the *correct* gradient, not merely that the
adjoints agree with each other. Three adjoints that share a bug would
pass an agree-with-each-other test.
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _harness import OmegaPredictor, make_oscillator_dataset, oscillator_state_to_output
from jax import Array

from jaxhybridmodels.data import BucketPayload, Dataset
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.prediction import predict_bucket
from jaxhybridmodels.solver import ADJOINT_REGISTRY, SolverConfig
from jaxhybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

ADJOINT_NAMES: tuple[str, ...] = ("Direct", "RecursiveCheckpoint")
# Backsolve needs the args-threading recipe, so it is excluded from the
# closed-over tests but included in the trainer end-to-end test, where the
# args-threaded simulate_fn is used.
ALL_ADJOINT_NAMES: tuple[str, ...] = ("Direct", "RecursiveCheckpoint", "Backsolve")

# Central-difference step for the finite-difference anchor. The harmonic
# oscillator loss is smooth in ``omega``, so a modest step keeps the
# cancellation noise well below the assertion tolerance.
FD_H: float = 1e-2

# Agreement tolerance against the finite-difference anchor. Measured
# agreement on this benign problem is ~1e-5 relative (Backsolve) to ~3e-6
# (Direct/RecursiveCheckpoint), so 1e-2 leaves ~100x margin while still
# catching percent-level gradient corruption.
GRAD_TOL: float = 1e-2


def make_closed_over_simulate_fn():
    """The canonical library shape: predictors closed over in the vector field.

    This is how every example writes ``simulate_fn``. Works with ``Direct``
    and ``RecursiveCheckpoint``; raises with ``Backsolve`` (pinned below).
    """

    def simulate_fn(predictor, ts, covariates, y0, solver):
        omega = predictor[0].omega

        def vector_field(t, y, args):
            return jnp.stack([y[1], -(omega**2) * y[0]])

        term = diffrax.ODETerm(vector_field)
        sol = diffrax.diffeqsolve(
            term,
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=solver.stepsize_controller(),
            adjoint=solver.adjoint,
            max_steps=solver.max_steps,
        )
        return sol.ys

    return simulate_fn


def make_args_threaded_simulate_fn():
    """The recipe that makes ``Backsolve`` work: predictors via ``args``.

    The vector field receives the predictors as ``args`` (an explicit
    input to the solve) instead of closing over them, so the Backsolve
    VJP rule can differentiate through them in the backward pass.
    """

    def simulate_fn(predictor, ts, covariates, y0, solver):
        def vector_field(t, y, args):
            omega = args[0].omega
            return jnp.stack([y[1], -(omega**2) * y[0]])

        term = diffrax.ODETerm(vector_field)
        sol = diffrax.diffeqsolve(
            term,
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            args=predictor,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=solver.stepsize_controller(),
            adjoint=solver.adjoint,
            max_steps=solver.max_steps,
        )
        return sol.ys

    return simulate_fn


def data_loss(predictors, bp: BucketPayload, simulate_fn, solver: SolverConfig) -> Array:
    """Public forward loss: predict every experiment, then ``masked_mse``."""
    pred_obs = predict_bucket(
        predictors,
        bp,
        simulate_fn=simulate_fn,
        state_to_output=oscillator_state_to_output,
        solver=solver,
    )
    return masked_mse(pred_obs, bp)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return make_oscillator_dataset()


@pytest.fixture
def closed_over_simulate_fn():
    return make_closed_over_simulate_fn()


@pytest.fixture
def args_threaded_simulate_fn():
    return make_args_threaded_simulate_fn()


@pytest.fixture
def predictors():
    return (OmegaPredictor(omega=1.5),)


@pytest.fixture
def bp(dataset: Dataset) -> BucketPayload:
    assert len(dataset.bucket_payloads) == 1
    return dataset.bucket_payloads[0]


def make_solver(adjoint_name: str) -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=4096,
        dt0=0.05,
        adjoint=ADJOINT_REGISTRY[adjoint_name](),
    )


def grads_for(simulate_fn, solver: SolverConfig, predictors, bp: BucketPayload):
    """Gradient of the data loss w.r.t. the predictors, through the solve."""
    grad_fn = eqx.filter_grad(lambda ps, bucket: data_loss(ps, bucket, simulate_fn, solver))
    return grad_fn(predictors, bp)


def finite_difference_grads(simulate_fn, solver: SolverConfig, predictors, bp: BucketPayload):
    """Central-difference gradient of the forward loss w.r.t. ``omega``.

    The forward loss does not depend on the adjoint (the adjoint only
    changes the backward pass), so this is a single fixed anchor every
    adjoint strategy must match.
    """

    def forward(ps):
        return float(data_loss(ps, bp, simulate_fn, solver))

    omega0 = float(predictors[0].omega)

    def at(omega: float):
        return eqx.tree_at(lambda ps: ps[0].omega, predictors, jnp.asarray(omega))

    return (forward(at(omega0 + FD_H)) - forward(at(omega0 - FD_H))) / (2.0 * FD_H)


def all_finite(grads) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))


def test_direct_and_recursive_agree_with_finite_difference(closed_over_simulate_fn, predictors, bp):
    """The gradient machinery through the solve is correct, not just consistent."""
    expected = finite_difference_grads(
        closed_over_simulate_fn, make_solver("Direct"), predictors, bp
    )
    assert expected != 0.0
    for name in ADJOINT_NAMES:
        grads = grads_for(closed_over_simulate_fn, make_solver(name), predictors, bp)
        assert all_finite(grads)
        assert abs(float(grads[0].omega) - expected) <= GRAD_TOL * abs(expected)


def test_direct_and_recursive_agree_with_each_other(closed_over_simulate_fn, predictors, bp):
    got = {
        name: float(grads_for(closed_over_simulate_fn, make_solver(name), predictors, bp)[0].omega)
        for name in ADJOINT_NAMES
    }
    assert abs(got["RecursiveCheckpoint"] - got["Direct"]) <= GRAD_TOL * abs(got["Direct"])


def test_backsolve_works_with_args_threading(args_threaded_simulate_fn, predictors, bp):
    """Backsolve needs predictors threaded through ``args``; then it agrees too."""
    expected = finite_difference_grads(
        args_threaded_simulate_fn, make_solver("Direct"), predictors, bp
    )
    grads = grads_for(args_threaded_simulate_fn, make_solver("Backsolve"), predictors, bp)
    assert all_finite(grads)
    assert abs(float(grads[0].omega) - expected) <= GRAD_TOL * abs(expected)


def test_backsolve_rejects_closed_over_predictors(closed_over_simulate_fn, predictors, bp):
    """Backsolve cannot differentiate through a value closed over in the vector field.

    Pinning the diffrax constraint so a user who switches
    ``SolverConfig.adjoint`` to ``"Backsolve"`` on the canonical
    closed-over ``simulate_fn`` gets an error they can act on, not a
    silently wrong gradient.
    """
    with pytest.raises(Exception, match="closed-over"):
        grads_for(closed_over_simulate_fn, make_solver("Backsolve"), predictors, bp)


def test_training_converges_with_every_adjoint(args_threaded_simulate_fn, dataset):
    """The whole training pipeline is adjoint-consistent, not just a bare grad.

    Drives ``train_with_optax`` end-to-end with each adjoint strategy and
    checks they all land at the same ``omega``. This is the integration
    test the module promises: the trainer forwards ``solver`` (and hence
    ``solver.adjoint``) into the user's ``simulate_fn``, so the adjoint
    flows through the whole compiled pipeline (dataset → bucket_step →
    apply_update → restore_best).

    The start (1.2) is deliberately inside the basin of the global loss
    minimum at ``omega=1.0``. The sampled loss has an aliasing local
    minimum near ``omega=2.4`` (cos(2.4t) aliases with cos(t) over 10
    samples on [0, 5]), so starting at 2.0 would converge to the wrong
    basin for *every* adjoint and the "moved toward truth" check would
    be meaningless.
    """
    trained: dict[str, float] = {}
    for name in ALL_ADJOINT_NAMES:
        final = train_with_optax(
            (OmegaPredictor(omega=1.2),),
            dataset,
            OptaxTrainingConfig(
                steps=(12,),
                lr=(5e-2,),
                optimizer=("adamw",),
                reset_optimiser_state=(False,),
                restore_best=False,
            ),
            simulate_fn=args_threaded_simulate_fn,
            state_to_output=oscillator_state_to_output,
            solver=make_solver(name),
            key=jax.random.PRNGKey(0),
        )[1]
        trained[name] = float(final[0].omega)
    # All adjoints must agree on where training lands (identical loss,
    # identical optimiser; only the backward pass differs).
    for other in ALL_ADJOINT_NAMES[1:]:
        assert abs(trained[other] - trained["Direct"]) <= 1e-3
    # And that landing must be a real improvement toward the truth (1.0),
    # proving the gradient through the whole pipeline is correct, not just
    # consistent between adjoints.
    assert all(abs(v - 1.0) < 0.1 for v in trained.values())
