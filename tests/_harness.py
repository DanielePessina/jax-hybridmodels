"""Scaffolding shared by more than one test module.

Only definitions that were **byte-identical** across test files live here.
Where two files looked similar but differed in substance they keep their own
copy, because merging them would silently change what a test integrates. The
known divergences, deliberately left alone:

* ``tests/test_prediction.py`` projects with ``full_state[:, :1]`` and drives
  diffrax through ``solver.stepsize_controller()`` with an explicit
  ``adjoint=``. The training tests project with ``state[..., 0:1]`` and build
  a ``PIDController`` inline with no adjoint. Different code paths on purpose.
* ``tests/test_prediction.py`` runs tighter tolerances (``rtol=1e-6``,
  ``atol=1e-8``) than the training tests. Hence the arguments on
  :func:`solver_config` rather than one frozen config.

Plain functions, not fixtures: the existing call sites are direct calls, many
of them from inside other module-level helpers where pytest cannot inject
anything. Importing keeps every call site as it was.

``from _harness import ...`` resolves because ``tests/`` has no ``__init__.py``,
so pytest's default ``prepend`` import mode puts that directory on ``sys.path``.
Adding ``tests/__init__.py`` would break every importer here.
"""

from __future__ import annotations

from io import StringIO

import diffrax
import jax.numpy as jnp
from jax import Array
from rich.console import Console

from jaxhybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from jaxhybridmodels.predictors.base import Predictor
from jaxhybridmodels.solver import SolverConfig

# ---------------------------------------------------------------------------
# Harmonic oscillator: state ``[position, velocity]``, dy/dt = [v, -w^2 x].
# Used by tests/test_train_optax.py, tests/test_ui_rich.py and
# tests/test_prediction.py.
# ---------------------------------------------------------------------------

OMEGA_TRUE: float = 1.0
N_TIMESTEPS: int = 10
T_MAX: float = 5.0
INITIAL_STATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, -0.5),
    (1.0, 1.0),
)


class OmegaPredictor(Predictor):
    """One scalar leaf, so the pytree stays trivial and shapes are the focus."""

    omega: Array

    def __init__(self, omega) -> None:
        # Strong-type the leaf so the predictor's pytree weak_type does not flip
        # after the first apply_update, which would force a retrace.
        self.omega = jnp.asarray(omega, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:  # type: ignore[override]
        return self.omega


def y0_fn_factory(y0: Array):
    """Return a ``y0_fn`` that ignores covariates and channels and yields ``y0``."""

    def _y0_fn(_cov, _channels):
        return y0

    return _y0_fn


def solver_config(
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_steps: int = 4096,
    dt0: float = 0.05,
) -> SolverConfig:
    """A Tsit5 config. Defaults match the training tests.

    ``tests/test_prediction.py`` passes ``rtol=1e-6, atol=1e-8``; its oracle
    compares an un-vmapped Python loop against the vmapped path, so it needs
    the integration error well below the comparison tolerance.
    """
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        dt0=dt0,
    )


def true_position(omega: float, t: Array, x0: float, v0: float) -> Array:
    """Analytic solution of the oscillator, the ground truth observations."""
    return x0 * jnp.cos(omega * t) + (v0 / omega) * jnp.sin(omega * t)


def oscillator_state_to_output(state: Array) -> Array:
    """Project ``[T, S]`` onto the single observed channel, position.

    A module-level function so every call site passes the same object; a
    fresh closure per call would cost retraces inside compiled kernels.
    """
    return state[..., 0:1]


def make_oscillator_dataset(
    initial_states: tuple[tuple[float, float], ...] = INITIAL_STATES,
    t_max: float = T_MAX,
    n_timesteps: int = N_TIMESTEPS,
) -> Dataset:
    """Experiments sharing one timestamp axis, position observed, velocity not."""
    ts = jnp.linspace(0.0, t_max, n_timesteps)
    experiments = []
    for i, (x0, v0) in enumerate(initial_states):
        x_obs = true_position(OMEGA_TRUE, ts, x0, v0)
        y0 = jnp.asarray([x0, v0])
        experiments.append(
            make_experiment(
                covariates={"id": float(i)},
                channels={"position": ChannelObs(ts=ts, values=x_obs)},
                y0_fn=y0_fn_factory(y0),
                exp_id=f"exp_{i}",
            )
        )
    return make_dataset(
        experiments,
        output_channel_names=("position",),
    )


def make_oscillator_simulate_fn():
    """Return a fresh ``simulate_fn`` closure over the oscillator vector field.

    Fresh per call by design: tests that count traces rely on getting a
    distinct callable each time.
    """

    def simulate_fn(predictor, ts, covariates, y0, solver):
        omega = predictor.omega

        def vector_field(t, y, args):
            return jnp.stack([y[1], -(omega**2) * y[0]])

        term = diffrax.ODETerm(vector_field)
        controller = diffrax.PIDController(rtol=solver.rtol, atol=solver.atol)
        sol = diffrax.diffeqsolve(
            term,
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=solver.max_steps,
        )
        return sol.ys

    return simulate_fn


# ---------------------------------------------------------------------------
# 4-D quadratic: a single trainable ``theta: [4]`` against a constant target.
# The "simulator" is the identity of the predictor, so no ODE runs and the
# evosax loop is isolated from numerics. Used by tests/test_train_evosax.py
# and tests/test_ui_rich_evosax.py.
# ---------------------------------------------------------------------------

THETA_STAR = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)
N_DIM = 4


class QuadraticPredictor(Predictor):
    """Single inexact-array leaf ``theta: [4]``; covariates are ignored on call."""

    theta: Array

    def __init__(self, theta: Array) -> None:
        # Float32 to keep CMA-ES (whose state is float32 by default) and the
        # predictor's leaf in one dtype, avoiding a silent upcast that would
        # force a re-trace inside population_eval.
        self.theta = jnp.asarray(theta, dtype=jnp.float32)

    def __call__(self, covariates):  # type: ignore[override]
        # The model is just the constant theta. Returning the leaf directly
        # lets simulate_fn fold it into [T, S] = [1, 4] without an ODE call.
        return self.theta


def quadratic_state_to_output(state: Array) -> Array:
    """Project ``[T, S]`` to ``[T, D]`` as the identity: every state component is observed."""
    return state


def quadratic_dataset() -> Dataset:
    """One bucket holding one experiment, T=1, D=4, mask all True."""
    ts = jnp.array([0.0], dtype=jnp.float32)
    channels = {f"c{i}": ChannelObs(ts=ts, values=THETA_STAR[i : i + 1]) for i in range(N_DIM)}
    exp = make_experiment(
        covariates={"id": 0.0},
        channels=channels,
        y0_fn=lambda _c, _ch: jnp.zeros(N_DIM, dtype=jnp.float32),
        exp_id="exp_0",
    )
    return make_dataset(
        [exp],
        output_channel_names=tuple(f"c{i}" for i in range(N_DIM)),
    )


def quadratic_simulate_fn(predictor, ts, covariates, y0, solver):
    """Identity-of-predictor "simulator": returns ``[T, 4]``, constant in time."""
    # predictor(covariates) -> [4]; broadcasting to [T, 4] gives the
    # per-experiment full-state trajectory the framework expects.
    return jnp.broadcast_to(predictor(covariates)[None, :], (ts.shape[0], N_DIM))


# ---------------------------------------------------------------------------
# Rich UI capture. Used by tests/test_ui_rich.py and
# tests/test_ui_rich_evosax.py.
# ---------------------------------------------------------------------------


def recording_console() -> Console:
    """Recording console with no TTY, so ``Live`` prints once at ``stop()``."""
    return Console(record=True, force_terminal=False, width=120, file=StringIO())
