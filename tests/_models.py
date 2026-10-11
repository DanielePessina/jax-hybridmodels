"""The small models every end-to-end test builds on.

First-order decay, ``dy/dt = -k * y + offset``, observed directly. The rate
``k`` is either a ``Parameter`` (one scalar to learn) or a
``BoundedPredictor`` network of the covariate ``temp``. The offset is a
``Constant`` so every fit also checks that fixed values stay fixed.

Everything here is a plain module-level function or constant, so every test
passes the same callable objects to the trainers. A fresh closure per call
would defeat JAX's trace cache and cost a recompile each time.

The solver is deliberately loose (``max_steps=64``, ``rtol=1e-4``). Decay
over ``t <= 2`` needs a handful of Tsit5 steps, and the compile cost of the
solve scales with ``max_steps`` under the default ``DirectAdjoint``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.training import OptaxTrainingConfig

TRUE_RATE = 0.8
CHANNELS = ("conc",)
SOLVER = hm.SolverConfig(solver=diffrax.Tsit5(), rtol=1e-4, atol=1e-6, max_steps=64, dt0=0.1)

# (exp_id, temp, observation times). Two union lengths, so two buckets, and
# irregular grids so bucket padding and masks are exercised.
EXPERIMENTS: tuple[tuple[str, float, tuple[float, ...]], ...] = (
    ("a", 0.5, (0.5, 1.0, 2.0)),
    ("b", 1.0, (1.0, 2.0)),
    ("c", 1.5, (0.25, 0.75, 1.5)),
    ("d", 1.0, (0.5, 1.5)),
)


def constant_rate(temp: float) -> float:
    del temp
    return TRUE_RATE


def temp_rate(temp: float) -> float:
    """The law the network model has to learn: faster decay when hotter."""
    return 0.4 + 0.4 * temp


def decay_dataset(rate_fn: Callable[[float], float] = constant_rate) -> hm.Dataset:
    """Exact decay curves from ``y0 = 1`` at ``t0 = 0`` under ``rate_fn(temp)``."""
    experiments = []
    for exp_id, temp, times in EXPERIMENTS:
        ts = jnp.asarray(times)
        values = jnp.exp(-rate_fn(temp) * ts)
        experiments.append(
            hm.make_experiment(
                channels={"conc": hm.ChannelObs(ts=ts, values=values, variance=1e-2)},
                covariates={"temp": temp},
                y0=[1.0],
                t0=0.0,
                exp_id=exp_id,
            )
        )
    return hm.make_dataset(experiments, output_channel_names=CHANNELS)


def _decay(rate: Array, offset: Array, ts: Array, y0: Array, solver: hm.SolverConfig) -> Array:
    term = diffrax.ODETerm(lambda t, y, args: -rate * y + offset)
    ys = solver.diffeqsolve(term, ts, y0).ys
    assert ys is not None
    return ys


def simulate_parameter(
    predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any
) -> Array:
    del covariates
    return _decay(predictors["rate"].value, predictors["offset"].value, ts, y0, solver)


def simulate_network(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    rate = predictors["rate"](covariates)
    return _decay(rate, predictors["offset"].value, ts, y0, solver)


def identity(state: Array) -> Array:
    return state


def parameter_predictors(rate: float = 0.3) -> dict[str, Any]:
    return {"rate": hm.Parameter(rate, bounds=(0.0, None)), "offset": hm.Constant(0.0)}


def network_predictors(key: Array | None = None) -> dict[str, Any]:
    key = jr.key(0) if key is None else key
    inner = hm.MLPPredictor(
        in_size=1, out_size=1, width_size=4, depth=1, activation_name="tanh", key=key
    )
    rate = hm.BoundedPredictor(
        inner=inner, input_keys=("temp",), input_bounds=((0.0, 2.0),), output_bounds=((0.05, 3.0),)
    )
    return {"rate": rate, "offset": hm.Constant(0.0)}


def parameter_model(rate: float = 0.3) -> hm.HybridModel:
    return hm.HybridModel(
        predictors=parameter_predictors(rate),
        simulate_fn=simulate_parameter,
        state_to_output=identity,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )


def network_model(key: Array | None = None) -> hm.HybridModel:
    return hm.HybridModel(
        predictors=network_predictors(key),
        simulate_fn=simulate_network,
        state_to_output=identity,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )


def quiet(**kwargs: Any) -> OptaxTrainingConfig:
    """An Optax config with the Rich UI off, which tests never want on by accident."""
    return OptaxTrainingConfig(verbose=False, **kwargs)


# Keyword arguments for the functional trainers (train_with_optax and
# friends) on the Parameter model.
PARAMETER_KWARGS: dict[str, Any] = dict(
    simulate_fn=simulate_parameter, state_to_output=identity, solver=SOLVER
)
NETWORK_KWARGS: dict[str, Any] = dict(
    simulate_fn=simulate_network, state_to_output=identity, solver=SOLVER
)
