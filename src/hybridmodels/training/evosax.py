"""Gradient-free training loop for small-parameter predictors, driven by evosax.

Evolutionary search instead of gradient descent. Each generation samples
a *population* of candidate parameter vectors, scores them all, and lets
CMA-ES move its sampling distribution towards the good ones. No
derivative is ever taken, which is what makes it useful when the ODE
adjoint is unreliable or the loss surface is full of local minima.

The cost is that evaluations needed grow quickly with parameter count.
This loop targets small kinetic predictors, roughly 4 to 10 trainable
scalars. Use ``hybridmodels.training.optax`` for network-sized fits, or
run this first and polish with Optax afterwards.

There are no phases. The run is ``num_generations`` iterations over a
population of ``population_size`` individuals.

JIT boundary
------------
The trainable parameters are ravelled to one flat vector via
``jax.flatten_util.ravel_pytree``. The per-individual loss
``single_eval(flat) -> scalar`` unflattens it, glues it back to the
static part with ``eqx.combine`` (closed over, since callables and
static fields cannot pass through ``vmap``), and runs the bucket
dispatch loop inside the trace, so the whole multi-bucket forward pass
becomes one fused kernel. ``population_eval =
eqx.filter_jit(jax.vmap(single_eval))`` evaluates the population in
parallel.

Initial-population modes
------------------------
``"warm"``
    CMA-ES starts with ``mean = flat`` and ``std = sigma_init``. Its
    first ``ask`` produces the initial population.
``"uniform_box"``
    A per-individual ``flat + Uniform(-extent, extent)`` is built
    host-side and **evaluated directly in generation 0**, bypassing
    CMA-ES's first ``ask``. The strategy is still initialised with
    ``mean = flat``, so the following ``tell`` updates its mean and
    covariance consistently with the rest of the run.
``"lhs_box"``
    Same shape as ``uniform_box`` with offsets from
    ``scipy.stats.qmc.LatinHypercube``, computed host-side (``scipy`` is
    a hard dependency). Skipping the first ``ask`` is required here: a
    prescribed LHS pattern reaches generation 0 only by direct injection.

Best-ever tracking
------------------
Each generation's ``argmin(fitness)`` is compared host-side against the
running best. The winning flat vector is turned back into predictors only
at run end. CMA-ES's own ``state.best_solution`` is deliberately unused,
so the contract stays the same whichever strategy is plugged in.
"""

# ruff: noqa: F722

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
from evosax.algorithms.distribution_based.cma_es import CMA_ES
from jax import Array
from scipy.stats import qmc

from hybridmodels.data import BucketPayload, Dataset
from hybridmodels.losses import _resolve_loss_fn
from hybridmodels.penalties import bound_penalty, collocation_grids
from hybridmodels.rng import fold
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.ui.base import EvosaxUI, SilentUI
from hybridmodels.ui.evosax import RichEvosaxUI

_SUPPORTED_INIT_MODES: tuple[str, ...] = ("warm", "uniform_box", "lhs_box")
_SUPPORTED_ALGORITHMS: tuple[str, ...] = ("CMA_ES",)


@dataclass(frozen=True)
class EvosaxTrainingConfig:
    """Configuration for :func:`train_with_evosax`.

    Unlike :class:`~hybridmodels.training.optax.OptaxTrainingConfig` there
    are no phase-keyed tuples: the run is a flat loop, so every field is a
    scalar.

    Attributes
    ----------
    algorithm
        Evosax strategy name. Only ``"CMA_ES"`` is wired in. The field
        exists so another strategy can slot in without an API break.
    population_size, num_generations
        Loop dimensions. The population is evaluated in parallel through
        ``vmap``; generations run in sequence.
    init
        Initial-population scheme, one of ``"warm"``, ``"uniform_box"``,
        ``"lhs_box"``. The module docstring explains the difference.
    penalty_weight
        Weight on the bound-saturation penalty, folded into each
        individual's fitness. ``0.0`` disables it. Scalar, not a tuple.
    penalty_grid_points
        Points per input dimension in the collocation grid the penalty is
        evaluated on.
    init_box_extent
        Half-width of the box for ``"uniform_box"`` and ``"lhs_box"``.
        Ignored by ``"warm"``.
    sigma_init
        Initial CMA-ES step size. Used directly by ``"warm"`` and as the
        prior step size for the box-init modes. CMA-ES adapts it after
        the first ``tell``.
    loss
        A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``,
        ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``.
    channel_idx, channel_weights
        Forwarded into the resolved loss. See ``hybridmodels.losses``.
    log_every
        UI heartbeat cadence. Honoured only by the Rich UIs; the silent
        and recording UIs see every generation.
    verbose
        Selects ``RichEvosaxUI`` over ``SilentUI`` when ``ui=None``. An
        explicit ``ui=...`` argument always wins.
    """

    algorithm: str = "CMA_ES"
    population_size: int = 64
    num_generations: int = 100
    init: Literal["warm", "uniform_box", "lhs_box"] = "warm"
    penalty_weight: float = 0.0
    penalty_grid_points: int = 5
    init_box_extent: float = 2.0
    sigma_init: float = 0.1
    loss: Callable[..., Array] | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    log_every: int = 1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"EvosaxTrainingConfig.algorithm={self.algorithm!r} is not supported; "
                f"available: {list(_SUPPORTED_ALGORITHMS)}"
            )
        if self.init not in _SUPPORTED_INIT_MODES:
            raise ValueError(
                f"EvosaxTrainingConfig.init={self.init!r} is not supported; "
                f"available: {list(_SUPPORTED_INIT_MODES)}"
            )
        if self.penalty_weight < 0.0:
            raise ValueError(
                "EvosaxTrainingConfig.penalty_weight must be non-negative; "
                f"got {self.penalty_weight}"
            )
        if self.population_size <= 0:
            raise ValueError("EvosaxTrainingConfig.population_size must be > 0")
        if self.num_generations <= 0:
            raise ValueError("EvosaxTrainingConfig.num_generations must be > 0")


def _build_single_eval(
    *,
    static_predictor: Any,
    unflatten: Callable[[Array], Any],
    bucket_payloads: tuple[BucketPayload, ...],
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
    penalty_grids: tuple[Array, ...],
    penalty_weight: float,
) -> Callable[[Array], Array]:
    """Return the per-individual loss closure ``single_eval(flat) -> scalar``.

    Closes over everything that cannot pass through ``vmap``: the static
    partition, the ``ravel_pytree`` unflatten closure, the bucket
    payloads, ``simulate_fn`` and ``state_to_output``, the solver config
    and the resolved loss. The bucket loop unrolls inside the trace, so
    the multi-bucket forward pass becomes one fused kernel.

    Loss aggregation across buckets is a **simple sum**, so a bucket with
    more experiments weighs more. That matches how the same dataset scores
    end to end, which keeps the ranking honest.
    """

    def single_eval(flat: Array) -> Array:
        params = unflatten(flat)
        predictor = eqx.combine(params, static_predictor)

        def per_experiment(ts: Array, covariates: dict[str, Array], y0: Array) -> Array:
            full_state = simulate_fn(predictor, ts, covariates, y0, solver)
            return state_to_output(full_state)

        total = jnp.asarray(0.0)
        for bp in bucket_payloads:
            pred_obs = jax.vmap(per_experiment, in_axes=(0, 0, 0))(bp.ts, bp.covariates, bp.y0)
            total = total + loss_fn(pred_obs, bp)
        if penalty_weight > 0.0:
            # CMA-ES searches the latent space with nothing holding it in
            # range, and the squash keeps the physical output legal however
            # far the latent drifts, so an individual parked deep in
            # saturation looks mediocre rather than broken. Charging
            # saturation makes the search prefer individuals with gradient
            # left, which matters if optax polishes the result later.
            #
            # Folded into the fitness because evosax ranks by one scalar
            # with no aux channel. The weight is a closed-over Python float,
            # never traced, so it stays out of the vmap.
            total = total + penalty_weight * bound_penalty(predictor, penalty_grids)
        return total

    return single_eval


def _build_strategy(
    *,
    config: EvosaxTrainingConfig,
    flat: Array,
) -> tuple[Any, Any]:
    """Instantiate the evosax strategy and return ``(strategy, params)``.

    Only ``CMA_ES`` is wired in. The returned ``params`` are CMA-ES's
    frozen hyperparameters with ``std_init`` set to ``config.sigma_init``.
    """
    if config.algorithm != "CMA_ES":  # defensive; __post_init__ rejects others
        raise ValueError(f"Unsupported algorithm: {config.algorithm!r}")
    strategy = CMA_ES(population_size=config.population_size, solution=flat)
    params = dataclasses.replace(strategy.default_params, std_init=config.sigma_init)
    return strategy, params


def _box_population(
    *,
    flat: Array,
    config: EvosaxTrainingConfig,
    key: Array,
) -> Array:
    """Build the gen-0 population for ``"uniform_box"`` / ``"lhs_box"`` init modes.

    Both modes return ``flat[None, :] + offsets`` of shape
    ``(population_size, n_params)``. ``"uniform_box"`` draws offsets
    i.i.d. from ``Uniform(-extent, +extent)``; ``"lhs_box"`` rescales
    ``scipy.stats.qmc.LatinHypercube`` samples to the same range,
    host-side because JAX has no LHS sampler. Its seed is derived from
    ``key``, so the same root key reproduces the same sample.
    """
    n_params = int(flat.shape[0])
    extent = float(config.init_box_extent)
    if config.init == "uniform_box":
        offsets = jr.uniform(
            key,
            shape=(config.population_size, n_params),
            minval=-extent,
            maxval=extent,
        )
        return flat[None, :] + offsets
    if config.init == "lhs_box":
        # SciPy QMC takes an int seed, so derive one host-side from the key.
        seed_arr = jr.bits(key, shape=(), dtype=jnp.uint32)
        sampler = qmc.LatinHypercube(d=n_params, seed=int(seed_arr))
        sample = sampler.random(n=config.population_size)  # shape [pop, n], [0, 1]
        scaled = (sample * 2.0 - 1.0) * extent
        return flat[None, :] + jnp.asarray(scaled, dtype=flat.dtype)
    # The "warm" mode is handled by the caller (CMA-ES's own ask); we should
    # never be invoked with it.
    raise ValueError(f"_box_population called with non-box init {config.init!r}")


def _select_ui(ui: EvosaxUI | None, verbose: bool) -> EvosaxUI:
    """Pick the concrete UI. An explicit ``ui`` always wins, and otherwise
    ``verbose`` toggles between ``RichEvosaxUI`` and ``SilentUI``."""
    if ui is not None:
        return ui
    return RichEvosaxUI() if verbose else SilentUI()


def train_with_evosax(
    predictors: Any,
    dataset: Dataset,
    config: EvosaxTrainingConfig,
    *,
    simulate_fn: Callable[..., Array],
    solver: SolverConfig,
    trainable: Any = None,
    key: Array,
    ui: EvosaxUI | None = None,
) -> tuple[list[float], Any]:
    """Train ``predictors`` against ``dataset`` with an evolutionary strategy.

    Runs ``config.num_generations`` generations of CMA-ES over a
    population of ``config.population_size`` candidates, taking no
    gradient. See the module docstring for the JIT boundary and the
    initial-population modes.

    ``predictors`` is a ``PyTree[eqx.Module]`` in any shape. ``key`` is
    keyword-only and required. ``trainable`` defaults to
    :func:`hybridmodels.trainable.trainable_mask`, and must select at
    least one scalar.

    Returns
    -------
    history : list[float]
        **Best loss so far** at the end of each generation, so the series
        is monotone non-increasing. Length ``config.num_generations``.

        It differs from
        :func:`~hybridmodels.training.optax.train_with_optax`, whose
        history is the raw per-step loss and can go up. Same type, same
        position, different meaning: plotting both on one axis misleads.

        When ``config.penalty_weight > 0`` the recorded value is the
        combined objective, since evosax ranks by one scalar. The optax
        history excludes its penalty.
    best_predictors : Any
        Predictors rebuilt from the lowest-loss flat vector seen in any
        generation, the warm-up evaluation of the input predictors
        included. Same container shape as the input.
    """
    if trainable is None:
        trainable = trainable_mask(predictors)

    bucket_payloads = dataset.bucket_payloads
    if not bucket_payloads:
        raise ValueError("train_with_evosax: dataset has no bucket payloads")

    loss_fn = _resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights)
    state_to_output = dataset.state_to_output

    params_pytree, static_predictors = eqx.partition(predictors, trainable)
    flat0, unflatten = jfu.ravel_pytree(params_pytree)
    if flat0.size == 0:
        raise ValueError(
            "train_with_evosax: trainable mask selected zero parameters; "
            "evosax requires at least one trainable scalar."
        )

    ui_ = _select_ui(ui, config.verbose)
    ui_.on_run_start(
        num_generations=int(config.num_generations),
        population_size=int(config.population_size),
    )

    single_eval = _build_single_eval(
        static_predictor=static_predictors,
        unflatten=unflatten,
        bucket_payloads=bucket_payloads,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        penalty_grids=collocation_grids(predictors, config.penalty_grid_points),
        penalty_weight=float(config.penalty_weight),
    )
    population_eval = eqx.filter_jit(jax.vmap(single_eval))

    # ``population_eval`` is one fused vmap+jit callable across all buckets,
    # so there is no per-bucket compile span to emit as the Optax loop does.
    # One synthetic span over a "union" shape keeps the UI contract
    # symmetric and the Rich progress bars rendering.
    union_shape = (int(config.population_size), int(flat0.shape[0]))
    ui_.on_compile_start(bucket_idx=0, bucket_shape=union_shape)
    warm_pop = jnp.broadcast_to(flat0[None, :], (config.population_size, flat0.shape[0]))
    warm_fitness = population_eval(warm_pop)
    jax.block_until_ready(warm_fitness)  # type: ignore[no-untyped-call]
    ui_.on_compile_done(bucket_idx=0)

    strategy, strat_params = _build_strategy(config=config, flat=flat0)
    init_key = fold(key, "evosax_init")
    state = strategy.init(init_key, flat0, strat_params)

    # Best-ever bookkeeping starts from the warm-up evaluation, so an
    # unlucky initial population cannot replace a perfectly good seed.
    best_idx = int(jnp.argmin(warm_fitness))
    best_loss = float(warm_fitness[best_idx])
    best_flat = jnp.asarray(warm_pop[best_idx])

    history: list[float] = []

    for gen in range(int(config.num_generations)):
        ask_key = fold(key, f"evosax_ask_{gen}")
        if gen == 0 and config.init in ("uniform_box", "lhs_box"):
            # Inject the box-init population directly, skipping CMA-ES's
            # first ``ask``. ``tell`` still consumes it below, so mean and
            # covariance update from the prescribed sample.
            population = _box_population(flat=flat0, config=config, key=fold(key, "evosax_init"))
        else:
            population, state = strategy.ask(ask_key, state, strat_params)

        fitness = population_eval(population)
        # CMA-ES tolerates NaN in the fitness vector: a NaN just makes that
        # individual the worst in the generation. Per-individual error
        # recovery is deliberately absent; wrap your own simulator for it.
        tell_key = fold(key, f"evosax_tell_{gen}")
        state, _metrics = strategy.tell(tell_key, population, fitness, state, strat_params)

        gen_best_idx = int(jnp.argmin(fitness))
        gen_best_loss = float(fitness[gen_best_idx])
        gen_mean_loss = float(jnp.mean(fitness))
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_flat = jnp.asarray(population[gen_best_idx])

        history.append(best_loss)
        ui_.on_generation_end(
            gen_idx=gen,
            best_fitness=gen_best_loss,
            mean_fitness=gen_mean_loss,
        )

    best_predictors = eqx.combine(unflatten(best_flat), static_predictors)
    ui_.on_run_end(best_fitness=best_loss)
    return history, best_predictors
