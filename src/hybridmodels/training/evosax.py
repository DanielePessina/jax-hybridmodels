"""Gradient-free training loop for small-parameter predictors, driven by evosax.

Evolutionary search instead of gradient descent. Each generation samples
a *population* of candidate parameter vectors, scores every one of them,
and lets the strategy (CMA-ES) move its sampling distribution towards
the good ones. No derivative of the loss is ever taken, which is what
makes it useful when the ODE adjoint is unreliable or the loss surface
is full of local minima.

The cost is that the number of evaluations needed grows quickly with the
number of parameters. This loop targets small kinetic predictors, a
handful of trainable scalars, roughly 4 to 10 dimensions. Use the Optax
loop in ``hybridmodels.training.optax`` for neural-network-sized fits,
or use this first and polish with Optax afterwards.

There are no phases here. The run is ``num_generations`` iterations over
a population of ``population_size`` individuals.

JIT boundary
------------
The trainable parameters are first ravelled to a single flat vector
``flat`` (via ``jax.flatten_util.ravel_pytree``). The per-individual
loss callable ``single_eval(flat) -> scalar`` then:

1. ``unflatten(flat)`` rebuilds the dynamic part of the predictor pytree;
2. ``eqx.combine(params, static)`` glues it back to the static part
   (closed over at construction time, since callables and static
   fields cannot pass through ``vmap``);
3. the bucket dispatch loop (Python ``for`` over
   ``dataset.bucket_payloads``) runs inside the traced region, so the
   whole multi-bucket forward pass becomes a single fused kernel
   after ``vmap`` and ``jit``.

``population_eval = eqx.filter_jit(jax.vmap(single_eval))`` then
evaluates the entire population in parallel.

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
    a hard dependency). Skipping the first ``ask`` is required here, not
    just convenient. A prescribed LHS pattern reaches generation 0 only
    by direct injection; handing LHS samples to ``tell`` alone would let
    the next ``ask`` overwrite them silently.

Best-ever tracking
------------------
Each generation's ``argmin(fitness)`` is compared host-side against the
running best loss. The winning flat-parameter vector is kept and turned
back into predictors only at run end. CMA-ES's own
``state.best_solution`` field is deliberately unused, so the contract
stays the same whichever strategy is plugged in.
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
from hybridmodels.losses import LOSS_REGISTRY
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

    There are no phase-keyed tuples here, unlike
    :class:`~hybridmodels.training.optax.OptaxTrainingConfig`. The run is
    a flat loop of ``num_generations`` over a population of
    ``population_size`` individuals, so every field is a scalar.

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


def _resolve_loss_fn(
    loss: Callable[..., Array] | str,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> Callable[[Array, BucketPayload], Array]:
    """Resolve a string-or-callable loss spec into a ``(pred_obs, bp) -> scalar``.

    Copy of the same-named helper in ``training/optax.py``. Kept
    duplicated rather than extracted, because there are only two call
    sites.
    """
    if isinstance(loss, str):
        key = loss.lower().strip()
        if key not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss name {loss!r}; available: {sorted(LOSS_REGISTRY)}")
        base = LOSS_REGISTRY[key]
    else:
        base = loss
    if channel_idx is None and channel_weights is None:
        return base

    def loss_fn(pred_obs: Array, bp: BucketPayload) -> Array:
        return base(pred_obs, bp, channel_idx=channel_idx, channel_weights=channel_weights)

    return loss_fn


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

    Closes over everything that cannot pass through ``vmap`` cleanly.
    That is the static partition of the predictor, the unflatten function
    (a Python closure produced by ``ravel_pytree``), the bucket payloads,
    the user-written ``simulate_fn`` and ``state_to_output``, the solver
    config, and the resolved loss. The bucket dispatch loop unrolls
    inside the trace, so the whole multi-bucket forward pass becomes one
    fused kernel after ``vmap`` and ``jit``.

    Loss aggregation across buckets is a **simple sum**, so a bucket
    holding more experiments weighs more in an individual's fitness. That
    matches how the same dataset scores end to end, which keeps the
    ranking of individuals honest.
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
            # range. BoundedPredictor's squash keeps the *physical* output
            # legal however far the latent drifts, so an individual parked
            # deep in saturation just looks mediocre rather than broken.
            # Charging saturation gives the search a reason to prefer
            # individuals that still have gradient left, which matters when
            # the result is later polished with optax.
            #
            # Folded into the fitness rather than reported next to it,
            # because evosax ranks individuals by one scalar and there is
            # no aux channel to separate the terms into. The weight is a
            # closed-over Python float rather than a traced value: it never
            # changes within a run, and closing over it keeps it out of the
            # vmap.
            total = total + penalty_weight * bound_penalty(predictor, penalty_grids)
        return total

    return single_eval


def _build_strategy(
    *,
    config: EvosaxTrainingConfig,
    flat: Array,
) -> tuple[Any, Any]:
    """Instantiate the evosax strategy and return ``(strategy, params)``.

    Only ``CMA_ES`` is wired in. A new algorithm slots in as another
    branch, or a small registry, once a use case calls for one. The
    returned ``params`` are CMA-ES's frozen hyperparameters with
    ``std_init`` overridden to ``config.sigma_init``.
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
    ``(population_size, n_params)``. ``"uniform_box"`` draws each offset
    i.i.d. from ``Uniform(-extent, +extent)``. ``"lhs_box"`` uses
    ``scipy.stats.qmc.LatinHypercube`` and rescales its ``[0, 1]``
    samples to ``[-extent, +extent]``. That runs host-side because JAX
    has no LHS sampler, which makes SciPy a hard dependency of this mode.

    The LHS seed is derived from ``key`` via a 32-bit unsigned hash so
    the same root key reproduces the same LHS sample across runs.
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
    population of ``config.population_size`` candidate parameter vectors.
    No gradient of the loss is taken. See the module docstring for the
    JIT boundary and the initial-population modes.

    ``predictors`` is a ``PyTree[eqx.Module]``. The convention is a tuple
    of ``BoundedPredictor`` leaves, but any pytree shape works. ``key``
    is keyword-only and required; calling without it raises
    ``TypeError`` before any work happens. ``trainable`` defaults to
    :func:`hybridmodels.trainable.trainable_mask` over the supplied
    pytree, marking every inexact-array leaf trainable. The mask has to
    select at least one scalar.

    Returns
    -------
    history : list[float]
        **Best loss so far** at the end of each generation, so the series
        is monotone non-increasing. Length ``config.num_generations``.

        This differs from
        :func:`~hybridmodels.training.optax.train_with_optax`, whose
        history is the raw per-step loss and can go up. Same type, same
        position in the return tuple, different meaning. Plotting the two
        together, or feeding both to a shared stopping rule, will
        mislead.

        When ``config.penalty_weight > 0`` the recorded value is the
        combined objective, because evosax ranks individuals by one
        scalar and the terms are never separated. The optax history
        excludes its penalty.
    best_predictors : Any
        The predictors rebuilt from the flat parameter vector with the
        lowest loss seen in any generation, including the warm-up
        evaluation of the input predictors. Same container shape as the
        input ``predictors``.
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

    # The Optax loop's UI sees one compile span per bucket shape. Here
    # ``population_eval`` is a single fused vmap+jit callable across all
    # buckets, so there is no per-bucket span to emit. Emit one synthetic
    # span over a "union" shape instead, so the Rich UI's progress bars
    # still render and the on_compile_start / on_compile_done contract
    # stays symmetric with the Optax loop.
    union_shape = (int(config.population_size), int(flat0.shape[0]))
    ui_.on_compile_start(bucket_idx=0, bucket_shape=union_shape)
    warm_pop = jnp.broadcast_to(flat0[None, :], (config.population_size, flat0.shape[0]))
    warm_fitness = population_eval(warm_pop)
    jax.block_until_ready(warm_fitness)  # type: ignore[no-untyped-call]
    ui_.on_compile_done(bucket_idx=0)

    strategy, strat_params = _build_strategy(config=config, flat=flat0)
    init_key = fold(key, "evosax_init")
    state = strategy.init(init_key, flat0, strat_params)

    # Best-ever bookkeeping starts from the warm-up evaluation, so the
    # user-supplied predictor stays in contention even if every sampled
    # individual turns out worse. Without that, an unlucky initial
    # population would replace a perfectly good seed.
    best_idx = int(jnp.argmin(warm_fitness))
    best_loss = float(warm_fitness[best_idx])
    best_flat = jnp.asarray(warm_pop[best_idx])

    history: list[float] = []

    for gen in range(int(config.num_generations)):
        ask_key = fold(key, f"evosax_ask_{gen}")
        if gen == 0 and config.init in ("uniform_box", "lhs_box"):
            # Inject the box-init population directly. CMA-ES's first
            # ``ask`` is skipped, but the strategy's ``tell`` still
            # consumes the same population below, so its mean and
            # covariance update from the prescribed sample rather than
            # from a Gaussian draw the user did not request.
            population = _box_population(flat=flat0, config=config, key=fold(key, "evosax_init"))
        else:
            population, state = strategy.ask(ask_key, state, strat_params)

        fitness = population_eval(population)
        # CMA-ES tolerates NaN in the fitness vector, so nothing scrubs or
        # replaces it here. A NaN just makes that individual the worst in
        # the generation. Per-individual error recovery, such as catching
        # exceptions raised inside simulate_fn, is deliberately absent; a
        # user who wants it wraps their own simulator.
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
