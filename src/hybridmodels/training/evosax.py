"""Evosax-driven training loop for small-parameter predictors.

Implements SPEC §5.8 / R-E1..R-E6 / R-J1..R-J3 / R-R1..R-R3 / R-A2 / R-L1.
Targeted at kinetic predictors (~4-10 dims) per R-E2; not optimised for
neural-network-sized search.

JIT boundary (R-E3 / R-J3)
--------------------------
``single_eval(flat_params)`` rebuilds the predictor inside the trace via
``unflatten(flat) -> params; eqx.combine(params, static)`` and runs the bucket
dispatch loop (Python ``for`` over ``dataset.bucket_payloads``) inside the
traced region. ``population_eval = eqx.filter_jit(jax.vmap(single_eval))``
maps that across the population. ``static`` is closed over at construction
time because callables (``simulate_fn``, ``state_to_output``, the inner
modules' static fields) cannot pass through ``vmap``.

Init modes (R-E5)
-----------------
* ``"warm"`` — CMA-ES starts with ``mean = flat`` and ``std = sigma_init``;
  the strategy's first ``ask`` produces the initial population.
* ``"uniform_box"`` — a per-individual ``flat + Uniform(-extent, extent)``
  is constructed host-side and **evaluated directly in generation 0**,
  bypassing CMA-ES's first ask. The CMA-ES strategy is still initialised
  with ``mean = flat`` so the subsequent ``tell`` updates its mean / cov
  consistently with the rest of the run.
* ``"lhs_box"`` — same as ``uniform_box`` but the offsets come from
  ``scipy.stats.qmc.LatinHypercube`` host-side (``scipy`` is a hard
  dependency). Skipping the first ``ask`` means the prescribed sampling
  pattern actually reaches generation 0 — feeding LHS samples to ``tell``
  alone would be silently overwritten by the next ``ask``.

Best-ever tracking (R-E6)
-------------------------
Each generation's ``argmin(fitness)`` is compared host-side against the
running best loss; the corresponding flat-parameter vector is kept and
reconstructed only at run end. We do **not** consume CMA-ES's
``state.best_solution`` field because we want a uniform contract regardless
of which strategy is plugged in.
"""

# ruff: noqa: F722

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

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
from hybridmodels.rng import fold
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.ui.base import EvosaxUI, SilentUI
from hybridmodels.ui.evosax import RichEvosaxUI

_SUPPORTED_INIT_MODES: tuple[str, ...] = ("warm", "uniform_box", "lhs_box")
_SUPPORTED_ALGORITHMS: tuple[str, ...] = ("CMA_ES",)


@dataclass(frozen=True)
class EvosaxTrainingConfig:
    """Configuration for ``train_with_evosax`` (SPEC §5.8).

    All phase semantics from optax (steps/lr/optimizer tuples) are absent here:
    evosax is a flat outer loop of ``num_generations`` over a population of
    ``population_size`` individuals.

    Attributes
    ----------
    algorithm
        Evosax strategy name. Only ``"CMA_ES"`` is supported in v1; the field
        exists to keep the surface symmetric with the source package and to
        accept future additions without an API break.
    population_size, num_generations
        Outer-loop dimensions. Population is evaluated in parallel via
        ``vmap``; generations are sequential.
    init
        Initial-population scheme — see module docstring R-E5 details.
    init_box_extent
        Half-width of the box for ``"uniform_box"`` and ``"lhs_box"``. Ignored
        for ``"warm"``.
    sigma_init
        Initial CMA-ES step size. Used in ``"warm"`` and as the strategy's
        prior step size for the box-init modes (CMA-ES adapts it after the
        first ``tell``).
    loss
        Either a ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``,
        ``"bal_mle"``) or a callable matching the
        ``loss(pred_obs, bp) -> scalar`` contract from R-L1.
    channel_idx, channel_weights
        Forwarded into the resolved loss; see ``hybridmodels.losses``.
    log_every
        UI heartbeat cadence (currently honoured only by Rich UIs; the silent
        / recording UIs see every generation).
    verbose
        Selects ``RichEvosaxUI`` vs ``SilentUI`` when ``ui=None``. Explicit
        ``ui=...`` always wins (R-U2).
    """

    algorithm: str = "CMA_ES"
    population_size: int = 64
    num_generations: int = 100
    init: Literal["warm", "uniform_box", "lhs_box"] = "warm"
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

    Mirrors the same-named helper in ``training/optax.py`` (kept duplicated
    rather than extracted: only two call sites and the spec deliberately
    keeps this slim).
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
) -> Callable[[Array], Array]:
    """Return ``single_eval(flat) -> scalar`` — the per-individual loss closure.

    Closes over everything that cannot pass through ``vmap`` cleanly: the
    static partition of the predictor, the unflatten function (a Python
    closure produced by ``ravel_pytree``), the bucket payloads, the
    user-written ``simulate_fn`` / ``state_to_output``, the solver config,
    and the resolved loss function. The bucket dispatch loop unrolls inside
    the trace per R-E3 / R-J2, producing one fused kernel for the whole
    multi-bucket forward pass after ``vmap`` + ``jit``.

    Loss aggregation is a **simple sum** across buckets (matching the
    convention spelled out in the build-plan task brief): longer datasets
    weight more heavily, which keeps the relative ranking of individuals
    consistent with how the same dataset would be scored end-to-end.
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
        return total

    return single_eval


def _build_strategy(
    *,
    config: EvosaxTrainingConfig,
    flat: Array,
) -> tuple[Any, Any]:
    """Instantiate the evosax strategy and return ``(strategy, params)``.

    Only ``CMA_ES`` is wired in v1; new algorithms slot in as additional
    branches (or a small registry) once R-E1 grows beyond the kinetic-only
    use case. The returned ``params`` are CMA-ES's frozen hyperparams with
    ``std_init`` overridden to ``config.sigma_init``.
    """
    if config.algorithm != "CMA_ES":  # defensive — __post_init__ rejects others
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
    ``(population_size, n_params)``. ``"uniform_box"`` draws each offset i.i.d.
    from ``Uniform(-extent, +extent)``. ``"lhs_box"`` runs
    ``scipy.stats.qmc.LatinHypercube`` host-side (R-E5 explicitly mandates
    SciPy here — JAX has no LHS sampler) and rescales from ``[0, 1]`` to
    ``[-extent, +extent]``.

    The LHS seed is derived from ``key`` via a 32-bit unsigned hash so the
    same root key produces the same LHS sample across runs.
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
        # SciPy QMC takes an int seed — derive one host-side from the key.
        seed_arr = jr.bits(key, shape=(), dtype=jnp.uint32)
        sampler = qmc.LatinHypercube(d=n_params, seed=int(seed_arr))
        sample = sampler.random(n=config.population_size)  # shape [pop, n], [0, 1]
        scaled = (sample * 2.0 - 1.0) * extent
        return flat[None, :] + jnp.asarray(scaled, dtype=flat.dtype)
    # The "warm" mode is handled by the caller (CMA-ES's own ask); we should
    # never be invoked with it.
    raise ValueError(f"_box_population called with non-box init {config.init!r}")


def _initial_population(
    predictor: Any,
    config: EvosaxTrainingConfig,
    *,
    trainable: Any,
    key: Array,
) -> Array:
    """Return the population evaluated in generation 0 (R-E5).

    Exposed as a module-private helper rather than buried in
    ``train_with_evosax`` so tests can pin the spread of each init mode
    without driving a full training loop.

    Shape: ``(config.population_size, n_params)`` where ``n_params`` is the
    flattened-trainable dimension.
    """
    params, _static = eqx.partition(predictor, trainable)
    flat, _unflatten = jfu.ravel_pytree(params)

    if config.init == "warm":
        # CMA-ES's own first ask: mean=flat, std=sigma_init. The state must be
        # initialised with the same std-overridden params or the first draw
        # would silently use std=1.0.
        strategy, params_ = _build_strategy(config=config, flat=flat)
        init_key = fold(key, "evosax_init")
        state = strategy.init(init_key, flat, params_)
        ask_key = fold(key, "evosax_ask_0")
        pop, _new_state = strategy.ask(ask_key, state, params_)
        return cast(Array, pop)
    return _box_population(flat=flat, config=config, key=fold(key, "evosax_init"))


def _select_ui(ui: EvosaxUI | None, verbose: bool) -> EvosaxUI:
    """Pick the concrete UI: explicit ``ui`` wins, else verbose toggles Rich/Silent (R-U2)."""
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
    """Train ``predictors`` against ``dataset`` with an evolutionary strategy (SPEC §5.8).

    ``predictors`` is a ``PyTree[eqx.Module]`` (R-A2 / ADR-0006); the canonical
    convention is a tuple of ``BoundedPredictor`` leaves. Required keyword-only
    ``key`` (R-R1); calling without it raises ``TypeError`` before any work
    happens. ``trainable`` defaults to :func:`hybridmodels.trainable.trainable_mask`
    (every inexact-array leaf).

    Returns
    -------
    history : list[float]
        Best-loss-so-far per generation (length ``config.num_generations``).
    best_predictors : Any
        The reconstructed predictors pytree whose flat-parameter vector
        minimised the loss across every generation (R-E6). Same container
        shape as the input ``predictors``.
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
    )
    population_eval = eqx.filter_jit(jax.vmap(single_eval))

    # Compile-event lifecycle (R-U3): per-bucket-shape isn't directly visible
    # here because we vmap+jit a single fused callable across all buckets.
    # We emit one synthetic compile span over the union shape so the UI gets
    # the same on_compile_start / on_compile_done bracket the optax loop
    # produces — useful for Rich progress bars and load-bearing in test
    # contracts (test_recording_ui_lifecycle_events_fire).
    union_shape = (int(config.population_size), int(flat0.shape[0]))
    ui_.on_compile_start(bucket_idx=0, bucket_shape=union_shape)
    warm_pop = jnp.broadcast_to(flat0[None, :], (config.population_size, flat0.shape[0]))
    warm_fitness = population_eval(warm_pop)
    jax.block_until_ready(warm_fitness)  # type: ignore[no-untyped-call]
    ui_.on_compile_done(bucket_idx=0)

    strategy, strat_params = _build_strategy(config=config, flat=flat0)
    init_key = fold(key, "evosax_init")
    state = strategy.init(init_key, flat0, strat_params)

    # Best-ever bookkeeping (R-E6). Compare against the warm-up evaluation so
    # the user-supplied predictor stays in contention even if every sampled
    # individual is worse.
    best_idx = int(jnp.argmin(warm_fitness))
    best_loss = float(warm_fitness[best_idx])
    best_flat = jnp.asarray(warm_pop[best_idx])

    history: list[float] = []

    for gen in range(int(config.num_generations)):
        ask_key = fold(key, f"evosax_ask_{gen}")
        if gen == 0 and config.init in ("uniform_box", "lhs_box"):
            # Inject the box-init population directly (see R-E5 / module doc).
            # CMA-ES's first ask is skipped; the strategy's tell still consumes
            # the same population, so its mean and covariance update from the
            # prescribed sample rather than from a Gaussian draw.
            population = _box_population(flat=flat0, config=config, key=fold(key, "evosax_init"))
        else:
            population, state = strategy.ask(ask_key, state, strat_params)

        fitness = population_eval(population)
        # CMA-ES expects a JAX array but tolerates NaN; we let the user's
        # error-mode policy bubble up unchanged (R-E2 / out-of-scope:
        # per-individual graceful error handling).
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
