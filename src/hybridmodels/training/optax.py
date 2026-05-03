"""Optax-driven training loop for hybrid mechanistic models.

Implements SPEC §5.7 / R-T1..R-T8 / R-J1..R-J3 / R-R1..R-R3 / R-A2 / R-L1.
A training step is one full pass over every bucket, accumulating gradients,
followed by a single ``optimizer.update``. Per-phase learning rate, optimizer
type, length-schedule mask and optimiser-state reset are configured via
``OptaxTrainingConfig`` (every phase-keyed field is a same-length tuple). The
shared tournament reuses the main loop's compiled ``make_step`` and
``apply_update`` (ADR-0002).
"""

# ruff: noqa: F722

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from hybridmodels.data import BucketPayload, Dataset
from hybridmodels.losses import LOSS_REGISTRY
from hybridmodels.predictors.base import Predictor, reinitialize_with_key
from hybridmodels.rng import fold
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.ui.base import SilentUI, TrainingUI

_PHASE_KEYED_FIELDS: tuple[str, ...] = (
    "lr",
    "optimizer",
    "reset_optimiser_state",
    "length_schedule",
)


@dataclass(frozen=True)
class OptaxTrainingConfig:
    steps: tuple[int, ...]
    lr: tuple[float, ...]
    optimizer: tuple[str, ...]
    reset_optimiser_state: tuple[bool, ...]
    length_schedule: tuple[float, ...] = (1.0,)
    loss: Callable[..., Array] | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    tournament_attempts: int = 1
    tournament_steps: int = 0
    tournament_lr: float = 1e-4
    log_every: int = 10
    patience: int = 0
    restore_best: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        n = len(self.steps)
        if n == 0:
            raise ValueError("OptaxTrainingConfig.steps must contain at least one phase")
        for name in _PHASE_KEYED_FIELDS:
            value = getattr(self, name)
            if len(value) != n:
                raise ValueError(
                    f"OptaxTrainingConfig: phase-keyed field {name!r} has length "
                    f"{len(value)}, expected {n} (matching steps)"
                )
        for fraction in self.length_schedule:
            f = float(fraction)
            if not (0.0 < f <= 1.0):
                raise ValueError(
                    "OptaxTrainingConfig.length_schedule entries must lie in (0, 1]; "
                    f"got {fraction}"
                )


def _build_optimizer(name: str, lr: float) -> optax.GradientTransformation:
    norm = name.lower().strip()
    if norm == "adamw":
        return optax.inject_hyperparams(optax.adamw)(learning_rate=lr)
    if norm == "adabelief":
        return optax.inject_hyperparams(optax.adabelief)(learning_rate=lr)
    raise ValueError(
        f"OptaxTrainingConfig.optimizer={name!r} is not supported; expected "
        "'adamw' or 'adabelief'."
    )


def _resolve_loss_fn(
    loss: Callable[..., Array] | str,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> Callable[[Array, BucketPayload], Array]:
    if isinstance(loss, str):
        key = loss.lower().strip()
        if key not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss name {loss!r}; available: {sorted(LOSS_REGISTRY)}"
            )
        base = LOSS_REGISTRY[key]
    else:
        base = loss
    if channel_idx is None and channel_weights is None:
        return base

    def loss_fn(pred_obs: Array, bp: BucketPayload) -> Array:
        return base(
            pred_obs, bp, channel_idx=channel_idx, channel_weights=channel_weights
        )

    return loss_fn


def _build_make_step(
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
    trainable: Any,
) -> Callable[[Predictor, BucketPayload, Array], tuple[Array, Any]]:
    def loss_eval(
        diff_predictor: Predictor,
        static_predictor: Predictor,
        bp_masked: BucketPayload,
    ) -> Array:
        predictor = eqx.combine(diff_predictor, static_predictor)

        def per_experiment(
            ts: Array, covariates: dict[str, Array], y0: Array
        ) -> Array:
            full_state = simulate_fn(predictor, ts, covariates, y0, solver)
            return state_to_output(full_state)

        pred_obs = jax.vmap(per_experiment, in_axes=(0, 0, 0))(
            bp_masked.ts, bp_masked.covariates, bp_masked.y0
        )
        return loss_fn(pred_obs, bp_masked)

    grad_fn = eqx.filter_value_and_grad(loss_eval)

    @eqx.filter_jit
    def make_step(
        predictor: Predictor, bp: BucketPayload, length_mask_fraction: Array
    ) -> tuple[Array, Any]:
        T = bp.ts.shape[1]
        cutoff = jnp.maximum(
            jnp.ceil(jnp.float32(T) * length_mask_fraction).astype(jnp.int32),
            jnp.int32(1),
        )
        sched_mask = (jnp.arange(T) < cutoff)[None, :, None]
        bp_masked = bp._replace(mask=bp.mask & sched_mask)
        diff_part, static_part = eqx.partition(predictor, trainable)
        loss, grads = grad_fn(diff_part, static_part, bp_masked)
        return loss, grads

    return make_step


def _build_apply_update(
    optimizer: optax.GradientTransformation, trainable: Any
) -> Callable[[Predictor, Any, Any], tuple[Predictor, Any]]:
    @eqx.filter_jit
    def apply_update(
        predictor: Predictor, grads: Any, opt_state: Any
    ) -> tuple[Predictor, Any]:
        params = eqx.filter(predictor, trainable)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_predictor = eqx.apply_updates(predictor, updates)
        return new_predictor, new_opt_state

    return apply_update


def _accumulate_step(
    predictor: Predictor,
    dataset: Dataset,
    make_step: Callable[..., tuple[Array, Any]],
    length_mask_fraction: Array,
    trainable: Any,
) -> tuple[Array, Any]:
    zero_grads = jax.tree.map(jnp.zeros_like, eqx.filter(predictor, trainable))
    acc_grads = zero_grads
    total_loss = jnp.asarray(0.0)
    n_batches = 0
    for bp in dataset.bucket_payloads:
        loss, grads = make_step(predictor, bp, length_mask_fraction)
        acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
        total_loss = total_loss + loss
        n_batches += 1
    denom = float(max(n_batches, 1))
    avg_grads = jax.tree.map(lambda g: g / denom, acc_grads)
    avg_loss = total_loss / denom
    return avg_loss, avg_grads


def _shared_tournament(
    predictor: Predictor,
    dataset: Dataset,
    *,
    make_step: Callable[..., tuple[Array, Any]],
    apply_update: Callable[..., tuple[Predictor, Any]],
    optimizer: optax.GradientTransformation,
    trainable: Any,
    tournament_attempts: int,
    tournament_steps: int,
    tournament_lr: float,
    key: Array,
    length_mask_fraction: Array,
) -> Predictor:
    for attempt in range(tournament_attempts):
        attempt_key = fold(key, f"tournament_attempt_{attempt}")
        try:
            candidate: Predictor = reinitialize_with_key(predictor, attempt_key)  # type: ignore[assignment]
            opt_state = optimizer.init(eqx.filter(candidate, trainable))
            opt_state.hyperparams["learning_rate"] = jnp.asarray(tournament_lr)

            for _ in range(tournament_steps):
                _avg_loss, avg_grads = _accumulate_step(
                    candidate, dataset, make_step, length_mask_fraction, trainable
                )
                candidate, opt_state = apply_update(candidate, avg_grads, opt_state)

            score = jnp.asarray(0.0)
            for bp in dataset.bucket_payloads:
                bucket_loss, _ = make_step(candidate, bp, length_mask_fraction)
                score = score + bucket_loss
            score_value = float(score)
            if not math.isfinite(score_value):
                raise FloatingPointError(f"non-finite tournament loss: {score_value}")
            return candidate
        except Exception:
            continue

    warnings.warn(
        "tournament: all attempts failed; falling back to initial predictor",
        RuntimeWarning,
        stacklevel=2,
    )
    return predictor


def _select_ui(ui: TrainingUI | None, verbose: bool) -> TrainingUI:
    if ui is not None:
        return ui
    # TODO Phase 10: switch to RichTrainingUI when verbose=True.
    # why: RichTrainingUI is implemented in Phase 10; until then both verbose
    # branches reduce to silent.
    return SilentUI()


def train_with_optax(
    predictor: Predictor,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    simulate_fn: Callable[..., Array],
    solver: SolverConfig,
    trainable: Any = None,
    key: Array,
    ui: TrainingUI | None = None,
) -> tuple[list[float], Predictor]:
    """Train ``predictor`` against ``dataset`` with Optax (SPEC §5.7).

    ``key`` is required keyword-only (R-R1); calling without it raises
    ``TypeError`` before any compilation. ``trainable`` defaults to
    :func:`hybridmodels.trainable.trainable_mask` (every inexact-array leaf).
    """
    if trainable is None:
        trainable = trainable_mask(predictor)

    bucket_payloads = dataset.bucket_payloads
    if not bucket_payloads:
        raise ValueError("train_with_optax: dataset has no bucket payloads")

    loss_fn = _resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights)
    state_to_output = dataset.state_to_output

    ui_ = _select_ui(ui, config.verbose)

    make_step = _build_make_step(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trainable=trainable,
    )

    n_phases = len(config.steps)
    total_steps = int(sum(config.steps))
    ui_.on_run_start(total_steps=total_steps, num_phases=n_phases)

    full_mask = jnp.asarray(1.0)
    for idx, bp in enumerate(bucket_payloads):
        bucket_shape = (int(bp.ts.shape[0]), int(bp.ts.shape[1]))
        ui_.on_compile_start(bucket_idx=idx, bucket_shape=bucket_shape)
        warm_loss, _ = make_step(predictor, bp, full_mask)
        jax.block_until_ready(warm_loss)  # type: ignore[no-untyped-call]
        ui_.on_compile_done(bucket_idx=idx)
        ui_.on_compile_progress(bucket_idx=idx, total_buckets=len(bucket_payloads))

    optimizer = _build_optimizer(config.optimizer[0], config.lr[0])
    apply_update = _build_apply_update(optimizer, trainable)

    if config.tournament_attempts > 1 and config.tournament_steps > 0:
        tournament_root = fold(key, "tournament")
        predictor = _shared_tournament(
            predictor,
            dataset,
            make_step=make_step,
            apply_update=apply_update,
            optimizer=optimizer,
            trainable=trainable,
            tournament_attempts=config.tournament_attempts,
            tournament_steps=config.tournament_steps,
            tournament_lr=config.tournament_lr,
            key=tournament_root,
            length_mask_fraction=full_mask,
        )

    opt_state = optimizer.init(eqx.filter(predictor, trainable))

    losses_history: list[float] = []
    best_loss = float("inf")
    best_predictor = predictor
    steps_since_improvement = 0

    for phase_idx, n_steps in enumerate(config.steps):
        if phase_idx > 0:
            if config.reset_optimiser_state[phase_idx]:
                optimizer = _build_optimizer(
                    config.optimizer[phase_idx], config.lr[phase_idx]
                )
                apply_update = _build_apply_update(optimizer, trainable)
                opt_state = optimizer.init(eqx.filter(predictor, trainable))
            else:
                opt_state.hyperparams["learning_rate"] = jnp.asarray(
                    config.lr[phase_idx]
                )

        length_mask_fraction = jnp.asarray(config.length_schedule[phase_idx])

        ui_.on_phase_start(
            phase_idx=phase_idx,
            phase_steps=int(n_steps),
            lr=float(config.lr[phase_idx]),
            optimizer=config.optimizer[phase_idx],
        )

        for step in range(int(n_steps)):
            avg_loss, avg_grads = _accumulate_step(
                predictor, dataset, make_step, length_mask_fraction, trainable
            )
            loss_value = float(avg_loss)
            losses_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_predictor = predictor
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            predictor, opt_state = apply_update(predictor, avg_grads, opt_state)

            ui_.on_step_end(step_idx=step, phase_idx=phase_idx, loss=loss_value)

            if config.patience > 0 and steps_since_improvement >= config.patience:
                break

        ui_.on_phase_end(phase_idx=phase_idx)

    final_predictor = best_predictor if config.restore_best else predictor
    final_loss = losses_history[-1] if losses_history else float("nan")
    ui_.on_run_end(final_loss=final_loss)
    return losses_history, final_predictor
