"""Optax-driven training loop for hybrid mechanistic models.

A training *step* here is one full pass over every bucket: the loop
computes per-bucket gradients via ``make_step`` (jitted, one trace per
bucket shape), accumulates and averages them across the dataset, and
applies a single ``optimizer.update`` via ``apply_update``. The
training run is divided into one or more *phases*; each phase has its
own learning rate, optimizer type, length-schedule mask, and optional
optimiser-state reset, all carried in ``OptaxTrainingConfig`` as
same-length tuples (one entry per phase).

Optionally the loop runs a *tournament* before the main phases: it
re-initialises the predictors several times under the same fresh key
discipline, takes a few short training steps with each candidate, and
keeps the candidate with the lowest training loss. The tournament is
intentionally implemented on top of the same compiled ``make_step`` and
``apply_update`` rather than as a parallel kernel — that way the
short-burst attempts share JIT cache entries with the main loop and pay
no extra compilation cost.
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
from hybridmodels.penalties import bound_penalty, collocation_grids
from hybridmodels.predictors.base import reinitialize_pytree_with_key
from hybridmodels.rng import fold
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.ui.base import SilentUI, TrainingUI
from hybridmodels.ui.optax import RichTrainingUI

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
    penalty_weight: tuple[float, ...] = (0.0,)
    penalty_grid_points: int = 5
    loss: Callable[..., Array] | str = "mse"
    channel_idx: tuple[int, ...] | None = None
    channel_weights: tuple[float, ...] | None = None
    tournament_attempts: int = 1
    tournament_steps: int = 0
    tournament_lr: float = 1e-4
    patience: int = 0
    restore_best: bool = True
    verbose: bool = True

    def penalty_weight_for_phase(self, phase_idx: int) -> float:
        """Penalty weight for ``phase_idx``, honouring the length-1 broadcast."""
        if len(self.penalty_weight) == 1:
            return float(self.penalty_weight[0])
        return float(self.penalty_weight[phase_idx])

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
        # penalty_weight is deliberately NOT in _PHASE_KEYED_FIELDS. R-T2
        # enumerates the fields that must be full-length tuples, and every
        # one of them lacks a safe default -- there is no "obvious" learning
        # rate, so demanding the user spell it out per phase is protection,
        # not ceremony. penalty_weight has an unambiguous off state, and
        # requiring `(0.0, 0.0)` from every multi-phase config that never
        # enables it would be ceremony. A length-1 tuple therefore
        # broadcasts; any other length must match `steps` exactly, so a
        # genuine per-phase schedule still cannot be silently truncated.
        if len(self.penalty_weight) not in (1, n):
            raise ValueError(
                "OptaxTrainingConfig.penalty_weight must have length 1 "
                f"(broadcast across phases) or {n} (one per phase); "
                f"got {len(self.penalty_weight)}"
            )
        for weight in self.penalty_weight:
            if float(weight) < 0.0:
                raise ValueError(
                    "OptaxTrainingConfig.penalty_weight entries must be non-negative; "
                    f"got {weight}"
                )
        if self.penalty_grid_points < 2:
            raise ValueError(
                "OptaxTrainingConfig.penalty_grid_points must be at least 2 "
                f"(one point per box edge); got {self.penalty_grid_points}"
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
        f"OptaxTrainingConfig.optimizer={name!r} is not supported; expected 'adamw' or 'adabelief'."
    )


def _resolve_loss_fn(
    loss: Callable[..., Array] | str,
    channel_idx: tuple[int, ...] | None,
    channel_weights: tuple[float, ...] | None,
) -> Callable[[Array, BucketPayload], Array]:
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


def _build_make_step(
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
    trainable: Any,
    penalty_grids: tuple[Array, ...],
) -> Callable[[Any, BucketPayload, Array, Array], tuple[Array, Array, Array, Any]]:
    def loss_eval(
        diff_predictors: Any,
        static_predictors: Any,
        bp_masked: BucketPayload,
        penalty_weight: Array,
    ) -> tuple[Array, tuple[Array, Array]]:
        # ``predictors`` here is whatever pytree the user passed in
        # (typically a tuple of BoundedPredictor leaves; could also be a
        # dict, NamedTuple, single Module, ...). ``eqx.combine`` walks
        # any pytree shape, so we never need to inspect the container —
        # we just hand the recombined pytree to the user's simulate_fn.
        predictors = eqx.combine(diff_predictors, static_predictors)

        def per_experiment(ts: Array, covariates: dict[str, Array], y0: Array) -> Array:
            full_state = simulate_fn(predictors, ts, covariates, y0, solver)
            return state_to_output(full_state)

        pred_obs = jax.vmap(per_experiment, in_axes=(0, 0, 0))(
            bp_masked.ts, bp_masked.covariates, bp_masked.y0
        )
        data = loss_fn(pred_obs, bp_masked)
        # The penalty reads the recombined pytree, so it sees frozen and
        # trainable leaves alike but only differentiates the former --
        # exactly the split ``eqx.partition`` already established above.
        # It does not depend on ``bp``: saturation is a property of the
        # predictor over its declared input box, not of any one bucket.
        penalty = bound_penalty(predictors, penalty_grids)
        return data + penalty_weight * penalty, (data, penalty)

    grad_fn = eqx.filter_value_and_grad(loss_eval, has_aux=True)

    @eqx.filter_jit
    def make_step(
        predictors: Any,
        bp: BucketPayload,
        length_mask_fraction: Array,
        penalty_weight: Array,
    ) -> tuple[Array, Array, Array, Any]:
        T = bp.ts.shape[1]
        cutoff = jnp.maximum(
            jnp.ceil(jnp.float32(T) * length_mask_fraction).astype(jnp.int32),
            jnp.int32(1),
        )
        sched_mask = (jnp.arange(T) < cutoff)[None, :, None]
        bp_masked = bp._replace(mask=bp.mask & sched_mask)
        diff_part, static_part = eqx.partition(predictors, trainable)
        (total, (data, penalty)), grads = grad_fn(
            diff_part, static_part, bp_masked, penalty_weight
        )
        return total, data, penalty, grads

    return make_step


def _build_apply_update(
    optimizer: optax.GradientTransformation, trainable: Any
) -> Callable[[Any, Any, Any], tuple[Any, Any]]:
    @eqx.filter_jit
    def apply_update(predictors: Any, grads: Any, opt_state: Any) -> tuple[Any, Any]:
        params = eqx.filter(predictors, trainable)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_predictors = eqx.apply_updates(predictors, updates)
        return new_predictors, new_opt_state

    return apply_update


def _accumulate_step(
    predictors: Any,
    dataset: Dataset,
    make_step: Callable[..., tuple[Array, Array, Array, Any]],
    length_mask_fraction: Array,
    trainable: Any,
    penalty_weight: Array,
) -> tuple[Array, Array, Array, Any]:
    """One training step: every bucket, gradients accumulated, then averaged.

    Returns the total, data, and penalty terms separately. Keeping them
    apart is not cosmetic: ``restore_best`` and ``losses_history`` track
    the *data* loss, so that "best" cannot drift merely because the
    penalty weight ramped between phases.

    The penalty is bucket-independent, so averaging it over buckets
    returns it unchanged; it is computed inside the per-bucket kernel
    anyway because that is what lets its gradient ride the same
    ``filter_value_and_grad`` as the data term instead of needing a second
    pass over the pytree.
    """
    zero_grads = jax.tree.map(jnp.zeros_like, eqx.filter(predictors, trainable))
    acc_grads = zero_grads
    total_loss = jnp.asarray(0.0)
    total_data = jnp.asarray(0.0)
    total_penalty = jnp.asarray(0.0)
    n_batches = 0
    for bp in dataset.bucket_payloads:
        loss, data, penalty, grads = make_step(
            predictors, bp, length_mask_fraction, penalty_weight
        )
        acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
        total_loss = total_loss + loss
        total_data = total_data + data
        total_penalty = total_penalty + penalty
        n_batches += 1
    denom = float(max(n_batches, 1))
    avg_grads = jax.tree.map(lambda g: g / denom, acc_grads)
    return total_loss / denom, total_data / denom, total_penalty / denom, avg_grads


def _shared_tournament(
    predictors: Any,
    dataset: Dataset,
    *,
    make_step: Callable[..., tuple[Array, Array, Array, Any]],
    apply_update: Callable[..., tuple[Any, Any]],
    optimizer: optax.GradientTransformation,
    trainable: Any,
    tournament_attempts: int,
    tournament_steps: int,
    tournament_lr: float,
    key: Array,
    length_mask_fraction: Array,
    penalty_weight: Array,
) -> Any:
    for attempt in range(tournament_attempts):
        attempt_key = fold(key, f"tournament_attempt_{attempt}")
        try:
            # Per-eqx.Module-leaf re-init across the predictors pytree:
            # ``reinitialize_pytree_with_key`` splits ``attempt_key``
            # into one subkey per Module leaf so that even
            # identical-shape sibling predictors get genuinely
            # different fresh weights for this attempt.
            candidate: Any = reinitialize_pytree_with_key(predictors, attempt_key)
            opt_state = optimizer.init(eqx.filter(candidate, trainable))
            opt_state.hyperparams["learning_rate"] = jnp.asarray(tournament_lr)

            for _ in range(tournament_steps):
                _total, _data, _pen, avg_grads = _accumulate_step(
                    candidate,
                    dataset,
                    make_step,
                    length_mask_fraction,
                    trainable,
                    penalty_weight,
                )
                candidate, opt_state = apply_update(candidate, avg_grads, opt_state)

            # Score on the DATA term alone. A candidate must win on fit,
            # not by having drifted somewhere the penalty happens to like.
            score = jnp.asarray(0.0)
            for bp in dataset.bucket_payloads:
                _bucket_total, bucket_data, _bucket_pen, _g = make_step(
                    candidate, bp, length_mask_fraction, penalty_weight
                )
                score = score + bucket_data
            score_value = float(score)
            if not math.isfinite(score_value):
                raise FloatingPointError(f"non-finite tournament loss: {score_value}")
            return candidate
        except Exception:
            continue

    warnings.warn(
        "tournament: all attempts failed; falling back to initial predictors",
        RuntimeWarning,
        stacklevel=2,
    )
    return predictors


def _select_ui(ui: TrainingUI | None, verbose: bool) -> TrainingUI:
    # An explicit ``ui=`` argument always wins. Otherwise ``verbose=True``
    # picks the Rich live dashboard and ``verbose=False`` silences output
    # entirely. This keeps the common case ergonomic ("just print stuff")
    # while still letting callers swap in a custom UI implementation
    # (e.g. a TensorBoard logger) without changing the loop.
    if ui is not None:
        return ui
    if verbose:
        return RichTrainingUI()
    return SilentUI()


def train_with_optax(
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    simulate_fn: Callable[..., Array],
    solver: SolverConfig,
    trainable: Any = None,
    key: Array,
    ui: TrainingUI | None = None,
) -> tuple[list[float], Any]:
    """Train ``predictors`` against ``dataset`` with Optax.

    ``predictors`` is a ``PyTree[eqx.Module]``: by convention a tuple of
    ``BoundedPredictor`` leaves, but any pytree shape is accepted (dict,
    NamedTuple, single Module — ``eqx.partition`` walks them uniformly).
    ``key`` is required keyword-only — calling without it raises
    ``TypeError`` before any compilation, so reproducibility never
    relies on an implicit default.

    The ``trainable`` argument is a boolean PyTree mask matching
    ``predictors``'s structure. When omitted, it defaults to
    :func:`hybridmodels.trainable.trainable_mask` over the supplied
    pytree, which marks every inexact-array leaf as trainable; pass a
    custom mask (typically built with the freezers in
    ``hybridmodels.trainable``) to hold specific leaves fixed during
    training.

    Returns
    -------
    tuple[list[float], PyTree[eqx.Module]]
        ``(loss_history, trained_predictors)``. ``loss_history`` is the
        training loss recorded once per step across every phase;
        ``trained_predictors`` is the predictors corresponding to the
        best-loss step seen so far when ``config.restore_best=True``,
        or to the final step otherwise.
    """
    if trainable is None:
        trainable = trainable_mask(predictors)

    bucket_payloads = dataset.bucket_payloads
    if not bucket_payloads:
        raise ValueError("train_with_optax: dataset has no bucket payloads")

    loss_fn = _resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights)
    state_to_output = dataset.state_to_output

    ui_ = _select_ui(ui, config.verbose)

    # Built once on the host from static ``bounds``; rebuilding them per
    # step would add trace work for a constant. Empty when the pytree
    # holds no BoundedPredictor, in which case the penalty is a no-op.
    penalty_grids = collocation_grids(predictors, config.penalty_grid_points)

    make_step = _build_make_step(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trainable=trainable,
        penalty_grids=penalty_grids,
    )

    n_phases = len(config.steps)
    total_steps = int(sum(config.steps))
    ui_.on_run_start(total_steps=total_steps, num_phases=n_phases)

    full_mask = jnp.asarray(1.0)
    for idx, bp in enumerate(bucket_payloads):
        bucket_shape = (int(bp.ts.shape[0]), int(bp.ts.shape[1]))
        ui_.on_compile_start(bucket_idx=idx, bucket_shape=bucket_shape)
        warm_loss, _d, _p, _g = make_step(
            predictors, bp, full_mask, jnp.asarray(config.penalty_weight_for_phase(0))
        )
        jax.block_until_ready(warm_loss)  # type: ignore[no-untyped-call]
        ui_.on_compile_done(bucket_idx=idx)
        ui_.on_compile_progress(bucket_idx=idx, total_buckets=len(bucket_payloads))

    optimizer = _build_optimizer(config.optimizer[0], config.lr[0])
    apply_update = _build_apply_update(optimizer, trainable)

    if config.tournament_attempts > 1 and config.tournament_steps > 0:
        tournament_root = fold(key, "tournament")
        predictors = _shared_tournament(
            predictors,
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
            penalty_weight=jnp.asarray(config.penalty_weight_for_phase(0)),
        )

    opt_state = optimizer.init(eqx.filter(predictors, trainable))

    losses_history: list[float] = []
    best_loss = float("inf")
    best_predictors = predictors
    steps_since_improvement = 0

    for phase_idx, n_steps in enumerate(config.steps):
        if phase_idx > 0:
            if config.reset_optimiser_state[phase_idx]:
                optimizer = _build_optimizer(config.optimizer[phase_idx], config.lr[phase_idx])
                apply_update = _build_apply_update(optimizer, trainable)
                opt_state = optimizer.init(eqx.filter(predictors, trainable))
            else:
                opt_state.hyperparams["learning_rate"] = jnp.asarray(config.lr[phase_idx])

        length_mask_fraction = jnp.asarray(config.length_schedule[phase_idx])
        # Traced, not closed over: a Python float that changed per phase
        # would retrace ``make_step`` at every phase boundary. Same reason
        # ``length_mask_fraction`` is passed rather than baked in.
        penalty_weight = jnp.asarray(config.penalty_weight_for_phase(phase_idx))

        ui_.on_phase_start(
            phase_idx=phase_idx,
            phase_steps=int(n_steps),
            lr=float(config.lr[phase_idx]),
            optimizer=config.optimizer[phase_idx],
        )

        for step in range(int(n_steps)):
            avg_total, avg_data, avg_penalty, avg_grads = _accumulate_step(
                predictors,
                dataset,
                make_step,
                length_mask_fraction,
                trainable,
                penalty_weight,
            )
            # History and early stopping follow the DATA term. Tracking the
            # combined objective would let "best" move when only the
            # penalty weight changed, and would make runs with different
            # weights incomparable.
            loss_value = float(avg_data)
            penalty_value = float(avg_penalty)
            losses_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_predictors = predictors
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            predictors, opt_state = apply_update(predictors, avg_grads, opt_state)

            ui_.on_step_end(
                step_idx=step,
                phase_idx=phase_idx,
                loss=loss_value,
                penalty=penalty_value,
            )

            if config.patience > 0 and steps_since_improvement >= config.patience:
                break

        ui_.on_phase_end(phase_idx=phase_idx)

    final_predictors = best_predictors if config.restore_best else predictors
    final_loss = losses_history[-1] if losses_history else float("nan")
    ui_.on_run_end(final_loss=final_loss)
    return losses_history, final_predictors
