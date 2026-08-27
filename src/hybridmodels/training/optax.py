"""Gradient training loop for hybrid mechanistic models, driven by Optax.

One training **step** is one full pass over every bucket: per-bucket
gradients from ``make_step`` (jitted, one trace per bucket shape),
accumulated and averaged, then a single ``optimizer.update``. A bucket is
not a step, and there is no minibatching.

A run is a sequence of **phases**, contiguous blocks of steps sharing
hyperparameters. Each has its own step count, learning rate, optimizer
name, length-schedule fraction and reset flag, carried in
``OptaxTrainingConfig`` as same-length tuples. Phases are how a run
changes strategy partway through, say a coarse pass at a high learning
rate followed by a slow refinement.

Before the phases the loop can run a **tournament**: re-initialise the
predictors several times from different keys, train each candidate
briefly, keep the one with the lowest data loss. This escapes an unlucky
initial draw, which matters because a hybrid ODE model can be
unrecoverable from a bad start. It reuses the main loop's compiled
kernels, so it adds no compilation cost.
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

# R-T7 allows a tournament attempt to fail on a diffrax error or a
# non-finite loss. Anything else is a bug in user or framework code and
# must reach the user instead of being retried with a different seed.
_TOURNAMENT_FAILURES: tuple[type[BaseException], ...] = (
    FloatingPointError,
    jax.errors.JaxRuntimeError,
)

_PHASE_KEYED_FIELDS: tuple[str, ...] = (
    "lr",
    "optimizer",
    "reset_optimiser_state",
    "length_schedule",
)


@dataclass(frozen=True)
class OptaxTrainingConfig:
    """Configuration for :func:`train_with_optax`.

    The first five fields are **phase-keyed**: ``steps``, ``lr``,
    ``optimizer``, ``reset_optimiser_state`` and ``length_schedule`` carry
    one entry per phase, must have the same length, and do not broadcast.
    None has a defensible default, so a single-phase run spells out
    one-element tuples::

        OptaxTrainingConfig(
            steps=(500,), lr=(1e-3,), optimizer=("adamw",),
            reset_optimiser_state=(False,),
        )

    ``penalty_weight`` is the exception. It has an unambiguous off state,
    so a length-1 tuple broadcasts across every phase.

    Attributes
    ----------
    steps : tuple[int, ...]
        Step budget per phase. Its length is the number of phases.
    lr : tuple[float, ...]
        Learning rate per phase. Applied to the live optimiser state
        unless that phase also resets it.
    optimizer : tuple[str, ...]
        Optimiser name per phase, ``"adamw"`` or ``"adabelief"``. A phase
        that changes the name must also set ``reset_optimiser_state``,
        because optimiser state belongs to the optimiser that built it.
    reset_optimiser_state : tuple[bool, ...]
        Per phase, rebuild the optimiser and discard its state at that
        boundary. Set it when switching optimiser, and when a
        length-schedule change has made the accumulated momentum wrong.
    length_schedule : tuple[float, ...]
        Fraction of each experiment's timeline the loss looks at, per
        phase, in ``(0, 1]``. It masks the **loss**, never the
        integration: the solver still runs the full trajectory, and only
        the first ``fraction`` of the observation times is scored, which
        stops a long-horizon divergence from drowning the gradient. Being
        a runtime mask rather than a shape change, a phase boundary costs
        no recompile. Default ``(1.0,)`` scores everything.
    penalty_weight : tuple[float, ...]
        Weight on the bound-saturation penalty. Length 1 broadcasts to
        every phase; any other length must match ``steps``. Entries must
        be non-negative. ``0.0`` disables the penalty.
    penalty_grid_points : int
        Points per input dimension in the collocation grid the penalty is
        evaluated on. At least 2 (one per box edge).
    loss : Callable | str
        A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``,
        ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``.
    channel_idx, channel_weights : tuple | None
        Forwarded into the resolved loss. See ``hybridmodels.losses``.
    tournament_attempts, tournament_steps : int
        The tournament runs only when ``tournament_steps > 0`` and
        ``tournament_attempts > 1``. Each attempt re-initialises the
        predictors, trains for ``tournament_steps`` steps, and is scored on
        the data term alone by a forward-only pass; the lowest wins. An
        attempt that raises a diffrax error or a non-finite loss is dropped
        and the next key tried. If all fail, the original predictors are
        used and a ``RuntimeWarning`` is raised.
    tournament_lr : float
        Learning rate for the tournament's short bursts, independent of
        ``lr``.
    patience : int
        Consecutive steps without a new best data loss before the current
        phase stops early. Counted within a phase and reset at every phase
        boundary, so a plateau at the end of one phase cannot kill the next
        before its new learning rate acts. ``0`` disables early stopping.
    restore_best : bool
        Return the predictors from the lowest-data-loss step instead of the
        last one. The running minimum resets whenever ``length_schedule``
        changes, since losses over different horizons are not comparable
        and the shortest horizon would otherwise always own the minimum.
    verbose : bool
        Selects ``RichTrainingUI`` over ``SilentUI`` when ``ui=None``. An
        explicit ``ui=...`` argument always wins.
    """

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
        # penalty_weight is deliberately NOT in _PHASE_KEYED_FIELDS: those
        # fields lack a safe default, while this one has an unambiguous off
        # state. A length-1 tuple broadcasts; any other length must match
        # `steps` exactly, so a real per-phase schedule cannot be truncated.
        if len(self.penalty_weight) not in (1, n):
            raise ValueError(
                "OptaxTrainingConfig.penalty_weight must have length 1 "
                f"(broadcast across phases) or {n} (one per phase); "
                f"got {len(self.penalty_weight)}"
            )
        for weight in self.penalty_weight:
            if float(weight) < 0.0:
                raise ValueError(
                    f"OptaxTrainingConfig.penalty_weight entries must be non-negative; got {weight}"
                )
        if self.penalty_grid_points < 2:
            raise ValueError(
                "OptaxTrainingConfig.penalty_grid_points must be at least 2 "
                f"(one point per box edge); got {self.penalty_grid_points}"
            )
        for phase_idx in range(1, n):
            # Without a reset only the learning rate is pushed into the
            # existing opt_state, so a changed optimizer name would be
            # accepted and then ignored. Refuse the combination instead.
            if (
                self.optimizer[phase_idx] != self.optimizer[phase_idx - 1]
                and not self.reset_optimiser_state[phase_idx]
            ):
                raise ValueError(
                    f"OptaxTrainingConfig: phase {phase_idx} changes optimizer from "
                    f"{self.optimizer[phase_idx - 1]!r} to {self.optimizer[phase_idx]!r} "
                    "but reset_optimiser_state[{0}] is False. Optimiser state is "
                    "specific to the optimiser that built it, so switching without a "
                    "reset would keep running the previous one. Set "
                    "reset_optimiser_state[{0}]=True.".format(phase_idx)
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
) -> Callable[[Any, BucketPayload, Array], tuple[Array, Any]]:
    def loss_eval(
        diff_predictors: Any,
        static_predictors: Any,
        bp_masked: BucketPayload,
    ) -> Array:
        # ``eqx.combine`` walks any pytree shape, so the container is never
        # inspected before the recombined tree goes to simulate_fn.
        predictors = eqx.combine(diff_predictors, static_predictors)

        def per_experiment(ts: Array, covariates: dict[str, Array], y0: Array) -> Array:
            full_state = simulate_fn(predictors, ts, covariates, y0, solver)
            return state_to_output(full_state)

        pred_obs = jax.vmap(per_experiment, in_axes=(0, 0, 0))(
            bp_masked.ts, bp_masked.covariates, bp_masked.y0
        )
        return loss_fn(pred_obs, bp_masked)

    grad_fn = eqx.filter_value_and_grad(loss_eval)

    @eqx.filter_jit
    def make_step(
        predictors: Any, bp: BucketPayload, length_mask_fraction: Array
    ) -> tuple[Array, Any]:
        T = bp.ts.shape[1]
        cutoff = jnp.maximum(
            jnp.ceil(jnp.float32(T) * length_mask_fraction).astype(jnp.int32),
            jnp.int32(1),
        )
        sched_mask = (jnp.arange(T) < cutoff)[None, :, None]
        bp_masked = bp._replace(mask=bp.mask & sched_mask)
        diff_part, static_part = eqx.partition(predictors, trainable)
        return grad_fn(diff_part, static_part, bp_masked)

    return make_step


def _build_score_bucket(
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    loss_fn: Callable[[Array, BucketPayload], Array],
) -> Callable[[Any, BucketPayload, Array], Array]:
    """Return a forward-only ``score_bucket(predictors, bp, fraction) -> loss``.

    Scoring through ``make_step`` would run a full ``value_and_grad`` and
    discard the gradients, roughly tripling the cost of a scoring sweep.
    This costs one extra compile per bucket shape and pays for itself above
    two attempts.
    """

    @eqx.filter_jit
    def score_bucket(predictors: Any, bp: BucketPayload, length_mask_fraction: Array) -> Array:
        T = bp.ts.shape[1]
        cutoff = jnp.maximum(
            jnp.ceil(jnp.float32(T) * length_mask_fraction).astype(jnp.int32),
            jnp.int32(1),
        )
        sched_mask = (jnp.arange(T) < cutoff)[None, :, None]
        bp_masked = bp._replace(mask=bp.mask & sched_mask)

        def per_experiment(ts: Array, covariates: dict[str, Array], y0: Array) -> Array:
            return state_to_output(simulate_fn(predictors, ts, covariates, y0, solver))

        pred_obs = jax.vmap(per_experiment, in_axes=(0, 0, 0))(
            bp_masked.ts, bp_masked.covariates, bp_masked.y0
        )
        return loss_fn(pred_obs, bp_masked)

    return score_bucket


def _build_penalty_step(
    *, penalty_grids: tuple[Array, ...], trainable: Any
) -> Callable[[Any, Array], tuple[Array, Any]]:
    """Return ``penalty_step(predictors, weight) -> (penalty, weighted_grads)``.

    Evaluated once per training step, outside the bucket loop: the penalty
    reads only the predictors pytree, so computing it inside ``make_step``
    would repeat one identical evaluation per bucket.

    Returns the gradient of ``weight * penalty``, to add straight onto the
    averaged data gradient. ``penalty`` comes back unweighted, since that
    is what gets reported.
    """

    def weighted(diff_predictors: Any, static_predictors: Any, weight: Array) -> Array:
        predictors = eqx.combine(diff_predictors, static_predictors)
        return weight * bound_penalty(predictors, penalty_grids)

    grad_fn = eqx.filter_value_and_grad(weighted)

    @eqx.filter_jit
    def penalty_step(predictors: Any, weight: Array) -> tuple[Array, Any]:
        diff_part, static_part = eqx.partition(predictors, trainable)
        weighted_value, grads = grad_fn(diff_part, static_part, weight)
        # Report the raw penalty; the weight is a scheduling choice and
        # folding it into the number would make phases incomparable.
        unweighted = jnp.where(weight > 0.0, weighted_value / jnp.maximum(weight, 1e-30), 0.0)
        return unweighted, grads

    return penalty_step


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
    make_step: Callable[..., tuple[Array, Any]],
    length_mask_fraction: Array,
    trainable: Any,
) -> tuple[Array, Any]:
    """One training step: every bucket, gradients accumulated, then averaged.

    Returns the data term only. The bound penalty is bucket-independent
    and is added once per step by the caller, via ``_build_penalty_step``.
    """
    zero_grads = jax.tree.map(jnp.zeros_like, eqx.filter(predictors, trainable))
    acc_grads = zero_grads
    total_loss = jnp.asarray(0.0)
    n_batches = 0
    for bp in dataset.bucket_payloads:
        loss, grads = make_step(predictors, bp, length_mask_fraction)
        acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
        total_loss = total_loss + loss
        n_batches += 1
    denom = float(max(n_batches, 1))
    avg_grads = jax.tree.map(lambda g: g / denom, acc_grads)
    return total_loss / denom, avg_grads


def _shared_tournament(
    predictors: Any,
    dataset: Dataset,
    *,
    make_step: Callable[..., tuple[Array, Any]],
    score_bucket: Callable[..., Array],
    apply_update: Callable[..., tuple[Any, Any]],
    optimizer: optax.GradientTransformation,
    trainable: Any,
    tournament_attempts: int,
    tournament_steps: int,
    tournament_lr: float,
    key: Array,
    length_mask_fraction: Array,
) -> Any:
    """Warm-start selection: train several fresh inits briefly, keep the best.

    Each candidate is re-initialised from its own subkey, trained for
    ``tournament_steps`` steps at ``tournament_lr``, then scored on the
    data term with a forward-only pass. Scoring on the data term alone,
    not the combined objective, stops a candidate winning by drifting
    somewhere the penalty likes rather than by fitting.

    An attempt that raises a diffrax error or a non-finite score is dropped
    and the next key tried. If every attempt fails, the original
    ``predictors`` come back with a ``RuntimeWarning``.
    """
    last_error: BaseException | None = None
    best_score = math.inf
    best_candidate: Any = None
    for attempt in range(tournament_attempts):
        attempt_key = fold(key, f"tournament_attempt_{attempt}")
        try:
            # One subkey per Module leaf, so identical-shape sibling
            # predictors get genuinely different fresh weights.
            candidate: Any = reinitialize_pytree_with_key(predictors, attempt_key)
            opt_state = optimizer.init(eqx.filter(candidate, trainable))
            opt_state.hyperparams["learning_rate"] = jnp.asarray(tournament_lr)

            for _ in range(tournament_steps):
                _loss, avg_grads = _accumulate_step(
                    candidate, dataset, make_step, length_mask_fraction, trainable
                )
                candidate, opt_state = apply_update(candidate, avg_grads, opt_state)

            # Forward-only scorer, not make_step, whose discarded backward
            # pass roughly tripled the cost of a scoring sweep. Scores the
            # data term alone, so a candidate wins on fit.
            score = jnp.asarray(0.0)
            for bp in dataset.bucket_payloads:
                score = score + score_bucket(candidate, bp, length_mask_fraction)
            score_value = float(score)
            if not math.isfinite(score_value):
                raise FloatingPointError(f"non-finite tournament loss: {score_value}")
            # Strict ``<``, so ties keep the earlier attempt and the
            # result stays a deterministic function of ``key``.
            if score_value < best_score:
                best_score = score_value
                best_candidate = candidate
        except _TOURNAMENT_FAILURES as exc:
            # Narrow on purpose: only a diffrax error and a non-finite loss
            # are attempt failures. Catching everything also swallowed shape
            # bugs in the user's simulate_fn and framework bugs, reporting
            # them as one warning while training carried on unmodified.
            last_error = exc
            continue

    if best_candidate is not None:
        return best_candidate

    warnings.warn(
        "tournament: all attempts failed; falling back to initial predictors. "
        f"Last failure: {last_error!r}",
        RuntimeWarning,
        stacklevel=2,
    )
    return predictors


def _select_ui(ui: TrainingUI | None, verbose: bool) -> TrainingUI:
    # An explicit ``ui=`` always wins, so a caller can swap in a custom UI
    # (a TensorBoard logger, say) without changing the loop.
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

    Runs the phases described by ``config``, optionally preceded by a
    tournament. See the module docstring for what a step, a phase, and
    the tournament are, and :class:`OptaxTrainingConfig` for the fields.

    ``predictors`` is a ``PyTree[eqx.Module]``, conventionally a tuple of
    ``BoundedPredictor`` leaves but any shape ``eqx.partition`` can walk.
    ``key`` is keyword-only and required, so reproducibility never rests on
    an implicit default.

    ``trainable`` is a boolean mask matching ``predictors``. Omitting it
    defaults to :func:`hybridmodels.trainable.trainable_mask`, which marks
    every inexact-array leaf trainable. Pass a custom mask, usually from
    the freezers in ``hybridmodels.trainable``, to hold leaves fixed;
    freezing ``BoundScaler`` leaves is the common case.

    Returns
    -------
    tuple[list[float], PyTree[eqx.Module]]
        ``(loss_history, trained_predictors)``.

        ``loss_history`` is the **raw per-step data loss**, one entry per
        step, concatenated across phases. It can go up. The bound penalty
        is excluded, so a ramping penalty weight cannot move the series and
        runs with different weights stay comparable, and nothing is
        smoothed: these are the values the optimiser saw.

        It differs from
        :func:`~hybridmodels.training.evosax.train_with_evosax`, whose
        history is best-so-far and therefore monotone. Same type, same
        position, different meaning: plotting both on one axis misleads.

        ``trained_predictors`` comes from the lowest-loss step when
        ``config.restore_best=True``, else the final step. That minimum
        resets whenever ``length_schedule`` changes, so the returned model
        always comes from the last horizon trained on.
    """
    if trainable is None:
        trainable = trainable_mask(predictors)

    bucket_payloads = dataset.bucket_payloads
    if not bucket_payloads:
        raise ValueError("train_with_optax: dataset has no bucket payloads")

    loss_fn = _resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights)
    state_to_output = dataset.state_to_output

    ui_ = _select_ui(ui, config.verbose)

    # Built once on the host from static ``bounds``. Empty when the pytree
    # holds no BoundedPredictor, in which case the penalty is a no-op.
    penalty_grids = collocation_grids(predictors, config.penalty_grid_points)

    make_step = _build_make_step(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trainable=trainable,
    )
    penalty_step = _build_penalty_step(penalty_grids=penalty_grids, trainable=trainable)
    score_bucket = _build_score_bucket(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
    )

    n_phases = len(config.steps)
    total_steps = int(sum(config.steps))
    ui_.on_run_start(total_steps=total_steps, num_phases=n_phases)

    full_mask = jnp.asarray(1.0)
    for idx, bp in enumerate(bucket_payloads):
        bucket_shape = (int(bp.ts.shape[0]), int(bp.ts.shape[1]))
        ui_.on_compile_start(bucket_idx=idx, bucket_shape=bucket_shape)
        warm_loss, _g = make_step(predictors, bp, full_mask)
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
            score_bucket=score_bucket,
            apply_update=apply_update,
            optimizer=optimizer,
            trainable=trainable,
            tournament_attempts=config.tournament_attempts,
            tournament_steps=config.tournament_steps,
            tournament_lr=config.tournament_lr,
            key=tournament_root,
            length_mask_fraction=full_mask,
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

        # Patience counts within a phase. A plateau at the end of one phase
        # would otherwise carry over and kill the next after a single step,
        # before its fresh learning rate had any chance to act.
        steps_since_improvement = 0

        # "Best" only means something among losses over the same horizon. A
        # phase with length_schedule=0.2 scores a fifth of each trajectory,
        # so carried across the boundary its minimum wins every time and
        # restore_best hands back the least-trained model in the run.
        if phase_idx > 0 and (
            config.length_schedule[phase_idx] != config.length_schedule[phase_idx - 1]
        ):
            best_loss = float("inf")
            best_predictors = predictors

        ui_.on_phase_start(
            phase_idx=phase_idx,
            phase_steps=int(n_steps),
            lr=float(config.lr[phase_idx]),
            optimizer=config.optimizer[phase_idx],
        )

        for step in range(int(n_steps)):
            avg_data, avg_grads = _accumulate_step(
                predictors, dataset, make_step, length_mask_fraction, trainable
            )
            avg_penalty, penalty_grads = penalty_step(predictors, penalty_weight)
            # Added after the bucket average, not inside it: the penalty is
            # charged once per step, not once per bucket.
            avg_grads = jax.tree.map(jnp.add, avg_grads, penalty_grads)
            # Dispatch the update before blocking on the loss values. Both
            # float() calls are host syncs; reading them first left the
            # accelerator idle through the Python bookkeeping every step.
            previous_predictors = predictors
            predictors, opt_state = apply_update(predictors, avg_grads, opt_state)

            # History and early stopping follow the DATA term: tracking the
            # combined objective would let "best" move when only the penalty
            # weight changed.
            loss_value = float(avg_data)
            penalty_value = float(avg_penalty)
            losses_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                # avg_data was measured at the pre-update parameters, so the
                # snapshot has to be those, not the ones just produced.
                best_predictors = previous_predictors
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

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
