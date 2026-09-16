"""Gradient training loop for hybrid mechanistic models, driven by Optax.

One training **step** is one full pass over every bucket: per-bucket
gradients from ``bucket_step`` (jitted, one trace per bucket shape),
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
briefly, keep the one with the lowest tracked loss (data plus any trajectory
penalty). This escapes an unlucky
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

from hybridmodels.data import BucketPayload, Dataset, make_bootstrap_dataset
from hybridmodels.losses import resolve_loss_fn
from hybridmodels.penalties import (
    PenaltyPointSource,
    bound_penalty,
    data_penalty_points,
    select_penalty_points,
    validate_penalty_points,
)
from hybridmodels.predictors.base import reinitialize_pytree_with_key
from hybridmodels.rng import fold
from hybridmodels.solver import SolverConfig
from hybridmodels.trainable import trainable_mask
from hybridmodels.training.kernels import (
    build_apply_update,
    build_bucket_step,
    build_penalty_step,
    build_score_bucket,
)
from hybridmodels.ui.base import SilentUI, TrainingUI
from hybridmodels.ui.optax import RichTrainingUI

# An optimizer entry may be a registered name ("adamw"), a factory taking
# ``learning_rate`` and returning an ``optax.GradientTransformation``, or a
# ready-made transformation instance. Names and factories are wrapped in
# ``optax.inject_hyperparams`` so a phase boundary can move the learning
# rate; a raw instance has its own state and cannot be re-hyperparametrised.
OptimizerSpec = (
    str | Callable[[float], optax.GradientTransformation] | optax.GradientTransformation
)

# R-T7 allows a tournament attempt to fail on a diffrax error or a
# non-finite loss. Anything else is a bug in user or framework code and
# must reach the user instead of being retried with a different seed.
_TOURNAMENT_FAILURES: tuple[type[BaseException], ...] = (
    FloatingPointError,
    jax.errors.JaxRuntimeError,
    eqx.EquinoxRuntimeError,
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
    optimizer : tuple[OptimizerSpec, ...]
        Optimiser per phase: a registered name (``"adamw"``,
        ``"adabelief"``), a factory taking ``learning_rate`` and returning
        an ``optax.GradientTransformation`` (e.g. ``optax.adamw``, or
        ``lambda learning_rate: optax.chain(optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate))``), or a ready-made transformation instance.
        Names and factories are wrapped in ``optax.inject_hyperparams``, so
        a phase boundary can move the learning rate without a rebuild; a
        raw instance cannot be re-hyperparametrised, so a phase that
        changes ``lr`` with a raw instance must also set
        ``reset_optimiser_state``. A phase that changes the optimiser must
        set ``reset_optimiser_state``, because optimiser state belongs to
        the optimiser that built it.
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
        be non-negative. ``0.0`` disables the penalty. The weight is
        relative to the per-bucket-averaged data term: the penalty is a
        mean over its points, charged once per step onto the averaged data
        gradient, so the same weight means the same thing whatever the
        dataset or point-count size.
    penalty_points : tuple[Array, ...] | None
        User-supplied penalty-only points for the bound penalty: one
        ``[G, n_inputs]`` array of physical input vectors per
        ``BoundedPredictor`` leaf, in traversal order, matching each leaf's
        ``input_keys`` column order. No measurements are needed there;
        saturation is charged at these points regardless of the data. When
        ``None`` the penalty uses only the measured points gathered from
        the dataset (leaves whose inputs do not all resolve to dataset
        covariates must be covered by an entry here, or the run raises).
        ``hybridmodels.penalties.box_grid`` builds a warp-uniform box sweep
        for the "police the whole box" recipe.
    penalty_fn : Callable | None
        The regulariser added to the data objective, defaulting to
        :func:`hybridmodels.penalties.bound_penalty` when ``None``. A
        custom callable ``(predictors, points) -> scalar`` replaces
        the bound penalty with e.g. weight decay on inner weights or a
        monotonicity term; one that ignores points simply does not use them.
    trajectory_penalty_fn : Callable | None
        Trajectory-aware penalty for **embedded** hybrid models (the
        predictor runs inside the vector field). Called as
        ``(full_state, bp) -> scalar`` with the full state ``[N, T, S]``
        *including* any penalty accumulators carried in the ODE state; add
        it to the data loss inside the same forward pass. ``None`` (the
        default) disables it. See the helpers in ``hybridmodels.penalties``
        (``attach_penalty_state`` / ``penalty_vector_field`` /
        ``strip_penalty_state`` / ``penalty_integral``).
    trajectory_penalty_weight : float
        Scalar weight on ``trajectory_penalty_fn``. ``0.0`` disables it
        even if a function is set. Non-negative.
    loss : Callable | str
        A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``,
        ``"bal_mle"``) or a callable matching ``loss(pred_obs, bp)``.
    channel_idx, channel_weights : tuple | None
        Forwarded into the resolved loss. See ``hybridmodels.losses``.
    tournament_attempts, tournament_steps : int
        The tournament runs only when ``tournament_steps > 0`` and
        ``tournament_attempts > 1``. Each attempt re-initialises the
        predictors, trains for ``tournament_steps`` steps, and is scored on
        the data term plus any configured trajectory penalty by a
        forward-only pass; the bound penalty remains excluded. The lowest
        score wins. An attempt that raises a diffrax error or a non-finite
        loss is dropped and the next key tried. If all fail, the original
        predictors are used and a ``RuntimeWarning`` is raised.
    tournament_lr : float
        Learning rate for the tournament's short bursts, independent of
        ``lr``.
    patience : int
        Consecutive steps without a new best tracked loss before the current
        phase stops early. The tracked loss is the data term plus any
        trajectory penalty, excluding the fixed-point bound penalty. Counted
        within a phase and reset at every phase boundary, so a plateau at the
        end of one phase cannot kill the next before its new learning rate
        acts. ``0`` disables early stopping.
    restore_best : bool
        Return the predictors from the lowest tracked-loss step instead of
        the last one. The running minimum resets whenever ``length_schedule``
        changes, since tracked losses over different horizons are not
        comparable and the shortest horizon would otherwise always own the
        minimum.
    verbose : bool
        Selects ``RichTrainingUI`` over ``SilentUI`` when ``ui=None``. An
        explicit ``ui=...`` argument always wins.
    """

    steps: tuple[int, ...]
    lr: tuple[float, ...]
    optimizer: tuple[OptimizerSpec, ...]
    reset_optimiser_state: tuple[bool, ...]
    length_schedule: tuple[float, ...] = (1.0,)
    penalty_weight: tuple[float, ...] = (0.0,)
    penalty_points: tuple[Array, ...] | None = None
    penalty_fn: Callable[..., Array] | None = None
    trajectory_penalty_fn: Callable[..., Array] | None = None
    trajectory_penalty_weight: float = 0.0
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
        # Lengths first: every later rule indexes a phase-keyed field, and on a
        # short tuple would raise IndexError instead of the real message.
        _validate_phase_lengths(self)
        _validate_penalty(self)
        _validate_penalty_points(self)
        _validate_optimizer_transitions(self)
        _validate_length_schedule(self)
        _validate_trajectory_penalty(self)
        _validate_tournament(self)


def _validate_phase_lengths(config: OptaxTrainingConfig) -> None:
    """Every phase-keyed field must be exactly as long as ``steps``.

    These four have no defensible default, so none of them broadcasts. A
    shorter tuple would either truncate the run or index out of range at a
    phase boundary, both silently.
    """
    n = len(config.steps)
    if n == 0:
        raise ValueError("OptaxTrainingConfig.steps must contain at least one phase")
    for step_count in config.steps:
        if step_count < 1:
            raise ValueError(
                f"OptaxTrainingConfig.steps entries must be at least 1; got {step_count}"
            )
    for learning_rate in config.lr:
        if not math.isfinite(float(learning_rate)) or float(learning_rate) < 0.0:
            raise ValueError(
                "OptaxTrainingConfig.lr entries must be finite and non-negative; "
                f"got {learning_rate}"
            )
    for name in _PHASE_KEYED_FIELDS:
        value = getattr(config, name)
        if len(value) != n:
            raise ValueError(
                f"OptaxTrainingConfig: phase-keyed field {name!r} has length "
                f"{len(value)}, expected {n} (matching steps)"
            )


def _validate_penalty(config: OptaxTrainingConfig) -> None:
    """``penalty_weight`` broadcasts from length 1, and entries are non-negative.

    It is deliberately not in ``_PHASE_KEYED_FIELDS``: those fields lack a
    safe default, while this one has an unambiguous off state. A length-1
    tuple broadcasts across every phase; any other length must match ``steps``
    exactly, so a real per-phase schedule cannot be silently truncated.
    """
    n = len(config.steps)
    if len(config.penalty_weight) not in (1, n):
        raise ValueError(
            "OptaxTrainingConfig.penalty_weight must have length 1 "
            f"(broadcast across phases) or {n} (one per phase); "
            f"got {len(config.penalty_weight)}"
        )
    for weight in config.penalty_weight:
        if not math.isfinite(float(weight)):
            raise ValueError(
                "OptaxTrainingConfig.penalty_weight entries must be finite; "
                f"got {weight}"
            )
        if float(weight) < 0.0:
            raise ValueError(
                f"OptaxTrainingConfig.penalty_weight entries must be non-negative; got {weight}"
            )


def _validate_penalty_points(config: OptaxTrainingConfig) -> None:
    """User-supplied penalty points are rank-2 physical input vectors.

    Length and input-column checks against the actual ``BoundedPredictor``
    leaves need the predictors pytree, so they happen at kernel build time
    (:func:`hybridmodels.penalties.validate_penalty_points`); here only the
    per-array shape is checkable without it.
    """
    for idx, points in enumerate(config.penalty_points or ()):
        if points.ndim != 2:
            raise ValueError(
                "OptaxTrainingConfig.penalty_points entries must be rank-2 "
                f"[G, n_inputs] arrays of physical input vectors; entry {idx} "
                f"has shape {points.shape}"
            )


def _validate_optimizer_transitions(config: OptaxTrainingConfig) -> None:
    """A phase that changes the optimiser must also reset its state.

    Without a reset only the learning rate is pushed into the existing
    ``opt_state``, so a changed optimiser would be accepted and then
    ignored, leaving the previous optimiser running for the rest of the
    run. This check fires only when the two phase specs are actually
    different. ``optax.GradientTransformation`` is a NamedTuple of
    closures, so equality is identity: two separately-built raw
    transformations are always unequal (and a raw optimiser's baked-in lr
    is fixed anyway, so a raw spec across an lr change needs a reset
    regardless).
    """
    for phase_idx in range(1, len(config.steps)):
        if (
            config.optimizer[phase_idx] != config.optimizer[phase_idx - 1]
            and not config.reset_optimiser_state[phase_idx]
        ):
            raise ValueError(
                f"OptaxTrainingConfig: phase {phase_idx} changes optimizer from "
                f"{config.optimizer[phase_idx - 1]!r} to {config.optimizer[phase_idx]!r} "
                f"but reset_optimiser_state[{phase_idx}] is False. Optimiser state is "
                "specific to the optimiser that built it, so switching without a "
                "reset would keep running the previous one. Set "
                f"reset_optimiser_state[{phase_idx}]=True."
            )


def _validate_length_schedule(config: OptaxTrainingConfig) -> None:
    """Every fraction lies in ``(0, 1]``.

    Zero would score no timestamps at all, and above one would claim more of
    the trajectory than there is.
    """
    for fraction in config.length_schedule:
        if not (0.0 < float(fraction) <= 1.0):
            raise ValueError(
                f"OptaxTrainingConfig.length_schedule entries must lie in (0, 1]; got {fraction}"
            )


def _validate_trajectory_penalty(config: OptaxTrainingConfig) -> None:
    """The trajectory-penalty weight is non-negative, and a set weight needs a function.

    ``trajectory_penalty_weight`` is a scalar (not per-phase): the
    trajectory penalty is a regulariser on the forward pass, so a per-phase
    schedule is a follow-up, not v1. A positive weight with no function
    would silently do nothing, so it is refused; a function with weight
    ``0.0`` is a no-op the user can flip on later without editing the hook.
    """
    if not math.isfinite(config.trajectory_penalty_weight):
        raise ValueError(
            "OptaxTrainingConfig.trajectory_penalty_weight must be finite; "
            f"got {config.trajectory_penalty_weight}"
        )
    if config.trajectory_penalty_weight < 0.0:
        raise ValueError(
            "OptaxTrainingConfig.trajectory_penalty_weight must be non-negative; "
            f"got {config.trajectory_penalty_weight}"
        )
    if config.trajectory_penalty_weight != 0.0 and config.trajectory_penalty_fn is None:
        raise ValueError(
            "OptaxTrainingConfig.trajectory_penalty_weight is non-zero but "
            "trajectory_penalty_fn is None. Provide a "
            "(full_state, bp) -> scalar function to charge."
        )


def _validate_tournament(config: OptaxTrainingConfig) -> None:
    """Validate the optional tournament budget and its learning rate."""
    if config.tournament_attempts < 1:
        raise ValueError("OptaxTrainingConfig.tournament_attempts must be at least 1")
    if config.tournament_steps < 0:
        raise ValueError("OptaxTrainingConfig.tournament_steps must be non-negative")
    if not math.isfinite(config.tournament_lr) or config.tournament_lr < 0.0:
        raise ValueError(
            "OptaxTrainingConfig.tournament_lr must be finite and non-negative"
        )
    if config.patience < 0:
        raise ValueError("OptaxTrainingConfig.patience must be non-negative")


def _build_optimizer(spec: OptimizerSpec, lr: float) -> optax.GradientTransformation:
    """Build the per-phase optimiser, honouring names, factories, and raw instances.

    A name string (``"adamw"``/``"adabelief"``) and a factory taking
    ``learning_rate`` are both wrapped in ``optax.inject_hyperparams``, so
    ``_begin_phase`` can move the learning rate in the live optimiser state
    without a rebuild. A raw ``optax.GradientTransformation`` instance is
    returned as-is: its state is not re-hyperparametrisable, so a phase that
    changes ``lr`` with a raw instance must also reset the optimiser.
    """
    if isinstance(spec, str):
        norm = spec.lower().strip()
        factory = {
            "adamw": optax.adamw,
            "adabelief": optax.adabelief,
        }.get(norm)
        if factory is None:
            raise ValueError(
                f"OptaxTrainingConfig.optimizer={spec!r} is not a supported name; "
                "expected 'adamw' or 'adabelief'. Pass a factory taking "
                "learning_rate (e.g. optax.adamw) or a ready-made "
                "optax.GradientTransformation for anything else."
            )
        return optax.inject_hyperparams(factory)(learning_rate=lr)
    if isinstance(spec, optax.GradientTransformation):
        return spec
    if callable(spec):
        try:
            return optax.inject_hyperparams(spec)(learning_rate=lr)
        except TypeError as exc:
            raise ValueError(
                f"OptaxTrainingConfig.optimizer factory {spec!r} failed when "
                "called with learning_rate=. A factory must accept a keyword "
                "parameter named 'learning_rate' (or use optax's own "
                "`inject_hyperparams` conventions)." 
            ) from exc
    raise ValueError(
        f"OptaxTrainingConfig.optimizer={spec!r} is not a name, a factory "
        "taking learning_rate, or an optax.GradientTransformation."
    )


def _optimiser_state_supports_lr(opt_state: Any) -> bool:
    """Does this optimiser state carry injectable hyperparams?

    ``optax.inject_hyperparams`` builds a state whose top-level
    ``hyperparams`` mapping holds the learning rate; a raw transformation's
    state (a plain optax tuple) has no such attribute. The check is
    structural, so the loop never pokes a ``hyperparams`` key that a raw
    optimiser does not have.
    """
    return hasattr(opt_state, "hyperparams")


def _build_step_update(
    *,
    optimizer: optax.GradientTransformation,
    trainable: Any,
) -> Callable[..., tuple[Any, Any]]:
    """Fused per-step tail: average, merge the bound penalty, and optimise.

    One jitted ``step_update(predictors, acc_grads, penalty_grads,
    n_buckets, opt_state) -> (predictors, opt_state)`` replacing what used
    to be two Python tree walks (normalise by bucket count, add the
    penalty grads) plus a separate ``apply_update`` launch every step.
    Each eager per-leaf tree operation dispatches one compiled kernel per
    parameter leaf — on CPU that measured ~2 ms per tree walk for a small
    MLP, several times the ODE solve itself — so folding them into the
    optimiser kernel removes most of the Python-side step cost without
    changing the math: the averaging and the penalty merge run in the
    same order, so the optimiser sees bit-identical gradients.

    ``n_buckets`` stays a traced argument rather than a closure because a
    bootstrap ensemble re-buckets every resample: closing over the source
    dataset's count would silently average with the wrong denominator.
    """

    @eqx.filter_jit
    def step_update(
        predictors: Any,
        acc_grads: Any,
        penalty_grads: Any,
        n_buckets: Array,
        opt_state: Any,
    ) -> tuple[Any, Any]:
        avg_grads = jax.tree.map(
            lambda a, p: a / n_buckets + p, acc_grads, penalty_grads
        )
        params = eqx.filter(predictors, trainable)
        updates, new_opt_state = optimizer.update(avg_grads, opt_state, params)
        return eqx.apply_updates(predictors, updates), new_opt_state

    return step_update


def _training_step(
    predictors: Any,
    dataset: Dataset,
    bucket_step: Callable[..., tuple[Array, Any]],
    length_mask_fraction: Array,
    trainable: Any,
) -> tuple[Array, Any, int]:
    """One training step: every bucket, gradients accumulated, raw sums back.

    Returns ``(total_loss, acc_grads, n_buckets)`` — the **raw** sum of
    per-bucket losses and gradients, not yet divided. The caller merges
    the bound-penalty gradients and normalises inside the fused
    ``step_update`` kernel (:func:`_build_step_update`), so the per-step
    Python side never walks the parameter tree. The first bucket's
    gradients seed the accumulator directly instead of adding to a
    ``zeros_like`` tree — one fewer pointless per-leaf pass per step.

    ``total_loss`` is the data term plus any configured trajectory penalty
    (charged inside ``bucket_step``'s forward pass). The bound penalty is
    bucket-independent and added once per step by the caller.
    """
    acc_grads: Any = None
    total_loss = jnp.asarray(0.0)
    n_buckets = 0
    for bp in dataset.bucket_payloads:
        loss, grads = bucket_step(predictors, bp, length_mask_fraction)
        if acc_grads is None:
            acc_grads = grads
        else:
            acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
        total_loss = total_loss + loss
        n_buckets += 1
    if acc_grads is None:  # defensive; callers validate a non-empty dataset
        acc_grads = jax.tree.map(jnp.zeros_like, eqx.filter(predictors, trainable))
    return total_loss, acc_grads, n_buckets


def _shared_tournament(
    predictors: Any,
    dataset: Dataset,
    *,
    bucket_step: Callable[..., tuple[Array, Any]],
    score_bucket: Callable[..., Array],
    apply_update: Callable[..., tuple[Any, Any]],
    optimizer: optax.GradientTransformation,
    trainable: Any,
    tournament_attempts: int,
    tournament_steps: int,
    tournament_lr: float,
    key: Array,
    length_mask_fraction: Array,
    top_k: int,
) -> list[tuple[float, Any]]:
    """Warm-start selection: train several fresh inits briefly, keep the best ``top_k``.

    Each candidate is re-initialised from its own subkey, trained for
    ``tournament_steps`` steps at ``tournament_lr``, then scored with a
    forward-only pass. The score is the data term — including any
    configured trajectory penalty, which the phases also charge — and
    *excluding* the bound penalty: that regulariser is charged
    once per step regardless of the data, so scoring on it too would let a
    candidate win by drifting where the penalty likes rather than by
    fitting.

    Returns the ``top_k`` candidates ranked ascending by score. Ties keep
    the earlier attempt (stable sort), so the result stays a deterministic
    function of ``key``. ``top_k=1`` reproduces the single-best behaviour
    the main loop uses.

    An attempt that raises a diffrax error or a non-finite score is dropped
    and the next key tried. If every attempt fails, the original
    ``predictors`` come back with score ``inf`` and a ``RuntimeWarning``.
    """
    last_error: BaseException | None = None
    scored: list[tuple[float, Any]] = []
    for attempt in range(tournament_attempts):
        attempt_key = fold(key, f"tournament_attempt_{attempt}")
        try:
            # One subkey per Module leaf, so identical-shape sibling
            # predictors get genuinely different fresh weights.
            candidate: Any = reinitialize_pytree_with_key(predictors, attempt_key)
            opt_state = optimizer.init(eqx.filter(candidate, trainable))
            if _optimiser_state_supports_lr(opt_state):
                opt_state.hyperparams["learning_rate"] = jnp.asarray(tournament_lr)
            else:
                warnings.warn(
                    "tournament_lr has no effect: the optimiser state carries no "
                    "injectable hyperparams. Wrap the optimiser in "
                    "optax.inject_hyperparams to make tournament_lr apply.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            for _ in range(tournament_steps):
                _loss, acc_grads, n_buckets = _training_step(
                    candidate, dataset, bucket_step, length_mask_fraction, trainable
                )
                denom = float(max(n_buckets, 1))

                def _normalised(g: Array, _d: float = denom) -> Array:
                    return g / _d

                avg_grads = jax.tree.map(_normalised, acc_grads)
                candidate, opt_state = apply_update(candidate, avg_grads, opt_state)

            # Forward-only scorer, not bucket_step, whose discarded backward
            # pass roughly tripled the cost of a scoring sweep. Scores the
            # data term (including any configured trajectory penalty, as the
            # phases charge it), so a candidate wins on fit.
            score = jnp.asarray(0.0)
            for bp in dataset.bucket_payloads:
                score = score + score_bucket(candidate, bp, length_mask_fraction)
            score = score / max(len(dataset.bucket_payloads), 1)
            score_value = float(score)
            if not math.isfinite(score_value):
                raise FloatingPointError(f"non-finite tournament loss: {score_value}")
            scored.append((score_value, candidate))
        except _TOURNAMENT_FAILURES as exc:
            # Narrow on purpose: only a diffrax error and a non-finite loss
            # are attempt failures. Catching everything also swallowed shape
            # bugs in the user's simulate_fn and framework bugs, reporting
            # them as one warning while training carried on unmodified.
            last_error = exc
            continue

    if not scored:
        warnings.warn(
            "tournament: all attempts failed; falling back to initial predictors. "
            f"Last failure: {last_error!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return [(math.inf, predictors)]

    scored.sort(key=lambda pair: pair[0])  # stable, so ties keep the earlier attempt
    return scored[:top_k]


def _tournament_enabled(config: OptaxTrainingConfig) -> bool:
    """The tournament is enabled implicitly, never by its own flag.

    One attempt has nothing to choose between, and zero steps trains no
    candidate, so either alone makes it a no-op.
    """
    return config.tournament_attempts > 1 and config.tournament_steps > 0


def _horizon_changed(config: OptaxTrainingConfig, phase_idx: int) -> bool:
    """Does this phase score a different slice of each trajectory than the last?

    ``length_schedule`` masks the loss to a prefix, so a change here means
    the losses either side of the boundary measure different quantities.
    Phase 0 has no predecessor and so never counts as a change.
    """
    if phase_idx == 0:
        return False
    return config.length_schedule[phase_idx] != config.length_schedule[phase_idx - 1]


def _has_weak_scalar_trainable(predictors: Any, trainable: Any) -> bool:
    """Does the trainable partition hold a weak-typed 0-d array leaf?

    JAX's compiled cache keys 0-d array leaves by value *and* weak type,
    so the first optimiser update flips a weak scalar leaf
    (``jnp.asarray(1.0)``, the default-constructed
    ``BoundScaler.temperature``) to strong-typed and invalidates every
    kernel that reads the predictor once. Strong-typed or frozen scalars
    never flip and never retrace (``tests/_harness.py``'s ``OmegaPredictor``
    documents the same rule). This is the predicate that decides whether
    :func:`_warmup_compile`'s settle cycle is worth paying.
    """
    for leaf in jax.tree.leaves(eqx.filter(predictors, trainable)):
        if isinstance(leaf, jax.Array) and leaf.ndim == 0 and jax.typeof(leaf).weak_type:
            return True
    return False


def _warmup_compile(
    predictors: Any,
    bucket_payloads: tuple[BucketPayload, ...],
    *,
    bucket_step: Callable[..., tuple[Array, Any]],
    ui: TrainingUI,
    length_mask_fraction: Array,
    penalty_step: Callable[..., tuple[Array, Any]] | None = None,
    optimizer: optax.GradientTransformation | None = None,
    step_update: Callable[..., tuple[Any, Any]] | None = None,
    trainable: Any = None,
    penalty_points: tuple[Array, ...] = (),
    score_bucket: Callable[..., Array] | None = None,
    settle: bool = False,
) -> None:
    """Force one trace per kernel before the run proper starts.

    Compilation dominates the first steps and can take tens of seconds per
    shape. Paying it here, bracketed by the compile events, is what stops a
    progress bar sitting at zero and making the run look hung. The warm-up
    loss is discarded; only the populated jit cache matters.

    With the trainer's ``optimizer``/``step_update``/``penalty_step``
    supplied, the non-bucket kernels are compiled here too, so the run
    starts with every kernel warm.

    ``settle=true`` additionally runs one update-and-re-evaluate round on
    the updated predictors. This is the weak-scalar safety net: JAX's
    compiled cache keys 0-d array leaves by value *and* weak type, so the
    first optimiser update flips a weak-typed scalar leaf in the user's own
    tree (``jnp.asarray(1.0)``) to strong-typed and would retrace every
    kernel once — seconds of ODE recompile mid-run. The framework's own
    scalars are strong-typed at construction
    (:class:`~hybridmodels.predictors.BoundScaler`), so the stock trainer
    requests the settle only when :func:`_has_weak_scalar_trainable` finds
    a weak trainable scalar in the supplied tree; strong-typed or frozen
    trees keep the one-trace-per-shape contract exactly. Discovered and
    verified empirically — see ``scripts/bench_hotpath.py``.
    """
    total_buckets = len(bucket_payloads)
    last_grads: Any = None
    for idx, bp in enumerate(bucket_payloads):
        bucket_shape = (int(bp.ts.shape[0]), int(bp.ts.shape[1]))
        ui.on_compile_start(bucket_idx=idx, bucket_shape=bucket_shape)
        warm_loss, warm_grads = bucket_step(predictors, bp, length_mask_fraction)
        jax.block_until_ready(warm_loss)  # type: ignore[no-untyped-call]
        last_grads = warm_grads
        ui.on_compile_progress(bucket_idx=idx, total_buckets=total_buckets)
        ui.on_compile_done(bucket_idx=idx)

    if (
        step_update is None
        or optimizer is None
        or trainable is None
        or penalty_step is None
    ):
        return

    opt_state = optimizer.init(eqx.filter(predictors, trainable))
    zero_weight = jnp.asarray(0.0)
    n_buckets_arg = jnp.asarray(len(bucket_payloads), dtype=jnp.int32)
    # Compile the loop's non-bucket kernels here (one call each) so the run
    # starts with every kernel warm. ``last_grads`` is a real gradient from
    # the warm-up pass; the values are irrelevant, only the shapes.
    warm_penalty, _ = penalty_step(predictors, zero_weight, penalty_points)
    jax.block_until_ready(warm_penalty)  # type: ignore[no-untyped-call]
    updated, opt_state = step_update(
        predictors, last_grads, last_grads, n_buckets_arg, opt_state
    )
    jax.block_until_ready(updated)  # type: ignore[no-untyped-call]
    # The tournament's forward-only scorer gets its first trace here too.
    if score_bucket is not None:
        for bp in bucket_payloads:
            warm_score = score_bucket(predictors, bp, length_mask_fraction)
            jax.block_until_ready(warm_score)  # type: ignore[no-untyped-call]

    if not settle:
        return

    # One update round, then re-trace every kernel on the updated (strong)
    # tree, absorbing the weak→strong retrace into the compile bracket.
    for bp in bucket_payloads:
        warm_loss, _ = bucket_step(updated, bp, length_mask_fraction)
        jax.block_until_ready(warm_loss)  # type: ignore[no-untyped-call]
    warm_penalty, _ = penalty_step(updated, zero_weight, penalty_points)
    jax.block_until_ready(warm_penalty)  # type: ignore[no-untyped-call]
    updated, opt_state = step_update(
        updated, last_grads, last_grads, n_buckets_arg, opt_state
    )
    jax.block_until_ready(updated)  # type: ignore[no-untyped-call]
    # The tournament's forward-only scorer reads the same post-update
    # predictors; settle it too so a scoring sweep never re-traces.
    if score_bucket is not None:
        for bp in bucket_payloads:
            warm_score = score_bucket(updated, bp, length_mask_fraction)
            jax.block_until_ready(warm_score)  # type: ignore[no-untyped-call]
    ui.on_compile_progress(
        bucket_idx=total_buckets, total_buckets=total_buckets
    )


def _begin_phase(
    phase_idx: int,
    predictors: Any,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    *,
    config: OptaxTrainingConfig,
    trainable: Any,
) -> tuple[optax.GradientTransformation, Any]:
    """Apply the phase-boundary optimiser policy and return the pair to run with.

    A phase either rebuilds the optimiser and discards its state, or keeps the
    live state and pushes the new learning rate into it. Phase 0 passes
    through untouched: its optimiser was built by the caller, and the
    tournament may already have trained against it.

    Both ``(optimizer, opt_state)`` come back together because a reset
    invalidates them jointly; :func:`_run_phases` rebuilds the fused
    ``step_update`` whenever the returned optimiser is a new instance.
    """
    if phase_idx == 0:
        return optimizer, opt_state

    if config.reset_optimiser_state[phase_idx]:
        optimizer = _build_optimizer(config.optimizer[phase_idx], config.lr[phase_idx])
        return optimizer, optimizer.init(eqx.filter(predictors, trainable))

    # No reset, so only the learning rate moves. ``__post_init__`` has already
    # refused a phase that changes the optimiser without a reset, so the live
    # state still belongs to the optimiser this phase names. A raw
    # ``optax.GradientTransformation`` has no injectable ``hyperparams``, so
    # an lr change on one is refused here rather than silently ignored.
    if config.lr[phase_idx] != config.lr[phase_idx - 1]:
        if not _optimiser_state_supports_lr(opt_state):
            raise ValueError(
                "OptaxTrainingConfig: phase "
                f"{phase_idx} changes lr with a raw optax.GradientTransformation "
                "that cannot be re-hyperparametrised. Set "
                f"reset_optimiser_state[{phase_idx}]=True to rebuild it, or pass "
                "an optimiser name / factory so the learning rate stays injectable."
            )
        opt_state.hyperparams["learning_rate"] = jnp.asarray(config.lr[phase_idx])
    return optimizer, opt_state


class _BestTracker:
    """Lowest-data-loss snapshot and the patience counter, held together.

    Host-side only. Never pass this into a jitted function: ``eqx.filter_jit``
    would treat it as a static argument and hash it by identity, retracing
    once per instance.

    It replaces three loop variables that always had to move together and
    carried two invariants between them that previously lived only in
    comments: the snapshot must be the parameters the loss was measured at,
    and both the minimum and the counter are scoped to a horizon rather than
    to the whole run.
    """

    def __init__(self, predictors: Any) -> None:
        self.best_loss = float("inf")
        self.best_predictors = predictors
        self.steps_since_improvement = 0

    def begin_phase(self, predictors: Any, *, horizon_changed: bool) -> None:
        """Phase-boundary bookkeeping for both pieces of state.

        The patience counter always resets. A plateau at the end of one phase
        would otherwise carry over and kill the next after a single step,
        before its fresh learning rate had any chance to act.

        The running minimum resets only when the scored horizon changed.
        Losses over a prefix and losses over the full window are not
        comparable, and a single minimum across both lands in the shortest
        phase, so ``restore_best`` would hand back the least-trained model in
        the run.
        """
        self.steps_since_improvement = 0
        if horizon_changed:
            self.best_loss = float("inf")
            self.best_predictors = predictors

    def update(self, loss: float, predictors: Any) -> None:
        """Record ``loss``, which must have been measured at ``predictors``.

        Pass the **pre-update** parameters. Passing the ones the optimiser
        just produced hands back a model whose loss is not the reported
        minimum, and ``loss_history`` then describes a different model.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_predictors = predictors
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def out_of_patience(self, patience: int) -> bool:
        """``patience`` consecutive steps with no new minimum. ``0`` disables it."""
        return patience > 0 and self.steps_since_improvement >= patience


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
    state_to_output: Callable[[Array], Array],
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

    ``state_to_output`` is the pure mapping ``[T, S] -> [T, D]`` from full
    simulator state to observed channels. It is a property of the model,
    passed here rather than stored on the ``Dataset``.

    ``trainable`` is a boolean mask matching ``predictors``. Omitting it
    defaults to :func:`hybridmodels.trainable.trainable_mask`, which marks
    every inexact-array leaf trainable. Pass a custom mask, usually from
    the freezers in ``hybridmodels.trainable``, to hold leaves fixed;
    freezing ``BoundScaler`` leaves is the common case.

    Returns
    -------
    tuple[list[float], PyTree[eqx.Module]]
        ``(loss_history, trained_predictors)``.

        ``loss_history`` is the **raw per-step loss**, one entry per
        step, concatenated across phases. It can go up. It is the data
        term plus any configured trajectory penalty (charged inside the
        bucket forward pass); the bound penalty is excluded, so a ramping
        bound-penalty weight cannot move the series and runs with
        different bound weights stay comparable, and nothing is smoothed:
        these are the values the optimiser saw.

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

    ui_ = _select_ui(ui, config.verbose)
    ui_.on_run_start(total_steps=int(sum(config.steps)), num_phases=len(config.steps))

    (
        bucket_step,
        penalty_step,
        score_bucket,
        optimizer,
        apply_update,
        step_update,
        sources,
        extras,
    ) = _build_training_kernels(
        predictors=predictors,
        dataset=dataset,
        config=config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        trainable=trainable,
        ui=ui_,
    )

    if _tournament_enabled(config):
        full_mask = jnp.asarray(1.0)
        ranked = _shared_tournament(
            predictors,
            dataset,
            bucket_step=bucket_step,
            score_bucket=score_bucket,
            apply_update=apply_update,
            optimizer=optimizer,
            trainable=trainable,
            tournament_attempts=config.tournament_attempts,
            tournament_steps=config.tournament_steps,
            tournament_lr=config.tournament_lr,
            key=fold(key, "tournament"),
            length_mask_fraction=full_mask,
            top_k=1,
        )
        predictors = ranked[0][1]

    _history, final_predictors, _final_loss = _run_phases(
        predictors,
        dataset,
        config,
        optimizer=optimizer,
        bucket_step=bucket_step,
        penalty_step=penalty_step,
        sources=sources,
        extras=extras,
        trainable=trainable,
        ui=ui_,
        step_update=step_update,
    )
    return _history, final_predictors


def _run_phases(
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    optimizer: optax.GradientTransformation,
    bucket_step: Callable[..., tuple[Array, Any]],
    penalty_step: Callable[..., tuple[Array, Any]],
    sources: tuple[PenaltyPointSource | None, ...],
    extras: tuple[Array, ...],
    trainable: Any,
    ui: TrainingUI,
    step_update: Callable[..., tuple[Any, Any]] | None = None,
) -> tuple[list[float], Any, float]:
    """Run the phase schedule from a starting ``predictors``.

    The shared main-loop body behind :func:`train_with_optax` and the
    ensemble entry points. Takes the pre-built kernels and a fresh
    optimiser; the caller owns warm-up compilation and any tournament
    selection.

    The caller is responsible for the ``on_run_start`` side of the UI
    bracket; this function fires ``on_run_end``. A caller that runs this
    body several times (the ensembles, once per member) fires
    ``on_run_start`` before each call, so every member is its own UI run.

    Returns ``(losses_history, final_predictors, final_loss)`` where
    ``final_loss`` is the loss that labels the returned predictors: the
    best-step loss when ``config.restore_best`` is set (measured exactly
    at the returned parameters), else the last step's reported loss
    (measured one update before the returned parameters, as the loop
    reports it). The ensembles rank members on this labelled loss, so a
    best-step-restored member is not reported at a loss it never had.

    ``step_update`` is the fused per-step tail built by
    :func:`_build_step_update` — average + penalty merge + optimiser
    update in one kernel. The stock callers pass the instance the
    warm-up settled; when ``None`` (or when a phase reset rebuilds the
    optimiser) it is rebuilt here from the live optimiser.
    """
    opt_state = optimizer.init(eqx.filter(predictors, trainable))
    if step_update is None:
        step_update = _build_step_update(optimizer=optimizer, trainable=trainable)
    step_update_optimizer = optimizer
    n_buckets_arg = jnp.asarray(len(dataset.bucket_payloads), dtype=jnp.int32)

    losses_history: list[float] = []
    best = _BestTracker(predictors)

    for phase_idx, n_steps in enumerate(config.steps):
        optimizer, opt_state = _begin_phase(
            phase_idx,
            predictors,
            optimizer,
            opt_state,
            config=config,
            trainable=trainable,
        )
        if optimizer is not step_update_optimizer:
            # A reset built a fresh optimiser; its fused kernel must too.
            step_update = _build_step_update(optimizer=optimizer, trainable=trainable)
            step_update_optimizer = optimizer

        length_mask_fraction = jnp.asarray(config.length_schedule[phase_idx])
        # Traced, not closed over: a Python float that changed per phase
        # would retrace ``bucket_step`` at every phase boundary. Same reason
        # ``length_mask_fraction`` is passed rather than baked in.
        penalty_weight = jnp.asarray(config.penalty_weight_for_phase(phase_idx))
        # Host-side per phase: the keep-mask must be concrete to index the
        # gathered points. A shape change across phases retraces only the
        # small penalty kernel, never ``bucket_step``.
        phase_points = select_penalty_points(
            sources, extras, config.length_schedule[phase_idx]
        )

        best.begin_phase(predictors, horizon_changed=_horizon_changed(config, phase_idx))

        ui.on_phase_start(
            phase_idx=phase_idx,
            phase_steps=int(n_steps),
            lr=float(config.lr[phase_idx]),
            optimizer=config.optimizer[phase_idx],
        )

        for step in range(int(n_steps)):
            total_loss, acc_grads, n_buckets = _training_step(
                predictors, dataset, bucket_step, length_mask_fraction, trainable
            )
            avg_penalty, penalty_grads = penalty_step(
                predictors, penalty_weight, phase_points
            )
            # Dispatch the update before blocking on the loss values. Both
            # float() calls are host syncs; reading them first left the
            # accelerator idle through the Python bookkeeping every step.
            # The fused kernel averages the accumulator, merges the penalty
            # grads (charged once per step, not once per bucket) and runs
            # the optimiser update in a single launch.
            previous_predictors = predictors
            predictors, opt_state = step_update(
                predictors, acc_grads, penalty_grads, n_buckets_arg, opt_state
            )

            # History and early stopping follow the per-bucket-averaged
            # loss (the data term plus any configured trajectory penalty):
            # tracking the bound-penalty weight too would let "best" move
            # when only the penalty weight changed.
            loss_value = float(total_loss / float(max(n_buckets, 1)))
            penalty_value = float(avg_penalty)
            losses_history.append(loss_value)
            # ``previous_predictors``, not ``predictors``: avg_data was
            # measured before the update was applied.
            best.update(loss_value, previous_predictors)

            ui.on_step_end(
                step_idx=step,
                phase_idx=phase_idx,
                loss=loss_value,
                penalty=penalty_value,
            )

            if best.out_of_patience(config.patience):
                break

        ui.on_phase_end(phase_idx=phase_idx)

    final_predictors = best.best_predictors if config.restore_best else predictors
    if not losses_history:
        final_loss = float("nan")
    elif config.restore_best:
        # ``best.best_loss`` was measured at ``best.best_predictors`` (the
        # pre-update parameters of the best step), so it labels the model
        # that is about to come back. ``history[-1]`` is the last step's
        # loss, which describes a different model.
        final_loss = best.best_loss
    else:
        final_loss = losses_history[-1]
    ui.on_run_end(final_loss=final_loss)
    return losses_history, final_predictors, final_loss


def _build_training_kernels(
    *,
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    trainable: Any,
    ui: TrainingUI,
) -> tuple[
    Callable[..., tuple[Array, Any]],
    Callable[..., tuple[Array, Any]],
    Callable[..., Array],
    optax.GradientTransformation,
    Callable[..., tuple[Any, Any]],
    Callable[..., tuple[Any, Any]],
    tuple[PenaltyPointSource | None, ...],
    tuple[Array, ...],
]:
    """Build the compiled kernels, optimiser, and warm-up the caches.

    Shared by :func:`train_with_optax` and the ensemble entry points so a
    seed/bootstrap ensemble reuses the exact same kernel construction.
    Returns ``(bucket_step, penalty_step, score_bucket, optimizer,
    apply_update, step_update, sources, extras)`` with the jit caches
    already populated by a warm-up pass; ``sources`` / ``extras`` are the
    penalty point sets
    the phases select from.

    The caller owns the ``on_run_start`` side of the UI bracket: the
    warm-up compile events below are meant to land inside a run, and the
    ensembles fire ``on_run_start`` before every member, of which this is
    the first.
    """
    bucket_payloads = dataset.bucket_payloads
    loss_fn = resolve_loss_fn(config.loss, config.channel_idx, config.channel_weights)
    penalty_fn = bound_penalty if config.penalty_fn is None else config.penalty_fn
    sources = data_penalty_points(predictors, dataset)
    extras = config.penalty_points or ()
    validate_penalty_points(
        predictors,
        sources,
        extras,
        # Coverage is the default bound penalty's contract; a custom
        # ``penalty_fn`` owns its points and needs no coverage check.
        enabled=any(w > 0.0 for w in config.penalty_weight) and config.penalty_fn is None,
    )

    bucket_step = build_bucket_step(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trainable=trainable,
        trajectory_penalty_fn=config.trajectory_penalty_fn,
        trajectory_penalty_weight=config.trajectory_penalty_weight,
    )
    penalty_step = build_penalty_step(
        penalty_fn=penalty_fn,
        trainable=trainable,
    )
    score_bucket = build_score_bucket(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trajectory_penalty_fn=config.trajectory_penalty_fn,
        trajectory_penalty_weight=config.trajectory_penalty_weight,
    )
    full_mask = jnp.asarray(1.0)
    # Settle only when the run can hit the weak→strong scalar flip: a
    # default ``trainable_mask`` treats ``BoundScaler.temperature``
    # (``jnp.asarray(1.0)``, weak-typed) as trainable, and JAX's cache keys
    # 0-d leaves by value+weak_type, so the first update would retrace
    # every kernel once (seconds-to-minutes of ODE compile) mid-run.
    # Strong-typed or frozen scalars never flip — skip the settle entirely
    # and keep the "one trace per bucket shape" contract untouched.
    settle = _has_weak_scalar_trainable(predictors, trainable)
    optimizer = _build_optimizer(config.optimizer[0], config.lr[0])
    apply_update = build_apply_update(optimizer, trainable)
    step_update = _build_step_update(optimizer=optimizer, trainable=trainable)
    settle_points = select_penalty_points(sources, extras, 1.0)
    _warmup_compile(
        predictors,
        bucket_payloads,
        bucket_step=bucket_step,
        ui=ui,
        length_mask_fraction=full_mask,
        penalty_step=penalty_step,
        optimizer=optimizer,
        step_update=step_update,
        trainable=trainable,
        penalty_points=settle_points,
        score_bucket=score_bucket if _tournament_enabled(config) else None,
        settle=settle,
    )
    return (
        bucket_step,
        penalty_step,
        score_bucket,
        optimizer,
        apply_update,
        step_update,
        sources,
        extras,
    )


def train_seed_ensemble(
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    n_seeds: int,
    k_best: int | None = None,
    trainable: Any = None,
    key: Array,
    ui: TrainingUI | None = None,
) -> list[tuple[float, Any]]:
    """Train a seed ensemble: rank warm starts, fully train the best ``k_best``.

    The **tournament** already ranks fresh initialisations cheaply (a few
    warm-up steps each, scored forward-only). This reuses that ranking to
    pick which seeds deserve a full training run, instead of fully training
    every seed: with ``n_seeds`` tournament attempts it keeps the best
    ``k_best`` (default: all ``n_seeds``) and runs the full phase schedule
    on each.

    Returns the fully trained members as a list of ``(final_loss,
    predictors)`` ranked ascending by final loss — the same container
    shape as the single ``predictors`` you passed in, so the result is an
    ensemble of model pytrees ready for :func:`ensemble_predictions`.
    ``final_loss`` labels the returned member: the best-step loss when
    ``config.restore_best`` is set (measured exactly at the returned
    parameters), else the last step's reported loss (measured one update
    before, as the loop reports it).

    ``config.tournament_steps`` controls how much warm-up each seed gets
    before ranking; the default ``0`` ranks by the initialisation score
    alone (still a valid, cheapest ranking). ``config`` is otherwise used
    exactly as in :func:`train_with_optax`.

    Each member is its own UI run: the warm-up compile and the tournament
    land inside the first member's bracket, and ``on_run_start`` /
    ``on_run_end`` are fired once per member, so a live dashboard shows
    each member's phases rather than freezing after the first.
    """
    if trainable is None:
        trainable = trainable_mask(predictors)
    if k_best is None:
        k_best = n_seeds
    if not (0 < k_best <= n_seeds):
        raise ValueError(
            f"train_seed_ensemble: k_best={k_best} must satisfy 0 < k_best <= n_seeds={n_seeds}"
        )

    bucket_payloads = dataset.bucket_payloads
    if not bucket_payloads:
        raise ValueError("train_seed_ensemble: dataset has no bucket payloads")

    ui_ = _select_ui(ui, config.verbose)
    ui_.on_run_start(total_steps=int(sum(config.steps)), num_phases=len(config.steps))
    (
        bucket_step,
        penalty_step,
        score_bucket,
        optimizer,
        apply_update,
        step_update,
        sources,
        extras,
    ) = _build_training_kernels(
        predictors=predictors,
        dataset=dataset,
        config=config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        trainable=trainable,
        ui=ui_,
    )

    full_mask = jnp.asarray(1.0)
    ranked = _shared_tournament(
        predictors,
        dataset,
        bucket_step=bucket_step,
        score_bucket=score_bucket,
        apply_update=apply_update,
        optimizer=optimizer,
        trainable=trainable,
        tournament_attempts=n_seeds,
        tournament_steps=config.tournament_steps,
        tournament_lr=config.tournament_lr,
        key=fold(key, "seed_ensemble_tournament"),
        length_mask_fraction=full_mask,
        top_k=k_best,
    )

    trained: list[tuple[float, Any]] = []
    for member_idx, (_seed_score, candidate) in enumerate(ranked):
        if not math.isfinite(_seed_score):
            continue
        # Every member is its own UI run: each gets a fresh on_run_start
        # (the first also brackets the warm-up compile), and _run_phases
        # fires the matching on_run_end.
        if member_idx > 0:
            ui_.on_run_start(total_steps=int(sum(config.steps)), num_phases=len(config.steps))
        _history, final, final_loss = _run_phases(
            candidate,
            dataset,
            config,
            optimizer=optimizer,
            bucket_step=bucket_step,
            penalty_step=penalty_step,
            sources=sources,
            extras=extras,
            trainable=trainable,
            ui=ui_,
            step_update=step_update,
        )
        trained.append((final_loss, final))

    if not trained:
        warnings.warn(
            "train_seed_ensemble: every seed failed its warm-up; returning the "
            "input predictors as a single-member ensemble.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Close the run bracket opened by the per-member on_run_start
        # above, so a live dashboard does not hang after the fallback.
        ui_.on_run_end(final_loss=math.inf)
        return [(math.inf, predictors)]

    trained.sort(key=lambda pair: pair[0])
    return trained


def train_bootstrap_ensemble(
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    n_bootstraps: int,
    n_seeds: int = 1,
    k_best: int | None = None,
    trainable: Any = None,
    key: Array,
    ui: TrainingUI | None = None,
) -> list[tuple[float, Any]]:
    """Train a bootstrap ensemble: one (or a seed-set of) model(s) per resample.

    For each of ``n_bootstraps`` resampled datasets (via
    :func:`hybridmodels.make_bootstrap_dataset`), trains a model. With
    ``n_seeds > 1`` each resample's member is itself seed-selected by
    :func:`train_seed_ensemble` (so every member both sees different data
    *and* is a good seed); with ``n_seeds == 1`` each resample contributes
    one fresh re-initialisation trained via the same phase schedule as
    :func:`train_with_optax`.

    Returns the members as a list of ``(final_loss, predictors)`` ranked
    ascending by final loss across **all** bootstrap samples. ``k_best``
    trims the ensemble to its best members overall (default: keep
    ``n_bootstraps * max(n_seeds, 1)``). ``final_loss`` labels the
    returned member: the best-step loss when ``config.restore_best`` is
    set (measured exactly at the returned parameters), else the last
    step's reported loss (measured one update before, as the loop
    reports it).

    Every resample and every training run is folded off the one ``key``, so
    the whole ensemble is deterministic given it. Each member is its own
    UI run, bracketed by ``on_run_start`` / ``on_run_end``.
    """
    if n_bootstraps < 1:
        raise ValueError(f"train_bootstrap_ensemble: n_bootstraps must be >= 1, got {n_bootstraps}")
    if n_seeds < 1:
        raise ValueError(f"train_bootstrap_ensemble: n_seeds must be >= 1, got {n_seeds}")
    if trainable is None:
        trainable = trainable_mask(predictors)

    bucket_payloads = dataset.bucket_payloads
    if not bucket_payloads:
        raise ValueError("train_bootstrap_ensemble: dataset has no bucket payloads")

    ui_ = _select_ui(ui, config.verbose)
    ui_.on_run_start(total_steps=int(sum(config.steps)), num_phases=len(config.steps))

    (
        bucket_step,
        penalty_step,
        score_bucket,
        optimizer,
        apply_update,
        step_update,
        sources,
        extras,
    ) = _build_training_kernels(
        predictors=predictors,
        dataset=dataset,
        config=config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        trainable=trainable,
        ui=ui_,
    )

    all_trained: list[tuple[float, Any]] = []
    for s in range(n_bootstraps):
        boot = make_bootstrap_dataset(dataset, key=fold(key, f"bootstrap_{s}"))
        sample_key = fold(key, f"bootstrap_seeds_{s}")
        full_mask = jnp.asarray(1.0)
        ranked = _shared_tournament(
            predictors,
            boot,
            bucket_step=bucket_step,
            score_bucket=score_bucket,
            apply_update=apply_update,
            optimizer=optimizer,
            trainable=trainable,
            tournament_attempts=n_seeds,
            tournament_steps=config.tournament_steps,
            tournament_lr=config.tournament_lr,
            key=fold(sample_key, "bootstrap_sample_tournament"),
            length_mask_fraction=full_mask,
            top_k=n_seeds if n_seeds > 1 else 1,
        )
        for member_idx, (_seed_score, candidate) in enumerate(ranked):
            if not math.isfinite(_seed_score):
                continue
            # Every member is its own UI run: each gets a fresh
            # on_run_start (the first also brackets the warm-up compile),
            # and _run_phases fires the matching on_run_end.
            if s > 0 or member_idx > 0:
                ui_.on_run_start(total_steps=int(sum(config.steps)), num_phases=len(config.steps))
            _history, final, final_loss = _run_phases(
                candidate,
                boot,
                config,
                optimizer=optimizer,
                bucket_step=bucket_step,
                penalty_step=penalty_step,
                # The resample changes the covariate values, so the measured
                # points are gathered from the boot sample, not the source.
                sources=data_penalty_points(predictors, boot),
                extras=extras,
                trainable=trainable,
                ui=ui_,
                step_update=step_update,
            )
            all_trained.append((final_loss, final))

    if not all_trained:
        warnings.warn(
            "train_bootstrap_ensemble: every member failed; returning the input "
            "predictors as a single-member ensemble.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Close the run bracket opened by the per-member on_run_start
        # above, so a live dashboard does not hang after the fallback.
        ui_.on_run_end(final_loss=math.inf)
        return [(math.inf, predictors)]

    all_trained.sort(key=lambda pair: pair[0])
    if k_best is not None:
        if not (0 < k_best <= len(all_trained)):
            raise ValueError(
                f"train_bootstrap_ensemble: k_best={k_best} must satisfy "
                f"0 < k_best <= ensemble size {len(all_trained)}"
            )
        all_trained = all_trained[:k_best]
    return all_trained
