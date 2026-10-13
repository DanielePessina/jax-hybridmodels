"""Gradient training loop for hybrid mechanistic models, driven by Optax.

One training **step** is one full pass over every bucket: per-bucket
gradients from ``bucket_step`` (jitted, one trace per bucket shape),
accumulated and averaged, then a single ``optimizer.update``. A bucket is
not a step, and there is no minibatching.

What is minimised (loss, channel weights, penalties) is an
:class:`~jaxhybridmodels.Objective`, passed as ``objective=``; the config
here holds only how to minimise it.

A run is a sequence of **phases**, contiguous blocks of steps sharing
hyperparameters. Each is a ``TrainingPhase`` with its own step count,
learning rate (a float or an optax schedule), optimizer, horizon, reset
flag and optional penalty-weight override, held in
``OptaxTrainingConfig.phases``. Phases are how a run
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

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from jaxhybridmodels.data import (
    BucketPayload,
    Data,
    Dataset,
    make_bootstrap_dataset,
)
from jaxhybridmodels.objective import Objective, ResolvedObjective, resolve_objective
from jaxhybridmodels.penalties import (
    PenaltyPointSource,
    data_penalty_points,
    select_penalty_points,
)
from jaxhybridmodels.predictors.base import reinitialize_pytree_with_key
from jaxhybridmodels.results import FitHistory, FitResult
from jaxhybridmodels.rng import fold
from jaxhybridmodels.solver import SolverConfig
from jaxhybridmodels.trainable import FreezeSpec, trainable_mask
from jaxhybridmodels.training.kernels import (
    build_apply_update,
    build_bucket_step,
    build_dataset_score,
    build_penalty_step,
    build_score_bucket,
)
from jaxhybridmodels.ui.base import SilentUI, TrainingUI, close_run_on_error
from jaxhybridmodels.ui.optax import RichTrainingUI

if TYPE_CHECKING:
    from jaxhybridmodels.model import HybridModel

# An optimizer entry may be a registered name ("adamw"), a factory taking
# ``learning_rate`` and returning an ``optax.GradientTransformation``, or a
# ready-made transformation instance. Names and factories are wrapped in
# ``optax.inject_hyperparams`` so a phase boundary can move the learning
# rate; a raw instance has its own state and cannot be re-hyperparametrised.
OptimizerSpec = str | Callable[[float], optax.GradientTransformation] | optax.GradientTransformation

# R-T7 allows a tournament attempt to fail on a diffrax error or a
# non-finite loss. Anything else is a bug in user or framework code and
# must reach the user instead of being retried with a different seed.
_TOURNAMENT_FAILURES: tuple[type[BaseException], ...] = (
    FloatingPointError,
    jax.errors.JaxRuntimeError,
    eqx.EquinoxRuntimeError,
)


@dataclass(frozen=True)
class TrainingPhase:
    """One contiguous block of Optax steps with its own schedule settings.

    Attributes
    ----------
    steps : int
        Step budget for the phase. At least 1.
    lr : float | optax.Schedule
        Learning rate: a finite non-negative float, or an optax schedule
        ``step -> lr`` whose step count starts at 0 at the start of this
        phase, whatever ran before it. Chain several curves inside one
        phase with ``optax.join_schedules``. Any callable works: one JAX
        cannot trace, such as ``lambda s: 0.1 if s < 2 else 0.01``, is
        evaluated step by step on the host. A float is applied to the live
        optimiser state unless the phase also sets ``reset_optimizer``. A
        schedule needs an optimiser name or factory; a ready-made
        ``optax.GradientTransformation`` owns its learning rate, so pairing
        one with a schedule raises.
    optimizer : OptimizerSpec
        A registered name (``"adamw"``, ``"adabelief"``), a factory taking
        ``learning_rate`` and returning an ``optax.GradientTransformation``
        (e.g. ``optax.adamw``, or ``lambda learning_rate: optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))``), or
        a ready-made transformation instance. Names and factories are
        wrapped in ``optax.inject_hyperparams``, so a phase boundary or a
        schedule can move the learning rate without a rebuild. A raw
        instance cannot be re-hyperparametrised, so a phase that changes
        ``lr`` with a raw instance must set ``reset_optimizer``. A phase
        that changes the optimiser must set it too, because optimiser state
        belongs to the optimiser that built it.
    horizon : float
        Fraction of each experiment's timeline the loss looks at, in
        ``(0, 1]``. It masks the **loss**, never the integration: the
        solver still runs the full trajectory, and only the first
        ``horizon`` of the observation times is scored, which stops a
        long-horizon divergence from drowning the gradient. Being a
        runtime mask rather than a shape change, a phase boundary costs no
        recompile.
    reset_optimizer : bool
        Rebuild the optimiser and discard its state at the start of this
        phase. Set it when switching optimiser, and when a horizon change
        has made the accumulated momentum wrong.
    penalty_weight : float | None
        Override of ``Objective.penalty_weight`` for this phase. ``None``
        (the default) uses the objective's weight; ``0.0`` switches the
        bound penalty off for the phase.
    """

    steps: int
    # ``steps`` may be positional (``TrainingPhase(200, lr=1e-2)``); the rest
    # are keyword-only so a stray positional cannot land in the wrong field.
    _: KW_ONLY
    lr: float | optax.Schedule = 1e-3
    optimizer: OptimizerSpec = "adamw"
    horizon: float = 1.0
    reset_optimizer: bool = False
    penalty_weight: float | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"TrainingPhase.steps must be at least 1; got {self.steps}")
        if callable(self.lr):
            if isinstance(self.optimizer, optax.GradientTransformation):
                raise ValueError(
                    "TrainingPhase.lr is a schedule but optimizer is a ready-made "
                    "optax.GradientTransformation, which owns its learning rate. Pass an "
                    "optimizer name or a factory taking learning_rate instead."
                )
        elif not math.isfinite(float(self.lr)) or float(self.lr) < 0.0:
            raise ValueError(
                f"TrainingPhase.lr must be a finite non-negative float or an optax "
                f"schedule; got {self.lr}"
            )
        if not 0.0 < float(self.horizon) <= 1.0:
            raise ValueError(f"TrainingPhase.horizon must lie in (0, 1]; got {self.horizon}")
        if self.penalty_weight is not None:
            weight = float(self.penalty_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"TrainingPhase.penalty_weight must be finite and non-negative; got {weight}"
                )


@dataclass(frozen=True, kw_only=True)
class OptaxTrainingConfig:
    """Search settings for :func:`train_with_optax` and Optax ``HybridModel.fit``.

    What is minimised (loss, channel weights, penalties) lives in
    :class:`~jaxhybridmodels.Objective`, passed separately as
    ``objective=``. This config says how to minimise it.

    A single phase is spelled with ``steps=`` (plus optional ``lr=`` and
    ``optimizer=``). A schedule is spelled with ``phases=``, a tuple of
    :class:`TrainingPhase`. The two forms do not mix; either way the config
    stores ``phases``::

        OptaxTrainingConfig(steps=500, lr=1e-3)
        OptaxTrainingConfig(steps=500, lr=optax.cosine_decay_schedule(1e-2, 500))
        OptaxTrainingConfig(
            phases=(
                TrainingPhase(steps=200, lr=3e-2, horizon=0.5),
                TrainingPhase(steps=200, lr=5e-3, reset_optimizer=True),
            )
        )

    Attributes
    ----------
    phases : tuple[TrainingPhase, ...]
        The schedule, run in order. See :class:`TrainingPhase` for the
        per-phase settings.
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
        the phase learning rates.
    patience : int
        Consecutive steps without a new best tracked loss before the current
        phase stops early. The tracked loss is the data term plus any
        trajectory penalty, excluding the fixed-point bound penalty. Counted
        within a phase and reset at every phase boundary, so a plateau at the
        end of one phase cannot kill the next before its new learning rate
        acts. ``0`` disables early stopping.
    restore_best : bool
        Return the predictors from the lowest tracked-loss step instead of
        the last one. The running minimum resets whenever the horizon
        changes, since tracked losses over different horizons are not
        comparable and the shortest horizon would otherwise always own the
        minimum.
    verbose : bool
        Selects ``RichTrainingUI`` over ``SilentUI`` when ``ui=None``. An
        explicit ``ui=...`` argument always wins.
    """

    # Single-phase shorthand. Init-only: after construction the schedule
    # lives in ``phases`` alone, so these class attributes read as None.
    steps: InitVar[int | None] = None
    lr: InitVar[float | optax.Schedule | None] = None
    optimizer: InitVar[OptimizerSpec | None] = None
    phases: tuple[TrainingPhase, ...] = ()
    tournament_attempts: int = 1
    tournament_steps: int = 0
    tournament_lr: float = 1e-4
    patience: int = 0
    restore_best: bool = True
    verbose: bool = True

    def __post_init__(
        self,
        steps: int | None,
        lr: float | optax.Schedule | None,
        optimizer: OptimizerSpec | None,
    ) -> None:
        if steps is not None:
            if self.phases:
                raise ValueError("Use steps= or phases=, not both")
            # None means "not given"; fall back to the TrainingPhase defaults.
            default = TrainingPhase(steps)
            phase = TrainingPhase(
                steps,
                lr=default.lr if lr is None else lr,
                optimizer=default.optimizer if optimizer is None else optimizer,
            )
            object.__setattr__(self, "phases", (phase,))
        elif lr is not None or optimizer is not None:
            raise ValueError("With phases=, set lr and optimizer per TrainingPhase")
        object.__setattr__(self, "phases", tuple(self.phases))
        if not self.phases:
            raise ValueError("Supply steps= for one phase or phases= for a schedule")
        _validate_optimizer_transitions(self)
        _validate_tournament(self)

    @property
    def total_steps(self) -> int:
        """Step budget summed over every phase."""
        return sum(phase.steps for phase in self.phases)


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
    for idx, (before, phase) in enumerate(itertools.pairwise(config.phases), start=1):
        if phase.optimizer != before.optimizer and not phase.reset_optimizer:
            raise ValueError(
                f"OptaxTrainingConfig: phase {idx} changes optimizer from "
                f"{before.optimizer!r} to {phase.optimizer!r} without reset_optimizer=True. "
                "Optimiser state is specific to the optimiser that built it, so switching "
                "without a reset would keep running the previous one."
            )


def _validate_tournament(config: OptaxTrainingConfig) -> None:
    """Validate the optional tournament budget and its learning rate."""
    if config.tournament_attempts < 1:
        raise ValueError("OptaxTrainingConfig.tournament_attempts must be at least 1")
    if config.tournament_steps < 0:
        raise ValueError("OptaxTrainingConfig.tournament_steps must be non-negative")
    if not math.isfinite(config.tournament_lr) or config.tournament_lr < 0.0:
        raise ValueError("OptaxTrainingConfig.tournament_lr must be finite and non-negative")
    if config.patience < 0:
        raise ValueError("OptaxTrainingConfig.patience must be non-negative")


def _build_optimizer(spec: OptimizerSpec, lr: float) -> optax.GradientTransformation:
    """Build the per-phase optimiser, honouring names, factories, and raw instances.

    A name string (``"adamw"``/``"adabelief"``) and a factory taking
    ``learning_rate`` are both wrapped in ``optax.inject_hyperparams``, so
    ``_begin_phase`` and a scheduled phase can move the learning rate in the
    live optimiser state without a rebuild. ``lr`` is the starting value.
    A raw ``optax.GradientTransformation`` instance is returned as-is: its
    state is not re-hyperparametrisable, so a phase that changes ``lr`` with
    a raw instance must also reset the optimiser.
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


def _start_lr(lr: float | optax.Schedule) -> float:
    """The learning rate a phase starts at: the float, or the schedule at step 0."""
    if callable(lr):
        return float(jnp.asarray(cast(optax.Schedule, lr)(0)))
    return float(lr)


def _scheduled_lrs(phase: TrainingPhase) -> Array | None:
    """Every step's learning rate for a scheduled phase, ``None`` for a float one.

    The schedule is evaluated once per phase, vectorised over the phase's
    own step indices (0 at the phase start), rather than once per step on
    the host. The loop then writes one entry into the optimiser state before
    each step.
    """
    if not callable(phase.lr):
        return None
    schedule = cast(optax.Schedule, phase.lr)
    try:
        return jax.vmap(schedule)(jnp.arange(phase.steps))
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerIntegerConversionError):
        # A plain Python schedule (``lambda s: 0.1 if s < 2 else 0.01``)
        # branches on the step, which a traced index cannot do. Evaluate it
        # on the host instead, once per step.
        return jnp.stack([jnp.asarray(schedule(i)) for i in range(phase.steps)])


def _set_lr(opt_state: Any, value: Any) -> None:
    """Write ``value`` into an injectable optimiser state's learning rate, in place.

    The new value takes the dtype of the one it replaces. ``inject_hyperparams``
    stores a strong-typed float32, and a weak-typed replacement (what
    ``jnp.asarray(0.1)`` gives) would change the jit cache key and retrace
    the fused update kernel.
    """
    current = opt_state.hyperparams["learning_rate"]
    opt_state.hyperparams["learning_rate"] = jnp.asarray(value, dtype=current.dtype)


def _build_step_update(
    *,
    optimizer: optax.GradientTransformation,
    trainable: Any,
) -> Callable[..., tuple[Any, Any]]:
    """Fused per-step tail: average, merge the bound penalty, and optimise.

    One jitted ``step_update(predictors, acc_grads, penalty_grads,
    n_buckets, opt_state) -> (predictors, opt_state)`` that normalises by
    bucket count, adds the penalty grads and applies the optimiser update.
    Each eager per-leaf tree operation dispatches one compiled kernel per
    parameter leaf. On CPU that measured ~2 ms per tree walk for a small
    MLP, several times the ODE solve itself, so folding them into the
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
        avg_grads = jax.tree.map(lambda a, p: a / n_buckets + p, acc_grads, penalty_grads)
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

    Returns ``(total_loss, acc_grads, n_buckets)``, the **raw** sum of
    per-bucket losses and gradients, not yet divided. The caller merges
    the bound-penalty gradients and normalises inside the fused
    ``step_update`` kernel (:func:`_build_step_update`), so the per-step
    Python side never walks the parameter tree. The first bucket's
    gradients seed the accumulator directly instead of adding to a
    ``zeros_like`` tree, which saves one per-leaf pass per step.

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
    forward-only pass. The score is the data term, including any
    configured trajectory penalty (the phases charge it too) and
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
            fresh = reinitialize_pytree_with_key(predictors, attempt_key)
            # Re-init draws every Module leaf, frozen ones included. Put the
            # frozen leaves back, so a frozen pre-trained subtree (or a
            # default-frozen BoundScaler) keeps its values instead of being
            # held fixed at random ones.
            candidate: Any = eqx.combine(
                eqx.filter(fresh, trainable),
                eqx.filter(predictors, trainable, inverse=True),
            )
            opt_state = optimizer.init(eqx.filter(candidate, trainable))
            if _optimiser_state_supports_lr(opt_state):
                _set_lr(opt_state, tournament_lr)
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

    ``TrainingPhase.horizon`` masks the loss to a prefix, so a change here means
    the losses either side of the boundary measure different quantities.
    Phase 0 has no predecessor and so never counts as a change.
    """
    if phase_idx == 0:
        return False
    return config.phases[phase_idx].horizon != config.phases[phase_idx - 1].horizon


def _has_weak_scalar_trainable(predictors: Any, trainable: Any) -> bool:
    """Does the trainable partition hold a weak-typed 0-d array leaf?

    JAX's compiled cache keys 0-d array leaves by value *and* weak type,
    so the first optimiser update flips a weak scalar leaf
    (``jnp.asarray(1.0)``, the default-constructed
    ``BoundScaler.temperature``) to strong-typed and invalidates every
    kernel that reads the predictor once. Strong-typed or frozen scalars
    never flip and never retrace. This is the predicate that decides whether
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
    kernel once, costing seconds of ODE recompile mid-run. The framework's own
    scalars are strong-typed at construction
    (:class:`~jaxhybridmodels.predictors.BoundScaler`), so the stock trainer
    requests the settle only when :func:`_has_weak_scalar_trainable` finds
    a weak trainable scalar in the supplied tree; strong-typed or frozen
    trees keep the one-trace-per-shape contract exactly. Discovered and
    verified empirically; see ``scripts/bench_hotpath.py``.
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

    if step_update is None or optimizer is None or trainable is None or penalty_step is None:
        return

    opt_state = optimizer.init(eqx.filter(predictors, trainable))
    zero_weight = jnp.asarray(0.0)
    n_buckets_arg = jnp.asarray(len(bucket_payloads), dtype=jnp.int32)
    # Compile the loop's non-bucket kernels here (one call each) so the run
    # starts with every kernel warm. ``last_grads`` is a real gradient from
    # the warm-up pass; the values are irrelevant, only the shapes.
    warm_penalty, _ = penalty_step(predictors, zero_weight, penalty_points)
    jax.block_until_ready(warm_penalty)  # type: ignore[no-untyped-call]
    updated, opt_state = step_update(predictors, last_grads, last_grads, n_buckets_arg, opt_state)
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
    updated, opt_state = step_update(updated, last_grads, last_grads, n_buckets_arg, opt_state)
    jax.block_until_ready(updated)  # type: ignore[no-untyped-call]
    # The tournament's forward-only scorer reads the same post-update
    # predictors; settle it too so a scoring sweep never re-traces.
    if score_bucket is not None:
        for bp in bucket_payloads:
            warm_score = score_bucket(updated, bp, length_mask_fraction)
            jax.block_until_ready(warm_score)  # type: ignore[no-untyped-call]
    ui.on_compile_progress(bucket_idx=total_buckets, total_buckets=total_buckets)


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

    phase = config.phases[phase_idx]
    if phase.reset_optimizer:
        optimizer = _build_optimizer(phase.optimizer, _start_lr(phase.lr))
        return optimizer, optimizer.init(eqx.filter(predictors, trainable))
    if callable(phase.lr):
        # A scheduled phase writes its learning rate before every step, and
        # TrainingPhase has refused a schedule on a raw optimiser.
        return optimizer, opt_state

    # No reset, so only the learning rate moves. ``__post_init__`` has already
    # refused a phase that changes the optimiser without a reset, so the live
    # state still belongs to the optimiser this phase names. A raw
    # ``optax.GradientTransformation`` has no injectable ``hyperparams``, so
    # an lr change on one is refused here rather than silently ignored.
    if phase.lr != config.phases[phase_idx - 1].lr:
        if not _optimiser_state_supports_lr(opt_state):
            raise ValueError(
                "OptaxTrainingConfig: phase "
                f"{phase_idx} changes lr with a raw optax.GradientTransformation "
                "that cannot be re-hyperparametrised. Set "
                f"reset_optimizer=True on that phase to rebuild it, or pass "
                "an optimiser name / factory so the learning rate stays injectable."
            )
        _set_lr(opt_state, phase.lr)
    return optimizer, opt_state


class _BestTracker:
    """Lowest-data-loss snapshot and the patience counter, held together.

    Host-side only. Never pass this into a jitted function: ``eqx.filter_jit``
    would treat it as a static argument and hash it by identity, retracing
    once per instance.

    Two invariants hold between the fields: the snapshot is the parameters
    the loss was measured at, and both the minimum and the counter are
    scoped to a horizon rather than to the whole run.
    """

    def __init__(self, predictors: Any) -> None:
        self.best_loss = float("inf")
        self.best_predictors = predictors
        self.best_step: int | None = None
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
            self.best_step = None

    def update(self, loss: float, predictors: Any, step: int) -> None:
        """Record ``loss``, which must have been measured at ``predictors``.

        Pass the **pre-update** parameters. Passing the ones the optimiser
        just produced hands back a model whose loss is not the reported
        minimum, and ``loss_history`` then describes a different model.
        ``step`` is the run-wide index of this step in the loss history.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_predictors = predictors
            self.best_step = step
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def out_of_patience(self, patience: int) -> bool:
        """``patience`` consecutive steps with no new minimum. ``0`` disables it."""
        return patience > 0 and self.steps_since_improvement >= patience


@dataclass
class _PhaseRun:
    """What one pass of :func:`_run_phases` produced, step by step.

    ``loss`` is the tracked loss (data term plus any trajectory penalty)
    and ``penalty`` the weighted bound penalty, so their sum is the
    objective the optimiser minimised. ``selection`` holds the
    ``select_score`` values and is empty when selection follows ``loss``.
    ``best_step`` indexes these lists. ``stopped_early`` is set when
    patience ended the final phase before its step budget.
    """

    predictors: Any
    final_loss: float
    loss: list[float]
    penalty: list[float]
    phase: list[int]
    selection: list[float]
    best_step: int | None
    stopped_early: bool


def _select_ui(ui: TrainingUI | None, verbose: bool) -> TrainingUI:
    # An explicit ``ui=`` always wins, so a caller can swap in a custom UI
    # (a TensorBoard logger, say) without changing the loop.
    if ui is not None:
        return ui
    if verbose:
        return RichTrainingUI()
    return SilentUI()


def train_with_optax(
    model: HybridModel,
    data: Data,
    config: OptaxTrainingConfig,
    *,
    key: Array,
    objective: Objective | None = None,
    freeze: Sequence[FreezeSpec] | FreezeSpec | None = None,
    trainable: Any = None,
    ui: TrainingUI | None = None,
    validation: Data | None = None,
) -> FitResult:
    """Train ``model``'s predictors with Optax; what ``model.fit`` runs for an Optax config.

    Runs the phases described by ``config``, optionally preceded by a
    tournament. See the module docstring for what a step, a phase, and the
    tournament are, and :class:`OptaxTrainingConfig` for the fields. Every
    keyword means what it does for :meth:`HybridModel.fit
    <jaxhybridmodels.HybridModel.fit>`, and ``data`` and ``validation`` may
    be a ``Dataset`` or a list of experiments.

    ``history.loss`` is the raw per-step loss, measured before that step's
    update, concatenated across phases. It can go up. It is the data term
    plus any trajectory penalty; the bound penalty is in
    ``history.penalty``, so a ramping bound weight does not move the loss
    series. Evosax records best-so-far instead, so the two series mean
    different things.

    The returned model comes from the lowest-loss step when
    ``config.restore_best=True``, else the final step. That minimum resets
    whenever the phase horizon changes, so the returned model always comes
    from the last horizon trained on.
    """
    if not isinstance(config, OptaxTrainingConfig):
        raise TypeError(
            f"train_with_optax takes an OptaxTrainingConfig; got {type(config).__name__}"
        )
    dataset, validation_set, mask = model._prepare_fit(
        data, validation=validation, freeze=freeze, trainable=trainable, caller="fit"
    )
    run = _train_with_optax(
        model.predictors,
        dataset,
        config,
        objective=objective,
        simulate_fn=model.simulate_fn,
        state_to_output=model.state_to_output,
        solver=model.solver,
        trainable=mask,
        key=key,
        ui=ui,
        validation=validation_set,
    )
    return _fit_result(model, run)


def _fit_result(model: HybridModel, run: _PhaseRun) -> FitResult:
    """Wrap one phase run as the user-facing :class:`FitResult`."""
    return FitResult(
        model=model._with_predictors(run.predictors),
        history=FitHistory(
            loss=tuple(run.loss),
            penalty=tuple(run.penalty),
            phase=tuple(run.phase),
            validation_loss=tuple(run.selection),
        ),
        best_step=run.best_step,
        stop_reason="patience" if run.stopped_early else "budget_exhausted",
    )


def _train_with_optax(
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    objective: Objective | None = None,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    trainable: Any = None,
    key: Array,
    ui: TrainingUI | None = None,
    validation: Dataset | None = None,
) -> _PhaseRun:
    """Train loose predictors and return the full per-step record.

    The engine behind :func:`train_with_optax`, on the pieces of a model
    rather than the model, after its checks have run.
    """
    if trainable is None:
        trainable = trainable_mask(predictors)

    if not dataset.bucket_payloads:
        raise ValueError("train_with_optax: dataset has no bucket payloads")
    # Checked before the warm-up compile, which a mismatch would waste.
    if validation is not None and validation.output_channel_names != dataset.output_channel_names:
        raise ValueError(
            f"validation channels {validation.output_channel_names} do not match the "
            f"training channels {dataset.output_channel_names}"
        )

    # Resolved before the dashboard opens, so a bad channel name or penalty
    # path raises without leaving a Rich Live display on the terminal.
    resolved = _resolve_objective(objective, predictors, dataset, config)
    ui_ = _select_ui(ui, config.verbose)
    ui_.on_run_start(total_steps=config.total_steps, num_phases=len(config.phases))
    with close_run_on_error(ui_):
        kernels = _build_training_kernels(
            predictors=predictors,
            dataset=dataset,
            config=config,
            resolved=resolved,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=solver,
            trainable=trainable,
            ui=ui_,
        )
        select_score = (
            None if validation is None else build_dataset_score(kernels.score_bucket, validation)
        )

        if _tournament_enabled(config):
            ranked = _shared_tournament(
                predictors,
                dataset,
                bucket_step=kernels.bucket_step,
                score_bucket=kernels.score_bucket,
                apply_update=kernels.apply_update,
                optimizer=kernels.optimizer,
                trainable=trainable,
                tournament_attempts=config.tournament_attempts,
                tournament_steps=config.tournament_steps,
                tournament_lr=config.tournament_lr,
                key=fold(key, "tournament"),
                length_mask_fraction=jnp.asarray(1.0),
                top_k=1,
            )
            predictors = ranked[0][1]

        return _run_phases(
            predictors,
            dataset,
            config,
            kernels=kernels,
            sources=kernels.sources,
            trainable=trainable,
            ui=ui_,
            select_score=select_score,
        )


def _phase_penalty_weight(phase: TrainingPhase, objective: Objective) -> float:
    """The phase's bound-penalty weight: its own override, else the objective's."""
    return float(objective.penalty_weight if phase.penalty_weight is None else phase.penalty_weight)


def _resolve_objective(
    objective: Objective | None, predictors: Any, dataset: Dataset, config: OptaxTrainingConfig
) -> ResolvedObjective:
    """:func:`resolve_objective` with the bound penalty on when any phase weights it."""
    objective = Objective() if objective is None else objective
    penalty_on = any(_phase_penalty_weight(p, objective) > 0.0 for p in config.phases)
    return resolve_objective(objective, predictors, dataset, penalty_on=penalty_on)


def _run_phases(
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    *,
    kernels: _Kernels,
    sources: tuple[PenaltyPointSource | None, ...],
    trainable: Any,
    ui: TrainingUI,
    select_score: Callable[[Any], float] | None = None,
) -> _PhaseRun:
    """Run the phase schedule from a starting ``predictors``.

    The shared main-loop body behind :func:`train_with_optax` and the
    ensemble entry points. Takes the pre-built kernels and starts a fresh
    optimiser state; the caller owns warm-up compilation and any tournament
    selection. ``sources`` are the measured penalty points for ``dataset``,
    passed separately because a bootstrap resample gathers its own.

    The caller is responsible for the ``on_run_start`` side of the UI
    bracket; this function fires ``on_run_end``. A caller that runs this
    body several times (the ensembles, once per member) fires
    ``on_run_start`` before each call, so every member is its own UI run.

    Returns a :class:`_PhaseRun`. Its ``final_loss`` labels the returned
    predictors: the
    best-step loss when ``config.restore_best`` is set (measured exactly
    at the returned parameters), else the last step's reported loss
    (measured one update before the returned parameters, as the loop
    reports it). The ensembles rank members on this labelled loss, so a
    best-step-restored member is not reported at a loss it never had.

    The fused ``step_update`` kernel comes from ``kernels`` and is reused
    while the optimiser is; a phase reset that rebuilds the optimiser
    rebuilds it too.

    ``select_score``, when given, replaces the tracked loss for picking the
    best step and for patience. It scores a fixed full-horizon target, so
    the running minimum then survives horizon changes.
    """
    optimizer = kernels.optimizer
    opt_state = optimizer.init(eqx.filter(predictors, trainable))
    step_update = kernels.step_update
    step_update_optimizer = optimizer
    n_buckets_arg = jnp.asarray(len(dataset.bucket_payloads), dtype=jnp.int32)

    losses_history: list[float] = []
    penalties: list[float] = []
    phases: list[int] = []
    selection: list[float] = []
    stopped_early = False
    best = _BestTracker(predictors)

    for phase_idx, phase in enumerate(config.phases):
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

        length_mask_fraction = jnp.asarray(phase.horizon)
        weight = _phase_penalty_weight(phase, kernels.objective)
        # Traced, not closed over: a Python float that changed per phase
        # would retrace ``bucket_step`` at every phase boundary. Same reason
        # ``length_mask_fraction`` is passed rather than baked in.
        penalty_weight = jnp.asarray(weight)
        # Host-side per phase: the keep-mask must be concrete to index the
        # gathered points. A shape change across phases retraces only the
        # small penalty kernel, never ``bucket_step``.
        phase_points = select_penalty_points(sources, kernels.extras, phase.horizon)
        lrs = _scheduled_lrs(phase)

        best.begin_phase(
            predictors,
            horizon_changed=select_score is None and _horizon_changed(config, phase_idx),
        )
        # Only the last phase decides the reported stop reason.
        stopped_early = False

        ui.on_phase_start(
            phase_idx=phase_idx,
            phase_steps=phase.steps,
            lr=_start_lr(phase.lr),
            optimizer=phase.optimizer,
        )

        for step in range(phase.steps):
            if lrs is not None:
                _set_lr(opt_state, lrs[step])
            total_loss, acc_grads, n_buckets = _training_step(
                predictors, dataset, kernels.bucket_step, length_mask_fraction, trainable
            )
            avg_penalty, penalty_grads = kernels.penalty_step(
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
            penalties.append(weight * penalty_value)
            phases.append(phase_idx)
            score = loss_value
            if select_score is not None:
                score = select_score(previous_predictors)
                selection.append(score)
            # ``previous_predictors``, not ``predictors``: avg_data was
            # measured before the update was applied.
            best.update(score, previous_predictors, len(losses_history) - 1)

            ui.on_step_end(
                step_idx=step,
                phase_idx=phase_idx,
                loss=loss_value,
                penalty=penalty_value,
            )

            if best.out_of_patience(config.patience):
                stopped_early = step < phase.steps - 1
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
    return _PhaseRun(
        predictors=final_predictors,
        final_loss=final_loss,
        loss=losses_history,
        penalty=penalties,
        phase=phases,
        selection=selection,
        best_step=best.best_step if config.restore_best else None,
        stopped_early=stopped_early,
    )


@dataclass(frozen=True)
class _Kernels:
    """The compiled pieces and resolved objective one run (or ensemble) shares.

    ``sources`` are the measured penalty points of the dataset the kernels
    were built against and ``extras`` the objective's extra points, one per
    ``BoundedPredictor`` in traversal order. Host-side only; never passed
    into a jitted function.
    """

    objective: Objective
    bucket_step: Callable[..., tuple[Array, Any]]
    penalty_step: Callable[..., tuple[Array, Any]]
    score_bucket: Callable[..., Array]
    optimizer: optax.GradientTransformation
    apply_update: Callable[..., tuple[Any, Any]]
    step_update: Callable[..., tuple[Any, Any]]
    sources: tuple[PenaltyPointSource | None, ...]
    extras: tuple[Array, ...]


def _build_training_kernels(
    *,
    predictors: Any,
    dataset: Dataset,
    config: OptaxTrainingConfig,
    resolved: ResolvedObjective,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
    trainable: Any,
    ui: TrainingUI,
) -> _Kernels:
    """Build the compiled kernels and optimiser for a resolved objective, and warm the caches.

    Shared by :func:`train_with_optax` and the ensemble entry points so a
    seed/bootstrap ensemble reuses the exact same kernel construction. The
    jit caches come back already populated by a warm-up pass.

    The caller owns the ``on_run_start`` side of the UI bracket: the
    warm-up compile events below are meant to land inside a run, and the
    ensembles fire ``on_run_start`` before every member, of which this is
    the first.
    """
    bucket_payloads = dataset.bucket_payloads
    objective, loss_fn = resolved.objective, resolved.loss_fn
    sources, extras = resolved.sources, resolved.extras

    bucket_step = build_bucket_step(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trainable=trainable,
        trajectory_penalty_fn=objective.trajectory_penalty_fn,
        trajectory_penalty_weight=objective.trajectory_penalty_weight,
    )
    penalty_step = build_penalty_step(penalty_fn=objective.bound_penalty_fn, trainable=trainable)
    score_bucket = build_score_bucket(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=loss_fn,
        trajectory_penalty_fn=objective.trajectory_penalty_fn,
        trajectory_penalty_weight=objective.trajectory_penalty_weight,
    )
    # Settle only when the run can hit the weak→strong scalar flip: a
    # custom masks may train weak-typed scalar leaves, and JAX's cache keys
    # 0-d leaves by value+weak_type, so the first update would retrace
    # every kernel once (seconds-to-minutes of ODE compile) mid-run.
    # Strong-typed or frozen scalars never flip, so skip the settle entirely
    # and keep the "one trace per bucket shape" contract untouched.
    settle = _has_weak_scalar_trainable(predictors, trainable)
    first = config.phases[0]
    optimizer = _build_optimizer(first.optimizer, _start_lr(first.lr))
    apply_update = build_apply_update(optimizer=optimizer, trainable=trainable)
    step_update = _build_step_update(optimizer=optimizer, trainable=trainable)
    _warmup_compile(
        predictors,
        bucket_payloads,
        bucket_step=bucket_step,
        ui=ui,
        length_mask_fraction=jnp.asarray(1.0),
        penalty_step=penalty_step,
        optimizer=optimizer,
        step_update=step_update,
        trainable=trainable,
        penalty_points=select_penalty_points(sources, extras, 1.0),
        score_bucket=score_bucket if _tournament_enabled(config) else None,
        settle=settle,
    )
    return _Kernels(
        objective=objective,
        bucket_step=bucket_step,
        penalty_step=penalty_step,
        score_bucket=score_bucket,
        optimizer=optimizer,
        apply_update=apply_update,
        step_update=step_update,
        sources=sources,
        extras=extras,
    )


def _fit_ensemble(
    model: HybridModel,
    data: Data,
    config: OptaxTrainingConfig,
    *,
    key: Array,
    n_members: int,
    n_seeds: int | None,
    bootstrap: bool,
    objective: Objective | None,
    freeze: Sequence[FreezeSpec] | FreezeSpec | None,
    trainable: Any,
    ui: TrainingUI | None,
) -> list[FitResult]:
    """The body of :meth:`HybridModel.fit_ensemble`; see there for the two modes.

    The kernels are built and warmed once and shared by every member. Each
    member is its own UI run: the warm-up compile and the first scoring
    sweep land inside the first member's bracket, and ``_run_phases`` fires
    each member's ``on_run_end``.
    """
    for name, value in (("n_members", n_members), ("n_seeds", n_seeds)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError(f"fit_ensemble: {name} must be an int; got {value!r}")
    if n_members < 1:
        raise ValueError(f"fit_ensemble: n_members must be >= 1, got {n_members}")
    if n_seeds is None:
        n_seeds = 1 if bootstrap else n_members
    if n_seeds < 1:
        raise ValueError(f"fit_ensemble: n_seeds must be >= 1, got {n_seeds}")
    if not bootstrap and n_members > n_seeds:
        raise ValueError(
            f"fit_ensemble: n_members={n_members} exceeds n_seeds={n_seeds}; a seed "
            "ensemble trains the best n_members of n_seeds seeds"
        )
    dataset, _, mask = model._prepare_fit(
        data, validation=None, freeze=freeze, trainable=trainable, caller="fit_ensemble"
    )
    predictors = model.predictors

    # Resolved before the dashboard opens; see _train_with_optax.
    resolved = _resolve_objective(objective, predictors, dataset, config)
    ui_ = _select_ui(ui, config.verbose)
    ui_.on_run_start(total_steps=config.total_steps, num_phases=len(config.phases))
    with close_run_on_error(ui_):
        kernels = _build_training_kernels(
            predictors=predictors,
            dataset=dataset,
            config=config,
            resolved=resolved,
            simulate_fn=model.simulate_fn,
            state_to_output=model.state_to_output,
            solver=model.solver,
            trainable=mask,
            ui=ui_,
        )

        def rank(data_: Dataset, seed_key: Array, top_k: int) -> list[tuple[float, Any]]:
            return _shared_tournament(
                predictors,
                data_,
                bucket_step=kernels.bucket_step,
                score_bucket=kernels.score_bucket,
                apply_update=kernels.apply_update,
                optimizer=kernels.optimizer,
                trainable=mask,
                tournament_attempts=n_seeds,
                tournament_steps=config.tournament_steps,
                tournament_lr=config.tournament_lr,
                key=seed_key,
                length_mask_fraction=jnp.asarray(1.0),
                top_k=top_k,
            )

        # (training data, seed) per member. The key labels are the ones the
        # earlier train_seed_ensemble and train_bootstrap_ensemble used, so a
        # given key still draws the same members.
        jobs: list[tuple[Dataset, tuple[float, Any]]]
        if bootstrap:
            jobs = []
            for s in range(n_members):
                boot = make_bootstrap_dataset(dataset, key=fold(key, f"bootstrap_{s}"))
                sample_key = fold(key, f"bootstrap_seeds_{s}")
                best = rank(boot, fold(sample_key, "bootstrap_sample_tournament"), 1)
                jobs.append((boot, best[0]))
        else:
            ranked = rank(dataset, fold(key, "seed_ensemble_tournament"), n_members)
            jobs = [(dataset, seed) for seed in ranked]

        trained: list[FitResult] = []
        for data_, (seed_score, candidate) in jobs:
            if not math.isfinite(seed_score):
                continue
            # The first member's on_run_start is the one opened above.
            if trained:
                ui_.on_run_start(total_steps=config.total_steps, num_phases=len(config.phases))
            run = _run_phases(
                candidate,
                data_,
                config,
                kernels=kernels,
                # A resample changes the covariate values, so its measured
                # penalty points are gathered from the resample itself.
                sources=data_penalty_points(predictors, data_) if bootstrap else kernels.sources,
                trainable=mask,
                ui=ui_,
            )
            trained.append(_fit_result(model, run))

        if not trained:
            raise RuntimeError(
                "fit_ensemble: every seed failed its warm-up (a solver error or a "
                "non-finite loss); no member was trained"
            )
    # Stable, so a tie keeps the seed ranking. FitResult.final_loss is the
    # loss _run_phases reports for the returned parameters. It is NaN when
    # restore_best found no finite loss, and NaN breaks the ordering of the
    # whole sort, so such a member ranks last.
    return sorted(trained, key=_rank_key)


def _rank_key(result: FitResult) -> float:
    """Sort key for ensemble members: final loss, with NaN ranked as ``inf``."""
    loss = result.final_loss
    return loss if not math.isnan(loss) else math.inf
