"""What a fit minimises, held apart from how it is minimised.

:class:`Objective` collects every term of the training objective: the data
loss and its channel weights, the bound penalty and where it is charged,
and the trajectory penalty. The optimiser configs
(:class:`~jaxhybridmodels.training.optax.OptaxTrainingConfig`,
:class:`~jaxhybridmodels.training.evosax.EvosaxTrainingConfig`) hold only
the search settings, so the same objective can be handed to an Evosax
search and then to an Optax polish without restating it::

    objective = hm.Objective(loss="bal_mse", penalty_weight=1e-3)
    coarse = model.fit(data, evosax_config, objective=objective, key=k1)
    fine = coarse.model.fit(data, optax_config, objective=objective, key=k2)

Validation scoring uses the same objective, minus the bound penalty.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from jaxhybridmodels.data import BucketPayload, Dataset
from jaxhybridmodels.losses import resolve_loss_fn
from jaxhybridmodels.penalties import (
    PenaltyPointSource,
    bound_penalty,
    data_penalty_points,
    resolve_penalty_points,
    validate_penalty_points,
)

__all__ = ("Objective",)


@dataclass(frozen=True, kw_only=True, eq=False)
class Objective:
    """The training objective: data loss plus optional penalties.

    Keyword-only. ``Objective()`` is plain MSE over every channel with no
    penalty.

    Attributes
    ----------
    loss : Callable | str
        A ``LOSS_REGISTRY`` key (``"mse"``, ``"mle"``, ``"bal_mse"``,
        ``"bal_mle"``) or a callable ``loss(pred_obs, bp) -> scalar``.
    channel_weights : Mapping[str, float] | Sequence[float] | None
        Per-channel multipliers on the loss. A mapping keyed by channel
        name scores only the channels it names (``{"conc": 1.0}`` fits one
        channel of several); a sequence gives one weight per output channel,
        in ``output_channel_names`` order, and keeps every channel's cells
        in the loss's denominator. ``None`` weights every channel 1. Names
        are checked against the dataset's channels when training starts.
        A named channel with weight ``0.0`` contributes nothing but its
        cells still count in the denominator, so ``{"a": 1.0, "b": 0.0}``
        gives half the loss of ``{"a": 1.0}``; leave a name out to drop it.
    penalty_weight : float
        Weight on the bound-saturation penalty, relative to the
        per-bucket-averaged data term. ``0.0`` disables it.
        ``TrainingPhase.penalty_weight`` overrides it for one Optax phase.
    penalty_points : Mapping[str, ArrayLike] | None
        Extra points where the bound penalty is charged, keyed by the path of
        a ``BoundedPredictor`` in the predictors pytree. The path is the
        dotted string ``freeze_paths`` uses, so for a dict of predictors it
        is the key: ``{"rate": grid}``. Each value is a ``[G, n_inputs]``
        array of physical input vectors in that predictor's ``input_keys``
        order. Predictors left out are charged at the measured points only.
        ``jaxhybridmodels.penalties.box_grid`` builds a sweep of an input box.
    penalty_fn : Callable | None
        Replaces :func:`~jaxhybridmodels.penalties.bound_penalty`. Called as
        ``(predictors, points) -> scalar`` where ``points`` is one
        ``[G, n_inputs]`` array per ``BoundedPredictor`` in traversal order;
        a custom term that needs no points ignores them.
    trajectory_penalty_fn : Callable | None
        Trajectory-aware penalty for embedded hybrid models, called as
        ``(full_state, bp) -> scalar`` on the full simulator state
        ``[N, T, S]``, penalty accumulators included. See the
        ``penalty_*`` helpers in ``jaxhybridmodels.penalties``.
    trajectory_penalty_weight : float
        Weight on ``trajectory_penalty_fn``. ``0.0`` disables it even when a
        function is set; a non-zero weight without a function raises.
    """

    loss: Callable[..., Array] | str = "mse"
    channel_weights: Mapping[str, float] | Sequence[float] | None = None
    penalty_weight: float = 0.0
    penalty_points: Mapping[str, ArrayLike] | None = None
    penalty_fn: Callable[..., Array] | None = None
    trajectory_penalty_fn: Callable[..., Array] | None = None
    trajectory_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        _check_weight("penalty_weight", self.penalty_weight)
        _check_weight("trajectory_penalty_weight", self.trajectory_penalty_weight)
        if self.trajectory_penalty_weight != 0.0 and self.trajectory_penalty_fn is None:
            raise ValueError(
                "Objective.trajectory_penalty_weight is non-zero but trajectory_penalty_fn "
                "is None. Provide a (full_state, bp) -> scalar function to charge."
            )
        # Copied so later mutation of the caller's container cannot change a run.
        object.__setattr__(self, "channel_weights", _copy_channel_weights(self.channel_weights))
        if self.penalty_points is not None:
            if not isinstance(self.penalty_points, Mapping):
                raise TypeError(
                    "Objective.penalty_points must be a mapping from predictor path to a "
                    "[G, n_inputs] array, e.g. {'rate': grid}; the tuple-in-traversal-order "
                    "form was removed (jaxhybridmodels.penalties.bounded_predictor_paths "
                    f"lists the paths). Got {type(self.penalty_points).__name__}."
                )
            frozen = {
                str(path): jnp.asarray(points) for path, points in self.penalty_points.items()
            }
            for path, points in frozen.items():
                if points.ndim != 2:
                    raise ValueError(
                        f"Objective.penalty_points[{path!r}] must be a rank-2 [G, n_inputs] "
                        f"array of physical input vectors; got shape {points.shape}"
                    )
            object.__setattr__(self, "penalty_points", frozen)

    def loss_fn(
        self, output_channel_names: Sequence[str]
    ) -> Callable[[Array, BucketPayload], Array]:
        """The resolved ``(pred_obs, bp) -> scalar`` for a dataset with these channels.

        Channel names in ``channel_weights`` are matched here, so an unknown
        name raises before anything compiles.
        """
        channel_idx, channel_weights = _channel_selection(
            self.channel_weights, tuple(output_channel_names)
        )
        return resolve_loss_fn(self.loss, channel_idx, channel_weights)

    @property
    def bound_penalty_fn(self) -> Callable[[Any, tuple[Array, ...]], Array]:
        """``penalty_fn`` when set, else :func:`~jaxhybridmodels.penalties.bound_penalty`."""
        return bound_penalty if self.penalty_fn is None else self.penalty_fn

    def extra_penalty_points(self, predictors: Any) -> tuple[Array, ...]:
        """``penalty_points`` as one array per ``BoundedPredictor``, in traversal order.

        Predictors with no entry get an empty ``[0, n_inputs]`` array. An
        unknown path raises, naming the valid ones.
        """
        return resolve_penalty_points(predictors, self.penalty_points)


@dataclass(frozen=True)
class ResolvedObjective:
    """An objective matched against one dataset and predictors pytree.

    ``loss_fn`` is the bound ``(pred_obs, bp) -> scalar``; ``sources`` and
    ``extras`` are the measured and extra penalty points, one per
    ``BoundedPredictor`` in traversal order. Host-side only.
    """

    objective: Objective
    loss_fn: Callable[[Array, BucketPayload], Array]
    sources: tuple[PenaltyPointSource | None, ...]
    extras: tuple[Array, ...]


def resolve_objective(
    objective: Objective, predictors: Any, dataset: Dataset, *, penalty_on: bool
) -> ResolvedObjective:
    """Check ``objective`` against the data and predictors; raise on any mismatch.

    Covers an unknown channel name, an unknown penalty-point path and, when
    ``penalty_on`` (some stage of the run weights the bound penalty above
    zero), a predictor no penalty point reaches. Cheap (no compile), so the
    trainers run it before opening the UI.
    """
    loss_fn = objective.loss_fn(dataset.output_channel_names)
    sources = data_penalty_points(predictors, dataset)
    extras = objective.extra_penalty_points(predictors)
    validate_penalty_points(
        predictors,
        sources,
        extras,
        # Coverage is the default bound penalty's contract; a custom
        # ``penalty_fn`` owns its points and needs no coverage check.
        enabled=penalty_on and objective.penalty_fn is None,
    )
    return ResolvedObjective(objective=objective, loss_fn=loss_fn, sources=sources, extras=extras)


def _check_weight(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"Objective.{name} must be finite and non-negative; got {value}")


def _copy_channel_weights(
    weights: Mapping[str, float] | Sequence[float] | None,
) -> dict[str, float] | tuple[float, ...] | None:
    """Validate ``channel_weights`` and copy it: a dict by name, or a tuple."""
    if weights is None:
        return None
    if isinstance(weights, str):
        raise ValueError("Objective.channel_weights must be a mapping or a sequence of floats")
    copied: dict[str, float] | tuple[float, ...]
    if isinstance(weights, Mapping):
        if not weights:
            raise ValueError("Objective.channel_weights mapping must name at least one channel")
        by_name = cast(Mapping[str, float], weights)
        copied = {str(name): float(w) for name, w in by_name.items()}
        values = list(copied.values())
    else:
        copied = tuple(float(w) for w in weights)
        values = list(copied)
    if any(not math.isfinite(w) or w < 0.0 for w in values):
        raise ValueError(
            f"Objective.channel_weights must be finite and non-negative; got {weights}"
        )
    return copied


def _channel_selection(
    weights: Mapping[str, float] | Sequence[float] | None,
    names: tuple[str, ...],
) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None]:
    """Turn frozen channel weights into the loss layer's ``(channel_idx, channel_weights)``.

    A by-name mapping becomes a channel selection in output order plus the
    matching weights. Unit weights collapse to ``None``, which keeps a plain
    ``(pred_obs, bp)`` custom loss usable with a by-name selection (the loss
    layer refuses weights it cannot apply inside a custom reduction).
    """
    if weights is None:
        return None, None
    if isinstance(weights, Mapping):
        by_name = cast(Mapping[str, float], weights)
        unknown = sorted(set(by_name) - set(names))
        if unknown:
            raise ValueError(
                f"Objective.channel_weights names unknown channels {unknown}; "
                f"the dataset's channels are {list(names)}"
            )
        idx: tuple[int, ...] | None = tuple(i for i, n in enumerate(names) if n in by_name)
        values = tuple(float(by_name[names[i]]) for i in idx or ())
        if idx == tuple(range(len(names))):
            idx = None
    else:
        values = tuple(float(w) for w in weights)
        if len(values) != len(names):
            raise ValueError(
                f"Objective.channel_weights has {len(values)} entries but the dataset has "
                f"{len(names)} channels {list(names)}; pass one weight per channel or a "
                "mapping by channel name"
            )
        idx = None
    return idx, (None if all(w == 1.0 for w in values) else values)
