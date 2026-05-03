"""Prediction entry points (SPEC §5.10 / R-J1 / R-J2 / R-J3).

``predict_bucket`` vmaps the user's ``simulate_fn`` across a bucket's N axis
and applies ``state_to_output`` per experiment. It is jitted independently
from training (R-J1). ``predict_dataset`` is the bucket-dispatch driver: a
plain Python ``for`` loop over ``dataset.bucket_payloads`` (R-J2), so each
bucket shape compiles once (R-J3).
"""

# ruff: noqa: F722

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float

from hybridmodels.data import BucketPayload, Dataset
from hybridmodels.solver import SolverConfig


@eqx.filter_jit
def predict_bucket(
    predictor: eqx.Module,
    bp: BucketPayload,
    *,
    simulate_fn: Callable[..., Array],
    state_to_output: Callable[[Array], Array],
    solver: SolverConfig,
) -> Float[Array, "N T D"]:
    """Vmap ``simulate_fn`` over the bucket's ``N`` axis and project to observed channels.

    The inner ``_per_experiment`` runs the user's ``simulate_fn`` once for
    one experiment to produce ``[T, S]`` and then projects to ``[T, D]`` via
    ``state_to_output``. ``jax.vmap`` lifts this over ``(ts, covariates, y0)``
    along the ``N`` axis to produce ``[N, T, D]``. ``predictor`` and
    ``solver`` are closed over (no vmap axis) — they are constant across
    experiments in the bucket.

    Jit caching: one trace per bucket *shape* (R-J3). The Python dispatch
    over ``dataset.bucket_payloads`` lives in ``predict_dataset``, never
    inside the jitted region (R-J2).

    Parameters
    ----------
    predictor : eqx.Module
        Trainable predictor (typically ``BoundedPredictor`` or ``RatePair``)
        consumed by ``simulate_fn``.
    bp : BucketPayload
        One bucket; ``ts``, ``covariates``, ``y0`` are vmapped along ``N``.
    simulate_fn
        User-supplied ``(predictor, ts, cov, y0, solver) -> [T, S]`` per R-A2.
    state_to_output
        Pure ``[T, S] -> [T, D]`` projector held on ``Dataset``.
    solver
        Static ``SolverConfig`` (R-S1).

    Returns
    -------
    Float[Array, "N T D"]
        Predicted output channels for every experiment in the bucket.
    """
    def _per_experiment(
        ts: Array, covariates: dict[str, Array], y0: Array
    ) -> Array:
        full_state = simulate_fn(predictor, ts, covariates, y0, solver)
        return state_to_output(full_state)

    return jax.vmap(_per_experiment, in_axes=(0, 0, 0))(
        bp.ts, bp.covariates, bp.y0
    )


def predict_dataset(
    predictor: eqx.Module,
    dataset: Dataset,
    *,
    simulate_fn: Callable[..., Array],
    solver: SolverConfig,
) -> tuple[Float[Array, "N T D"], ...]:
    """Run ``predict_bucket`` over every bucket in ``dataset`` and return the stack tuple.

    The Python ``for`` loop over ``dataset.bucket_payloads`` is the dispatch
    driver (R-J2); each bucket shape compiles ``predict_bucket`` exactly once.
    Returns a tuple aligned with ``dataset.bucket_payloads`` order, *not* a
    flat concatenation — each entry has its own ``[N_b, T_b, D]`` shape and
    cannot be stacked into a single tensor (different ``T``).

    Returns
    -------
    tuple[Float[Array, "N T D"], ...]
        One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order.
    """
    state_to_output = dataset.state_to_output
    return tuple(
        predict_bucket(
            predictor,
            bp,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=solver,
        )
        for bp in dataset.bucket_payloads
    )
