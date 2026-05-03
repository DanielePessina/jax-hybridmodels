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
