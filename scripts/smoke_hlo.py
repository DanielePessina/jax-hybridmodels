"""Inspect the compiled HLO of the library's core kernels for review.

A review habit from the JAX scalability research: the compiled kernel can
surprise you even when the Python source looks right. This script lowers
``predict_bucket`` and a training-style ``bucket_step`` (value_and_grad)
to HLO and reports any unexpected dtype ``convert``, layout ``copy``, or
``retile`` ops, which XLA inserts "sometimes at non-trivial overhead".

It is a reviewer's tool, not a hard CI gate: exact HLO op counts drift
across JAX/XLA versions, so the script prints the summary and exits
non-zero only when the suspicious-op count jumps far outside the normal
range. Read the printed op census, don't just trust the exit code.

Run:
    uv run python scripts/smoke_hlo.py
"""

from __future__ import annotations

import re
import sys

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    make_experiment,
    predict_bucket,
)
from hybridmodels.solver import ADJOINT_REGISTRY

# How many layout-related ops (copy/retile/transpose) are "normal" for
# these tiny kernels. A few ``copy`` ops are ordinary buffer-assignment
# noise, and the training kernel (``value_and_grad``) carries the backward
# pass's data-flow copies on top of the forward solve. A jump far above
# this is what the reviewer should inspect. This is a drift detector, not
# a pass/fail for layout quality, and exact counts shift across
# JAX/XLA versions.
COPY_OPS_LIMIT = 160
# ``retile``/``transpose`` should be zero: they indicate XLA re-layout the
# library's shapes forced on it, which the scalability research flags as
# "sometimes at non-trivial overhead".
RETILE_OPS_LIMIT = 10

# Dtype casts are expected inside diffrax's adaptive step-size controller
# (it converts between floats and ints for the step counter), so they are
# reported for information, not counted as suspicious.
CAST_OPS = ("convert", "bitcast-convert", "convert_element_type")


def _y0_fn(covariates, channels):
    return jnp.array([float(channels["x"].values[0]), 0.0])


def _state_to_output(state):
    return state[..., :1]


def _simulate_fn(predictor, ts, covariates, y0, solver):
    def vector_field(t, y, args):
        omega = predictor({"t": t}).reshape(())
        omega_sq = omega * omega
        return jnp.stack([y[1], -(omega_sq) * y[0]])

    term = diffrax.ODETerm(vector_field)
    sol = diffrax.diffeqsolve(
        term,
        solver.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=solver.dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=solver.stepsize_controller(),
        adjoint=solver.adjoint,
        max_steps=solver.max_steps,
    )
    return sol.ys


def census(hlo_text: str) -> dict[str, int]:
    """Count op kinds in an HLO module text, summarised by op name.

    An HLO instruction line looks like::

        %param_3.8 = f32[1]{0} parameter(3)
        ROOT %arg_tuple.3 = (f32[1,10,2]{2,1,0}) parameter(0)

    The op name is the token immediately before the first ``(`` that
    *follows the type*. Types may themselves be parenthesised tuples, so
    we strip one balanced leading paren group before locating the op name.
    """
    counts: dict[str, int] = {}
    for line in hlo_text.splitlines():
        line = re.sub(r"/\*.*?\*/", "", line.strip())
        if not line.startswith("%") and not line.startswith("ROOT %"):
            continue
        if "=" not in line:
            continue
        rhs = line.split("=", 1)[1].strip()
        if rhs.startswith("("):
            depth = 0
            for i, ch in enumerate(rhs):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        rhs = rhs[i + 1 :].strip()
                        break
        before_paren = rhs.split("(", 1)[0].strip()
        if not before_paren:
            continue
        op = before_paren.split()[-1]
        counts[op] = counts.get(op, 0) + 1
    return counts


def main() -> int:
    ts = jnp.linspace(0.0, 5.0, 10)
    exp = make_experiment(
        covariates={"id": 0.0},
        channels={"x": ChannelObs(ts=ts, values=jnp.cos(ts))},
        y0_fn=_y0_fn,
        exp_id="exp_0",
    )
    ds = make_dataset(
        [exp],
        output_channel_names=("x",),
    )
    bp = ds.bucket_payloads[0]

    predictor = BoundedPredictor(
        input_keys=("t",),
        in_scaler=BoundScaler(bounds=((0.0, 5.0),)),
        inner=MLPPredictor(
            in_size=1,
            out_size=1,
            width_size=8,
            depth=1,
            activation_name="tanh",
            key=jax.random.PRNGKey(0),
        ),
        out_scaler=BoundScaler(bounds=((0.5, 2.0),)),
    )
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=4096,
        dt0=0.05,
        adjoint=ADJOINT_REGISTRY["RecursiveCheckpoint"](),
    )

    kernels = {}

    compiled_predict = predict_bucket.lower(
        predictor,
        bp,
        simulate_fn=_simulate_fn,
        state_to_output=_state_to_output,
        solver=solver,
    ).compile()
    kernels["predict_bucket"] = compiled_predict.compiled.as_text()

    def loss(predictors, bucket):
        pred = predict_bucket(
            predictors,
            bucket,
            simulate_fn=_simulate_fn,
            state_to_output=_state_to_output,
            solver=solver,
        )
        return jnp.sum((pred - bucket.y_observed) ** 2)

    grad_step = eqx.filter_jit(eqx.filter_value_and_grad(loss)).lower(predictor, bp).compile()
    kernels["bucket_step (value_and_grad)"] = grad_step.compiled.as_text()

    all_ok = True
    for name, hlo in kernels.items():
        c = census(hlo)
        copies = sum(c.get(op, 0) for op in ("copy", "copy-start", "copy-done"))
        retiles = sum(c.get(op, 0) for op in ("retile", "transpose"))
        casts = sum(c.get(op, 0) for op in CAST_OPS)
        print(f"\n=== {name} ===")
        print(
            f"  total ops: {sum(c.values())}, copies: {copies}, "
            f"retile/transpose: {retiles}, dtype casts: {casts}"
        )
        common = sorted(c.items(), key=lambda kv: -kv[1])[:8]
        print("  top ops:", ", ".join(f"{op} x{n}" for op, n in common))
        if copies > COPY_OPS_LIMIT:
            print(
                f"  WARNING: {copies} copy ops exceed limit {COPY_OPS_LIMIT}; "
                "inspect the kernel for layout churn"
            )
            all_ok = False
        if retiles > RETILE_OPS_LIMIT:
            print(
                f"  WARNING: {retiles} retile/transpose ops exceed limit "
                f"{RETILE_OPS_LIMIT}; XLA is re-laying-out the shapes"
            )
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
