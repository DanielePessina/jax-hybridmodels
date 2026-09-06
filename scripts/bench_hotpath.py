"""Benchmark the training hot path and lock the steady-state per-step cost.

Measures, for a small synthetic hybrid model with an MLP inside the ODE:
  - prediction forward pass (compile + run)
  - the production kernel build incl. its warm-up/settle (compile phase)
  - one training step: bucket_step + penalty_step + apply_update

Run manually with `uv run python scripts/bench_hotpath.py`.

History (CPU, single bucket, MLP 1->16->16->1 inside Tsit5):
  before anything:            first _run_phases ~123-150 ms/step
  first pass (settle only):   ~17-24 ms/step
  root fix + warm-all + fused: ~3.1-3.3 ms/step, zero mid-run retraces
  (BoundScaler strong-types its own scalars; warm compiles every kernel
  once; the per-step tail is one fused jitted update)

This is not a correctness test. It is the regression harness for the
jit-cache settle logic in ``training/optax.py:_warmup_compile``.
"""

import time

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    MLPPredictor,
    SolverConfig,
    make_dataset,
    make_experiment,
    predict_dataset,
)
from hybridmodels.penalties import select_penalty_points
from hybridmodels.trainable import trainable_mask
from hybridmodels.training.optax import (
    OptaxTrainingConfig,
    _build_training_kernels,
    _run_phases,
)
from hybridmodels.ui.base import SilentUI

N_EXPERIMENTS = 12
N_STEPS_BENCH = 20


def make_synth_dataset(key):
    exps = []
    for i in range(N_EXPERIMENTS):
        t = jnp.linspace(0.0, 10.0, 21)
        T = float(300.0 + 10 * (i % 3))
        y = jnp.sin(t * (1.0 + 0.1 * i)) + 0.05 * t
        exps.append(
            make_experiment(
                covariates={"temperature": T},
                channels={
                    "y": ChannelObs(ts=t, values=y, variance=jnp.full_like(y, 0.01)),
                },
                y0_fn=lambda cov, ch: jnp.array([ch["y"].values[0], 0.0]),
                exp_id=f"e{i}",
            )
        )
    return make_dataset(exps, output_channel_names=("y",))


def make_predictor(key):
    in_scaler = BoundScaler(bounds=((280.0, 320.0),), transform="sigmoid", warp="linear")
    inner = MLPPredictor(
        in_size=1, out_size=1, width_size=16, depth=2, activation_name="tanh", key=key
    )
    out_scaler = BoundScaler(bounds=((-1.0, 1.0),), transform="sigmoid")
    return BoundedPredictor(
        input_keys=("temperature",),
        in_scaler=in_scaler,
        inner=inner,
        out_scaler=out_scaler,
    )


def simulate_fn(predictor, ts, covariates, y0, solver):
    k = predictor({"temperature": covariates["temperature"]})

    def vf(t, y, args):
        y1, y2 = y[0], y[1]
        dy1 = -k[0] * y1 + y2
        dy2 = -k[0] * y2 - y1
        return jnp.stack([dy1, dy2])

    return jnp.asarray(
        diffrax.diffeqsolve(
            diffrax.ODETerm(vf),
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=solver.stepsize_controller(),
            adjoint=solver.adjoint,
            max_steps=solver.max_steps,
        ).ys
    )


def state_to_output(state):
    return state[..., :1]


def main():
    print("jax devices:", jax.devices())
    key = jr.PRNGKey(0)
    dataset = make_synth_dataset(key)
    predictor = make_predictor(key)
    print(f"buckets: {len(dataset.bucket_payloads)}")

    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=1e-5,
        max_steps=100_000,
        dt0=0.01,
    )

    # ---- prediction ----
    t0 = time.perf_counter()
    preds = predict_dataset(
        predictor, dataset, simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver
    )
    for p in preds:
        jax.block_until_ready(p)
    t_pred_total = time.perf_counter() - t0

    # ---- production kernel build (warm-up + settle cycle) ----
    trainable = trainable_mask(predictor)
    config = OptaxTrainingConfig(
        steps=(N_STEPS_BENCH,),
        lr=(1e-3,),
        optimizer=(optax.adamw,),
        reset_optimiser_state=(False,),
    )
    ui = SilentUI()
    ui.on_run_start(total_steps=N_STEPS_BENCH, num_phases=1)
    t0 = time.perf_counter()
    (bucket_step, penalty_step, score_bucket, optimizer, apply_update,
     step_update, sources, extras) = _build_training_kernels(
        predictors=predictor, dataset=dataset, config=config,
        simulate_fn=simulate_fn, state_to_output=state_to_output,
        solver=solver, trainable=trainable, ui=ui,
    )
    t_build = time.perf_counter() - t0

    bp0 = dataset.bucket_payloads[0]
    opt_state = optimizer.init(eqx.filter(predictor, trainable))
    loss, grads = bucket_step(predictor, bp0, jnp.asarray(1.0))
    pen_points = select_penalty_points(sources, extras, 1.0)
    p, pg = penalty_step(predictor, jnp.asarray(0.0), pen_points)
    np_, os_ = apply_update(predictor, grads, opt_state)
    jax.block_until_ready((loss, p, np_))

    def bench(fn, *args, n=50):
        for _ in range(3):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(*args)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n * 1000

    print(f"\nprediction (compile+run, all buckets): {t_pred_total*1000:.1f} ms")
    print(f"kernel build + warm-up + settle: {t_build*1000:.1f} ms")
    print(f"  bucket_step steady-state: "
          f"{bench(bucket_step, predictor, bp0, jnp.asarray(1.0)):.3f} ms")
    print(f"  penalty_step steady-state: "
          f"{bench(penalty_step, predictor, jnp.asarray(0.0), pen_points):.3f} ms")
    print(f"  apply_update steady-state: "
          f"{bench(apply_update, predictor, grads, opt_state):.3f} ms")

    # full _run_phases loop timing (host overhead incl.) — caches pre-settled
    ui2 = SilentUI()
    t0 = time.perf_counter()
    hist, final, fl = _run_phases(
        predictor,
        dataset,
        config,
        optimizer=optimizer,
        apply_update=apply_update,
        step_update=step_update,        bucket_step=bucket_step,
        penalty_step=penalty_step,
        sources=sources,
        extras=extras,
        trainable=trainable,
        ui=ui2,
    )
    t_loop = time.perf_counter() - t0
    print(f"\n_run_phases {N_STEPS_BENCH} steps (settled): {t_loop*1000:.1f} ms total, "
          f"{t_loop/N_STEPS_BENCH*1000:.1f} ms/step (host+device)")

    floor = (
        bench(bucket_step, predictor, bp0, jnp.asarray(1.0))
        + bench(penalty_step, predictor, jnp.asarray(0.0), pen_points)
        + bench(apply_update, predictor, grads, opt_state)
    )
    print(f"  sum of 3 steady-state device calls: {floor:.3f} ms/step")
    print(f"  => host/orchestration overhead per step: "
          f"{max(0.0, t_loop/N_STEPS_BENCH*1000 - floor):.3f} ms")


if __name__ == "__main__":
    main()
