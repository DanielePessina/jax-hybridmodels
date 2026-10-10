"""Write your own training loop from the public gradient kernels.

``train_with_optax`` is assembled from the pieces in
``jaxhybridmodels.training.kernels``. When you need a loop it cannot
express, compose the same pieces instead of forking the trainer. This
script recovers a harmonic oscillator's frequency with a loop that weights
each bucket's loss and gradients by its experiment count, which the stock
trainer has no option for.

Shows:

- ``build_bucket_step``, ``build_penalty_step`` and ``build_apply_update``
- a custom loss (masked Huber) and a custom regulariser (L2 on a second
  ``Parameter``) entering a hand-written loop
- ``predict_dataset`` and ``compute_metrics`` on raw predictors

Run::

    uv run python examples/custom_loop/train_custom_loop.py
    uv run python examples/custom_loop/train_custom_loop.py --steps 5 --no-plot

Prints the loss, penalty and ``omega`` every 25 steps, the recovered values
and metrics; writes ``figures/parity.png``. Under a minute on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.training.kernels import (
    build_apply_update,
    build_bucket_step,
    build_penalty_step,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
)

OMEGA_TRUE: float = 1.0
NOISE_STD: float = 0.05

# Two time grids of different lengths give two buckets: three experiments
# on the short grid, one on the long grid. Unequal buckets are what give
# the per-bucket weight below something to do.
BUCKET_A_TS: int = 20
BUCKET_A_MAX: float = 4.0
BUCKET_B_TS: int = 40
BUCKET_B_MAX: float = 8.0
INITIAL_STATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, -0.5),
    (1.0, 1.0),
)


def simulate_fn(predictors, ts, covariates, y0, solver):
    """Undamped oscillator. ``scale`` does not enter the dynamics."""
    omega = predictors["omega"].value

    def vector_field(t, y, args):
        return jnp.stack([y[1], -(omega**2) * y[0]])

    return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)


def state_to_output(state: Array) -> Array:
    """Only position is measured."""
    return state[..., :1]


def huber_loss(pred_obs: Array, bp) -> Array:
    """Masked Huber loss. Any ``(pred_obs, bp) -> scalar`` callable works."""
    abs_res = jnp.abs(pred_obs - bp.y_observed)
    delta = 0.1
    quadratic = jnp.minimum(abs_res, delta)
    loss = 0.5 * quadratic**2 + delta * (abs_res - quadratic)
    return jnp.sum(jnp.where(bp.mask, loss, 0.0)) / jnp.maximum(jnp.sum(bp.mask), 1)


def weight_decay(predictors, points):
    """L2 on ``scale``, replacing the bound penalty.

    ``points`` are the per-leaf penalty points the bound penalty would use.
    A regulariser that needs none ignores them.
    """
    return predictors["scale"].value ** 2


def bucket_weight(bp) -> float:
    """Weight a bucket by its experiment count, so 3 experiments outweigh 1."""
    return float(bp.y_observed.shape[0])


def build_dataset() -> hm.Dataset:
    """Noisy positions on two time grids, so the dataset forms two buckets."""
    ts_a = jnp.linspace(0.0, BUCKET_A_MAX, BUCKET_A_TS)
    ts_b = jnp.linspace(0.0, BUCKET_B_MAX, BUCKET_B_TS)
    grids = (ts_a, ts_a, ts_a, ts_b)
    experiments = []
    for i, ((x0, v0), ts) in enumerate(zip(INITIAL_STATES, grids, strict=True)):
        clean = x0 * jnp.cos(OMEGA_TRUE * ts) + (v0 / OMEGA_TRUE) * jnp.sin(OMEGA_TRUE * ts)
        noise = NOISE_STD * jax.random.normal(jax.random.key(i), ts.shape)
        experiments.append(
            hm.make_experiment(
                channels={"position": hm.ChannelObs(ts=ts, values=clean + noise)},
                y0=[x0, v0],
                exp_id=f"osc_{i}",
            )
        )
    return hm.make_dataset(experiments, output_channel_names=("position",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    ds = build_dataset()
    print("[data]")
    print(hm.describe_buckets(ds))

    # ``scale`` exists only for the regulariser to pull toward zero.
    predictors = {
        "omega": hm.Parameter(1.5, bounds=(0.5, 2.0)),
        "scale": hm.Parameter(1.0),
    }
    solver = hm.SolverConfig(solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=4096, dt0=0.05)
    # Every latent trains; scalers stay fixed. Freezing a leaf is a mask edit.
    mask = hm.trainable_mask(predictors)

    bucket_step = build_bucket_step(
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        loss_fn=huber_loss,
        trainable=mask,
    )
    penalty_step = build_penalty_step(penalty_fn=weight_decay, trainable=mask)
    optimizer = optax.adamw(learning_rate=1e-2)
    apply_update = build_apply_update(optimizer, mask)
    opt_state = optimizer.init(eqx.filter(predictors, mask))

    # The fraction of each time series the loss sees. 1.0 scores all of it;
    # a schedule over this value is how the stock trainer's horizon works.
    horizon = jnp.asarray(1.0)
    penalty_weight = jnp.asarray(0.1)
    print("\n[fit]")
    for step in range(args.steps):
        # Weighted mean over buckets of the loss and its gradient. Dividing
        # by the total weight keeps the weights relative.
        total_loss = jnp.asarray(0.0)
        total_weight = 0.0
        grads_sum = jax.tree.map(jnp.zeros_like, eqx.filter(predictors, mask))
        for bp in ds.bucket_payloads:
            loss, grads = bucket_step(predictors, bp, horizon)
            w = bucket_weight(bp)
            total_loss = total_loss + w * loss
            total_weight += w
            grads_sum = jax.tree.map(lambda a, g, w=w: a + w * g, grads_sum, grads)
        data_loss = total_loss / total_weight
        grads = jax.tree.map(lambda g, t=total_weight: g / t, grads_sum)

        # The regulariser is charged once per step, outside the bucket loop.
        # It ignores penalty points, so none are passed.
        penalty, penalty_grads = penalty_step(predictors, penalty_weight, ())
        grads = jax.tree.map(jnp.add, grads, penalty_grads)

        predictors, opt_state = apply_update(predictors, grads, opt_state)
        if step % 25 == 0:
            print(
                f"step {step:3d}  data={float(data_loss):.5f}  penalty={float(penalty):.3e}  "
                f"omega={float(predictors['omega'].value):.4f}"
            )

    print("\n[results]")
    print(f"recovered omega = {float(predictors['omega'].value):.4f}  (truth {OMEGA_TRUE})")
    print(f"final scale     = {float(predictors['scale'].value):.6f}  (regularised toward 0)")

    preds = hm.predict_dataset(
        predictors, ds, simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver
    )
    hm.print_metrics(hm.compute_metrics(preds, ds), header="custom-loop result")

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            parity_diagnostics(preds, ds),
            title="Custom loop parity",
            save_path=args.plot_dir / "parity.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
