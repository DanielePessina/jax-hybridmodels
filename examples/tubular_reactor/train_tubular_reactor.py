"""Learn a reaction rate inside a PDE: a tubular reactor with axial dispersion.

A liquid flows through a tube at speed ``u`` and reacts as it goes. The
concentration ``c(z, t)`` obeys an advection-dispersion-reaction equation,

    dc/dt = -u dc/dz + D d2c/dz2 - r(c, T),

discretised on ``N_CELLS`` cells (the method of lines), which turns the
PDE into an ODE with ``N_CELLS`` states. Transport is known physics. The
rate law is not: the truth is Langmuir-Hinshelwood,
``r = k(T) c / (1 + K c)``, and the model only knows that ``r = g(c, T) c``
with ``g`` positive. A network learns ``g``. Four sensors along the tube
see the concentration at irregular times; the other cells are never
measured.

A mechanism-only model that assumes first-order Arrhenius kinetics,
``r = k(T) c``, is fitted to the same data for comparison.

Shows:

- a 50-state ``simulate_fn`` with a ``BoundedPredictor`` evaluated at every
  cell, and a ``state_to_output`` that picks sensor cells
- covariates feeding a time-varying inlet through ``hm.step_profile``
- ``Parameter`` with a log warp for the mechanistic constants
- ``RecursiveCheckpointAdjoint`` for a network inside the vector field
- ``HybridModel.simulate`` for the full space-time field

Run::

    uv run python examples/tubular_reactor/train_tubular_reactor.py
    uv run python examples/tubular_reactor/train_tubular_reactor.py --steps 5 --no-plot

Prints the data layout, both fits' metrics and the learned rate against
the truth. Writes three figures to ``figures/``. About two minutes on CPU.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import jaxhybridmodels as hm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import apply_default_style  # noqa: E402

# Tube and flow. Residence time L / u = 1; Peclet number u L / D = 100.
LENGTH, VELOCITY, DISPERSION = 1.0, 1.0, 0.01
N_CELLS = 50
DZ = LENGTH / N_CELLS
Z = (jnp.arange(N_CELLS) + 0.5) * DZ  # cell centres

# Sensors at a quarter, half and three quarters of the tube, and the outlet.
SENSORS = {"z25": 12, "z50": 24, "z75": 37, "outlet": N_CELLS - 1}
CHANNELS = tuple(SENSORS)
SENSOR_CELLS = jnp.array(list(SENSORS.values()))

# The feed drops to 30% of its initial concentration at T_DROP.
T_END, T_DROP = 3.0, 1.5
# (temperature K, feed concentration mol/L), one experiment each.
CONDITIONS = ((310.0, 2.0), (320.0, 1.0), (325.0, 1.5), (335.0, 2.0), (345.0, 0.5), (350.0, 1.5))
NOISE_STD = 0.01
T_REF = 330.0

# Not stiff at this resolution: the fastest mode is dispersion, about
# 4 D / DZ^2 = 100 per unit time, so explicit Tsit5 needs a few hundred
# steps. Halving DZ quadruples that rate; past roughly 200 cells an implicit
# solver such as Kvaerno3 wins. RecursiveCheckpoint keeps the backward pass
# through the network in the vector field at O(log steps) memory; the
# default DirectAdjoint stores every step and compiles in time that grows
# with max_steps.
SOLVER = hm.SolverConfig(
    solver=diffrax.Tsit5(),
    rtol=1e-5,
    atol=1e-7,
    max_steps=4096,
    dt0=1e-3,
    adjoint=diffrax.RecursiveCheckpointAdjoint(),
)


def true_g(c: ArrayLike, temp: ArrayLike) -> Array:
    """Effective first-order coefficient of the true rate law, ``r = g c``."""
    c, temp = jnp.asarray(c), jnp.asarray(temp)
    k = 3.0 * jnp.exp(-4000.0 * (1.0 / temp - 1.0 / T_REF))
    return k / (1.0 + 2.0 * c)


def transport(c: Array, c_in: Array) -> Array:
    """Upwind advection and central dispersion, inlet held at ``c_in``, zero-gradient outlet."""
    left = jnp.concatenate([c_in[None], c[:-1]])
    right = jnp.concatenate([c[1:], c[-1:]])
    return -VELOCITY * (c - left) / DZ + DISPERSION * (left - 2.0 * c + right) / DZ**2


def solve_tube(g_fn, ts: Array, covariates, y0: Array, solver: hm.SolverConfig) -> Array:
    """Integrate the discretised tube for a rate ``g_fn(c, T) * c``."""
    temp, feed = covariates["T"], covariates["c_feed"]
    c_in = hm.step_profile(feed, 0.3 * feed, T_DROP)

    def vector_field(t, c, args):
        return transport(c, c_in(t)) - g_fn(c, temp) * c

    ys = solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys
    assert ys is not None
    return ys


def simulate_hybrid(predictors, ts, covariates, y0, solver):
    net = predictors["g"]
    # One network call per cell; T is shared along the tube.
    g_fn = jax.vmap(lambda c, temp: net({"c": c, "T": temp})[0], in_axes=(0, None))
    return solve_tube(g_fn, ts, covariates, y0, solver)


def simulate_mechanistic(predictors, ts, covariates, y0, solver):
    k_ref, ea_over_r = predictors["k_ref"].value, predictors["Ea_over_R"].value

    def g_fn(c, temp):
        return k_ref * jnp.exp(-ea_over_r * (1.0 / temp - 1.0 / T_REF)) * jnp.ones_like(c)

    return solve_tube(g_fn, ts, covariates, y0, solver)


def state_to_output(state: Array) -> Array:
    return state[..., SENSOR_CELLS]


def exp_name(temp: float, feed: float) -> str:
    return f"T{temp:.0f}_c{feed:g}"


def make_data(key: Array) -> hm.Dataset:
    """Each sensor samples 8 random times per experiment, so no two grids match."""
    truth = hm.HybridModel(
        predictors={},
        simulate_fn=lambda p, ts, cov, y0, s: solve_tube(true_g, ts, cov, y0, s),
        state_to_output=state_to_output,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    experiments = []
    for (temp, feed), k in zip(CONDITIONS, jr.split(key, len(CONDITIONS)), strict=True):
        k_times, k_noise = jr.split(k)
        times = jnp.sort(jr.uniform(k_times, (len(CHANNELS), 8), minval=0.1, maxval=T_END), axis=1)
        noise = NOISE_STD * jr.normal(k_noise, times.shape)
        covariates: dict[str, float | Array] = {"T": temp, "c_feed": feed}
        channels = {}
        for i, name in enumerate(CHANNELS):
            clean = truth.simulate(
                ts=times[i], y0=jnp.zeros(N_CELLS), covariates=covariates, t0=0.0
            )
            channels[name] = hm.ChannelObs(times[i], clean.outputs[name] + noise[i])
        experiments.append(
            hm.make_experiment(
                channels=channels,
                covariates=covariates,
                y0=jnp.zeros(N_CELLS),
                t0=0.0,
                exp_id=exp_name(temp, feed),
            )
        )
    return hm.make_dataset(experiments, output_channel_names=CHANNELS)


def build_models(key: Array) -> tuple[hm.HybridModel, hm.HybridModel]:
    net = hm.BoundedPredictor(
        inner=hm.MLPPredictor(
            in_size=2, out_size=1, width_size=16, depth=2, activation_name="tanh", key=key
        ),
        input_keys=("c", "T"),
        input_bounds=((0.0, 2.2), (300.0, 360.0)),
        # g spans two decades over the box, so the output is log-warped.
        out_scaler=hm.BoundScaler(bounds=((0.05, 20.0),), warp="log"),
    )
    hybrid = hm.HybridModel(
        predictors={"g": net},
        simulate_fn=simulate_hybrid,
        state_to_output=state_to_output,
        output_channel_names=CHANNELS,
        solver=SOLVER,
    )
    # Same sensors and solver; only the rate law differs.
    mechanistic = replace(
        hybrid,
        predictors={
            "k_ref": hm.Parameter(1.0, bounds=(0.05, 50.0), warp="log"),
            "Ea_over_R": hm.Parameter(2000.0, bounds=(0.0, 10000.0)),
        },
        simulate_fn=simulate_mechanistic,
    )
    return hybrid, mechanistic


def plot(args, dataset, hybrid, mechanistic) -> None:
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    pred_h = hybrid.predict(dataset)
    dense = jnp.linspace(0.0, T_END, 200)

    # Sensor traces for the coldest and hottest runs.
    shown = (CONDITIONS[0], CONDITIONS[-1])
    fig, axes = plt.subplots(len(shown), len(CHANNELS), figsize=(13, 5.5), sharex=True)
    for row, (temp, feed) in zip(axes, shown, strict=True):
        cov = {"T": temp, "c_feed": feed}
        sim_h = hybrid.simulate(ts=dense, y0=jnp.zeros(N_CELLS), covariates=cov, t0=0.0)
        sim_m = mechanistic.simulate(ts=dense, y0=jnp.zeros(N_CELLS), covariates=cov, t0=0.0)
        for ax, name in zip(row, CHANNELS, strict=True):
            obs = pred_h[exp_name(temp, feed)]
            mask = np.asarray(obs.observation_mask[name])
            ax.plot(np.asarray(obs.times)[mask], np.asarray(obs.observed[name])[mask], "ko", ms=3)
            ax.plot(dense, sim_h.outputs[name], color="tab:red", label="hybrid")
            ax.plot(dense, sim_m.outputs[name], color="tab:blue", ls="--", label="first order")
            ax.set_title(f"{exp_name(temp, feed)}, {name}", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time")
    axes[0, 0].set_ylabel("c")
    axes[1, 0].set_ylabel("c")
    fig.savefig(args.plot_dir / "sensors.png")
    plt.close(fig)

    # The learned rate r = g c against the truth over the concentration range.
    c = jnp.linspace(0.0, 2.0, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for temp, colour in zip(
        (310.0, 330.0, 350.0), ("tab:blue", "tab:green", "tab:red"), strict=True
    ):
        learned = jax.vmap(lambda ci, t=temp: hybrid.predictors["g"]({"c": ci, "T": t})[0])(c)
        ax.plot(c, true_g(c, temp) * c, color=colour, ls="--", lw=1.2)
        ax.plot(c, learned * c, color=colour, label=f"T = {temp:.0f} K")
    ax.set_xlabel("concentration c")
    ax.set_ylabel("rate r = g c")
    ax.set_title("Learned rate (solid) against truth (dashed)")
    ax.legend(fontsize=8)
    fig.savefig(args.plot_dir / "rate_law.png")
    plt.close(fig)

    # Space-time field of the hottest run, which no sensor sees whole.
    temp, feed = shown[-1]
    sim = hybrid.simulate(
        ts=dense, y0=jnp.zeros(N_CELLS), covariates={"T": temp, "c_feed": feed}, t0=0.0
    )
    assert sim.state is not None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    image = ax.pcolormesh(np.asarray(Z), np.asarray(dense), np.asarray(sim.state), shading="auto")
    ax.grid(False)
    for cell in SENSORS.values():
        ax.axvline(float(Z[cell]), color="white", lw=0.8, ls=":")
    fig.colorbar(image, ax=ax, label="c")
    ax.set_xlabel("position z")
    ax.set_ylabel("time")
    ax.set_title(f"Hybrid model field, {exp_name(temp, feed)} (dotted: sensors)")
    fig.savefig(args.plot_dir / "field.png")
    plt.close(fig)
    print(f"\n[plot] figures written to {args.plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="Adam steps for each model")
    parser.add_argument("--plot-dir", type=Path, default=Path(__file__).parent / "figures")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    apply_default_style()

    k_data, k_model, k_fit = jr.split(jr.key(0), 3)
    print("[data]")
    dataset = make_data(k_data)
    print(hm.describe_buckets(dataset))

    hybrid, mechanistic = build_models(k_model)
    config = hm.OptaxTrainingConfig(
        phases=(
            # Fit the start-up front first, then the whole run with the feed drop.
            hm.TrainingPhase(args.steps // 3, lr=1e-2, horizon=0.5),
            hm.TrainingPhase(args.steps - args.steps // 3, lr=3e-3),
        ),
        verbose=False,
    )
    print("\n[fit]")
    fits = {}
    for name, model in (("hybrid", hybrid), ("first order", mechanistic)):
        fits[name] = model.fit(dataset, config, key=k_fit)
        print(
            f"  {name}: loss {fits[name].history.loss[0]:.3e} -> {min(fits[name].history.loss):.3e}"
        )

    print("\n[results]")
    for name, result in fits.items():
        hm.print_metrics(
            hm.compute_metrics(result.model.predict(dataset).buckets, dataset), header=name
        )
    hybrid, mechanistic = fits["hybrid"].model, fits["first order"].model
    k_ref, ea = (float(mechanistic.predictors[k].value) for k in ("k_ref", "Ea_over_R"))
    print(f"\n  first order fit: k_ref = {k_ref:.3f}, Ea/R = {ea:.0f} K (truth saturates in c)")
    print("  rate r = g c at:  c     T     truth   hybrid")
    for c, temp in ((0.2, 310.0), (1.5, 310.0), (0.2, 350.0), (1.5, 350.0)):
        learned = hm.evaluate_predictor(hybrid.predictors["g"], {"c": c, "T": temp}) * c
        truth = float(true_g(c, temp)) * c
        print(f"{'':18}{c:4.1f}  {temp:4.0f}  {truth:6.3f}   {learned:6.3f}")

    if not args.no_plot:
        plot(args, dataset, hybrid, mechanistic)


if __name__ == "__main__":
    main()
