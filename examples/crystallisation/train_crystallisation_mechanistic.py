"""Crystallisation mechanistic-only training example (CNT + power-law growth).

Same hardcoded LowData4 cut and method-of-moments backbone as
``train_kinetic.py``, but the two MLP rate predictors are replaced by a
four-scalar mechanistic kinetic law that is global across experiments:

* CNT nucleation:    ``J = exp(logA) * S * exp(-16π γ³ v² / (3 (k_B T)³ ln²S))``
* Power-law growth:  ``G = (10**Ag / 60) * max(S - 1, 0)**g``

The four trainable scalars ``(logA, gamma, Ag, g)`` are stored in latent space
inside a small :class:`KineticParameters` eqx.Module and pushed through a
sigmoid :class:`BoundScaler` so the optimiser sees an unconstrained problem
while the simulator always receives parameters inside the physical box.

Optimisation uses ``train_with_evosax`` (CMA-ES). Four parameters is the
regime the evosax loop was sized for; gradient-based optimisation is overkill
and the population-based search avoids the local minima that the CNT
exponential creates near the metastable limit.

Run: ``uv run python examples/crystallisation/train_crystallisation_mechanistic.py``
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax

# x64 must be enabled before any other JAX-touching import. The population-
# balance moments span ~18 decades during integration; float32 mass balance
# drifts visibly within a single experiment.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
from jax import Array  # noqa: E402
from jaxtyping import Float  # noqa: E402

from hybridmodels import (  # noqa: E402
    BoundScaler,
    ChannelObs,
    Experiment,
    SolverConfig,
    make_dataset,
    make_experiment,
    predict_dataset,
)
from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
    trajectory_plot,
)

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Four hardcoded experiments from the thesis ``Unseeded_LowData4`` sheet,
# identical to the dataset used by ``train_kinetic.py`` so the two scripts are
# directly comparable. Conc and d43 rounded to 1 dp; d43 variance to 3 dp;
# concentration variance is a single made-up scalar (``CONC_VAR``) applied per
# row. The thesis ``Loading`` column is uniformly zero across LowData4 and is
# intentionally omitted as a covariate.
EXPERIMENTS_DATA: tuple[dict[str, object], ...] = (
    {
        "exp_id": "LowData4_3",
        "temperature_C": 17.0,
        "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
        "conc": (14.7, 13.7, 7.7, 7.0, 5.7, 5.5, 5.4, 5.1, 5.2),
        "d43_time_min": 270.0,
        "d43": 9.2,
        "d43_var": 5.345,
    },
    {
        "exp_id": "LowData4_4",
        "temperature_C": 17.0,
        "time_min": (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0),
        "conc": (11.6, 10.8, 10.8, 10.5, 10.9, 9.5, 6.8, 6.1, 5.7),
        "d43_time_min": 270.0,
        "d43": 7.7,
        "d43_var": 3.734,
    },
    {
        "exp_id": "LowData4_7",
        "temperature_C": 21.0,
        "time_min": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
        "conc": (16.8, 12.1, 8.6, 7.6, 7.2, 6.8, 6.4),
        "d43_time_min": 360.0,
        "d43": 10.5,
        "d43_var": 1.421,
    },
    {
        "exp_id": "LowData4_9",
        "temperature_C": 21.0,
        "time_min": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 375.0),
        "conc": (14.4, 14.2, 13.6, 10.1, 9.3, 7.6, 7.0),
        "d43_time_min": 375.0,
        "d43": 11.9,
        "d43_var": 0.267,
    },
)
# Made-up uniform concentration variance, broadcast across every row.
CONC_VAR = 0.1

# Bounds for the four mechanistic kinetic parameters. Reproduced from the
# original thesis-package bounds in ``hybridcrystals/regressor_constants.py``
# so the trained scalars sit in the same physical box as the source-package
# runs. ``out_scaler`` interprets these as ``(low, high)`` per-component.
LOGA_BOUNDS = (20.0, 65.0)  # ln(A) for CNT pre-exponential
GAMMA_BOUNDS = (0.15, 1.0)  # interfacial energy [mJ/m^2]
AG_BOUNDS = (-20.0, -5.0)  # log10 of growth pre-factor [m/s]
G_BOUNDS = (1.0, 3.5)  # power-law growth exponent

# ODE constants (from hybridcrystals/mechanistic.py).
RHO_C = 1370.0  # crystal density [kg/m^3]
K_V = 0.81  # volumetric shape factor
M_V = 2.97e-26  # molecular volume [m^3]
K_B = 1.38064852e-23  # Boltzmann constant [J/K]
D43_MAX = 55.0  # upper guard on d43 [um]
D43_MU3_EPS = 1e-6  # mu3 floor for the d43 ratio
META_EPS = 1e-5  # supersaturation must exceed 1 + eps for nucleation/growth

OUTPUT_CHANNELS = ("conc", "d43")


# --------------------------------------------------------------------------- #
# Trainable component: four mechanistic kinetic scalars                       #
# --------------------------------------------------------------------------- #


class KineticParameters(eqx.Module):
    """Four global mechanistic kinetic constants ``[logA, gamma, Ag, g]``.

    The framework's ``BoundedPredictor`` is built around a covariate-keyed
    ``__call__(dict | Array) -> Array`` and requires at least one input;
    here the parameters are *global* (no covariate dependence at all), so we
    sidestep ``BoundedPredictor`` and define a minimal ``eqx.Module`` whose
    only inexact-array leaf is the four-vector latent. The :class:`BoundScaler`
    output mapping is the same primitive the MLP-based predictor uses, so the
    optimiser still operates in an unbounded latent space and the simulator
    still sees physical-units parameters.

    Calling produces ``[logA, gamma, Ag, g]`` in physical units. CMA-ES sees
    a four-dimensional unbounded search; the sigmoid in ``out_scaler`` keeps
    every candidate inside the physical box no matter how wide the search
    spreads.
    """

    latent: Float[Array, " 4"]
    out_scaler: BoundScaler

    def __init__(self, *, key: Array) -> None:
        # Small Gaussian init in latent space puts the physical parameters
        # near the centre of each bound at gen 0; CMA-ES expands from there
        # under ``sigma_init``. Zero init would also work but a tiny noise
        # break ties between identical re-init seeds in tests.
        self.latent = jr.normal(key, (4,)) * 0.1
        self.out_scaler = BoundScaler(
            bounds=(LOGA_BOUNDS, GAMMA_BOUNDS, AG_BOUNDS, G_BOUNDS),
            transform="sigmoid",
        )

    def __call__(self) -> Float[Array, " 4"]:
        """Return the four mechanistic parameters in physical units."""
        return self.out_scaler.from_latent(self.latent)


# --------------------------------------------------------------------------- #
# Hook callables consumed by the framework                                    #
# --------------------------------------------------------------------------- #
# These three functions are passed into ``make_experiment``, ``make_dataset``,
# and ``train_with_evosax`` respectively. They have to live at module level so
# their identities are stable across calls.


def y0_fn(covariates: dict[str, Array], channels: dict[str, ChannelObs]) -> Float[Array, " 6"]:
    """Initial state ``[mu0..mu4, conc]``: moments at zero, conc at first observation."""
    init_conc = jnp.asarray(channels["conc"].values[0])
    return jnp.concatenate([jnp.zeros(5, dtype=init_conc.dtype), init_conc[None]])


def state_to_output(state: Float[Array, "T 6"]) -> Float[Array, "T 2"]:
    """Project full state ``[mu0..mu4, conc]`` to observed channels ``[conc, d43]``."""
    mu3 = state[..., 3]
    mu4 = state[..., 4]
    conc = state[..., 5]
    # d43 = (mu4 / mu3) * 1e6 [um], guarded against the early-time near-zero-
    # moments regime. The double-``where`` pattern is the canonical JAX-grad-
    # safe guarded division.
    safe_mu3 = jnp.where(mu3 > D43_MU3_EPS, mu3, 1.0)
    ratio = jnp.where(mu3 > D43_MU3_EPS, (mu4 / safe_mu3) * 1e6, 0.0)
    d43 = jnp.clip(jnp.where(jnp.isfinite(ratio) & (ratio > 0.0), ratio, 0.0), 0.0, D43_MAX)
    return jnp.stack([conc, d43], axis=-1)


def simulate_fn(
    predictor: KineticParameters,
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 6"],
    solver: SolverConfig,
) -> Float[Array, "T 6"]:
    """Integrate the method-of-moments ODE with CNT nucleation + power-law growth.

    The four physical parameters ``[logA, gamma, Ag, g]`` are evaluated once
    at the top of the call (they are global, not state-dependent) and closed
    over by the vector field. The ``(S > 1 + eps)`` mask gates both rates so
    the ODE stops moving below the metastable limit.
    """
    params = predictor()  # [4] in physical units, sigmoid-bounded
    logA = params[0]
    gamma_J_m2 = params[1] * 1e-3  # bounds are in [mJ/m^2]; CNT formula is [J/m^2]
    Ag = params[2]
    g_exp = params[3]

    temperature_C = covariates["temperature_C"]
    T_K = temperature_C + 273.15

    # Empirical saturation polynomial in °C (from hybridcrystals/mechanistic.py).
    conc_sat = (
        0.3705
        + 7.171e-2 * temperature_C
        - 1.924e-3 * temperature_C**2
        + 17.97e-5 * temperature_C**3
    )

    def vector_field(t: Array, y: Float[Array, " 6"], args: object) -> Float[Array, " 6"]:
        mu0, mu1, mu2, mu3, _mu4, conc = y[0], y[1], y[2], y[3], y[4], y[5]
        S = conc / conc_sat
        meta_mask = (S > 1.0 + META_EPS).astype(y.dtype)

        # CNT nucleation J = exp(logA) * S * exp(-16π γ³ v² / (3 (k_B T)³ ln²S)).
        # Clip S to a hair above 1 inside the log so the exponent stays finite
        # when meta_mask is zero — the mask wipes the contribution out anyway,
        # but the gradient through the clipped log must stay finite or autodiff
        # will return NaN at every ODE step on a sub-saturated trajectory.
        S_safe = jnp.clip(S, min=1.0 + 1e-12)
        logS = jnp.log(S_safe)
        cnt_exp = -16.0 * jnp.pi * gamma_J_m2**3 * M_V**2 / (3.0 * (K_B * T_K) ** 3 * logS**2)
        J = meta_mask * jnp.exp(logA) * S_safe * jnp.exp(cnt_exp)

        # Power-law growth G = (10**Ag / 60) * max(S - 1, 0)**g. The /60 maps
        # the per-minute pre-factor convention in the source package onto the
        # per-second SI integration.
        growth_drive = jnp.maximum(S - 1.0, 0.0)
        G = meta_mask * jnp.power(10.0, Ag) / 60.0 * jnp.power(growth_drive, g_exp)

        # Hulburt-Katz form for size-independent nucleation at zero size and
        # pure linear growth, plus a mass balance: solute lost equals crystal
        # volume gained.
        dmu0 = J
        dmu1 = G * mu0
        dmu2 = 2.0 * G * mu1
        dmu3 = 3.0 * G * mu2
        dmu4 = 4.0 * G * mu3
        dconc = -3.0 * K_V * RHO_C * G * mu2
        return jnp.stack([dmu0, dmu1, dmu2, dmu3, dmu4, dconc])

    # Dataset stores time in minutes; rate constants are SI-second-based.
    times_sec = ts * 60.0
    if solver.dt0 is None:
        # 1/1000 of the full span, floored at 1 s so the first step doesn't
        # underflow on short trajectories.
        span = times_sec[-1] - times_sec[0]
        dt0 = jnp.maximum(span / 1000.0, jnp.asarray(1.0, dtype=times_sec.dtype))
    else:
        dt0 = jnp.asarray(solver.dt0, dtype=times_sec.dtype)

    # diffrax.PIDController needs atol as a scalar/array — tuples don't
    # broadcast against the y_error PyTree leaves.
    atol = (
        jnp.asarray(solver.atol, dtype=times_sec.dtype)
        if isinstance(solver.atol, tuple)
        else solver.atol
    )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=times_sec[0],
        t1=times_sec[-1],
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=times_sec),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


# --------------------------------------------------------------------------- #
# Script body                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=80)
    parser.add_argument("--sigma-init", type=float, default=0.5)
    parser.add_argument(
        "--init",
        choices=("warm", "uniform_box", "lhs_box"),
        default="lhs_box",
        help="Gen-0 sampling mode for CMA-ES; lhs_box gives the broadest 4-D coverage.",
    )
    parser.add_argument("--init-box-extent", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    apply_default_style()
    k_init, k_train = jr.split(jr.PRNGKey(args.seed), 2)

    # ---- Build experiments from the hardcoded LowData4 cut ----------------- #
    # Each entry in ``EXPERIMENTS_DATA`` is converted into one
    # ``hybridmodels.Experiment`` carrying two channels with their own ``ts``
    # axes — the d43 channel has a single terminal observation, so
    # ``make_dataset`` will form per-experiment union axes and bucket on
    # length parity. ``CONC_VAR`` is broadcast to a per-row variance vector;
    # the d43 variance is scalar (one observation per experiment).
    experiments: list[Experiment] = []
    for data in EXPERIMENTS_DATA:
        time_min = jnp.asarray(data["time_min"], dtype=float)
        conc = jnp.asarray(data["conc"], dtype=float)
        d43_ts = jnp.asarray((data["d43_time_min"],), dtype=float)
        d43_vals = jnp.asarray((data["d43"],), dtype=float)
        d43_var = jnp.asarray((data["d43_var"],), dtype=float)

        experiments.append(
            make_experiment(
                covariates={"temperature_C": float(data["temperature_C"])},  # type: ignore[arg-type]
                channels={
                    "conc": ChannelObs(
                        ts=time_min,
                        values=conc,
                        variance=jnp.full_like(conc, CONC_VAR),
                    ),
                    "d43": ChannelObs(ts=d43_ts, values=d43_vals, variance=d43_var),
                },
                y0_fn=y0_fn,
                exp_id=str(data["exp_id"]),
            )
        )

    print(f"[load] {len(experiments)} hardcoded experiments")
    for exp in experiments:
        print(
            f"    {exp.exp_id}: T={float(exp.covariates['temperature_C']):.1f}°C, "
            f"conc obs={exp.channels['conc'].values.shape[0]}, "
            f"d43 obs={exp.channels['d43'].values.shape[0]}"
        )

    # ---- Build dataset (bucket + union-axis logic) ------------------------ #
    print("\n[build] dataset")
    dataset = make_dataset(
        experiments,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(dataset.bucket_payloads)} bucket(s)")
    for i, bp in enumerate(dataset.bucket_payloads):
        print(
            f"    bucket {i}: ts={tuple(bp.ts.shape)}, "
            f"y_observed={tuple(bp.y_observed.shape)}, "
            f"mask={tuple(bp.mask.shape)}, n_obs={int(bp.n_obs)}"
        )

    # ---- Solver ----------------------------------------------------------- #
    # Per-state atol matched to the natural moment magnitudes (``mu0 ~ 1e11``,
    # ``mu1 ~ 1e6``, ..., ``mu4 ~ 1e-7``, ``conc ~ 1``). A uniform atol
    # forces the PIDController to over-resolve small components and
    # under-resolve large ones, and the integrator hits ``max_steps`` before
    # finishing one trajectory. Setting atol per-component ~9 decades below
    # each natural magnitude lets ``rtol`` dominate once values are
    # appreciable, with atol acting as a near-zero floor.
    solver = SolverConfig(
        solver=diffrax.Kvaerno3(),
        rtol=1e-4,
        atol=(1e3, 1e-2, 1e-6, 1e-10, 1e-14, 1e-5),
        max_steps=500_000,
        dt0=None,
    )

    # ---- Build the four-scalar mechanistic predictor ---------------------- #
    predictor = KineticParameters(key=k_init)

    # Sanity-check evaluation. Same call shape ``simulate_fn`` makes inside
    # the trace.
    init_params = predictor()
    print(
        f"\n[init] logA={float(init_params[0]):.2f}, "
        f"gamma={float(init_params[1]):.3f} mJ/m^2, "
        f"Ag={float(init_params[2]):.2f}, "
        f"g={float(init_params[3]):.2f}"
    )

    # ---- Train ------------------------------------------------------------ #
    print("\n[train] evosax (CMA-ES, 4 parameters)")
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=args.population_size,
        num_generations=args.num_generations,
        init=args.init,
        init_box_extent=args.init_box_extent,
        sigma_init=args.sigma_init,
        loss="mse",
        verbose=True,
    )
    history, trained_predictor = train_with_evosax(
        predictor,
        dataset,
        config,
        simulate_fn=simulate_fn,
        solver=solver,
        key=k_train,
    )
    print(f"  {len(history)} generations; final best loss {history[-1]:.6f}")
    sample_every = max(1, len(history) // 10)
    print(
        f"  best-loss trace every ~{sample_every} gens: "
        f"{[f'{loss:.4f}' for loss in history[::sample_every]]}"
    )

    final_params = trained_predictor()
    print(
        f"  trained -> logA={float(final_params[0]):.2f}, "
        f"gamma={float(final_params[1]):.3f} mJ/m^2, "
        f"Ag={float(final_params[2]):.2f}, "
        f"g={float(final_params[3]):.2f}"
    )

    # ---- Diagnostics + plots --------------------------------------------- #
    # ``predict_dataset`` returns one ``[N, T, D]`` array per bucket; the
    # helpers walk it in lockstep with the dataset.
    print("\n[diagnostics] per-channel parity stats over the training set")
    predictions = predict_dataset(
        trained_predictor,
        dataset,
        simulate_fn=simulate_fn,
        solver=solver,
    )
    diag = compute_diagnostics(predictions, dataset)
    print_diagnostics(diag)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            diag,
            title="Crystallisation parity (mechanistic CNT + power-law)",
            save_path=args.plot_dir / "parity_mechanistic.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained_predictor,
            simulate_fn=simulate_fn,
            solver=solver,
            max_experiments=6,
            title="Crystallisation trajectories (mechanistic, first 6 experiments)",
            save_path=args.plot_dir / "trajectories_mechanistic.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
