"""How long a drug keeps its potency, from an accelerated stability study.

A drug degrades by first-order kinetics whose rate follows Arrhenius. At the
storage temperature, 25 °C, it degrades too slowly to measure in a few months, so
the study stores it at 40 to 70 °C instead, measures potency for three months,
and extrapolates the rate down to 25 °C. The shelf life is the time until potency
falls to 90% of the label claim.

The answer that matters is not the best estimate. A label claim must hold with
confidence, so it is set from the one-sided 95% lower bound on the shelf life,
and that bound comes entirely from how uncertain the extrapolated rate is.

Shows:

- ``InferenceProblem.optimize`` for the maximum likelihood, with no prior
- ``fisher`` and ``derived`` for the rate's uncertainty at each temperature, and
  how it grows away from the data
- ``profile`` for the lower bound on the shelf life, which sets the label claim,
  checked against the delta method's one-sided bound
- ``FisherResult.sample`` with ``predict`` for a potency band at 25 °C

Run::

    uv run python examples/shelf_life/shelf_life.py
    uv run python examples/shelf_life/shelf_life.py --no-plot

Writes ``figures/shelf_life.png``. About 20 seconds on CPU.
"""

from __future__ import annotations

import jax

# Curvatures and their inverses need double precision.
jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import jaxhybridmodels as hm  # noqa: E402

# ``examples/_shared`` is a sibling directory; put it on sys.path so the
# helpers import without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import apply_default_style  # noqa: E402

STORAGE_C = 25.0
STORAGE_K = 273.15 + STORAGE_C
TEST_TEMPERATURES_C = (40.0, 50.0, 60.0, 70.0)
MONTHS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
SPECIFICATION = 0.90  # potency, as a fraction of the label claim
ASSAY_SD = 0.01  # the assay's standard deviation, in the same units

# The truth the data are drawn from: 90% potency after 30 months at 25 °C, and an
# activation energy of 100 kJ/mol.
TRUE_K25 = math.log(1 / SPECIFICATION) / 30.0  # per month
TRUE_EA_OVER_R = 12000.0  # kelvin

FIGURES = Path(__file__).resolve().parent / "figures"


# region model
def rate(log_k25: Any, ea_over_r: Any, temperature_c: Any) -> Any:
    """Arrhenius, written around 25 °C so ``log_k25`` is the rate the label needs."""
    return jnp.exp(log_k25 - ea_over_r * (1 / (273.15 + temperature_c) - 1 / STORAGE_K))


def simulate_fn(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
    k = rate(
        predictors["log_k25"].value, predictors["Ea_over_R"].value, covariates["temperature_C"]
    )
    return solver.diffeqsolve(diffrax.ODETerm(lambda t, y, args: -k * y), ts, y0)


model = hm.HybridModel(
    predictors={
        "log_k25": hm.Parameter(-4.0, bounds=(-12.0, 0.0)),
        "Ea_over_R": hm.Parameter(10000.0, bounds=(2000.0, 30000.0)),
    },
    simulate_fn=simulate_fn,
    state_to_output=lambda state: state,
    output_channel_names=("potency",),
    solver=hm.SolverConfig(rtol=1e-8, atol=1e-10),
)
# endregion model


def shelf_life(k25: Any) -> Any:
    """Months until potency falls to the specification, at 25 °C."""
    return jnp.log(1 / SPECIFICATION) / k25


def make_study(key: jax.Array) -> list[hm.Experiment]:
    """Potency at each test temperature, from the closed form plus assay noise."""
    ts = jnp.asarray(MONTHS)
    study = []
    for temperature in TEST_TEMPERATURES_C:
        key, sub = jr.split(key)
        k = rate(math.log(TRUE_K25), TRUE_EA_OVER_R, temperature)
        potency = jnp.exp(-k * ts) + ASSAY_SD * jr.normal(sub, ts.shape)
        study.append(
            hm.make_experiment(
                channels={"potency": hm.ChannelObs(ts=ts, values=potency, variance=ASSAY_SD**2)},
                covariates={"temperature_C": temperature},
                y0=[1.0],
                exp_id=f"{temperature:.0f} C",
            )
        )
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Shelf life from an accelerated stability study")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = parser.parse_args()
    study = make_study(jr.key(0))

    # region fisher
    fit = model.fit(
        study, hm.OptaxTrainingConfig(steps=1500, lr=0.05), key=jr.key(1), verbose=False
    )
    problem = hm.inference.make_problem(fit.model, study, log_prior=None, noise="data")
    mle = problem.optimize().problem
    fisher = mle.fisher()
    print(fisher.summary())
    # endregion fisher

    # region extrapolation
    # log k(T) depends on both values, so its standard error uses their covariance,
    # correlation included. derived() takes the gradient and applies it.
    print("\nRelative uncertainty of the rate, one standard error:")
    spreads = {}
    for temperature in (STORAGE_C, *TEST_TEMPERATURES_C):
        log_k = fisher.derived(
            lambda values, t=temperature: jnp.log(rate(values["log_k25"], values["Ea_over_R"], t))
        )
        spreads[temperature] = float(log_k.standard_error)
        note = "  (storage, no data)" if temperature == STORAGE_C else ""
        print(f"  {temperature:4.0f} °C  ±{100 * spreads[temperature]:.1f}%{note}")
    # endregion extrapolation

    # region shelf-life
    # A two-sided 90% interval has a one-sided 95% bound at each end. The fastest
    # plausible rate gives the shortest plausible shelf life.
    profile = mle.profile("log_k25", confidence=0.90)
    slowest, fastest = profile.bounds
    if slowest is None or fastest is None:
        raise SystemExit(
            f"the data do not bound the rate, so no claim can be set: {profile.stop_reasons}"
        )
    estimate = shelf_life(math.exp(float(mle.physical(mle.position)["log_k25"])))
    lower = shelf_life(math.exp(fastest))
    label = math.floor(lower)
    print(f"\nShelf life at 25 °C: {estimate:.1f} months, best estimate")
    print(f"  one-sided 95% lower bound {lower:.1f} months, so a label claim of {label} months")
    # endregion shelf-life

    # region delta-method
    # The delta method gives the same kind of bound for any function of the values,
    # in one line, but assumes the shelf life itself is close to Gaussian.
    life = fisher.derived(lambda values: shelf_life(jnp.exp(values["log_k25"])))
    wald, _ = life.interval(confidence=0.95, side="lower")
    print(f"  delta method: {float(life.estimate):.1f} ± {float(life.standard_error):.1f} months,")
    print(f"  one-sided 95% lower bound {float(wald):.1f} months")
    # endregion delta-method

    # region band
    # Draws from the estimate's sampling distribution carry its uncertainty into
    # any function of it: here the shelf life, and the potency curve at 25 °C.
    draws = fisher.sample(key=jr.key(2), n_draws=4000)
    sampled = shelf_life(np.exp(np.asarray(mle.physical(draws)["log_k25"])))
    storage = hm.make_experiment(
        channels={"potency": hm.ChannelObs(ts=jnp.asarray([0.0]), values=jnp.asarray([1.0]))},
        covariates={"temperature_C": STORAGE_C},
        y0=[1.0],
        exp_id="25 C",
    )
    months = jnp.linspace(0.0, 48.0, 97)
    band = mle.predict(
        draws, [storage], ts=months, n_draws=400, quantiles=(0.05, 0.95), key=jr.key(3)
    )["25 C"]
    # endregion band
    print(f"  from 4000 draws: 5th percentile {np.percentile(sampled, 5):.1f} months")

    true_life = shelf_life(TRUE_K25)
    if not lower < true_life < shelf_life(math.exp(slowest)):
        raise SystemExit(f"the 90% interval should contain the true {true_life:.1f} months")
    if abs(np.percentile(sampled, 5) - lower) > 0.05 * lower:
        raise SystemExit("the draws' 5th percentile should agree with the profile bound")
    if abs(float(wald) - lower) > 0.05 * lower:
        raise SystemExit("the delta method's bound should agree with the profile's here")
    tested = [spreads[t] for t in TEST_TEMPERATURES_C]
    if not spreads[STORAGE_C] > max(tested) and spreads[STORAGE_C] > 3 * min(tested):
        raise SystemExit("extrapolating to 25 °C should widen the rate's uncertainty")

    if not args.no_plot:
        plot(mle, study, band, months, estimate=estimate, label=label)


def plot(
    mle: Any, study: list[hm.Experiment], band: Any, months: Any, *, estimate: float, label: int
) -> None:
    """The accelerated data with the fit, and the extrapolated 25 °C band with the claim."""
    import matplotlib.pyplot as plt

    apply_default_style()
    best = mle.to_model(mle.position)
    figure, (accelerated, storage) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = jnp.linspace(0.0, 3.0, 61)
    for experiment in study:
        observed = experiment.channels["potency"]
        (line,) = accelerated.plot(
            grid, best.simulate(experiment, ts=grid).outputs["potency"], lw=1.5
        )
        accelerated.plot(
            observed.ts,
            observed.values,
            "o",
            color=line.get_color(),
            ms=4,
            label=f"{experiment.covariates['temperature_C']:.0f} °C",
        )
    accelerated.set(
        xlabel="months", ylabel="potency (fraction of label)", title="Accelerated study: the data"
    )
    accelerated.legend(frameon=False)

    low, high = band.quantiles["potency"][0.05], band.quantiles["potency"][0.95]
    storage.fill_between(months, low, high, alpha=0.25, label="90% band")
    storage.plot(months, band.outputs["potency"], lw=1.5, label="prediction")
    # Known here only because the data are synthetic.
    storage.plot(months, np.exp(-TRUE_K25 * np.asarray(months)), "k:", lw=1, label="truth")
    storage.axhline(SPECIFICATION, color="k", lw=0.8, ls="--")
    storage.axvline(
        estimate, color="0.5", lw=1.0, ls="-.", label=f"best estimate, {estimate:.0f} months"
    )
    storage.axvline(label, color="C3", lw=1.4, label=f"label claim, {label} months")
    storage.set(
        xlabel="months",
        ylabel="potency (fraction of label)",
        title="25 °C storage: extrapolated, never measured",
        ylim=(0.8, 1.01),
    )
    storage.legend(frameon=False, loc="lower left")
    FIGURES.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / "shelf_life.png", dpi=150)
    print(f"[plot] {FIGURES / 'shelf_life.png'}")


if __name__ == "__main__":
    main()
