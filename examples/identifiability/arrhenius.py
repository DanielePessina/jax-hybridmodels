"""How well the data pin an Arrhenius rate law, without sampling.

First-order decay ``dy/dt = -k(T) y`` with ``k = A exp(-E / T)``, measured at
seven temperatures between 290 and 350 K. Written with ``log A`` and ``E``, the
two are almost perfectly correlated: the data fix the rate over a narrow band of
temperatures, and many ``(log A, E)`` pairs give nearly the same rates there.

Shows:

- ``InferenceProblem.optimize`` for the maximum likelihood and the posterior mode
- ``fisher`` for standard errors, correlations and the condition number
- the same law centred at a reference temperature, which removes the correlation
- ``profiles`` for likelihood-based intervals that need no Gaussian assumption
- ``laplace`` for a Gaussian posterior and draws that ``physical`` reads

Run::

    uv run python examples/identifiability/arrhenius.py

About 30 seconds on CPU.
"""

from __future__ import annotations

import jax

# Curvatures and their inverses need double precision.
jax.config.update("jax_enable_x64", True)

from typing import Any  # noqa: E402

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

import jaxhybridmodels as hm  # noqa: E402

TEMPERATURES = (290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0)
TRUE_LOG_A, TRUE_E = 17.14, 6000.0  # k(320 K) = 0.2
NOISE_SD = 0.01
CONFIG = hm.OptaxTrainingConfig(steps=1500, lr=0.05)


def make_data(key: jax.Array) -> list[hm.Experiment]:
    """One noisy decay curve per temperature, from the closed form."""
    ts = jnp.linspace(0.0, 25.0, 15)
    experiments = []
    for temperature in TEMPERATURES:
        key, sub = jr.split(key)
        rate = jnp.exp(TRUE_LOG_A - TRUE_E / temperature)
        values = jnp.exp(-rate * ts) + NOISE_SD * jr.normal(sub, ts.shape)
        experiments.append(
            hm.make_experiment(
                channels={"y": hm.ChannelObs(ts=ts, values=values, variance=NOISE_SD**2)},
                covariates={"temperature": temperature},
                y0=[1.0],
                exp_id=f"T={temperature:.0f}",
            )
        )
    return experiments


def decay_model(predictors: dict[str, Any], rate: Any) -> hm.HybridModel:
    """First-order decay whose rate is ``rate(predictors, temperature)``."""

    def simulate_fn(p: Any, ts: jax.Array, covariates: Any, y0: jax.Array, solver: Any) -> Any:
        k = rate(p, covariates["temperature"])
        return solver.diffeqsolve(diffrax.ODETerm(lambda t, y, args: -k * y), ts, y0)

    return hm.HybridModel(
        predictors=predictors,
        simulate_fn=simulate_fn,
        state_to_output=lambda state: state,
        output_channel_names=("y",),
        solver=hm.SolverConfig(rtol=1e-8, atol=1e-10),
    )


# region model
def textbook_model() -> hm.HybridModel:
    def rate(p: Any, temperature: jax.Array) -> jax.Array:
        return jnp.exp(p["log_A"].value - p["E"].value / temperature)

    return decay_model(
        {
            "log_A": hm.Parameter(15.0, bounds=(0.0, 40.0)),
            "E": hm.Parameter(5000.0, bounds=(0.0, 2e4)),
        },
        rate,
    )


def centred_model(reference: float) -> hm.HybridModel:
    """The same law, with ``log k`` at ``reference`` kelvin in place of ``log A``."""

    def rate(p: Any, temperature: jax.Array) -> jax.Array:
        return jnp.exp(p["log_k_ref"].value - p["E"].value * (1 / temperature - 1 / reference))

    return decay_model(
        {
            "log_k_ref": hm.Parameter(-2.0, bounds=(-10.0, 5.0)),
            "E": hm.Parameter(5000.0, bounds=(0.0, 2e4)),
        },
        rate,
    )


# endregion model


def main() -> None:
    data = make_data(jr.key(0))

    # region fisher
    fit = textbook_model().fit(data, CONFIG, key=jr.key(1), verbose=False)
    problem = hm.inference.make_problem(fit.model, data, log_prior=None, noise="data")
    mle = problem.optimize()  # the fit averages buckets, so it is close but not the MLE
    fisher = mle.problem.fisher()
    print(fisher.summary())
    # endregion fisher
    if not mle.converged or abs(float(fisher.correlation()[0, 1])) < 0.99:
        raise SystemExit("expected a converged MLE with log A and E almost fully correlated")

    # region profile
    profiles = mle.problem.profiles()
    for profile in profiles.values():
        print(profile.summary())
    # endregion profile
    wald = fisher.intervals()
    for name, profile in profiles.items():
        low, high = profile.bounds
        if low is None or high is None or abs(low - wald[name][0]) > 0.1 * (high - low):
            raise SystemExit(f"{name}: the profile should close near the Wald interval here")

    # region centred
    # log k(T_ref) = log A - E / T_ref is uncorrelated with E when
    # T_ref = var(E) / cov(log A, E), which the Fisher covariance gives directly.
    covariance = fisher.physical_covariance()
    e, log_a = fisher.names.index("E"), fisher.names.index("log_A")
    reference = float(covariance[e, e] / covariance[e, log_a])
    print(f"reference temperature {reference:.1f} K")

    fit = centred_model(reference).fit(data, CONFIG, key=jr.key(1), verbose=False)
    problem = hm.inference.make_problem(fit.model, data, log_prior=None, noise="data")
    centred = problem.optimize().problem.fisher()
    print(centred.summary())
    # endregion centred
    if abs(float(centred.correlation()[0, 1])) > 0.05:
        raise SystemExit("centring at the reference temperature should remove the correlation")
    if centred.condition_number > fisher.condition_number / 10:
        raise SystemExit("centring should improve the condition number by orders of magnitude")

    # region laplace
    def log_prior(predictors: Any, noise: Any) -> jax.Array:
        log_k_ref, energy = predictors["log_k_ref"].value, predictors["E"].value
        return -0.5 * ((log_k_ref + 2.0) / 3.0) ** 2 - 0.5 * ((energy - 6000.0) / 3000.0) ** 2

    bayesian = hm.inference.make_problem(fit.model, data, log_prior=log_prior, noise="data")
    laplace = bayesian.optimize(target="posterior").problem.laplace()
    print(laplace.summary())
    draws = bayesian.physical(laplace.sample(key=jr.key(2), n_draws=2000))
    print({name: f"{float(v.mean()):.4g} +- {float(v.std()):.3g}" for name, v in draws.items()})
    # endregion laplace


if __name__ == "__main__":
    main()
