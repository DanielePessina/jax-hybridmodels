"""Put uncertainty on a rate law: NUTS through BlackJAX on the tubular reactor.

Uses the tube from ``train_tubular_reactor.py``. BlackJAX does the sampling, and
``hm.inference.make_problem`` supplies the log-density it targets.

First the mechanistic model, on data from a first-order Arrhenius law so it is
well specified. ``k_ref`` and ``Ea_over_R`` are the whole of what it fits, so this
is a complete posterior. With ``--last-layer``, the hybrid model too: the data
come from the saturating rate law, and only the network's last layer is sampled.
That posterior is conditional on the other layers, so its band is a lower bound
on the network's uncertainty.

Shows:

- ``make_problem`` with a prior on physical values and known noise
- ``InferenceProblem.preconditioned``, which evens out the scales the sampler sees
- BlackJAX window adaptation and NUTS, with chains on separate threads
- R-hat and effective sample size from ``blackjax.diagnostics``
- ``InferenceProblem.predict`` for a posterior predictive band
- ``free=`` naming a network layer, and ``conditional_on`` saying what stayed fixed

Needs ``pip install jax-hybridmodels[bayes,examples]``. Run::

    uv run python examples/tubular_reactor/sample_tubular_reactor.py
    uv run python examples/tubular_reactor/sample_tubular_reactor.py --last-layer
    uv run python examples/tubular_reactor/sample_tubular_reactor.py \\
        --warmup 10 --draws 10 --chains 2 --steps 5

Prints each fit, the posterior against the truth with its diagnostics, and the
time taken. On 8 CPU cores the mechanistic posterior takes under a minute, and
``--last-layer`` about ten more.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax

# Before anything uses JAX: HMC's energy check is sensitive to round-off.
jax.config.update("jax_enable_x64", True)

import blackjax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import jax.scipy.stats as jstats  # noqa: E402
import numpy as np  # noqa: E402
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction  # noqa: E402
from train_tubular_reactor import (  # noqa: E402
    CHANNELS,
    CONDITIONS,
    NOISE_STD,
    arrhenius_g,
    build_models,
    exp_name,
    make_data,
    true_g,
)

import jaxhybridmodels as hm  # noqa: E402

TRUE_K_REF, TRUE_EA = 1.5, 4000.0


def damped(model: hm.HybridModel) -> hm.HybridModel:
    """PID gains that reject fewer steps than the default, about 12% off each gradient."""
    solver = dataclasses.replace(model.solver, pcoeff=0.3, icoeff=0.4)
    return dataclasses.replace(model, solver=solver)


def first_order_g(c: Any, temp: Any) -> Any:
    """The law the mechanistic model assumes, so this data set is well specified."""
    return arrhenius_g(TRUE_K_REF, TRUE_EA, c, temp)


# region sample
def sample(problem: hm.inference.InferenceProblem, key: Any, args: Any) -> tuple[Any, Any]:
    """Window adaptation then NUTS, one thread per chain. Returns ``[chain, draw, ...]``."""

    def one_chain(chain_key: Any) -> Any:
        adapt_key, draw_key = jr.split(chain_key)
        adaptation = blackjax.window_adaptation(blackjax.nuts, problem.log_density)
        (state, params), _ = adaptation.run(
            adapt_key,
            problem.position,
            num_steps=args.warmup,  # ty: ignore[unknown-argument]
        )
        kernel = blackjax.nuts(problem.log_density, **params)

        def step(state: Any, step_key: Any) -> Any:
            state, info = kernel.step(step_key, state)
            return state, (state.position, info.is_divergent)

        _, out = jax.lax.scan(step, state, jr.split(draw_key, args.draws))
        return out

    # Compiled code releases the GIL, so threads run chains in parallel on
    # CPU. Vmapping them instead makes every chain wait for the deepest NUTS tree.
    keys = jr.split(key, args.chains)
    compiled = jax.jit(one_chain).lower(keys[0]).compile()
    with ThreadPoolExecutor(args.chains) as pool:
        chains = list(pool.map(compiled, keys))
    return jax.tree_util.tree_map(lambda *leaf: jnp.stack(leaf), *chains)
    # endregion sample


def report(named: dict[str, Any], divergent: Any, truth: dict[str, float]) -> None:
    print(f"  divergences: {int(np.asarray(divergent).sum())}")
    print(f"  {'value':<12} {'mean':>10} {'sd':>10} {'5%':>10} {'95%':>10} {'R-hat':>7} {'ESS':>7}")
    for name, value in named.items():
        chains = jnp.squeeze(value)
        low, high = np.quantile(np.asarray(chains), [0.05, 0.95])
        rhat = float(potential_scale_reduction(chains))
        ess = float(effective_sample_size(chains))
        print(
            f"  {name:<12} {float(chains.mean()):>10.4g} {float(chains.std()):>10.3g} "
            f"{low:>10.4g} {high:>10.4g} {rhat:>7.3f} {ess:>7.0f}  truth {truth[name]:g}"
        )


def rhat_range(positions: Any) -> tuple[float, float]:
    """The smallest and largest R-hat over every sampled element."""
    values = jnp.ravel(potential_scale_reduction(positions))
    return float(values.min()), float(values.max())


def mechanistic_posterior(args: Any, key: Any) -> None:
    k_data, k_fit, k_chain, k_predict = jr.split(key, 4)
    print("[mechanistic] data from a first-order law, full posterior")
    dataset = make_data(k_data, first_order_g, NOISE_STD**2)
    _, mechanistic = build_models(k_fit)
    config = hm.OptaxTrainingConfig(steps=args.steps, lr=1e-2)
    result = mechanistic.fit(
        dataset, config, objective=hm.Objective(loss="mle"), key=k_fit, verbose=False
    )
    print(result.summary())

    # region problem
    def log_prior(predictors: Any, noise: Any) -> Any:
        # Weakly informative, and written on physical values.
        k_ref, ea = predictors["k_ref"].value, predictors["Ea_over_R"].value
        return jstats.norm.logpdf(jnp.log(k_ref), 0.0, 1.5) + jstats.norm.logpdf(ea, 3000.0, 3000.0)

    problem = hm.inference.make_problem(
        damped(result.model), dataset, log_prior=log_prior, noise="data"
    ).preconditioned()
    # endregion problem
    start = time.perf_counter()
    positions, divergent = jax.block_until_ready(sample(problem, k_chain, args))
    print(f"\n  {args.chains} chains in {time.perf_counter() - start:.0f} s")
    named = jax.vmap(jax.vmap(problem.physical))(positions)
    report(named, divergent, {"k_ref": TRUE_K_REF, "Ea_over_R": TRUE_EA})
    correlation = np.corrcoef(
        np.asarray(named["k_ref"]).ravel(), np.asarray(named["Ea_over_R"]).ravel()
    )[0, 1]
    print(f"  correlation of k_ref and Ea/R: {correlation:.2f}")

    # region predict
    start = time.perf_counter()
    predictive = problem.predict(positions, key=k_predict, n_draws=args.predict_draws)
    # endregion predict
    print(f"  {len(predictive.members)} predictive draws in {time.perf_counter() - start:.0f} s")
    run = predictive[exp_name(*CONDITIONS[0])]
    name = CHANNELS[-1]
    print(
        f"  largest 90% band width at {name}, {exp_name(*CONDITIONS[0])}: "
        f"{float(jnp.max(run.quantiles[name][0.95] - run.quantiles[name][0.05])):.4f}"
    )


def last_layer_posterior(args: Any, key: Any) -> None:
    k_data, k_model, k_fit, k_chain = jr.split(key, 4)
    print("\n[hybrid] data from the saturating law, last layer only")
    dataset = make_data(k_data, true_g, NOISE_STD**2)
    hybrid, _ = build_models(k_model)
    config = hm.OptaxTrainingConfig(
        phases=(
            hm.TrainingPhase(args.steps // 3, lr=1e-2, horizon=0.5),
            hm.TrainingPhase(args.steps - args.steps // 3, lr=3e-3),
        ),
    )
    result = hybrid.fit(dataset, config, key=k_fit, verbose=False)
    print(result.summary())

    # region last-layer
    fitted = result.model.predictors["g"].inner.mlp.layers[-1]

    def log_prior(predictors: Any, noise: Any) -> Any:
        # Network weights have no physical meaning, so centre a prior on the fit.
        layer = predictors["g"].inner.mlp.layers[-1]
        return jnp.sum(jstats.norm.logpdf(layer.weight, fitted.weight, 1.0)) + jnp.sum(
            jstats.norm.logpdf(layer.bias, fitted.bias, 1.0)
        )

    problem = hm.inference.make_problem(
        damped(result.model),
        dataset,
        log_prior=log_prior,
        noise="data",
        free="g.inner.mlp.layers.2",
    ).preconditioned()
    print(f"  conditional on: {', '.join(problem.conditional_on)}")
    # endregion last-layer
    start = time.perf_counter()
    positions, divergent = jax.block_until_ready(sample(problem, k_chain, args))
    print(f"  {args.chains} chains in {time.perf_counter() - start:.0f} s")
    low, high = rhat_range(positions)
    print(f"  divergences: {int(np.asarray(divergent).sum())}; R-hat from {low:.3f} to {high:.3f}")
    flat = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), positions)
    weights = jax.tree_util.tree_map(lambda x: x[:: max(1, x.shape[0] // 100)], flat)
    rates = jax.vmap(
        lambda p: hm.evaluate_predictor(
            problem.to_model(p).predictors["g"], {"c": jnp.asarray(1.0), "T": jnp.asarray(330.0)}
        )
    )(weights)
    print(
        f"  g(c=1, T=330): {float(rates.mean()):.3f} +/- {float(rates.std()):.3f}, "
        f"truth {float(true_g(1.0, 330.0)):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, help="Adam steps before sampling")
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--predict-draws", type=int, default=200)
    parser.add_argument("--last-layer", action="store_true", help="also sample the hybrid model")
    args = parser.parse_args()
    k_mechanistic, k_hybrid = jr.split(jr.key(0))
    mechanistic_posterior(args, k_mechanistic)
    if args.last_layer:
        last_layer_posterior(args, k_hybrid)


if __name__ == "__main__":
    main()
