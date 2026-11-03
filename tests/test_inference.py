"""The inference seam, checked against answers known in closed form.

The model is linear in its rate, ``y(t) = k t``, so with a Gaussian prior and
Gaussian noise the posterior is known exactly. One session fixture runs NUTS
through BlackJAX, as a user would, and the short tests read it.
"""

from __future__ import annotations

from typing import Any

import blackjax
import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import numpy as np
import pytest
from _models import identity
from blackjax.diagnostics import effective_sample_size
from jax.flatten_util import ravel_pytree
from scipy import stats as scipy_stats

import jaxhybridmodels as hm
from jaxhybridmodels.inference import flat_prior, make_problem

TRUE_RATE = 0.7
SIGMA = 0.2
PRIOR_MEAN, PRIOR_SD = 0.0, 1.0
SOLVER = hm.SolverConfig(rtol=1e-8, atol=1e-10, dt0=0.05, max_steps=512)
# Two grid lengths, so the data falls into two buckets.
GRIDS = {"short": (0.5, 1.0, 1.5), "mid": (0.5, 1.0, 2.0), "long": (0.25, 0.75, 1.25, 1.75, 2.5)}


def field(t: Any, y: Any, predictors: Any, covariates: Any) -> Any:
    return predictors["k"].value * jnp.ones_like(y)


def linear_model(k: Any = None, solver: hm.SolverConfig = SOLVER, **extra: Any) -> hm.HybridModel:
    return hm.HybridModel.from_vector_field(
        vector_field=field,
        predictors={"k": hm.Parameter(0.5) if k is None else k, **extra},
        state_to_output=identity,
        output_channel_names=("y",),
        solver=solver,
    )


def measurements(
    rate: Any = TRUE_RATE, key: Any = None
) -> tuple[list[hm.Experiment], np.ndarray, np.ndarray]:
    """Noisy data, returning the times and values for the oracles."""
    experiments, all_ts, all_ys = [], [], []
    key = jr.key(11) if key is None else key
    for name, grid in GRIDS.items():
        key, sub = jr.split(key)
        ts = jnp.asarray(grid)
        ys = rate * ts + SIGMA * jr.normal(sub, ts.shape, dtype=ts.dtype)
        experiments.append(
            hm.make_experiment(
                channels={"y": hm.ChannelObs(ts=ts, values=ys, variance=SIGMA**2)},
                y0=[0.0],
                t0=0.0,
                exp_id=name,
            )
        )
        all_ts.append(np.asarray(ts))
        all_ys.append(np.asarray(ys))
    return experiments, np.concatenate(all_ts), np.concatenate(all_ys)


def conjugate(ts: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """The posterior mean and sd of ``k`` for the linear-Gaussian model."""
    precision = 1 / PRIOR_SD**2 + (ts**2).sum() / SIGMA**2
    mean = (PRIOR_MEAN / PRIOR_SD**2 + (ts * ys).sum() / SIGMA**2) / precision
    return float(mean), float(precision**-0.5)


def least_squares(ts: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """The maximum likelihood of ``k`` for the linear-Gaussian model, and its standard error."""
    return float((ts * ys).sum() / (ts**2).sum()), SIGMA / float(np.sqrt((ts**2).sum()))


def normal_prior(predictors: Any, noise: Any) -> Any:
    return jnp.sum(jstats.norm.logpdf(predictors["k"].value, PRIOR_MEAN, PRIOR_SD))


def run_nuts(problem: Any, key: Any, warmup: int = 300, draws: int = 1500) -> tuple[Any, Any]:
    """The recipe from the docs: BlackJAX window adaptation, then a scan."""
    adapt_key, sample_key = jr.split(key)
    adaptation = blackjax.window_adaptation(blackjax.nuts, problem.log_density)
    (state, params), _ = adaptation.run(adapt_key, problem.position, num_steps=warmup)  # ty: ignore[unknown-argument]
    kernel = blackjax.nuts(problem.log_density, **params)

    def step(state: Any, step_key: Any) -> tuple[Any, Any]:
        state, info = kernel.step(step_key, state)
        return state, (state.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(step, state, jr.split(sample_key, draws))
    return positions, divergent


def assert_matches_conjugate(draws: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> None:
    mean, sd = conjugate(ts, ys)
    ess = float(effective_sample_size(jnp.asarray(draws)[None]))
    assert draws.mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(ess))
    assert draws.std(ddof=1) == pytest.approx(sd, rel=0.1)


@pytest.fixture(scope="module")
def x64():
    with jax.enable_x64():
        yield


@pytest.fixture(scope="module")
def linear(x64):
    experiments, ts, ys = measurements()
    problem = make_problem(linear_model(), experiments, log_prior=normal_prior, noise="data")
    positions, divergent = run_nuts(problem, jr.key(0))
    draws = np.asarray(problem.physical(positions)["k"]).ravel()
    return problem, experiments, ts, ys, draws, np.asarray(divergent)


def test_density_is_the_written_out_log_posterior(linear):
    problem, _, ts, ys, _, _ = linear

    def by_hand(k: float) -> float:
        normaliser = -0.5 * ts.size * np.log(2 * np.pi * SIGMA**2)
        likelihood = normaliser - 0.5 * ((ys - k * ts) ** 2).sum() / SIGMA**2
        prior = float(jstats.norm.logpdf(k, PRIOR_MEAN, PRIOR_SD))
        return likelihood + prior

    for k in (0.3, 0.7, 1.4):
        at_k = make_problem(
            linear_model(hm.Parameter(k)), problem.dataset, log_prior=normal_prior, noise="data"
        )
        assert float(at_k.log_density(at_k.position)) == pytest.approx(by_hand(k), rel=1e-8)


def test_nuts_recovers_the_conjugate_posterior(linear):
    _, _, ts, ys, draws, divergent = linear
    assert not divergent.any()
    assert_matches_conjugate(draws, ts, ys)


def test_density_does_not_depend_on_the_bucketing(linear):
    problem, experiments, *_ = linear
    merged = hm.make_dataset(experiments, output_channel_names=("y",), max_buckets=1)
    assert len(merged.bucket_payloads) < len(problem.dataset.bucket_payloads)
    one = make_problem(linear_model(), merged, log_prior=normal_prior, noise="data")
    assert float(one.log_density(one.position)) == pytest.approx(
        float(problem.log_density(problem.position)), rel=1e-7
    )


def test_residuals_whiten_and_keep_a_fixed_shape(linear):
    problem, _, ts, ys, _, _ = linear
    k = float(problem.physical(problem.position)["k"].squeeze())
    residuals = np.asarray(problem.residuals(problem.position))
    measured = residuals[residuals != 0.0]
    assert sorted(measured) == pytest.approx(sorted((k * ts - ys) / SIGMA), rel=1e-6)
    gradient = jax.jacfwd(problem.residuals)(problem.position)
    assert jax.tree_util.tree_leaves(gradient)[0].shape[0] == residuals.size


@pytest.mark.parametrize(
    "parameter",
    [
        pytest.param(lambda: hm.Parameter(0.5, bounds=(0.0, 2.0)), id="sigmoid"),
        pytest.param(lambda: hm.Parameter(0.5, bounds=(0.0, None)), id="one-sided"),
        pytest.param(lambda: hm.Parameter(1.0, bounds=(1e-2, 1e2), warp="log10"), id="log10"),
    ],
)
def test_the_jacobian_makes_a_physical_prior_integrate_to_one(x64, parameter):
    """With no information from data, the density over the latent is the prior's."""
    start = parameter()
    low, high = start.scaler.bounds[0]

    def prior(predictors: Any, noise: Any) -> Any:
        # Uniform on the finite box, or exponential on the half line.
        k = predictors["k"].value
        if np.isfinite(high):
            return -jnp.log(jnp.asarray(high - low))
        return -k.sum()

    # One observation with a huge variance, so the likelihood is flat in the rate.
    flat = hm.make_experiment(
        channels={
            "y": hm.ChannelObs(ts=jnp.asarray([1.0]), values=jnp.asarray([0.0]), variance=1e12)
        },
        y0=[0.0],
        t0=0.0,
    )
    problem = make_problem(linear_model(start), [flat], log_prior=prior, noise="data")
    likelihood = float(problem.log_likelihood(problem.position))
    leaf, treedef = jax.tree_util.tree_flatten(problem.position)
    zs = jnp.linspace(-40.0, 40.0, 4001)

    def density(z: Any) -> Any:
        return problem.log_density(treedef.unflatten([z.reshape(leaf[0].shape)]))

    values = np.exp(np.asarray(jax.vmap(density)(zs)) - likelihood)
    assert np.trapezoid(values, np.asarray(zs)) == pytest.approx(1.0, abs=1e-3)


def test_inferred_noise_matches_quadrature(x64):
    """NUTS on the rate and the noise, against a 2-D grid of the same density."""
    experiments, _, _ = measurements()

    def prior(predictors: Any, noise: Any) -> Any:
        return normal_prior(predictors, noise) + jstats.norm.logpdf(jnp.log(noise["y"]), -1.5, 1.0)

    problem = make_problem(
        linear_model(),
        experiments,
        log_prior=prior,
        noise={"y": hm.Parameter(0.3, bounds=(1e-3, 10.0), warp="log10")},
    )
    assert set(problem.physical(problem.position)) == {"k", "noise.y"}
    positions, divergent = run_nuts(problem, jr.key(4))
    assert not np.asarray(divergent).any()
    named = problem.physical(positions)
    draws = np.asarray(named["noise.y"]).ravel()

    leaves, treedef = jax.tree_util.tree_flatten(problem.position)
    centres = [float(jnp.mean(leaf)) for leaf in jax.tree_util.tree_leaves(positions)]
    spreads = [float(jnp.std(leaf)) for leaf in jax.tree_util.tree_leaves(positions)]
    axes = [jnp.linspace(c - 8 * w, c + 8 * w, 201) for c, w in zip(centres, spreads, strict=True)]
    first, second = jnp.meshgrid(*axes, indexing="ij")

    def at(a: Any, b: Any) -> Any:
        point = treedef.unflatten(
            [v.reshape(leaf.shape) for v, leaf in zip((a, b), leaves, strict=True)]
        )
        return problem.log_density(point), problem.physical(point)["noise.y"].squeeze()

    logs, sigmas = jax.vmap(at)(first.ravel(), second.ravel())
    weights = np.exp(np.asarray(logs) - float(jnp.max(logs)))
    weights /= weights.sum()
    mean = float((weights * np.asarray(sigmas)).sum())
    sd = float(np.sqrt((weights * (np.asarray(sigmas) - mean) ** 2).sum()))
    ess = float(effective_sample_size(jnp.asarray(draws)[None]))
    assert draws.mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(ess))
    assert draws.std(ddof=1) == pytest.approx(sd, rel=0.15)


def test_a_custom_likelihood_can_restate_the_gaussian(linear):
    problem, *_ = linear

    def gaussian(pred_obs: Any, bp: Any, noise: Any) -> Any:
        cells = jstats.norm.logpdf(
            jnp.where(bp.mask, bp.observed, 0.0),
            pred_obs,
            jnp.sqrt(jnp.where(bp.mask, bp.variance, 1.0)),
        )
        return jnp.sum(jnp.where(bp.mask, cells, 0.0))

    custom = make_problem(
        linear_model(),
        problem.dataset,
        log_prior=normal_prior,
        noise="data",
        log_likelihood=gaussian,
    )
    assert float(custom.log_density(custom.position)) == pytest.approx(
        float(problem.log_density(problem.position)), rel=1e-9
    )


def test_the_documented_ar1_likelihood_is_the_multivariate_normal(x64):
    """The recipe in the uncertainty guide, against its covariance matrix."""

    def ar1(pred_obs: Any, bp: Any, noise: Any) -> Any:
        sigma, rho = noise["sigma"], noise["rho"]
        residual = bp.observed[..., 0] - pred_obs[..., 0]
        innovation = residual[:, 1:] - rho * residual[:, :-1]
        variance = sigma**2 * (1 - rho**2)
        first = jstats.norm.logpdf(residual[:, 0], 0.0, sigma)
        rest = jstats.norm.logpdf(innovation, 0.0, jnp.sqrt(variance))
        return first.sum() + rest.sum()

    ts = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    ys = 0.5 * ts + jnp.asarray([0.1, 0.05, -0.1, 0.2])
    experiment = hm.make_experiment(
        channels={"y": hm.ChannelObs(ts=ts, values=ys, variance=1.0)}, y0=[0.5 * 1.0], t0=1.0
    )
    problem = make_problem(
        linear_model(hm.Parameter(0.5)),
        [experiment],
        log_prior=lambda p, n: jnp.asarray(0.0),
        noise={
            "sigma": hm.Parameter(0.3, bounds=(0.01, 2.0)),
            "rho": hm.Parameter(0.6, bounds=(-0.99, 0.99)),
        },
        log_likelihood=ar1,
    )
    lags = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    covariance = 0.3**2 * 0.6**lags
    residual = np.asarray(ys - 0.5 * ts)
    expected = np.asarray(jstats.multivariate_normal.logpdf(residual, np.zeros(4), covariance))
    assert float(problem.log_likelihood(problem.position)) == pytest.approx(float(expected))


def test_a_failed_solve_is_a_rejection_and_the_chain_survives(x64):
    """A rate above the threshold blows up the solve, which must not end the chain."""
    limit = 1.2

    def stiff_field(t: Any, y: Any, predictors: Any, covariates: Any) -> Any:
        k = predictors["k"].value
        return jnp.where(k > limit, jnp.nan, k) * jnp.ones_like(y)

    experiments, *_ = measurements()
    model = hm.HybridModel.from_vector_field(
        vector_field=stiff_field,
        predictors={"k": hm.Parameter(0.7)},
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    problem = make_problem(
        model, experiments, log_prior=lambda p, n: jnp.asarray(0.0), noise="data"
    )
    beyond = jax.tree_util.tree_map(lambda leaf: leaf + 5.0, problem.position)
    value, gradient = jax.value_and_grad(problem.log_density)(beyond)
    assert value == -jnp.inf
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(gradient))

    positions, _ = run_nuts(problem, jr.key(3), warmup=100, draws=200)
    draws = np.asarray(problem.physical(positions)["k"]).ravel()
    assert np.isfinite(draws).all() and draws.max() <= limit


def test_mixed_models_state_what_they_condition_on(x64):
    network = hm.BoundedPredictor(
        input_keys=("T",),
        input_bounds=(0.0, 1.0),
        inner=hm.MLPPredictor(in_size=1, out_size=1, width_size=4, depth=1, key=jr.key(0)),
        out_scaler=hm.BoundScaler(bounds=(0.0, 1.0)),
    )
    experiments, *_ = measurements()
    with pytest.warns(UserWarning, match="conditional on"):
        problem = make_problem(
            linear_model(net=network), experiments, log_prior=normal_prior, noise="data"
        )
    assert problem.conditional_on and all(p.startswith("net.") for p in problem.conditional_on)
    assert set(problem.physical(problem.position)) == {"k"}
    # Leading axes stay on the values, as a [chain, draw] sampler output has them.
    batched = jax.tree.map(lambda leaf: jnp.stack([leaf] * 3)[None], problem.position)
    assert problem.physical(batched)["k"].shape[:2] == (1, 3)

    last_layer = make_problem(
        linear_model(net=network),
        experiments,
        log_prior=normal_prior,
        noise="data",
        estimate=["k", "net.inner.mlp.layers.1"],
    )
    named = set(last_layer.physical(last_layer.position))
    assert named == {"k", "net.inner.mlp.layers.1.weight", "net.inner.mlp.layers.1.bias"}
    moved = last_layer.to_model(last_layer.position)
    assert isinstance(moved, hm.HybridModel) and moved.solver.throw

    # A fixed Parameter is named as physical() names it, without ".latent".
    with pytest.warns(UserWarning, match="conditional on"):
        layer_only = make_problem(
            linear_model(net=network),
            experiments,
            log_prior=normal_prior,
            noise="data",
            estimate="net.inner.mlp.layers.1",
        )
    assert "k" in layer_only.conditional_on
    assert not any(name.endswith(".latent") for name in layer_only.conditional_on)


def test_to_model_predicts_at_a_draw(linear):
    problem, experiments, *_ = linear
    model = problem.to_model(problem.position)
    assert model.solver.throw, "a model handed back keeps the raising solver"
    assert set(model.predict(experiments)) == set(GRIDS)


def test_forward_and_reverse_give_one_density_and_gradient(linear):
    problem, *_ = linear
    forward = make_problem(
        linear_model(), problem.dataset, log_prior=normal_prior, noise="data", gradient="forward"
    )
    reverse = make_problem(
        linear_model(), problem.dataset, log_prior=normal_prior, noise="data", gradient="reverse"
    )
    assert (forward.gradient, reverse.gradient) == ("forward", "reverse")
    value_f, grad_f = jax.value_and_grad(forward.log_density)(forward.position)
    value_r, grad_r = jax.value_and_grad(reverse.log_density)(reverse.position)
    assert float(value_f) == pytest.approx(float(value_r), rel=1e-9)
    for a, b in zip(
        jax.tree_util.tree_leaves(grad_f), jax.tree_util.tree_leaves(grad_r), strict=True
    ):
        assert np.allclose(a, b, rtol=1e-5)


def test_preconditioning_keeps_the_posterior_and_flattens_the_geometry(linear):
    problem, _, ts, ys, *_ = linear
    whitened = problem.preconditioned()
    # z = 0 is the fit, and the density there is the original's, since the map is linear.
    assert float(whitened.log_density(whitened.position)) == pytest.approx(
        float(problem.log_density(problem.position)), rel=1e-9
    )
    start = float(whitened.physical(whitened.position)["k"].squeeze())
    assert start == pytest.approx(float(problem.physical(problem.position)["k"].squeeze()))

    positions, divergent = run_nuts(whitened, jr.key(7))
    draws = np.asarray(jax.vmap(whitened.physical)(positions)["k"]).ravel()
    assert not np.asarray(divergent).any()
    assert_matches_conjugate(draws, ts, ys)
    # A unit-scale target: the sampler's coordinates have about unit spread.
    assert float(jnp.std(positions)) == pytest.approx(1.0, rel=0.2)
    result = whitened.predict(positions, key=jr.key(8), n_draws=20)
    assert len(result.members) == 20


def test_preconditioning_rejects_what_it_cannot_use(linear):
    problem, *_ = linear

    def custom(pred_obs: Any, bp: Any, noise: Any) -> Any:
        return jnp.asarray(0.0)

    with pytest.raises(ValueError, match="already preconditioned"):
        problem.preconditioned().preconditioned()
    with pytest.raises(ValueError, match="must have shape"):
        problem.preconditioned(precision=jnp.eye(3))
    other = make_problem(
        linear_model(), problem.dataset, log_prior=normal_prior, noise="data", log_likelihood=custom
    )
    with pytest.raises(ValueError, match="pass precision="):
        other.preconditioned()
    assert other.preconditioned(precision=jnp.eye(1)).position.shape == (1,)


def test_a_simulator_without_forward_mode_still_preconditions(x64):
    """A hand-written solve keeps diffrax's reverse-only adjoint, so auto falls back."""

    def simulate(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
        rate = predictors["k"].value

        def rhs(t: Any, y: Any, args: Any) -> Any:
            return rate * jnp.ones_like(y)

        return diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=solver.stepsize_controller(),
            throw=solver.throw,
        ).ys

    experiments, ts, ys = measurements()
    model = hm.HybridModel(
        predictors={"k": hm.Parameter(0.5)},
        simulate_fn=simulate,
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    problem = make_problem(model, experiments, log_prior=normal_prior, noise="data")
    assert problem.gradient == "reverse"
    whitened = problem.preconditioned()
    # In the whitened coordinate the likelihood's curvature is about one.
    assert whitened._whitening is not None
    assert float(jnp.abs(whitened._whitening[1]).squeeze()) == pytest.approx(
        conjugate(ts, ys)[1], rel=0.01
    )
    positions, _ = run_nuts(whitened, jr.key(9), warmup=100, draws=300)
    assert_matches_conjugate(
        np.asarray(jax.vmap(whitened.physical)(positions)["k"]).ravel(), ts, ys
    )


def test_a_bad_prior_raises_when_the_problem_is_built():
    experiments, *_ = measurements()

    def bad(predictors: Any, noise: Any) -> Any:
        return predictors["missing"].value

    with pytest.raises(KeyError, match="missing"):
        make_problem(linear_model(), experiments, log_prior=bad, noise="data")


def test_a_nan_gradient_at_a_finite_density_stays_visible(x64):
    """Zeroing it would silently run the sampler with the wrong dynamics.

    Reverse mode multiplies the dead branch's NaN slope by zero, which is NaN, so the
    density is finite and its gradient is not. Forward mode never visits that branch.
    """
    experiments, *_ = measurements()

    def nan_slope(predictors: Any, noise: Any) -> Any:
        k = jnp.squeeze(predictors["k"].value)
        return jnp.where(k > 1e6, jnp.sqrt(-k), 0.0)

    problem = make_problem(
        linear_model(), experiments, log_prior=nan_slope, noise="data", gradient="reverse"
    )
    value, gradient = jax.value_and_grad(problem.log_density)(problem.position)
    assert np.isfinite(float(value))
    assert np.isnan(np.asarray(jax.tree_util.tree_leaves(gradient)[0])).any()


def test_a_bad_input_to_a_warped_scaler_rejects_the_proposal(x64):
    """A proposal that drives a log-warped input non-positive must not end the chain."""

    def field_with_network(t: Any, y: Any, predictors: Any, covariates: Any) -> Any:
        unused = hm.evaluate_predictor(predictors["net"], {"y": y[0]})
        return (predictors["k"].value - 0.5) * jnp.ones_like(y) + 0.0 * unused

    network = hm.BoundedPredictor(
        input_keys=("y",),
        in_scaler=hm.BoundScaler(bounds=(1e-3, 10.0), warp="log"),
        inner=hm.MLPPredictor(in_size=1, out_size=1, width_size=4, depth=1, key=jr.key(0)),
        out_scaler=hm.BoundScaler(bounds=(0.0, 1.0)),
    )
    ts = jnp.asarray([0.5, 1.0, 2.0])
    experiment = hm.make_experiment(
        channels={"y": hm.ChannelObs(ts=ts, values=jnp.ones_like(ts), variance=0.01)},
        y0=[1.0],
        t0=0.0,
    )
    model = hm.HybridModel.from_vector_field(
        vector_field=field_with_network,
        predictors={"k": hm.Parameter(0.5), "net": network},
        state_to_output=identity,
        output_channel_names=("y",),
        solver=hm.SolverConfig(dt0=0.05, max_steps=256),
    )
    problem = make_problem(
        model, [experiment], log_prior=normal_prior, noise="data", gradient="reverse"
    )
    beyond = jax.tree_util.tree_map(lambda leaf: leaf - 3.0, problem.position)
    value, gradient = jax.value_and_grad(problem.log_density)(beyond)
    assert float(value) == -np.inf
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(gradient))
    # The likelihood that optimize and profile search is guarded the same way.
    flat, unravel = ravel_pytree(beyond)
    assert float(problem._scalar("likelihood", unravel, forward=False)(flat)) == -np.inf
    # Outside the density the guard is still on.
    broken = problem.to_model(beyond)
    with pytest.raises(Exception, match="strictly positive"):
        broken.predict([experiment])


def test_predict_drops_a_draw_that_fails_to_solve(x64):
    limit = 1.2

    def failing_field(t: Any, y: Any, predictors: Any, covariates: Any) -> Any:
        k = predictors["k"].value
        return jnp.where(k > limit, jnp.nan, k) * jnp.ones_like(y)

    experiments, *_ = measurements()
    model = hm.HybridModel.from_vector_field(
        vector_field=failing_field,
        predictors={"k": hm.Parameter(0.7)},
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    problem = make_problem(model, experiments, log_prior=normal_prior, noise="data")
    good = jax.tree_util.tree_map(lambda leaf: leaf[None], problem.position)
    bad = jax.tree_util.tree_map(lambda leaf: leaf + 5.0, good)
    mixed = jax.tree_util.tree_map(lambda *leaf: jnp.concatenate(leaf), good, bad, good)
    with pytest.warns(RuntimeWarning, match="dropped 1 of 3"):
        result = problem.predict(mixed, key=jr.key(0))
    assert len(result.members) == 2 and np.all(np.isfinite(result["long"].outputs["y"]))
    with pytest.raises(ValueError, match="every draw failed"):
        problem.predict(bad, key=jr.key(0))


def test_preconditioning_scales_inferred_noise_from_a_poor_start(x64):
    """A noise guess 5 times too large must not leave its direction far too wide."""
    experiments, *_ = measurements()

    def prior(predictors: Any, noise: Any) -> Any:
        return normal_prior(predictors, noise) + jstats.norm.logpdf(jnp.log(noise["y"]), -1.5, 1.0)

    problem = make_problem(
        linear_model(hm.Parameter(TRUE_RATE)),
        experiments,
        log_prior=prior,
        noise={"y": hm.Parameter(SIGMA * 5, bounds=(1e-3, 10.0), warp="log10")},
    ).preconditioned()
    positions, divergent = run_nuts(problem, jr.key(10), warmup=200, draws=400)
    assert not np.asarray(divergent).any()
    # The position flattens "noise" before "predictors". The rate's own spread is about
    # 1/5 here: its curvature scales with 1/sigma^2, so a poor guess shrinks it, and the
    # sampler's mass adaptation rescales that. Only the noise needs the Fisher term.
    noise_spread = float(np.std(np.asarray(positions)[:, 0]))
    assert noise_spread == pytest.approx(1.0, rel=0.3)


def test_explicit_forward_mode_fails_when_the_problem_is_built(x64):
    def simulate(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
        rate = predictors["k"].value

        def rhs(t: Any, y: Any, args: Any) -> Any:
            return rate * jnp.ones_like(y)

        return diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=solver.stepsize_controller(),
        ).ys

    experiments, *_ = measurements()
    model = hm.HybridModel(
        predictors={"k": hm.Parameter(0.5)},
        simulate_fn=simulate,
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    with pytest.raises(TypeError):
        make_problem(model, experiments, log_prior=normal_prior, noise="data", gradient="forward")


@jax.custom_vjp
def _reverse_only(x: Any) -> Any:
    return x


_reverse_only.defvjp(lambda x: (x, None), lambda _, g: (g,))


def reverse_only_prior(predictors: Any, noise: Any) -> Any:
    """A prior with no forward-mode rule, which forward mode must refuse up front."""
    return _reverse_only(normal_prior(predictors, noise))


def _bad_problem(**overrides: Any) -> Any:
    arguments: dict[str, Any] = dict(log_prior=normal_prior, noise="data")
    arguments.update(overrides)
    experiments, *_ = measurements()
    model = arguments.pop("model", linear_model())
    return make_problem(model, experiments, **arguments)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        pytest.param({"noise": "unit"}, ValueError, "noise must be", id="noise-name"),
        pytest.param({"noise": 0.1}, TypeError, "noise must be", id="noise-type"),
        pytest.param({"noise": {"y": 0.1}}, ValueError, "scalar hm.Parameter", id="noise-float"),
        pytest.param(
            {"noise": {"z": hm.Parameter(0.1, bounds=(0.0, 1.0))}},
            ValueError,
            "one per channel",
            id="noise-channel",
        ),
        pytest.param({"gradient": "fast"}, ValueError, "gradient must", id="gradient"),
        pytest.param(
            {"gradient": "forward", "log_prior": reverse_only_prior},
            TypeError,
            "custom_vjp",
            id="forward-prior",
        ),
        pytest.param({"noise": {"y": hm.Parameter(0.1)}}, ValueError, "positive", id="noise-sign"),
        pytest.param({"estimate": "nope"}, ValueError, "nope", id="estimate-path"),
        pytest.param({"log_prior": flat_prior}, ValueError, "improper", id="flat-prior-open-bound"),
        pytest.param(
            {"model": linear_model(hm.Constant(0.5))}, ValueError, "no Parameter", id="nothing"
        ),
    ],
)
def test_make_problem_rejects_bad_input(overrides, error, match):
    with pytest.raises(error, match=match):
        _bad_problem(**overrides)


def test_flat_prior_is_accepted_on_a_bounded_box_and_unset_variance_warns(x64):
    experiments = [
        hm.make_experiment(
            channels={"y": hm.ChannelObs(ts=jnp.asarray([1.0]), values=jnp.asarray([0.5]))},
            y0=[0.0],
            t0=0.0,
        )
    ]
    with pytest.warns(UserWarning, match="default for"):
        problem = make_problem(
            linear_model(hm.Parameter(0.5, bounds=(0.0, 2.0))),
            experiments,
            log_prior=flat_prior,
            noise="data",
        )
    assert jnp.isfinite(problem.log_density(problem.position))


def test_a_solver_records_throw_only_when_it_is_off():
    assert "throw" not in hm.SolverConfig().to_dict()
    off = hm.SolverConfig(solver=diffrax.Tsit5(), throw=False)
    assert hm.SolverConfig.from_dict(off.to_dict()).throw is False


def test_posterior_predictive_matches_the_closed_form(linear):
    """``y(t) = k t``, so the band at time ``t`` is the posterior of ``k`` scaled by ``t``."""
    problem, _, ts, ys, _, _ = linear
    mean, sd = conjugate(ts, ys)
    positions, *_ = run_nuts(problem, jr.key(5), warmup=200, draws=600)
    result = problem.predict(positions, key=jr.key(6), n_draws=400)
    assert len(result.members) == 400
    long = result["long"]
    last = -1
    t_last = float(long.ts[last])
    assert float(long.outputs["y"][last]) == pytest.approx(mean * t_last, abs=4 * sd * t_last / 15)
    assert float(long.std["y"][last]) == pytest.approx(sd * t_last, rel=0.2)
    low, high = (float(long.quantiles["y"][q][last]) for q in (0.025, 0.975))
    assert high - low == pytest.approx(2 * 1.96 * sd * t_last, rel=0.2)
    frame = result.to_dataframe()
    assert {"q0.025", "q0.975", "std"} <= set(frame.columns)


def test_observation_noise_widens_the_interval_by_the_noise(x64):
    experiments, ts, ys = measurements()

    problem = make_problem(
        linear_model(hm.Parameter(0.7)),
        experiments,
        log_prior=normal_prior,
        noise={"y": hm.Parameter(0.2, bounds=(0.01, 1.0))},
    )
    positions = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (500, *leaf.shape)), problem.position
    )
    # Every draw is the same point, so the model's own band is zero and only noise is left.
    plain = problem.predict(positions, key=jr.key(0), n_draws=500)
    assert float(jnp.max(plain["long"].std["y"])) < 1e-6
    noisy = problem.predict(positions, key=jr.key(0), n_draws=500, observation_noise=True)
    assert float(jnp.mean(noisy["long"].std["y"])) == pytest.approx(0.2, rel=0.15)
    assert np.allclose(noisy["long"].outputs["y"], plain["long"].outputs["y"])


def test_arviz_export_names_physical_values_and_records_the_conditioning(linear):
    pytest.importorskip("arviz_base")
    problem, *_ = linear
    positions = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 5, *leaf.shape)), problem.position
    )
    tree = problem.to_arviz(positions, sample_stats={"diverging": np.zeros((2, 5), bool)})
    assert tree["posterior"]["k"].sizes["chain"] == 2
    assert tree["posterior"]["k"].sizes["draw"] == 5
    assert "diverging" in tree["sample_stats"]
    assert tree["posterior"].attrs["conditional_on"] == ""
    one_axis = jax.tree_util.tree_map(lambda leaf: leaf[None], problem.position)
    with pytest.raises(ValueError, match="chain"):
        problem.to_arviz(one_axis)


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        pytest.param(lambda p, x: p.predict(x, key=0), TypeError, "key", id="key"),
        pytest.param(
            lambda p, x: p.predict(x, key=jr.key(0), quantiles=(1.5,)),
            ValueError,
            "strictly",
            id="quantile",
        ),
        pytest.param(
            lambda p, x: p.predict(x, key=jr.key(0), n_draws=0),
            ValueError,
            "n_draws",
            id="draws",
        ),
        pytest.param(
            lambda p, x: p.predict(p.position, key=jr.key(0)),
            ValueError,
            "leading sample axes",
            id="no-sample-axis",
        ),
        pytest.param(
            lambda p, x: p.predict(x, key=jr.key(0), observation_noise=True),
            ValueError,
            "inferred",
            id="noise-from-data",
        ),
    ],
)
def test_prediction_rejects_bad_input(linear, call, error, match):
    problem, *_ = linear
    positions = jax.tree_util.tree_map(lambda leaf: leaf[None], problem.position)
    with pytest.raises(error, match=match):
        call(problem, positions)


# ---------------------------------------------------------------------------
# Frequentist: the optimum, Fisher information and the Laplace approximation
# ---------------------------------------------------------------------------


def test_optimum_fisher_and_laplace_match_the_closed_form(linear):
    """For ``y = k t`` with known noise every answer is exact: the MLE is
    ``sum(t y) / sum(t^2)``, its standard error ``sigma / sqrt(sum(t^2))``, and the
    Laplace approximation is the conjugate posterior itself."""
    _, experiments, ts, ys, _, _ = linear
    mle, error = least_squares(ts, ys)

    frequentist = make_problem(linear_model(), experiments, log_prior=None, noise="data")
    with pytest.warns(UserWarning, match="not at the likelihood's maximum"):
        frequentist.fisher()
    optimum = frequentist.optimize()
    assert optimum.converged and set(optimum.gradient) == {"k"}
    assert optimum.gradient_norm < 1e-6
    fisher = optimum.problem.fisher()
    assert float(fisher.estimate[0]) == pytest.approx(mle, rel=1e-6)
    assert fisher.standard_errors()["k"] == pytest.approx(error, rel=1e-6)
    observed = optimum.problem.fisher(curvature="exact")
    assert observed.standard_errors()["k"] == pytest.approx(error, rel=1e-6)
    low, high = fisher.intervals(confidence=0.95)["k"]
    assert (low, high) == pytest.approx((mle - 1.959964 * error, mle + 1.959964 * error))
    draws = fisher.sample(key=jr.key(4), n_draws=4000)
    estimates = np.asarray(optimum.problem.physical(draws)["k"])
    assert estimates.mean() == pytest.approx(mle, abs=4 * error / np.sqrt(4000))
    assert estimates.std() == pytest.approx(error, rel=0.05)
    assert "Condition number" in fisher.summary()
    # The delta method is exact for a linear function, and a one-sided bound uses
    # the one-sided quantile.
    doubled = fisher.derived(lambda values: 2.0 * values["k"])
    assert float(doubled.estimate) == pytest.approx(2 * mle, rel=1e-6)
    assert float(doubled.standard_error) == pytest.approx(2 * error, rel=1e-6)
    low, high = fisher.derived(lambda values: values["k"]).interval(confidence=0.95, side="lower")
    assert float(low) == pytest.approx(mle - 1.644854 * error, rel=1e-6) and high == np.inf

    bayesian = make_problem(linear_model(), experiments, log_prior=normal_prior, noise="data")
    laplace = bayesian.optimize(target="posterior").problem.laplace()
    mean, sd = conjugate(ts, ys)
    assert float(laplace.estimate[0]) == pytest.approx(mean, rel=1e-6)
    assert laplace.standard_errors()["k"] == pytest.approx(sd, rel=1e-6)
    draws = laplace.sample(key=jr.key(5), n_draws=4000)
    values = np.asarray(bayesian.physical(draws)["k"])
    assert values.shape == (4000,)
    assert values.mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(4000))
    assert values.std() == pytest.approx(sd, rel=0.05)
    band = bayesian.predict(draws, key=jr.key(6), n_draws=50)
    assert set(band) == {e.exp_id for e in experiments}
    whitened = bayesian.optimize(target="posterior").problem.preconditioned()
    redrawn = whitened.laplace().sample(key=jr.key(5), n_draws=4000)
    again = np.asarray(whitened.physical(redrawn)["k"])
    assert again == pytest.approx(values, rel=1e-8)


@pytest.mark.parametrize(
    ("rate", "bounds", "warp", "profiled"),
    [
        pytest.param(50.0, (0.0, None), "linear", True, id="far-above-a-lower-bound"),
        pytest.param(-50.0, (None, 0.0), "linear", True, id="far-below-an-upper-bound"),
        pytest.param(-5.0, (None, 0.0), "linear", True, id="below-an-upper-bound"),
        pytest.param(1e13, (0.0, None), "log", False, id="log-warp-far-out"),
    ],
)
def test_a_value_far_from_its_only_bound_is_free(x64, rate, bounds, warp, profiled):
    """The far side of a one-sided bound is not a bound.

    The standard error and the profile interval are the closed-form ones, and a
    value with only an upper bound reports its interval low end first.
    """
    experiments, ts, ys = measurements(rate)
    mle, error = least_squares(ts, ys)
    model = linear_model(hm.Parameter(rate, bounds=bounds, warp=warp))
    problem = make_problem(model, experiments, log_prior=None, noise="data").optimize().problem
    fisher = problem.fisher()
    assert fisher.standard_errors()["k"] == pytest.approx(error, rel=1e-5)
    if not profiled:
        return
    profile = problem.profile("k")
    assert profile.stop_reasons == ("threshold", "threshold")
    assert profile.bounds == pytest.approx(
        (mle - 1.959964 * error, mle + 1.959964 * error), abs=0.01 * error
    )
    assert np.all(np.diff(profile.grid) > 0)


@pytest.mark.parametrize("warp", ["linear", "log"])
def test_a_value_pinned_at_its_only_bound_profiles_away_from_it(x64, warp):
    """The far side closes where the closed form says, however huge the latent spread."""
    experiments, ts, ys = measurements()
    mle, error = least_squares(ts, ys)
    edge = mle + 2.0 * error
    model = linear_model(hm.Parameter(edge + 0.5, bounds=(edge, None), warp=warp))
    problem = make_problem(model, experiments, log_prior=None, noise="data").optimize().problem
    profile = problem.profile("k")
    assert profile.stop_reasons == ("bound", "threshold")
    high = mle + np.sqrt(3.841459 * error**2 + (edge - mle) ** 2)
    assert profile.bounds[1] == pytest.approx(high, abs=0.01 * error)


def test_the_condition_number_ignores_how_a_value_is_bounded(x64):
    """Rescaling a latent through its bounds leaves the identifiability readout alone."""
    experiments, *_ = measurements()
    numbers = []
    for bounds in (None, (0.0, 1e4)):
        model = linear_model(hm.Parameter(TRUE_RATE, bounds=bounds))
        noise = {"y": hm.Parameter(SIGMA, bounds=(1e-3, 1.0))}
        problem = make_problem(model, experiments, log_prior=None, noise=noise)
        numbers.append(problem.optimize().problem.fisher().condition_number)
    assert numbers[0] == pytest.approx(numbers[1], rel=1e-4)


def test_a_value_the_data_barely_see_is_not_identified(x64):
    """No correlation, so the scale-free test passes it; its huge spread does not."""

    def faint_field(t: Any, y: Any, predictors: Any, covariates: Any) -> Any:
        # b's effect grows with t, so it is not a multiple of k's: its own
        # information is tiny but nothing is collinear.
        return (predictors["k"].value + 1e-9 * predictors["b"].value * t) * jnp.ones_like(y)

    experiments, *_ = measurements()
    model = hm.HybridModel.from_vector_field(
        vector_field=faint_field,
        predictors={"k": hm.Parameter(0.7), "b": hm.Parameter(0.5)},
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    problem = make_problem(model, experiments, log_prior=None, noise="data").optimize().problem
    fisher = problem.fisher()
    errors = fisher.standard_errors()
    assert np.isfinite(errors["k"]) and errors["b"] == np.inf
    assert {"b": 1.0} in fisher.unidentified()
    assert "Not identified: b" in fisher.summary()
    assert np.isinf(float(fisher.derived(lambda v: v["b"]).standard_error))
    assert np.isfinite(float(fisher.derived(lambda v: v["k"]).standard_error))


def test_a_tiny_measurement_variance_is_not_floored(x64):
    """Data in small units keep their likelihood: every curvature agrees."""
    rate, sigma = 1e-4, 1e-7
    experiments = [
        hm.make_experiment(
            channels={
                "y": hm.ChannelObs(
                    ts=jnp.asarray(grid),
                    values=rate * jnp.asarray(grid),
                    variance=sigma**2,
                )
            },
            y0=[0.0],
            t0=0.0,
            exp_id=name,
        )
        for name, grid in GRIDS.items()
    ]
    ts = np.concatenate([np.asarray(grid) for grid in GRIDS.values()])
    model = linear_model(hm.Parameter(2e-4, bounds=(0.0, 1e-3)))
    problem = make_problem(model, experiments, log_prior=None, noise="data").optimize().problem
    expected = sigma / np.sqrt(np.sum(ts**2))
    for curvature in ("gauss_newton", "exact"):
        error = problem.fisher(curvature=curvature).standard_errors()["k"]
        assert error == pytest.approx(expected, rel=1e-4), curvature


def product_field(t: Any, y: Any, predictors: Any, covariates: Any) -> Any:
    return predictors["a"].value * predictors["b"].value * jnp.ones_like(y)


def product_model(a: hm.Parameter) -> hm.HybridModel:
    """``y = a b t``: the data see only the product, so ``a`` and ``b`` trade off freely."""
    return hm.HybridModel.from_vector_field(
        vector_field=product_field,
        predictors={"a": a, "b": hm.Parameter(0.7)},
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )


def test_fisher_reports_physical_errors_and_what_the_data_cannot_separate(x64):
    experiments, ts, ys = measurements()
    # The latent is a squashed log of k, so the information is computed far from
    # physical units. At the MLE the delta method maps it back exactly.
    warped = linear_model(hm.Parameter(0.7, bounds=(0.01, 10.0), warp="log"))
    problem = make_problem(warped, experiments, log_prior=None, noise="data").optimize().problem
    result = problem.fisher()
    assert result.standard_errors()["k"] == pytest.approx(least_squares(ts, ys)[1], rel=1e-6)
    assert result.standard_errors(latent=True)["k"] != pytest.approx(result.standard_errors()["k"])
    low, high = result.intervals(confidence=0.95)["k"]
    centre = float(result.estimate[0])
    # Built in the latent and mapped back, so it leans up as log k does and stays in bounds.
    assert 0.01 < low < centre < high < 10.0 and high - centre > centre - low
    # An interior optimum a millionth of the way along a huge box is in the squash's
    # tail but not held there by the bound, so it is reported in full.
    for wide in (
        hm.Parameter(0.7, bounds=(0.0, 1e7)),
        hm.Parameter(0.7, bounds=(1e-12, 1e4), warp="log10"),
    ):
        problem = make_problem(linear_model(wide), experiments, log_prior=None, noise="data")
        result = problem.optimize().problem.fisher()
        assert not result.at_bound.any()
        assert result.standard_errors()["k"] == pytest.approx(least_squares(ts, ys)[1], rel=1e-5)

    # With inferred noise the Gauss-Newton information must count the noise once:
    # the MLE's standard error is sigma / sqrt(2 n), as the observed information says.
    noisy = (
        make_problem(
            linear_model(),
            experiments,
            log_prior=None,
            noise={"y": hm.Parameter(0.3, bounds=(1e-3, 10.0), warp="log")},
        )
        .optimize()
        .problem
    )
    sigma = float(noisy.physical(noisy.position)["noise.y"])
    expected = sigma / np.sqrt(2 * ys.size)
    assert noisy.fisher().standard_errors()["noise.y"] == pytest.approx(expected, rel=1e-6)
    assert noisy.fisher(curvature="exact").standard_errors()["noise.y"] == pytest.approx(
        expected, rel=1e-6
    )

    product = product_model(hm.Parameter(1.0))
    problem = make_problem(product, experiments, log_prior=None, noise="data").optimize().problem
    together = problem.fisher()
    (direction,) = together.unidentified()
    assert set(direction) == {"a", "b"}
    assert together.standard_errors() == {"a": np.inf, "b": np.inf}
    # The product is what the data see, so its error is finite: that of the slope.
    product_error = together.derived(lambda values: values["a"] * values["b"]).standard_error
    assert float(product_error) == pytest.approx(least_squares(ts, ys)[1], rel=1e-5)
    assert np.isinf(together.derived(lambda values: values["a"]).standard_error)
    assert together.intervals()["a"] == (-np.inf, np.inf)
    with pytest.raises(ValueError, match="do not constrain"):
        together.sample(key=jr.key(0), n_draws=10)
    assert "Not identified: " in together.summary()


def test_a_profile_closes_where_the_closed_form_says_and_compiles_once(linear):
    """For a linear-Gaussian model the profile is exactly quadratic, so its 95%
    interval is the Wald one, ``mle -/+ 1.96 se``. With a second, separately
    identified value, re-optimising it at each point must not move the interval."""
    _, experiments, ts, ys, _, _ = linear
    mle, error = least_squares(ts, ys)
    traces = []

    def counted(predictors: Any, ts: Any, covariates: Any, y0: Any, solver: Any) -> Any:
        traces.append(1)  # once per trace of whatever calls the model
        term = diffrax.ODETerm(lambda t, y, args: field(t, y, predictors, covariates))
        return solver.diffeqsolve(term, ts, y0)

    model = hm.HybridModel(
        predictors={"k": hm.Parameter(0.5)},
        simulate_fn=counted,
        state_to_output=identity,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    problem = make_problem(model, experiments, log_prior=None, noise="data").optimize().problem
    before = len(traces)
    first = problem.profile("k", max_points=4)
    with_compile = len(traces) - before
    before = len(traces)
    profile = problem.profile("k")
    # Ten times the points, and only the setup's eager Fisher is traced again: the
    # BFGS objective is compiled once per problem and reused for every point.
    assert len(traces) - before < with_compile
    assert profile.identifiable and profile.converged.all()
    assert profile.stop_reasons == ("threshold", "threshold")
    assert profile.bounds == pytest.approx((mle - 1.959964 * error, mle + 1.959964 * error))
    assert list(profile.grid) == sorted(profile.grid)
    assert profile.log_likelihood_ratio.min() == pytest.approx(0.0, abs=1e-9)
    assert "95% profile interval" in profile.summary() and first.name == "k"


def test_a_profile_stays_open_where_the_data_do_not_bound_the_value(x64):
    experiments, *_ = measurements()
    product = product_model(hm.Parameter(1.0, bounds=(0.1, 10.0)))
    problem = make_problem(product, experiments, log_prior=None, noise="data").optimize().problem
    profiles = problem.profiles()
    assert set(profiles) == {"a", "b"}
    assert profiles["a"].bounds == (None, None) and not profiles["a"].identifiable
    assert profiles["a"].stop_reasons == ("bound", "bound") and not profiles["a"].at_bound
    # b compensates for a along the whole profile, keeping the product fixed.
    product_along = profiles["a"].grid * profiles["a"].others["b"]
    assert np.ptp(product_along) == pytest.approx(0.0, abs=1e-5)
    assert "reaches its bound" in profiles["a"].summary()
    # Profiling b, a compensates until it is pinned, and only then does the profile
    # drop: the interval holds for a's bounds, and the result says which bound.
    with pytest.warns(RuntimeWarning, match="reached its bound along the profile"):
        b = problem.profile("b")
    assert b.identifiable and set(b.at_bound) == {"a"}
    assert "a pinned at its" in b.summary()


def test_a_value_against_its_bound_still_closes_on_the_other_side(x64):
    """The data prefer a rate above the upper bound, so the estimate sits on it and
    its latent is far out in the squash's tail, where steps barely move the value.
    The profile must still walk down, and its crossing has a closed form: the ratio
    is ``((k - mle)^2 - (bound - mle)^2) / se^2`` for a linear-Gaussian model."""
    experiments, ts, ys = measurements()
    mle, error = least_squares(ts, ys)
    bound = mle - 2.0 * error
    model = linear_model(hm.Parameter(mle - 4.0 * error, bounds=(0.5 * mle, bound)))
    problem = make_problem(model, experiments, log_prior=None, noise="data").optimize().problem
    profile = problem.profile("k")
    assert profile.stop_reasons == ("threshold", "bound")
    threshold = scipy_stats.chi2.ppf(0.95, 1)
    expected = mle - float(np.sqrt(threshold * error**2 + (bound - mle) ** 2))
    low, high = profile.bounds
    # Not exact: the ratio is quadratic about the unconstrained optimum, not about
    # the estimate on the bound, so its root is only nearly linear in the value.
    # Bisecting the bracket close to the threshold keeps that error small.
    assert high is None and low == pytest.approx(expected, rel=1e-3)
    assert list(profile.grid) == sorted(profile.grid)
    # The curvature in the squash's tail says nothing about k, so Fisher reports
    # no number for it rather than a misleading one.
    with pytest.warns(RuntimeWarning, match="sit against their bounds"):
        fisher = problem.fisher()
    assert np.isnan(fisher.standard_errors()["k"]) and np.isnan(fisher.intervals()["k"]).all()
    assert np.isnan(fisher.derived(lambda values: values["k"]).standard_error)
    assert "at bound" in fisher.summary()


def test_profile_rejects_what_it_cannot_profile(x64):
    experiments, *_ = measurements()
    problem = make_problem(linear_model(), experiments, log_prior=None, noise="data")
    with pytest.raises(ValueError, match="not free Parameter entries"):
        problem.profile("q")
    with pytest.raises(ValueError, match="confidence"):
        problem.profile("k", confidence=1.5)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda problem: problem.log_density(problem.position), id="log_density"),
        pytest.param(lambda problem: problem.laplace(), id="laplace"),
        pytest.param(lambda problem: problem.preconditioned(), id="preconditioned"),
        pytest.param(lambda problem: problem.optimize(target="posterior"), id="posterior-mode"),
    ],
)
def test_without_a_prior_the_posterior_methods_say_so(x64, call):
    experiments, *_ = measurements()
    problem = make_problem(linear_model(), experiments, log_prior=None, noise="data")
    with pytest.raises(ValueError, match="needs a prior"):
        call(problem)


@pytest.mark.slow
def test_simulation_based_calibration(x64):
    """Draw a rate from the prior and data from the rate: its posterior rank must be uniform.

    Fails if the density, the prior's change of variables or the bucket sum is wrong
    in a way a single posterior check at one data set could miss. The parameter is
    bounded, so the Jacobian is exercised. A prior draw is checked against thinned
    NUTS draws, Talts et al. (2018).
    """
    low, high = 0.0, 2.0
    n_replicates, kept = 60, 50
    ranks = []
    key = jr.key(21)
    for _ in range(n_replicates):
        key, k_rate, k_noise, k_chain = jr.split(key, 4)
        rate = jr.uniform(k_rate, minval=low, maxval=high)
        experiments, *_ = measurements(rate, k_noise)
        problem = make_problem(
            linear_model(hm.Parameter(1.0, bounds=(low, high))),
            experiments,
            log_prior=flat_prior,
            noise="data",
        )
        positions, _ = run_nuts(problem, k_chain, warmup=150, draws=kept * 4)
        draws = np.asarray(problem.physical(positions)["k"]).ravel()[::4]
        ranks.append(int((draws < float(rate)).sum()))
    fraction = (np.asarray(ranks) + 0.5) / (kept + 1)
    assert scipy_stats.kstest(fraction, "uniform").pvalue > 0.01
