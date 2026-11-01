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


def state_to_output(state: Any) -> Any:
    return state


def linear_model(k: Any = None, solver: hm.SolverConfig = SOLVER, **extra: Any) -> hm.HybridModel:
    return hm.HybridModel.from_vector_field(
        vector_field=field,
        predictors={"k": hm.Parameter(0.5) if k is None else k, **extra},
        state_to_output=state_to_output,
        output_channel_names=("y",),
        solver=solver,
    )


def measurements() -> tuple[list[hm.Experiment], np.ndarray, np.ndarray]:
    """Noisy data from a fixed key, returning the times and values for the oracles."""
    experiments, all_ts, all_ys = [], [], []
    key = jr.key(11)
    for name, grid in GRIDS.items():
        key, sub = jr.split(key)
        ts = jnp.asarray(grid)
        ys = TRUE_RATE * ts + SIGMA * jr.normal(sub, ts.shape, dtype=ts.dtype)
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


def effective_size(chain: np.ndarray) -> float:
    """Geyer's initial positive sequence, enough to set a Monte Carlo tolerance."""
    centred = chain - chain.mean()
    n = centred.size
    spectrum = np.fft.rfft(centred, 2 * n)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    rho = autocovariance / autocovariance[0]
    total = 0.0
    for i in range(0, n - 1, 2):
        pair = rho[i] + rho[i + 1]
        if pair < 0:
            break
        total += pair
    return n / max(2 * total - 1, 1.0)


@pytest.fixture(scope="module")
def x64():
    with jax.enable_x64():
        yield


@pytest.fixture(scope="module")
def linear(x64):
    experiments, ts, ys = measurements()
    problem = make_problem(linear_model(), experiments, log_prior=normal_prior, noise="data")
    positions, divergent = run_nuts(problem, jr.key(0))
    draws = np.asarray(jax.vmap(problem.physical)(positions)["k"]).ravel()
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
    mean, sd = conjugate(ts, ys)
    tolerance = 4 * sd / np.sqrt(effective_size(draws))
    assert not divergent.any()
    assert draws.mean() == pytest.approx(mean, abs=tolerance)
    assert draws.std(ddof=1) == pytest.approx(sd, rel=0.1)


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
    named = jax.vmap(problem.physical)(positions)
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
    assert draws.mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(effective_size(draws)))
    assert draws.std(ddof=1) == pytest.approx(sd, rel=0.15)


def test_a_custom_likelihood_can_restate_the_gaussian(linear):
    problem, *_ = linear

    def gaussian(pred_obs: Any, bp: Any, noise: Any) -> Any:
        cells = jstats.norm.logpdf(
            jnp.where(bp.mask, bp.y_observed, 0.0),
            pred_obs,
            jnp.sqrt(jnp.where(bp.mask, bp.yvar, 1.0)),
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
        state_to_output=state_to_output,
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
    draws = np.asarray(jax.vmap(problem.physical)(positions)["k"]).ravel()
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

    last_layer = make_problem(
        linear_model(net=network),
        experiments,
        log_prior=normal_prior,
        noise="data",
        free=["k", "net.inner.mlp.layers.1"],
    )
    named = set(last_layer.physical(last_layer.position))
    assert named == {"k", "net.inner.mlp.layers.1.weight", "net.inner.mlp.layers.1.bias"}
    moved = last_layer.to_model(last_layer.position)
    assert isinstance(moved, hm.HybridModel) and moved.solver.throw


def test_to_model_predicts_at_a_draw(linear):
    problem, experiments, *_ = linear
    model = problem.to_model(problem.position)
    assert model.solver.throw, "a model handed back keeps the raising solver"
    assert set(model.predict(experiments)) == set(GRIDS)


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
        pytest.param({"free": "nope"}, ValueError, "nope", id="free-path"),
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
    t_last = float(long.times[last])
    assert float(long.outputs["y"][last]) == pytest.approx(mean * t_last, abs=4 * sd * t_last / 15)
    assert float(long.std["y"][last]) == pytest.approx(sd * t_last, rel=0.2)
    low, high = (float(long.quantiles["y"][q][last]) for q in (0.05, 0.95))
    assert high - low == pytest.approx(2 * 1.6449 * sd * t_last, rel=0.2)
    frame = result.to_dataframe()
    assert {"q0.05", "q0.95", "std"} <= set(frame.columns)


def test_observation_noise_widens_the_interval_by_the_noise(x64):
    experiments, ts, ys = measurements()

    def prior(predictors: Any, noise: Any) -> Any:
        return normal_prior(predictors, noise)

    problem = make_problem(
        linear_model(hm.Parameter(0.7)),
        experiments,
        log_prior=prior,
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
        pytest.param(lambda p, x: p.to_arviz(x), ValueError, "chain", id="arviz-axes"),
    ],
)
def test_prediction_rejects_bad_input(linear, call, error, match):
    pytest.importorskip("arviz_base")
    problem, *_ = linear
    positions = jax.tree_util.tree_map(lambda leaf: leaf[None], problem.position)
    with pytest.raises(error, match=match):
        call(problem, positions)
