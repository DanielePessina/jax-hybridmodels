"""Penalties: the bound penalty and trajectory penalties through the trainers.

The end-to-end tests come first. They drive the decay models from
``tests/_models.py`` through ``train_with_optax``, ``train_with_evosax``
and a bootstrap ``fit_ensemble``, with the penalties set on an
``Objective``, and check that a penalty moves the parameters it should.
The property tests at the bottom guard the gradient-safe primitives the
penalties are built from; they are cheap and catch the class of bug where
a guard silently zeroes a gradient.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _models import SOLVER, make_model, network_predictors, quiet, simulate_network
from jax import Array

import jaxhybridmodels as hm
from jaxhybridmodels.losses import masked_mse
from jaxhybridmodels.objective import Objective
from jaxhybridmodels.penalties import (
    _bounded_leaves,
    attach_penalty_state,
    bound_penalty,
    bounded_predictor_paths,
    box_grid,
    box_violation,
    clip_ste,
    data_penalty_points,
    length_mask_keep,
    penalty_integral,
    penalty_vector_field,
    resolve_penalty_points,
    soft_logit,
    softclip,
    strip_penalty_state,
    trajectory_saturation_penalty,
)
from jaxhybridmodels.trainable import freeze_mask, trainable_mask
from jaxhybridmodels.training import (
    EvosaxTrainingConfig,
    TrainingPhase,
    train_with_evosax,
    train_with_optax,
)
from jaxhybridmodels.training.kernels import apply_length_mask, build_score_bucket
from jaxhybridmodels.ui.testing import RecordingUI


def saturate(rate: hm.BoundedPredictor, scale: float = 50.0) -> hm.BoundedPredictor:
    """Scale the readout weights so the output squash is pinned against its bound."""
    return eqx.tree_at(
        lambda r: r.inner.mlp.layers[-1].weight, rate, rate.inner.mlp.layers[-1].weight * scale
    )


def saturated_predictors() -> dict[str, Any]:
    predictors = network_predictors()
    return dict(predictors, rate=saturate(predictors["rate"]))


def grid_penalty(predictors: Any) -> float:
    """Bound penalty on a sweep of each leaf's input box, for before/after comparisons."""
    points = tuple(box_grid(leaf.in_scaler, 5) for leaf in _bounded_leaves(predictors))
    return float(bound_penalty(predictors, points))


# ---------------------------------------------------------------------------
# Bound penalty through the Optax trainer
# ---------------------------------------------------------------------------


def step_penalties(ui: RecordingUI) -> list[float]:
    """The unweighted bound penalty the trainer reported at every step."""
    return [kw["penalty"] for name, kw in ui.events if name == "on_step_end"]


@pytest.fixture(scope="module")
def penalty_runs(temp_data):
    """Two 30-step fits from the same saturated start.

    ``off`` never charges the penalty. ``switched`` charges nothing for 15
    steps (the phase overrides the objective's weight with 0), then the
    objective's weight 100 for 15, with a box sweep as extra penalty points
    keyed by the predictor's path. In deep saturation the data gradient has
    underflowed, so only the penalty can move the readout.
    """
    start = saturated_predictors()
    runs = {}
    for name, objective, phases in (
        ("off", None, (TrainingPhase(30, lr=0.05),)),
        (
            "switched",
            Objective(
                penalty_weight=1e2, penalty_points={"rate": box_grid(start["rate"].in_scaler, 5)}
            ),
            (TrainingPhase(15, lr=0.05, penalty_weight=0.0), TrainingPhase(15, lr=0.05)),
        ),
    ):
        ui = RecordingUI()
        result = train_with_optax(
            make_model(start, simulate_network),
            temp_data,
            quiet(phases=phases, restore_best=False),
            objective=objective,
            key=jr.key(0),
            ui=ui,
        )
        runs[name] = (list(result.history.loss), result.model.predictors, step_penalties(ui))
    return start, runs


def test_zero_weight_phase_reproduces_the_unpenalised_run(penalty_runs):
    start, runs = penalty_runs
    off_history, _, off_penalty = runs["off"]
    history, _, penalty = runs["switched"]
    # Anti-vacuity: the start really is saturated, so a zero penalty is the
    # weight's doing, not an empty penalty.
    assert grid_penalty(start) > 1.0
    assert history[:15] == off_history[:15]
    assert all(p == 0.0 for p in penalty[:15])
    assert all(p == 0.0 for p in off_penalty)


def test_penalty_phase_pulls_the_network_out_of_saturation(penalty_runs):
    start, runs = penalty_runs
    _, off, _ = runs["off"]
    _, switched, penalty = runs["switched"]
    assert len(penalty) == 30 and all(p > 0.0 for p in penalty[15:])
    assert grid_penalty(switched) < 0.5 * grid_penalty(start)
    assert grid_penalty(switched) < grid_penalty(off)


def simulate_pressure(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """``simulate_network`` feeding ``temp`` to a rate network that calls it ``pressure``."""
    return simulate_network(predictors, ts, {"pressure": covariates["temp"]}, y0, solver)


def test_an_uncovered_leaf_raises_before_training(temp_data):
    # input_keys that match no covariate leave the leaf with no measured
    # points. With the penalty on, that must fail loudly, not skip the leaf.
    predictors = network_predictors()
    inner = predictors["rate"].inner
    predictors["rate"] = hm.BoundedPredictor(
        inner=inner, input_keys=("pressure",), input_bounds=(0.0, 2.0), output_bounds=(0.05, 3.0)
    )
    with pytest.raises(ValueError, match="'rate'.*trajectory_penalty_fn"):
        train_with_optax(
            make_model(predictors, simulate_pressure),
            temp_data,
            quiet(steps=2),
            objective=Objective(penalty_weight=1e-2),
            key=jr.key(0),
        )
    # A phase override switches the penalty on for that phase alone, and
    # the coverage check follows it.
    with pytest.raises(ValueError, match="no penalty points"):
        train_with_optax(
            make_model(predictors, simulate_pressure),
            temp_data,
            quiet(phases=(TrainingPhase(1), TrainingPhase(1, penalty_weight=1e-2))),
            key=jr.key(0),
        )


@pytest.mark.parametrize(
    "points,match",
    [
        ({"rates": jnp.zeros((2, 1))}, r"valid paths are \['rate'\]"),
        ({"rate": jnp.zeros((2, 3))}, "input columns"),
    ],
)
def test_malformed_penalty_points_raise(temp_data, points, match):
    objective = Objective(penalty_weight=1e-2, penalty_points=points)
    with pytest.raises(ValueError, match=match):
        train_with_optax(
            make_model(network_predictors(), simulate_network),
            temp_data,
            quiet(steps=2),
            objective=objective,
            key=jr.key(0),
        )


def test_penalty_points_are_keyed_by_the_freeze_path_syntax():
    """A predictor's penalty-points key is the dotted path ``freeze_mask`` takes.

    Nested containers join with dots, and a predictor nested inside another
    is reached through ``inner``. Each path prefixes the mask leaves of that
    predictor, so the same string freezes it.
    """
    hot, cold = saturate(network_predictors()["rate"]), network_predictors(jr.key(1))["rate"]
    outer = hm.BoundedPredictor(
        inner=cold, input_keys=("temp",), input_bounds=(0.0, 2.0), output_bounds=(0.05, 3.0)
    )
    predictors = {"growth": {"rate": hot}, "nucleation": outer, "offset": hm.Constant(0.0)}
    paths = bounded_predictor_paths(predictors)
    assert paths == ("growth.rate", "nucleation", "nucleation.inner")
    mask = trainable_mask(predictors)
    for path in paths:
        assert sum(map(bool, jax.tree.leaves(freeze_mask(predictors, path)))) < sum(
            map(bool, jax.tree.leaves(mask))
        )

    grid = box_grid(hot.in_scaler, 3)
    extras = resolve_penalty_points(predictors, {"growth.rate": grid})
    assert [e.shape for e in extras] == [(3, 1), (0, 1), (0, 1)]
    assert jnp.array_equal(extras[0], grid)
    assert resolve_penalty_points(predictors, None) == ()
    with pytest.raises(ValueError, match="valid paths"):
        resolve_penalty_points(predictors, {"growth": grid})


# ---------------------------------------------------------------------------
# Trajectory penalties
# ---------------------------------------------------------------------------

N_ACC = 2  # saturation accumulator, input-violation accumulator


class Latent(hm.Predictor):
    """One trainable latent ``c``, ignoring its input.

    Restarts keep ``c``, so "training moved c" cannot be satisfied by a
    tournament re-draw alone.
    """

    c: Array

    def __init__(self, c: float) -> None:
        self.c = jnp.asarray(c, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:
        return self.c

    def initialized_with_key(self, key: Array) -> Latent:
        return self


def latent_predictors(c: float) -> tuple[hm.BoundedPredictor]:
    return (
        hm.BoundedPredictor(
            inner=Latent(c), input_keys=("u",), input_bounds=(-3.0, 3.0), output_bounds=(0.0, 1.0)
        ),
    )


def simulate_embedded(predictors: Any, ts: Array, covariates: Any, y0: Array, solver: Any) -> Array:
    """Decay whose state ``x`` feeds the predictor; two accumulators ride along.

    The accumulators integrate the predictor's output saturation and its
    input violation, so the final state holds both time integrals.
    """
    (p,) = predictors
    rate = jnp.reshape(p({"u": jnp.asarray(1.0)}), ())

    def physics(t, y, args):
        return -rate * y

    def penalty_rates(t, y, args):
        u = y[0]
        z = p.inner(p.in_scaler.to_latent(u))
        return jnp.stack([p.out_scaler.saturation(z), p.in_scaler.input_violation(u)])

    term = diffrax.ODETerm(penalty_vector_field(physics, penalty_rates))
    return solver.diffeqsolve(term, ts, y0).ys


def strip_accumulators(state: Array) -> Array:
    return strip_penalty_state(state, N_ACC)


def accumulated_penalty(full_state: Array, bp: Any) -> Array:
    return jnp.sum(penalty_integral(full_state, N_ACC))


EMBEDDED_KWARGS: dict[str, Any] = dict(
    simulate_fn=simulate_embedded, state_to_output=strip_accumulators, solver=SOLVER
)


def embedded_model(predictors: Any) -> hm.HybridModel:
    """The embedded-penalty model: one observed channel ``x``, accumulators stripped."""
    return hm.HybridModel(predictors=predictors, output_channel_names=("x",), **EMBEDDED_KWARGS)


def embedded_dataset(*, probe: bool = False) -> hm.Dataset:
    """One experiment observed at four times, or a probe with no observations.

    A probe keeps its time grid, so the solver still integrates over it, but
    its mask is all False: the data loss is exactly zero and only a
    trajectory penalty can produce a gradient.
    """
    ts = jnp.array([0.5, 1.0, 1.5, 2.0])
    values = jnp.array([]) if probe else jnp.exp(-0.5 * ts)
    experiment = hm.make_experiment(
        channels={"x": hm.ChannelObs(ts=ts, values=values)},
        y0=attach_penalty_state(jnp.array([1.0]), N_ACC),
        t0=0.0,
        exp_id="probe" if probe else "real",
    )
    return hm.make_dataset([experiment], output_channel_names=("x",))


def test_the_score_charges_the_trajectory_penalty():
    data_only = build_score_bucket(loss_fn=masked_mse, **EMBEDDED_KWARGS)
    penalised = build_score_bucket(
        loss_fn=masked_mse,
        trajectory_penalty_fn=accumulated_penalty,
        trajectory_penalty_weight=1.0,
        **EMBEDDED_KWARGS,
    )
    predictors = latent_predictors(5.0)
    full = jnp.asarray(1.0)
    # Both datasets share one grid shape, so each kernel compiles once.
    for probe in (False, True):
        bp = embedded_dataset(probe=probe).bucket_payloads[0]
        data = float(data_only(predictors, bp, full))
        if probe:
            assert int(bp.mask.sum()) == 0
            assert data == pytest.approx(0.0, abs=1e-7)
        assert float(penalised(predictors, bp, full)) > data + 1.0


TRAJECTORY = Objective(trajectory_penalty_fn=accumulated_penalty, trajectory_penalty_weight=1.0)


def test_evosax_ranks_by_the_trajectory_penalty():
    config = EvosaxTrainingConfig(population_size=8, steps=10, sigma_init=0.5, verbose=False)
    result = train_with_evosax(
        embedded_model(latent_predictors(5.0)),
        embedded_dataset(),
        config,
        objective=TRAJECTORY,
        key=jr.key(0),
    )
    assert len(result.history.loss) == 10
    assert float(result.model.predictors[0].inner.c) < 5.0


def test_a_probe_alone_drives_every_bootstrap_member():
    # The probe has no observations, so the penalty is the only gradient.
    members = embedded_model(latent_predictors(5.0)).fit_ensemble(
        embedded_dataset(probe=True),
        quiet(steps=10, lr=1e-1, restore_best=False),
        objective=TRAJECTORY,
        n_members=2,
        bootstrap=True,
        key=jr.key(0),
    )
    assert len(members) == 2
    for member in members:
        assert np.isfinite(member.final_loss)
        assert float(member.model.predictors[0].inner.c) < 4.5


def test_parallel_saturation_penalty_moves_the_latent_through_the_trainer():
    # Parallel hybrid: the predictor's output is the observed channel, so no
    # accumulators are needed; the hook inverts outputs back to latents.
    out_scaler = latent_predictors(0.0)[0].out_scaler

    def simulate(predictors, ts, covariates, y0, solver):
        out = predictors[0]({"u": jnp.asarray(0.0)})
        return jnp.broadcast_to(out[None], (ts.shape[0], 1))

    def saturation(full_state, bp):
        return trajectory_saturation_penalty(full_state, out_scaler)

    ts = jnp.linspace(0.0, 1.0, 4)
    experiment = hm.make_experiment(
        channels={"y": hm.ChannelObs(ts=ts, values=jnp.full_like(ts, 0.99))}, y0=[0.0], exp_id="e"
    )
    dataset = hm.make_dataset([experiment], output_channel_names=("y",))
    start = latent_predictors(8.0)
    # Anti-vacuity: the penalty is live at the start.
    assert float(saturation(jnp.full((4, 1), start[0]({"u": 0.0})), None)) > 0.0
    model = hm.HybridModel(
        predictors=start,
        simulate_fn=simulate,
        state_to_output=lambda s: s,
        output_channel_names=("y",),
        solver=SOLVER,
    )
    result = train_with_optax(
        model,
        dataset,
        quiet(steps=20, lr=1e-1, restore_best=False),
        objective=Objective(trajectory_penalty_fn=saturation, trajectory_penalty_weight=1.0),
        key=jr.key(0),
    )
    assert float(result.model.predictors[0].inner.c) < 8.0


def test_penalty_state_helpers():
    y0 = jnp.array([1.0, 2.0])
    full = jnp.stack([attach_penalty_state(y0, 2), jnp.array([1.0, 2.0, 5.0, 6.0])])
    assert full[0].tolist() == [1.0, 2.0, 0.0, 0.0]
    assert strip_penalty_state(full, 2).shape == (2, 2)
    assert penalty_integral(full, 2).tolist() == [5.0, 6.0]

    # The base field sees only the physical state; the rates are appended.
    seen = []

    def base(t, y, args):
        seen.append(y.shape)
        return -y

    vf = penalty_vector_field(base, lambda t, y, args: jnp.array([y[0] ** 2, 7.0]))
    assert vf(0.0, jnp.array([3.0, 0.0, 0.0]), None).tolist() == [-3.0, 9.0, 7.0]
    assert seen == [(1,)]

    for helper, arg in (
        (attach_penalty_state, y0),
        (strip_penalty_state, full),
        (penalty_integral, full),
    ):
        with pytest.raises(ValueError, match="at least one"):
            helper(arg, n=0)


# ---------------------------------------------------------------------------
# Penalty points
# ---------------------------------------------------------------------------


def test_data_penalty_points_sit_at_observed_cells(temp_data):
    preds = network_predictors()
    (source,) = data_penalty_points(preds, temp_data)
    assert source is not None
    # Ten observed cells across the four experiments, one temp column.
    assert source.points.shape == (10, 1)
    assert set(np.asarray(source.points[:, 0]).tolist()) == {0.5, 1.0, 1.5}
    # Horizon masking follows the loss's own prefix mask, cell for cell.
    for fraction in (1.0, 0.5, 0.05):
        expected = jnp.concatenate(
            [
                apply_length_mask(bp, jnp.asarray(fraction)).mask.any(-1)[bp.mask.any(-1)]
                for bp in temp_data.bucket_payloads
            ]
        )
        assert jnp.array_equal(length_mask_keep(source, fraction), expected)


def test_data_penalty_points_resolution():
    ts = jnp.array([0.5, 1.0])
    experiments = [
        hm.make_experiment(
            channels={"y": hm.ChannelObs(ts=ts, values=ts)},
            covariates={"a": 1.0, "b": 10.0},
            y0=[0.0],
        ),
        hm.make_experiment(
            channels={"y": hm.ChannelObs(ts=ts, values=jnp.array([]))},
            covariates={"a": 9.0, "b": 99.0},
            y0=[0.0],
        ),
    ]
    dataset = hm.make_dataset(experiments, output_channel_names=("y",))

    def leaf(keys):
        inner = hm.MLPPredictor(in_size=2, out_size=1, width_size=2, depth=1, key=jr.key(0))
        return hm.BoundedPredictor(inner=inner, input_keys=keys, input_bounds=(0.0, 100.0))

    swapped, missing = data_penalty_points((leaf(("b", "a")), leaf(("a", "c"))), dataset)
    assert swapped is not None and missing is None
    # Columns follow input_keys order, and the probe contributes nothing.
    assert swapped.points.tolist() == [[10.0, 1.0], [10.0, 1.0]]
    assert data_penalty_points(hm.Constant(1.0), dataset) == ()


def test_box_grid():
    linear = hm.BoundScaler(((0.0, 10.0), (0.0, 5.0)))
    grid = box_grid(linear, n_per_dim=5)
    mesh = jnp.meshgrid(jnp.linspace(0.0, 10.0, 5), jnp.linspace(0.0, 5.0, 5), indexing="ij")
    assert jnp.array_equal(grid, jnp.stack([m.reshape(-1) for m in mesh], axis=-1))
    # Under a log warp the sweep is uniform in decades, so the low end is
    # not starved of points.
    log = box_grid(hm.BoundScaler(((1e-6, 1e2),), warp="log10"), n_per_dim=5)
    np.testing.assert_allclose(jnp.log10(log[:, 0]), jnp.linspace(-6.0, 2.0, 5), atol=1e-5)
    with pytest.raises(ValueError, match="n_per_dim"):
        box_grid(linear, n_per_dim=1)


# ---------------------------------------------------------------------------
# bound_penalty on a pytree
# ---------------------------------------------------------------------------


def test_bound_penalty_ignores_the_pytree_shape():
    hot, cold = saturate(network_predictors()["rate"]), network_predictors(jr.key(1))["rate"]

    class Pair(NamedTuple):
        a: hm.BoundedPredictor
        b: hm.BoundedPredictor

    shapes = [
        (hot, cold),
        [hot, cold],
        {"a": hot, "b": cold},
        Pair(hot, cold),
        ((hot,), {"b": cold}),
    ]
    values = {round(grid_penalty(shape), 4) for shape in shapes}
    assert len(values) == 1 and values.pop() > 0.0
    assert float(bound_penalty(cold, (jnp.zeros((0, 1)),))) == 0.0
    assert float(bound_penalty(hm.Constant(1.0), ())) == 0.0


def test_bound_penalty_finds_a_predictor_nested_inside_another():
    # An is_leaf traversal that stops at the outer BoundedPredictor would
    # skip the inner one's box entirely.
    inner = saturate(network_predictors()["rate"])
    outer = hm.BoundedPredictor(
        inner=inner, input_keys=("temp",), input_bounds=(0.0, 2.0), output_bounds=(0.05, 3.0)
    )
    assert len(_bounded_leaves(outer)) == 2
    with pytest.raises(ValueError, match="points"):
        bound_penalty(outer, (box_grid(outer.in_scaler, 3),))
    fresh = hm.BoundedPredictor(
        inner=network_predictors()["rate"],
        input_keys=("temp",),
        input_bounds=(0.0, 2.0),
        output_bounds=(0.05, 3.0),
    )
    assert grid_penalty(outer) > grid_penalty(fresh)


def test_bound_penalty_gradient_reaches_the_inner_weights():
    rate = saturate(network_predictors()["rate"])
    points = (box_grid(rate.in_scaler, 5),)
    grads = eqx.filter_grad(lambda p: bound_penalty(p, points))(rate)
    w = grads.inner.mlp.layers[-1].weight
    assert jnp.all(jnp.isfinite(w)) and float(jnp.linalg.norm(w)) > 1.0
    jitted = eqx.filter_jit(lambda p: bound_penalty(p, points))(rate)
    np.testing.assert_allclose(jitted, bound_penalty(rate, points), rtol=1e-6)


class Latent1(Latent):
    """``Latent`` with a rank-1 output, the shape before one-output predictors went scalar."""

    def __init__(self, c: float) -> None:
        super().__init__(c)

    def __call__(self, x: Array) -> Array:
        return self.c[None]


def test_penalties_accept_scalar_and_rank_1_outputs():
    """One-output predictors may return ``()`` or ``(1,)``; the penalties agree on both.

    ``saturation`` reads the trailing axis as outputs, so a ``[G]`` batch of
    scalar latents read as ``G`` outputs of one point would charge only the
    first point.
    """
    points = (jnp.linspace(-3.0, 3.0, 7)[:, None],)

    def bounded(inner: hm.Predictor) -> hm.BoundedPredictor:
        return hm.BoundedPredictor(
            inner=inner, input_keys=("u",), input_bounds=(-3.0, 3.0), output_bounds=(0.0, 1.0)
        )

    scalar = float(bound_penalty(bounded(Latent(6.0)), points))
    rank_1 = float(bound_penalty(bounded(Latent1(6.0)), points))
    assert scalar > 0.0 and scalar == pytest.approx(rank_1)

    scaler = latent_predictors(0.0)[0].out_scaler
    outputs = scaler.from_latent(jnp.linspace(0.0, 8.0, 5))
    np.testing.assert_allclose(
        trajectory_saturation_penalty(outputs, scaler),
        trajectory_saturation_penalty(outputs[:, None], scaler),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Gradient-safe primitives
# ---------------------------------------------------------------------------


def test_soft_logit_is_exact_inside_and_linear_outside():
    s = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(soft_logit(s), jax.scipy.special.logit(s), atol=1e-5)
    grad = jax.vmap(jax.grad(soft_logit))
    # The stop_gradient on the clamp is load-bearing: without it the
    # interior derivative picks up a spurious term.
    np.testing.assert_allclose(grad(s), 1.0 / (s * (1.0 - s)), rtol=1e-4)
    outside = jnp.array([-5.0, -0.01, 1.01, 7.0])
    assert jnp.all(jnp.isfinite(soft_logit(outside))) and jnp.all(grad(outside) > 0.0)
    # The hard clip this replaced has exactly zero gradient there.
    hard = jax.grad(lambda x: jax.scipy.special.logit(jnp.clip(x, 1e-3, 1 - 1e-3)))
    assert float(hard(jnp.asarray(2.0))) == 0.0
    # eps sets the continuation slope.
    slope = lambda eps: float(jax.grad(lambda v: soft_logit(v, eps))(jnp.asarray(3.0)))  # noqa: E731
    assert slope(1e-4) > slope(1e-2)


def test_box_violation_pushes_back_from_outside_only():
    lows, highs = jnp.array([0.0]), jnp.array([10.0])
    grad = jax.grad(lambda v: box_violation(v, lows, highs))
    assert float(box_violation(jnp.array([4.0]), lows, highs)) == 0.0
    assert float(grad(jnp.array([4.0]))[0]) == 0.0
    assert float(grad(jnp.array([13.0]))[0]) > 0.0 > float(grad(jnp.array([-3.0]))[0])
    pushes = [float(grad(jnp.array([x]))[0]) for x in (20.0, 200.0, 2000.0)]
    assert pushes[0] < pushes[1] < pushes[2]
    # Width-normalised, so one weight works across channels.
    np.testing.assert_allclose(
        box_violation(jnp.array([2.0]), jnp.array([0.0]), jnp.array([1.0])),
        box_violation(jnp.array([2000.0]), jnp.array([0.0]), jnp.array([1000.0])),
    )


def test_clip_ste_and_softclip():
    xs = jnp.array([-2.0, 0.4, 3.0])
    np.testing.assert_allclose(clip_ste(xs, 0.0, 1.0), [0.0, 0.4, 1.0])
    assert jax.vmap(jax.grad(lambda v: clip_ste(v, 0.0, 1.0)))(xs).tolist() == [1.0, 1.0, 1.0]

    out = softclip(jnp.linspace(-10.0, 10.0, 41), 0.0, 1.0)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert 0.0 < float(jax.grad(lambda v: softclip(v, 0.0, 1.0))(jnp.asarray(1.05))) < 1.0
    # Interior distortion is O(1 / beta), which is why softclip is not the
    # logit guard. Measured near an edge; the shoulders cancel at the centre.
    error = lambda beta: abs(float(softclip(jnp.asarray(0.05), 0.0, 1.0, beta)) - 0.05)  # noqa: E731
    assert error(20.0) > error(200.0) and error(20.0) > 1e-3
