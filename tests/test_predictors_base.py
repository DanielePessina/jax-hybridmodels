# ruff: noqa: F722

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, Float

from hybridmodels.predictors import (
    BoundedPredictor,
    BoundScaler,
    Predictor,
    reinitialize_with_key,
)


class _LinearPredictor(Predictor):
    weights: Float[Array, "in_dim out_dim"]

    def __call__(self, x: Array) -> Array:
        return x @ self.weights


class _PredictorWithIntField(Predictor):
    weights: Float[Array, " n"]
    counter: Array

    def __call__(self, x: Array) -> Array:
        return x * self.counter.astype(x.dtype)


class _PredictorWithProtocol(Predictor):
    weights: Float[Array, " n"]
    sentinel: float = 42.0

    def __call__(self, x: Array) -> Array:
        return x * self.weights

    def initialized_with_key(self, key: Array) -> _PredictorWithProtocol:
        return _PredictorWithProtocol(
            weights=jnp.full_like(self.weights, self.sentinel),
            sentinel=self.sentinel,
        )


class _ConstantPredictor(Predictor):
    value: Array

    def __call__(self, x: Array) -> Array:
        return self.value


class TestBoundScaler:
    def test_unsupported_transform_raises(self):
        with pytest.raises(ValueError):
            BoundScaler(bounds=((0.0, 1.0),), transform="tanh")

    def test_sigmoid_inverse_one_dim(self):
        scaler = BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid")
        x = jnp.array([3.5])
        z = scaler.to_latent(x)
        x_back = scaler.from_latent(z)
        assert jnp.allclose(x, x_back, atol=1e-5)

    def test_sigmoid_inverse_vector_bounds(self):
        scaler = BoundScaler(
            bounds=((0.0, 10.0), (-1.0, 1.0), (100.0, 200.0)),
            transform="sigmoid",
        )
        x = jnp.array([3.5, 0.25, 150.0])
        z = scaler.to_latent(x)
        x_back = scaler.from_latent(z)
        assert jnp.allclose(x, x_back, atol=1e-5)

    def test_from_latent_at_zero_returns_midpoint(self):
        scaler = BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid")
        out = scaler.from_latent(jnp.array([0.0]))
        assert jnp.allclose(out, jnp.array([5.0]), atol=1e-6)

    def test_from_latent_respects_bounds(self):
        scaler = BoundScaler(
            bounds=((0.0, 10.0), (-1.0, 1.0)),
            transform="sigmoid",
        )
        zs = jnp.array([-100.0, 100.0])
        out = scaler.from_latent(zs)
        assert out[0] >= 0.0 and out[0] <= 10.0
        assert out[1] >= -1.0 and out[1] <= 1.0

    def test_temperature_changes_latent_value(self):
        s1 = BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid", temperature=1.0)
        s2 = BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid", temperature=2.0)
        x = jnp.array([7.0])
        z1 = s1.to_latent(x)
        z2 = s2.to_latent(x)
        assert not jnp.allclose(z1, z2)
        assert jnp.allclose(s2.from_latent(z2), x, atol=1e-5)
        assert jnp.allclose(s1.from_latent(z1), x, atol=1e-5)


def _identity_bp(
    *,
    input_keys: tuple[str, ...] | None = None,
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0)),
) -> BoundedPredictor:
    """Two-in / two-out identity-through-latent BoundedPredictor for tests."""
    n = len(bounds)
    return BoundedPredictor(
        input_keys=input_keys,
        in_scaler=BoundScaler(bounds=bounds, transform="sigmoid"),
        inner=_LinearPredictor(weights=jnp.eye(n)),
        out_scaler=BoundScaler(bounds=bounds, transform="sigmoid"),
    )


class TestBoundedPredictorInputKeys:
    """Contract for the ``input_keys`` static field and the polymorphic call.

    The selector module was folded into BoundedPredictor: ``input_keys``
    declares the named-input order, defaulting to ``("x1", ..., "xN")`` if
    omitted, with cardinality matching ``in_scaler.bounds``. ``__call__``
    accepts either a ``dict[str, Array]`` (subset extraction in declared
    order) or a rank-1 ``Array`` (passed through unchanged).
    """

    def test_default_input_keys_autofill(self):
        bp = _identity_bp(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
        assert bp.input_keys == ("x1", "x2", "x3")

    def test_explicit_input_keys_preserved(self):
        bp = _identity_bp(input_keys=("a", "b"))
        assert bp.input_keys == ("a", "b")

    def test_cardinality_mismatch_raises(self):
        with pytest.raises(ValueError):
            BoundedPredictor(
                input_keys=("only_one",),
                in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
                inner=_LinearPredictor(weights=jnp.eye(2)),
                out_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
            )

    def test_empty_bounds_raises(self):
        # at-least-one-input rule per the user's "no training otherwise". The
        # constructor checks bounds first, so input_keys can be omitted: this
        # pins that the empty-bounds branch fires before the cardinality check.
        with pytest.raises(ValueError):
            BoundedPredictor(
                in_scaler=BoundScaler(bounds=(), transform="sigmoid"),
                inner=_LinearPredictor(weights=jnp.zeros((0, 1))),
                out_scaler=BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid"),
            )

    def test_dict_input_extracts_in_declared_order(self):
        bp = _identity_bp(input_keys=("a", "b"))
        cov = {"a": jnp.array(0.4), "b": jnp.array(0.7)}
        out = bp(cov)
        assert jnp.allclose(out, jnp.array([0.4, 0.7]), atol=1e-5)

    def test_dict_input_reordered_input_keys_swap_output(self):
        bp_ab = _identity_bp(input_keys=("a", "b"))
        bp_ba = _identity_bp(input_keys=("b", "a"))
        cov = {"a": jnp.array(0.4), "b": jnp.array(0.7)}
        assert jnp.allclose(bp_ab(cov), jnp.array([0.4, 0.7]), atol=1e-5)
        assert jnp.allclose(bp_ba(cov), jnp.array([0.7, 0.4]), atol=1e-5)

    def test_dict_input_extra_keys_ignored(self):
        # subset-extraction contract: extra keys in the inputs dict are fine.
        bp = _identity_bp(input_keys=("a", "b"))
        cov = {"a": jnp.array(0.4), "b": jnp.array(0.7), "extra": jnp.array(99.0)}
        out = bp(cov)
        assert jnp.allclose(out, jnp.array([0.4, 0.7]), atol=1e-5)

    def test_dict_input_missing_key_raises(self):
        bp = _identity_bp(input_keys=("a", "b"))
        with pytest.raises(KeyError):
            bp({"a": jnp.array(0.4)})

    def test_array_input_passthrough(self):
        bp = _identity_bp(input_keys=("a", "b"))
        x = jnp.array([0.4, 0.7])
        out = bp(x)
        assert jnp.allclose(out, jnp.array([0.4, 0.7]), atol=1e-5)

    def test_array_input_wrong_length_raises(self):
        # eqx.error_if: rank-1 array must match len(input_keys).
        bp = _identity_bp(input_keys=("a", "b"))
        with pytest.raises(Exception):  # noqa: B017 — eqx.error_if raises a concrete subtype
            bp(jnp.array([0.4, 0.7, 0.1]))

    def test_array_input_wrong_rank_raises(self):
        bp = _identity_bp(input_keys=("a", "b"))
        with pytest.raises(Exception):  # noqa: B017 — eqx.error_if raises a concrete subtype
            bp(jnp.array([[0.4, 0.7]]))

    def test_array_and_dict_yield_same_result(self):
        # The dict path is just stack-then-Array. They must agree.
        bp = _identity_bp(input_keys=("a", "b"))
        x = jnp.array([0.4, 0.7])
        cov = {"a": jnp.array(0.4), "b": jnp.array(0.7)}
        assert jnp.allclose(bp(x), bp(cov), atol=1e-6)


class TestBoundedPredictor:
    def test_pipeline_round_trip_with_identity_inner(self):
        bp = BoundedPredictor(
            input_keys=("a", "b"),
            in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
            inner=_LinearPredictor(weights=jnp.eye(2)),
            out_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
        )
        cov = {"a": jnp.array(0.5), "b": jnp.array(0.3)}
        out = bp(cov)
        assert out.shape == (2,)
        assert jnp.allclose(out, jnp.array([0.5, 0.3]), atol=1e-5)

    def test_dimensions_independent(self):
        bp = BoundedPredictor(
            input_keys=("x",),
            in_scaler=BoundScaler(bounds=((0.0, 10.0),), transform="sigmoid"),
            inner=_LinearPredictor(weights=jnp.array([[1.0, 1.0]])),
            out_scaler=BoundScaler(
                bounds=((0.0, 100.0), (-50.0, 50.0)),
                transform="sigmoid",
            ),
        )
        out = bp({"x": jnp.array(5.0)})
        assert out.shape == (2,)
        assert 0.0 <= float(out[0]) <= 100.0
        assert -50.0 <= float(out[1]) <= 50.0


class TestPredictorsTuple:
    """Multi-rate composition via a tuple of predictors.

    The framework's convention for multi-rate hybrid models is to pass a
    tuple of ``BoundedPredictor``s to training and unpack it at the top
    of the user's vector field — there is no framework wrapper class for
    "two rates" or "a list of rates". These tests pin the behaviour the
    user gets from operating directly on those tuple primitives:
    composition, indexing, partial freezing, and so on.
    """

    def _bp(self, out_low: float, out_high: float) -> BoundedPredictor:
        return BoundedPredictor(
            input_keys=("T",),
            in_scaler=BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid"),
            inner=_LinearPredictor(weights=jnp.array([[1.0]])),
            out_scaler=BoundScaler(bounds=((out_low, out_high),), transform="sigmoid"),
        )

    def test_two_predictor_tuple_yields_two_rates(self):
        # Convention: predictors = (growth_BP, nucleation_BP). Each call
        # produces one bounded scalar; user stacks at vector-field call site.
        predictors = (self._bp(10.0, 20.0), self._bp(1.0, 2.0))
        nucleation_bp, growth_bp = predictors
        inputs = {"T": jnp.array(0.5)}
        n_out = nucleation_bp(inputs)
        g_out = growth_bp(inputs)
        assert n_out.shape == (1,)
        assert g_out.shape == (1,)
        assert 10.0 <= float(n_out[0]) <= 20.0
        assert 1.0 <= float(g_out[0]) <= 2.0

    def test_tuple_partition_walks_into_branches(self):
        # eqx.partition + the default trainable mask should walk into both
        # tuple branches uniformly — no special tuple-aware code needed.
        import equinox as eqx

        from hybridmodels.trainable import trainable_mask

        predictors = (self._bp(10.0, 20.0), self._bp(1.0, 2.0))
        mask = trainable_mask(predictors)
        params, static = eqx.partition(predictors, mask)
        # Both branches contribute params; static side preserves shape.
        assert isinstance(params, tuple) and len(params) == 2
        assert isinstance(static, tuple) and len(static) == 2


class TestReinitializeWithKey:
    def _predictor(self) -> _LinearPredictor:
        return _LinearPredictor(weights=jnp.ones((3, 2)))

    def test_changes_inexact_leaves(self):
        p = self._predictor()
        p_new = reinitialize_with_key(p, jr.PRNGKey(0))
        assert p_new.weights.shape == p.weights.shape
        assert p_new.weights.dtype == p.weights.dtype
        assert not jnp.allclose(p_new.weights, p.weights)

    def test_deterministic_for_same_key(self):
        p = self._predictor()
        a = reinitialize_with_key(p, jr.PRNGKey(0))
        b = reinitialize_with_key(p, jr.PRNGKey(0))
        assert jnp.allclose(a.weights, b.weights)

    def test_different_keys_diverge(self):
        p = self._predictor()
        a = reinitialize_with_key(p, jr.PRNGKey(0))
        b = reinitialize_with_key(p, jr.PRNGKey(1))
        assert not jnp.allclose(a.weights, b.weights)

    def test_preserves_non_float_leaves(self):
        p = _PredictorWithIntField(
            weights=jnp.ones((3,)),
            counter=jnp.asarray(5, dtype=jnp.int32),
        )
        p_new = reinitialize_with_key(p, jr.PRNGKey(0))
        assert jnp.array_equal(p_new.counter, p.counter)
        assert p_new.counter.dtype == p.counter.dtype
        assert not jnp.allclose(p_new.weights, p.weights)

    def test_uses_protocol_when_present(self):
        p = _PredictorWithProtocol(weights=jnp.ones((3,)))
        p_new = reinitialize_with_key(p, jr.PRNGKey(0))
        assert jnp.allclose(p_new.weights, jnp.full((3,), 42.0))

    def test_works_on_bounded_predictor(self):
        bp = BoundedPredictor(
            input_keys=("a",),
            in_scaler=BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid"),
            inner=_LinearPredictor(weights=jnp.ones((1, 1))),
            out_scaler=BoundScaler(bounds=((0.0, 1.0),), transform="sigmoid"),
        )
        bp_new = reinitialize_with_key(bp, jr.PRNGKey(0))
        assert not jnp.allclose(bp_new.inner.weights, bp.inner.weights)
        assert bp_new.input_keys == bp.input_keys
        assert bp_new.in_scaler.bounds == bp.in_scaler.bounds
        assert bp_new.in_scaler.transform == bp.in_scaler.transform


def test_jit_traces_through_bounded_predictor():
    bp = BoundedPredictor(
        input_keys=("a", "b"),
        in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
        inner=_LinearPredictor(weights=jnp.eye(2)),
        out_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
    )

    @jax.jit
    def call(predictor: BoundedPredictor, cov: dict[str, Array]) -> Array:
        return predictor(cov)

    cov = {"a": jnp.array(0.5), "b": jnp.array(0.3)}
    out = call(bp, cov)
    assert out.shape == (2,)


def test_jit_traces_through_bounded_predictor_array_input():
    bp = BoundedPredictor(
        input_keys=("a", "b"),
        in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
        inner=_LinearPredictor(weights=jnp.eye(2)),
        out_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0)), transform="sigmoid"),
    )

    @jax.jit
    def call(predictor: BoundedPredictor, x: Array) -> Array:
        return predictor(x)

    out = call(bp, jnp.array([0.5, 0.3]))
    assert out.shape == (2,)
