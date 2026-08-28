# ruff: noqa: F722

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import Array, Float

from hybridmodels.predictors import (
    BoundedPredictor,
    BoundScaler,
    MLPPredictor,
    Predictor,
)

# ``NeuralNPolynomial`` is not part of the public predictor surface yet
# (the file is kept in-tree as a future candidate; see
# ``hybridmodels/predictors/__init__.py``). We import it directly from
# the submodule so these tests can still pin the implementation's
# contract while it lives outside the public API.
from hybridmodels.predictors.neural_npoly import NeuralNPolynomial


class _ConstantPredictor(Predictor):
    """Test fixture: returns ``constant`` regardless of input.

    Lets the math-correctness test pin the output of ``NeuralNPolynomial``
    to the analytic formula without depending on the inner network's
    activation/initialisation.
    """

    constant: Float[Array, " out_size"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        del x
        return self.constant


def _array_leaves(predictor: eqx.Module) -> list:
    arrays, _ = eqx.partition(predictor, eqx.is_array)
    return jtu.tree_leaves(arrays)


def _make_npoly(
    *,
    in_size: int = 3,
    out_size: int = 2,
    exponents: tuple[float, ...] = (0.0, 1.0, 2.0),
    width_size: int = 8,
    depth: int = 2,
    seed: int = 0,
) -> NeuralNPolynomial:
    coeff_net = MLPPredictor(
        in_size=in_size,
        out_size=out_size * len(exponents),
        width_size=width_size,
        depth=depth,
        activation_name="tanh",
        key=jr.PRNGKey(seed),
    )
    return NeuralNPolynomial(
        coeff_net=coeff_net,
        exponents=exponents,
        in_size=in_size,
        out_size=out_size,
    )


class TestConstructionGuard:
    def test_mismatched_coeff_net_out_size_raises(self):
        # coeff_net.out_size = 5 but out_size * len(exponents) = 2 * 3 = 6.
        coeff_net = MLPPredictor(
            in_size=3,
            out_size=5,
            width_size=8,
            depth=2,
            activation_name="tanh",
            key=jr.PRNGKey(0),
        )
        with pytest.raises(ValueError, match="out_size"):
            NeuralNPolynomial(
                coeff_net=coeff_net,
                exponents=(0.0, 1.0, 2.0),
                in_size=3,
                out_size=2,
            )

    def test_matched_coeff_net_constructs(self):
        npoly = _make_npoly()
        assert npoly.in_size == 3
        assert npoly.out_size == 2
        assert npoly.exponents == (0.0, 1.0, 2.0)


class TestForwardShape:
    def test_returns_out_size_vector(self):
        npoly = _make_npoly(in_size=3, out_size=2, exponents=(0.0, 1.0, 2.0))
        out = npoly(jnp.array([0.5, -0.2, 0.1]))
        assert out.shape == (2,)

    def test_single_exponent(self):
        npoly = _make_npoly(in_size=2, out_size=3, exponents=(1.0,))
        out = npoly(jnp.zeros((2,)))
        assert out.shape == (3,)


class TestMathCorrectness:
    def test_matches_analytic_polynomial(self):
        out_size = 2
        exponents = (0.0, 1.0, 2.0)
        in_size = 3
        # Constant inner predictor isolates the polynomial evaluation from
        # any nonlinearity in the coefficient network.
        constant = jnp.arange(out_size * len(exponents), dtype=jnp.float32)
        coeff_net = _ConstantPredictor(
            constant=constant,
            in_size=in_size,
            out_size=out_size * len(exponents),
        )
        npoly = NeuralNPolynomial(
            coeff_net=coeff_net,
            exponents=exponents,
            in_size=in_size,
            out_size=out_size,
        )
        x = jnp.array([0.5, 1.5, 2.0])
        coeffs = constant.reshape((out_size, len(exponents)))
        basis = jnp.sum(x)
        powers = basis ** jnp.asarray(exponents)
        expected = jnp.einsum("od,d->o", coeffs, powers)
        actual = npoly(x)
        assert actual.shape == expected.shape
        assert jnp.allclose(actual, expected, atol=1e-6)


class TestJitCompatibility:
    def test_jits_without_error(self):
        npoly = _make_npoly()
        out = eqx.filter_jit(lambda p, x: p(x))(npoly, jnp.ones((3,)))
        assert out.shape == (2,)


class TestBoundedPredictorComposition:
    def test_runs_inside_bounded_predictor(self):
        in_keys = ("a", "b")
        coeff_net = MLPPredictor(
            in_size=len(in_keys),
            out_size=2 * 3,
            width_size=4,
            depth=1,
            activation_name="tanh",
            key=jr.PRNGKey(0),
        )
        npoly = NeuralNPolynomial(
            coeff_net=coeff_net,
            exponents=(0.0, 1.0, 2.0),
            in_size=len(in_keys),
            out_size=2,
        )
        bp = BoundedPredictor(
            input_keys=in_keys,
            in_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 1.0))),
            inner=npoly,
            out_scaler=BoundScaler(bounds=((0.0, 1.0), (0.0, 2.0))),
        )
        covariates = {"a": jnp.array(0.3), "b": jnp.array(0.7)}
        out = bp(covariates)
        assert out.shape == (2,)
        assert jnp.all(jnp.isfinite(out))


class TestInitializedWithKey:
    def test_preserves_static_fields(self):
        npoly = _make_npoly()
        fresh = npoly.initialized_with_key(jr.PRNGKey(11))
        assert fresh.in_size == npoly.in_size
        assert fresh.out_size == npoly.out_size
        assert fresh.exponents == npoly.exponents
        assert type(fresh.coeff_net) is type(npoly.coeff_net)
        assert fresh.coeff_net.in_size == npoly.coeff_net.in_size
        assert fresh.coeff_net.out_size == npoly.coeff_net.out_size

    def test_changes_inexact_leaves(self):
        npoly = _make_npoly(seed=0)
        fresh = npoly.initialized_with_key(jr.PRNGKey(11))
        leaves_old = _array_leaves(npoly)
        leaves_new = _array_leaves(fresh)
        assert len(leaves_old) == len(leaves_new)
        assert any(
            not jnp.array_equal(lo, ln) for lo, ln in zip(leaves_old, leaves_new, strict=True)
        )


class TestZeroExponentNonNaN:
    def test_zero_exponent_at_zero_basis_is_finite(self):
        # x sums to zero so basis = 0; exponent tuple includes 0.0 so the
        # polynomial expansion contains the term ``0 ** 0`` which JAX
        # evaluates to 1 (IEEE convention). The output must remain finite.
        npoly = _make_npoly(in_size=3, out_size=2, exponents=(0.0, 1.0, 2.0))
        x = jnp.zeros((3,))
        out = npoly(x)
        assert jnp.all(jnp.isfinite(out))


def test_is_in_public_export() -> None:
    """``NeuralNPolynomial`` is a supported public predictor family.

    It lives in ``hybridmodels.predictors`` and the top-level namespace,
    alongside ``MLPPredictor`` and ``KANPredictor``. Pin the export so a
    future removal is a deliberate edit, not an accidental leak.
    """
    import hybridmodels

    assert hasattr(hybridmodels, "NeuralNPolynomial")
    assert "NeuralNPolynomial" in hybridmodels.__all__
