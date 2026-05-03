"""``NeuralNPolynomial``: composition of a coefficient network and exponents.

This file owns the ``NeuralNPolynomial`` predictor (SPEC §5.2; build-plan
Phase 14). It is the canonical example of *composition* over inheritance for
a Predictor: the trainable component is a wrapped inner ``Predictor`` (any
concrete one — MLP, KAN, ...) whose output is reshaped into a
``[out_size, len(exponents)]`` coefficient matrix and contracted with
powers of a single scalar basis derived from the input.

Mathematical contract
---------------------
For ``x: [in_size]``::

    coeffs = coeff_net(x).reshape((out_size, len(exponents)))   # [O, D]
    basis  = sum(x)                                             # scalar
    powers = basis ** asarray(exponents)                        # [D]
    out    = einsum("od,d->o", coeffs, powers)                  # [O]

Per output channel ``o`` this reads as a scalar polynomial in ``basis``
whose ``D`` coefficients are nonlinear functions of the full ``x`` vector
produced by the inner network.

Design choices (Option A from the build-plan brief):
  - ``input_to_basis = sum(x)`` collapses the input to one scalar that
    drives every polynomial. Mirrors the source-package
    ``NeuralNPolynomialRegressor`` conception of "one scalar polynomial
    per output, coefficients are functions of x". Option B
    (``norm(x)`` for rotation invariance) was considered and rejected
    for v1; revisit if a domain calls for it.
  - ``coeff_net`` is consumed as-is; the constructor does **not** key-init
    it. The user builds whichever inner predictor they want and passes
    it in. This keeps the class agnostic to the inner predictor's
    construction conventions and makes the trainable-leaves story
    completely owned by the inner network.
"""

# ruff: noqa: F722

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from hybridmodels.predictors.base import Predictor, reinitialize_with_key


class NeuralNPolynomial(Predictor):
    """Polynomial-in-``sum(x)`` whose coefficients come from ``coeff_net``.

    Pure composition wrapper. The trainable piece is the inner ``coeff_net``
    Predictor; ``exponents``, ``in_size``, and ``out_size`` are static
    metadata that survive serialisation and reconstruction (R-A5).

    Static vs dynamic split
    -----------------------
    Dynamic (PyTree leaves):
        ``coeff_net`` — its own inexact-array leaves are the trainable
        coefficients (modulated per-input by the network).
    Static fields:
        ``exponents`` — tuple of polynomial exponents (e.g. ``(0., 1., 2.)``
                       for a quadratic). Must be non-empty.
        ``in_size``  — input dimension; metadata for the user, also pinned
                       so it can be cross-checked by composition wrappers.
        ``out_size`` — output dimension; the polynomial is evaluated
                       independently per output channel.

    Coefficient layout
    ------------------
    ``coeff_net(x)`` must produce a flat ``[out_size * len(exponents)]``
    vector; the constructor enforces this contract by validating
    ``coeff_net.out_size`` at construction time. Inside ``__call__`` the
    flat output is reshaped row-major to ``[out_size, len(exponents)]``
    so that row ``o`` is the coefficient vector for output channel ``o``.

    Example
    -------
    A 3-term quadratic with two output channels backed by an MLP::

        coeff_net = MLPPredictor(in_size=3, out_size=6, width_size=8,
                                 depth=2, activation_name="tanh", key=key)
        npoly = NeuralNPolynomial(coeff_net=coeff_net,
                                  exponents=(0.0, 1.0, 2.0),
                                  in_size=3, out_size=2)
    """

    coeff_net: Predictor
    exponents: tuple[float, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        coeff_net: Predictor,
        exponents: tuple[float, ...],
        in_size: int,
        out_size: int,
    ) -> None:
        """Validate the shape contract and store hyperparameters.

        ``coeff_net.out_size`` must equal ``out_size * len(exponents)`` so
        that the call-time reshape into ``[out_size, len(exponents)]`` is
        exact. A mismatch is a static (architectural) error and raises
        ``ValueError``; the inner network is otherwise stored verbatim
        (no re-initialisation, no copy).
        """
        if not exponents:
            raise ValueError("exponents must contain at least one entry.")

        expected_out = int(out_size) * len(exponents)
        actual_out = getattr(coeff_net, "out_size", None)
        if actual_out is None or int(actual_out) != expected_out:
            raise ValueError(
                "coeff_net.out_size must equal out_size * len(exponents) = "
                f"{int(out_size)} * {len(exponents)} = {expected_out}; "
                f"got coeff_net.out_size={actual_out!r}."
            )

        self.coeff_net = coeff_net
        self.exponents = tuple(float(e) for e in exponents)
        self.in_size = int(in_size)
        self.out_size = int(out_size)

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        """Evaluate the polynomial-in-``sum(x)`` per output channel.

        Returns ``out[o] = sum_d coeffs[o, d] * basis ** exponents[d]`` where
        ``coeffs`` is the row-major reshape of ``coeff_net(x)`` and
        ``basis = sum(x)``. ``exponents`` is converted to a JAX array each
        call but, since it is a static tuple, JAX folds the conversion at
        trace time and the array becomes a constant in the compiled program.
        """
        coeffs = jnp.asarray(self.coeff_net(x)).reshape(
            (self.out_size, len(self.exponents))
        )
        basis = jnp.sum(x)
        powers = basis ** jnp.asarray(self.exponents)
        return jnp.einsum("od,d->o", coeffs, powers)

    def initialized_with_key(self, key: Array) -> NeuralNPolynomial:
        """Re-initialise the inner ``coeff_net``; keep the polynomial structure.

        Implements the R-T8 protocol consumed by the optax tournament.
        Delegation order matches what the prompt pins: prefer the inner
        predictor's own ``initialized_with_key`` (so e.g. ``MLPPredictor``
        re-runs Equinox's Glorot init), and otherwise fall back to
        ``reinitialize_with_key`` which samples replacement leaves
        elementwise. The free function ``reinitialize_with_key`` already
        encodes this delegation, so we just call through. ``exponents``,
        ``in_size`` and ``out_size`` are static and unchanged.
        """
        new_coeff_net = cast(Predictor, reinitialize_with_key(self.coeff_net, key))
        return NeuralNPolynomial(
            coeff_net=new_coeff_net,
            exponents=self.exponents,
            in_size=self.in_size,
            out_size=self.out_size,
        )
