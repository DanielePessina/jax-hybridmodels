"""``NeuralNPolynomial``: composition of a coefficient network and exponents.

Experimental, and not re-exported from ``hybridmodels.predictors``.

A ``NeuralNPolynomial`` predictor evaluates one scalar polynomial per
output channel. Its *coefficients* come from a trainable inner
``Predictor`` held as a field, so any concrete ``Predictor`` can be the
coefficient network as long as its output width matches the layout below.

Mathematical contract
---------------------
For ``x: [in_size]``::

    coeffs = coeff_net(x).reshape((out_size, len(exponents)))   # [O, D]
    basis  = sum(x)                                             # scalar
    powers = basis ** asarray(exponents)                        # [D]
    out    = einsum("od,d->o", coeffs, powers)                  # [O]

Per output channel ``o`` this reads as a scalar polynomial in ``basis``
whose ``D`` coefficients are nonlinear functions of the full ``x``
vector produced by the inner network.

Design choices
--------------
* ``sum(x)`` collapses the input to the single scalar each per-channel
  polynomial is evaluated at. An alternative such as ``norm(x)`` would
  drop straight in; only one is committed to, to keep the contract simple.
* ``coeff_net`` is stored as given, never key-initialised here. The inner
  network owns the whole trainable-leaves story.
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

    Pure composition wrapper. Only the inner ``coeff_net`` is trainable;
    ``exponents`` (non-empty), ``in_size`` and ``out_size`` are static
    metadata, so only the inner network's float leaves reach the binary
    checkpoint.

    Coefficient layout
    ------------------
    ``coeff_net(x)`` must produce a flat ``[out_size * len(exponents)]``
    vector, validated at construction. ``__call__`` reshapes it row-major
    to ``[out_size, len(exponents)]``, so row ``o`` holds the coefficients
    for output channel ``o``.

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
        the call-time reshape is exact; a mismatch raises. The inner network
        is stored verbatim, with no re-initialisation and no copy.
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

        Returns ``out[o] = sum_d coeffs[o, d] * basis ** exponents[d]``,
        where ``coeffs`` is the row-major reshape of ``coeff_net(x)`` and
        ``basis = sum(x)``. ``exponents`` is a static tuple, so its
        per-call array conversion folds into a compile-time constant.
        """
        coeffs = jnp.asarray(self.coeff_net(x)).reshape((self.out_size, len(self.exponents)))
        basis = jnp.sum(x)
        powers = basis ** jnp.asarray(self.exponents)
        return jnp.einsum("od,d->o", coeffs, powers)

    def initialized_with_key(self, key: Array) -> NeuralNPolynomial:
        """Re-initialise the inner ``coeff_net``; keep the polynomial structure.

        Calls straight through to :func:`reinitialize_with_key`, which
        prefers the inner predictor's own ``initialized_with_key`` and falls
        back to elementwise sampling only when it offers no scheme.
        ``exponents``, ``in_size`` and ``out_size`` are static.
        """
        new_coeff_net = cast(Predictor, reinitialize_with_key(self.coeff_net, key))
        return NeuralNPolynomial(
            coeff_net=new_coeff_net,
            exponents=self.exponents,
            in_size=self.in_size,
            out_size=self.out_size,
        )
