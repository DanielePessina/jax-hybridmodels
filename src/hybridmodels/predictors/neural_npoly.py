"""``NeuralNPolynomial``: composition of a coefficient network and exponents.

Experimental, and not re-exported from ``hybridmodels.predictors``.

A ``NeuralNPolynomial`` predictor evaluates one scalar polynomial per
output channel. Its *coefficients* come from a trainable inner
``Predictor`` (an MLP, a KAN, anything). The package's clearest example
of composition over inheritance: the trainable component is a field
rather than a base class, and any concrete ``Predictor`` can be the
coefficient network as long as its output width matches the coefficient
layout below.

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
* ``sum(x)`` collapses the input to the single scalar that every
  per-channel polynomial is evaluated at. It is the simplest choice that
  matches the reading "one scalar polynomial per output, coefficients
  are functions of x". A rotation-invariant alternative such as
  ``norm(x)`` would drop straight in if a domain wanted it. Only one is
  committed to here, to keep the contract simple.
* ``coeff_net`` is stored as given. The constructor does **not** key-init
  it. The user builds whichever inner predictor they want, however they
  prefer to build it, and passes it in. This class stays ignorant of the
  inner predictor's construction, and the inner network owns the whole
  trainable-leaves story.
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

    Pure composition wrapper. The trainable piece is the inner
    ``coeff_net`` Predictor. ``exponents``, ``in_size``, and ``out_size``
    are static metadata that survive serialisation and reconstruction,
    and only the inner network's float leaves reach the binary
    checkpoint.

    Static vs dynamic split
    -----------------------
    Dynamic (PyTree leaves):
        ``coeff_net``, whose own inexact-array leaves are the trainable
        weights that produce the coefficients for a given input.
    Static fields:
        ``exponents``, the polynomial exponents (``(0., 1., 2.)`` for a
                       quadratic). Must be non-empty.
        ``in_size``, the input width. Metadata for the user, and pinned
                       so composition wrappers can cross-check it.
        ``out_size``, the output width. One polynomial is evaluated per
                       output channel.

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

        ``coeff_net.out_size`` must equal ``out_size * len(exponents)``, so
        that the call-time reshape into ``[out_size, len(exponents)]`` is
        exact. A mismatch is an architectural error and raises
        ``ValueError``. Otherwise the inner network is stored verbatim,
        with no re-initialisation and no copy.
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
        ``basis = sum(x)``. ``exponents`` is converted to a JAX array on
        every call, but it is a static tuple, so JAX folds the conversion
        at trace time and the array becomes a compile-time constant.
        """
        coeffs = jnp.asarray(self.coeff_net(x)).reshape((self.out_size, len(self.exponents)))
        basis = jnp.sum(x)
        powers = basis ** jnp.asarray(self.exponents)
        return jnp.einsum("od,d->o", coeffs, powers)

    def initialized_with_key(self, key: Array) -> NeuralNPolynomial:
        """Re-initialise the inner ``coeff_net``; keep the polynomial structure.

        Implements the re-init protocol consumed by the training
        tournament. The delegation order matters. Prefer the inner
        predictor's own ``initialized_with_key``, so that
        ``MLPPredictor`` re-runs Equinox's LeCun-uniform init rather than
        having its weights resampled elementwise; fall back to
        elementwise sampling only when the inner predictor offers no
        scheme. :func:`reinitialize_with_key` already encodes that
        order, so this calls straight through. ``exponents``, ``in_size``
        and ``out_size`` are static and unchanged.
        """
        new_coeff_net = cast(Predictor, reinitialize_with_key(self.coeff_net, key))
        return NeuralNPolynomial(
            coeff_net=new_coeff_net,
            exponents=self.exponents,
            in_size=self.in_size,
            out_size=self.out_size,
        )
