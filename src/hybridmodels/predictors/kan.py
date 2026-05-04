"""Concrete ``KANPredictor`` wrapping a ``jaxkan`` Kolmogorov-Arnold network.

A KAN predictor differs from an MLP in that the learnable nonlinearities live
on the *edges* (parameterised as splines / radial bases / orthogonal polynomial
expansions) rather than as fixed activations on the *nodes*. The trainable
content per layer is the set of basis-function coefficients (``c_basis``,
``c_spl``, ``c_res``, ``bias``), not weight matrices.

Static-vs-dynamic split
-----------------------
``jaxkan.models.KAN`` is a Flax NNX ``Module``: when flattened as a JAX
pytree its leaves include ``PRNGKeyArray`` and ``uint32`` rng-counter
scalars alongside the float ``nnx.Param`` arrays.
``eqx.tree_serialise_leaves`` refuses to ``np.save`` the typed PRNG-key
leaves, so the raw KAN model cannot be exposed as a dynamic field on
this predictor — the binary checkpoint must contain only float arrays.

The workaround:

* Static fields hold the architecture (``in_size``, ``out_size``,
  ``hidden_widths``, ``grid_size``, ``basis``) and the integer ``seed``
  used to build the underlying jaxkan model.
* The single dynamic field ``params`` is the ``nnx.State`` returned by
  ``nnx.split(model, nnx.Param, ...)``: it contains *only* float Param
  leaves with shapes determined by the static config, so it is
  identical-structured across different seeds and round-trips
  cleanly through Equinox's leaf serialisation.
* The forward pass rebuilds the rng-bearing rest-state of the KAN from
  ``seed`` (deterministic) and merges in ``self.params`` via
  ``nnx.merge``. Every per-call construction is driven by static
  values plus the dynamic ``params`` state, so the merged model is
  JIT-traceable.

This couples us to ``flax.nnx`` pytree internals via ``jaxkan``. A
future jaxkan release that stops exposing ``Param``-based filtering
would force the wrapper to reach into individual layers; a leaner
alternative path is to re-implement KAN directly on top of
``equinox``, which would remove the static/dynamic split entirely.
"""

# ruff: noqa: F722

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jaxkan.models.KAN import KAN as _JaxKAN
from jaxtyping import Array, Float

from hybridmodels.predictors.base import Predictor

# Currently supported bases share the ``{k, G}`` parameter shape that
# jaxkan threads through its ``required_parameters`` dict. Other bases
# (rbf, chebyshev, ...) need a different parameter set and would require
# extending the wrapper to know which parameters each basis demands.
_SUPPORTED_BASES: tuple[str, ...] = ("spline", "base")
_SPLINE_ORDER_K: int = 3


def _seed_from_key(key: Array) -> int:
    """Derive a non-negative 31-bit Python ``int`` seed from a JAX PRNG key.

    ``jaxkan.models.KAN`` takes ``seed: int`` rather than a key; this helper
    closes the gap so the framework's user-facing ``key: Array`` contract is
    preserved. The conversion is host-side (``int(...)`` forces a sync) which
    is acceptable because predictor construction never happens inside JIT.
    """
    return int(jr.randint(key, (), 0, 2**31 - 1))


class KANPredictor(Predictor):
    """Kolmogorov-Arnold network predictor: ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

    Wraps ``jaxkan.models.KAN`` with a serialisation-clean static/dynamic split
    (see module docstring for the full rationale). The trainable content is
    exposed as a single ``params`` ``nnx.State`` field whose leaves are float
    Param arrays; everything else (architecture, rng seed, basis kind) is
    static.

    Attributes
    ----------
    in_size, out_size : int
        Input / output dimensionality. Static; pinned by ``layer_dims`` in the
        underlying KAN.
    hidden_widths : tuple[int, ...]
        Widths of the hidden KAN layers in order. Static; combined with
        ``in_size`` and ``out_size`` to form ``layer_dims = [in_size,
        *hidden_widths, out_size]``.
    grid_size : int
        Number of spline grid intervals (``G``). Static; passed as
        ``required_parameters['G']`` to jaxkan.
    basis : str
        Layer-type code, one of ``"spline"`` or ``"base"``. Static.
    seed : int
        Integer seed used to build the underlying jaxkan model and to
        recreate its rng-state on demand inside ``__call__``. Derived from
        the user-supplied ``key`` at construction; static thereafter.
    params : nnx.State
        Dynamic field — the ``nnx.Param`` slice of the KAN's state, all
        float arrays. Trainable; serialised round-trip via
        ``eqx.tree_serialise_leaves``.
    """

    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    hidden_widths: tuple[int, ...] = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    basis: str = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    # ``nnx.State`` is generic over ``MutableMapping[K, V]``; the concrete
    # parameterisation is internal to flax-nnx and not part of our public
    # surface, so the annotation here uses ``Any, Any`` to keep the type
    # checker quiet without leaking flax internals.
    params: nnx.State[Any, Any]

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        hidden_widths: tuple[int, ...] = (8,),
        grid_size: int = 5,
        basis: str = "spline",
        key: Array,
    ) -> None:
        """Build the KAN, split off its Param state, and pin the rest as static config.

        ``key`` is required (the framework refuses silent default keys
        for reproducibility) and used to derive the integer ``seed``
        threaded into ``jaxkan.models.KAN``. The constructed KAN is
        split immediately via ``nnx.split(model, nnx.Param, ...)`` so
        the non-serialisable rng-state never lands on this module — only
        the float Param leaves are kept as the dynamic ``params`` field.
        """
        if basis not in _SUPPORTED_BASES:
            raise ValueError(f"Basis {basis!r} not supported. Available: {list(_SUPPORTED_BASES)}")
        self.in_size = int(in_size)
        self.out_size = int(out_size)
        self.hidden_widths = tuple(int(w) for w in hidden_widths)
        self.grid_size = int(grid_size)
        self.basis = basis
        self.seed = _seed_from_key(key)

        model = self._build_model(self.seed)
        # Filter the model's state into Params (trainable, float-only) and the
        # rng-rest (uint32 counters + PRNGKeyArrays). We intentionally drop
        # the rest here — it is regenerated on demand from the same seed.
        # nnx.split returns ``(GraphDef, State, *State)`` (variadic in the
        # number of filters), hence the trailing star unpack.
        _, params, *_ = nnx.split(model, nnx.Param, ...)
        self.params = params

    def _build_model(self, seed: int) -> Any:
        """Construct a fresh ``jaxkan.models.KAN`` matching this predictor's static config.

        Used both at ``__init__`` (to obtain the initial Param state) and in
        ``__call__`` (to recover the rng-state needed for ``nnx.merge``).
        Centralising the construction here keeps ``layer_dims`` and
        ``required_parameters`` in one place.
        """
        layer_dims = [self.in_size, *self.hidden_widths, self.out_size]
        required_parameters = {"k": _SPLINE_ORDER_K, "G": self.grid_size}
        return _JaxKAN(
            layer_dims=layer_dims,
            layer_type=self.basis,
            required_parameters=required_parameters,
            seed=seed,
        )

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        """Forward pass: ``[in_size] -> [out_size]``.

        jaxkan's KAN expects a leading batch axis; we add and strip it around
        the call. The merge pattern (build fresh -> split -> swap params)
        gives us a usable model whose Param leaves are the trainable
        ``self.params`` while the rng-state comes from a deterministic
        rebuild (so the forward pass is a pure function of static config and
        dynamic params).
        """
        scaffold = self._build_model(self.seed)
        graphdef, _, *rest_states = nnx.split(scaffold, nnx.Param, ...)
        merged = nnx.merge(graphdef, self.params, *rest_states)
        # Add/strip the batch axis required by jaxkan's per-layer matmuls.
        y = merged(x[None, :])
        return jnp.asarray(y[0])

    def initialized_with_key(self, key: Array) -> KANPredictor:
        """Return a same-architecture KANPredictor with freshly initialised parameters.

        Implements the re-init protocol used by the training tournament
        loop. Building a whole new ``KANPredictor`` (rather than
        tweaking ``self.params`` in place) lets jaxkan's per-layer init
        logic — truncated-normal spline weights, ones bias, identity
        residual — drive the initialisation instead of replacing it with
        leaf-level standard-normal samples that would skew the
        distribution.
        """
        return KANPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            hidden_widths=self.hidden_widths,
            grid_size=self.grid_size,
            basis=self.basis,
            key=key,
        )
