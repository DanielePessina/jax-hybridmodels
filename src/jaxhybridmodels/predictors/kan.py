"""``KANPredictor``, wrapping a ``jaxkan`` Kolmogorov-Arnold network.

A Kolmogorov-Arnold network (KAN) puts the learnable nonlinearity on the
*edges*, as a basis expansion per connection, where an MLP uses a fixed
activation on the nodes and learns a weight matrix. The trainable content
of a KAN layer is therefore basis coefficients (``c_basis``, ``c_spl``,
``c_res``, ``bias``).

Static-vs-dynamic split
-----------------------
``jaxkan.models.KAN`` is a Flax NNX ``Module`` whose pytree leaves include
``PRNGKeyArray`` and ``uint32`` rng counters alongside the float
``nnx.Param`` arrays. ``eqx.tree_serialise_leaves`` refuses the typed
PRNG-key leaves, so the raw KAN model cannot be a dynamic field here.
Instead:

* Static fields hold the architecture (``in_size``, ``out_size``,
  ``hidden_widths``, ``grid_size``, ``basis``) and the integer ``seed``.
* The one dynamic field ``params`` is the ``nnx.State`` from
  ``nnx.split(model, nnx.Param, ...)``, float leaves only, with a
  seed-independent structure that round-trips through Equinox.
* The forward pass rebuilds the rng-bearing rest-state from ``seed`` and
  merges ``self.params`` back in, so the merged model is JIT-traceable.

This ties the wrapper to ``flax.nnx`` pytree internals through
``jaxkan``. The escape route, if jaxkan stops exposing ``Param``-based
filtering, is to reimplement KAN directly on ``equinox``.
"""

# ruff: noqa: F722

from __future__ import annotations

import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jaxkan.models.KAN import KAN as _JaxKAN
from jaxtyping import Array, Float

from jaxhybridmodels.predictors.base import Predictor

# Currently supported bases share the ``{k, G}`` parameter shape that
# jaxkan threads through its ``required_parameters`` dict. Other bases
# (rbf, chebyshev, ...) need a different parameter set and would require
# extending the wrapper to know which parameters each basis demands.
_SUPPORTED_BASES: tuple[str, ...] = ("spline", "base")
_SPLINE_ORDER_K: int = 3


def _seed_from_key(key: Array) -> int:
    """Derive a non-negative 31-bit Python ``int`` seed from a JAX PRNG key.

    ``jaxkan.models.KAN`` takes ``seed: int`` rather than a key, and this
    keeps the framework's ``key: Array`` contract. The ``int(...)`` forces a
    host sync, which is fine because construction never happens inside JIT.
    """
    return int(jr.randint(key, (), 0, 2**31 - 1))


def _build_kan(
    in_size: int,
    out_size: int,
    hidden_widths: tuple[int, ...],
    grid_size: int,
    basis: str,
    seed: int,
) -> Any:
    """Construct a ``jaxkan.models.KAN`` from the static architecture fields.

    Module-level rather than a method so the cache below can key on the
    fields alone. Keeps ``layer_dims`` and ``required_parameters`` in one
    place for both the constructor and the forward pass.
    """
    return _JaxKAN(
        layer_dims=[in_size, *hidden_widths, out_size],
        layer_type=basis,
        required_parameters={"k": _SPLINE_ORDER_K, "G": grid_size},
        seed=seed,
    )


@functools.cache
def _scaffold_parts(
    in_size: int,
    out_size: int,
    hidden_widths: tuple[int, ...],
    grid_size: int,
    basis: str,
    seed: int,
) -> tuple[Any, tuple[Any, ...]]:
    """Cached ``(graphdef, rest_states)`` for one KAN architecture.

    Keyed on the static fields alone, which are exactly what determines
    the graph. The result holds no trainable parameters;
    ``KANPredictor.__call__`` merges its own ``self.params`` back in.

    Cached because a KAN inside a vector field is traced once per solver
    stage, and rebuilding the jaxkan model in Python each time dominated
    trace cost. A run uses very few distinct architectures, so the
    unbounded cache is not a leak in practice.

    ``jax.ensure_compile_time_eval`` is load-bearing. The first call
    usually lands inside a trace, and without the context jaxkan's grid
    construction stages out into that trace, so the cache stores its
    tracers and the next trace raises ``UnexpectedTracerError``.
    """
    with jax.ensure_compile_time_eval():
        scaffold = _build_kan(in_size, out_size, hidden_widths, grid_size, basis, seed)
        graphdef, _params, *rest_states = nnx.split(scaffold, nnx.Param, ...)
        rest = tuple(jax.tree.map(jnp.asarray, rest_states))
    return graphdef, rest


class KANPredictor(Predictor):
    """Kolmogorov-Arnold network ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

    A KAN learns a basis expansion on each edge instead of a weight matrix
    per layer. Wraps ``jaxkan.models.KAN`` with a serialisation-clean
    static/dynamic split (module docstring has the reasoning): the one
    dynamic field is ``params``, everything else is static.

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
        The one dynamic field. The ``nnx.Param`` slice of the KAN's
        state, all float arrays. Trainable, and round-trips through
        ``eqx.tree_serialise_leaves``.
    """

    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    hidden_widths: tuple[int, ...] = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    basis: str = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    # ``nnx.State`` is generic over ``MutableMapping[K, V]``. The concrete
    # parameterisation is internal to flax-nnx and not part of this package's
    # public API, so the annotation stays at ``Any, Any``: it keeps the type
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

        ``key`` is required, not defaulted, and is used to derive the
        integer ``seed`` threaded into ``jaxkan.models.KAN``. The KAN is
        split immediately with ``nnx.split(model, nnx.Param, ...)``, so the
        non-serialisable rng-state never lands on this module.
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
        # rng-rest (uint32 counters and PRNGKeyArrays). Dropping the rest is
        # deliberate: it is regenerated on demand from the same seed.
        # nnx.split returns ``(GraphDef, State, *State)``, variadic in the
        # number of filters, hence the trailing star unpack.
        _, params, *_ = nnx.split(model, nnx.Param, ...)
        self.params = params

    def _build_model(self, seed: int) -> Any:
        """Construct a fresh ``jaxkan.models.KAN`` matching this predictor's static config."""
        return _build_kan(
            self.in_size,
            self.out_size,
            self.hidden_widths,
            self.grid_size,
            self.basis,
            seed,
        )

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        """Forward pass: ``[in_size] -> [out_size]``.

        jaxkan's KAN expects a leading batch axis, added and stripped here.
        The merged model takes its Param leaves from the trainable
        ``self.params`` and its rng-state from a deterministic rebuild, so
        the forward pass is a pure function of static config and params.

        The graph and its split come from :func:`_scaffold_parts`, cached on
        the static architecture. XLA folds them away either way, but the
        Python-level rebuild ran on every retrace before the cache existed.
        """
        graphdef, rest_states = _scaffold_parts(
            self.in_size,
            self.out_size,
            self.hidden_widths,
            self.grid_size,
            self.basis,
            self.seed,
        )
        merged = nnx.merge(graphdef, self.params, *rest_states)
        # Add/strip the batch axis required by jaxkan's per-layer matmuls.
        y = merged(x[None, :])
        return jnp.asarray(y[0])

    def initialized_with_key(self, key: Array) -> KANPredictor:
        """Return a same-architecture KANPredictor with freshly initialised parameters.

        The re-init protocol used by the tournament. Building a new
        ``KANPredictor`` rather than editing ``self.params`` keeps jaxkan's
        per-layer init in charge (truncated-normal spline weights, ones
        bias, identity residual), which leaf-level normal sampling would
        replace with a badly scaled scheme.
        """
        return KANPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            hidden_widths=self.hidden_widths,
            grid_size=self.grid_size,
            basis=self.basis,
            key=key,
        )

    def with_zero_final_head(self) -> KANPredictor:
        """Return a copy whose final KAN layer's parameters are all zero.

        The final head is the readout layer at index ``len(hidden_widths)``.
        Zeroing its four trainable arrays makes it produce
        ``zeros(out_size)`` for any input, which inside a
        ``BoundedPredictor`` maps to the midpoint of the physical box.

        Earlier layers keep their jaxkan-default init, so the feature
        transformation stays non-degenerate. Same intent as
        :meth:`MLPPredictor.with_zero_final_head` over a different
        parameterisation.
        """
        last_idx = len(self.hidden_widths)

        def _zero_if_in_last_layer(path: Any, leaf: Any) -> Any:
            # Path through ``nnx.State`` looks like
            # (DictKey('layers'), DictKey(<int>), GetAttrKey('value')), so
            # match on the second element being the final-layer index. The
            # ``getattr`` survives jax-tree internals wrapping paths
            # differently.
            if len(path) >= 2:
                k0 = getattr(path[0], "key", None)
                k1 = getattr(path[1], "key", None)
                if k0 == "layers" and k1 == last_idx:
                    return jnp.zeros_like(leaf)
            return leaf

        new_params = jax.tree_util.tree_map_with_path(_zero_if_in_last_layer, self.params)
        return eqx.tree_at(lambda kp: kp.params, self, new_params)
