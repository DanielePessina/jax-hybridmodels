"""``KANPredictor``, wrapping a ``jaxkan`` Kolmogorov-Arnold network.

A Kolmogorov-Arnold network (KAN) puts the learnable nonlinearity on the
*edges* of the network, as a spline or other basis expansion per
connection, where an MLP uses a fixed activation on the *nodes* and
learns a weight matrix. So the trainable content of a KAN layer is a set
of basis coefficients (``c_basis``, ``c_spl``, ``c_res``, ``bias``)
rather than a weight matrix.

Static-vs-dynamic split
-----------------------
``jaxkan.models.KAN`` is a Flax NNX ``Module``. Flattened as a JAX
pytree, its leaves include ``PRNGKeyArray`` and ``uint32`` rng-counter
scalars alongside the float ``nnx.Param`` arrays.
``eqx.tree_serialise_leaves`` refuses to ``np.save`` the typed PRNG-key
leaves, and the binary checkpoint must hold only float arrays, so the
raw KAN model cannot be a dynamic field on this predictor.

The workaround:

* Static fields hold the architecture (``in_size``, ``out_size``,
  ``hidden_widths``, ``grid_size``, ``basis``) and the integer ``seed``
  used to build the underlying jaxkan model.
* The single dynamic field ``params`` is the ``nnx.State`` returned by
  ``nnx.split(model, nnx.Param, ...)``. It holds *only* float Param
  leaves whose shapes come from the static config, so its structure is
  the same across seeds and it round-trips through Equinox's leaf
  serialisation.
* The forward pass rebuilds the rng-bearing rest-state of the KAN from
  ``seed``, deterministically, and merges ``self.params`` back in with
  ``nnx.merge``. Every per-call construction is driven by static values
  plus the dynamic ``params``, so the merged model is JIT-traceable.

This ties the wrapper to ``flax.nnx`` pytree internals through
``jaxkan``. A future jaxkan release that stops exposing ``Param``-based
filtering would force this code to reach into individual layers. The
leaner escape route is to reimplement KAN directly on ``equinox``, which
removes the static/dynamic split altogether.
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
    the graph, so the cache can never hand back a mismatched one. The
    result holds no trainable parameters. ``nnx.split`` peels those off
    and ``KANPredictor.__call__`` merges its own ``self.params`` back in.

    Cached because a KAN evaluated inside a vector field is traced once
    per solver stage, and rebuilding the jaxkan model in Python each time
    dominated trace cost. The cached values are architecture-shaped and
    small, and a run uses very few distinct architectures, so the
    unbounded cache is not a leak in practice.

    ``jax.ensure_compile_time_eval`` is load-bearing, not an
    optimisation. The first call usually lands *inside* a trace, because
    the first thing a program does with a KAN is evaluate it under
    ``jit``. Without the context, jaxkan's grid construction stages out
    into that trace and the cache stores tracers belonging to it. The
    next trace merges those tracers and JAX raises
    ``UnexpectedTracerError``. Forcing eager evaluation makes the cached
    values concrete arrays, which is what everything downstream assumes.
    """
    with jax.ensure_compile_time_eval():
        scaffold = _build_kan(in_size, out_size, hidden_widths, grid_size, basis, seed)
        graphdef, _params, *rest_states = nnx.split(scaffold, nnx.Param, ...)
        rest = tuple(jax.tree.map(jnp.asarray, rest_states))
    return graphdef, rest


class KANPredictor(Predictor):
    """Kolmogorov-Arnold network ``Float[Array, "in_size"] -> Float[Array, "out_size"]``.

    A KAN learns a basis expansion on each edge instead of a weight
    matrix per layer. Wraps ``jaxkan.models.KAN`` with a
    serialisation-clean static/dynamic split; the module docstring has
    the full reasoning. The trainable content is one ``params``
    ``nnx.State`` field whose leaves are float Param arrays. Everything
    else (architecture, rng seed, basis kind) is static.

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

        ``key`` is required (the framework refuses silent default keys, so
        reproducibility never rests on a hidden global RNG) and is used to
        derive the integer ``seed`` threaded into ``jaxkan.models.KAN``.
        The constructed KAN is split immediately with
        ``nnx.split(model, nnx.Param, ...)``, so the non-serialisable
        rng-state never lands on this module. Only the float Param leaves
        are kept, as the dynamic ``params`` field.
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

        jaxkan's KAN expects a leading batch axis, so this adds and strips
        one around the call. The merge pattern (build fresh, split, swap
        params) yields a usable model whose Param leaves are the trainable
        ``self.params`` while the rng-state comes from a deterministic
        rebuild. The forward pass is therefore a pure function of the
        static config and the dynamic params.

        The graph and its split come from :func:`_scaffold_parts`, cached
        on the static architecture. They are trace-time constants either
        way, so XLA folds them away, but the Python-level jaxkan
        construction and ``nnx.split`` ran on every retrace before the
        cache existed. A KAN called from inside a vector field is traced
        once per solver stage, which made that a real compile-time cost.
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

        Implements the re-init protocol used by the training tournament
        loop. Building a whole new ``KANPredictor``, rather than editing
        ``self.params`` in place, keeps jaxkan's per-layer init logic in
        charge (truncated-normal spline weights, ones bias, identity
        residual). Leaf-level standard-normal samples would replace that
        scheme with a badly scaled one.
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

        For a KAN, the "final head" is the readout layer at index
        ``len(hidden_widths)`` in the underlying ``layer_dims`` list. Each
        layer carries four trainable arrays (``c_basis``, ``c_spl``,
        ``c_res``, ``bias``); zeroing all of them makes the layer produce
        ``zeros(out_size)`` regardless of input, which composed inside a
        ``BoundedPredictor`` maps via ``out_scaler.from_latent(0)`` to the
        midpoint of the physical bound box.

        The earlier KAN layers keep their jaxkan-default init, so the
        input feature transformation stays non-degenerate. Only the
        readout is locked. Same intent as
        :meth:`MLPPredictor.with_zero_final_head`, a seed-independent
        initial physical output, over a different parameterisation.
        """
        last_idx = len(self.hidden_widths)

        def _zero_if_in_last_layer(path: Any, leaf: Any) -> Any:
            # Path through ``nnx.State`` looks like
            # (DictKey('layers'), DictKey(<int>), GetAttrKey('value')), so
            # match on the second element being the final-layer index.
            # ``DictKey`` exposes its key via ``.key``; the defensive
            # ``getattr`` survives future jax-tree internals that wrap path
            # entries differently.
            if len(path) >= 2:
                k0 = getattr(path[0], "key", None)
                k1 = getattr(path[1], "key", None)
                if k0 == "layers" and k1 == last_idx:
                    return jnp.zeros_like(leaf)
            return leaf

        new_params = jax.tree_util.tree_map_with_path(_zero_if_in_last_layer, self.params)
        return eqx.tree_at(lambda kp: kp.params, self, new_params)
