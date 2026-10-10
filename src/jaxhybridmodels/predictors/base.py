"""Predictors and the wrappers that compose them.

A *predictor* is a trainable ``eqx.Module`` whose ``__call__`` maps one
array to another. That is the whole contract. A predictor knows nothing
about covariate names, physical units, or bounds.

Everything else is composition rather than inheritance. ``Predictor``
is an abstract marker, concrete predictors (``MLPPredictor``,
``KANPredictor``, ...) are final, and shared concerns live in wrappers
that hold a ``Predictor`` as a field. ``BoundScaler`` maps a physical
range onto an unbounded latent. ``BoundedPredictor`` chains an input
scaler, an inner predictor, and an output scaler. Adding a predictor
family means subclassing ``Predictor`` and writing ``__call__``. No
method is ever overridden.

Named-input ordering lives on ``BoundedPredictor.input_keys``, a static
``tuple[str, ...]``. The keys travel through
``eqx.tree_serialise_leaves`` as static metadata, so a reload site can
read which inputs a saved predictor expects, and in which order, without
consulting the code that built it. ``__call__`` accepts either a
``dict[str, Array]`` (subset extraction in ``input_keys`` order) or a
rank-1 ``Array``.

Multi-rate models (simultaneous nucleation and growth, say) compose as a
*tuple* of predictors the user unpacks at the top of their
``simulate_fn``. There is deliberately no framework "rate-pair" class:
unpacking names each predictor's role in user code, and adding a third
rate is one more tuple entry.

The trainable component handed to training kernels is therefore a
``PyTree[eqx.Module]``: a tuple by convention, but any pytree shape
works. :func:`reinitialize_pytree_with_key` walks that pytree and gives
each ``eqx.Module`` leaf an independent subkey, so identical-shape
siblings get different fresh weights when the tournament restarts a run.
"""

# ruff: noqa: F722

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.typing import ArrayLike
from jaxtyping import Array

from jaxhybridmodels.penalties import soft_inverse
from jaxhybridmodels.transforms import BOUND_TRANSFORMS, WARPS, check_bounds, warp_bounds

# What this module exports.
__all__ = (
    "Predictor",
    "BoundScaler",
    "BoundedPredictor",
    "Constant",
    "Parameter",
    "reinitialize_with_key",
    "reinitialize_pytree_with_key",
)


class Predictor(eqx.Module):
    """Abstract marker for a trainable ``Array -> Array`` module.

    Carries no behaviour. It exists so the rest of the framework can say
    "this leaf is a trainable function approximator" and so composition
    wrappers have one type to accept. The base ``__call__`` raises.

    To add your own family, subclass ``Predictor``, declare trainable
    arrays as ordinary fields and hyperparameters as
    ``eqx.field(static=True)``, and implement
    ``__call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]``.
    Dynamic leaves must be JAX float arrays and static fields must be
    JSON-encodable, so the module round-trips through
    ``eqx.tree_serialise_leaves``. Optionally implement
    ``initialized_with_key(self, key) -> Self`` to control how the
    tournament restarts your weights; without it,
    :func:`reinitialize_with_key` replaces every float leaf with a
    standard-normal sample, which skews any considered init scheme.

    Do not subclass to add bound handling or named inputs.
    ``BoundedPredictor`` supplies both by composition.
    """

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError("Predictor is abstract; concrete subclasses implement __call__")


def _float_array(value: Any) -> Array:
    """``jnp.asarray`` that promotes integer input to the default float dtype."""
    return jnp.asarray(value, dtype=jnp.result_type(jnp.asarray(value).dtype, float))


def _is_pair(spec: Any) -> bool:
    """True for one ``(low, high)`` pair, as opposed to a sequence of pairs."""
    return len(spec) == 2 and all(v is None or isinstance(v, Real) for v in spec)


def _bound_pairs(spec: Any, size: int) -> tuple[tuple[Any, Any], ...]:
    """Expand None, one shared pair, or one pair per component to ``size`` pairs."""
    if spec is None:
        return ((None, None),) * size
    if _is_pair(spec):
        return (tuple(spec),) * size
    pairs = tuple(tuple(pair) for pair in spec)
    if len(pairs) != size:
        raise ValueError(f"Expected {size} bound pairs, got {len(pairs)}")
    return pairs


class Constant(eqx.Module):
    """Fixed physical values, read through ``.value``.

    Lists become arrays; tuples keep their structure and numeric leaves
    become JAX arrays. The default training mask and tournament restarts
    leave constants unchanged.

    Examples
    --------
    ::

        damping = Constant(0.0)
        damping.value  # Array(0.)
    """

    value: Any

    def __init__(self, value: Any) -> None:
        def convert(item: Any) -> Any:
            if isinstance(item, tuple):
                return tuple(convert(child) for child in item)
            return _float_array(item)

        self.value = convert(value)

    def initialized_with_key(self, key: Array) -> Constant:
        """Keep fixed values unchanged when restarting a model."""
        return self


class BoundScaler(eqx.Module):
    """Two-way map between a physical range ``[low, high]`` and an unbounded latent.

    Why this exists
    ---------------
    Physical quantities have ranges, and an ODE solver handed a value
    outside one either fails or returns nonsense. An optimiser knows
    none of that; it proposes whatever number lowers the loss. Clipping
    the proposal is not usable, because a clip has exactly zero
    derivative outside the range, so a parameter that leaves the box has
    no gradient to pull it back. This scaler reparameterises instead: the
    inner predictor reads and writes a *latent* value, any real number,
    and the scaler squashes it into the physical range. An out-of-range
    physical value has no latent, so there is nothing to clip.

    Vocabulary
    ----------
    latent
        The unbounded real number the inner predictor works in. Written
        ``z`` below.
    physical
        The value in the units the simulator uses. Always inside
        ``[low, high]``.
    warp
        Decides what "halfway between the bounds" means. ``"linear"``
        puts the midpoint of ``(1e-6, 1e2)`` at 50 and collapses the
        eight-decade low end into a sliver; ``"log10"`` puts it at
        ``1e-2``. Bounds are declared in physical units either way.
        Name-keyed registry in ``transforms.py``.
    squash
        The map from latent onto ``(0, 1)``, applied before the affine
        rescale onto the box. ``"sigmoid"`` (default), ``"algebraic"``,
        ``"softsign"``. Same module, ``BOUND_TRANSFORMS``. Stored as the
        ``transform`` field.
    temperature
        Divides the latent before the squash. A larger ``T`` spreads the
        same box over a wider latent range, so the squash saturates more
        slowly. ``T = 1.0`` is the plain squash.
    knee
        The ``z_knee`` field. Latent magnitude past which
        :meth:`saturation` starts charging. Derived per transform from
        one physical criterion, the outer 5% of the box.

    The two maps
    ------------
    :meth:`to_latent` warps the physical value, normalises it to
    ``[0, 1]`` against the warped bounds, applies the squash inverse, and
    multiplies by ``T``. :meth:`from_latent` inverts that. With the
    default ``"linear"`` warp and ``"sigmoid"`` squash they read
    ``z = logit((x - low) / (high - low)) * T`` and
    ``x = low + (high - low) * sigmoid(z / T)``.

    The round trip is the identity strictly inside the open box. Outside
    a narrow band at the endpoints ``to_latent`` continues linearly (see
    its docstring), so it deviates there rather than hitting a pole.

    The cost
    --------
    Bounds hold by construction; the price is gradient. The squash
    derivative decays as ``|z|`` grows, so a predictor pinned against a
    bound has little signal left to pull it back. Sigmoid's decay is
    exponential and dies at ``z = 16.8`` in float32; ``"algebraic"`` and
    ``"softsign"`` decay polynomially and buy far more runway (numbers in
    ``transforms.py``). Runway alone is not enough, since escape time
    still grows fast with ``|z|``. :meth:`saturation` charges the output
    end for sitting deep in the squash, :meth:`input_violation` charges
    the input end for arriving outside its box. Both are pure queries;
    ``__call__`` invokes neither.

    ``temperature`` is a dynamic leaf, frozen by the default training mask.
    An explicit custom mask can opt into training it. The scaler
    defines the activation shape and the inner predictor learns inside
    it; a trainable ``T`` moves gradient between the two and tends to
    slow convergence.

    Attributes
    ----------
    bounds : tuple[tuple[float, float], ...]
        Per-component ``(low, high)`` pairs in physical units. Length
        sets the input/output dimension and the pairs apply elementwise
        to the last axis. Use None or signed infinity for a missing endpoint.
        Finite pairs must satisfy ``low < high``. Open pairs require a linear warp.
    transform : str
        Static. Squash name, a key of ``BOUND_TRANSFORMS``. Register your
        own with ``register_bound_transform``.
    temperature : Array
        Scalar (or per-component) latent sharpness. ``T = 1.0`` recovers
        the plain squash and its inverse.
    warp : str
        Static. Warp name, a key of ``WARPS``. Register your own with
        ``register_warp``.
    warped_bounds : tuple[tuple[float, float], ...]
        Static. ``bounds`` pushed through the warp once at construction.
        Resolved eagerly so no call has to warp the edges under a trace.
    logit_eps : float
        Static. Half-width of the band at each end of ``[0, 1]`` outside
        which ``to_latent`` continues linearly instead of running into
        the squash inverse's pole. Also sets the continuation slope
        (roughly ``1 / logit_eps``).
    z_knee : float
        Static. The knee. Defaults to the transform's own value, which
        for sigmoid is ``2.944`` (``logit(0.95)``), the outer 5% of the
        box. Do not share one number across transforms: 2.944 is 12.5%
        from the bound on softsign, which would charge 2.7x too hard.
    """

    bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    transform: str = eqx.field(static=True)
    temperature: Array
    warp: str = eqx.field(static=True)
    warped_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    logit_eps: float = eqx.field(static=True)
    z_knee: float = eqx.field(static=True)

    def __init__(
        self,
        bounds: tuple[tuple[float | None, float | None], ...],
        transform: str = "sigmoid",
        temperature: Any = 1.0,
        warp: str = "linear",
        logit_eps: float = 1e-3,
        z_knee: float | None = None,
    ) -> None:
        if transform not in BOUND_TRANSFORMS:
            raise ValueError(
                f"Unknown transform {transform!r}; available: "
                f"{sorted(BOUND_TRANSFORMS)}. Register custom squashes via "
                "register_bound_transform(name, transform)."
            )
        if warp not in WARPS:
            raise ValueError(
                f"Unknown warp {warp!r}; available: {sorted(WARPS)}. "
                "Register custom warps via register_warp(name, warp)."
            )
        bounds = tuple(
            (
                float("-inf") if low is None else float(low),
                float("inf") if high is None else float(high),
            )
            for low, high in bounds
        )
        check_bounds(bounds, warp)
        temperature_arr = jnp.asarray(temperature)
        # Kick the weak type. ``jnp.asarray(1.0)`` is weak-typed, and JAX's
        # compiled cache keys 0-d leaves by value *and* weak type, so the
        # first optimiser update would flip it strong and force a one-time
        # full retrace of every training kernel. A scalar this framework
        # constructs is strong from birth, so default models never pay that
        # (see ``_has_weak_scalar_trainable`` in ``training/optax.py``).
        if jax.typeof(temperature_arr).weak_type:
            temperature_arr = jnp.asarray(temperature_arr, dtype=temperature_arr.dtype)
        if temperature_arr.ndim not in (0, 1) or (
            temperature_arr.ndim == 1 and temperature_arr.shape[0] != len(bounds)
        ):
            raise ValueError(
                "BoundScaler temperature must be a positive scalar or a vector "
                f"with one entry per bound; got shape {temperature_arr.shape}."
            )
        try:
            valid_temperature = bool(
                jnp.all(jnp.isfinite(temperature_arr)) & jnp.all(temperature_arr > 0.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BoundScaler temperature must be finite and strictly positive"
            ) from exc
        if not valid_temperature:
            raise ValueError("BoundScaler temperature must be finite and strictly positive")
        if not math.isfinite(float(logit_eps)) or not 0.0 < float(logit_eps) < 0.5:
            raise ValueError(
                "BoundScaler logit_eps must be finite and lie strictly between 0 and 0.5"
            )
        self.bounds = bounds
        self.transform = transform
        self.warp = warp
        # Resolved once here, not per call: jnp ops inside a jit trace are
        # staged out, so warping the edges lazily would hand back tracers.
        self.warped_bounds = tuple(
            warp_bounds((pair,), warp)[0] if all(map(math.isfinite, pair)) else pair
            for pair in bounds
        )
        self.temperature = temperature_arr
        self.logit_eps = float(logit_eps)
        # Resolved to a float now, not looked up per call. The static field
        # stays JSON-friendly, and a saved scaler keeps the knee it trained
        # with even if the registry default later changes.
        self.z_knee = float(BOUND_TRANSFORMS[transform].knee) if z_knee is None else float(z_knee)
        if not math.isfinite(self.z_knee) or self.z_knee < 0.0:
            raise ValueError("BoundScaler z_knee must be finite and non-negative")

    def _component(self, v: Array, i: int, *, inverse: bool, guard: bool = True) -> Array:
        """Map component ``i`` between physical and latent units.

        The bound kind is static, so only the branch for this component is
        traced and no infinite endpoint ever enters the arithmetic. A finite
        interval uses the warp and squash. One finite endpoint uses softplus
        distance from it. No endpoints is the identity. ``guard=False``
        gives the exact inverse, used to initialise a ``Parameter`` close to
        an endpoint.
        """
        low, high = self.bounds[i]
        temperature = self.temperature if self.temperature.ndim == 0 else self.temperature[i]
        if math.isfinite(low) and math.isfinite(high):
            wl, wh = self.warped_bounds[i]
            warp = WARPS[self.warp]
            squash = BOUND_TRANSFORMS[self.transform]
            if not inverse:
                return warp.inverse(wl + (wh - wl) * squash.forward(v / temperature))
            normalized = (warp.forward(v) - wl) / (wh - wl)
            if not guard:
                return squash.inverse(normalized) * temperature
            return (
                soft_inverse(normalized, squash.inverse, squash.inverse_slope, self.logit_eps)
                * temperature
            )
        if not (math.isfinite(low) or math.isfinite(high)):
            return v
        edge, sign = (low, 1.0) if math.isfinite(low) else (high, -1.0)
        if not inverse:
            return edge + sign * jax.nn.softplus(v / temperature)
        distance = sign * (v - edge)
        if not guard:
            # Stable softplus inverse: log(expm1(d)) = d + log(1 - exp(-d)).
            return (distance + jnp.log(-jnp.expm1(-distance))) * temperature
        # Below logit_eps the inverse continues linearly, as in soft_inverse,
        # so an input at or past the endpoint keeps a finite sensitivity.
        safe = jnp.maximum(distance, self.logit_eps)
        latent = safe + jnp.log(-jnp.expm1(-safe))
        slope = 1.0 / (-jnp.expm1(-self.logit_eps))
        return (latent + jnp.minimum(distance - self.logit_eps, 0.0) * slope) * temperature

    def _map(self, x: Array, *, inverse: bool, guard: bool = True) -> Array:
        """Apply :meth:`_component` along the last axis; a scalar broadcasts.

        A one-component scaler acts elementwise on any shape, so a batch of
        values against a single box needs no reshaping.
        """
        x = jnp.atleast_1d(x)
        if len(self.bounds) == 1:
            return self._component(x, 0, inverse=inverse, guard=guard)
        x = jnp.broadcast_to(x, (*x.shape[:-1], len(self.bounds)))
        parts = [
            self._component(x[..., i], i, inverse=inverse, guard=guard)
            for i in range(len(self.bounds))
        ]
        return jnp.stack(parts, axis=-1)

    def to_latent(self, x: Array) -> Array:
        """Map a physical value to its latent representative.

        Warp ``x``, normalise it to ``[0, 1]`` against the warped bounds,
        apply the squash inverse through
        :func:`~jaxhybridmodels.penalties.soft_inverse`, multiply by
        ``temperature``.

        The squash inverse has a pole at each end of ``[0, 1]``, guarded by
        a linear continuation rather than a hard ``jnp.clip``. A clip has
        zero derivative outside the box, and since the guard sits mid-graph
        that zero propagates to every upstream parameter. Predictor inputs
        are often state-derived (supersaturation in the crystallisation
        example), so a clipped input silently drops a real sensitivity from
        the adjoint.

        Inside ``[logit_eps, 1 - logit_eps]`` the map is exactly the plain
        inverse in value and derivative. Outside it continues linearly at
        the inverse's slope at the crossing: finite values, non-zero
        constant gradient, and a C^1 join so an adaptive step controller
        sees no kink.

        The continuation reports direction, not magnitude. Pair it with
        :meth:`input_violation` when an input can leave its box.
        """
        warp = WARPS[self.warp]
        if warp.requires_positive:
            nonpositive = jnp.any(x <= 0.0)
            try:
                if bool(nonpositive):
                    raise ValueError(
                        f"BoundScaler warp={self.warp!r} requires strictly positive runtime inputs"
                    )
            except jax.errors.TracerBoolConversionError:
                x = eqx.error_if(
                    x,
                    nonpositive,
                    f"BoundScaler warp={self.warp!r} requires strictly positive runtime inputs",
                )
        return self._map(x, inverse=True)

    def input_violation(self, x: Array) -> Array:
        """Scalar squared hinge on how far ``x`` fell outside ``bounds``.

        Zero in value *and* gradient strictly inside the box, so adding it
        to a loss never perturbs the feasible interior. Outside, it grows
        quadratically in the width-normalised overshoot.

        The push-back half of a pair: :meth:`to_latent` keeps the forward
        pass finite and differentiable near the box, this term supplies a
        restoring force that keeps working far outside it. Pure, so the
        caller decides whether and where to pay for it.
        """
        x = jnp.atleast_1d(x)
        terms = []
        for i, (low, high) in enumerate(self.bounds):
            width = high - low if math.isfinite(low) and math.isfinite(high) else 1.0
            below = jnp.maximum(low - x[..., i], 0.0) if math.isfinite(low) else 0.0
            above = jnp.maximum(x[..., i] - high, 0.0) if math.isfinite(high) else 0.0
            terms.append((below / width) ** 2 + (above / width) ** 2)
        return jnp.sum(jnp.stack(terms, axis=-1)) if terms else jnp.asarray(0.0)

    def saturation(self, z: Array) -> Array:
        """Scalar squared overshoot of ``|z / temperature|`` past ``z_knee``.

        Measures how hard the output squash is pinned against its bound.
        For sigmoid the knee is ``2.944``, where ``sigmoid(2.944) = 0.95``,
        the outer 5% of the physical box on each side.

        The argument is the latent, never the physical value it maps to.
        ``from_latent``'s derivative carries a ``sigma'(z / T)`` factor that
        underflows to 0.0 past ``|z / T| ~ 15``, so a penalty written
        against the physical output dies exactly where saturation is worst.
        Reading ``|z| / T`` gives a gradient linear in the overshoot.

        Reduced with ``mean``, not ``sum``, so one weight means the same for
        a one-output and a six-output predictor.
        """
        z = jnp.atleast_1d(z)
        terms = []
        for i, (low, high) in enumerate(self.bounds):
            temperature = self.temperature if self.temperature.ndim == 0 else self.temperature[i]
            u = z[..., i] / temperature
            if math.isfinite(low) and math.isfinite(high):
                terms.append(jnp.maximum(jnp.abs(u) - self.z_knee, 0.0) ** 2)
            elif math.isfinite(low) or math.isfinite(high):
                terms.append(jnp.maximum(-u - self.z_knee, 0.0) ** 2)
            else:
                terms.append(jnp.zeros_like(u))
        return jnp.mean(jnp.stack(terms, axis=-1)) if terms else jnp.asarray(0.0)

    def from_latent(self, z: Array) -> Array:
        """Map a latent value back into the physical box ``[low, high]``.

        Squash ``z / temperature`` into ``(0, 1)``, affine rescale onto the
        warped box, then unwarp. The result is finite and inside the box for
        any finite ``z``, so this direction needs no guard.
        """
        return self._map(z, inverse=False)

    def scale_inputs(self, x: Array) -> Array:
        """Normalise network inputs: finite intervals map to latent, open ones pass through.

        A one-sided input bound says where the input may go, not what scale
        it has, so the network sees such an input unchanged.
        """
        if all(all(map(math.isfinite, pair)) for pair in self.bounds):
            return self.to_latent(x)
        x = jnp.atleast_1d(x)
        parts = [
            self._component(x[..., i], i, inverse=True)
            if all(map(math.isfinite, pair))
            else x[..., i]
            for i, pair in enumerate(self.bounds)
        ]
        return jnp.stack(parts, axis=-1)


class Parameter(eqx.Module):
    """A trainable physical scalar or array, read through ``.value``.

    The optimiser updates ``latent``. ``.value`` maps it through ``scaler``
    into physical units, so the value respects ``bounds`` at every step
    without clipping.

    Parameters
    ----------
    value
        Initial physical value: a scalar or an array of any shape. Must be
        finite and strictly inside its bounds.
    bounds
        One ``(low, high)`` pair shared by every element, or one pair per
        element in flattened order. None for a missing endpoint. Omit for
        an unbounded value. Two finite endpoints use the ``transform``
        squash. One endpoint uses softplus distance from it.
    transform, warp
        Passed to :class:`BoundScaler`. ``warp="log10"`` suits rate
        constants that span decades and needs finite positive bounds.

    Examples
    --------
    ::

        rate = Parameter(0.1, bounds=(0.0, None))
        rate.value  # Array(0.1)
    """

    latent: Array
    scaler: BoundScaler
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        value: Any,
        bounds: Any = None,
        *,
        transform: str = "sigmoid",
        warp: str = "linear",
    ) -> None:
        physical = _float_array(value)
        if physical.size == 0 or not bool(jnp.all(jnp.isfinite(physical))):
            raise ValueError("Parameter value must be nonempty and finite")
        flat = physical.reshape(-1)
        scaler = BoundScaler(_bound_pairs(bounds, flat.size), transform=transform, warp=warp)
        for (low, high), v in zip(scaler.bounds, flat.tolist(), strict=True):
            if not low < v < high:
                raise ValueError(f"Parameter value {v} must lie strictly inside ({low}, {high})")
        # The exact inverse, not to_latent's guarded one, so a value close to
        # an endpoint still round-trips through .value.
        self.latent = scaler._map(flat, inverse=True, guard=False)
        self.scaler = scaler
        self.shape = tuple(physical.shape)

    @property
    def value(self) -> Array:
        """Current value in physical units, with the shape given at construction."""
        return self.scaler.from_latent(self.latent).reshape(self.shape)

    def initialized_with_key(self, key: Array) -> Parameter:
        """Draw a fresh standard-normal latent; the scaler is left untouched."""
        fresh = jr.normal(key, self.latent.shape, dtype=self.latent.dtype)
        return eqx.tree_at(lambda p: p.latent, self, fresh)


class BoundedPredictor(Predictor):
    """A network in physical units, built from an inner network that never sees a bound.

    This is the thing a user's vector field calls. It takes named inputs
    in physical units, returns an output in physical units, and keeps
    the output inside its declared range.

    Three stages
    ------------
    1. **Normalise the input.** ``in_scaler.scale_inputs`` maps each input
       from its physical range onto an unbounded latent, so the network
       receives numbers of comparable size whether the input was a
       temperature in the tens or a concentration in the thousandths.
       Inputs with an open bound pass through unchanged.
    2. **Run the network.** The trainable ``inner`` ``Predictor`` maps
       latent to latent. It is handed no bound information at all.
    3. **Squash the output.** ``out_scaler.from_latent`` maps the
       network's unbounded output into the physical output range.

    The network is better off never seeing a bound. Given one it would
    have to enforce the range itself, with either a clip (zero gradient
    outside, so a parameter that leaves cannot come back) or a final
    squash it would have to learn to aim. Moving the squash into
    ``out_scaler`` makes an out-of-range output unrepresentable, leaves
    ``inner`` free to be any ``Array -> Array`` function, and lets the
    same network be reused under different bounds.

    For a single trainable number or array with no inputs, use
    :class:`Parameter`. For a fixed value, use :class:`Constant`.

    Calling it
    ----------
    The user builds predictor inputs in the vector field by mixing
    constant covariates with state-derived or exogenous time-dependent
    values (CONTEXT.md, "Predictor inputs"). ``__call__`` accepts:

    - A mapping from name to value. Extra keys are allowed; only the subset named
      in ``self.input_keys`` is pulled, in declared order. A missing key
      raises ``KeyError``.
    - An array, rank-1 of length ``len(input_keys)``, shape-checked.

    Construction
    ------------
    Give the scalers either as ``in_scaler`` / ``out_scaler`` instances or
    as ``input_bounds`` / ``output_bounds`` pairs (None for a missing
    endpoint; one pair broadcasts to every component). Sizes not fixed by
    the bounds come from ``input_keys`` or ``inner.in_size`` /
    ``inner.out_size``. At least one input is required: a zero-input
    network has no training signal, and that case is a ``Parameter``.
    Omitting ``input_keys`` auto-fills ``("x1", ..., "xN")``, so a saved
    predictor always describes its own input contract.

    Attributes
    ----------
    input_keys : tuple[str, ...]
        Static. Declared order of the physical-units inputs; one entry per
        ``in_scaler.bounds`` row. Drives subset extraction when ``__call__``
        receives a dict.
    in_scaler : BoundScaler
        Maps physical-space inputs into the inner network's latent space.
    inner : Predictor
        Trainable Array -> Array module operating in latent space.
    out_scaler : BoundScaler
        Maps the inner network's latent output back to physical units.
    """

    input_keys: tuple[str, ...] = eqx.field(static=True)
    in_scaler: BoundScaler
    inner: Predictor
    out_scaler: BoundScaler

    def __init__(
        self,
        *,
        inner: Predictor,
        in_scaler: BoundScaler | None = None,
        out_scaler: BoundScaler | None = None,
        input_keys: tuple[str, ...] | None = None,
        input_bounds: Any = None,
        output_bounds: Any = None,
    ) -> None:
        if in_scaler is not None and input_bounds is not None:
            raise ValueError("Use in_scaler= or input_bounds=, not both")
        if out_scaler is not None and output_bounds is not None:
            raise ValueError("Use out_scaler= or output_bounds=, not both")

        def size_of(bounds: Any, *fallbacks: int | None) -> int | None:
            # A sequence of pairs fixes the size; one shared pair does not.
            if bounds is not None and not _is_pair(bounds):
                return len(bounds)
            known = next((s for s in fallbacks if s is not None), None)
            return 1 if known is None and bounds is not None else known

        if in_scaler is None:
            n_in = size_of(
                input_bounds,
                None if input_keys is None else len(input_keys),
                getattr(inner, "in_size", None),
            )
            if n_in is None:
                raise ValueError("Supply in_scaler, input_bounds, input_keys, or inner.in_size")
            in_scaler = BoundScaler(_bound_pairs(input_bounds, int(n_in)))
        if out_scaler is None:
            n_out = size_of(output_bounds, getattr(inner, "out_size", None))
            if n_out is None:
                raise ValueError("Supply out_scaler, output_bounds, or inner.out_size")
            out_scaler = BoundScaler(_bound_pairs(output_bounds, int(n_out)))
        n = len(in_scaler.bounds)
        if n < 1:
            raise ValueError(
                "BoundedPredictor requires at least one input; "
                "use Parameter for a trainable value with no inputs."
            )
        if input_keys is None:
            resolved_keys: tuple[str, ...] = tuple(f"x{i + 1}" for i in range(n))
        else:
            resolved_keys = tuple(input_keys)
            if len(resolved_keys) != n:
                raise ValueError(
                    f"input_keys length {len(resolved_keys)} does not match "
                    f"in_scaler.bounds length {n}; they must agree."
                )
        inner_in_size = getattr(inner, "in_size", None)
        if inner_in_size is not None and int(inner_in_size) != n:
            raise ValueError(
                "BoundedPredictor inner input dimension must match in_scaler.bounds; "
                f"got inner.in_size={inner_in_size} and {n} input bounds."
            )
        inner_out_size = getattr(inner, "out_size", None)
        n_outputs = len(out_scaler.bounds)
        if inner_out_size is not None and int(inner_out_size) != n_outputs:
            raise ValueError(
                "BoundedPredictor inner output dimension must match out_scaler.bounds; "
                f"got inner.out_size={inner_out_size} and {n_outputs} output bounds."
            )
        self.input_keys = resolved_keys
        self.in_scaler = in_scaler
        self.inner = inner
        self.out_scaler = out_scaler

    def __call__(self, inputs: Mapping[str, ArrayLike] | ArrayLike) -> Array:
        n = len(self.input_keys)
        if isinstance(inputs, Mapping):
            # A missing key raises KeyError from the dict access itself.
            # eqx.error_if does not fit here: it needs an array to attach the
            # check to, and the failure happens before any array exists.
            named = cast(Mapping[str, ArrayLike], inputs)  # ty narrows to bare Mapping
            x = jnp.stack([jnp.asarray(named[k]) for k in self.input_keys])
        else:
            x = jnp.asarray(inputs)
            # Array path contract: rank-1 with length matching input_keys. The
            # predicate is Python-static (shapes are known at trace time), so
            # this raises at trace time under jit.
            x = eqx.error_if(
                x,
                x.ndim != 1 or x.shape[0] != n,
                f"BoundedPredictor expected rank-1 array of length {n} "
                f"(matching input_keys={self.input_keys}); got shape {x.shape}.",
            )
        z_in = self.in_scaler.scale_inputs(x)
        z_out = jnp.asarray(self.inner(z_in))
        if z_out.ndim == 0 and len(self.out_scaler.bounds) == 1:
            z_out = jnp.reshape(z_out, (1,))
        z_out = eqx.error_if(
            z_out,
            z_out.ndim != 1 or z_out.shape[0] != len(self.out_scaler.bounds),
            "BoundedPredictor inner output must be rank-1 with one entry per "
            f"out_scaler bound; got shape {z_out.shape}.",
        )
        return self.out_scaler.from_latent(z_out)

    def initialized_with_key(self, key: Array) -> BoundedPredictor:
        """Re-initialise ``inner`` only, leaving both scalers untouched.

        Without this method, ``reinitialize_with_key`` takes its generic
        branch and replaces every inexact leaf, ``BoundScaler.temperature``
        included, with a sample from ``N(0, 1)``. A temperature near zero
        or negative inverts and blows up both ``to_latent`` and
        ``from_latent``.

        Delegating to the free function also restores the inner predictor's
        own init scheme rather than leaf-level normal sampling. The scalers
        hold bound geometry, not learned state, so a restart has no reason
        to touch them.
        """
        return eqx.tree_at(
            lambda bp: bp.inner,
            self,
            reinitialize_with_key(self.inner, key),
        )


def reinitialize_with_key(predictor: eqx.Module, key: Array) -> eqx.Module:
    """Return a fresh copy of ``predictor`` with its float leaves re-initialised.

    Single-Module helper. If ``predictor`` implements
    ``initialized_with_key`` (a ``self -> key -> self`` method for classes
    that want their own re-init scheme, such as a KAN that has to rebuild
    its grid), this delegates to it. Otherwise every inexact-array leaf is
    replaced with a standard-normal sample of matching shape and dtype.

    For a *pytree* of predictors, the convention at the ``simulate_fn``
    boundary, use :func:`reinitialize_pytree_with_key`.
    """
    if hasattr(predictor, "initialized_with_key"):
        return cast(eqx.Module, predictor.initialized_with_key(key))

    leaves, treedef = jtu.tree_flatten(predictor)
    inexact_indices = [i for i, leaf in enumerate(leaves) if eqx.is_inexact_array(leaf)]
    if not inexact_indices:
        return predictor
    subkeys = jr.split(key, len(inexact_indices))
    new_leaves = list(leaves)
    for k, idx in zip(subkeys, inexact_indices, strict=True):
        leaf = leaves[idx]
        new_leaves[idx] = jr.normal(k, leaf.shape, dtype=leaf.dtype)
    return cast(eqx.Module, jtu.tree_unflatten(treedef, new_leaves))


def reinitialize_pytree_with_key(predictors: Any, key: Array) -> Any:
    """Per-``eqx.Module``-leaf re-initialisation across a ``predictors`` pytree.

    Used by the training tournament to escape bad initial weights: when
    an attempt diverges or stalls, the loop draws a fresh per-attempt key,
    calls this, and restarts. Each ``eqx.Module`` leaf gets its own
    independent subkey, so identical-shape siblings re-init differently.

    Subkeys are handed out in traversal order, which is enough because the
    pytree shape is fixed across re-inits within one run. Path-derived
    subkeys would be path-stable at the cost of a hash per leaf.

    Accepts any pytree shape ``jax.tree_util`` can walk, including a bare
    ``eqx.Module``. Returns a structurally identical pytree with fresh
    weights on every ``eqx.Module`` leaf.
    """

    def _is_module(node: Any) -> bool:
        return isinstance(node, eqx.Module)

    module_leaves = [
        leaf for leaf in jtu.tree_leaves(predictors, is_leaf=_is_module) if _is_module(leaf)
    ]
    n_modules = len(module_leaves)
    if n_modules == 0:
        return predictors

    subkeys = jr.split(key, n_modules)
    keys_iter = iter(subkeys)

    def _per_leaf(node: Any) -> Any:
        if _is_module(node):
            return reinitialize_with_key(node, next(keys_iter))
        return node

    return jtu.tree_map(_per_leaf, predictors, is_leaf=_is_module)
