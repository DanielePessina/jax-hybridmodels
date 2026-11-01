"""A log-density for a hybrid model, for whichever sampler you want to run.

This module does no sampling. :func:`make_problem` turns a model, its data, a
prior and a noise model into a :class:`InferenceProblem`, whose ``log_density``
is a pure JAX function of a pytree position. BlackJAX, NumPyro's ``potential_fn``
or an Optimistix MAP search can all take it::

    def log_prior(predictors, noise):
        return -0.5 * jnp.log(predictors["k"].value / 0.3) ** 2

    problem = hm.inference.make_problem(model, data, log_prior=log_prior, noise="data")
    warmup = blackjax.window_adaptation(blackjax.nuts, problem.log_density)
    (state, params), _ = warmup.run(key, problem.position, num_steps=1000)

Three things make this density differ from the training loss.

- It sums over buckets. The trainers average them, which tempers even the
  ``"mle"`` loss by one over the number of buckets, and makes a posterior depend
  on how experiments happened to be grouped.
- The position holds latents, while priors are written on physical values, so
  the change-of-variables term is added here for every free ``Parameter``.
  Leaving it out would imply a prior nobody chose, with mass piled at the bounds.
- A failed solve gives ``-inf`` and a zero gradient, so a sampler sees a rejected
  proposal instead of an exception that ends the chain.

A sampler's cost is its number of gradients, often tens of thousands, so two
things matter more than solver settings. ``gradient="auto"`` differentiates a few
parameters in forward mode, which is cheaper than reversing through the solve.
:meth:`InferenceProblem.preconditioned` whitens the position with the Gauss-Newton
curvature at the fit, which removes the correlations and scale differences that
otherwise force a tiny step size and deep NUTS trees.

The default target is the ``Parameter`` values. ``free=`` can name any subtree,
such as a network's last layer. Whatever stays fixed is conditioned on, and
``InferenceProblem.conditional_on`` lists it: a posterior over a physical
parameter with a fitted network held fixed is a conditional posterior, and the
network can absorb what the physics should explain.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from jaxtyping import Array

from jaxhybridmodels.data import BucketPayload, Data, Dataset
from jaxhybridmodels.model import HybridModel
from jaxhybridmodels.prediction import check_quantiles, member_quantiles, predict_ensemble
from jaxhybridmodels.predictors import Parameter
from jaxhybridmodels.results import PredictionResult
from jaxhybridmodels.rng import check_key
from jaxhybridmodels.trainable import (
    FreezeSpec,
    _path_to_dotted,
    freeze_mask,
    trainable_mask,
)
from jaxhybridmodels.training.kernels import predict_bucket_obs

__all__ = ("InferenceProblem", "flat_prior", "make_problem")

_PREDICTORS = "predictors"
_NOISE = "noise"

# gradient="auto" uses forward mode up to this many free values. Forward cost grows
# with the count, and reverse does not: on the tubular reactor forward took 8 ms
# against 25 ms at 2 values, and 368 ms against 187 ms at 17.
_FORWARD_MODE_LIMIT = 8


def flat_prior(predictors: Any, noise: Mapping[str, Array]) -> Array:
    """A prior flat in physical units, for free ``Parameter``s with finite bounds.

    Proper only on a bounded box, so :func:`make_problem` refuses it when a free
    value has an open bound or is a network weight. A flat prior is not neutral
    under a warp: on a log-warped parameter it still favours the upper decades.
    """
    return jnp.asarray(0.0)


@jax.custom_vjp
def _finite_gradients(position: Any) -> Any:
    """The identity, except that non-finite gradients flowing back through it become 0.

    Gating a value with ``jnp.where`` cannot do this: reverse mode still
    multiplies the failed solve's NaN sensitivities by zero, which is NaN.
    """
    return position


def _finite_gradients_fwd(position: Any) -> tuple[Any, None]:
    return position, None


def _finite_gradients_bwd(_: None, cotangent: Any) -> tuple[Any]:
    return (jtu.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), cotangent),)


_finite_gradients.defvjp(_finite_gradients_fwd, _finite_gradients_bwd)


def _forward_mode_density(density: Callable[[Any], Array]) -> Callable[[Any], Array]:
    """``density``, whose reverse-mode gradient is computed in forward mode.

    Samplers call ``jax.grad``, so forward mode has to hide behind a custom VJP. A
    non-finite tangent from a failed solve becomes zero, as in reverse mode.
    """

    def with_gradient(position: Any) -> Any:
        def twice(x: Any) -> tuple[Array, Array]:
            value = density(x)
            return value, value

        gradient, value = jax.jacfwd(twice, has_aux=True)(position)
        return value, jtu.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), gradient)

    @jax.custom_vjp
    def wrapped(position: Any) -> Array:
        return density(position)

    def backward(gradient: Any, cotangent: Array) -> tuple[Any]:
        return (jtu.tree_map(lambda g: g * cotangent, gradient),)

    wrapped.defvjp(with_gradient, backward)
    return wrapped


def _log_abs_jacobian(parameter: Parameter) -> Array:
    """``sum log|d value / d latent|`` for one free ``Parameter``.

    The squash maps each latent to its own value, so a tangent of ones reads off
    the whole diagonal in one ``jvp``. Far into a squash's tail the derivative
    underflows to zero, and the term is held at the smallest positive float there
    so the density stays finite and carries no gradient from this term.
    """
    latent = parameter.latent
    _, derivative = jax.jvp(parameter.scaler.from_latent, (latent,), (jnp.ones_like(latent),))
    size = jnp.abs(jnp.ravel(derivative))
    smallest = jnp.finfo(size.dtype).tiny
    positive = size > 0.0
    return jnp.sum(jnp.where(positive, jnp.log(jnp.where(positive, size, 1.0)), jnp.log(smallest)))


def _is_parameter(node: Any) -> bool:
    return isinstance(node, Parameter)


def _dotted(path: tuple[Any, ...]) -> str:
    """A leaf's path in the position, without the ``predictors`` level users never wrote."""
    dotted = _path_to_dotted(path)
    prefix = f"{_PREDICTORS}."
    return dotted[len(prefix) :] if dotted.startswith(prefix) else dotted


@dataclass(frozen=True, eq=False)
class InferenceProblem:
    """A model, its data, a prior and a noise model, as a function of a position.

    Build one with :func:`make_problem`. Identity-hashed, so bound methods such as
    ``problem.log_density`` can be passed to ``jax.jit`` and to samplers.

    Attributes
    ----------
    model : HybridModel
        The model as given. Solves here run through a copy with
        ``SolverConfig.throw`` off.
    dataset : Dataset
        The data the likelihood reads.
    position : PyTree
        The starting position: the free leaves' current values. Latents for a
        ``Parameter``, weights for a network, and inferred noise under ``"noise"``.
        After :meth:`preconditioned`, a flat array of zeros instead. ``physical``,
        ``to_model``, ``predict`` and ``to_arviz`` read either form.
    conditional_on : tuple[str, ...]
        Dotted paths of the trainable leaves held fixed. The posterior is
        conditional on these values and its width does not account for them.
    """

    model: HybridModel
    dataset: Dataset
    position: Any
    conditional_on: tuple[str, ...]
    _log_prior: Callable[..., Array]
    _log_likelihood: Callable[..., Array] | None
    _static: Any
    _mask: Any
    _sampling_model: HybridModel
    _forward_model: HybridModel
    _gradient: str
    _forward_density: Callable[[Any], Array] | None = None
    _whitening: tuple[Array, Array, Callable[[Array], Any]] | None = None

    def _tree(self, position: Any) -> dict[str, Any]:
        if self._whitening is not None:
            shift, factor, unravel = self._whitening
            position = unravel(shift + factor @ position)
        return cast(dict[str, Any], eqx.combine(position, self._static))

    def _predictions(self, predictors: Any, bp: BucketPayload) -> Array:
        return predict_bucket_obs(
            predictors,
            bp,
            simulate_fn=self._sampling_model.simulate_fn,
            state_to_output=self._sampling_model.state_to_output,
            solver=self._sampling_model.solver,
        )

    def _whitened(self, pred_obs: Array, bp: BucketPayload, noise: Mapping[str, Array]) -> Any:
        """``(residuals, variances)``, both ``[N, T, D]``, with benign values at masked cells."""
        if noise:
            sigma = jnp.stack([noise[name] for name in self.dataset.output_channel_names], axis=-1)
            variance = jnp.broadcast_to(sigma**2, bp.yvar.shape)
        else:
            variance = bp.yvar
        # Unmeasured cells hold placeholders, so they are replaced before any
        # arithmetic and cannot reach a value or a gradient.
        safe_variance = jnp.where(bp.mask, variance, 1.0)
        safe_prediction = jnp.where(bp.mask, pred_obs, 0.0)
        observed = jnp.where(bp.mask, bp.y_observed, 0.0)
        residuals = (safe_prediction - observed) / jnp.sqrt(safe_variance)
        return jnp.where(bp.mask, residuals, 0.0), safe_variance

    def residuals(self, position: Any) -> Array:
        """Whitened residuals ``(prediction - observed) / sigma``, flattened over buckets.

        Length is every ``[N, T, D]`` cell, with zero at unmeasured cells, so the
        shape does not depend on the data's mask and ``jax.jacfwd`` applies.
        ``log_likelihood`` is built on the same quantity for a Gaussian.
        """
        tree = self._tree(position)
        noise = self._noise_values(tree)
        parts = [
            self._whitened(self._predictions(tree[_PREDICTORS], bp), bp, noise)[0].ravel()
            for bp in self.dataset.bucket_payloads
        ]
        return jnp.concatenate(parts)

    def _noise_values(self, tree: Mapping[str, Any]) -> dict[str, Array]:
        return {name: jnp.squeeze(p.value) for name, p in tree[_NOISE].items()}

    def log_likelihood(self, position: Any) -> Array:
        """The log-likelihood of the data, summed over every experiment.

        ``-inf`` when any solve fails or any prediction is non-finite, including
        at a row no channel measured, since a failed solve leaves such rows
        non-finite too.
        """
        tree = self._tree(position)
        predictors = tree[_PREDICTORS]
        noise = self._noise_values(tree)
        total = jnp.asarray(0.0)
        finite = jnp.asarray(True)
        for bp in self.dataset.bucket_payloads:
            pred_obs = self._predictions(predictors, bp)
            finite = finite & jnp.all(jnp.isfinite(pred_obs))
            clean = jnp.where(jnp.isfinite(pred_obs), pred_obs, 0.0)
            if self._log_likelihood is not None:
                total = total + self._log_likelihood(clean, bp, noise)
            else:
                residuals, variance = self._whitened(clean, bp, noise)
                normaliser = jnp.where(bp.mask, jnp.log(2.0 * jnp.pi * variance), 0.0)
                total = total - 0.5 * (jnp.sum(residuals**2) + jnp.sum(normaliser))
        return jnp.where(finite & jnp.isfinite(total), total, -jnp.inf)

    def _log_jacobian(self, tree: Mapping[str, Any]) -> Array:
        total = jnp.asarray(0.0)
        leaves = jtu.tree_leaves(tree, is_leaf=_is_parameter)
        flags = jtu.tree_leaves(self._mask, is_leaf=_is_parameter)
        for leaf, flag in zip(leaves, flags, strict=True):
            # A partly fixed Parameter cannot occur: its one latent is free or not.
            if _is_parameter(leaf) and flag.latent:
                total = total + _log_abs_jacobian(leaf)
        return total

    def _density(self, position: Any) -> Array:
        tree = self._tree(position)
        prior = self._log_prior(tree[_PREDICTORS], self._noise_values(tree))
        total = self.log_likelihood(position) + prior + self._log_jacobian(tree)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    def log_density(self, position: Any) -> Array:
        """Unnormalised log-posterior at ``position``, for a sampler to target.

        Log-likelihood, plus the prior on physical values, plus the
        change-of-variables term for free ``Parameter`` latents. It is a density
        over the latents, so it is ``-inf`` with a zero gradient wherever a solve
        fails, and any NaN term also becomes ``-inf``.

        Take its gradient with ``jax.grad``, as samplers do. Its custom gradient
        rule rules out ``jax.jvp`` and ``jax.jacfwd`` on this method, so use them on
        ``log_likelihood`` or ``residuals``.
        """
        if self._forward_density is not None:
            return self._forward_density(position)
        return self._density(_finite_gradients(position))

    def preconditioned(self, *, precision: Any = None, ridge: float = 1.0) -> InferenceProblem:
        """This problem, sampling a whitened position so the target is close to a unit normal.

        ``position = fit + L z``, where ``L L^T`` is the inverse of the curvature
        at the current position. It is a constant linear map, so the posterior over
        the physical values is unchanged and only the sampler's geometry improves.
        Start from a fitted model: far from a mode the curvature says little, and
        sampling is then no better but still correct.

        Parameters
        ----------
        precision : Array | None
            The ``[P, P]`` curvature to use, for a custom likelihood or a better
            estimate. ``None`` computes the Gauss-Newton ``J^T J + ridge I`` from
            the whitened residuals, which needs the Gaussian likelihood.
        ridge : float
            Added to the diagonal of the Gauss-Newton curvature in latent units,
            standing in for the prior so an unidentified direction keeps a finite
            scale. Ignored with ``precision``.
        """
        if self._whitening is not None:
            raise ValueError("preconditioned: this problem is already preconditioned")
        flat, unravel = ravel_pytree(self.position)
        if precision is None:
            if self._log_likelihood is not None:
                raise ValueError(
                    "preconditioned: Gauss-Newton needs the Gaussian likelihood; pass precision= "
                    "for a custom log_likelihood"
                )
            forward = dataclasses.replace(self, _sampling_model=self._forward_model)
            jacobian = jax.jacfwd(lambda x: forward.residuals(unravel(x)))(flat)
            precision = jacobian.T @ jacobian + ridge * jnp.eye(flat.size, dtype=flat.dtype)
        precision = jnp.asarray(precision, dtype=flat.dtype)
        if precision.shape != (flat.size, flat.size):
            raise ValueError(
                f"preconditioned: precision must have shape {(flat.size, flat.size)}; "
                f"got {precision.shape}"
            )
        # precision = C C^T, so the covariance's square root is C^-T.
        cholesky = jnp.linalg.cholesky(precision)
        identity = jnp.eye(flat.size, dtype=flat.dtype)
        factor = jax.scipy.linalg.solve_triangular(cholesky.T, identity, lower=False)
        if not bool(jnp.all(jnp.isfinite(factor))):
            raise ValueError(
                "preconditioned: the curvature could not be factorised. Check that the model "
                "solves at this position; in single precision, enable x64."
            )
        whitened = dataclasses.replace(
            self,
            position=jnp.zeros_like(flat),
            _forward_density=None,
            _whitening=(flat, factor, unravel),
        )
        return _choose_gradient(whitened, self._gradient)

    def to_model(self, position: Any) -> HybridModel:
        """The given model with this position's values, for ``predict`` and ``save``.

        It keeps the model's own solver, so its solves raise on failure as usual.
        Inferred noise is not part of a model.
        """
        return eqx.tree_at(lambda m: m.predictors, self.model, self._tree(position)[_PREDICTORS])

    def physical(self, position: Any) -> dict[str, Array]:
        """The free values in physical units, keyed by dotted path.

        A ``Parameter`` appears under its path with its ``.value``, a free network
        leaf under its path as it is, and inferred noise as ``"noise.<name>"``.
        """
        tree = self._tree(position)
        flat_mask = jtu.tree_leaves(self._mask, is_leaf=_is_parameter)
        flat_tree = jtu.tree_flatten_with_path(tree, is_leaf=_is_parameter)[0]
        out: dict[str, Array] = {}
        for (path, leaf), flag in zip(flat_tree, flat_mask, strict=True):
            if _is_parameter(leaf):
                if flag.latent:
                    out[_dotted(path)] = leaf.value
            elif flag:
                out[_dotted(path)] = leaf
        return out

    def predict(
        self,
        positions: Any,
        data: Data | None = None,
        *,
        key: Array,
        ts: ArrayLike | None = None,
        n_points: int | None = None,
        n_draws: int = 200,
        quantiles: Sequence[float] = (0.05, 0.95),
        observation_noise: bool = False,
    ) -> PredictionResult:
        """Predict with draws from a chain and report their spread.

        ``positions`` is the sampler's output: the position pytree with any
        leading sample axes, such as ``[draw, ...]`` or ``[chain, draw, ...]``.
        ``n_draws`` of them are chosen at random, without replacement. ``data``
        defaults to the problem's own. ``ts`` and ``n_points`` are as for
        :meth:`HybridModel.predict`.

        ``outputs`` is the mean over draws. ``std`` and ``quantiles`` describe the
        model's uncertainty, a credible band for the noiseless trajectory. With
        ``observation_noise``, they include measurement noise instead, an interval
        for a new measurement. That needs a standard deviation inferred for each
        channel, since the variances in data exist only at measured cells.
        """
        check_key(key, "InferenceProblem.predict")
        levels = check_quantiles(quantiles, "InferenceProblem.predict")
        if n_draws < 1:
            raise ValueError(f"InferenceProblem.predict: n_draws must be positive; got {n_draws}")
        channels = self.dataset.output_channel_names
        if observation_noise and not set(channels) <= set(self._tree(self.position)[_NOISE]):
            raise ValueError(
                "InferenceProblem.predict: observation_noise needs noise inferred for every "
                f"channel {list(channels)}; make_problem(noise={{...}}) provides it"
            )
        choice_key, noise_key = jr.split(key)
        draws = self._unstack(positions, n_draws, choice_key)
        result = predict_ensemble(
            [self.to_model(draw) for draw in draws],
            self.dataset if data is None else data,
            ts=ts,
            n_points=n_points,
            quantiles=levels,
        )
        if not observation_noise:
            return result
        sigmas = [self._noise_values(self._tree(draw)) for draw in draws]
        return _with_observation_noise(result, sigmas, levels, noise_key)

    def _unstack(self, positions: Any, n_draws: int, key: Array) -> list[Any]:
        """``positions`` as a list of up to ``n_draws`` single positions."""
        reference = jtu.tree_leaves(self.position)[0]
        first = jtu.tree_leaves(positions)[0]
        extra = first.ndim - reference.ndim
        if extra < 1 or first.shape[extra:] != reference.shape:
            raise ValueError(
                f"InferenceProblem: positions need leading sample axes before the shape "
                f"{reference.shape}; got a leaf of shape {first.shape}"
            )
        flat = jtu.tree_map(
            lambda leaf: leaf.reshape((-1, *leaf.shape[extra:])),
            positions,
        )
        count = jtu.tree_leaves(flat)[0].shape[0]
        chosen = jr.permutation(key, count)[: min(n_draws, count)]
        return [jtu.tree_map(lambda leaf, i=i: leaf[i], flat) for i in chosen.tolist()]

    def to_arviz(self, positions: Any, *, sample_stats: Mapping[str, Any] | None = None) -> Any:
        """An ArviZ ``DataTree`` of the physical values, for R-hat, ESS and plots.

        ``positions`` must have leading ``[chain, draw]`` axes; add one with
        ``jax.tree.map(lambda x: x[None], positions)`` for a single chain.
        Variables are named by dotted path. ``sample_stats`` holds per-draw
        sampler output with the same axes, such as ``{"diverging": ...}``.
        ``conditional_on`` is recorded in the attributes. Needs
        ``pip install jax-hybridmodels[bayes]``, and ArviZ needs Python 3.12.
        """
        try:
            from arviz_base import from_dict
        except ImportError as exc:
            raise ImportError(
                "to_arviz needs ArviZ, which needs Python 3.12 or later; install it with: "
                "pip install jax-hybridmodels[bayes]"
            ) from exc
        reference = jtu.tree_leaves(self.position)[0]
        first = jtu.tree_leaves(positions)[0]
        if first.ndim != reference.ndim + 2:
            raise ValueError(
                "to_arviz: positions need leading [chain, draw] axes before the shape "
                f"{reference.shape}; got a leaf of shape {first.shape}"
            )
        named = jax.vmap(jax.vmap(self.physical))(positions)
        groups = {"posterior": {name: np.asarray(value) for name, value in named.items()}}
        if sample_stats:
            groups["sample_stats"] = {k: np.asarray(v) for k, v in sample_stats.items()}
        # A string, as a NetCDF attribute cannot hold an empty list.
        conditioning = ", ".join(self.conditional_on)
        return from_dict(groups, attrs={"posterior": {"conditional_on": conditioning}})


def _choose_gradient(problem: InferenceProblem, gradient: str) -> InferenceProblem:
    """``problem`` with its gradient mode set, trying forward mode for ``"auto"``."""
    if gradient == "reverse":
        return problem
    forward = dataclasses.replace(
        problem, _sampling_model=problem._forward_model, _forward_density=None
    )
    if gradient == "auto":
        if ravel_pytree(problem.position)[0].size > _FORWARD_MODE_LIMIT:
            return problem
        try:
            eqx.filter_eval_shape(jax.jacfwd(forward._density), problem.position)
        except (TypeError, ValueError, NotImplementedError):
            # A simulate_fn with its own adjoint, or code with no forward rule.
            return problem
    return dataclasses.replace(problem, _forward_density=_forward_mode_density(forward._density))


def _with_observation_noise(
    result: PredictionResult,
    sigmas: Sequence[Mapping[str, Array]],
    levels: Sequence[float],
    key: Array,
) -> PredictionResult:
    """``result`` with ``std`` and ``quantiles`` widened by each draw's own noise.

    ``outputs`` stay the draws' mean: zero-mean noise does not move it, and adding
    sampled noise there would only make it rougher.
    """
    experiments = {}
    for exp_key, (exp_id, experiment) in zip(
        jr.split(key, len(result)), result.items(), strict=True
    ):
        noisy = {}
        channel_keys = jr.split(exp_key, len(experiment.outputs))
        for channel_key, name in zip(channel_keys, experiment.outputs, strict=True):
            means = jnp.stack([member[exp_id].outputs[name] for member in result.members])
            scale = jnp.stack([sigma[name] for sigma in sigmas])[:, None]
            noisy[name] = means + scale * jr.normal(channel_key, means.shape, dtype=means.dtype)
        experiments[exp_id] = dataclasses.replace(
            experiment,
            std={name: jnp.std(values, axis=0) for name, values in noisy.items()},
            quantiles=member_quantiles(noisy, levels),
        )
    return dataclasses.replace(result, experiments=experiments)


def make_problem(
    model: HybridModel,
    data: Data,
    *,
    log_prior: Callable[[Any, Mapping[str, Array]], Array],
    noise: Literal["data"] | Mapping[str, Parameter],
    free: Sequence[FreezeSpec] | FreezeSpec | None = None,
    log_likelihood: Callable[[Array, BucketPayload, Mapping[str, Array]], Array] | None = None,
    gradient: Literal["auto", "forward", "reverse"] = "auto",
) -> InferenceProblem:
    """Build the log-density for ``model`` on ``data``.

    Parameters
    ----------
    model : HybridModel
        Its current values are the starting position, so a fitted model starts a
        chain near the mode.
    data : Dataset | Sequence[Experiment]
        The observations. Bucketing does not change the density.
    log_prior : Callable
        ``(predictors, noise) -> scalar``, the log-prior on physical values: read
        ``predictors["k"].value``, and ``noise["a"]`` for inferred noise. There is
        no default because no prior is neutral. :func:`flat_prior` is available
        for a bounded box. A free network's weights need a prior written here.
    noise : "data" | Mapping[str, Parameter]
        ``"data"`` trusts the variances in the data, and warns for a channel left
        at the default of 1.0. A mapping infers a standard deviation per channel,
        each a ``Parameter`` naming its starting value and bounds. With
        ``log_likelihood`` its keys are free-form, such as an AR(1) coefficient.
    free : str | type | Callable | Sequence | None
        What to sample, in the syntax of ``fit(freeze=...)``. ``None`` means every
        ``Parameter``. A dotted path can name a network part, such as
        ``"g.inner.mlp.layers.2"``.
    log_likelihood : Callable | None
        Replaces the Gaussian: ``(pred_obs, bp, noise) -> scalar``, called once per
        bucket and summed. ``pred_obs`` is ``[N, T, D]``, and ``bp.mask`` says
        which cells were measured.
    gradient : {"auto", "forward", "reverse"}
        How the density is differentiated. ``"auto"`` uses forward mode for a few
        free values when the model allows it, and otherwise reverse mode with the
        model's own adjoint. Forward mode replaces the adjoint with
        ``diffrax.ForwardMode``, so a ``simulate_fn`` must use ``solver.diffeqsolve``
        or pass ``adjoint=solver.adjoint``.

    Solves inside the density run with ``SolverConfig.throw`` off. A ``simulate_fn``
    that calls ``diffrax.diffeqsolve`` itself must pass ``throw=solver.throw``, or
    a failed solve will end the chain with an exception.
    """
    if gradient not in ("auto", "forward", "reverse"):
        raise ValueError(
            f"make_problem: gradient must be 'auto', 'forward' or 'reverse'; got {gradient!r}"
        )
    dataset = model._dataset(data, "make_problem")
    model._check_shapes(dataset.bucket_payloads[0], "make_problem")
    predictors = model.predictors
    noise_map = _check_noise(noise, dataset, custom=log_likelihood is not None)

    base = trainable_mask(predictors)
    free_mask = _free_mask(predictors, base, free)
    if not any(jtu.tree_leaves(free_mask)):
        raise ValueError("make_problem: free= selects no trainable leaf; nothing to sample")
    if log_prior is flat_prior:
        _check_flat_prior(predictors, free_mask, noise_map)

    conditional_on = tuple(
        _path_to_dotted(path)
        for (path, in_base), in_free in zip(
            jtu.tree_flatten_with_path(base)[0], jtu.tree_leaves(free_mask), strict=True
        )
        if in_base and not in_free
    )
    if conditional_on:
        shown = ", ".join(conditional_on[:4]) + (", ..." if len(conditional_on) > 4 else "")
        warnings.warn(
            f"make_problem: {len(conditional_on)} trainable leaves stay fixed ({shown}), so this "
            "is a posterior conditional on their values. If a network can absorb what the free "
            "values explain, it can be too narrow. Compare the free values across fit_ensemble "
            "members to check.",
            UserWarning,
            stacklevel=2,
        )
    if jnp.asarray(0.0).dtype != jnp.float64:
        warnings.warn(
            "make_problem: x64 is off. Single precision and solver tolerance noise can degrade "
            "HMC energy conservation; enable x64 before importing anything else that uses JAX.",
            UserWarning,
            stacklevel=2,
        )

    tree = {_PREDICTORS: predictors, _NOISE: dict(noise_map)}
    mask = {_PREDICTORS: free_mask, _NOISE: trainable_mask(dict(noise_map))}
    position, static = eqx.partition(tree, mask)
    sampling_solver = dataclasses.replace(model.solver, throw=False)
    problem = InferenceProblem(
        model=model,
        dataset=dataset,
        position=position,
        conditional_on=conditional_on,
        _log_prior=log_prior,
        _log_likelihood=log_likelihood,
        _static=static,
        _mask=mask,
        _sampling_model=dataclasses.replace(model, solver=sampling_solver),
        _forward_model=dataclasses.replace(
            model, solver=dataclasses.replace(sampling_solver, adjoint=diffrax.ForwardMode())
        ),
        _gradient=gradient,
    )
    return _choose_gradient(problem, gradient)


def _free_mask(predictors: Any, base: Any, free: Any) -> Any:
    """``base`` narrowed to what ``free`` matches, reusing ``freeze_mask``'s matching."""
    if free is None:
        leaves = jtu.tree_leaves(predictors, is_leaf=_is_parameter)
        if not any(_is_parameter(leaf) for leaf in leaves):
            raise ValueError(
                "make_problem: this model has no Parameter to sample. Name a network part with "
                "free=, such as a last layer, or use fit_ensemble for whole-network uncertainty."
            )
        free = Parameter
    try:
        unmatched = freeze_mask(predictors, free, mask=base)
    except ValueError as exc:
        exc.add_note("make_problem: free= takes the same entries as fit(freeze=...).")
        raise
    return jtu.tree_map(lambda in_base, kept: in_base and not kept, base, unmatched)


def _check_noise(noise: Any, dataset: Dataset, *, custom: bool) -> dict[str, Parameter]:
    if isinstance(noise, str):
        if noise != "data":
            raise ValueError(f"make_problem: noise must be 'data' or a mapping; got {noise!r}")
        if not custom:
            _warn_on_default_variance(dataset)
        return {}
    if not isinstance(noise, Mapping):
        raise TypeError(
            f"make_problem: noise must be 'data' or a mapping; got {type(noise).__name__}"
        )
    for name, value in noise.items():
        if not isinstance(value, Parameter) or value.latent.size != 1:
            raise ValueError(f"make_problem: noise[{name!r}] must be a scalar hm.Parameter")
    if not custom and set(noise) != set(dataset.output_channel_names):
        raise ValueError(
            f"make_problem: noise names {sorted(noise)} but the channels are "
            f"{sorted(dataset.output_channel_names)}; the Gaussian needs one per channel"
        )
    return dict(noise)


def _warn_on_default_variance(dataset: Dataset) -> None:
    """Warn for channels whose every measured variance equals the unset default of 1.0."""
    for d, name in enumerate(dataset.output_channel_names):
        cells = [
            bool(jnp.all(bp.yvar[..., d][bp.mask[..., d]] == 1.0)) for bp in dataset.bucket_payloads
        ]
        if all(cells):
            warnings.warn(
                f"make_problem: every variance in channel {name!r} is 1.0, the default for "
                "unset data. If that is not the real noise, the posterior's width is wrong; "
                "set variance= on the data, or infer it with noise={...}.",
                UserWarning,
                stacklevel=3,
            )


def _check_flat_prior(predictors: Any, free_mask: Any, noise: Mapping[str, Parameter]) -> None:
    """Refuse ``flat_prior`` when a free value has no bounded box, as it can be improper."""
    leaves = jtu.tree_leaves(predictors, is_leaf=_is_parameter)
    flags = jtu.tree_leaves(free_mask, is_leaf=_is_parameter)
    named = [(leaf, flag) for leaf, flag in zip(leaves, flags, strict=True)]
    named += [(value, True) for value in noise.values()]
    for leaf, flag in named:
        if _is_parameter(leaf):
            open_box = any(
                not (math.isfinite(low) and math.isfinite(high)) for low, high in leaf.scaler.bounds
            )
            if open_box and (flag is True or flag.latent):
                raise ValueError(
                    "make_problem: flat_prior on a Parameter with an open bound can be improper; "
                    "write a log_prior or give it finite bounds"
                )
        elif flag:
            raise ValueError(
                "make_problem: flat_prior on network weights is improper; write a log_prior"
            )
