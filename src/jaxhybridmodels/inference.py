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

Without a sampler, :meth:`InferenceProblem.optimize` finds the maximum likelihood
or the posterior mode, :meth:`InferenceProblem.fisher` gives standard errors and the
directions the data cannot separate, :meth:`InferenceProblem.profile` profile
likelihood intervals, and :meth:`InferenceProblem.laplace` a Gaussian approximation
to the posterior. ``log_prior=None`` is enough for all but the last.

Three things make this density differ from the training loss.

- It is the likelihood itself. The trainers minimise ``Objective.loss``, which
  matches it only for ``loss="mle"`` with the data's variances: ``"mse"`` weighs
  every measurement the same whatever its variance.
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

The default target is the ``Parameter`` values. ``estimate=`` can name any subtree,
such as a network's last layer. Whatever stays fixed is conditioned on, and
``InferenceProblem.conditional_on`` lists it: a posterior over a physical
parameter with a fitted network held fixed is a conditional posterior, and the
network can absorb what the physics should explain.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from jaxtyping import Array
from scipy import stats

from jaxhybridmodels.data import BucketPayload, Data, Dataset
from jaxhybridmodels.losses import _gaussian_nll_terms, _masked_safe
from jaxhybridmodels.model import HybridModel
from jaxhybridmodels.prediction import _check_quantiles, _member_quantiles, predict_ensemble
from jaxhybridmodels.predictors import Parameter
from jaxhybridmodels.predictors.base import _is_parameter, _nan_on_bad_input
from jaxhybridmodels.results import PredictionResult
from jaxhybridmodels.rng import check_key
from jaxhybridmodels.solver import SolverConfig
from jaxhybridmodels.trainable import (
    FreezeSpec,
    _path_to_dotted,
    freeze_mask,
    trainable_mask,
)
from jaxhybridmodels.training.kernels import predict_bucket_obs

__all__ = (
    "DerivedQuantity",
    "FisherResult",
    "InferenceProblem",
    "LaplaceResult",
    "OptimizeResult",
    "ProfileResult",
    "flat_prior",
    "make_problem",
)

_PREDICTORS = "predictors"
_NOISE = "noise"

# gradient="auto" uses forward mode up to this many free values. Forward cost grows
# with the count, and reverse does not: on the tubular reactor forward took 8 ms
# against 25 ms at 2 values, and 368 ms against 187 ms at 17.
_FORWARD_MODE_LIMIT = 8

# fisher and laplace warn when a Newton step would still raise the log-likelihood or
# log-posterior by more than this, in log units. A tenth of a unit is well inside the
# 1.92 that sets a 95% profile interval, so a result past it is about a different point.
_OPTIMUM_TOLERANCE = 0.1

# An eigenvalue of the information below this fraction of the largest marks a direction
# the data does not constrain. Double precision resolves about 1e-16 of the largest, so
# this leaves room for solver noise.
_UNIDENTIFIED = 1e-8
# The scale-free test above sees only directions the data cannot separate. A value
# the data barely see, with tiny information and no correlation to anything, passes
# it. A latent standard error beyond this many latent units marks such a value
# unidentified too, for a value whose latent has a fixed scale: a box, where the
# squash spans its range within about +-10, or a log-warped one-sided bound, where
# a latent unit is a factor of e. An unbounded value or a one-sided linear one has
# its latent in physical units, where no fixed cutoff means anything, so only the
# collinearity test applies to it.
_WEAK = 1e3
# A profile point closes a side only when the log-likelihood could rise by no more
# than this by re-optimising the other values further: a hundredth of the 95%
# threshold's 1.92.
_NEWTON_DECREMENT = 0.02


def flat_prior(predictors: Any, noise: Mapping[str, Array]) -> Array:
    """A prior flat in physical units, for free ``Parameter``s with finite bounds.

    Proper only on a bounded box, so :func:`make_problem` refuses it when a free
    value has an open bound or is a network weight. A flat prior is not neutral
    under a warp: on a log-warped parameter it still favours the upper decades.
    """
    return jnp.asarray(0.0)


def _forward_value_and_grad(density: Callable[[Any], Array]) -> Callable[[Any], Any]:
    """``jax.value_and_grad(density)``, computed in forward mode from one solve."""

    def value_and_grad(position: Any) -> tuple[Array, Any]:
        def with_value(x: Any) -> tuple[Array, Array]:
            value = density(x)
            return value, value

        gradient, value = jax.jacfwd(with_value, has_aux=True)(position)
        return value, gradient

    return value_and_grad


def _guarded(density: Callable[[Any], Array], value_and_grad: Callable[[Any], Any]) -> Any:
    """``density``, with a zero gradient wherever it is not finite.

    A failed solve's sensitivities are NaN, and gating the value with ``jnp.where``
    cannot remove them from a gradient, since reverse mode multiplies them by zero.
    The rule depends on the value, so a NaN gradient at a finite density stays
    visible to the sampler. Samplers call ``jax.grad``, so this is also how forward
    mode hides behind a reverse-mode interface.
    """

    @jax.custom_vjp
    def guarded(position: Any) -> Array:
        return density(position)

    def forward(position: Any) -> tuple[Array, Any]:
        value, gradient = value_and_grad(position)
        keep = jnp.isfinite(value)
        return value, jtu.tree_map(lambda g: jnp.where(keep, g, jnp.zeros_like(g)), gradient)

    def backward(gradient: Any, cotangent: Array) -> tuple[Any]:
        return (jtu.tree_map(lambda g: g * cotangent, gradient),)

    guarded.defvjp(forward, backward)
    return guarded


def _linearizable(
    value_and_grad: Callable[[Array], tuple[Array, Array]],
) -> Callable[[Array], Array]:
    """A scalar function of a flat array whose JVP is its gradient times the tangent.

    Optimistix linearises its objective, which a ``custom_vjp`` such as
    :func:`_guarded` cannot be, and which a solve under a reverse-only adjoint
    cannot be either. This rule computes the gradient however ``value_and_grad``
    does, in forward or reverse mode, and the tangent map it returns is a dot
    product, which Optimistix can transpose. A non-finite value gets a zero
    gradient, so a failed solve looks like a wall to the line search.
    """

    @jax.custom_jvp
    def function(flat: Array) -> Array:
        return value_and_grad(flat)[0]

    @function.defjvp
    def function_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
        value, gradient = value_and_grad(primals[0])
        gradient = jnp.where(jnp.isfinite(value), gradient, jnp.zeros_like(gradient))
        return value, jnp.vdot(gradient, tangents[0])

    return function


def _log_abs_jacobian(parameter: Parameter) -> Array:
    """``sum log|d value / d latent|`` for one free ``Parameter``.

    The squash maps each latent to its own value, so a tangent of ones reads off
    the whole diagonal in one ``jvp``. Far into a squash's tail the derivative
    underflows to zero, and it is floored at the smallest positive float so the
    density stays finite and carries no gradient from this term.
    """
    latent = parameter.latent
    _, derivative = jax.jvp(parameter.scaler.from_latent, (latent,), (jnp.ones_like(latent),))
    size = jnp.abs(derivative)
    return jnp.sum(jnp.log(jnp.maximum(size, jnp.finfo(size.dtype).tiny)))


def _dotted(path: tuple[Any, ...]) -> str:
    """A leaf's path in the position, without the ``predictors`` level users never wrote."""
    return _path_to_dotted(path).removeprefix(f"{_PREDICTORS}.")


def _free_leaves(tree: Any, mask: Any) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """``(path, leaf)`` for the free leaves, with a ``Parameter`` as one whole leaf."""
    flat = jtu.tree_flatten_with_path(tree, is_leaf=_is_parameter)[0]
    flags = jtu.tree_leaves(mask, is_leaf=_is_parameter)
    for (path, leaf), flag in zip(flat, flags, strict=True):
        if flag.latent if _is_parameter(leaf) else flag:
            yield path, leaf


@dataclass(frozen=True, eq=False)
class InferenceProblem:
    """A model, its data, a prior and a noise model, as a function of a position.

    Build one with :func:`make_problem`. Identity-hashed, so bound methods such as
    ``problem.log_density`` can be passed to ``jax.jit`` and to samplers.

    Attributes
    ----------
    model : HybridModel
        The model as given. Solves in the density run with ``SolverConfig.throw`` off.
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
    gradient : str
        The mode in use, ``"forward"`` or ``"reverse"``, after ``"auto"`` chose.
    """

    model: HybridModel
    dataset: Dataset
    position: Any
    conditional_on: tuple[str, ...]
    gradient: str
    _log_prior: Callable[..., Array] | None
    _log_likelihood: Callable[..., Array] | None
    _static: Any
    _mask: Any
    _solver: SolverConfig
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
            simulate_fn=self.model.simulate_fn,
            state_to_output=self.model.state_to_output,
            solver=self._solver,
        )

    def _forward_view(self) -> InferenceProblem:
        """This problem with solves that support forward-mode differentiation."""
        solver = dataclasses.replace(self._solver, adjoint=diffrax.ForwardMode())
        return dataclasses.replace(self, _solver=solver)

    def _variance(self, bp: BucketPayload, noise: Mapping[str, Array]) -> Array:
        """The ``[N, T, D]`` observation variance: inferred if there is any, else the data's."""
        if not noise:
            return bp.variance
        sigma = jnp.stack([noise[name] for name in self.dataset.output_channel_names], axis=-1)
        return jnp.broadcast_to(sigma**2, bp.variance.shape)

    def residuals(self, position: Any) -> Array:
        """Whitened residuals ``(prediction - observed) / sigma``, flattened over buckets.

        Length is every ``[N, T, D]`` cell, with zero at unmeasured cells, so the
        shape does not depend on the data's mask. ``jax.jacfwd`` applies when the
        model's adjoint supports forward mode.
        """
        return self._whitened_residuals(position, noise_fixed=False)

    def _whitened_residuals(self, position: Any, *, noise_fixed: bool) -> Array:
        """``residuals``, with the noise held constant under differentiation if asked.

        ``noise_fixed`` stops the gradient through the variance, so ``J^T J``
        carries only the mean's information. The noise's own comes from the
        log-variance term in ``_curvature_residuals``.
        """
        tree = self._tree(position)
        noise = self._noise_values(tree)
        parts = []
        for bp in self.dataset.bucket_payloads:
            predicted, observed = _masked_safe(
                self._predictions(tree[_PREDICTORS], bp), bp.observed, bp.mask
            )
            variance = jnp.where(bp.mask, self._variance(bp, noise), 1.0)
            if noise_fixed:
                variance = jax.lax.stop_gradient(variance)
            parts.append(((predicted - observed) / jnp.sqrt(variance)).ravel())
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
                variance = self._variance(bp, noise)
                total = total - _gaussian_nll_terms(clean, bp.observed, variance, bp.mask).sum()
        return jnp.where(finite & jnp.isfinite(total), total, -jnp.inf)

    def _log_jacobian(self, tree: Mapping[str, Any]) -> Array:
        terms = [
            _log_abs_jacobian(leaf)
            for _, leaf in _free_leaves(tree, self._mask)
            if _is_parameter(leaf)
        ]
        return sum(terms, jnp.asarray(0.0))

    def _density(self, position: Any) -> Array:
        tree = self._tree(position)
        prior = self._log_prior(tree[_PREDICTORS], self._noise_values(tree))
        with _nan_on_bad_input():
            likelihood = self.log_likelihood(position)
        total = likelihood + prior + self._log_jacobian(tree)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    def _curvature_residuals(self, position: Any) -> Array:
        """``residuals``, plus the terms that make ``J^T J`` the Fisher information.

        A Gaussian's Fisher information has ``0.5 (d log var)^2`` per cell on top of
        the mean's ``J^T J``, which is the Jacobian of ``log var / sqrt(2)``. It is
        what gives inferred noise a curvature, however far its start is from its mode.
        The mean's residuals hold the noise fixed: through their division by
        ``sigma`` they would add ``r^2 (d log sigma)^2`` per cell, about one more
        unit on top of the two the log-variance term gives, and overstate the
        noise's information by half.
        """
        parts = [self._whitened_residuals(position, noise_fixed=True)]
        noise = self._noise_values(self._tree(position))
        if noise:
            parts += [
                (
                    jnp.where(bp.mask, jnp.log(self._variance(bp, noise)), 0.0) / jnp.sqrt(2.0)
                ).ravel()
                for bp in self.dataset.bucket_payloads
            ]
        return jnp.concatenate(parts)

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
        self._require_prior("log_density")
        if self.gradient == "forward":
            density = self._forward_view()._density
            return _guarded(density, _forward_value_and_grad(density))(position)
        return _guarded(self._density, jax.value_and_grad(self._density))(position)

    def preconditioned(self, *, precision: Any = None, ridge: float = 1.0) -> InferenceProblem:
        """This problem, sampling a whitened position so the target is close to a unit normal.

        ``position = fit + L z``, where ``L L^T`` is the inverse of the curvature
        at the current position. It is a constant linear map, so the posterior over
        the physical values is unchanged and only the sampler's geometry improves.
        Start from a fitted model: far from a mode the curvature says little, and
        sampling is then no better but still correct. A fit does not set inferred noise,
        and the other values' curvature scales with ``1 / sigma^2``, so a good starting
        guess for the noise helps too.

        Parameters
        ----------
        precision : Array | None
            The ``[P, P]`` curvature to use, for a custom likelihood or a better
            estimate. ``None`` computes the Gauss-Newton ``J^T J + ridge I`` from
            the whitened residuals and any inferred noise, which needs the Gaussian
            likelihood.
        ridge : float
            Added to the diagonal of the Gauss-Newton curvature in latent units,
            standing in for the prior so an unidentified direction keeps a finite
            scale. Ignored with ``precision``.
        """
        if self._whitening is not None:
            raise ValueError("preconditioned: this problem is already preconditioned")
        self._require_prior("preconditioned")
        flat, unravel = ravel_pytree(self.position)
        if precision is None:
            if self._log_likelihood is not None:
                raise ValueError(
                    "preconditioned: Gauss-Newton needs the Gaussian likelihood; pass precision= "
                    "for a custom log_likelihood"
                )
            precision = self._gauss_newton(flat, unravel) + ridge * jnp.eye(
                flat.size, dtype=flat.dtype
            )
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
        return dataclasses.replace(
            self, position=jnp.zeros_like(flat), _whitening=(flat, factor, unravel)
        )

    def _require_prior(self, caller: str) -> None:
        if self._log_prior is None:
            raise ValueError(
                f"{caller} needs a prior, and make_problem was given log_prior=None. Without "
                "one, optimize(target='likelihood'), fisher and profile are available."
            )

    def _unwhitened(self) -> InferenceProblem:
        """This problem in its own latent coordinates, undoing ``preconditioned``."""
        if self._whitening is None:
            return self
        shift, factor, unravel = self._whitening
        position = unravel(shift + factor @ self.position)
        return dataclasses.replace(self, position=position, _whitening=None)

    def _gauss_newton(self, flat: Array, unravel: Callable[[Array], Any]) -> Array:
        """``J^T J`` of the curvature residuals at ``flat``: the Fisher information of the Gaussian.

        P + 1 solves in forward mode, against one reverse pass per cell. Reverse is for
        a simulate_fn whose own adjoint has no forward rule.
        """
        if self.gradient == "forward" or _supports_forward(self):
            residuals = self._forward_view()._curvature_residuals
            jacobian = jax.jacfwd(lambda x: residuals(unravel(x)))(flat)
        else:
            jacobian = jax.jacrev(lambda x: self._curvature_residuals(unravel(x)))(flat)
        return jacobian.T @ jacobian

    def _scalar(
        self, target: str, unravel: Callable[[Array], Any], *, forward: bool
    ) -> Callable[[Array], Array]:
        """The log-likelihood or log-posterior as a function of the flat position."""
        problem = self._forward_view() if forward else self

        def value(flat: Array) -> Array:
            position = unravel(flat)
            if target == "likelihood":
                # A trial point can push a warped scaler's input out of range; that
                # is -inf for a line search to back away from, as in _density.
                with _nan_on_bad_input():
                    return problem.log_likelihood(position)
            return problem._density(position)

        return value

    def _target(self, position: Any) -> Array:
        """What the problem's own methods differentiate: the density, or the likelihood alone."""
        if self._log_prior is None:
            return self.log_likelihood(position)
        return self._density(position)

    def _value_and_grad(
        self, target: str, unravel: Callable[[Array], Any]
    ) -> Callable[[Array], tuple[Array, Array]]:
        """Value and gradient of the log-likelihood or log-posterior, in this problem's mode."""
        if self.gradient == "forward":
            return _forward_value_and_grad(self._scalar(target, unravel, forward=True))
        return jax.value_and_grad(self._scalar(target, unravel, forward=False))

    def _hessian(self, target: str, flat: Array, unravel: Callable[[Array], Any]) -> Array:
        """The Hessian of the log-likelihood or log-posterior at ``flat``.

        Forward over forward for the few values that use forward mode, and otherwise
        forward over reverse, which needs an adjoint that supports both, such as the
        default ``DirectAdjoint``.
        """
        if self.gradient == "forward":
            return jax.jacfwd(jax.jacfwd(self._scalar(target, unravel, forward=True)))(flat)
        try:
            return jax.hessian(self._scalar(target, unravel, forward=False))(flat)
        except (TypeError, ValueError, NotImplementedError) as exc:
            exc.add_note(
                "A Hessian differentiates the solve twice. Use diffrax.DirectAdjoint, the "
                "SolverConfig default, or the Gauss-Newton option where there is one."
            )
            raise

    def _objective(self, target: str, unravel: Callable[[Array], Any]) -> Callable[[Array], Array]:
        """Minus the log-likelihood or log-posterior, in a form Optimistix can linearise."""
        value_and_grad = self._value_and_grad(target, unravel)

        def negated(flat: Array) -> tuple[Array, Array]:
            value, gradient = value_and_grad(flat)
            return -value, -gradient

        return _linearizable(negated)

    def _information(
        self, *, caller: str, gauss_newton: bool, target: str, coordinates: _Coordinates
    ) -> Array:
        """The symmetric curvature of the log-likelihood or log-posterior, checked finite.

        Gauss-Newton is ``J^T J``, plus for the posterior the exact curvature of the
        prior and change-of-variables terms, which need no solve. Otherwise it is the
        negative Hessian.
        """
        flat, unravel = coordinates.flat, coordinates.unravel
        if gauss_newton:
            if self._log_likelihood is not None:
                raise ValueError(
                    f"{caller}: Gauss-Newton needs the Gaussian likelihood; with a custom "
                    "log_likelihood, use the Hessian option"
                )
            matrix = self._gauss_newton(flat, unravel)
            if target == "posterior":
                matrix = matrix - jax.hessian(lambda x: self._prior_terms(unravel(x)))(flat)
        else:
            matrix = -self._hessian(target, flat, unravel)
        matrix = 0.5 * (matrix + matrix.T)
        if not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError(
                f"{caller}: the curvature is not finite here. Check that the model solves at "
                "this position; in single precision, enable x64."
            )
        return matrix

    def optimize(
        self,
        *,
        target: Literal["likelihood", "posterior"] = "likelihood",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 1000,
    ) -> OptimizeResult:
        """Find the maximum of the log-likelihood, or of the log-posterior, with BFGS.

        Fisher information and profiles describe the likelihood at its maximum, and a
        Laplace approximation the posterior at its mode. A training fit is close but
        not there: the optimiser stops after a finite budget, and a loss other than
        ``"mle"`` has a different minimum when the variances differ. The search runs in the
        latent coordinates, so a ``Parameter`` stays inside its bounds; the posterior
        mode is therefore the mode of the density over latents.

        Uses ``optimistix.BFGS``. ``rtol`` and ``atol`` are its tolerances on the
        latent values and on the objective. The result's ``problem`` sits at the
        optimum and ``converged`` says whether BFGS met its tolerances.
        """
        if target not in ("likelihood", "posterior"):
            raise ValueError(
                f"optimize: target must be 'likelihood' or 'posterior'; got {target!r}"
            )
        problem = self._unwhitened()
        if target == "posterior":
            problem._require_prior("optimize(target='posterior')")
        flat, unravel = ravel_pytree(problem.position)
        if not bool(jnp.isfinite(problem._scalar(target, unravel, forward=False)(flat))):
            raise ValueError(
                "optimize: the model does not solve at the starting position, or the "
                f"log-{target} is not finite there; start from a fitted model"
            )
        objective = problem._objective(target, unravel)
        found = _maximise(
            lambda x, args: objective(x), flat, rtol=rtol, atol=atol, max_steps=max_steps
        )
        at_optimum = dataclasses.replace(problem, position=unravel(found.position))
        gradient = np.asarray(found.gradient)
        return OptimizeResult(
            problem=at_optimum,
            target=target,
            value=found.value,
            converged=found.converged,
            steps=found.steps,
            gradient=dict(zip(at_optimum._coordinates().names, gradient.tolist(), strict=True)),
            gradient_norm=float(np.max(np.abs(gradient), initial=0.0)),
        )

    def fisher(
        self, *, curvature: Literal["gauss_newton", "exact"] = "gauss_newton"
    ) -> FisherResult:
        """The Fisher information at the current position, with what it says about identifiability.

        Call it at the maximum likelihood, from :meth:`optimize`; it warns when the
        position is far from one. ``"gauss_newton"`` is ``J^T J`` of the whitened
        residuals, plus the noise terms when noise is inferred: the expected
        information of the Gaussian likelihood. ``"exact"`` is the negative
        Hessian of the log-likelihood, the observed information, which also works
        with a custom ``log_likelihood``. ``laplace`` takes the same choice.

        The matrices are in latent coordinates, where the problem lives. Standard
        errors and intervals are reported in physical units by default.
        """
        if curvature not in ("gauss_newton", "exact"):
            raise ValueError(
                f"fisher: curvature must be 'gauss_newton' or 'exact'; got {curvature!r}"
            )
        problem = self._unwhitened()
        coordinates = problem._coordinates()
        result = problem._fisher(coordinates, gauss_newton=curvature == "gauss_newton")
        problem._warn_if_not_at_optimum("fisher", "likelihood", coordinates, result.covariance)
        _warn_about_pinned("fisher", result.names, result.at_bound)
        return result

    def _fisher(self, coordinates: _Coordinates, *, gauss_newton: bool) -> FisherResult:
        matrix = self._information(
            caller="fisher", gauss_newton=gauss_newton, target="likelihood", coordinates=coordinates
        )
        covariance, eigenvalues, eigenvectors = _scaled_pseudo_inverse(matrix, _UNIDENTIFIED)
        return FisherResult(
            **coordinates.summary_fields(covariance),
            _to_position=self._position_from(coordinates),
            information=matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    def laplace(self, *, curvature: Literal["exact", "gauss_newton"] = "exact") -> LaplaceResult:
        """A Gaussian approximation to the posterior over latents, at the current position.

        Call it at the posterior mode, from ``optimize(target="posterior")``; it warns
        when the position is far from one. ``"exact"`` uses the Hessian of the log
        density. ``"gauss_newton"`` replaces the likelihood's part with ``J^T J``,
        which is cheaper for many values and always positive semi-definite, and keeps
        the exact Hessian of the prior and change-of-variables terms, which need no
        solve.

        ``sample`` draws positions that ``physical``, ``predict`` and ``to_arviz``
        take as they take a sampler's output.
        """
        if curvature not in ("exact", "gauss_newton"):
            raise ValueError(
                f"laplace: curvature must be 'exact' or 'gauss_newton'; got {curvature!r}"
            )
        problem = self._unwhitened()
        problem._require_prior("laplace")
        coordinates = problem._coordinates("posterior")
        precision = problem._information(
            caller="laplace",
            gauss_newton=curvature == "gauss_newton",
            target="posterior",
            coordinates=coordinates,
        )
        covariance, eigenvalues, eigenvectors = _pseudo_inverse(precision, 0.0)
        if float(eigenvalues[0]) <= 0.0:
            flattest = _loading_text(_loadings(eigenvectors[:, 0], coordinates.names))
            raise ValueError(
                "laplace: the log density is not concave here, so this is not a mode. Its "
                f"flattest direction loads on {flattest}. Run optimize(target='posterior') "
                "first; if this persists, that direction is not identified and needs a prior."
            )
        problem._warn_if_not_at_optimum("laplace", "posterior", coordinates, covariance)
        result = LaplaceResult(
            **coordinates.summary_fields(covariance), _to_position=self._position_from(coordinates)
        )
        _warn_about_pinned("laplace", result.names, result.at_bound)
        return result

    def _position_from(self, coordinates: _Coordinates) -> Callable[[Array], Any]:
        """Map a flat latent vector to a position of this problem, whitened or not.

        Draws must be readable by the problem they were asked of, so a whitened
        problem gets its whitened coordinates back.
        """
        if self._whitening is None:
            return coordinates.unravel
        shift, factor, _ = self._whitening
        return functools.partial(_whiten, shift, factor)

    def profile(
        self,
        name: str,
        *,
        confidence: float = 0.95,
        max_points: int = 40,
        max_steps: int = 200,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> ProfileResult:
        """The profile likelihood of one value, and the confidence interval it gives.

        Steps away from the current position, which should be the maximum
        likelihood from :meth:`optimize`, in each direction. At each point the
        named value is held fixed and BFGS maximises the log-likelihood over all
        the others, starting from the previous point. A side closes when twice the
        drop in log-likelihood passes the chi-square quantile for ``confidence``,
        and the interval's end is interpolated there. A side that reaches the
        value's bound, or runs out of ``max_points`` first, stays open: the data
        do not bound the value on that side. The walk is local: a second mode
        elsewhere is not seen. A point that fits better than the start warns,
        since the start was then not the maximum.

        Steps adapt so each raises the drop by about a tenth of the threshold, as
        in pyPESTO. They are taken in the latent coordinate, so profiling is
        unaffected by a warp. Profiles do not depend on the parametrisation at
        all, unlike Wald intervals.

        ``name`` is a path as ``physical`` names it, with ``[i]`` for an element
        of an array ``Parameter``, such as ``"k"`` or ``"rates[2]"``.
        """
        profiler = self._profiler(
            "profile",
            [name],
            confidence=confidence,
            max_points=max_points,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
        )
        return profiler.run(name)

    def profiles(
        self,
        names: Sequence[str] | None = None,
        *,
        confidence: float = 0.95,
        max_points: int = 40,
        max_steps: int = 200,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> dict[str, ProfileResult]:
        """:meth:`profile` for each name, by default every free ``Parameter`` entry.

        The setup, the Fisher step sizes and the compiled objective are shared, so
        each extra name costs only its own points. One that fails, for instance
        because the model stops solving far from the optimum, is skipped with a
        warning and the others still run.
        """
        profiler = self._profiler(
            "profiles",
            names,
            confidence=confidence,
            max_points=max_points,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
        )
        results = {}
        for name in profiler.parameters if names is None else names:
            try:
                results[name] = profiler.run(name)
            except (ValueError, FloatingPointError, eqx.EquinoxRuntimeError) as exc:
                warnings.warn(f"profiles: skipped {name!r}: {exc}", RuntimeWarning, stacklevel=2)
        return results

    def _profiler(
        self,
        caller: str,
        names: Sequence[str] | None,
        *,
        confidence: float,
        max_points: int,
        max_steps: int,
        rtol: float,
        atol: float,
    ) -> _Profiler:
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"{caller}: confidence must be in (0, 1); got {confidence}")
        if max_points < 2:
            raise ValueError(f"{caller}: max_points must be at least 2; got {max_points}")
        problem = self._unwhitened()
        coordinates = problem._coordinates()
        parameters = tuple(
            name
            for name, is_parameter in zip(coordinates.names, coordinates.is_parameter, strict=True)
            if is_parameter
        )
        unknown = [name for name in names or () if name not in parameters]
        if unknown:
            raise ValueError(
                f"{caller}: {unknown} are not free Parameter entries; "
                f"choose from {list(parameters)}"
            )
        fisher = problem._fisher(coordinates, gauss_newton=problem._log_likelihood is None)
        problem._warn_if_not_at_optimum(caller, "likelihood", coordinates, fisher.covariance)
        return _Profiler(
            coordinates=coordinates,
            parameters=parameters,
            # The raw latent spread, finite even where the summary reports inf:
            # it only sizes the first step, and a weakly seen value needs a long one.
            spreads=dict(
                zip(
                    fisher.names,
                    np.sqrt(np.maximum(np.diagonal(np.asarray(fisher.covariance)), 0.0)).tolist(),
                    strict=True,
                )
            ),
            objective=problem._profile_objective,
            best=float(
                problem._scalar("likelihood", coordinates.unravel, forward=False)(coordinates.flat)
            ),
            threshold=float(stats.chi2.ppf(confidence, df=1)),
            confidence=confidence,
            max_points=max_points,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
        )

    @functools.cached_property
    def _profile_objective(self) -> Callable[[Array, Any], Array]:
        """Minus the log-likelihood with one entry held fixed, given as ``(mask, value)``.

        The fixed entry is written over the full vector, so one compiled objective
        serves every name, every point and every later ``profile`` call on this
        problem: Optimistix caches its compile on the function's identity.
        """
        objective = self._objective("likelihood", ravel_pytree(self.position)[1])
        return lambda x, args: objective(jnp.where(args[0], args[1], x))

    def _prior_terms(self, position: Any) -> Array:
        """The log-prior plus the change-of-variables term: the density's solve-free part."""
        tree = self._tree(position)
        return self._log_prior(tree[_PREDICTORS], self._noise_values(tree)) + self._log_jacobian(
            tree
        )

    def _warn_if_not_at_optimum(
        self, caller: str, target: str, coordinates: _Coordinates, covariance: Array
    ) -> None:
        """Warn when a Newton step from here would still raise the objective by much.

        ``0.5 g^T C g`` is that rise in log units, so the test does not depend on the
        values' scale. An entry against its bound is left out: the maximum within
        the bounds is there, and its huge latent variance would turn a tiny residual
        slope into a large predicted rise that no step inside the box can deliver.
        """
        _, gradient = self._value_and_grad(target, coordinates.unravel)(coordinates.flat)
        pinned = coordinates.at_bound != ""
        gradient = jnp.where(pinned, 0.0, gradient)
        rise = 0.5 * float(gradient @ covariance @ gradient)
        if rise > _OPTIMUM_TOLERANCE:
            method = "optimize()" if target == "likelihood" else "optimize(target='posterior')"
            warnings.warn(
                f"{caller}: this position is not at the {target}'s maximum; a Newton step "
                f"would raise the log-{target} by about {rise:.3g}. The result describes a "
                f"maximum, so run {method} first.",
                UserWarning,
                stacklevel=3,
            )

    def _coordinates(self, target: str = "likelihood") -> _Coordinates:
        """The flat latent position with a name, a physical value and a slope per entry.

        ``target`` is the objective whose maximum the position is, which decides
        whether a value against its bound is held there by it.

        Entries follow the flat position, which is the tree order ``physical`` also
        walks. Each physical value depends on its own latent only, so the Jacobian is
        diagonal, and one JVP with a tangent of ones gives the delta method's slopes.
        """
        flat, unravel = ravel_pytree(self.position)
        names, is_parameter = [], []
        for path, leaf in _free_leaves(self._tree(self.position), self._mask):
            shape = np.shape(leaf.value if _is_parameter(leaf) else leaf)
            entries = (
                [""] if not shape else [f"[{','.join(map(str, i))}]" for i in np.ndindex(shape)]
            )
            names += [_dotted(path) + entry for entry in entries]
            is_parameter += [_is_parameter(leaf)] * len(entries)
        if len(names) != flat.size:
            raise ValueError(
                "InferenceProblem: each free value needs one latent entry per physical entry; "
                f"got {flat.size} latents for {len(names)} values"
            )
        to_physical = _PhysicalMap(static=self._static, mask=self._mask, unravel=unravel)
        estimate, slopes = jax.jvp(to_physical, (flat,), (jnp.ones_like(flat),))
        return _Coordinates(
            names=tuple(names),
            is_parameter=tuple(is_parameter),
            flat=flat,
            unravel=unravel,
            estimate=estimate,
            slopes=slopes,
            limits=to_physical.limits(flat),
            to_physical=to_physical,
            objective=functools.partial(self._scalar, target, unravel),
        )

    def to_model(self, position: Any) -> HybridModel:
        """The given model with this position's values, for ``predict`` and ``save``.

        It keeps the model's own solver, so its solves raise on failure as usual.
        Inferred noise is not part of a model.
        """
        return self.model.with_predictors(self._tree(position)[_PREDICTORS])

    def physical(self, position: Any) -> dict[str, Array]:
        """The free values in physical units, keyed by dotted path.

        A ``Parameter`` appears under its path with its ``.value``, a free network
        leaf under its path as it is, and inferred noise as ``"noise.<name>"``.
        ``position`` may be a sampler's output: leading axes such as
        ``[chain, draw]`` stay on every value.
        """
        reference = jtu.tree_leaves(self.position)[0]
        if jtu.tree_leaves(position)[0].ndim == reference.ndim:
            return self._physical_one(position)
        convert = self._physical_one
        for _ in range(self._sample_axes(position)):
            convert = jax.vmap(convert)
        return convert(position)

    def _physical_one(self, position: Any) -> dict[str, Array]:
        tree = self._tree(position)
        return {
            _dotted(path): leaf.value if _is_parameter(leaf) else leaf
            for path, leaf in _free_leaves(tree, self._mask)
        }

    def predict(
        self,
        positions: Any,
        data: Data | None = None,
        *,
        key: Array,
        ts: ArrayLike | None = None,
        n_points: int | None = None,
        n_draws: int = 200,
        quantiles: Sequence[float] = (0.025, 0.975),
        observation_noise: bool = False,
    ) -> PredictionResult:
        """Predict with draws from a chain and report their spread.

        ``positions`` is the sampler's output: the position pytree with any
        leading sample axes, such as ``[draw, ...]`` or ``[chain, draw, ...]``.
        ``n_draws`` of them are chosen at random, without replacement. ``data``
        defaults to the problem's own. ``ts`` and ``n_points`` are as for
        :meth:`HybridModel.predict`.

        ``outputs`` is the mean over draws. ``std`` and ``quantiles``, by default a
        central 95% band, describe the model's uncertainty: a credible band for the
        noiseless trajectory. With ``observation_noise``, they include measurement
        noise instead, an interval for a new measurement. That needs a standard
        deviation inferred for each channel, since the variances in data exist only
        at measured cells.

        A draw whose solve fails on this grid is dropped with a warning, so one bad
        draw on new data or a longer grid does not cost the others.
        """
        check_key(key, "InferenceProblem.predict")
        levels = _check_quantiles(quantiles, "InferenceProblem.predict")
        if n_draws < 1:
            raise ValueError(f"InferenceProblem.predict: n_draws must be positive; got {n_draws}")
        channels = self.dataset.output_channel_names
        if observation_noise and self._log_likelihood is not None:
            # A custom likelihood gives its noise entries its own meaning (a
            # variance, a Student-t scale, an AR coefficient), so adding them as
            # Gaussian standard deviations would draw from the wrong model.
            raise ValueError(
                "InferenceProblem.predict: observation_noise draws Gaussian noise, which a "
                "custom log_likelihood does not define; draw new measurements from your "
                "own noise model on the noiseless predictions instead"
            )
        if observation_noise and not set(channels) <= set(self._mask[_NOISE]):
            raise ValueError(
                "InferenceProblem.predict: observation_noise needs noise inferred for every "
                f"channel {list(channels)}; make_problem(noise={{...}}) provides it"
            )
        choice_key, noise_key = jr.split(key)
        draws = self._unstack(positions, n_draws, choice_key)
        members = [dataclasses.replace(self.to_model(draw), solver=self._solver) for draw in draws]

        def run(models: Sequence[HybridModel]) -> PredictionResult:
            return predict_ensemble(
                models,
                self.dataset if data is None else data,
                ts=ts,
                n_points=n_points,
                # Noise replaces the spread below, so it would go unread.
                quantiles=() if observation_noise else levels,
            )

        result = run(members)
        solved = [_is_finite(member) for member in result.members]
        if not all(solved):
            if not any(solved):
                raise ValueError("InferenceProblem.predict: every draw failed to solve")
            warnings.warn(
                f"InferenceProblem.predict: dropped {solved.count(False)} of {len(solved)} draws "
                "whose solve failed on this grid",
                RuntimeWarning,
                stacklevel=2,
            )
            draws = [draw for draw, kept in zip(draws, solved, strict=True) if kept]
            result = run([model for model, kept in zip(members, solved, strict=True) if kept])
        if not observation_noise:
            return result
        sigmas = [self._noise_values(self._tree(draw)) for draw in draws]
        return _with_observation_noise(result, sigmas, levels, noise_key)

    def _sample_axes(self, positions: Any) -> int:
        """How many leading sample axes ``positions`` has, raising if it has none."""
        reference = jtu.tree_leaves(self.position)[0]
        first = jtu.tree_leaves(positions)[0]
        extra = first.ndim - reference.ndim
        if extra < 1 or first.shape[extra:] != reference.shape:
            raise ValueError(
                f"InferenceProblem: positions need leading sample axes before the shape "
                f"{reference.shape}; got a leaf of shape {first.shape}"
            )
        return extra

    def _unstack(self, positions: Any, n_draws: int, key: Array) -> list[Any]:
        """``positions`` as a list of up to ``n_draws`` single positions."""
        extra = self._sample_axes(positions)
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
        if self._sample_axes(positions) != 2:
            raise ValueError("to_arviz: positions need exactly leading [chain, draw] axes")
        named = jax.vmap(jax.vmap(self.physical))(positions)
        groups = {"posterior": {name: np.asarray(value) for name, value in named.items()}}
        if sample_stats:
            groups["sample_stats"] = {k: np.asarray(v) for k, v in sample_stats.items()}
        # A string, as a NetCDF attribute cannot hold an empty list.
        conditioning = ", ".join(self.conditional_on)
        return from_dict(groups, attrs={"posterior": {"conditional_on": conditioning}})


def _whiten(shift: Array, factor: Array, flat: Array) -> Array:
    """The whitened ``z`` with ``flat = shift + factor z``, for an upper-triangular ``factor``."""
    return jax.scipy.linalg.solve_triangular(factor, flat - shift, lower=False)


@dataclass(frozen=True)
class _PhysicalMap:
    """The flat latent position to the free values in physical units, entry by entry.

    A class rather than a closure over the problem, so a result that keeps it does
    not also keep the problem's dataset alive.
    """

    static: Any
    mask: Any
    unravel: Callable[[Array], Any]

    def __call__(self, flat: Array) -> Array:
        return jnp.concatenate([jnp.ravel(value) for value in self.values(flat).values()])

    def values(self, flat: Array) -> dict[str, Array]:
        """The free values keyed by dotted path, as ``InferenceProblem.physical`` gives them."""
        tree = eqx.combine(self.unravel(flat), self.static)
        return {
            _dotted(path): leaf.value if _is_parameter(leaf) else leaf
            for path, leaf in _free_leaves(tree, self.mask)
        }

    def squash_positions(self, flat: Array) -> Array:
        """Each entry's place between its bounds, from 0 to 1, as ``BoundScaler.squash_position``.

        A free entry that is not a ``Parameter``, such as a network weight, has no
        bounds and reads 0.5.
        """
        tree = eqx.combine(self.unravel(flat), self.static)
        return jnp.concatenate(
            [
                jnp.ravel(leaf.scaler.squash_position(leaf.latent))
                if _is_parameter(leaf)
                else jnp.full(np.size(leaf), 0.5)
                for _, leaf in _free_leaves(tree, self.mask)
            ]
        )

    def with_position(self, flat: Array, index: int, position: Array) -> Array:
        """``flat`` with entry ``index`` moved to squash position ``position``."""
        tree = eqx.combine(self.unravel(flat), self.static)
        start = 0
        for _, leaf in _free_leaves(tree, self.mask):
            size = int(np.size(leaf.value if _is_parameter(leaf) else leaf))
            if start <= index < start + size:
                if not _is_parameter(leaf):
                    raise ValueError("with_position: only a Parameter entry has bounds")
                element = index - start
                component = 0 if len(leaf.scaler.bounds) == 1 else element
                return flat.at[index].set(leaf.scaler.latent_at_position(position, component))
            start += size
        raise IndexError(f"with_position: no entry {index}")

    def limits(self, flat: Array) -> tuple[np.ndarray, np.ndarray]:
        """Every entry's lower and upper bound in physical units, infinite when it has none.

        Read from each ``Parameter``'s scaler rather than by mapping infinite
        latents through the transform, which gives ``nan`` for squashes such as
        ``"algebraic"`` that reach their ends only in the limit.
        """
        tree = eqx.combine(self.unravel(flat), self.static)
        lows, highs = [], []
        for _, leaf in _free_leaves(tree, self.mask):
            size = int(np.size(leaf.value if _is_parameter(leaf) else leaf))
            pairs = np.array(leaf.scaler.bounds if _is_parameter(leaf) else [(-np.inf, np.inf)])
            pairs = np.broadcast_to(pairs, (size, 2))
            lows.append(pairs[:, 0])
            highs.append(pairs[:, 1])
        return np.concatenate(lows), np.concatenate(highs)

    def scaled(self, flat: Array) -> np.ndarray:
        """Entries whose latent has a fixed scale: a box, or a log-warped one-sided bound.

        Elsewhere, an unbounded value or a one-sided linear one, the latent is in
        the value's own units.
        """
        tree = eqx.combine(self.unravel(flat), self.static)
        flags = []
        for _, leaf in _free_leaves(tree, self.mask):
            size = int(np.size(leaf.value if _is_parameter(leaf) else leaf))
            if not _is_parameter(leaf):
                flags.append(np.zeros(size, dtype=bool))
                continue
            pairs = np.broadcast_to(np.array(leaf.scaler.bounds, dtype=float), (size, 2))
            box = np.isfinite(pairs).all(axis=1)
            one_sided = np.isfinite(pairs).sum(axis=1) == 1
            logged = leaf.scaler.warp in ("log", "log10")
            flags.append(box | (one_sided & logged))
        return np.concatenate(flags)


@dataclass(frozen=True)
class _Coordinates:
    """The flat latent position with a name, a physical value and a slope per entry."""

    names: tuple[str, ...]
    is_parameter: tuple[bool, ...]
    flat: Array
    unravel: Callable[[Array], Any]
    estimate: Array
    slopes: Array
    limits: tuple[np.ndarray, np.ndarray]
    to_physical: _PhysicalMap
    objective: Callable[..., Callable[[Array], Array]] | None = None

    def sides(self, flat: Array) -> np.ndarray:
        """Which bound each entry of a latent is pinned at: ``"lower"``, ``"upper"`` or ``""``.

        Pinned means the bound holds the value back: the data would move it
        past the bound. Closeness alone does not say that. A value of 0.9 in a
        box from 0 to 1e6 sits a millionth of the way along it, and is still a
        free interior optimum. So each entry in its squash's tail is tested on
        the squash's own ``[0, 1]`` scale, which is linear in the warped value:
        with the others held, a Newton step on the objective, free of the
        bound, would carry it past the edge. That needs no physical scale and
        no tolerance on the gradient, which vanishes in the tail either way.
        """
        positions = np.asarray(self.to_physical.squash_positions(flat))
        sides = np.full(positions.shape, "", dtype="<U5")
        tail = np.flatnonzero((positions < _TAIL) | (positions > 1.0 - _TAIL))
        if tail.size == 0 or self.objective is None:
            return sides
        for index in tail.tolist():
            lower = bool(positions[index] < 0.5)
            if _edge_side(float(positions[index])):
                # On the edge to rounding: nothing interior is that close, and
                # the derivatives there are not finite.
                sides[index] = "lower" if lower else "upper"
                continue
            slope, curvature = self._position_derivatives(flat, index, positions[index])
            outward = slope < 0.0 if lower else slope > 0.0
            gap = positions[index] if lower else 1.0 - positions[index]
            step = math.inf if curvature >= 0.0 else abs(slope / curvature)
            if outward and step > gap:
                sides[index] = "lower" if lower else "upper"
        return sides

    def _position_derivatives(
        self, flat: Array, index: int, position: float
    ) -> tuple[float, float]:
        """First and second derivative of the objective in one entry's squash position.

        A function of one variable, so forward over forward is cheapest, and it
        falls back to reverse mode for a model without forward mode.
        """
        assert self.objective is not None

        def along(at: Array, objective: Callable[[Array], Array]) -> Array:
            return objective(self.to_physical.with_position(flat, index, at))

        at = jnp.asarray(position, dtype=flat.dtype)
        try:
            objective = self.objective(forward=True)
            first = jax.jacfwd(lambda x: along(x, objective))
            values = (first(at), jax.jacfwd(first)(at))
        except (TypeError, ValueError, NotImplementedError):
            objective = self.objective(forward=False)
            first = jax.grad(lambda x: along(x, objective))
            values = (first(at), jax.grad(first)(at))
        return float(values[0]), float(values[1])

    @functools.cached_property
    def at_bound(self) -> np.ndarray:
        """``sides`` at the estimate."""
        return self.sides(self.flat)

    def summary_fields(self, covariance: Array) -> dict[str, Any]:
        """The fields every Gaussian result shares, around a latent ``covariance``.

        The covariance is conditioned on the entries pinned at their bounds,
        whose rows and columns become zero. Their latent spread comes from the
        squash's flat tail, and letting the other values trade off against it
        would report errors for them that no draw, and no fit with the bound as
        set, could show. Every result method then reads one covariance: the
        others' uncertainty given the bound.
        """
        pinned = np.asarray(self.at_bound != "")
        if pinned.any():
            free = ~pinned
            cross = covariance[np.ix_(free, pinned)]
            conditional = (
                covariance[np.ix_(free, free)]
                - cross @ jnp.linalg.pinv(covariance[np.ix_(pinned, pinned)]) @ cross.T
            )
            covariance = jnp.zeros_like(covariance).at[np.ix_(free, free)].set(conditional)
        return {
            "names": self.names,
            "estimate": self.estimate,
            "latent": self.flat,
            "covariance": covariance,
            "slopes": self.slopes,
            "at_bound": self.at_bound != "",
            "scaled": self.to_physical.scaled(self.flat),
            "limits": self.limits,
            "_to_physical": self.to_physical,
        }


class _Maximum(NamedTuple):
    position: Array
    value: float
    converged: bool
    steps: int
    gradient: Array


def _maximise(
    objective: Callable[[Array, Any], Array],
    start: Array,
    *,
    args: Any = None,
    rtol: float,
    atol: float,
    max_steps: int,
) -> _Maximum:
    """Minimise ``objective`` (minus a log density) with BFGS, and report the maximum.

    The value and the gradient come from the solver's state at the returned point,
    so they cost no extra solve. Both are of the maximised log density, so the
    signs are flipped from the minimised objective's.
    """
    solution = optx.minimise(
        objective,
        optx.BFGS(rtol=rtol, atol=atol),
        start,
        args=args,
        max_steps=max_steps,
        throw=False,
    )
    return _Maximum(
        position=solution.value,
        value=-float(solution.state.f_info.f),
        converged=bool(solution.result == optx.RESULTS.successful),
        steps=int(solution.stats["num_steps"]),
        gradient=-solution.state.f_info.grad,
    )


def _pseudo_inverse(matrix: Array, cutoff: float) -> tuple[Array, Array, Array]:
    """``(inverse, eigenvalues, eigenvectors)`` of a symmetric matrix.

    Directions with an eigenvalue at or below ``cutoff`` times the largest are left
    out of the inverse instead of dividing by a near-zero. Their variance is
    unbounded, which ``standard_errors`` reports as infinite rather than as the
    inverse's zero.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    largest = jnp.maximum(eigenvalues[-1], jnp.finfo(eigenvalues.dtype).tiny)
    kept = eigenvalues > cutoff * largest
    scale = jnp.where(kept, 1.0 / jnp.where(kept, eigenvalues, 1.0), 0.0)
    return (eigenvectors * scale) @ eigenvectors.T, eigenvalues, eigenvectors


def _unit_scale(matrix: Array) -> Array:
    """``1 / sqrt(diag)``, the scaling that gives ``matrix`` a unit diagonal.

    An entry with no information keeps a scale of 1: its row is zero anyway,
    so it shows up as a null direction along that entry alone.
    """
    diagonal = jnp.diagonal(matrix)
    positive = diagonal > 0.0
    return jnp.where(positive, 1.0 / jnp.sqrt(jnp.where(positive, diagonal, 1.0)), 1.0)


def _scaled_pseudo_inverse(matrix: Array, cutoff: float) -> tuple[Array, Array, Array]:
    """:func:`_pseudo_inverse` of the information scaled to a unit diagonal.

    The eigenvalues of the raw latent information depend on how each value
    reaches its latent: changing a bound or a warp rescales a row and column,
    which moves the condition number and can push an identified value under the
    cutoff. Scaling to a unit diagonal, the correlation form of the information,
    removes that, so the condition number and the identifiability test depend
    on the data and the model only. The returned eigenvectors are in the scaled
    coordinates, and the inverse is mapped back to the latents.
    """
    scale = _unit_scale(matrix)
    scaled = matrix * scale[:, None] * scale[None, :]
    inverse, eigenvalues, eigenvectors = _pseudo_inverse(scaled, cutoff)
    return inverse * scale[:, None] * scale[None, :], eigenvalues, eigenvectors


def _loadings(vector: Array, names: Sequence[str]) -> dict[str, float]:
    """The entries a direction loads on, largest first, leaving out loadings below 0.1."""
    values = np.asarray(vector)
    return {
        names[i]: float(values[i]) for i in np.argsort(-np.abs(values)) if abs(values[i]) >= 0.1
    }


def _loading_text(loadings: Mapping[str, float]) -> str:
    return ", ".join(f"{name} ({value:+.2f})" for name, value in loadings.items())


@dataclass(frozen=True)
class OptimizeResult:
    """What :meth:`InferenceProblem.optimize` found.

    Attributes
    ----------
    problem : InferenceProblem
        The problem positioned at the optimum, for ``fisher``, ``laplace`` or
        ``profile``. ``problem.to_model(problem.position)`` is the fitted model.
    target : str
        ``"likelihood"`` or ``"posterior"``.
    value : float
        The log-likelihood, or the log-posterior density over latents, at the optimum.
    converged : bool
        Whether BFGS met its tolerances within ``max_steps``.
    steps : int
        BFGS steps taken.
    gradient : dict[str, float]
        The gradient of the log-likelihood or log-posterior at the optimum, per
        entry as ``physical`` names it, taken in the latent coordinates the search
        runs in. Near zero at a maximum. An entry that stays large points at the
        value BFGS could not settle. A value against its bound shows a gradient
        near zero too, since the squash is flat there; ``fisher`` and the
        profile flag those.
    gradient_norm : float
        The largest entry of ``gradient`` in size, a one-number convergence check.
        How small is small depends on the curvature: ``fisher`` and ``laplace``
        apply the scale-free test, and warn when a Newton step from here would
        still raise the log density by more than 0.1.
    """

    problem: InferenceProblem
    target: str
    value: float
    converged: bool
    steps: int
    gradient: dict[str, float]
    gradient_norm: float


@dataclass(frozen=True)
class _GaussianSummary:
    """A Gaussian over the latent position, read in physical units.

    ``names`` labels every scalar entry, as ``physical`` names it, with ``[i]`` for
    an element of an array. ``estimate`` holds the physical values at the centre,
    ``latent`` the latent ones, and ``covariance`` the latent covariance, all in
    ``names`` order. ``slopes`` is ``d value / d latent`` per entry. ``at_bound``
    flags the entries whose estimate sits against a bound of their ``Parameter``,
    and ``limits`` holds every entry's lower and upper bounds, infinite without.
    """

    names: tuple[str, ...]
    estimate: Array
    latent: Array
    covariance: Array
    slopes: Array
    at_bound: np.ndarray = dataclasses.field(kw_only=True)
    scaled: np.ndarray = dataclasses.field(kw_only=True, repr=False)
    limits: tuple[np.ndarray, np.ndarray] = dataclasses.field(kw_only=True, repr=False)
    _to_physical: _PhysicalMap = dataclasses.field(repr=False)
    _to_position: Callable[[Array], Any] = dataclasses.field(repr=False, kw_only=True)

    def sample(self, *, key: Array, n_draws: int) -> Any:
        """``n_draws`` draws from this Gaussian, as positions with a leading axis of that size.

        ``problem.physical``, ``problem.predict`` and, with an extra chain axis,
        ``problem.to_arviz`` take them as they take a sampler's output. From a
        :class:`LaplaceResult` they approximate the posterior. From a
        :class:`FisherResult` they follow the estimate's asymptotic sampling
        distribution, which carries its uncertainty into any function of it, such as
        a prediction at new conditions.
        """
        check_key(key, f"{type(self).__name__}.sample")
        if n_draws < 1:
            raise ValueError(
                f"{type(self).__name__}.sample: n_draws must be positive; got {n_draws}"
            )
        unbounded = [
            name for name, flag in zip(self.names, self._unbounded().tolist(), strict=True) if flag
        ]
        if unbounded:
            raise ValueError(
                f"{type(self).__name__}.sample: the data do not constrain {unbounded}, so "
                "their spread is unbounded; fix or reparametrise them, or add a prior and "
                "use laplace"
            )
        # A value against its bound has no Gaussian spread: its latent sits in the
        # squash's tail, where a draw would land anywhere in the box. It is held
        # at its estimate, and the others are drawn from their Gaussian given it.
        pinned = np.asarray(self.at_bound, dtype=bool)
        covariance = self.covariance
        if pinned.any():
            names = [name for name, flag in zip(self.names, pinned, strict=True) if flag]
            warnings.warn(
                f"{type(self).__name__}.sample: {names} sit against their bounds, so they "
                "are held at their estimates and the others are drawn given them; "
                "profile them for their uncertainty",
                RuntimeWarning,
                stacklevel=2,
            )
            # The covariance is already conditioned on them (``summary_fields``),
            # so their rows are zero and every draw holds them at the estimate.
        # A square root from the eigendecomposition, which, unlike a Cholesky factor,
        # cannot fail on rounding in an ill-conditioned covariance.
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        factor = eigenvectors * jnp.sqrt(jnp.clip(eigenvalues, 0.0))
        draws = jr.normal(key, (n_draws, len(self.names)), dtype=self.latent.dtype)
        return jax.vmap(self._to_position)(self.latent + draws @ factor.T)

    def _unbounded(self) -> Array:
        """Entries with unbounded variance. None here; ``FisherResult`` finds some."""
        return jnp.zeros(len(self.names), dtype=bool)

    def _loose_directions(self) -> Array:
        """``[P, K]`` latent directions with unbounded variance. None here."""
        return jnp.zeros((len(self.names), 0))

    def _spread(self, covariance: Array) -> Array:
        spread = jnp.sqrt(jnp.clip(jnp.diagonal(covariance), 0.0))
        return jnp.where(self._unbounded(), jnp.inf, spread)

    def standard_errors(self, *, latent: bool = False) -> dict[str, float]:
        """The standard error of each entry: physical by the delta method, or latent.

        An entry that loads on a direction the data does not constrain has an
        infinite standard error. An entry against its bound has ``nan`` as its
        physical one: the curvature there describes the squash's flat tail, not
        how far the data let the value move, so no number would be honest.
        """
        if latent:
            spread = np.asarray(self._spread(self.covariance))
        else:
            spread = np.where(self.at_bound, np.nan, self._spread(self.physical_covariance()))
        return dict(zip(self.names, spread.tolist(), strict=True))

    def physical_covariance(self) -> Array:
        """The ``[P, P]`` covariance in physical units, by the delta method, in ``names`` order.

        ``S C S`` with ``S`` the diagonal of ``d value / d latent``. It describes the
        physical values only as far as their transforms are close to linear over
        the latent spread; ``intervals`` does not rely on that.
        """
        return self.slopes[:, None] * self.covariance * self.slopes[None, :]

    def correlation(self) -> Array:
        """The ``[P, P]`` correlation matrix in ``names`` order.

        Physical and latent correlations are the same here, since each physical
        value is a monotone function of its own latent and the delta method only
        rescales each entry. An entry with unbounded spread gets ``nan``: the
        pseudo-inverse describes only the directions the data constrain, so its
        correlation there could even have the wrong sign.
        """
        spread = jnp.sqrt(jnp.clip(jnp.diagonal(self.covariance), jnp.finfo(float).tiny))
        correlation = self.covariance / spread[:, None] / spread[None, :]
        # A pinned entry is left out too: its covariance is conditioned away.
        hidden = self._unbounded() | jnp.asarray(self.at_bound, dtype=bool)
        return jnp.where(hidden[:, None] | hidden[None, :], jnp.nan, correlation)

    def intervals(self, *, confidence: float = 0.95) -> dict[str, tuple[float, float]]:
        """A Wald interval per entry, in physical units.

        Built in latent space and mapped through each value's transform, so it is
        asymmetric on a log warp and stays inside a ``Parameter``'s bounds. An
        unconstrained entry's interval reaches its bounds, or infinity without one.
        An entry against its bound gets ``(nan, nan)``, as in ``standard_errors``;
        a profile tells whether the data bound it away from there.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"intervals: confidence must be in (0, 1); got {confidence}")
        z = float(stats.norm.ppf(0.5 + confidence / 2))
        unbounded = np.asarray(self._unbounded())
        # An unbounded entry's interval runs to its bounds, or plus and minus
        # infinity. Its reach is set to 0 here and replaced below, since an
        # infinite latent gives nan through some squashes.
        reach = jnp.where(unbounded, 0.0, z * self._spread(self.covariance))
        # Each value depends on its own latent only, so both ends map in one call each.
        ends = np.asarray(
            jnp.stack(
                [self._to_physical(self.latent - reach), self._to_physical(self.latent + reach)]
            )
        )
        low = np.where(unbounded, self.limits[0], ends.min(axis=0))
        high = np.where(unbounded, self.limits[1], ends.max(axis=0))
        low = np.where(self.at_bound, np.nan, low)
        high = np.where(self.at_bound, np.nan, high)
        return {name: (float(low[i]), float(high[i])) for i, name in enumerate(self.names)}

    def derived(self, fn: Callable[[dict[str, Array]], Any]) -> DerivedQuantity:
        """The estimate and standard error of a function of the values, by the delta method.

        ``fn`` takes the free values as ``physical`` gives them, a dict keyed by
        dotted path, and returns a scalar or an array, such as a shelf life from a
        rate constant or a prediction at a new condition. Its gradient comes from
        ``jax.jacobian``, so ``fn`` must be written in ``jax.numpy``. The standard
        error is ``sqrt(g^T C g)`` with ``C`` the physical covariance, and so is
        exact for a function linear in the values over their spread. For a
        strongly curved ``fn``, push draws from ``sample`` through it instead.

        A result that moves along a direction the data do not constrain has an
        infinite standard error. One the data fix is finite even when the values
        it combines are not, such as the product of two rates when the data see
        only the product. One that depends on an entry against its bound is
        ``nan``.
        """
        flat = jnp.asarray(self.estimate)
        template = self._to_physical.values(self.latent)
        sizes = [int(np.size(value)) for value in template.values()]

        def from_flat(values: Array) -> Array:
            parts = jnp.split(values, np.cumsum(sizes)[:-1])
            tree = {
                key: part.reshape(np.shape(old))
                for (key, old), part in zip(template.items(), parts, strict=True)
            }
            return jnp.asarray(fn(tree))

        estimate = from_flat(flat)
        gradient = jax.jacobian(from_flat)(flat).reshape(-1, flat.size)
        variance = jnp.einsum("op,pq,oq->o", gradient, self.physical_covariance(), gradient)
        uses = np.asarray(gradient != 0.0)
        spread = np.sqrt(np.clip(np.asarray(variance), 0.0, None))
        # A function of values the data cannot separate can still be fixed by
        # them, such as the product of two rates when only the product is seen.
        # Its spread is unbounded only if its latent gradient has a component
        # along a direction the data leave free; the pseudo-inverse gives the rest.
        # The loose direction found at a point on a ridge is only close to the
        # ridge's own, so an identified function shows a small component along
        # it; a thousandth of its size separates that from a real dependence.
        latent_gradient = np.asarray(gradient * self.slopes[None, :])
        loose = np.asarray(self._loose_directions())
        along = np.abs(latent_gradient @ loose).max(axis=1, initial=0.0)
        size = np.linalg.norm(latent_gradient, axis=1)
        spread = np.where(along > 1e-3 * np.maximum(size, np.finfo(float).tiny), np.inf, spread)
        spread = np.where((uses & self.at_bound).any(axis=1), np.nan, spread)
        return DerivedQuantity(
            estimate=np.asarray(estimate), standard_error=spread.reshape(np.shape(estimate))
        )

    def _table(self, title: str, *, confidence: float = 0.95) -> list[str]:
        errors = self.standard_errors()
        bounds = self.intervals(confidence=confidence)
        width = max(len(name) for name in self.names)
        lines = [
            title,
            f"  {'':{width}}  {'estimate':>11}  {'std error':>11}  "
            f"{round(confidence * 100)}% interval",
        ]
        for i, name in enumerate(self.names):
            low, high = bounds[name]
            if self.at_bound[i]:
                spread, reach = f"{'at bound':>11}", "not reported"
            elif math.isinf(errors[name]):
                # The interval would be the whole box, which is not information
                # from the data.
                spread, reach = f"{'inf':>11}", "not identified"
            else:
                spread, reach = f"{errors[name]:11.4g}", f"[{low:.4g}, {high:.4g}]"
            lines.append(f"  {name:{width}}  {float(self.estimate[i]):11.4g}  {spread}  {reach}")
        correlation = np.asarray(self.correlation())
        pairs = sorted(
            (
                (abs(correlation[i, j]), i, j)
                for i in range(len(self.names))
                for j in range(i + 1, len(self.names))
                if abs(correlation[i, j]) >= 0.5
            ),
            reverse=True,
        )[:5]
        if pairs:
            lines.append("  Strongest correlations")
            lines += [
                f"    {self.names[i]} and {self.names[j]}: {correlation[i, j]:+.3f}"
                for _, i, j in pairs
            ]
        return lines


@dataclass(frozen=True)
class FisherResult(_GaussianSummary):
    """The Fisher information at a position, from :meth:`InferenceProblem.fisher`.

    ``information`` is the latent ``[P, P]`` information, and ``covariance`` its
    inverse with unconstrained directions left out. ``eigenvalues`` and
    ``eigenvectors`` belong to the information scaled to a unit diagonal, its
    correlation form, so they do not change when a bound or a warp rescales a
    latent. ``eigenvalues`` ascend, and the columns of ``eigenvectors`` are the
    matching directions over ``names``.
    """

    information: Array = dataclasses.field(kw_only=True)
    eigenvalues: Array = dataclasses.field(kw_only=True)
    eigenvectors: Array = dataclasses.field(kw_only=True)

    @property
    def condition_number(self) -> float:
        """Largest over smallest eigenvalue; ``inf`` when the smallest is not positive."""
        smallest = float(self.eigenvalues[0])
        return float(self.eigenvalues[-1]) / smallest if smallest > 0.0 else math.inf

    def unidentified(self, *, threshold: float = _UNIDENTIFIED) -> list[dict[str, float]]:
        """The directions the data does not constrain, as ``{name: loading}`` per direction.

        A direction counts when its eigenvalue, in the correlation form of the
        information, is below ``threshold`` times the largest. Loadings below 0.1
        in size are left out, so each dict names the values that trade off along
        it, such as two rates whose product is all the data sees. A value the data
        barely see on its own is listed alone as ``{name: 1.0}`` when its latent
        has a fixed scale (a box, or a log-warped one-sided bound) and its latent
        standard error passes 1e3. An unbounded value has no such scale, so only
        its collinearity is tested.
        """
        # At or below, so an information that is exactly zero counts as loose.
        loose = np.asarray(self.eigenvalues <= threshold * self.eigenvalues[-1])
        directions = [_loadings(self.eigenvectors[:, k], self.names) for k in np.flatnonzero(loose)]
        # A direction along pinned values only is their flat squash, reported as
        # pinned instead.
        pinned = {name for name, flag in zip(self.names, self.at_bound, strict=True) if flag}
        directions = [d for d in directions if not set(d) <= pinned]
        listed = {name for direction in directions for name in direction}
        weak = np.asarray(self._weak())
        directions += [
            {name: 1.0}
            for name, flag in zip(self.names, weak, strict=True)
            if flag and name not in listed
        ]
        return directions

    def _weak(self) -> Array:
        """Entries whose latent standard error spans far more than any latent range.

        A value against its bound is left out: its latent spread is huge because
        the squash is flat there, not because the data cannot see it, and it is
        reported as pinned instead.
        """
        spread = jnp.sqrt(jnp.clip(jnp.diagonal(self.covariance), 0.0))
        bounded = jnp.asarray(self.scaled, dtype=bool)
        weak = bounded & (spread > _WEAK)
        return weak & ~jnp.asarray(self.at_bound, dtype=bool)

    def _loose_directions(self) -> Array:
        # Back from the scaled coordinates to latent directions, unit length, and
        # one axis per weakly seen entry.
        loose = np.asarray(self.eigenvalues <= _UNIDENTIFIED * self.eigenvalues[-1])
        directions = _unit_scale(self.information)[:, None] * self.eigenvectors[:, loose]
        directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)
        axes = jnp.eye(len(self.names), dtype=directions.dtype)[:, np.asarray(self._weak())]
        return jnp.concatenate([directions, axes], axis=1)

    def _unbounded(self) -> Array:
        # Any loading on an unconstrained direction makes that entry's variance
        # unbounded, not only the loadings large enough to list. 1e-3 is well above
        # the rounding in the eigenvectors of a near-null direction.
        loose = self.eigenvalues <= _UNIDENTIFIED * self.eigenvalues[-1]
        collinear = jnp.any(loose[None, :] & (jnp.abs(self.eigenvectors) >= 1e-3), axis=1)
        # A pinned value is reported as pinned, not as unconstrained: its flat
        # squash, not the data, is what leaves its own row near zero.
        pinned = jnp.asarray(self.at_bound, dtype=bool)
        return (collinear & ~pinned) | self._weak()

    def summary(self, *, confidence: float = 0.95) -> str:
        """Estimates, standard errors, Wald intervals, correlations and identifiability."""
        lines = self._table("Fisher information", confidence=confidence)
        lines.append(f"  Condition number {self.condition_number:.3g}")
        lines += [f"  Not identified: {_loading_text(d)}" for d in self.unidentified()]
        return "\n".join(lines)


@dataclass(frozen=True)
class LaplaceResult(_GaussianSummary):
    """A Gaussian approximation to the posterior, from :meth:`InferenceProblem.laplace`.

    ``latent`` is its mean and ``covariance`` its covariance, over the latent
    position in ``names`` order.
    """

    def summary(self, *, confidence: float = 0.95) -> str:
        """Posterior means, standard deviations, Wald intervals and correlations."""
        return "\n".join(self._table("Laplace approximation", confidence=confidence))


# Squash positions within this of 0 or 1 are in the tail, where a value may be
# pinned and ``_Coordinates.sides`` tests whether it is.
_TAIL = 1e-3
# Within this of 0 or 1 a profiled value has reached its bound for every
# purpose: a millionth of a millionth of the way along its warped box.
_AT_EDGE = 1e-12


def _edge_side(position: float) -> str:
    """``"lower"`` or ``"upper"`` when a squash position is at that edge, else ``""``."""
    if position <= _AT_EDGE:
        return "lower"
    if position >= 1.0 - _AT_EDGE:
        return "upper"
    return ""


def _warn_about_pinned(caller: str, names: Sequence[str], at_bound: np.ndarray) -> None:
    """Warn that the Gaussian results leave out entries against their bound."""
    pinned = [name for name, flag in zip(names, at_bound.tolist(), strict=True) if flag]
    if pinned:
        warnings.warn(
            f"{caller}: {pinned} sit against their bounds, so their standard errors and "
            "Wald intervals are nan: the curvature there says nothing about how far the "
            "data let them move. The other values are the best fit with these held at "
            "their bounds, so they and anything derived from them hold only for those "
            "bounds, and a value correlated with a pinned one is biased if its bound is "
            "wrong. Profile the pinned values to see whether the data bound them, and "
            "widen a bound that is not physical.",
            RuntimeWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class _Profiler:
    """Everything a profile needs that does not depend on which value is profiled."""

    coordinates: _Coordinates
    parameters: tuple[str, ...]
    spreads: dict[str, float]
    objective: Callable[[Array, Any], Array]
    best: float
    threshold: float
    confidence: float
    max_points: int
    max_steps: int
    rtol: float
    atol: float

    def run(self, name: str) -> ProfileResult:
        """Profile ``name``, which ``InferenceProblem._profiler`` has checked."""
        coordinates = self.coordinates
        index = coordinates.names.index(name)
        fixed_entry = jnp.arange(coordinates.flat.size) == index
        spread = self.spreads[name]
        first_step = 0.2 * spread if math.isfinite(spread) and spread > 0 else 0.1
        low, high = (float(limit[index]) for limit in coordinates.limits)
        positions = coordinates.to_physical.squash_positions
        if (math.isfinite(low) and math.isfinite(high)) or coordinates.at_bound[index]:
            # A latent unit already crosses much of a box. A value pressed against
            # its bound sits far out in the squash's tail, where its latent spread
            # is huge and meaningless, and a step sized from it would jump clean
            # across the box, or, for a one-sided bound, to a latent so large the
            # value overflows. Far from a one-sided bound the latent is near the
            # value itself, so its spread is kept there.
            first_step = min(first_step, 1.0)
        # Find which way the value moves when the latent rises: up for two bounds,
        # a lower one or none, down for an upper one alone. The squash position
        # moves the same way, except for an unbounded value, where it is flat.
        to_physical = coordinates.to_physical
        rising = float(to_physical(coordinates.flat.at[index].set(1.0))[index]) > float(
            to_physical(coordinates.flat.at[index].set(-1.0))[index]
        )

        start_value = float(coordinates.to_physical(coordinates.flat)[index])

        def reached_bound(x: Array, direction: float) -> bool:
            """Has the walk in ``direction`` reached the bound it is heading for?

            Only that bound counts. An estimate pinned at its upper bound must
            still walk down, since the data may bound it from below. Read on the
            squash's scale, so a one-sided bound counts too. The algebraic and
            softsign squashes approach their edges so slowly that a walk would
            run out of points first, so a value within 1e-5 of the estimate's
            distance to the bound counts as well.
            """
            target = "upper" if (direction > 0) == rising else "lower"
            if _edge_side(float(positions(x)[index])) == target:
                return True
            edge = high if target == "upper" else low
            if not math.isfinite(edge):
                return False
            value = float(coordinates.to_physical(x)[index])
            # Relative to how far the estimate sits from that bound, so a value
            # in small units, or one whose interval ends just short of the
            # bound, is not stopped early.
            return abs(value - edge) <= 1e-5 * abs(start_value - edge)

        def release(x: Array) -> Array:
            """``x`` with every other value that sits in its squash's tail moved to the tail's edge.

            Deep in the tail a value's gradient is about 1e-17, so BFGS never
            moves it, even where the data, with the profiled value held
            elsewhere, now want it off its bound. From the tail's edge the
            search can go either way: back to the bound if the data still
            push there, or inward.
            """
            for other, side in enumerate(coordinates.sides(x).tolist()):
                if other == index or not side:
                    continue
                edge = _TAIL if side == "lower" else 1.0 - _TAIL
                x = coordinates.to_physical.with_position(x, other, jnp.asarray(edge))
            return x

        def solve_at(x: Array, fixed: Array) -> _Maximum:
            found = _maximise(
                self.objective,
                release(x),
                args=(fixed_entry, fixed),
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
            )
            if not found.converged and math.isfinite(found.value):
                # One restart from where it stopped, before the point is flagged.
                found = _maximise(
                    self.objective,
                    found.position,
                    args=(fixed_entry, fixed),
                    rtol=self.rtol,
                    atol=self.atol,
                    max_steps=self.max_steps,
                )
            return found

        # Compiled once per profile, since every crossing calls it.
        objective_hessian = jax.jit(jax.hessian(self.objective))

        def settled(found: _Maximum, fixed: Array) -> bool:
            """Is ``found`` a maximum over the other values, not just a stalled search?

            BFGS stops when a step changes the objective by less than its
            tolerance, which a value the data barely constrain can satisfy far from
            its maximum: its gradient is tiny, but so is its curvature, so the
            distance to go is not. The Newton decrement ``g' H^-1 g / 2`` is how
            much the log-likelihood would still rise; a crossing counts only when
            it is small and the curvature is that of a maximum.
            """
            # A value pinned at its bound has a flat squash, so its curvature is
            # rounding noise either side of zero; it is not part of the question.
            free = ~np.asarray(fixed_entry) & (coordinates.sides(found.position) == "")
            if not free.any():
                return True
            try:
                hessian = np.asarray(objective_hessian(found.position, (fixed_entry, fixed)))[
                    np.ix_(free, free)
                ]
            except (TypeError, ValueError, NotImplementedError):
                # No second derivative through this solve: BFGS's own test stands.
                return True
            gradient = np.asarray(found.gradient)[free]
            if not np.all(np.isfinite(hessian)):
                return False
            # In the eigenbasis: a direction of clear positive curvature adds its
            # share of the decrement; a flat one (no information, as for a value
            # the model ignores) counts only if the gradient still pushes along
            # it; clear negative curvature is not a maximum.
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            along = eigenvectors.T @ gradient
            floor = 1e-10 * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if np.any(eigenvalues < -floor):
                return False
            curved = eigenvalues > floor
            if np.any(np.abs(along[~curved]) > 1e-6):
                return False
            decrement = 0.5 * float(np.sum(along[curved] ** 2 / eigenvalues[curved]))
            return decrement <= _NEWTON_DECREMENT

        def narrow(
            inside: Array, inside_ratio: float, outside: Array, outside_ratio: float
        ) -> list[tuple[Array, float, bool]]:
            """Bisect a wide bracket of the threshold, so the crossing is interpolated.

            The interpolation in :func:`end` is exact for a quadratic profile. A
            step can jump far past the threshold where the profile is not
            quadratic, for instance from a flat stretch where a bounded value sits
            against its bound into steep ground, and interpolating across that
            jump misplaces the crossing badly. Bisecting until both ends lie within
            a quarter of the threshold of it keeps the bracket small; an ordinary
            walk, whose steps aim at a tenth of the threshold, rarely needs any.
            Returns the new inside points, nearest the estimate first, and then
            the outside point.
            """
            insides: list[tuple[Array, float, bool]] = []
            for _ in range(30):
                if outside_ratio <= 1.25 * self.threshold and inside_ratio >= 0.75 * self.threshold:
                    break
                fixed = 0.5 * (inside[index] + outside[index])
                found = solve_at(inside, fixed)
                mid = jnp.where(fixed_entry, fixed, found.position)
                mid_ratio = 2.0 * (self.best - found.value)
                if not (found.converged and math.isfinite(mid_ratio)):
                    break
                if mid_ratio > self.threshold:
                    outside, outside_ratio = mid, mid_ratio
                else:
                    insides.append((mid, mid_ratio, True))
                    inside, inside_ratio = mid, mid_ratio
            return [*insides, (outside, outside_ratio, True)]

        higher_warned: list[bool] = []

        def walk(direction: float) -> tuple[list[tuple[Array, float, bool]], str]:
            """The points on one side, and why the walk stopped there."""
            points: list[tuple[Array, float, bool]] = []
            x, step, ratio = coordinates.flat, first_step, 0.0
            pinned = str(coordinates.at_bound[index])
            heading = "upper" if (direction > 0) == rising else "lower"
            # A value pinned at its bound can sit so deep in the squash's tail,
            # under a squash with a polynomial tail, that steps of a latent unit
            # would never get it out. Leaving the bound, the first step jumps to
            # the tail's edge; if that passes the threshold, ``narrow`` bisects
            # back to the crossing.
            leaving = bool(pinned) and pinned != heading
            if leaving:
                edge = _TAIL if pinned == "lower" else 1.0 - _TAIL
                inside = coordinates.to_physical.with_position(x, index, jnp.asarray(edge))
                step = max(abs(float(inside[index] - x[index])), first_step)
            for point in range(self.max_points // 2):
                # A long step can land where the model cannot be solved from the
                # previous point's other values; shorter ones may not.
                for _attempt in range(6):
                    fixed = x[index] + direction * step
                    found = solve_at(x, fixed)
                    new_ratio = 2.0 * (self.best - found.value)
                    if math.isfinite(new_ratio):
                        break
                    step *= 0.25
                new = jnp.where(fixed_entry, fixed, found.position)
                if not math.isfinite(new_ratio):
                    points.append((new, math.inf, False))
                    return points, "solve_failed"
                # A long step across a flat ridge can land where BFGS cannot
                # finish. Shorter steps, before the side is given up on.
                retries = 0
                while new_ratio > self.threshold and retries < 4:
                    if found.converged and settled(found, fixed):
                        break
                    retries += 1
                    step *= 0.25
                    fixed = x[index] + direction * step
                    found = solve_at(x, fixed)
                    new_ratio = 2.0 * (self.best - found.value)
                    new = jnp.where(fixed_entry, fixed, found.position)
                    if not math.isfinite(new_ratio):
                        points.append((new, math.inf, False))
                        return points, "solve_failed"
                if new_ratio < -0.1 and found.converged and not higher_warned:
                    higher_warned.append(True)
                    warnings.warn(
                        f"profile {name!r}: a point along the profile fits better than the "
                        f"estimate, by {-new_ratio / 2:.3g} in log-likelihood, so the start was "
                        "not the maximum. Optimize again from there; the interval here is "
                        "relative to the wrong maximum.",
                        RuntimeWarning,
                        stacklevel=4,
                    )
                if new_ratio > self.threshold:
                    if not (found.converged and settled(found, fixed)):
                        # A ratio from an unfinished search can only be too high, so
                        # it cannot show the data close this side.
                        points.append((new, new_ratio, False))
                        return points, "not_converged"
                    points += narrow(x, ratio, new, new_ratio)
                    return points, "threshold"
                points.append((new, new_ratio, found.converged))
                if reached_bound(new, direction):
                    return points, "bound"
                # Aim each step at a tenth of the threshold, never more than doubling.
                rise = max(new_ratio - ratio, 1e-12)
                if leaving and point == 0:
                    step = first_step
                step *= float(np.clip(0.1 * self.threshold / rise, 0.5, 2.0))
                x, ratio = new, new_ratio
            return points, "max_points"

        def end(side: list[tuple[Array, float, bool]], stopped: str) -> float | None:
            if stopped != "threshold":
                return None
            *inside, (outside_x, outside_ratio, _) = side
            converged = [(x, r) for x, r, ok in inside if ok]
            inside_x, inside_ratio = converged[-1] if converged else (coordinates.flat, 0.0)
            # The signed root of the ratio is linear in the value when the profile is
            # quadratic in it, so interpolating it is exact there. In the physical
            # value, not the latent: a linear-Gaussian model's profile is quadratic
            # in the value whatever squash or warp carries it to the latent.
            root, inside_root = math.sqrt(outside_ratio), math.sqrt(max(inside_ratio, 0.0))
            weight = (math.sqrt(self.threshold) - inside_root) / (root - inside_root)
            inside_value = float(coordinates.to_physical(inside_x)[index])
            outside_value = float(coordinates.to_physical(outside_x)[index])
            return inside_value + weight * (outside_value - inside_value)

        # Walk down and up in the latent, then order the two sides by physical
        # value: a value with only an upper bound falls as its latent rises.
        down, up = walk(-1.0), walk(1.0)
        (lower, stopped_lower), (upper, stopped_upper) = (down, up) if rising else (up, down)
        points = [*reversed(lower), (coordinates.flat, 0.0, True), *upper]
        physical = np.asarray(jax.vmap(coordinates.to_physical)(jnp.stack([p[0] for p in points])))
        converged = np.array([p[2] for p in points])
        # Another value pinned at any converged point could not move further to
        # compensate, so the drop there may be steeper than with the bound widened.
        sides = np.stack([coordinates.sides(p[0]) for p in points if p[2]])
        at_bound = {
            other: str(next(side for side in sides[:, i] if side))
            for i, other in enumerate(coordinates.names)
            if i != index and sides[:, i].any()
        }
        if at_bound:
            pinned = ", ".join(f"{other} ({side})" for other, side in at_bound.items())
            warnings.warn(
                f"profile {name!r}: {pinned} reached its bound along the profile, where "
                "re-optimising cannot move it further. The interval holds for these bounds; "
                "if one is not physical, widen it, since the profile may then close later.",
                RuntimeWarning,
                stacklevel=3,
            )
        return ProfileResult(
            name=name,
            confidence=self.confidence,
            threshold=self.threshold,
            grid=physical[:, index],
            log_likelihood_ratio=np.array([p[1] for p in points]),
            others={
                other: physical[:, i] for i, other in enumerate(coordinates.names) if i != index
            },
            converged=converged,
            bounds=(end(lower, stopped_lower), end(upper, stopped_upper)),
            stop_reasons=(stopped_lower, stopped_upper),
            at_bound=at_bound,
        )


@dataclass(frozen=True)
class ProfileResult:
    """The profile likelihood of one value, from :meth:`InferenceProblem.profile`.

    Attributes
    ----------
    name : str
        The profiled entry, as ``physical`` names it.
    confidence : float
        The level the interval is for.
    threshold : float
        The chi-square(1) quantile for ``confidence``: the interval is where
        ``log_likelihood_ratio`` stays below it.
    grid : ndarray
        The profiled value at each point, physical and ascending, the maximum
        likelihood included.
    log_likelihood_ratio : ndarray
        ``2 (max log L - log L)`` at each point, with every other value
        re-optimised.
    others : dict[str, ndarray]
        Every other free entry along the profile, physical. How they move shows
        what compensates for the profiled value.
    converged : ndarray
        Whether BFGS met its tolerances at each point, after one restart. The
        interval ignores points that did not.
    bounds : tuple[float | None, float | None]
        The confidence interval's ends, ``None`` on a side that did not close.
    stop_reasons : tuple[str, str]
        Why each side, below and above, stopped. ``"threshold"`` closed it. The
        others leave it open: ``"bound"`` (the value reached its bound, so the data
        do not bound it), ``"max_points"`` (still inside; raise ``max_points``),
        ``"solve_failed"`` (the model stopped solving) and ``"not_converged"``
        (BFGS did not converge where the ratio crossed).
    at_bound : dict[str, str]
        Other values that sat against a bound, ``"lower"`` or ``"upper"``, at some
        converged point. Re-optimising could not move them further, so the profile
        may drop faster than it would with that bound widened, and the interval
        holds for the bounds as set. Empty when nothing was pinned.
    """

    name: str
    confidence: float
    threshold: float
    grid: np.ndarray
    log_likelihood_ratio: np.ndarray
    others: dict[str, np.ndarray]
    converged: np.ndarray
    bounds: tuple[float | None, float | None]
    stop_reasons: tuple[str, str]
    at_bound: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def identifiable(self) -> bool:
        """Whether both sides closed at the threshold."""
        return self.stop_reasons == ("threshold", "threshold")

    def summary(self) -> str:
        """One line: the interval, and why a side that did not close stopped."""
        low, high = self.bounds
        shown = (
            f"[{'open' if low is None else f'{low:.4g}'}, "
            f"{'open' if high is None else f'{high:.4g}'}]"
        )
        notes = [
            f"{side} {_STOP_REASONS[reason]}"
            for side, reason in zip(("below", "above"), self.stop_reasons, strict=True)
            if reason != "threshold"
        ]
        unconverged = int((~self.converged).sum())
        if unconverged:
            notes.append(f"{unconverged} points did not converge")
        notes += [f"{other} pinned at its {side} bound" for other, side in self.at_bound.items()]
        note = "".join(f"; {text}" for text in notes)
        return f"{self.name}: {round(self.confidence * 100)}% profile interval {shown}{note}"


@dataclass(frozen=True)
class DerivedQuantity:
    """A function of the values with its delta-method standard error, from ``derived``.

    Attributes
    ----------
    estimate : ndarray
        The function at the estimate, in the shape ``fn`` returns.
    standard_error : ndarray
        Its standard error, same shape. ``inf`` where it depends on a value the
        data do not constrain, ``nan`` where it depends on one against its bound.
    """

    estimate: np.ndarray
    standard_error: np.ndarray

    def interval(
        self, *, confidence: float = 0.95, side: Literal["both", "lower", "upper"] = "both"
    ) -> tuple[np.ndarray, np.ndarray]:
        """A Wald interval, ``estimate`` plus or minus ``z`` standard errors.

        ``side="lower"`` gives a one-sided bound, ``(estimate - z se, inf)`` with
        ``z`` the ``confidence`` quantile, as a shelf-life claim needs;
        ``"upper"`` the mirror image.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"interval: confidence must be in (0, 1); got {confidence}")
        if side not in ("both", "lower", "upper"):
            raise ValueError(f"interval: side must be 'both', 'lower' or 'upper'; got {side!r}")
        tail = (1 - confidence) / 2 if side == "both" else 1 - confidence
        reach = float(stats.norm.ppf(1 - tail)) * self.standard_error
        infinite = np.full(np.shape(self.estimate), np.inf)
        low = -infinite if side == "upper" else self.estimate - reach
        high = infinite if side == "lower" else self.estimate + reach
        return np.asarray(low), np.asarray(high)


_STOP_REASONS = {
    "bound": "reaches its bound, so the data do not bound it",
    "max_points": "is still inside after max_points; raise it",
    "solve_failed": "stops where the model fails to solve",
    "not_converged": "crosses where BFGS did not converge",
}


def _is_finite(prediction: PredictionResult) -> bool:
    """Whether every output of one member's prediction is finite."""
    return all(
        bool(jnp.all(jnp.isfinite(values)))
        for experiment in prediction.values()
        for values in experiment.outputs.values()
    )


def _supports_forward(problem: InferenceProblem) -> bool:
    """Whether the density has a forward-mode rule, by tracing it without solving.

    The density itself has already been traced by ``make_problem``, so a failure
    here can only mean no such rule, as for a ``simulate_fn`` with its own adjoint.
    """
    try:
        eqx.filter_eval_shape(jax.jacfwd(problem._forward_view()._target), problem.position)
    except (TypeError, ValueError, NotImplementedError):
        return False
    return True


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
    scales = {name: jnp.stack([sigma[name] for sigma in sigmas])[:, None] for name in sigmas[0]}
    experiments = {}
    for exp_key, (exp_id, experiment) in zip(
        jr.split(key, len(result)), result.items(), strict=True
    ):
        noisy = {}
        channel_keys = jr.split(exp_key, len(experiment.outputs))
        for channel_key, name in zip(channel_keys, experiment.outputs, strict=True):
            means = jnp.stack([member[exp_id].outputs[name] for member in result.members])
            draws = jr.normal(channel_key, means.shape, dtype=means.dtype)
            noisy[name] = means + scales[name] * draws
        experiments[exp_id] = dataclasses.replace(
            experiment,
            std={name: jnp.std(values, axis=0) for name, values in noisy.items()},
            quantiles=_member_quantiles(noisy, levels),
        )
    return dataclasses.replace(result, experiments=experiments)


def make_problem(
    model: HybridModel,
    data: Data,
    *,
    log_prior: Callable[[Any, Mapping[str, Array]], Array] | None,
    noise: Literal["data"] | Mapping[str, Parameter],
    estimate: Sequence[FreezeSpec] | FreezeSpec | None = None,
    log_likelihood: Callable[[Array, BucketPayload, Mapping[str, Array]], Array] | None = None,
    gradient: Literal["auto", "forward", "reverse"] = "auto",
) -> InferenceProblem:
    """Build the log-density for ``model`` on ``data``.

    Solves inside the density run with ``SolverConfig.throw`` off. A ``simulate_fn``
    that calls ``diffrax.diffeqsolve`` itself must pass ``throw=solver.throw``, or
    a failed solve will end the chain with an exception.

    Parameters
    ----------
    model : HybridModel
        Its current values are the starting position, so a fitted model starts a
        chain near the mode.
    data : Dataset | Sequence[Experiment]
        The observations. Bucketing does not change the density.
    log_prior : Callable | None
        ``(predictors, noise) -> scalar``, the log-prior on physical values: read
        ``predictors["k"].value``, and ``noise["a"]`` for inferred noise. There is
        no default because no prior is neutral. :func:`flat_prior` is available
        for a bounded box. A free network's weights need a prior written here.
        ``None`` is for frequentist work: ``optimize``, ``fisher`` and ``profile``
        use only the likelihood, and the posterior methods then raise.
    noise : "data" | Mapping[str, Parameter]
        ``"data"`` trusts the variances in the data, and warns for a channel left
        at the default of 1.0. A mapping infers a standard deviation per channel,
        each a ``Parameter`` naming its starting value and bounds, which must keep it
        positive. With ``log_likelihood`` its keys are
        free-form, such as an AR(1) coefficient.
    estimate : str | type | Callable | Sequence | None
        What to infer, named in the syntax ``fit(freeze=...)`` uses, though it
        selects rather than excludes. ``None`` means every ``Parameter``. A dotted
        path can name a network part, such as ``"g.inner.mlp.layers.2"``.
    log_likelihood : Callable | None
        Replaces the Gaussian: ``(pred_obs, bp, noise) -> scalar``, called once per
        bucket and summed. ``pred_obs`` is ``[N, T, D]``, and ``bp.mask`` says
        which cells were measured.
    gradient : {"auto", "forward", "reverse"}
        How the density is differentiated. ``"auto"`` uses forward mode for a few
        free values when the model allows it, and otherwise reverse mode with the
        model's own adjoint. Forward mode replaces the adjoint with
        ``diffrax.ForwardMode``, so a ``simulate_fn`` must use ``solver.diffeqsolve``
        or pass ``adjoint=solver.adjoint``. ``problem.gradient`` says which was used.
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
    free_mask = _free_mask(predictors, base, estimate)
    if not any(jtu.tree_leaves(free_mask)):
        raise ValueError("make_problem: estimate= selects no trainable leaf; nothing to infer")
    tree = {_PREDICTORS: predictors, _NOISE: noise_map}
    mask = {_PREDICTORS: free_mask, _NOISE: trainable_mask(noise_map)}
    if log_prior is flat_prior:
        _check_flat_prior(tree, mask)

    # A Parameter is named as everywhere else, without its ``.latent`` field.
    conditional_on = tuple(
        dict.fromkeys(
            _path_to_dotted(path).removesuffix(".latent")
            for (path, in_base), in_free in zip(
                jtu.tree_flatten_with_path(base)[0], jtu.tree_leaves(free_mask), strict=True
            )
            if in_base and not in_free
        )
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

    position, static = eqx.partition(tree, mask)
    problem = InferenceProblem(
        model=model,
        dataset=dataset,
        position=position,
        conditional_on=conditional_on,
        gradient="reverse" if gradient == "auto" else gradient,
        _log_prior=log_prior,
        _log_likelihood=log_likelihood,
        _static=static,
        _mask=mask,
        _solver=dataclasses.replace(model.solver, throw=False),
    )
    # A bug in simulate_fn or log_prior raises here, not at the sampler's first step.
    eqx.filter_eval_shape(problem._target, position)
    if gradient == "forward":
        eqx.filter_eval_shape(jax.jacfwd(problem._forward_view()._target), position)
    few = ravel_pytree(position)[0].size <= _FORWARD_MODE_LIMIT
    if gradient == "auto" and few and _supports_forward(problem):
        return dataclasses.replace(problem, gradient="forward")
    return problem


def _free_mask(predictors: Any, base: Any, estimate: Any) -> Any:
    """``base`` narrowed to what ``estimate`` matches, reusing ``freeze_mask``'s matching."""
    if estimate is None:
        leaves = jtu.tree_leaves(predictors, is_leaf=_is_parameter)
        if not any(_is_parameter(leaf) for leaf in leaves):
            raise ValueError(
                "make_problem: this model has no Parameter to infer. Name a network part "
                "with estimate=, such as a last layer, or use fit_ensemble for "
                "whole-network uncertainty."
            )
        estimate = Parameter
    try:
        unmatched = freeze_mask(predictors, estimate, mask=base)
    except ValueError as exc:
        exc.add_note("make_problem: estimate= takes the same entries as fit(freeze=...).")
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
    if not custom:
        for name, value in noise.items():
            if value.scaler.bounds[0][0] < 0.0:
                raise ValueError(
                    f"make_problem: noise[{name!r}] is a standard deviation, so its bounds must "
                    f"keep it positive, as in bounds=(1e-4, 1.0); got {value.scaler.bounds[0]}"
                )
    if not custom and set(noise) != set(dataset.output_channel_names):
        raise ValueError(
            f"make_problem: noise names {sorted(noise)} but the channels are "
            f"{sorted(dataset.output_channel_names)}; the Gaussian needs one per channel"
        )
    return dict(noise)


def _warn_on_default_variance(dataset: Dataset) -> None:
    """Warn for channels whose every measured variance equals the unset default of 1.0."""
    for d, name in enumerate(dataset.output_channel_names):
        if all(
            bool(jnp.all(bp.variance[..., d][bp.mask[..., d]] == 1.0))
            for bp in dataset.bucket_payloads
        ):
            warnings.warn(
                f"make_problem: every variance in channel {name!r} is 1.0, the default for "
                "unset data. If that is not the real noise, the posterior's width is wrong; "
                "set variance= on the data, or infer it with noise={...}.",
                UserWarning,
                stacklevel=4,
            )


def _check_flat_prior(tree: Any, mask: Any) -> None:
    """Refuse ``flat_prior`` when a free value has no bounded box, as it can be improper."""
    for _, leaf in _free_leaves(tree, mask):
        if not _is_parameter(leaf):
            raise ValueError(
                "make_problem: flat_prior on network weights is improper; write a log_prior"
            )
        if any(not (math.isfinite(a) and math.isfinite(b)) for a, b in leaf.scaler.bounds):
            raise ValueError(
                "make_problem: flat_prior on a Parameter with an open bound can be improper; "
                "write a log_prior or give it finite bounds"
            )
