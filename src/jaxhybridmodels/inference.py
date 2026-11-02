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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

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
            return bp.yvar
        sigma = jnp.stack([noise[name] for name in self.dataset.output_channel_names], axis=-1)
        return jnp.broadcast_to(sigma**2, bp.yvar.shape)

    def residuals(self, position: Any) -> Array:
        """Whitened residuals ``(prediction - observed) / sigma``, flattened over buckets.

        Length is every ``[N, T, D]`` cell, with zero at unmeasured cells, so the
        shape does not depend on the data's mask. ``jax.jacfwd`` applies when the
        model's adjoint supports forward mode.
        """
        tree = self._tree(position)
        noise = self._noise_values(tree)
        parts = []
        for bp in self.dataset.bucket_payloads:
            predicted, observed = _masked_safe(
                self._predictions(tree[_PREDICTORS], bp), bp.y_observed, bp.mask
            )
            variance = jnp.where(bp.mask, self._variance(bp, noise), 1.0)
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
                total = total - _gaussian_nll_terms(clean, bp.y_observed, variance, bp.mask).sum()
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
        """
        parts = [self.residuals(position)]
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

    def _require_unwhitened(self, caller: str) -> None:
        if self._whitening is not None:
            raise ValueError(
                f"{caller}: call it on the problem from make_problem, not preconditioned"
            )

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

    def _hessian(
        self,
        value: Callable[[InferenceProblem, Any], Array],
        flat: Array,
        unravel: Callable[[Array], Any],
    ) -> Array:
        """The Hessian of ``value(problem, position)`` in the flat latent coordinates.

        Forward over forward for the few values that use forward mode, and otherwise
        forward over reverse, which needs an adjoint that supports both, such as the
        default ``DirectAdjoint``.
        """
        if self.gradient == "forward":
            view = self._forward_view()
            return jax.jacfwd(jax.jacfwd(lambda x: value(view, unravel(x))))(flat)
        try:
            return jax.hessian(lambda x: value(self, unravel(x)))(flat)
        except (TypeError, ValueError, NotImplementedError) as exc:
            exc.add_note(
                "A Hessian differentiates the solve twice. Use diffrax.DirectAdjoint, the "
                "SolverConfig default, or the Gauss-Newton option where there is one."
            )
            raise

    def _value_and_grad(
        self, target: str, unravel: Callable[[Array], Any]
    ) -> Callable[[Array], tuple[Array, Array]]:
        """Value and gradient of the log-likelihood or log-posterior, in this problem's mode."""

        def value(problem: InferenceProblem, flat: Array) -> Array:
            position = unravel(flat)
            if target == "likelihood":
                return problem.log_likelihood(position)
            return problem._density(position)

        if self.gradient == "forward":
            view = self._forward_view()
            return _forward_value_and_grad(lambda flat: value(view, flat))
        return jax.value_and_grad(lambda flat: value(self, flat))

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
        not there, because it averages the loss over buckets. The search runs in the
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
        self._require_unwhitened("optimize")
        if target == "posterior":
            self._require_prior("optimize(target='posterior')")
        flat, unravel = ravel_pytree(self.position)
        value_and_grad = self._value_and_grad(target, unravel)
        if not bool(jnp.isfinite(value_and_grad(flat)[0])):
            raise ValueError(
                "optimize: the model does not solve at the starting position, or the "
                f"log-{target} is not finite there; start from a fitted model"
            )

        def negated(x: Array) -> tuple[Array, Array]:
            value, gradient = value_and_grad(x)
            return -value, -gradient

        objective = _linearizable(negated)
        solution = optx.minimise(
            lambda x, args: objective(x),
            optx.BFGS(rtol=rtol, atol=atol),
            flat,
            max_steps=max_steps,
            throw=False,
        )
        return OptimizeResult(
            problem=dataclasses.replace(self, position=unravel(solution.value)),
            target=target,
            value=float(value_and_grad(solution.value)[0]),
            converged=bool(solution.result == optx.RESULTS.successful),
            steps=int(solution.stats["num_steps"]),
        )

    def fisher(
        self, *, information: Literal["gauss_newton", "observed"] = "gauss_newton"
    ) -> FisherResult:
        """The Fisher information at the current position, with what it says about identifiability.

        Call it at the maximum likelihood, from :meth:`optimize`; it warns when the
        position is far from one. ``"gauss_newton"`` is ``J^T J`` of the whitened
        residuals, plus the noise terms when noise is inferred: the expected
        information of the Gaussian likelihood. ``"observed"`` is the negative
        Hessian of the log-likelihood, which also works with a custom
        ``log_likelihood``.

        The matrices are in latent coordinates, where the problem lives. Standard
        errors and intervals are reported in physical units by default.
        """
        self._require_unwhitened("fisher")
        coordinates = self._coordinates()
        result = self._fisher(information, coordinates)
        self._warn_if_not_at_optimum("fisher", "likelihood", coordinates, result.covariance)
        return result

    def _fisher(self, information: str, coordinates: _Coordinates) -> FisherResult:
        flat, unravel = coordinates.flat, coordinates.unravel
        if information == "gauss_newton":
            if self._log_likelihood is not None:
                raise ValueError(
                    "fisher: information='gauss_newton' needs the Gaussian likelihood; use "
                    "information='observed' with a custom log_likelihood"
                )
            matrix = self._gauss_newton(flat, unravel)
        elif information == "observed":
            matrix = -self._hessian(
                lambda problem, position: problem.log_likelihood(position), flat, unravel
            )
        else:
            raise ValueError(
                f"fisher: information must be 'gauss_newton' or 'observed'; got {information!r}"
            )
        matrix = coordinates.reorder(matrix)
        if not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError(
                "fisher: the information is not finite here. Check that the model solves at "
                "this position; in single precision, enable x64."
            )
        covariance, eigenvalues, eigenvectors = _pseudo_inverse(matrix)
        return FisherResult(
            names=coordinates.names,
            estimate=coordinates.estimate,
            latent=coordinates.latent,
            covariance=covariance,
            _to_physical=coordinates.to_physical,
            information=matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    def laplace(self, *, hessian: Literal["exact", "gauss_newton"] = "exact") -> LaplaceResult:
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
        self._require_unwhitened("laplace")
        self._require_prior("laplace")
        coordinates = self._coordinates()
        flat, unravel = coordinates.flat, coordinates.unravel
        if hessian == "exact":
            precision = -self._hessian(
                lambda problem, position: problem._density(position), flat, unravel
            )
        elif hessian == "gauss_newton":
            if self._log_likelihood is not None:
                raise ValueError(
                    "laplace: hessian='gauss_newton' needs the Gaussian likelihood; use "
                    "hessian='exact' with a custom log_likelihood"
                )
            prior_curvature = jax.hessian(lambda x: self._prior_terms(unravel(x)))(flat)
            precision = self._gauss_newton(flat, unravel) - prior_curvature
        else:
            raise ValueError(f"laplace: hessian must be 'exact' or 'gauss_newton'; got {hessian!r}")
        precision = coordinates.reorder(precision)
        precision = 0.5 * (precision + precision.T)
        if not bool(jnp.all(jnp.isfinite(precision))):
            raise ValueError(
                "laplace: the Hessian is not finite here. Check that the model solves at this "
                "position; in single precision, enable x64."
            )
        eigenvalues, eigenvectors = jnp.linalg.eigh(precision)
        if float(eigenvalues[0]) <= 0.0:
            flattest = _loading_text(eigenvectors[:, 0], coordinates.names)
            raise ValueError(
                "laplace: the log density is not concave here, so this is not a mode. Its "
                f"flattest direction loads on {flattest}"
                ". Run optimize(target='posterior') first; if this persists, that direction is "
                "not identified and needs a prior."
            )
        covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
        self._warn_if_not_at_optimum("laplace", "posterior", coordinates, covariance)
        return LaplaceResult(
            names=coordinates.names,
            estimate=coordinates.estimate,
            latent=coordinates.latent,
            covariance=covariance,
            _to_physical=coordinates.to_physical,
            _to_positions=coordinates.to_positions,
        )

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
        do not bound the value on that side.

        Steps adapt so each raises the drop by about a tenth of the threshold, as
        in pyPESTO. They are taken in the latent coordinate, so profiling is
        unaffected by a warp. Profiles do not depend on the parametrisation at
        all, unlike Wald intervals.

        ``name`` is a path as ``physical`` names it, with ``[i]`` for an element
        of an array ``Parameter``, such as ``"k"`` or ``"rates[2]"``.
        """
        self._require_unwhitened("profile")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"profile: confidence must be in (0, 1); got {confidence}")
        if max_points < 2:
            raise ValueError(f"profile: max_points must be at least 2; got {max_points}")
        coordinates = self._coordinates()
        parameters = {
            entry for entry in coordinates.names if entry.split("[")[0] in self._parameter_paths()
        }
        if name not in parameters:
            raise ValueError(
                f"profile: {name!r} is not a free Parameter entry; choose from {sorted(parameters)}"
            )
        index = coordinates.names.index(name)
        threshold = float(stats.chi2.ppf(confidence, df=1))
        fisher = self._fisher(
            "gauss_newton" if self._log_likelihood is None else "observed", coordinates
        )
        self._warn_if_not_at_optimum("profile", "likelihood", coordinates, fisher.covariance)
        spread = fisher.standard_errors(latent=True)[name]
        first_step = 0.2 * spread if math.isfinite(spread) and spread > 0 else 0.1

        inverse = np.argsort(coordinates.order)
        unravel = coordinates.unravel

        def from_names(x: Array) -> Any:
            return unravel(x[inverse])

        value_and_grad = self._value_and_grad("likelihood", lambda flat: unravel(flat))

        def negated(x: Array) -> tuple[Array, Array]:
            value, gradient = value_and_grad(x[inverse])
            return -value, -gradient[coordinates.order]

        objective = _linearizable(negated)

        def with_fixed(rest: Array, fixed: Array) -> Array:
            return jnp.concatenate([rest[:index], fixed[None], rest[index:]])

        def fixed_objective(rest: Array, fixed: Array) -> Array:
            return objective(with_fixed(rest, fixed))

        solver = optx.BFGS(rtol=rtol, atol=atol)
        log_likelihood = eqx.filter_jit(lambda x: self.log_likelihood(from_names(x)))
        centre = coordinates.latent
        best = float(log_likelihood(centre))

        def walk(direction: float) -> list[tuple[Array, float, bool]]:
            points: list[tuple[Array, float, bool]] = []
            x, step, ratio = centre, first_step, 0.0
            for _ in range(max_points // 2):
                fixed = x[index] + direction * step
                solution = optx.minimise(
                    fixed_objective,
                    solver,
                    jnp.delete(x, index),
                    args=fixed,
                    max_steps=max_steps,
                    throw=False,
                )
                new = with_fixed(solution.value, fixed)
                new_ratio = 2.0 * (best - float(log_likelihood(new)))
                converged = bool(solution.result == optx.RESULTS.successful)
                moved = abs(
                    float(coordinates.to_physical(new)[index] - coordinates.to_physical(x)[index])
                )
                points.append((new, new_ratio, converged))
                if new_ratio > threshold:
                    break
                scale = 1.0 + abs(float(coordinates.to_physical(new)[index]))
                if moved < 1e-10 * scale:
                    break  # at the value's bound: the squash no longer moves it
                # Aim each step at a tenth of the threshold, never more than doubling.
                rise = max(new_ratio - ratio, 1e-12)
                step *= float(np.clip(0.1 * threshold / rise, 0.5, 2.0))
                x, ratio = new, new_ratio
            return points

        below, above = walk(-1.0), walk(1.0)
        points = [*reversed(below), (centre, 0.0, True), *above]
        latents = jnp.stack([point[0] for point in points])
        physical = jax.vmap(coordinates.to_physical)(latents)
        ratios = np.array([point[1] for point in points])
        converged = np.array([point[2] for point in points])

        def end(side: list[tuple[Array, float, bool]]) -> float | None:
            inside_x, inside_ratio = centre, 0.0
            for x, ratio, ok in side:
                if not ok:
                    continue
                if ratio > threshold:
                    # The signed root of the ratio is linear in the value when the
                    # profile is quadratic, so interpolating it is exact there.
                    root, inside_root = math.sqrt(ratio), math.sqrt(max(inside_ratio, 0.0))
                    weight = (math.sqrt(threshold) - inside_root) / (root - inside_root)
                    crossing = inside_x + weight * (x - inside_x)
                    return float(coordinates.to_physical(crossing)[index])
                inside_x, inside_ratio = x, ratio
            return None

        return ProfileResult(
            name=name,
            confidence=confidence,
            threshold=threshold,
            values=np.asarray(physical[:, index]),
            log_likelihood_ratio=ratios,
            others={
                other: np.asarray(physical[:, i])
                for i, other in enumerate(coordinates.names)
                if i != index
            },
            converged=converged,
            interval=(end(below), end(above)),
        )

    def profiles(
        self, names: Sequence[str] | None = None, **options: Any
    ) -> dict[str, ProfileResult]:
        """:meth:`profile` for each name, by default every free ``Parameter`` entry.

        Each profile is independent, so one that fails, for instance because the
        model stops solving far from the optimum, is skipped with a warning and
        the others still run. ``options`` go to :meth:`profile`.
        """
        if names is None:
            names = [
                entry
                for entry in self._coordinates().names
                if entry.split("[")[0] in self._parameter_paths()
            ]
        results = {}
        for name in names:
            try:
                results[name] = self.profile(name, **options)
            except (ValueError, FloatingPointError, eqx.EquinoxRuntimeError) as exc:
                warnings.warn(f"profiles: skipped {name!r}: {exc}", RuntimeWarning, stacklevel=2)
        return results

    def _parameter_paths(self) -> set[str]:
        """The ``physical`` names of the free ``Parameter``s, inferred noise included."""
        tree = self._tree(self.position)
        return {
            _dotted(path) for path, leaf in _free_leaves(tree, self._mask) if _is_parameter(leaf)
        }

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
        values' scale.
        """
        _, gradient = self._value_and_grad(target, coordinates.unravel)(coordinates.flat)
        gradient = gradient[coordinates.order]
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

    def _coordinates(self) -> _Coordinates:
        """The flat latent position with a name, a physical value and a slope per entry.

        Entries are ordered as ``physical`` names them, sorted by path. Each physical
        value depends on its own latent only, so the Jacobian is diagonal and its
        diagonal is the delta method's slope.
        """
        flat, unravel = ravel_pytree(self.position)
        physical = self._physical_one(self.position)
        names = _entry_names(physical)
        # Where each leaf's entries sit in the flat position. A scalar Parameter keeps
        # a (1,) latent for a () value, so leaves are paired by path and size.
        offsets: dict[str, tuple[int, int]] = {}
        start = 0
        for path, leaf in jtu.tree_flatten_with_path(self.position)[0]:
            offsets[_dotted(path).removesuffix(".latent")] = (start, int(np.size(leaf)))
            start += int(np.size(leaf))
        order_parts = []
        for path in sorted(physical):
            begin, size = offsets.get(path, (0, -1))
            if size != int(np.size(physical[path])):
                raise ValueError(
                    f"InferenceProblem: could not pair the physical value {path!r} with its "
                    "latent; each free value needs one latent entry per physical entry"
                )
            order_parts.append(np.arange(begin, begin + size))
        order = np.concatenate(order_parts)
        # The flat (ravel) position of each named entry, and the inverse map.
        inverse = np.argsort(order)

        def to_physical(x: Array) -> Array:
            return ravel_pytree(self._physical_one(unravel(x[inverse])))[0]

        latent = flat[order]
        slopes = jnp.diagonal(jax.jacfwd(to_physical)(latent))
        return _Coordinates(
            names=tuple(names),
            order=order,
            flat=flat,
            unravel=unravel,
            latent=latent,
            estimate=to_physical(latent),
            slopes=slopes,
            to_physical=to_physical,
            to_positions=jax.vmap(lambda x: unravel(x[inverse])),
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

        A draw whose solve fails on this grid is dropped with a warning, so one bad
        draw on new data or a longer grid does not cost the others.
        """
        check_key(key, "InferenceProblem.predict")
        levels = _check_quantiles(quantiles, "InferenceProblem.predict")
        if n_draws < 1:
            raise ValueError(f"InferenceProblem.predict: n_draws must be positive; got {n_draws}")
        channels = self.dataset.output_channel_names
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


@dataclass(frozen=True)
class _Coordinates:
    """The flat latent position, ordered and named as ``InferenceProblem.physical`` names it."""

    names: tuple[str, ...]
    order: np.ndarray
    flat: Array
    unravel: Callable[[Array], Any]
    latent: Array
    estimate: Array
    slopes: Array
    to_physical: Callable[[Array], Array]
    to_positions: Callable[[Array], Any]

    def reorder(self, matrix: Array) -> Array:
        """A matrix over the flat position, reordered to ``names``."""
        return matrix[np.ix_(self.order, self.order)]


def _entry_names(values: Mapping[str, Any]) -> list[str]:
    """One name per scalar entry, in ``ravel_pytree``'s order: sorted paths, then C order."""
    names = []
    for path in sorted(values):
        shape = np.shape(values[path])
        if not shape:
            names.append(path)
        else:
            names += [f"{path}[{','.join(map(str, index))}]" for index in np.ndindex(shape)]
    return names


def _pseudo_inverse(matrix: Array) -> tuple[Array, Array, Array]:
    """``(inverse, eigenvalues, eigenvectors)`` of a symmetric PSD matrix.

    Directions below ``_UNIDENTIFIED`` of the largest eigenvalue are left out of the
    inverse instead of dividing by a near-zero. Their variance is unbounded, which
    ``standard_errors`` reports as infinite rather than as the inverse's zero.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    largest = jnp.maximum(eigenvalues[-1], jnp.finfo(eigenvalues.dtype).tiny)
    kept = eigenvalues > _UNIDENTIFIED * largest
    scale = jnp.where(kept, 1.0 / jnp.where(kept, eigenvalues, 1.0), 0.0)
    return (eigenvectors * scale) @ eigenvectors.T, eigenvalues, eigenvectors


def _loading_text(vector: Array, names: Sequence[str]) -> str:
    """The names a direction loads on most, with their loadings, for a message."""
    loadings = np.asarray(vector)
    picked = [i for i in np.argsort(-np.abs(loadings)) if abs(loadings[i]) >= 0.1][:4]
    return ", ".join(f"{names[i]} ({loadings[i]:+.2f})" for i in picked)


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
    """

    problem: InferenceProblem
    target: str
    value: float
    converged: bool
    steps: int


@dataclass(frozen=True)
class _GaussianSummary:
    """A Gaussian over the latent position, read in physical units.

    ``names`` labels every scalar entry, as ``physical`` names it, with ``[i]`` for
    an element of an array. ``estimate`` holds the physical values at the centre,
    ``latent`` the latent ones, and ``covariance`` the latent covariance, all in
    ``names`` order.
    """

    names: tuple[str, ...]
    estimate: Array
    latent: Array
    covariance: Array
    _to_physical: Callable[[Array], Array] = dataclasses.field(repr=False)

    def _unbounded(self) -> Array:
        """Entries with unbounded variance. None here; ``FisherResult`` finds some."""
        return jnp.zeros(len(self.names), dtype=bool)

    def standard_errors(self, *, latent: bool = False) -> dict[str, float]:
        """The standard error of each entry: physical by the delta method, or latent.

        An entry that loads on a direction the data does not constrain has an
        infinite standard error.
        """
        covariance = self.covariance if latent else self.physical_covariance()
        spread = jnp.sqrt(jnp.clip(jnp.diagonal(covariance), 0.0))
        spread = jnp.where(self._unbounded(), jnp.inf, spread)
        return {name: float(value) for name, value in zip(self.names, spread, strict=True)}

    def physical_covariance(self) -> Array:
        """The ``[P, P]`` covariance in physical units, by the delta method, in ``names`` order.

        ``S C S`` with ``S`` the diagonal of ``d value / d latent``. It describes the
        physical values only as far as their transforms are close to linear over
        the latent spread; ``intervals`` does not rely on that.
        """
        slopes = jnp.diagonal(jax.jacfwd(self._to_physical)(self.latent))
        return slopes[:, None] * self.covariance * slopes[None, :]

    def correlation(self) -> Array:
        """The ``[P, P]`` correlation matrix in ``names`` order.

        Physical and latent correlations are the same here, since each physical
        value is a monotone function of its own latent and the delta method only
        rescales each entry.
        """
        spread = jnp.sqrt(jnp.clip(jnp.diagonal(self.covariance), jnp.finfo(float).tiny))
        return self.covariance / spread[:, None] / spread[None, :]

    def intervals(self, confidence: float = 0.95) -> dict[str, tuple[float, float]]:
        """A Wald interval per entry, in physical units.

        Built in latent space and mapped through each value's transform, so it is
        asymmetric on a log warp and stays inside a ``Parameter``'s bounds. An
        unconstrained entry's interval reaches its bounds.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"intervals: confidence must be in (0, 1); got {confidence}")
        z = float(stats.norm.ppf(0.5 + confidence / 2))
        spread = jnp.array(list(self.standard_errors(latent=True).values()))
        # Far enough to reach a squash's bound or a log warp's zero, and still finite.
        spread = jnp.minimum(spread, 1e3)
        result = {}
        for i, name in enumerate(self.names):
            ends = [
                float(self._to_physical(self.latent.at[i].add(sign * z * spread[i]))[i])
                for sign in (-1.0, 1.0)
            ]
            result[name] = (min(ends), max(ends))
        return result

    def _table(self, title: str, confidence: float = 0.95) -> list[str]:
        errors = self.standard_errors()
        bounds = self.intervals(confidence)
        width = max(len(name) for name in self.names)
        lines = [
            title,
            f"  {'':{width}}  {'estimate':>11}  {'std error':>11}  "
            f"{int(confidence * 100)}% interval",
        ]
        for i, name in enumerate(self.names):
            low, high = bounds[name]
            lines.append(
                f"  {name:{width}}  {float(self.estimate[i]):11.4g}  {errors[name]:11.4g}  "
                f"[{low:.4g}, {high:.4g}]"
            )
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
    inverse with unconstrained directions left out. ``eigenvalues`` ascend, and the
    columns of ``eigenvectors`` are the matching directions over ``names``.
    """

    information: Array = dataclasses.field(kw_only=True)
    eigenvalues: Array = dataclasses.field(kw_only=True)
    eigenvectors: Array = dataclasses.field(kw_only=True)

    @property
    def condition_number(self) -> float:
        """Largest over smallest eigenvalue; ``inf`` when the smallest is not positive."""
        smallest = float(self.eigenvalues[0])
        return float(self.eigenvalues[-1]) / smallest if smallest > 0.0 else math.inf

    def unidentified(self, threshold: float = _UNIDENTIFIED) -> list[dict[str, float]]:
        """The directions the data does not constrain, as ``{name: loading}`` per direction.

        A direction counts when its eigenvalue is below ``threshold`` times the
        largest. Loadings below 0.1 in size are left out, so each dict names the
        values that trade off along it, such as two rates whose product is all the
        data sees.
        """
        largest = float(self.eigenvalues[-1])
        directions = []
        for k in range(len(self.names)):
            if float(self.eigenvalues[k]) >= threshold * largest:
                break
            vector = np.asarray(self.eigenvectors[:, k])
            directions.append(
                {
                    self.names[i]: float(vector[i])
                    for i in np.argsort(-np.abs(vector))
                    if abs(vector[i]) >= 0.1
                }
            )
        return directions

    def _unbounded(self) -> Array:
        loose = self.eigenvalues < _UNIDENTIFIED * self.eigenvalues[-1]
        return jnp.any(loose[None, :] & (jnp.abs(self.eigenvectors) >= 0.1), axis=1)

    def summary(self, confidence: float = 0.95) -> str:
        """Estimates, standard errors, Wald intervals, correlations and identifiability."""
        lines = self._table("Fisher information", confidence)
        lines.append(f"  Condition number {self.condition_number:.3g}")
        for direction in self.unidentified():
            shown = ", ".join(f"{name} ({value:+.2f})" for name, value in direction.items())
            lines.append(f"  Not identified: {shown}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LaplaceResult(_GaussianSummary):
    """A Gaussian approximation to the posterior, from :meth:`InferenceProblem.laplace`.

    ``latent`` is its mean and ``covariance`` its covariance, over the latent
    position in ``names`` order.
    """

    _to_positions: Callable[[Array], Any] = dataclasses.field(repr=False, kw_only=True)

    def sample(self, key: Array, n: int) -> Any:
        """``n`` draws as positions with a leading ``[n]`` axis.

        ``problem.physical``, ``problem.predict`` and, with an extra chain axis,
        ``problem.to_arviz`` take them as they take a sampler's output.
        """
        check_key(key, "LaplaceResult.sample")
        if n < 1:
            raise ValueError(f"LaplaceResult.sample: n must be positive; got {n}")
        factor = jnp.linalg.cholesky(self.covariance)
        draws = jr.normal(key, (n, len(self.names)), dtype=self.latent.dtype)
        return self._to_positions(self.latent + draws @ factor.T)

    def summary(self, confidence: float = 0.95) -> str:
        """Posterior means, standard deviations, Wald intervals and correlations."""
        return "\n".join(self._table("Laplace approximation", confidence))


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
    values : ndarray
        The profiled value at each point, physical and ascending, the maximum
        likelihood included.
    log_likelihood_ratio : ndarray
        ``2 (max log L - log L)`` at each point, with every other value
        re-optimised.
    others : dict[str, ndarray]
        Every other free entry along the profile, physical. How they move shows
        what compensates for the profiled value.
    converged : ndarray
        Whether BFGS met its tolerances at each point. The interval ignores points
        that did not.
    interval : tuple[float | None, float | None]
        The confidence interval's ends, ``None`` on a side the data leave open.
    """

    name: str
    confidence: float
    threshold: float
    values: np.ndarray
    log_likelihood_ratio: np.ndarray
    others: dict[str, np.ndarray]
    converged: np.ndarray
    interval: tuple[float | None, float | None]

    @property
    def identifiable(self) -> bool:
        """Whether both sides of the interval close."""
        return None not in self.interval

    def summary(self) -> str:
        """One line: the interval, or which side the data leave open."""
        low, high = self.interval
        shown = (
            f"[{'open' if low is None else f'{low:.4g}'}, "
            f"{'open' if high is None else f'{high:.4g}'}]"
        )
        note = "" if self.identifiable else ", not identified on the open side"
        unconverged = int((~self.converged).sum())
        if unconverged:
            note += f", {unconverged} points did not converge"
        return f"{self.name}: {int(self.confidence * 100)}% profile interval {shown}{note}"


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
        eqx.filter_eval_shape(jax.jacfwd(problem._forward_view().log_likelihood), problem.position)
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
    free: Sequence[FreezeSpec] | FreezeSpec | None = None,
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
        positive. Below about 1e-6 the likelihood's variance floor flattens the density,
        so keep the lower bound above that. With ``log_likelihood`` its keys are
        free-form, such as an AR(1) coefficient.
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
    free_mask = _free_mask(predictors, base, free)
    if not any(jtu.tree_leaves(free_mask)):
        raise ValueError("make_problem: free= selects no trainable leaf; nothing to sample")
    tree = {_PREDICTORS: predictors, _NOISE: noise_map}
    mask = {_PREDICTORS: free_mask, _NOISE: trainable_mask(noise_map)}
    if log_prior is flat_prior:
        _check_flat_prior(tree, mask)

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
    eqx.filter_eval_shape(
        problem.log_likelihood if log_prior is None else problem._density, position
    )
    if gradient == "forward":
        eqx.filter_eval_shape(jax.jacfwd(problem._forward_view().log_likelihood), position)
    few = ravel_pytree(position)[0].size <= _FORWARD_MODE_LIMIT
    if gradient == "auto" and few and _supports_forward(problem):
        return dataclasses.replace(problem, gradient="forward")
    return problem


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
            bool(jnp.all(bp.yvar[..., d][bp.mask[..., d]] == 1.0)) for bp in dataset.bucket_payloads
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
