# Inference: Log-Density, Fisher, Laplace and Profiles

## Quick links

- [`make_problem`](#make_problem)
- [`InferenceProblem`](#inferenceproblem)
- [`OptimizeResult`](#optimizeresult)
- [`FisherResult`](#fisherresult)
- [`LaplaceResult`](#laplaceresult)
- [`DerivedQuantity`](#derivedquantity)
- [`ProfileResult`](#profileresult)
- [`flat_prior`](#flat_prior)

---

<a id="make_problem"></a>

### `make_problem()`

<small>`from jaxhybridmodels.inference import make_problem`</small>

```python
make_problem(
    model: 'HybridModel',
    data: 'Data',
    *,
    log_prior: 'Callable[[Any, Mapping[str, Array]], Array] | None',
    noise: "Literal['data'] | Mapping[str, Parameter]",
    estimate: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
    log_likelihood: 'Callable[[Array, BucketPayload, Mapping[str, Array]], Array] | None' = None,
    gradient: "Literal['auto', 'forward', 'reverse']" = 'auto',
) -> InferenceProblem
```

Build the log-density for ``model`` on ``data``.

Solves inside the density run with ``SolverConfig.throw`` off. A ``simulate_fn``
that calls ``diffrax.diffeqsolve`` itself must pass ``throw=solver.throw``, or
a failed solve will end the chain with an exception.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `HybridModel` | Its current values are the starting position, so a fitted model starts a chain near the mode. |
| `data` | `Dataset | Sequence[Experiment]` | The observations. Bucketing does not change the density. |
| `log_prior` | `Callable | None` | ``(predictors, noise) -> scalar``, the log-prior on physical values: read ``predictors["k"].value``, and ``noise["a"]`` for inferred noise. There is no default because no prior is neutral. [`flat_prior`](/api/inference#flat_prior) is available for a bounded box. A free network's weights need a prior written here. ``None`` is for frequentist work: ``optimize``, ``fisher`` and ``profile`` use only the likelihood, and the posterior methods then raise. |
| `noise` | `"data" | Mapping[str, Parameter]` | ``"data"`` trusts the variances in the data, and warns for a channel left at the default of 1.0. A mapping infers a standard deviation per channel, each a ``Parameter`` naming its starting value and bounds, which must keep it positive. With ``log_likelihood`` its keys are free-form, such as an AR(1) coefficient. |
| `estimate` | `str | type | Callable | Sequence | None` | What to infer, named in the syntax ``fit(freeze=...)`` uses, though it selects rather than excludes. ``None`` means every ``Parameter``. A dotted path can name a network part, such as ``"g.inner.mlp.layers.2"``. |
| `log_likelihood` | `Callable | None` | Replaces the Gaussian: ``(pred_obs, bp, noise) -> scalar``, called once per bucket and summed. ``pred_obs`` is ``[N, T, D]``, and ``bp.mask`` says which cells were measured. |
| `gradient` | `{"auto", "forward", "reverse"}` | How the density is differentiated. ``"auto"`` uses forward mode for a few free values when the model allows it, and otherwise reverse mode with the model's own adjoint. Forward mode replaces the adjoint with ``diffrax.ForwardMode``, so a ``simulate_fn`` must use ``solver.diffeqsolve`` or pass ``adjoint=solver.adjoint``. ``problem.gradient`` says which was used. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L2201)</small>

---

<a id="inferenceproblem"></a>

### `InferenceProblem`

<small>`from jaxhybridmodels.inference import InferenceProblem`</small>

```python
InferenceProblem(
    model: 'HybridModel',
    dataset: 'Dataset',
    position: 'Any',
    conditional_on: 'tuple[str, ...]',
    gradient: 'str',
) -> None
```

A model, its data, a prior and a noise model, as a function of a position.

Build one with [`make_problem`](/api/inference#make_problem). Identity-hashed, so bound methods such as
``problem.log_density`` can be passed to ``jax.jit`` and to samplers.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `model` | `HybridModel` | The model as given. Solves in the density run with ``SolverConfig.throw`` off. |
| `dataset` | `Dataset` | The data the likelihood reads. |
| `position` | `PyTree` | The starting position: the free leaves' current values. Latents for a ``Parameter``, weights for a network, and inferred noise under ``"noise"``. After `preconditioned`, a flat array of zeros instead. ``physical``, ``to_model``, ``predict`` and ``to_arviz`` read either form. |
| `conditional_on` | `tuple[str, ...]` | Dotted paths of the trainable leaves held fixed. The posterior is conditional on these values and its width does not account for them. |
| `gradient` | `str` | The mode in use, ``"forward"`` or ``"reverse"``, after ``"auto"`` chose. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L233)</small>

<a id="inferenceproblemfisher"></a>

#### `InferenceProblem.fisher()`

```python
fisher(
    self,
    *,
    curvature: "Literal['gauss_newton', 'exact']" = 'gauss_newton',
) -> FisherResult
```

The Fisher information at the current position, with what it says about identifiability.

Call it at the maximum likelihood, from `optimize`; it warns when the
position is far from one. ``"gauss_newton"`` is ``J^T J`` of the whitened
residuals, plus the noise terms when noise is inferred: the expected
information of the Gaussian likelihood. ``"exact"`` is the negative
Hessian of the log-likelihood, the observed information, which also works
with a custom ``log_likelihood``. ``laplace`` takes the same choice.

The matrices are in latent coordinates, where the problem lives. Standard
errors and intervals are reported in physical units by default.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L629)</small>

<a id="inferenceproblemlaplace"></a>

#### `InferenceProblem.laplace()`

```python
laplace(
    self,
    *,
    curvature: "Literal['exact', 'gauss_newton']" = 'exact',
) -> LaplaceResult
```

A Gaussian approximation to the posterior over latents, at the current position.

Call it at the posterior mode, from ``optimize(target="posterior")``; it warns
when the position is far from one. ``"exact"`` uses the Hessian of the log
density. ``"gauss_newton"`` replaces the likelihood's part with ``J^T J``,
which is cheaper for many values and always positive semi-definite, and keeps
the exact Hessian of the prior and change-of-variables terms, which need no
solve.

``sample`` draws positions that ``physical``, ``predict`` and ``to_arviz``
take as they take a sampler's output.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L668)</small>

<a id="inferenceproblemlog_density"></a>

#### `InferenceProblem.log_density()`

```python
log_density(self, position: 'Any') -> 'Array'
```

Unnormalised log-posterior at ``position``, for a sampler to target.

Log-likelihood, plus the prior on physical values, plus the
change-of-variables term for free ``Parameter`` latents. It is a density
over the latents, so it is ``-inf`` with a zero gradient wherever a solve
fails, and any NaN term also becomes ``-inf``.

Take its gradient with ``jax.grad``, as samplers do. Its custom gradient
rule rules out ``jax.jvp`` and ``jax.jacfwd`` on this method, so use them on
``log_likelihood`` or ``residuals``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L390)</small>

<a id="inferenceproblemlog_likelihood"></a>

#### `InferenceProblem.log_likelihood()`

```python
log_likelihood(self, position: 'Any') -> 'Array'
```

The log-likelihood of the data, summed over every experiment.

``-inf`` when any solve fails or any prediction is non-finite, including
at a row no channel measured, since a failed solve leaves such rows
non-finite too.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L329)</small>

<a id="inferenceproblemoptimize"></a>

#### `InferenceProblem.optimize()`

```python
optimize(
    self,
    *,
    target: "Literal['likelihood', 'posterior']" = 'likelihood',
    rtol: 'float' = 1e-08,
    atol: 'float' = 1e-08,
    max_steps: 'int' = 1000,
) -> OptimizeResult
```

Find the maximum of the log-likelihood, or of the log-posterior, with BFGS.

Fisher information and profiles describe the likelihood at its maximum, and a
Laplace approximation the posterior at its mode. A training fit is close but
not there: the optimiser stops after a finite budget, and a loss other than
``"mle"`` has a different minimum when the variances differ. The search runs in the
latent coordinates, so a ``Parameter`` stays inside its bounds; the posterior
mode is therefore the mode of the density over latents.

Uses ``optimistix.BFGS``. ``rtol`` and ``atol`` are its tolerances on the
latent values and on the objective. The result's ``problem`` sits at the
optimum and ``converged`` says whether BFGS met its tolerances.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L579)</small>

<a id="inferenceproblemphysical"></a>

#### `InferenceProblem.physical()`

```python
physical(self, position: 'Any') -> 'dict[str, Array]'
```

The free values in physical units, keyed by dotted path.

A ``Parameter`` appears under its path with its ``.value``, a free network
leaf under its path as it is, and inferred noise as ``"noise.<name>"``.
``position`` may be a sampler's output: leading axes such as
``[chain, draw]`` stay on every value.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L936)</small>

<a id="inferenceproblempreconditioned"></a>

#### `InferenceProblem.preconditioned()`

```python
preconditioned(
    self,
    *,
    precision: 'Any' = None,
    ridge: 'float' = 1.0,
) -> InferenceProblem
```

This problem, sampling a whitened position so the target is close to a unit normal.

``position = fit + L z``, where ``L L^T`` is the inverse of the curvature
at the current position. It is a constant linear map, so the posterior over
the physical values is unchanged and only the sampler's geometry improves.
Start from a fitted model: far from a mode the curvature says little, and
sampling is then no better but still correct. A fit does not set inferred noise,
and the other values' curvature scales with ``1 / sigma^2``, so a good starting
guess for the noise helps too.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `precision` | `Array | None` | The ``[P, P]`` curvature to use, for a custom likelihood or a better estimate. ``None`` computes the Gauss-Newton ``J^T J + ridge I`` from the whitened residuals and any inferred noise, which needs the Gaussian likelihood. |
| `ridge` | `float` | Added to the diagonal of the Gauss-Newton curvature in latent units, standing in for the prior so an unidentified direction keeps a finite scale. Ignored with ``precision``. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L408)</small>

<a id="inferenceproblempredict"></a>

#### `InferenceProblem.predict()`

```python
predict(
    self,
    positions: 'Any',
    data: 'Data | None' = None,
    *,
    key: 'Array',
    ts: 'ArrayLike | None' = None,
    n_points: 'int | None' = None,
    n_draws: 'int' = 200,
    quantiles: 'Sequence[float]' = (0.025, 0.975),
    observation_noise: 'bool' = False,
) -> PredictionResult
```

Predict with draws from a chain and report their spread.

``positions`` is the sampler's output: the position pytree with any
leading sample axes, such as ``[draw, ...]`` or ``[chain, draw, ...]``.
``n_draws`` of them are chosen at random, without replacement. ``data``
defaults to the problem's own. ``ts`` and ``n_points`` are as for
[`HybridModel.predict`](/api/model#hybridmodelpredict).

``outputs`` is the mean over draws. ``std`` and ``quantiles``, by default a
central 95% band, describe the model's uncertainty: a credible band for the
noiseless trajectory. With ``observation_noise``, they include measurement
noise instead, an interval for a new measurement. That needs a standard
deviation inferred for each channel, since the variances in data exist only
at measured cells.

A draw whose solve fails on this grid is dropped with a warning, so one bad
draw on new data or a longer grid does not cost the others.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L959)</small>

<a id="inferenceproblemprofile"></a>

#### `InferenceProblem.profile()`

```python
profile(
    self,
    name: 'str',
    *,
    confidence: 'float' = 0.95,
    max_points: 'int' = 40,
    max_steps: 'int' = 200,
    rtol: 'float' = 1e-06,
    atol: 'float' = 1e-06,
) -> ProfileResult
```

The profile likelihood of one value, and the confidence interval it gives.

Steps away from the current position, which should be the maximum
likelihood from `optimize`, in each direction. At each point the
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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L720)</small>

<a id="inferenceproblemprofiles"></a>

#### `InferenceProblem.profiles()`

```python
profiles(
    self,
    names: 'Sequence[str] | None' = None,
    *,
    confidence: 'float' = 0.95,
    max_points: 'int' = 40,
    max_steps: 'int' = 200,
    rtol: 'float' = 1e-06,
    atol: 'float' = 1e-06,
) -> dict[str, ProfileResult]
```

`profile` for each name, by default every free ``Parameter`` entry.

The setup, the Fisher step sizes and the compiled objective are shared, so
each extra name costs only its own points. One that fails, for instance
because the model stops solving far from the optimum, is skipped with a
warning and the others still run.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L760)</small>

<a id="inferenceproblemresiduals"></a>

#### `InferenceProblem.residuals()`

```python
residuals(self, position: 'Any') -> 'Array'
```

Whitened residuals ``(prediction - observed) / sigma``, flattened over buckets.

Length is every ``[N, T, D]`` cell, with zero at unmeasured cells, so the
shape does not depend on the data's mask. ``jax.jacfwd`` applies when the
model's adjoint supports forward mode.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L297)</small>

<a id="inferenceproblemto_arviz"></a>

#### `InferenceProblem.to_arviz()`

```python
to_arviz(
    self,
    positions: 'Any',
    *,
    sample_stats: 'Mapping[str, Any] | None' = None,
) -> Any
```

An ArviZ ``DataTree`` of the physical values, for R-hat, ESS and plots.

``positions`` must have leading ``[chain, draw]`` axes; add one with
``jax.tree.map(lambda x: x[None], positions)`` for a single chain.
Variables are named by dotted path. ``sample_stats`` holds per-draw
sampler output with the same axes, such as ``{"diverging": ...}``.
``conditional_on`` is recorded in the attributes. Needs
``pip install jax-hybridmodels[bayes]``, and ArviZ needs Python 3.12.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1063)</small>

<a id="inferenceproblemto_model"></a>

#### `InferenceProblem.to_model()`

```python
to_model(self, position: 'Any') -> 'HybridModel'
```

The given model with this position's values, for ``predict`` and ``save``.

It keeps the model's own solver, so its solves raise on failure as usual.
Inferred noise is not part of a model.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L928)</small>

---

<a id="optimizeresult"></a>

### `OptimizeResult`

<small>`from jaxhybridmodels.inference import OptimizeResult`</small>

```python
OptimizeResult(
    problem: 'InferenceProblem',
    target: 'str',
    value: 'float',
    converged: 'bool',
    steps: 'int',
    gradient: 'dict[str, float]',
    gradient_norm: 'float',
) -> None
```

What [`InferenceProblem.optimize`](/api/inference#inferenceproblemoptimize) found.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `problem` | `InferenceProblem` | The problem positioned at the optimum, for ``fisher``, ``laplace`` or ``profile``. ``problem.to_model(problem.position)`` is the fitted model. |
| `target` | `str` | ``"likelihood"`` or ``"posterior"``. |
| `value` | `float` | The log-likelihood, or the log-posterior density over latents, at the optimum. |
| `converged` | `bool` | Whether BFGS met its tolerances within ``max_steps``. |
| `steps` | `int` | BFGS steps taken. |
| `gradient` | `dict[str, float]` | The gradient of the log-likelihood or log-posterior at the optimum, per entry as ``physical`` names it, taken in the latent coordinates the search runs in. Near zero at a maximum. An entry that stays large points at the value BFGS could not settle. A value against its bound shows a gradient near zero too, since the squash is flat there; ``fisher`` and the profile flag those. |
| `gradient_norm` | `float` | The largest entry of ``gradient`` in size, a one-number convergence check. How small is small depends on the curvature: ``fisher`` and ``laplace`` apply the scale-free test, and warn when a Newton step from here would still raise the log density by more than 0.1. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1373)</small>

---

<a id="fisherresult"></a>

### `FisherResult`

<small>`from jaxhybridmodels.inference import FisherResult`</small>

```python
FisherResult(
    names: 'tuple[str, ...]',
    estimate: 'Array',
    latent: 'Array',
    covariance: 'Array',
    slopes: 'Array',
    *,
    at_bound: 'np.ndarray',
    scaled: 'np.ndarray',
    limits: 'tuple[np.ndarray, np.ndarray]',
    information: 'Array',
    eigenvalues: 'Array',
    eigenvectors: 'Array',
) -> None
```

The Fisher information at a position, from [`InferenceProblem.fisher`](/api/inference#inferenceproblemfisher).

``information`` is the latent ``[P, P]`` information, and ``covariance`` its
inverse with unconstrained directions left out. ``eigenvalues`` and
``eigenvectors`` belong to the information scaled to a unit diagonal, its
correlation form, so they do not change when a bound or a warp rescales a
latent. ``eigenvalues`` ascend, and the columns of ``eigenvectors`` are the
matching directions over ``names``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1650)</small>

<a id="fisherresultcondition_number"></a>

#### `FisherResult.condition_number`

Largest over smallest eigenvalue; ``inf`` when the smallest is not positive.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1666)</small>

<a id="fisherresultcorrelation"></a>

#### `FisherResult.correlation()`

```python
correlation(self) -> 'Array'
```

The ``[P, P]`` correlation matrix in ``names`` order.

Physical and latent correlations are the same here, since each physical
value is a monotone function of its own latent and the delta method only
rescales each entry. An entry with unbounded spread gets ``nan``: the
pseudo-inverse describes only the directions the data constrain, so its
correlation there could even have the wrong sign.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1523)</small>

<a id="fisherresultderived"></a>

#### `FisherResult.derived()`

```python
derived(self, fn: 'Callable[[dict[str, Array]], Any]') -> 'DerivedQuantity'
```

The estimate and standard error of a function of the values, by the delta method.

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1566)</small>

<a id="fisherresultintervals"></a>

#### `FisherResult.intervals()`

```python
intervals(self, *, confidence: 'float' = 0.95) -> 'dict[str, tuple[float, float]]'
```

A Wald interval per entry, in physical units.

Built in latent space and mapped through each value's transform, so it is
asymmetric on a log warp and stays inside a ``Parameter``'s bounds. An
unconstrained entry's interval reaches its bounds, or infinity without one.
An entry against its bound gets ``(nan, nan)``, as in ``standard_errors``;
a profile tells whether the data bound it away from there.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1537)</small>

<a id="fisherresultphysical_covariance"></a>

#### `FisherResult.physical_covariance()`

```python
physical_covariance(self) -> 'Array'
```

The ``[P, P]`` covariance in physical units, by the delta method, in ``names`` order.

``S C S`` with ``S`` the diagonal of ``d value / d latent``. It describes the
physical values only as far as their transforms are close to linear over
the latent spread; ``intervals`` does not rely on that.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1514)</small>

<a id="fisherresultsample"></a>

#### `FisherResult.sample()`

```python
sample(self, *, key: 'Array', n_draws: 'int') -> 'Any'
```

``n_draws`` draws from this Gaussian, as positions with a leading axis of that size.

``problem.physical``, ``problem.predict`` and, with an extra chain axis,
``problem.to_arviz`` take them as they take a sampler's output. From a
[`LaplaceResult`](/api/inference#laplaceresult) they approximate the posterior. From a
[`FisherResult`](/api/inference#fisherresult) they follow the estimate's asymptotic sampling
distribution, which carries its uncertainty into any function of it, such as
a prediction at new conditions.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1436)</small>

<a id="fisherresultstandard_errors"></a>

#### `FisherResult.standard_errors()`

```python
standard_errors(self, *, latent: 'bool' = False) -> 'dict[str, float]'
```

The standard error of each entry: physical by the delta method, or latent.

An entry that loads on a direction the data does not constrain has an
infinite standard error. An entry against its bound has ``nan`` as its
physical one: the curvature there describes the squash's flat tail, not
how far the data let the value move, so no number would be honest.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1500)</small>

<a id="fisherresultsummary"></a>

#### `FisherResult.summary()`

```python
summary(self, *, confidence: 'float' = 0.95) -> 'str'
```

Estimates, standard errors, Wald intervals, correlations and identifiability.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1732)</small>

<a id="fisherresultunidentified"></a>

#### `FisherResult.unidentified()`

```python
unidentified(self, *, threshold: 'float' = 1e-08) -> 'list[dict[str, float]]'
```

The directions the data does not constrain, as ``{name: loading}`` per direction.

A direction counts when its eigenvalue, in the correlation form of the
information, is below ``threshold`` times the largest. Loadings below 0.1
in size are left out, so each dict names the values that trade off along
it, such as two rates whose product is all the data sees. A value the data
barely see on its own is listed alone as ``{name: 1.0}`` when its latent
has a fixed scale (a box, or a log-warped one-sided bound) and its latent
standard error passes 1e3. An unbounded value has no such scale, so only
its collinearity is tested.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1672)</small>

---

<a id="laplaceresult"></a>

### `LaplaceResult`

<small>`from jaxhybridmodels.inference import LaplaceResult`</small>

```python
LaplaceResult(
    names: 'tuple[str, ...]',
    estimate: 'Array',
    latent: 'Array',
    covariance: 'Array',
    slopes: 'Array',
    *,
    at_bound: 'np.ndarray',
    scaled: 'np.ndarray',
    limits: 'tuple[np.ndarray, np.ndarray]',
) -> None
```

A Gaussian approximation to the posterior, from [`InferenceProblem.laplace`](/api/inference#inferenceproblemlaplace).

``latent`` is its mean and ``covariance`` its covariance, over the latent
position in ``names`` order.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1740)</small>

<a id="laplaceresultcorrelation"></a>

#### `LaplaceResult.correlation()`

```python
correlation(self) -> 'Array'
```

The ``[P, P]`` correlation matrix in ``names`` order.

Physical and latent correlations are the same here, since each physical
value is a monotone function of its own latent and the delta method only
rescales each entry. An entry with unbounded spread gets ``nan``: the
pseudo-inverse describes only the directions the data constrain, so its
correlation there could even have the wrong sign.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1523)</small>

<a id="laplaceresultderived"></a>

#### `LaplaceResult.derived()`

```python
derived(self, fn: 'Callable[[dict[str, Array]], Any]') -> 'DerivedQuantity'
```

The estimate and standard error of a function of the values, by the delta method.

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1566)</small>

<a id="laplaceresultintervals"></a>

#### `LaplaceResult.intervals()`

```python
intervals(self, *, confidence: 'float' = 0.95) -> 'dict[str, tuple[float, float]]'
```

A Wald interval per entry, in physical units.

Built in latent space and mapped through each value's transform, so it is
asymmetric on a log warp and stays inside a ``Parameter``'s bounds. An
unconstrained entry's interval reaches its bounds, or infinity without one.
An entry against its bound gets ``(nan, nan)``, as in ``standard_errors``;
a profile tells whether the data bound it away from there.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1537)</small>

<a id="laplaceresultphysical_covariance"></a>

#### `LaplaceResult.physical_covariance()`

```python
physical_covariance(self) -> 'Array'
```

The ``[P, P]`` covariance in physical units, by the delta method, in ``names`` order.

``S C S`` with ``S`` the diagonal of ``d value / d latent``. It describes the
physical values only as far as their transforms are close to linear over
the latent spread; ``intervals`` does not rely on that.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1514)</small>

<a id="laplaceresultsample"></a>

#### `LaplaceResult.sample()`

```python
sample(self, *, key: 'Array', n_draws: 'int') -> 'Any'
```

``n_draws`` draws from this Gaussian, as positions with a leading axis of that size.

``problem.physical``, ``problem.predict`` and, with an extra chain axis,
``problem.to_arviz`` take them as they take a sampler's output. From a
[`LaplaceResult`](/api/inference#laplaceresult) they approximate the posterior. From a
[`FisherResult`](/api/inference#fisherresult) they follow the estimate's asymptotic sampling
distribution, which carries its uncertainty into any function of it, such as
a prediction at new conditions.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1436)</small>

<a id="laplaceresultstandard_errors"></a>

#### `LaplaceResult.standard_errors()`

```python
standard_errors(self, *, latent: 'bool' = False) -> 'dict[str, float]'
```

The standard error of each entry: physical by the delta method, or latent.

An entry that loads on a direction the data does not constrain has an
infinite standard error. An entry against its bound has ``nan`` as its
physical one: the curvature there describes the squash's flat tail, not
how far the data let the value move, so no number would be honest.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1500)</small>

<a id="laplaceresultsummary"></a>

#### `LaplaceResult.summary()`

```python
summary(self, *, confidence: 'float' = 0.95) -> 'str'
```

Posterior means, standard deviations, Wald intervals and correlations.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L1748)</small>

---

<a id="derivedquantity"></a>

### `DerivedQuantity`

<small>`from jaxhybridmodels.inference import DerivedQuantity`</small>

```python
DerivedQuantity(estimate: 'np.ndarray', standard_error: 'np.ndarray') -> None
```

A function of the values with its delta-method standard error, from ``derived``.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `estimate` | `ndarray` | The function at the estimate, in the shape ``fn`` returns. |
| `standard_error` | `ndarray` | Its standard error, same shape. ``inf`` where it depends on a value the data do not constrain, ``nan`` where it depends on one against its bound. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L2104)</small>

<a id="derivedquantityinterval"></a>

#### `DerivedQuantity.interval()`

```python
interval(
    self,
    *,
    confidence: 'float' = 0.95,
    side: "Literal['both', 'lower', 'upper']" = 'both',
) -> tuple[np.ndarray, np.ndarray]
```

A Wald interval, ``estimate`` plus or minus ``z`` standard errors.

``side="lower"`` gives a one-sided bound, ``(estimate - z se, inf)`` with
``z`` the ``confidence`` quantile, as a shelf-life claim needs;
``"upper"`` the mirror image.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L2120)</small>

---

<a id="profileresult"></a>

### `ProfileResult`

<small>`from jaxhybridmodels.inference import ProfileResult`</small>

```python
ProfileResult(
    name: 'str',
    confidence: 'float',
    threshold: 'float',
    grid: 'np.ndarray',
    log_likelihood_ratio: 'np.ndarray',
    others: 'dict[str, np.ndarray]',
    converged: 'np.ndarray',
    bounds: 'tuple[float | None, float | None]',
    stop_reasons: 'tuple[str, str]',
    at_bound: 'dict[str, str]' = <factory>,
) -> None
```

The profile likelihood of one value, from [`InferenceProblem.profile`](/api/inference#inferenceproblemprofile).

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | The profiled entry, as ``physical`` names it. |
| `confidence` | `float` | The level the interval is for. |
| `threshold` | `float` | The chi-square(1) quantile for ``confidence``: the interval is where ``log_likelihood_ratio`` stays below it. |
| `grid` | `ndarray` | The profiled value at each point, physical and ascending, the maximum likelihood included. |
| `log_likelihood_ratio` | `ndarray` | ``2 (max log L - log L)`` at each point, with every other value re-optimised. |
| `others` | `dict[str, ndarray]` | Every other free entry along the profile, physical. How they move shows what compensates for the profiled value. |
| `converged` | `ndarray` | Whether BFGS met its tolerances at each point, after one restart. The interval ignores points that did not. |
| `bounds` | `tuple[float | None, float | None]` | The confidence interval's ends, ``None`` on a side that did not close. |
| `stop_reasons` | `tuple[str, str]` | Why each side, below and above, stopped. ``"threshold"`` closed it. The others leave it open: ``"bound"`` (the value reached its bound, so the data do not bound it), ``"max_points"`` (still inside; raise ``max_points``), ``"solve_failed"`` (the model stopped solving) and ``"not_converged"`` (BFGS did not converge where the ratio crossed). |
| `at_bound` | `dict[str, str]` | Other values that sat against a bound, ``"lower"`` or ``"upper"``, at some converged point. Re-optimising could not move them further, so the profile may drop faster than it would with that bound widened, and the interval holds for the bounds as set. Empty when nothing was pinned. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L2028)</small>

<a id="profileresultidentifiable"></a>

#### `ProfileResult.identifiable`

Whether both sides closed at the threshold.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L2079)</small>

<a id="profileresultsummary"></a>

#### `ProfileResult.summary()`

```python
summary(self) -> 'str'
```

One line: the interval, and why a side that did not close stopped.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L2084)</small>

---

<a id="flat_prior"></a>

### `flat_prior()`

<small>`from jaxhybridmodels.inference import flat_prior`</small>

```python
flat_prior(predictors: 'Any', noise: 'Mapping[str, Array]') -> 'Array'
```

A prior flat in physical units, for free ``Parameter``s with finite bounds.

Proper only on a bounded box, so [`make_problem`](/api/inference#make_problem) refuses it when a free
value has an open bound or is a network weight. A flat prior is not neutral
under a warp: on a log-warped parameter it still favours the upper decades.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L129)</small>
