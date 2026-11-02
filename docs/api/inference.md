# Inference: A Log-Density for Samplers

## Quick links

- [`make_problem`](#make_problem)
- [`InferenceProblem`](#inferenceproblem)
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
    log_prior: 'Callable[[Any, Mapping[str, Array]], Array]',
    noise: "Literal['data'] | Mapping[str, Parameter]",
    free: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
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
| `log_prior` | `Callable` | ``(predictors, noise) -> scalar``, the log-prior on physical values: read ``predictors["k"].value``, and ``noise["a"]`` for inferred noise. There is no default because no prior is neutral. [`flat_prior`](/api/inference#flat_prior) is available for a bounded box. A free network's weights need a prior written here. |
| `noise` | `"data" | Mapping[str, Parameter]` | ``"data"`` trusts the variances in the data, and warns for a channel left at the default of 1.0. A mapping infers a standard deviation per channel, each a ``Parameter`` naming its starting value and bounds, which must keep it positive. Below about 1e-6 the likelihood's variance floor flattens the density, so keep the lower bound above that. With ``log_likelihood`` its keys are free-form, such as an AR(1) coefficient. |
| `free` | `str | type | Callable | Sequence | None` | What to sample, in the syntax of ``fit(freeze=...)``. ``None`` means every ``Parameter``. A dotted path can name a network part, such as ``"g.inner.mlp.layers.2"``. |
| `log_likelihood` | `Callable | None` | Replaces the Gaussian: ``(pred_obs, bp, noise) -> scalar``, called once per bucket and summed. ``pred_obs`` is ``[N, T, D]``, and ``bp.mask`` says which cells were measured. |
| `gradient` | `{"auto", "forward", "reverse"}` | How the density is differentiated. ``"auto"`` uses forward mode for a few free values when the model allows it, and otherwise reverse mode with the model's own adjoint. Forward mode replaces the adjoint with ``diffrax.ForwardMode``, so a ``simulate_fn`` must use ``solver.diffeqsolve`` or pass ``adjoint=solver.adjoint``. ``problem.gradient`` says which was used. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L589)</small>

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
    _log_prior: 'Callable[..., Array]',
    _log_likelihood: 'Callable[..., Array] | None',
    _static: 'Any',
    _mask: 'Any',
    _solver: 'SolverConfig',
    _whitening: 'tuple[Array, Array, Callable[[Array], Any]] | None' = None,
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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L166)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L308)</small>

<a id="inferenceproblemlog_likelihood"></a>

#### `InferenceProblem.log_likelihood()`

```python
log_likelihood(self, position: 'Any') -> 'Array'
```

The log-likelihood of the data, summed over every experiment.

``-inf`` when any solve fails or any prediction is non-finite, including
at a row no channel measured, since a failed solve leaves such rows
non-finite too.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L251)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L392)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L325)</small>

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
    quantiles: 'Sequence[float]' = (0.05, 0.95),
    observation_noise: 'bool' = False,
) -> PredictionResult
```

Predict with draws from a chain and report their spread.

``positions`` is the sampler's output: the position pytree with any
leading sample axes, such as ``[draw, ...]`` or ``[chain, draw, ...]``.
``n_draws`` of them are chosen at random, without replacement. ``data``
defaults to the problem's own. ``ts`` and ``n_points`` are as for
[`HybridModel.predict`](/api/model#hybridmodelpredict).

``outputs`` is the mean over draws. ``std`` and ``quantiles`` describe the
model's uncertainty, a credible band for the noiseless trajectory. With
``observation_noise``, they include measurement noise instead, an interval
for a new measurement. That needs a standard deviation inferred for each
channel, since the variances in data exist only at measured cells.

A draw whose solve fails on this grid is dropped with a warning, so one bad
draw on new data or a longer grid does not cost the others.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L415)</small>

<a id="inferenceproblemresiduals"></a>

#### `InferenceProblem.residuals()`

```python
residuals(self, position: 'Any') -> 'Array'
```

Whitened residuals ``(prediction - observed) / sigma``, flattened over buckets.

Length is every ``[N, T, D]`` cell, with zero at unmeasured cells, so the
shape does not depend on the data's mask. ``jax.jacfwd`` applies when the
model's adjoint supports forward mode.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L230)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L509)</small>

<a id="inferenceproblemto_model"></a>

#### `InferenceProblem.to_model()`

```python
to_model(self, position: 'Any') -> 'HybridModel'
```

The given model with this position's values, for ``predict`` and ``save``.

It keeps the model's own solver, so its solves raise on failure as usual.
Inferred noise is not part of a model.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L384)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L88)</small>
