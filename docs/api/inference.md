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
    log_prior: 'Callable[[Any, Mapping[str, Array]], Array]',
    noise: "Literal['data'] | Mapping[str, Parameter]",
    free: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
    log_likelihood: 'Callable[[Array, BucketPayload, Mapping[str, Array]], Array] | None' = None,
) -> InferenceProblem
```

Build the log-density for ``model`` on ``data``.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `HybridModel` | Its current values are the starting position, so a fitted model starts a chain near the mode. |
| `data` | `Dataset | Sequence[Experiment]` | The observations. Bucketing does not change the density. |
| `log_prior` | `Callable` | ``(predictors, noise) -> scalar``, the log-prior on physical values: read ``predictors["k"].value``, and ``noise["a"]`` for inferred noise. There is no default because no prior is neutral. [`flat_prior`](/api/inference#flat_prior) is available for a bounded box. A free network's weights need a prior written here. |
| `noise` | `"data" | Mapping[str, Parameter]` | ``"data"`` trusts the variances in the data, and warns for a channel left at the default of 1.0. A mapping infers a standard deviation per channel, each a ``Parameter`` naming its starting value and bounds. With ``log_likelihood`` its keys are free-form, such as an AR(1) coefficient. |
| `free` | `str | type | Callable | Sequence | None` | What to sample, in the syntax of ``fit(freeze=...)``. ``None`` means every ``Parameter``. A dotted path can name a network part, such as ``"g.inner.mlp.layers.2"``. |
| `log_likelihood` | `Callable | None` | Replaces the Gaussian: ``(pred_obs, bp, noise) -> scalar``, called once per bucket and summed. ``pred_obs`` is ``[N, T, D]``, and ``bp.mask`` says which cells were measured. |
| `Solves inside the density run with ``SolverConfig.throw`` off. A ``simulate_fn``` |  |  |
| `that calls ``diffrax.diffeqsolve`` itself must pass ``throw=solver.throw``, or` |  |  |
| `a failed solve will end the chain with an exception.` |  |  |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L275)</small>

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
    _log_prior: 'Callable[..., Array]',
    _log_likelihood: 'Callable[..., Array] | None',
    _static: 'Any',
    _mask: 'Any',
    _sampling_model: 'HybridModel',
) -> None
```

A model, its data, a prior and a noise model, as a function of a position.

Build one with [`make_problem`](/api/inference#make_problem). Identity-hashed, so bound methods such as
``problem.log_density`` can be passed to ``jax.jit`` and to samplers.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `model` | `HybridModel` | The model as given. Solves here run through a copy with ``SolverConfig.throw`` off. |
| `dataset` | `Dataset` | The data the likelihood reads. |
| `position` | `PyTree` | The starting position: the free leaves' current values. Latents for a ``Parameter``, weights for a network, and inferred noise under ``"noise"``. |
| `conditional_on` | `tuple[str, ...]` | Dotted paths of the trainable leaves held fixed. The posterior is conditional on these values and its width does not account for them. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L123)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L234)</small>

<a id="inferenceproblemlog_likelihood"></a>

#### `InferenceProblem.log_likelihood()`

```python
log_likelihood(self, position: 'Any') -> 'Array'
```

The log-likelihood of the data, summed over every experiment.

``-inf`` when any solve fails or any prediction is non-finite, including
at a row no channel measured, since a failed solve leaves such rows
non-finite too.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L200)</small>

<a id="inferenceproblemphysical"></a>

#### `InferenceProblem.physical()`

```python
physical(self, position: 'Any') -> 'dict[str, Array]'
```

The free values in physical units, keyed by dotted path.

A ``Parameter`` appears under its path with its ``.value``, a free network
leaf under its path as it is, and inferred noise as ``"noise.<name>"``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L256)</small>

<a id="inferenceproblemresiduals"></a>

#### `InferenceProblem.residuals()`

```python
residuals(self, position: 'Any') -> 'Array'
```

Whitened residuals ``(prediction - observed) / sigma``, flattened over buckets.

Length is every ``[N, T, D]`` cell, with zero at unmeasured cells, so the
shape does not depend on the data's mask and ``jax.jacfwd`` applies.
``log_likelihood`` is built on the same quantity for a Gaussian.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L182)</small>

<a id="inferenceproblemto_model"></a>

#### `InferenceProblem.to_model()`

```python
to_model(self, position: 'Any') -> 'HybridModel'
```

The given model with this position's values, for ``predict`` and ``save``.

It keeps the model's own solver, so its solves raise on failure as usual.
Inferred noise is not part of a model.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L248)</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/inference.py#L65)</small>
