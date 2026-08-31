# Prediction: Forward Simulation

Forward-simulate trained predictors against a dataset. [`predict_bucket`](#predict_bucket) is the single-bucket primitive (JIT-compiled, vmapped over the bucket's `N` axis); [`predict_dataset`](#predict_dataset) walks every bucket and returns one `[N, T, D]` array per bucket.

## Quick links

- [`predict_bucket`](#predict_bucket)
- [`predict_dataset`](#predict_dataset)
- [`predict_dense`](#predict_dense)
- [`ensemble_predictions`](#ensemble_predictions)
- [`evaluate_predictor`](#evaluate_predictor)

---

<a id="predict_bucket"></a>

### `predict_bucket()`

<small>`from hybridmodels.prediction import predict_bucket` &nbsp;·&nbsp; also re-exported as `hybridmodels.predict_bucket`</small>

```python
predict_bucket(
    predictors: 'Any',
    bp: 'BucketPayload',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
) -> Float[Array, 'N T D']
```

Vmap ``simulate_fn`` over the bucket's ``N`` axis and project to observed channels.

``predict_bucket_obs`` runs ``simulate_fn`` for one experiment, giving a
state trajectory ``[T, S]``, and maps it to observed channels ``[T, D]``.
``jax.vmap`` lifts that over ``(ts, covariates, y0)`` along ``N``.
``predictors`` and ``solver`` are closed over with no vmap axis, being
the same for every experiment in the bucket.

One compiled kernel per bucket shape. Python dispatch over buckets lives
in ``predict_dataset``, never inside the compiled region.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` | `PyTree[eqx.Module]` | The trainable part of the model, typically a tuple of ``BoundedPredictor`` leaves. Forwarded to ``simulate_fn`` unchanged. |
| `bp` | `BucketPayload` | One bucket. Its ``ts``, ``covariates``, and ``y0`` are vmapped along ``N``. |
| `simulate_fn` |  | User-supplied integrator with signature ``(predictors, ts, covariates, y0, solver) -> [T, S]``. |
| `state_to_output` |  | Pure ``[T, S] -> [T, D]`` map from full state to observed channels. A property of the model, passed explicitly. |
| `solver` |  | ``SolverConfig``. All its fields are static, so it enters the compiled kernel as configuration rather than as data. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Float[Array, "N T D"]` |  | Predicted output channels for every experiment in the bucket. |

---

<a id="predict_dataset"></a>

### `predict_dataset()`

<small>`from hybridmodels.prediction import predict_dataset` &nbsp;·&nbsp; also re-exported as `hybridmodels.predict_dataset`</small>

```python
predict_dataset(
    predictors: 'Any',
    dataset: 'Dataset',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
) -> tuple[Float[Array, 'N T D'], ...]
```

Run ``predict_bucket`` over every bucket in ``dataset`` and return the stack tuple.

Each bucket shape compiles ``predict_bucket`` exactly once. The result
is a tuple rather than one array, because buckets differ precisely in
``T`` and cannot be stacked.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `state_to_output` |  | Pure mapping ``[T, S] -> [T, D]`` from full simulator state to the observed channels. A property of the model, passed here rather than stored on the ``Dataset``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Float[Array, "N T D"], ...]` |  | One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/prediction.py#L88)</small>

---

<a id="predict_dense"></a>

### `predict_dense()`

<small>`from hybridmodels.prediction import predict_dense` &nbsp;·&nbsp; also re-exported as `hybridmodels.predict_dense`</small>

```python
predict_dense(
    predictors: 'Any',
    dataset: 'Dataset',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    ts_grid: 'Array | None' = None,
    n_points: 'int' = 100,
) -> tuple[Float[Array, 'N T_d D'], ...]
```

Evaluate a trained model on a dense time grid, one array per bucket.

``predict_dataset`` returns predictions only at the *measured*
timestamps. For smooth trajectory plots or dense evaluation you usually
want more points than that. This builds a fine grid per experiment and
reuses the same compiled forward pass, so no new kernel or dependency
is needed (the diffraxtra ``VectorizedDenseInterpolation`` equivalent,
folded in ~20 lines).

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `ts_grid` |  | Optional shared grid ``[T_d]`` to evaluate every experiment on. If ``None``, each experiment gets its own ``linspace`` from its first to its last measured time with ``n_points`` points. |
| `n_points` |  | Points per experiment when ``ts_grid`` is ``None``. Ignored otherwise. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Float[Array, "N T_d D"], ...]` |  | One ``[N, T_d, D]`` array per bucket, in bucket-payload order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/prediction.py#L138)</small>

---

<a id="ensemble_predictions"></a>

### `ensemble_predictions()`

<small>`from hybridmodels.prediction import ensemble_predictions` &nbsp;·&nbsp; also re-exported as `hybridmodels.ensemble_predictions`</small>

```python
ensemble_predictions(
    members: 'Sequence[Any]',
    dataset: 'Dataset',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
) -> tuple[Float[Array, 'N T D'], ...]
```

Average per-bucket predictions across an ensemble of models.

``members`` is a sequence of predictor pytrees — each the ``predictors``
argument you would pass to :func:`predict_dataset` alone. Each member is
run forward and the per-bucket predictions are averaged, so the result
is the same shape as a single ``predict_dataset`` return.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `members` |  | Non-empty sequence of predictor pytrees. Every member must be compatible with the same ``simulate_fn``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Float[Array, "N T D"], ...]` |  | The member-mean prediction per bucket, in bucket-payload order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/prediction.py#L195)</small>

---

<a id="evaluate_predictor"></a>

### `evaluate_predictor()`

<small>`from hybridmodels.prediction import evaluate_predictor` &nbsp;·&nbsp; also re-exported as `hybridmodels.evaluate_predictor`</small>

```python
evaluate_predictor(predictor: 'Any', covariates: 'dict[str, float]') -> 'float'
```

Evaluate a scalar-valued predictor at named inputs, as a Python float.

Shortcut for the recovered-physics readout every example writes by hand
(``float(predictor({"k": jnp.asarray(v)}).reshape(()))``). Takes the
predictor's named inputs as plain Python floats, calls it, and flattens
the scalar result to a ``float``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/prediction.py#L126)</small>
