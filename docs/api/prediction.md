# Prediction: Forward Simulation

Forward-simulate trained predictors against a dataset. [`predict_bucket`](#predict_bucket) is the single-bucket primitive (JIT-compiled, vmapped over the bucket's `N` axis); [`predict_dataset`](#predict_dataset) walks every bucket and returns one `[N, T, D]` array per bucket.

## Quick links

- [`predict_bucket`](#predict_bucket)
- [`predict_dataset`](#predict_dataset)

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

The inner ``_per_experiment`` runs the user's ``simulate_fn`` once for
one experiment, producing a full state trajectory ``[T, S]``, and maps
it to the observed channels ``[T, D]`` with ``state_to_output``.
``jax.vmap`` lifts that over ``(ts, covariates, y0)`` along the ``N``
axis, giving ``[N, T, D]``. ``predictors`` and ``solver`` are closed
over with no vmap axis, since they are the same for every experiment in
the bucket.

One compiled kernel exists per bucket shape. The Python dispatch over
``dataset.bucket_payloads`` lives in ``predict_dataset`` and never
inside the compiled region. That boundary is what keeps the cache
predictable.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` | `PyTree[eqx.Module]` | The trainable part of the model, typically a tuple of ``BoundedPredictor`` leaves, accepted in any pytree shape. Forwarded to ``simulate_fn`` unchanged; this module does not inspect the container. |
| `bp` | `BucketPayload` | One bucket. Its ``ts``, ``covariates``, and ``y0`` are vmapped along ``N``. |
| `simulate_fn` |  | User-supplied integrator with signature ``(predictors, ts, covariates, y0, solver) -> [T, S]``. |
| `state_to_output` |  | Pure ``[T, S] -> [T, D]`` map from full state to observed channels, held on the ``Dataset``. |
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
    solver: 'SolverConfig',
) -> tuple[Float[Array, 'N T D'], ...]
```

Run ``predict_bucket`` over every bucket in ``dataset`` and return the stack tuple.

The Python ``for`` loop over ``dataset.bucket_payloads`` drives the
dispatch, and each bucket shape compiles ``predict_bucket`` exactly
once.

The result is a tuple in ``dataset.bucket_payloads`` order rather than
one concatenated array. Buckets differ precisely in ``T``, so each
entry has its own ``[N_b, T_b, D]`` shape and they cannot be stacked.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Float[Array, "N T D"], ...]` |  | One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/prediction.py#L98)</small>
