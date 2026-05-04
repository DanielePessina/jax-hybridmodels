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
one experiment to produce a full state trajectory ``[T, S]`` and then
projects to the observed channels ``[T, D]`` via ``state_to_output``.
``jax.vmap`` lifts this over ``(ts, covariates, y0)`` along the ``N``
axis to produce ``[N, T, D]``. ``predictors`` and ``solver`` are
closed over (no vmap axis) — they are constant across the bucket.

JIT caching: one compiled trace per bucket *shape*. The Python
dispatch over ``dataset.bucket_payloads`` lives in
``predict_dataset``, never inside the jitted region — that boundary
is what keeps the cache predictable.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` | `PyTree[eqx.Module]` | Trainable component, typically a tuple of ``BoundedPredictor`` leaves but accepted as any pytree shape. Forwarded to ``simulate_fn`` as-is; this module does not inspect the container. |
| `bp` | `BucketPayload` | One bucket; ``ts``, ``covariates``, ``y0`` are vmapped along ``N``. |
| `simulate_fn` |  | User-supplied integrator with signature ``(predictors, ts, covariates, y0, solver) -> [T, S]``. |
| `state_to_output` |  | Pure ``[T, S] -> [T, D]`` projector held on ``Dataset``. |
| `solver` |  | Static ``SolverConfig``. |

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

The Python ``for`` loop over ``dataset.bucket_payloads`` is the dispatch
driver: each bucket shape compiles ``predict_bucket`` exactly once.
Returns a tuple aligned with ``dataset.bucket_payloads`` order, *not*
a flat concatenation — each entry has its own ``[N_b, T_b, D]``
shape and cannot be stacked into a single tensor (the buckets differ
precisely in ``T``).

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `tuple[Float[Array, "N T D"], ...]` |  | One ``[N_b, T_b, D]`` array per bucket, in bucket-payload order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/prediction.py#L95)</small>
