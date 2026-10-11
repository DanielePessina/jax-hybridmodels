# Prediction: Forward Simulation

`HybridModel.predict` and `HybridModel.simulate` cover one model. [`predict_ensemble`](#predict_ensemble) predicts with several and returns the member mean as a `PredictionResult`, with the per-channel spread in each experiment's `std` and each member's own result in `members`. [`evaluate_predictor`](#evaluate_predictor) calls one predictor outside any ODE.

## Quick links

- [`predict_ensemble`](#predict_ensemble)
- [`evaluate_predictor`](#evaluate_predictor)

---

<a id="predict_ensemble"></a>

### `predict_ensemble()`

<small>`from jaxhybridmodels.prediction import predict_ensemble` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.predict_ensemble`</small>

```python
predict_ensemble(
    models: 'Sequence[HybridModel | FitResult]',
    data: 'Data',
    ts: 'ArrayLike | None' = None,
    n_points: 'int | None' = None,
) -> PredictionResult
```

Predict with every model and return the member mean, with the spread.

``outputs`` (and ``buckets``) hold the mean across members, so
``compute_metrics`` scores the ensemble mean. Each experiment's ``std``
holds the per-channel standard deviation across members (population,
``ddof=0``; zero for one member), and ``members`` holds each model's
own [`PredictionResult`](/api/model#predictionresult).

``models`` holds ``HybridModel``\ s or ``FitResult``\ s, so the list
``fit_ensemble`` returns can be passed as is. They must share
``output_channel_names``. ``data``, ``ts`` and ``n_points`` are as for
[`HybridModel.predict`](/api/model#hybridmodelpredict).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/prediction.py#L201)</small>

---

<a id="evaluate_predictor"></a>

### `evaluate_predictor()`

<small>`from jaxhybridmodels.prediction import evaluate_predictor` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.evaluate_predictor`</small>

```python
evaluate_predictor(predictor: 'Any', inputs: 'Mapping[str, ArrayLike]') -> 'Array'
```

Evaluate a predictor at named inputs, vectorised over array-valued ones.

The inputs are broadcast against each other, so scalars, a 1-D sweep,
or a 2-D grid built from ``x[:, None]`` and ``y[None, :]`` all work.
The predictor is called once per broadcast point with a dict of scalars,
the way a ``simulate_fn`` calls it, and the result has shape
``broadcast_shape + output_shape``. A one-output ``BoundedPredictor``
has output shape ``()``, so a sweep over 50 temperatures gives ``[50]``.

**Examples**

::

    hm.evaluate_predictor(predictors["rate"], {"T": 300.0})  # shape ()
    hm.evaluate_predictor(predictors["rate"], {"T": temps, "pH": 6.5})  # [len(temps)]

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `Array` |  | Call ``float(...)`` on a scalar result for a Python number. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/prediction.py#L268)</small>
