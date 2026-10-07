# Models and Results

## Quick links

- [`HybridModel`](#hybridmodel)
- [`FitResult`](#fitresult)
- [`FitHistory`](#fithistory)
- [`PredictionResult`](#predictionresult)
- [`ExperimentPrediction`](#experimentprediction)
- [`fit`](#fit)
- [`predict`](#predict)

---

<a id="hybridmodel"></a>

### `HybridModel`

<small>`from jaxhybridmodels.model import HybridModel` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.HybridModel`</small>

```python
HybridModel(
    predictors: 'Any',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    output_channels: 'tuple[str, ...]',
    solver: 'SolverConfig',
) -> None
```

Hold the predictors, simulator, observation map, and solver together.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` |  | Any predictor PyTree, including named dictionaries. |
| `simulate_fn` |  | Pure ``(predictors, ts, covariates, y0, solver) -> [T, S]`` callback. |
| `state_to_output` |  | Pure ``[T, S] -> [T, D]`` callback. |
| `output_channels` |  | Ordered names matching the columns returned by state_to_output. |
| `solver` |  | Integration settings. |

**Notes**

Fitting returns a new model; the original remains unchanged. The
functional APIs accept these same pieces separately.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L148)</small>

<a id="hybridmodelfit"></a>

#### `HybridModel.fit()`

```python
fit(
    self,
    dataset: 'Dataset',
    config: 'OptaxTrainingConfig | EvosaxTrainingConfig | None' = None,
    key: 'Array',
    steps: 'int | None' = None,
    lr: 'float | None' = None,
    trainable: 'Any' = None,
    ui: 'Any' = None,
    validation: 'Dataset | None' = None,
    select_best: 'str | None' = None,
) -> FitResult
```

Fit with Optax by default, or with an Evosax config.

Supply a config, or use ``steps=`` and ``lr=`` for a simple Optax
run. ``select_best`` is ``training_loss`` or ``validation_loss``;
it defaults to validation when supplied, otherwise training.
Validation scoring is an additional forward pass at every step.
Validation selection and patience are supported for Optax only.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L188)</small>

<a id="hybridmodelpredict"></a>

#### `HybridModel.predict()`

```python
predict(self, dataset: 'Dataset') -> 'PredictionResult'
```

Predict measured union grids, with named experiment/channel access.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L222)</small>

<a id="hybridmodelsimulate"></a>

#### `HybridModel.simulate()`

```python
simulate(
    self,
    ts: 'Any',
    y0: 'Any',
    covariates: 'Mapping[str, Any] | None' = None,
    t0: 'float | None' = None,
) -> ExperimentPrediction
```

Simulate new conditions without an observation dataset.

``y0`` is the state at ``t0`` (default: the first requested time).
Times must be finite, strictly increasing, and at or after t0.
A leading integration point is added when needed and omitted from
the returned times and outputs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L226)</small>

---

<a id="fitresult"></a>

### `FitResult`

<small>`from jaxhybridmodels.model import FitResult` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.FitResult`</small>

```python
FitResult(
    model: 'HybridModel',
    history: 'FitHistory',
    best_step: 'int | None',
    stop_reason: 'str',
    selected_by: 'str',
) -> None
```

A fitted model, loss histories, and the reason fitting stopped.

``best_step`` is zero-based, referring to Optax's pre-update parameters
or the first Evosax generation reporting the best fitness (which may
include the input seed). It is None
when Optax returns the final update instead of a selected checkpoint.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L131)</small>

---

<a id="fithistory"></a>

### `FitHistory`

<small>`from jaxhybridmodels.model import FitHistory` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.FitHistory`</small>

```python
FitHistory(
    data_loss: 'tuple[float, ...]' = (),
    tracked_loss: 'tuple[float, ...]' = (),
    objective: 'tuple[float, ...]' = (),
    validation_loss: 'tuple[float, ...]' = (),
    phase: 'tuple[int, ...]' = (),
    best_fitness: 'tuple[float, ...]' = (),
) -> None
```

Optax pre-update losses, or Evosax best-so-far fitness.

``data_loss`` excludes all penalties. ``tracked_loss`` includes the
trajectory penalty, while ``objective`` includes both trajectory and
fixed-point penalties. Validation uses full-horizon data loss. Evosax
fills only ``best_fitness``: its generation history has different
semantics from Optax steps.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L112)</small>

---

<a id="predictionresult"></a>

### `PredictionResult`

<small>`from jaxhybridmodels.model import PredictionResult` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.PredictionResult`</small>

```python
PredictionResult(
    experiments: 'dict[str, ExperimentPrediction]',
    buckets: 'tuple[Array, ...]' = (),
) -> None
```

Predictions keyed by experiment ID, retaining input experiment order.

``buckets`` exposes the original arrays for use with ``compute_metrics``
or custom JAX analysis. ``to_dataframe`` requires optional pandas.
Missing experiment IDs receive ``experiment_<index>`` names; collisions
raise rather than overwrite experiments.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L47)</small>

<a id="predictionresultto_dataframe"></a>

#### `PredictionResult.to_dataframe()`

```python
to_dataframe(self) -> 'Any'
```

Export a long table with experiment, time, channel, and prediction.

Columns also include ``observed`` and ``is_observed``. Install
pandas separately, or use the package's ``examples`` extra.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L69)</small>

---

<a id="experimentprediction"></a>

### `ExperimentPrediction`

<small>`from jaxhybridmodels.model import ExperimentPrediction` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.ExperimentPrediction`</small>

```python
ExperimentPrediction(
    times: 'Array',
    outputs: 'dict[str, Array]',
    observed: 'dict[str, Array]' = <factory>,
    observation_mask: 'dict[str, Array]' = <factory>,
    state: 'Array | None' = None,
) -> None
```

One experiment's times, named outputs, and optional observed values.

Unobserved entries in ``observed`` are NaN. ``observation_mask`` states
which cells are real measurements. ``state`` is present for simulations
made directly with [`HybridModel.simulate`](/api/model#hybridmodelsimulate).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L28)</small>

---

<a id="fit"></a>

### `fit()`

<small>`from jaxhybridmodels.model import fit` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.fit`</small>

```python
fit(
    model: 'HybridModel',
    dataset: 'Dataset',
    config: 'OptaxTrainingConfig | EvosaxTrainingConfig | None' = None,
    key: 'Array',
    steps: 'int | None' = None,
    lr: 'float | None' = None,
    trainable: 'Any' = None,
    ui: 'Any' = None,
    validation: 'Dataset | None' = None,
    select_best: 'str | None' = None,
) -> FitResult
```

Functional equivalent of ``model.fit``; return a new fitted model.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L320)</small>

---

<a id="predict"></a>

### `predict()`

<small>`from jaxhybridmodels.model import predict` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.predict`</small>

```python
predict(model: 'HybridModel', dataset: 'Dataset') -> 'PredictionResult'
```

Functional equivalent of ``model.predict(dataset)``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L277)</small>
