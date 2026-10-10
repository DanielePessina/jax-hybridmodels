# Models and Results

## Quick links

- [`HybridModel`](#hybridmodel)
- [`FitResult`](#fitresult)
- [`FitHistory`](#fithistory)
- [`PredictionResult`](#predictionresult)
- [`ExperimentPrediction`](#experimentprediction)

---

<a id="hybridmodel"></a>

### `HybridModel`

<small>`from jaxhybridmodels.model import HybridModel` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.HybridModel`</small>

```python
HybridModel(
    predictors: 'Any',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    output_channel_names: 'tuple[str, ...]',
    solver: 'SolverConfig',
) -> None
```

The four pieces of a hybrid model, bound together.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` |  | Any predictor pytree, usually a dict of ``Parameter``, ``Constant`` and ``BoundedPredictor`` leaves. |
| `simulate_fn` |  | Pure ``(predictors, ts, covariates, y0, solver) -> [T, S]``. |
| `state_to_output` |  | Pure ``[T, S] -> [T, D]``. |
| `output_channel_names` |  | Names of the ``D`` output columns, in order. Must match the ``output_channel_names`` of every dataset passed in. |
| `solver` |  | Integration settings. |
| `Fitting returns a new model; the original is unchanged.` |  |  |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L141)</small>

<a id="hybridmodelfit"></a>

#### `HybridModel.fit()`

```python
fit(
    self,
    dataset: 'Dataset',
    config: 'OptaxTrainingConfig | EvosaxTrainingConfig',
    key: 'Array',
    trainable: 'Any' = None,
    ui: 'Any' = None,
    validation: 'Dataset | None' = None,
) -> FitResult
```

Train the predictors and return a new model with the result.

``config`` picks the optimiser: Optax for gradient training, Evosax
for an evolutionary search. ``trainable`` defaults to
[`trainable_mask`](/api/trainable#trainable_mask).

With ``validation`` (Optax only), the validation data loss replaces
the training loss for choosing the returned step and for
``patience``. It costs one extra forward pass over the validation
set per step.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L184)</small>

<a id="hybridmodelpredict"></a>

#### `HybridModel.predict()`

```python
predict(self, dataset: 'Dataset') -> 'PredictionResult'
```

Predict every experiment on its measured time grid, keyed by experiment ID.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L266)</small>

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

Simulate new conditions without a dataset.

``y0`` is the state at ``t0``, which defaults to ``ts[0]``. ``ts``
must be finite, strictly increasing, and not before ``t0``. When
``t0`` precedes ``ts[0]`` it is integrated from but left out of
the returned times and outputs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L303)</small>

---

<a id="fitresult"></a>

### `FitResult`

<small>`from jaxhybridmodels.model import FitResult` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.FitResult`</small>

```python
FitResult(
    model: 'HybridModel',
    history: 'FitHistory',
    best_step: 'int | None',
    stop_reason: 'StopReason',
) -> None
```

The fitted model, its history, and how the run ended.

``best_step`` indexes the history at the returned parameters. It is
None when ``restore_best=False`` returned the final update instead.
``stop_reason`` is ``"patience"`` when early stopping cut the final
phase short, else ``"budget_exhausted"``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L125)</small>

---

<a id="fithistory"></a>

### `FitHistory`

<small>`from jaxhybridmodels.model import FitHistory` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.FitHistory`</small>

```python
FitHistory(
    loss: 'tuple[float, ...]' = (),
    penalty: 'tuple[float, ...]' = (),
    phase: 'tuple[int, ...]' = (),
    validation_loss: 'tuple[float, ...]' = (),
    best_fitness: 'tuple[float, ...]' = (),
) -> None
```

Per-step records from a fit.

Optax fills ``loss``, ``penalty`` and ``phase``, one entry per step,
measured at the parameters before that step's update. ``loss`` is the
data term plus any trajectory penalty; ``penalty`` is the weighted
bound penalty, so ``loss + penalty`` is the objective the optimiser
minimised. ``validation_loss`` is the full-horizon data loss on the
validation set, present when one was given.

Evosax fills only ``best_fitness``, the best-so-far fitness per
generation. It is monotone and not comparable to ``loss``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L103)</small>

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

Predictions keyed by experiment ID, in the order the experiments were given.

``buckets`` keeps the raw ``[N, T, D]`` arrays from ``predict_dataset``
for ``compute_metrics`` or custom JAX analysis. Experiments without an
ID are named ``experiment_<index>``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L50)</small>

<a id="predictionresultto_dataframe"></a>

#### `PredictionResult.to_dataframe()`

```python
to_dataframe(self) -> 'Any'
```

Long table: one row per experiment, time and channel.

Columns are ``experiment_id``, ``time``, ``channel``,
``prediction``, ``observed`` and ``is_observed``. Needs pandas,
which the ``examples`` extra installs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L71)</small>

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

``outputs`` maps each channel name to its ``[T]`` prediction.
``observed`` holds the measurements on the same grid, NaN where a cell
was not measured, and ``observation_mask`` says which cells were.
Both are empty for [`HybridModel.simulate`](/api/model#hybridmodelsimulate), which instead fills
``state`` with the full ``[T, S]`` simulator state.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L32)</small>
