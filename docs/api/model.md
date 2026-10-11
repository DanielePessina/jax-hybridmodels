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
    simulate_fn: 'Callable[..., Any]',
    state_to_output: 'Callable[[Array], Array]',
    output_channel_names: 'Sequence[str]',
    solver: 'SolverConfig | None' = None,
) -> None
```

The four pieces of a hybrid model, bound together.

An ``eqx.Module``: ``predictors`` is its only array subtree, and the
other fields are static. Equinox tools work on it directly, for example
``eqx.tree_at(lambda m: m.predictors["rate"], model, new_rate)`` to swap
one predictor, or ``eqx.filter_vmap`` over a function that builds models
from different keys.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictors` |  | Any predictor pytree, usually a dict of ``Parameter``, ``Constant`` and ``BoundedPredictor`` leaves. |
| `simulate_fn` |  | Pure ``(predictors, ts, covariates, y0, solver)``, returning the ``[T, S]`` state at ``ts`` or the ``diffrax.Solution`` that holds it. `from_vector_field` writes one from a vector field. |
| `state_to_output` |  | Pure ``[T, S] -> [T, D]``. |
| `output_channel_names` |  | Names of the ``D`` output columns, in order. Must match the ``output_channel_names`` of every dataset passed in. |
| `solver` |  | Integration settings, default ``SolverConfig()``. |
| `Fitting returns a new model; the original is unchanged.` |  |  |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L71)</small>

<a id="hybridmodelfit"></a>

#### `HybridModel.fit()`

```python
fit(
    self,
    data: 'Data',
    config: 'OptaxTrainingConfig | EvosaxTrainingConfig',
    key: 'Array',
    objective: 'Objective | None' = None,
    freeze: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
    trainable: 'Any' = None,
    ui: 'Any' = None,
    validation: 'Data | None' = None,
) -> FitResult
```

Train the predictors and return a new model with the result.

``config`` says how to search: Optax for gradient training, Evosax
for an evolutionary search. ``objective`` says what is minimised;
``None`` means ``Objective()``, plain MSE with no penalty.

``freeze`` holds fixed, on top of the default
[`trainable_mask`](/api/trainable#trainable_mask), whatever its
entries match: dotted paths (``"rate"``, ``"rate.inner"``),
predictor classes (``hm.MLPPredictor``), or predicates
``(module) -> bool``. ``trainable`` is an explicit mask for full
control; pass one or the other.

With ``validation``, the validation data loss replaces the training
loss for choosing the returned step (Optax) or generation (Evosax),
and for ``patience``. It costs one extra forward pass over the
validation set per step or generation.

``data`` and ``validation`` may be a ``Dataset`` or a sequence of
experiments. ``simulate_fn`` and ``state_to_output`` shapes are
checked once on the first experiment before anything compiles.

Same as ``train_with_optax(model, ...)`` or
``train_with_evosax(model, ...)`` in `jaxhybridmodels.training`,
picked by the type of ``config``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L249)</small>

<a id="hybridmodelfit_ensemble"></a>

#### `HybridModel.fit_ensemble()`

```python
fit_ensemble(
    self,
    data: 'Data',
    config: 'OptaxTrainingConfig',
    key: 'Array',
    n_members: 'int',
    n_seeds: 'int | None' = None,
    bootstrap: 'bool' = False,
    objective: 'Objective | None' = None,
    freeze: 'Sequence[FreezeSpec] | FreezeSpec | None' = None,
    trainable: 'Any' = None,
    ui: 'Any' = None,
) -> list[FitResult]
```

Train an ensemble of up to ``n_members`` models, best first.

Every member starts from a fresh initialisation of the predictors
drawn from ``key``. A candidate seed is trained briefly
(``config.tournament_steps`` steps at ``config.tournament_lr``) and
scored, as in the tournament, then the chosen seeds get the full
phase schedule.

Seed ensemble (``bootstrap=False``)
    Score ``n_seeds`` seeds on all of ``data`` and fully train the
    best ``n_members``. ``n_seeds`` defaults to ``n_members``, which
    trains every seed.
Bootstrap ensemble (``bootstrap=True``)
    Draw ``n_members`` resamples of the experiments with replacement
    ([`make_bootstrap_dataset`](/api/data#make_bootstrap_dataset)) and train one
    member on each: the best of ``n_seeds`` seeds scored on that
    resample. ``n_seeds`` defaults to 1, one fresh seed per resample.

``config.tournament_attempts`` is not used: ``n_seeds`` takes its
place. ``objective``, ``freeze``, ``trainable`` and ``ui`` are as
for `fit`; each member is its own UI run. The result is sorted
by each member's final loss, the loss at its returned parameters
when ``config.restore_best`` is set. A seed whose warm-up fails is
dropped, so fewer than ``n_members`` may come back; if every seed
fails it raises ``RuntimeError``. Optax only. The list goes straight
into [`predict_ensemble`](/api/prediction#predict_ensemble).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L304)</small>

<a id="hybridmodelpredict"></a>

#### `HybridModel.predict()`

```python
predict(
    self,
    data: 'Data',
    ts: 'Any' = None,
    n_points: 'int | None' = None,
) -> PredictionResult
```

Predict every experiment, keyed by experiment ID.

By default each experiment is predicted on its measured time grid
(``t0`` included when set), with its observations alongside, ready
for ``compute_metrics``. ``n_points`` instead gives each experiment
an even grid of that many points over its own time span, from its
``t0`` (or first observation) to its last observation, for smooth
plots. ``ts`` gives one explicit grid for every experiment; it must
not start before any experiment's start. Off the measured grid,
``observed`` and ``observation_mask`` are empty. Each experiment
still integrates from its own start, so ``y0`` stays in place.

``data`` may be a ``Dataset`` from ``make_dataset`` or a sequence of
experiments.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L364)</small>

<a id="hybridmodelsimulate"></a>

#### `HybridModel.simulate()`

```python
simulate(
    self,
    experiment: 'Experiment | None' = None,
    ts: 'Any' = None,
    n_points: 'int | None' = None,
    y0: 'Any' = None,
    covariates: 'Mapping[str, Any] | None' = None,
    t0: 'float | None' = None,
) -> ExperimentPrediction
```

Simulate one set of conditions, from an experiment or given directly.

With ``experiment``, its ``y0``, ``t0`` and covariates are used, and
``ts`` defaults to its measured times (``t0`` included when set).
``n_points`` instead gives an even grid of that many points from the
experiment's start to its last observation, as in `predict`.
``y0`` is placed at the experiment's ``t0``, or at its first
observation time when ``t0`` is unset, whatever ``ts`` is passed; a
``ts`` starting before that time is rejected.
Otherwise pass ``ts`` and ``y0``, and optionally ``covariates`` and
``t0``.

``y0`` is the state at ``t0``, which defaults to ``ts[0]``. ``ts``
must be finite, strictly increasing, and not before ``t0``. When
``t0`` precedes ``ts[0]`` it is integrated from but left out of the
returned times and outputs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/model.py#L390)</small>

---

<a id="fitresult"></a>

### `FitResult`

<small>`from jaxhybridmodels.results import FitResult` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.FitResult`</small>

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

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/results.py#L132)</small>

---

<a id="fithistory"></a>

### `FitHistory`

<small>`from jaxhybridmodels.results import FitHistory` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.FitHistory`</small>

```python
FitHistory(
    loss: 'tuple[float, ...]' = (),
    penalty: 'tuple[float, ...]' = (),
    phase: 'tuple[int, ...]' = (),
    validation_loss: 'tuple[float, ...]' = (),
) -> None
```

Per-step records from a fit.

``loss`` has one entry per step (Optax) or per generation (Evosax).

Optax measures it at the parameters before that step's update: the data
term plus any trajectory penalty. ``penalty`` is the weighted bound
penalty at the same step, so ``loss + penalty`` is what the optimiser
minimised, and ``phase`` is the step's phase index.

Evosax records the best fitness found so far, so its ``loss`` is
monotone non-increasing; with a bound penalty on, that fitness includes
it. ``penalty`` and ``phase`` stay empty.

``validation_loss`` is the full-horizon data loss on the validation set,
present when one was given: per step for Optax, and for each
generation's best individual for Evosax.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/results.py#L106)</small>

---

<a id="predictionresult"></a>

### `PredictionResult`

<small>`from jaxhybridmodels.results import PredictionResult` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.PredictionResult`</small>

```python
PredictionResult(
    experiments: 'dict[str, ExperimentPrediction]',
    buckets: 'tuple[Array, ...]' = (),
    members: 'tuple[PredictionResult, ...]' = (),
) -> None
```

Predictions keyed by experiment ID, in the order the experiments were given.

Pass it straight to ``compute_metrics`` or ``print_metrics``.
``buckets`` keeps the raw ``[N, T, D]`` arrays, one per bucket of the
dataset predicted on, for custom JAX analysis. For an ensemble,
``outputs`` and ``buckets`` hold the member mean and ``members`` holds
each member's own result, in the order the models were given.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/results.py#L49)</small>

<a id="predictionresultto_dataframe"></a>

#### `PredictionResult.to_dataframe()`

```python
to_dataframe(self) -> 'Any'
```

Long table: one row per experiment, time and channel.

Columns are ``experiment_id``, ``time``, ``channel``,
``prediction``, ``observed`` and ``is_observed``, plus ``std`` for
an ensemble. Needs pandas, which the ``examples`` extra installs.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/results.py#L73)</small>

---

<a id="experimentprediction"></a>

### `ExperimentPrediction`

<small>`from jaxhybridmodels.results import ExperimentPrediction` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.ExperimentPrediction`</small>

```python
ExperimentPrediction(
    times: 'Array',
    outputs: 'dict[str, Array]',
    observed: 'dict[str, Array]' = <factory>,
    observation_mask: 'dict[str, Array]' = <factory>,
    state: 'Array | None' = None,
    std: 'dict[str, Array]' = <factory>,
) -> None
```

One experiment's times, named outputs, and optional observed values.

``outputs`` maps each channel name to its ``[T]`` prediction.
``observed`` holds the measurements on the same grid, NaN where a cell
was not measured, and ``observation_mask`` says which cells were. Both
are empty when the times are not the measured grid: for
[`HybridModel.simulate`](/api/model#hybridmodelsimulate), and for ``predict`` with ``ts=`` or
``n_points=``. ``simulate`` fills ``state`` with the full ``[T, S]``
simulator state. ``std`` is the per-channel spread across members, set
only by [`predict_ensemble`](/api/prediction#predict_ensemble).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/results.py#L27)</small>
