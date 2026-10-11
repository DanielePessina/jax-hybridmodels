# Serialise: Save & Load Runs

[`save`](#save) writes a `HybridModel` or a `FitResult` to a directory, with the solver, and optionally the training config and objective. [`load`](#load) reads it back into a [`SavedRun`](#savedrun).

**Loading needs a template.** Equinox cannot rebuild module shapes from the leaf file, so `load(path, like=model)` takes a model built the same way and fills its leaves. The template also supplies `simulate_fn` and `state_to_output`, which are code and are not saved. [`save_predictors`](#save_predictors) / [`load_predictors`](#load_predictors) do the same for a bare predictor tree.

## Quick links

- [`save`](#save)
- [`load`](#load)
- [`SavedRun`](#savedrun)
- [`save_predictors`](#save_predictors)
- [`load_predictors`](#load_predictors)

---

<a id="save"></a>

### `save()`

<small>`from jaxhybridmodels.serialise import save` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.save`</small>

```python
save(
    path: 'str | Path',
    run: 'HybridModel | FitResult',
    config: 'OptaxTrainingConfig | EvosaxTrainingConfig | None' = None,
    objective: 'Objective | None' = None,
    extras: 'dict[str, Any] | None' = None,
) -> None
```

Write a model, or a fit with its history, to the directory ``path``.

``run`` is a ``HybridModel`` or the ``FitResult`` from ``fit``; a
``FitResult`` also saves its ``history``, ``best_step`` and
``stop_reason``. ``config`` and ``objective`` record how the model was
trained, and ``extras`` is any JSON-encodable dict of your own.
Read it back with [`load`](/api/serialise#load).

**Layout written**

``path/predictors.eqx``
    Binary leaves of the predictors pytree (``save_predictors`` output).
``path/metadata.json``
    ``timestamp`` (UTC, ISO 8601), ``version``, ``predictors`` (the
    structural fingerprint ``load`` checks ``like`` against),
    ``output_channel_names``, ``solver`` (``SolverConfig.to_dict()``),
    ``config`` (``{"kind": "optax" | "evosax", "fields": {...}}``),
    ``objective``, ``history`` (the four ``FitHistory`` series),
    ``best_step``, ``stop_reason`` and ``extras``. Callables are
    described, not pickled (see the module docstring).

The directory is created, parents included. Existing files are
overwritten.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L329)</small>

---

<a id="load"></a>

### `load()`

<small>`from jaxhybridmodels.serialise import load` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.load`</small>

```python
load(path: 'str | Path', *, like: 'HybridModel') -> 'SavedRun'
```

Read back a run written by [`save`](/api/serialise#save).

``like`` is a model built the same way as the saved one: same
predictor architecture and static configuration, same
``output_channel_names``. It supplies ``simulate_fn`` and
``state_to_output``, which are not saved, and the solver. The returned
``SavedRun.model`` is ``like`` with the saved predictors. The saved
solver record holds class names only (constructor arguments such as
``RecursiveCheckpointAdjoint(checkpoints=7)`` are not kept), so
``like.solver`` is used as is, with a ``RuntimeWarning`` when it
differs from the record.

Raises ``ValueError`` when the saved package version differs from the
installed one, when ``like`` does not match the saved predictors or
channel names, or when a saved config has a field the current class
does not know.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L466)</small>

---

<a id="savedrun"></a>

### `SavedRun`

<small>`from jaxhybridmodels.results import SavedRun` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.SavedRun`</small>

```python
SavedRun(
    model: 'HybridModel',
    history: 'FitHistory',
    best_step: 'int | None',
    stop_reason: 'StopReason | None',
    config: 'OptaxTrainingConfig | EvosaxTrainingConfig | dict[str, Any] | None',
    objective: 'Objective | dict[str, Any] | None',
    extras: 'dict[str, Any]',
) -> None
```

A run read back by [`load`](/api/serialise#load).

``model`` is the ``like=`` model with the saved predictors; it keeps
``like``'s solver.
``history``, ``best_step`` and ``stop_reason`` are the saved
``FitResult``'s; a saved bare model leaves them as an empty
``FitHistory`` and None.

``config`` and ``objective`` come back as typed objects. A section that
names a callable (a custom loss, a ``penalty_fn``, an optimizer factory,
an lr schedule) cannot be rebuilt, so it comes back as the raw dict
with each callable as ``{"__callable__": "module.qualname"}``; rebind
those fields yourself. ``extras`` is the dict passed to ``save``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/results.py#L160)</small>

---

<a id="save_predictors"></a>

### `save_predictors()`

<small>`from jaxhybridmodels.serialise import save_predictors`</small>

```python
save_predictors(path: 'str | Path', predictors: 'Any') -> 'None'
```

Write ``predictors`` to ``path`` via ``eqx.tree_serialise_leaves``.

``eqx.tree_serialise_leaves`` walks the leaves the same way whatever
the container shape, writing a flat binary stream of ``np.save``-encoded
leaves. The caller picks the file extension; ``save`` uses ``.eqx``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L77)</small>

---

<a id="load_predictors"></a>

### `load_predictors()`

<small>`from jaxhybridmodels.serialise import load_predictors`</small>

```python
load_predictors(path: 'str | Path', *, like: 'Any') -> 'Any'
```

Restore a ``predictors`` pytree from ``path``, using ``like`` as the skeleton.

``like`` must share the saved pytree's container shape and per-leaf
static configuration, such as the same ``in_size`` and ``depth`` on
every ``MLPPredictor`` leaf. The file's values overwrite its array
leaves; its static fields supply the structure ``equinox`` needs.
``like`` is not mutated.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L88)</small>
