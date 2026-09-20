# Serialise: Save & Load Runs

Serialisation uses Equinox's `tree_serialise_leaves` / `tree_deserialise_leaves` under the hood. [`save_predictors`](#save_predictors) / [`load_predictors`](#load_predictors) round-trip a single predictor tree; [`save_run`](#save_run) / [`load_run`](#load_run) bundle predictors, solver config, training config, and loss history into one directory.

**Loading requires a template.** Equinox can't reconstruct module shapes from a binary blob, so you instantiate the same predictor structure (same shapes, same Module types) and the loader fills its leaves.

## Quick links

- [`save_predictors`](#save_predictors)
- [`load_predictors`](#load_predictors)
- [`save_run`](#save_run)
- [`load_run`](#load_run)

---

<a id="save_predictors"></a>

### `save_predictors()`

<small>`from jaxhybridmodels.serialise import save_predictors` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.save_predictors`</small>

```python
save_predictors(path: 'str | Path', predictors: 'Any') -> 'None'
```

Write ``predictors`` to ``path`` via ``eqx.tree_serialise_leaves``.

``eqx.tree_serialise_leaves`` walks the leaves the same way whatever
the container shape, writing a flat binary stream of ``np.save``-encoded
leaves. The caller picks the file extension; ``save_run`` uses ``.eqx``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L68)</small>

---

<a id="load_predictors"></a>

### `load_predictors()`

<small>`from jaxhybridmodels.serialise import load_predictors` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.load_predictors`</small>

```python
load_predictors(path: 'str | Path', predictors_template: 'Any') -> 'Any'
```

Restore a ``predictors`` pytree from ``path`` using ``predictors_template`` as the skeleton.

``predictors_template`` must share the saved pytree's container shape
and per-leaf static configuration, such as the same ``in_size`` and
``depth`` on every ``MLPPredictor`` leaf. The file's values overwrite
the template's array leaves; its static fields supply the structure
``equinox`` needs. The template is not mutated.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L79)</small>

---

<a id="save_run"></a>

### `save_run()`

<small>`from jaxhybridmodels.serialise import save_run` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.save_run`</small>

```python
save_run(
    directory: 'str | Path',
    predictors: 'Any',
    solver: 'SolverConfig',
    optax_config: 'OptaxTrainingConfig | None' = None,
    evosax_config: 'EvosaxTrainingConfig | None' = None,
    loss_history: 'list[float] | None' = None,
    extras: 'dict[str, Any] | None' = None,
) -> None
```

Persist a complete training run to ``directory``.

**Layout written**

``directory/predictors.eqx``
    Binary leaves of the predictors pytree (``save_predictors`` output).
``directory/metadata.json``
    JSON dict with ``timestamp`` (UTC, ISO 8601, microsecond
    precision), ``version`` (``importlib.metadata.version``-resolved),
    ``predictors`` (``{tree_structure, leaves: [{path, class}, ...]}``
    per ``_describe_predictors``), ``solver`` (``solver.to_dict()``),
    ``optax_config`` / ``evosax_config`` (``dataclasses.asdict`` with
    stringified ``loss``, see module docstring), ``loss_history``,
    and ``extras``.

    ``loss_history_kind``, inferred from whichever config was passed,
    records which entry point produced ``loss_history``:
    ``"per_step_data"`` for Optax (can go up) or ``"best_so_far"`` for
    Evosax (monotone). The two series share a type and mean different
    things.

The directory is created, parents included. Existing files are
overwritten: this saves rather than appends.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L263)</small>

---

<a id="load_run"></a>

### `load_run()`

<small>`from jaxhybridmodels.serialise import load_run` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.load_run`</small>

```python
load_run(
    directory: 'str | Path',
    predictors_template: 'Any',
    optax_cls: 'type[OptaxTrainingConfig] | None' = None,
    evosax_cls: 'type[EvosaxTrainingConfig] | None' = None,
) -> dict[str, Any]
```

Reconstruct a run from ``directory``.

Inverse of [`save_run`](/api/serialise#save_run). ``predictors_template`` is required, since
the framework owns no builder registry, and must share the saved
pytree's container shape and per-leaf static configuration.

Pass ``optax_cls`` / ``evosax_cls`` to get the saved config dicts
rebuilt as typed dataclass instances; leave them out and the raw dicts
come back.

A ``loss`` field that was a callable at save time arrives as the string
``"{module}.{qualname}"`` and is returned as it stands. Turning it back
into a function is the caller's job, usually
``importlib.import_module(module).qualname``.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `dict` |  | Keys: ``predictors`` (``PyTree[eqx.Module]``), ``solver`` (``SolverConfig``), ``optax_config`` (``OptaxTrainingConfig`` \| dict \| None), ``evosax_config`` (``EvosaxTrainingConfig`` \| dict \| None), ``loss_history`` (``list[float] \| None``), ``loss_history_kind`` (``str \| None``), ``extras`` (``dict``). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/serialise.py#L388)</small>
