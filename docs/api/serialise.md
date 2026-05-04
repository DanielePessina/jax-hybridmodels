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

<small>`from hybridmodels.serialise import save_predictors` &nbsp;·&nbsp; also re-exported as `hybridmodels.save_predictors`</small>

```python
save_predictors(path: 'str | Path', predictors: 'Any') -> 'None'
```

Write ``predictors`` to ``path`` via ``eqx.tree_serialise_leaves``.

``predictors`` is a ``PyTree[eqx.Module]`` of any container shape
(tuple / list / dict / NamedTuple / bare Module). The helper
delegates to ``eqx.tree_serialise_leaves``, which walks the pytree's
leaves uniformly regardless of container type, producing a flat
binary stream of ``np.save``-encoded leaves. The caller picks the
file extension; ``.eqx`` is the convention used by ``save_run``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/serialise.py#L78)</small>

---

<a id="load_predictors"></a>

### `load_predictors()`

<small>`from hybridmodels.serialise import load_predictors` &nbsp;·&nbsp; also re-exported as `hybridmodels.load_predictors`</small>

```python
load_predictors(path: 'str | Path', predictors_template: 'Any') -> 'Any'
```

Restore a ``predictors`` pytree from ``path`` using ``predictors_template`` as the skeleton.

``predictors_template`` must share the saved pytree's container shape
and per-leaf static configuration (e.g. matching ``MLPPredictor``
``in_size`` / ``out_size`` / ``width_size`` / ``depth`` for every
predictor leaf). Its dynamic leaves are overwritten by the values
stored in the file; its static fields are kept and provide the
structure ``equinox`` needs to reconstruct the tree.

Returns the restored pytree — does **not** mutate
``predictors_template``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/serialise.py#L92)</small>

---

<a id="save_run"></a>

### `save_run()`

<small>`from hybridmodels.serialise import save_run` &nbsp;·&nbsp; also re-exported as `hybridmodels.save_run`</small>

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
    stringified ``loss`` — see module docstring), ``loss_history``,
    and ``extras``.

The directory is created (parents included) if missing. Pre-existing
files are overwritten — this is a save, not an append.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/serialise.py#L219)</small>

---

<a id="load_run"></a>

### `load_run()`

<small>`from hybridmodels.serialise import load_run` &nbsp;·&nbsp; also re-exported as `hybridmodels.load_run`</small>

```python
load_run(
    directory: 'str | Path',
    predictors_template: 'Any',
    optax_cls: 'type[OptaxTrainingConfig] | None' = None,
    evosax_cls: 'type[EvosaxTrainingConfig] | None' = None,
) -> dict[str, Any]
```

Reconstruct a run from ``directory``.

Inverse of :func:`save_run`. ``predictors_template`` is required (the
framework deliberately does not own a builder registry — see the
module docstring) and must share the saved pytree's container shape
and per-leaf static configuration. ``optax_cls`` and ``evosax_cls``
are optional — pass them only when you want the metadata's
``optax_config`` / ``evosax_config`` dicts reconstituted into typed
dataclass instances. When omitted, the raw dicts flow through.

Loss-field policy
-----------------
If a config's ``loss`` field was a callable at save time, it lands here
as a string ``"{module}.{qualname}"``. The dataclass field type
(``Callable | str``) accepts the string verbatim — no implicit re-import
happens. Re-resolution is the caller's responsibility (typically
``importlib.import_module(module).qualname``); we keep that boundary
explicit because dynamic imports inside a load helper invite confusing
failure modes.

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `dict` |  | Keys: ``predictors`` (``PyTree[eqx.Module]``), ``solver`` (``SolverConfig``), ``optax_config`` (``OptaxTrainingConfig`` \| dict \| None), ``evosax_config`` (``EvosaxTrainingConfig`` \| dict \| None), ``loss_history`` (``list[float] \| None``), ``extras`` (``dict``). |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/serialise.py#L325)</small>
