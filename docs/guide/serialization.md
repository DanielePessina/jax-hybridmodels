# Saving and loading

Use `save_predictors` for predictor weights. Use `save_run` when you also
want solver settings, training configuration, loss history, and application
metadata.

## Save predictor weights

```python
hm.save_predictors("predictors.eqx", trained_predictors)
```

Equinox writes the array leaves of the predictor PyTree. Static fields such as
dimensions, bounds, and activation names are supplied again through a
template when loading.

```python
template = build_predictors(key=jr.PRNGKey(0))
restored = hm.load_predictors("predictors.eqx", template)
```

The template must have the same container shape, predictor classes, and static
fields as the saved PyTree. `load_predictors` does not import user code or
construct a predictor from a class name.

## Save a complete run

```python
hm.save_run(
    "runs/oscillator",
    predictors=trained_predictors,
    solver=solver,
    optax_config=config,
    loss_history=history,
    extras={"dataset": "synthetic-oscillator"},
)
```

The directory contains:

```text
runs/oscillator/
├── predictors.eqx
└── metadata.json
```

The metadata records the package version, solver settings, predictor
structure, static predictor fields, optional training configuration, loss
history, and `extras`.

## Load a complete run

```python
loaded = hm.load_run(
    "runs/oscillator",
    predictors_template=build_predictors(key=jr.PRNGKey(0)),
)

trained_predictors = loaded["predictors"]
solver = loaded["solver"]
history = loaded["loss_history"]
```

Pass `optax_cls=OptaxTrainingConfig` or `evosax_cls=EvosaxTrainingConfig` if
the saved configuration contains only reconstructible values. Configurations
with callable factories, losses, or regularisers must be rebound explicitly;
loading without a class returns their metadata markers.

## What is not saved

The run format does not store `simulate_fn`, `state_to_output`, `Dataset`, or
the trainability mask. Keep those in importable user code and rebuild them
when you resume or evaluate a run.

