# Saving and loading

Only the predictors and the solver settings are saved. `simulate_fn`,
`state_to_output`, datasets and trainable masks are code; keep them
importable and rebuild them when you load.

## Predictor weights

```python
hm.save_predictors("predictors.eqx", result.model.predictors)

template = build_predictors()           # same code that built the trained model
predictors = hm.load_predictors("predictors.eqx", template)
model = dataclasses.replace(result.model, predictors=predictors)
```

The file holds array leaves only. Static fields (sizes, bounds, warps,
activation names) come from the template, so it must have the same
container, the same predictor classes and the same static values as the
saved pytree. Its array values do not matter. The library never imports
your code or builds a predictor from a class name.

## A complete run

`save_run` writes the weights plus a `metadata.json` with the package
version, the solver, a description of the predictor pytree, an optional
training config and loss history, and anything you put in `extras`:

```python
hm.save_run(
    "runs/oscillator",
    predictors=result.model.predictors,
    solver=result.model.solver,
    optax_config=config,
    loss_history=list(result.history.loss),
    extras={"dataset": "synthetic-oscillator"},
)

loaded = hm.load_run("runs/oscillator", predictors_template=build_predictors())
loaded["predictors"], loaded["solver"], loaded["loss_history"], loaded["extras"]
```

`load_run` checks the template against the saved description and raises
with the mismatching part named, instead of Equinox's generic shape error.
It also raises if the run was saved by a different package version.

The training config comes back as a plain dict. Pass
`optax_cls=hm.OptaxTrainingConfig` (or `evosax_cls=`) to rebuild the
dataclass. A config holding callables, such as a custom loss or optimiser
factory, cannot be rebuilt that way and raises; load it as a dict and set
those fields yourself. A callable `loss` is stored as its
`"module.qualname"` string.
