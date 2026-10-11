# Saving and loading

Only the predictors and the solver settings are saved. `simulate_fn`,
`state_to_output`, datasets and trainable masks are code; keep them
importable and rebuild them when you load.

## Predictor weights

```python
hm.save_predictors("predictors.eqx", result.model.predictors)

template = build_predictors()           # same code that built the trained model
predictors = hm.load_predictors("predictors.eqx", template)
model = eqx.tree_at(lambda m: m.predictors, result.model, predictors)
```

The file holds array leaves only. Static fields (sizes, bounds, warps,
activation names) come from the template, so it must have the same
container, the same predictor classes and the same static values as the
saved pytree. Its array values do not matter. The library never imports
your code or builds a predictor from a class name.

## A complete run

`save_run` writes the weights plus a `metadata.json` with the package
version, the solver, a description of the predictor pytree, an optional
training config, objective and loss history, and anything you put in
`extras`:

```python
hm.save_run(
    "runs/oscillator",
    predictors=result.model.predictors,
    solver=result.model.solver,
    optax_config=config,
    objective=objective,
    loss_history=list(result.history.loss),
    extras={"dataset": "synthetic-oscillator"},
)

loaded = hm.load_run("runs/oscillator", predictors_template=build_predictors())
loaded["predictors"], loaded["solver"], loaded["objective"], loaded["loss_history"]
```

`load_run` checks the template against the saved description and raises
with the mismatching part named, instead of Equinox's generic shape error.
It also raises if the run was saved by a different package version.

The training config and objective come back as plain dicts. Pass
`optax_cls=hm.OptaxTrainingConfig` (or `evosax_cls=`) and
`objective_cls=hm.Objective` to rebuild them. Each `penalty_points` grid
is stored with its shape and comes back as an array of that shape, empty
grids included. A section holding callables, such as a custom loss, an
optimiser factory or an lr schedule, cannot be rebuilt that way and raises;
load it as a dict and set those fields yourself. A callable is stored as
`{"__callable__": "module.qualname"}`.

`loaded["loss_history_kind"]` says which trainer wrote the history:
`"per_step_data"` for Optax, `"best_so_far"` for Evosax.
