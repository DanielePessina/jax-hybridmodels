# Saving and loading

`hm.save` writes a model, or a fit with its history, to a directory.
`hm.load` reads it back. Only the predictors, the solver settings and the
records you pass are saved. `simulate_fn`, `state_to_output`, datasets and
trainable masks are code; keep them importable and rebuild them when you
load.

## Save and load

```python
result = model.fit(data, config, key=key, objective=objective)
hm.save(
    "runs/oscillator",
    result,
    config=config,
    objective=objective,
    extras={"dataset": "synthetic-oscillator"},
)

run = hm.load("runs/oscillator", like=build_model())
run.model, run.history, run.best_step, run.config, run.objective, run.extras
```

`save` takes a `HybridModel` or a `FitResult`. A `FitResult` also saves
its `history`, `best_step` and `stop_reason`; a bare model leaves them
empty on load. `config`, `objective` and `extras` (any JSON-encodable
dict) are optional.

`like` is a model built by the same code as the saved one. Its array
values do not matter. It supplies `simulate_fn` and `state_to_output`, and
the static fields of the predictors (sizes, bounds, warps, activation
names), which the leaf file does not hold. `run.model` is `like` with the
saved predictors. It keeps `like.solver`: the saved solver record names
the solver and adjoint classes but not their constructor arguments, so it
cannot rebuild, say, `RecursiveCheckpointAdjoint(checkpoints=7)`. The
library never imports your code or builds a predictor from a class name.

## What load checks

`load` compares `like` against the saved description of the predictors
and the saved `output_channel_names`, and raises with the mismatching part
named instead of Equinox's generic shape error. It also raises if the run
was saved by a different package version, and warns when `like.solver`
differs from the saved solver record.

## Configs and objectives

`run.config` comes back as an `OptaxTrainingConfig` or
`EvosaxTrainingConfig`, and `run.objective` as an `Objective`. Each
`penalty_points` grid is stored with its shape and comes back as an array
of that shape, empty grids included. A section holding callables, such as
a custom loss, an optimiser factory or an lr schedule, cannot be rebuilt,
so it comes back as its raw dict with each callable stored as
`{"__callable__": "module.qualname"}`. Set those fields yourself.

## Predictor weights only

For a bare predictor pytree, `jaxhybridmodels.serialise` has the
lower-level pair:

```python
from jaxhybridmodels.serialise import load_predictors, save_predictors

save_predictors("predictors.eqx", result.model.predictors)
predictors = load_predictors("predictors.eqx", like=build_predictors())
```

The file holds array leaves only, and nothing checks the template beyond
Equinox's shape check.
