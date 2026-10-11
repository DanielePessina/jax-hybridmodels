# Ensembles

An ensemble is several trained predictor pytrees whose predictions you
average. Seed ensembles vary the initialisation; bootstrap ensembles vary
the data. Both helpers return `(final_loss, predictors)` pairs sorted best
first.

The helpers use the functional calling convention. Unlike `fit`, they take
only a `Dataset`, not a list of experiments, have no `freeze=`, and need
`solver=` passed explicitly. From a `HybridModel`, pass its parts:

```python
dataset = hm.make_dataset(experiments, output_channel_names=model.output_channel_names)
parts = dict(
    simulate_fn=model.simulate_fn,
    state_to_output=model.state_to_output,
    solver=model.solver,
)
```

Both helpers take `objective=` as `fit` does.

## Seed ensembles

`train_seed_ensemble` scores `n_seeds` fresh initialisations with the
[tournament](/guide/training#the-tournament), then fully trains the best
`k_best`:

```python
ranked = hm.train_seed_ensemble(
    model.predictors, dataset, config, **parts, n_seeds=16, k_best=4, key=key
)
```

`config.tournament_steps` sets how many warm-up steps each seed gets before
it is scored. The default `0` scores the initialisation as it stands.

## Bootstrap ensembles

Each member trains on a resample of the experiments, drawn with
replacement:

```python
ranked = hm.train_bootstrap_ensemble(
    model.predictors, dataset, config, **parts, n_bootstraps=10, key=key
)
```

With `n_seeds > 1`, each resample's member is itself chosen by a seed
ensemble. `make_bootstrap_dataset(dataset, key=key)` gives a single
resample if you want to train on it yourself. A resample keeps IDs unique
by naming an experiment's second and later draws `<exp_id>#2`, `<exp_id>#3`
and so on.

## Averaging predictions

```python
mean = hm.ensemble_predictions([p for _, p in ranked], dataset, **parts)
```

The result has the shape of one `predict_dataset` call: a tuple with one
`[N, T, D]` array per bucket. `PredictionResult.from_buckets` keys it by
experiment so it can be scored:

```python
hm.print_metrics(hm.PredictionResult.from_buckets(mean, dataset))
```

For a smooth curve, call `predict_dense` per member and average those.

To use a member as a model again:
`model_k = eqx.tree_at(lambda m: m.predictors, model, ranked[k][1])`.

For members that differ only in their initial key, `eqx.filter_vmap` over a
function that builds the model gives one batched model; see
[Model interface](/guide/model-interface#the-model-is-an-equinox-module).
