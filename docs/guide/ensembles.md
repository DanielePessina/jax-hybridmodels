# Ensembles

An ensemble is several trained models whose predictions you average. Seed
ensembles vary the initialisation; bootstrap ensembles vary the data.
`model.fit_ensemble` trains either kind and returns a list of `FitResult`s,
best first. `hm.predict_ensemble` averages their predictions.

`fit_ensemble` takes `fit`'s arguments (`data`, `config`, `key`,
`objective`, `freeze`, `trainable`, `ui`) plus `n_members`, `n_seeds` and
`bootstrap`. It trains with Optax only. Every member starts from a fresh
initialisation of the predictors drawn from `key`.

## Seed ensembles

Score `n_seeds` initialisations with the
[tournament](/guide/training#the-tournament) on all of `data`, then fully
train the best `n_members`:

```python
results = model.fit_ensemble(data, config, key=key, n_members=4, n_seeds=16)
```

`config.tournament_steps` sets how many warm-up steps each seed gets before
it is scored. The default `0` scores the initialisation as it stands.
`n_seeds` defaults to `n_members`, which trains every seed.
`config.tournament_attempts` is not used here; `n_seeds` replaces it.

## Bootstrap ensembles

With `bootstrap=True`, each member trains on its own resample of the
experiments, drawn with replacement:

```python
results = model.fit_ensemble(data, config, key=key, n_members=10, bootstrap=True)
```

`n_seeds` defaults to 1 here: one fresh seed per resample. With
`n_seeds > 1`, each resample's member is the best of that many seeds,
scored on the resample. `make_bootstrap_dataset(dataset, key=key)` gives a
single resample if you want to train on it yourself. A resample keeps IDs
unique by naming an experiment's second and later draws `<exp_id>#2`,
`<exp_id>#3` and so on.

## Ranking and failures

The list is sorted by `FitResult.final_loss`, the training loss at each
member's returned parameters. A seed whose warm-up fails is dropped, so
fewer than `n_members` may come back. If every seed fails, `fit_ensemble`
raises `RuntimeError`.

## Averaging predictions

```python
pred = hm.predict_ensemble(results, data)
hm.print_metrics(pred)
```

`predict_ensemble` takes the `FitResult` list from `fit_ensemble` as is,
or a list of `HybridModel`s. `pred` is a `PredictionResult` whose outputs
are the member mean, so
`compute_metrics` scores the mean. Each experiment's `std` holds the
per-channel standard deviation across members (`ddof=0`), and
`pred.members` holds each model's own result. `to_dataframe()` adds a
`std` column. `ts=` and `n_points=` work as in
[`predict`](/guide/model-interface#predicting), so
`hm.predict_ensemble(results, data, n_points=200)` gives a smooth mean
curve with its band.

The members must predict the same `output_channel_names`; usually they
are all copies of one model.

For members that differ only in their initial key, `eqx.filter_vmap` over a
function that builds the model gives one batched model; see
[Model interface](/guide/model-interface#the-model-is-an-equinox-module).
