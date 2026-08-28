# Ensembles of hybrid models

Sometimes one fitted model is not enough. You want several, and you want
their predictions averaged — to smooth over unlucky initialisations, or
to quantify how much the answer moves when the training data moves.

The library supports two ways to build an ensemble, both returning the
same thing: a list of `(final_loss, trained_predictors)` pairs, ranked
ascending by loss. Feed the predictors into
`ensemble_predictions` to average their per-bucket forecasts.

## Seed ensembles: same data, different starts

A single `train_with_optax` run starts from one initialisation. The
**tournament** already ranks several fresh initialisations cheaply (a few
warm-up steps each, scored forward-only). `train_seed_ensemble` reuses
that ranking to decide which seeds deserve a *full* training run — so you
get the benefit of many restarts without fully training every one.

```python
ranked = hm.train_seed_ensemble(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, n_seeds=16, k_best=4, key=key,
)
# ranked: [(loss, predictors), ...] best-first; 4 fully-trained members
```

`config.tournament_steps` controls how much warm-up each seed gets before
ranking (default `0` ranks by the initialisation score alone).

## Bootstrap ensembles: resampled data, one model per sample

Bagging means fitting the same model on different **bootstrap resamples**
of the experiments — draws with replacement, so each member sees a
slightly different dataset. `make_bootstrap_dataset` produces one
resample; `train_bootstrap_ensemble` runs the whole recipe:

```python
# The data half, if you want it explicitly:
boot = hm.make_bootstrap_dataset(dataset, key=key, n_experiments=20)

# Or train the whole ensemble in one call:
ranked = hm.train_bootstrap_ensemble(
    predictors, dataset, config,
    simulate_fn=simulate_fn, state_to_output=state_to_output,
    solver=solver, n_bootstraps=10, n_seeds=1, k_best=None, key=key,
)
```

`n_seeds > 1` combines the two: each bootstrap sample gets its own
seed-selected member, so every member sees different data *and* is a good
seed. Irregular per-channel timestamps are handled automatically —
resampling re-buckets, and duplicated experiments just add rows to a
bucket.

## Averaging predictions

`ensemble_predictions` runs every member forward and averages the
per-bucket predictions, so the result is the same shape as a single
`predict_dataset` call:

```python
mean_pred = hm.ensemble_predictions(
    tuple(p for _, p in ranked), dataset,
    simulate_fn=simulate_fn, state_to_output=state_to_output, solver=solver,
)
```

For dense trajectory plots, evaluate on a fine grid first with
`predict_dense`, then average.