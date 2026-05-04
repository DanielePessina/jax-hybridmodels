# API Reference

The public surface is split across ten pages, grouped by concern. Every symbol below is also re-exported at the top level — `from hybridmodels import MLPPredictor` works exactly like `from hybridmodels.predictors import MLPPredictor`.

## [Data: Experiments, Channels, Datasets](/api/data)

[`ChannelObs`](/api/data#channelobs), [`Experiment`](/api/data#experiment), [`make_experiment`](/api/data#make_experiment), [`BucketPayload`](/api/data#bucketpayload), [`Dataset`](/api/data#dataset), [`make_dataset`](/api/data#make_dataset), [`split_dataset`](/api/data#split_dataset)

## [Predictors: Trainable Components](/api/predictors)

[`Predictor`](/api/predictors#predictor), [`BoundScaler`](/api/predictors#boundscaler), [`BoundedPredictor`](/api/predictors#boundedpredictor), [`MLPPredictor`](/api/predictors#mlppredictor), [`KANPredictor`](/api/predictors#kanpredictor), [`reinitialize_with_key`](/api/predictors#reinitialize_with_key), [`reinitialize_pytree_with_key`](/api/predictors#reinitialize_pytree_with_key)

## [Solver: ODE Integration](/api/solver)

[`SolverConfig`](/api/solver#solverconfig), [`SOLVER_REGISTRY`](/api/solver#solver_registry), [`register_solver`](/api/solver#register_solver)

## [Training: Optax & Evosax Loops](/api/training)

[`OptaxTrainingConfig`](/api/training#optaxtrainingconfig), [`train_with_optax`](/api/training#train_with_optax), [`EvosaxTrainingConfig`](/api/training#evosaxtrainingconfig), [`train_with_evosax`](/api/training#train_with_evosax)

## [Losses: Masked & Balanced Objectives](/api/losses)

[`masked_mse`](/api/losses#masked_mse), [`masked_mle`](/api/losses#masked_mle), [`bal_mse`](/api/losses#bal_mse), [`bal_mle`](/api/losses#bal_mle), [`LOSS_REGISTRY`](/api/losses#loss_registry)

## [Trainable Masks: Freezing Leaves](/api/trainable)

[`default_trainable`](/api/trainable#default_trainable), [`trainable_mask`](/api/trainable#trainable_mask), [`freeze_paths`](/api/trainable#freeze_paths), [`freeze_modules_of_type`](/api/trainable#freeze_modules_of_type), [`freeze_where`](/api/trainable#freeze_where)

## [Prediction: Forward Simulation](/api/prediction)

[`predict_bucket`](/api/prediction#predict_bucket), [`predict_dataset`](/api/prediction#predict_dataset)

## [Serialise: Save & Load Runs](/api/serialise)

[`save_predictors`](/api/serialise#save_predictors), [`load_predictors`](/api/serialise#load_predictors), [`save_run`](/api/serialise#save_run), [`load_run`](/api/serialise#load_run)

## [UI: Training Dashboards](/api/ui)

[`TrainingUI`](/api/ui#trainingui), [`EvosaxUI`](/api/ui#evosaxui), [`SilentUI`](/api/ui#silentui), [`RichTrainingUI`](/api/ui#richtrainingui), [`RichEvosaxUI`](/api/ui#richevosaxui)

## [RNG: Named-Fold Keys](/api/rng)

[`fold`](/api/rng#fold)
