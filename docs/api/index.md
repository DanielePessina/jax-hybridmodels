# API Reference

The public surface is split across the pages below, grouped by concern. Every symbol below is also re-exported at the top level — `from hybridmodels import MLPPredictor` works exactly like `from hybridmodels.predictors import MLPPredictor`.

## [Data: Experiments, Channels, Datasets](/api/data)

[`ChannelObs`](/api/data#channelobs), [`Experiment`](/api/data#experiment), [`make_experiment`](/api/data#make_experiment), [`BucketPayload`](/api/data#bucketpayload), [`Dataset`](/api/data#dataset), [`make_dataset`](/api/data#make_dataset), [`make_bootstrap_dataset`](/api/data#make_bootstrap_dataset), [`split_dataset`](/api/data#split_dataset), [`describe_buckets`](/api/data#describe_buckets)

## [Predictors: Trainable Components](/api/predictors)

[`Predictor`](/api/predictors#predictor), [`BoundScaler`](/api/predictors#boundscaler), [`BoundedPredictor`](/api/predictors#boundedpredictor), [`MLPPredictor`](/api/predictors#mlppredictor), [`KANPredictor`](/api/predictors#kanpredictor), [`NeuralNPolynomial`](/api/predictors#neuralnpolynomial), [`reinitialize_with_key`](/api/predictors#reinitialize_with_key), [`reinitialize_pytree_with_key`](/api/predictors#reinitialize_pytree_with_key)

## [Penalties: Gradient-Safe Bound Handling](/api/penalties)

[`attach_penalty_state`](/api/penalties#attach_penalty_state), [`bound_penalty`](/api/penalties#bound_penalty), [`box_violation`](/api/penalties#box_violation), [`clip_ste`](/api/penalties#clip_ste), [`collocation_grids`](/api/penalties#collocation_grids), [`penalty_integral`](/api/penalties#penalty_integral), [`penalty_vector_field`](/api/penalties#penalty_vector_field), [`soft_inverse`](/api/penalties#soft_inverse), [`soft_logit`](/api/penalties#soft_logit), [`softclip`](/api/penalties#softclip), [`strip_penalty_state`](/api/penalties#strip_penalty_state), [`trajectory_saturation_penalty`](/api/penalties#trajectory_saturation_penalty)

## [Transforms: Squash Shapes and Axis Warps](/api/transforms)

[`BoundTransform`](/api/transforms#boundtransform), [`BOUND_TRANSFORMS`](/api/transforms#bound_transforms), [`register_bound_transform`](/api/transforms#register_bound_transform), [`Warp`](/api/transforms#warp), [`WARPS`](/api/transforms#warps), [`register_warp`](/api/transforms#register_warp)

## [Solver: ODE Integration](/api/solver)

[`SolverConfig`](/api/solver#solverconfig), [`SOLVER_REGISTRY`](/api/solver#solver_registry), [`register_solver`](/api/solver#register_solver), [`ADJOINT_REGISTRY`](/api/solver#adjoint_registry), [`register_adjoint`](/api/solver#register_adjoint)

## [Training: Optax & Evosax Loops](/api/training)

[`OptaxTrainingConfig`](/api/training#optaxtrainingconfig), [`train_with_optax`](/api/training#train_with_optax), [`train_seed_ensemble`](/api/training#train_seed_ensemble), [`train_bootstrap_ensemble`](/api/training#train_bootstrap_ensemble), [`EvosaxTrainingConfig`](/api/training#evosaxtrainingconfig), [`train_with_evosax`](/api/training#train_with_evosax), [`register_algorithm`](/api/training#register_algorithm)

## [Training Kernels: Build Your Own Loop](/api/kernels)

[`apply_length_mask`](/api/kernels#apply_length_mask), [`predict_bucket_obs`](/api/kernels#predict_bucket_obs), [`build_bucket_step`](/api/kernels#build_bucket_step), [`build_score_bucket`](/api/kernels#build_score_bucket), [`build_penalty_step`](/api/kernels#build_penalty_step), [`build_apply_update`](/api/kernels#build_apply_update)

## [Losses: Masked & Balanced Objectives](/api/losses)

[`masked_mse`](/api/losses#masked_mse), [`masked_mle`](/api/losses#masked_mle), [`bal_mse`](/api/losses#bal_mse), [`bal_mle`](/api/losses#bal_mle), [`LOSS_REGISTRY`](/api/losses#loss_registry), [`resolve_loss_fn`](/api/losses#resolve_loss_fn)

## [Trainable Masks: Freezing Leaves](/api/trainable)

[`default_trainable`](/api/trainable#default_trainable), [`trainable_mask`](/api/trainable#trainable_mask), [`freeze_paths`](/api/trainable#freeze_paths), [`freeze_modules_of_type`](/api/trainable#freeze_modules_of_type), [`freeze_where`](/api/trainable#freeze_where), [`frozen_default_mask`](/api/trainable#frozen_default_mask), [`count_trainable_params`](/api/trainable#count_trainable_params)

## [Prediction: Forward Simulation](/api/prediction)

[`predict_bucket`](/api/prediction#predict_bucket), [`predict_dataset`](/api/prediction#predict_dataset), [`predict_dense`](/api/prediction#predict_dense), [`ensemble_predictions`](/api/prediction#ensemble_predictions), [`evaluate_predictor`](/api/prediction#evaluate_predictor)

## [Metrics: Per-Channel Evaluation](/api/metrics)

[`ChannelMetrics`](/api/metrics#channelmetrics), [`compute_metrics`](/api/metrics#compute_metrics), [`print_metrics`](/api/metrics#print_metrics)

## [Profiles: Time-Varying Inputs](/api/profiles)

[`constant_profile`](/api/profiles#constant_profile), [`step_profile`](/api/profiles#step_profile), [`ramp_profile`](/api/profiles#ramp_profile), [`piecewise_linear_profile`](/api/profiles#piecewise_linear_profile)

## [Schedules: Epoch-Scaled Annealing](/api/schedules)

[`annealing_schedule`](/api/schedules#annealing_schedule)

## [Serialise: Save & Load Runs](/api/serialise)

[`save_predictors`](/api/serialise#save_predictors), [`load_predictors`](/api/serialise#load_predictors), [`save_run`](/api/serialise#save_run), [`load_run`](/api/serialise#load_run)

## [UI: Training Dashboards](/api/ui)

[`TrainingUI`](/api/ui#trainingui), [`EvosaxUI`](/api/ui#evosaxui), [`SilentUI`](/api/ui#silentui), [`RichTrainingUI`](/api/ui#richtrainingui), [`RichEvosaxUI`](/api/ui#richevosaxui)

## [RNG: Named-Fold Keys](/api/rng)

[`fold`](/api/rng#fold)
