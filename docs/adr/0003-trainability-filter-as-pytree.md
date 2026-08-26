# Trainability filter is a boolean PyTree, not a per-class registry

Trainable/frozen status is expressed as a boolean PyTree mask matching the predictor's tree structure. The same mask is consumed by both Optax (`eqx.filter_value_and_grad(..., filter_spec=mask)`) and Evosax (`eqx.partition(predictor, mask)` to derive the flat parameter vector). Default predicate: `eqx.is_inexact_array`. Customisation is via composable free-function freezers (`freeze_paths`, `freeze_modules_of_type`, `freeze_where`) that take a mask and return a new mask.

## Why this is non-obvious

The source package had a 100+ line `_build_filter_spec` switch in `regressor_registry.py` that dispatched on model type to decide what was trainable. A future reader may try to add a `Predictor.trainable_filter` method on the abstract base. Don't. It violates Equinox's no-method-overriding pattern, scatters the freezing logic across N classes, and makes "freeze all `BoundScaler` instances regardless of where they appear" awkward to express. Free-function freezers compose cleanly, extend without touching predictor code, and produce an inspectable artifact (the mask itself).

## Considered alternatives

- Per-class `trainable_filter` method on `Predictor`. Rejected for the reasons above.
- A `set_trainable(...)` mutator on the predictor. Rejected because it feels mutable on a frozen pytree and breaks Equinox's "all init in `__init__`" rule.
- Predicate-only API (no concrete mask materialised). Rejected because a materialised mask is inspectable and shape-checkable; predicates derive a mask anyway under the hood.
