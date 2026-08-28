# `state_to_output` belongs to the model, not the Dataset

`state_to_output` — the pure callable mapping a full-state trajectory `[T, S]` to the observed output channels `[T, D]` — lives with the model setup, not the data. Prediction and training receive it as a parameter; the `Dataset` is pure data (experiments, channels, masks, the union timestamp axis, `y0`).

## Why this is non-obvious

The obvious home for `state_to_output` is the data container, and v1 shipped it there: `Dataset.state_to_output` stored the projection function alongside the observations, so prediction and training could read it off the dataset. That convenience is real — the projection is shared across scripts, and `_model.py` already re-uses it.

It is the wrong default here, for three reasons:

1. **The Dataset never sees states.** A dataset holds `[N, T, D]` observations — measured channels at measured times. The full state `[T, S]` exists only *inside* a solve, produced by `simulate_fn` from the model's predictors and solver. "Which states exist, and in what order" is a property of the *model*, not the data. Storing the projection on the data ties a model-owned function to a model-independent container.
2. **It breaks the no-`Model`-wrapper invariant.** ADR-0001 defines the model as the loose triple `(predictors, simulate_fn, solver_config)`. `state_to_output` is the fourth, model-shaped piece. Putting it on the Dataset smuggles a piece of the model into a place the ADRs explicitly keep model-free, and makes the Dataset not-pure-data by construction.
3. **It couples two halves that should vary independently.** "Train on the same physics with a different observation map" (e.g. change what's measured) currently means changing the data container, even though the physics is unchanged. When the projection is a parameter, that's a drop-in swap at the call site.

The fix is a mechanical move: `state_to_output` becomes a keyword-only parameter to `predict_bucket`/`predict_dataset` and to the training loops, exactly like `simulate_fn` and `solver` already are. The Dataset drops the field. Loss functions already receive `bp` (which carries `y_observed`/`yvar`/`mask`), so they stay signature-stable: `loss(pred_obs, bp) -> scalar` is untouched, and the *dataset plus the model with its attached `state_to_output`* is what a loss computation needs — which is precisely the "correct model with the correct projection attached" the plan requires.

## Why this is hard to reverse

It changes a public constructor field on `Dataset` and every call site of prediction and training, plus the docs' framing ("the model triple" becomes "the model triple plus a projection function"). Reversing it later — moving the projection back onto the data — would be a second breaking change for the same API surface. Getting it right before publication is much cheaper than shipping the coupled version and unwinding it under a versioning constraint.

## Considered alternatives

- Keep `state_to_output` on the `Dataset` (status quo). Rejected for the three reasons above: states depend on the model, the Dataset is not pure data, and the two halves should vary independently. The convenience is real but belongs in the examples (a shared builder that returns both the model triple and the projection), not in the container.
- A `HybridModel` wrapper class holding `(predictors, simulate_fn, solver_config, state_to_output)`. Rejected: that is ADR-0001's `Model` wrapper, which the no-wrapper decision exists to avoid. The four pieces travel as independent parameters.
- Attach `state_to_output` to `simulate_fn` (e.g. a closure that already projects). Rejected: it would blur the ADR-0005 contract — `simulate_fn` must return the full state `[T, S]` so that trajectory-dependent penalties (ADR-0007) and multi-output projections remain possible. The projection is deliberately separate.

## Consequences

- `Dataset` becomes pure data: `make_dataset` and the container carry no model-shaped callables.
- `predict_bucket`/`predict_dataset`/`train_with_optax`/`train_with_evosax` take `state_to_output` as a keyword-only parameter.
- The docs' "model" framing gains a fourth member: the model triple plus `state_to_output`.
- Examples' shared builders return both the model pieces and the projection, so the convenience is preserved without the coupling.
