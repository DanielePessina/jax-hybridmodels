# Training Kernels: Build Your Own Loop

The compiled pieces the stock trainers are assembled from. `build_bucket_step` is the jitted per-bucket `(loss, grads)` kernel (one trace per bucket shape); `build_score_bucket` is a forward-only scorer; `build_penalty_step` charges a regulariser once per step outside the bucket loop; `build_apply_update` is the single optimiser update. Write your own loop by composing these over `dataset.bucket_payloads` in Python.

## Quick links

- [`apply_length_mask`](#apply_length_mask)
- [`predict_bucket_obs`](#predict_bucket_obs)
- [`build_bucket_step`](#build_bucket_step)
- [`build_score_bucket`](#build_score_bucket)
- [`build_penalty_step`](#build_penalty_step)
- [`build_apply_update`](#build_apply_update)

---

<a id="apply_length_mask"></a>

### `apply_length_mask()`

<small>`from hybridmodels.training.kernels import apply_length_mask` &nbsp;·&nbsp; also re-exported as `hybridmodels.apply_length_mask`</small>

```python
apply_length_mask(bp: 'BucketPayload', length_mask_fraction: 'Array') -> 'BucketPayload'
```

Narrow the bucket's mask to the first ``fraction`` of its timestamps.

This masks the **loss**, never the integration: the solver still runs the
full trajectory, and only the leading prefix of the observation times is
scored. That is what keeps a long-horizon divergence from drowning the
gradient early in a run.

``length_mask_fraction`` stays traced rather than becoming a Python
branch, so a phase that changes the fraction costs no recompile.

The cutoff is clamped at 1. A fraction small enough to floor to zero
would otherwise give an all-false mask, and every loss here divides by a
count clamped at 1, so the step would silently score nothing.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/kernels.py#L56)</small>

---

<a id="predict_bucket_obs"></a>

### `predict_bucket_obs()`

<small>`from hybridmodels.training.kernels import predict_bucket_obs` &nbsp;·&nbsp; also re-exported as `hybridmodels.predict_bucket_obs`</small>

```python
predict_bucket_obs(
    predictors: 'Any',
    bp: 'BucketPayload',
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
) -> Array
```

Simulate every experiment in the bucket and project to ``[N, T, D]``.

The uncompiled shared core behind both `predict_bucket
<hybridmodels.prediction.predict_bucket>` and the training kernels.
Each caller wraps it in its own ``eqx.filter_jit``, which is what keeps
the training and prediction jit caches separate (R-J1): this body is
traced into whichever kernel calls it.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/kernels.py#L102)</small>

---

<a id="build_bucket_step"></a>

### `build_bucket_step()`

<small>`from hybridmodels.training.kernels import build_bucket_step` &nbsp;·&nbsp; also re-exported as `hybridmodels.build_bucket_step`</small>

```python
build_bucket_step(
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    loss_fn: 'Callable[[Array, BucketPayload], Array]',
    trainable: 'Any',
    trajectory_penalty_fn: 'Callable[[Array, BucketPayload], Array] | None' = None,
    trajectory_penalty_weight: 'float' = 0.0,
) -> Callable[[Any, BucketPayload, Array], tuple[Array, Any]]
```

Return a jitted ``bucket_step(predictors, bp, fraction) -> (loss, grads)``.

One trace per bucket shape (R-T5). Takes no ``opt_state``: the optimiser
update lives in a separate jitted ``apply_update``, and the **bound**
penalty is charged once per step by
[`build_penalty_step`](/api/kernels#build_penalty_step), outside the bucket loop — this kernel
charges the data loss (plus any configured trajectory penalty, below).

``trajectory_penalty_fn`` is the trajectory-aware counterpart, and the
exception to that: it reads the **full state** ``[N, T, S]`` (penalty
accumulators included) and the bucket payload, and returns a scalar
added to the data loss inside the same forward pass. Because it reads
per-bucket trajectories, it is charged **per bucket** — every bucket
in a step contributes its own trajectory penalty — unlike the bound
penalty's once-per-step charge. When it is ``None`` (the default) this
kernel is byte-for-byte what it was before — no extra simulate, no
behaviour change.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/kernels.py#L122)</small>

---

<a id="build_score_bucket"></a>

### `build_score_bucket()`

<small>`from hybridmodels.training.kernels import build_score_bucket` &nbsp;·&nbsp; also re-exported as `hybridmodels.build_score_bucket`</small>

```python
build_score_bucket(
    simulate_fn: 'Callable[..., Array]',
    state_to_output: 'Callable[[Array], Array]',
    solver: 'SolverConfig',
    loss_fn: 'Callable[[Array, BucketPayload], Array]',
    trajectory_penalty_fn: 'Callable[[Array, BucketPayload], Array] | None' = None,
    trajectory_penalty_weight: 'float' = 0.0,
) -> Callable[[Any, BucketPayload, Array], Array]
```

Return a forward-only ``score_bucket(predictors, bp, fraction) -> loss``.

Scoring through ``bucket_step`` would run a full ``value_and_grad`` and
discard the gradients, roughly tripling the cost of a scoring sweep.
This costs one extra compile per bucket shape and pays for itself above
two attempts. The returned score includes the data loss and, when
configured, the trajectory penalty; the bound penalty remains
outside this per-bucket scorer.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/kernels.py#L184)</small>

---

<a id="build_penalty_step"></a>

### `build_penalty_step()`

<small>`from hybridmodels.training.kernels import build_penalty_step` &nbsp;·&nbsp; also re-exported as `hybridmodels.build_penalty_step`</small>

```python
build_penalty_step(
    penalty_fn: 'Callable[[Any, tuple[Array, ...]], Array]',
    trainable: 'Any',
) -> Callable[[Any, Array, tuple[Array, ...]], tuple[Array, Any]]
```

Return ``penalty_step(predictors, weight, points=()) -> (penalty, weighted_grads)``.

Evaluated once per training step, outside the bucket loop: the penalty
reads only the predictors tree and its point arrays, so computing it
inside ``bucket_step`` would repeat one identical evaluation per
bucket.

``points`` are the per-leaf point arrays the penalty is evaluated at —
the output of [`select_penalty_points`](/api/penalties#select_penalty_points) —
passed as traced arrays, so a shape change (a phase boundary) retraces
this small kernel and nothing else. The default ``()`` suits a custom
``penalty_fn`` that ignores points.

``penalty_fn`` is the regulariser, required here and defaulted to
[`bound_penalty`](/api/penalties#bound_penalty) by the stock trainers.
Passing a different callable (weight decay on inner weights, a
monotonicity term, ...) is how a custom regulariser composes with the
loop. It must take ``(predictors, points)`` and return a scalar; a
custom term that does not need points just ignores them.

Returns the gradient of ``weight * penalty``, to add straight onto the
averaged data gradient. ``penalty`` comes back unweighted, since that
is what gets reported.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/kernels.py#L220)</small>

---

<a id="build_apply_update"></a>

### `build_apply_update()`

<small>`from hybridmodels.training.kernels import build_apply_update` &nbsp;·&nbsp; also re-exported as `hybridmodels.build_apply_update`</small>

```python
build_apply_update(
    optimizer: 'optax.GradientTransformation',
    trainable: 'Any',
) -> Callable[[Any, Any, Any], tuple[Any, Any]]
```

Return a jitted ``apply_update(predictors, grads, opt_state)``.

The single optimiser update per training step. Build once per
optimiser; reuse its returned state across steps, rebuilding only at a
reset/phase boundary.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/training/kernels.py#L269)</small>
