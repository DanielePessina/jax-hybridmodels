# JAX/Equinox audit research findings

Date: 2026-08-29  
Scope: read-only review of `jax-hybridmodels`, the supplied `jax-equinox-numerics`
and `jax-project-engineering` skills, and the official JAX, Equinox, and Optax
documentation/source linked below.

This is a research sidecar, not an implementation report. No source files or
tests were modified.

## Executive summary

The package uses the right broad composition model: predictors are Equinox
PyTrees, the ODE simulator remains user-owned, bucket dispatch stays outside
compiled kernels, and the data losses use the JAX-safe “sanitize before
`where`” pattern. The main correctness risks are at boundaries where static
metadata, runtime domains, precision, and objective reduction meet the JAX
transform model.

## Findings

### F1 — Checkpoint loading does not authenticate static predictor semantics (P1)

`save_predictors` writes only `eqx.tree_serialise_leaves`, and
`load_predictors` intentionally accepts a caller-built template
([`serialise.py`](../../src/jaxhybridmodels/serialise.py#L66-L87)). Equinox’s
contract is that deserialisation receives a `like` PyTree with the same
structure and leaf types; non-leaf/static values are retained from that
template ([Equinox serialisation](https://docs.kidger.site/equinox/api/serialisation/)).

That is safe only if the template is independently guaranteed to have the same
static semantics. This package’s run metadata records a container structure and
module class names, but not static values such as `BoundScaler.bounds`,
`transform`, `warp`, `input_keys`, `z_knee`, or `MLPPredictor.activation_name`
([`serialise.py`](../../src/jaxhybridmodels/serialise.py#L142-L175)). A same-shape
template with different bounds or activation therefore loads successfully and
keeps the template’s semantics while receiving the saved arrays. A local
smoke check reproduced this with a saved `tanh`/`[0, 1]` predictor loaded into
a `relu`/`[-10, 10]` template.

The supplied [project-engineering skill](https://github.com/mancusolab/coding-skills/tree/main/plugins/coding-skills/skills/jax-project-engineering)
also requires persisted state to carry a version tag and validate it on load;
`save_run` writes `version`, but `load_run` does not validate it. It also
silently drops unknown training-config fields in
`_filter_dataclass_kwargs` ([`serialise.py`](../../src/jaxhybridmodels/serialise.py#L259-L267)).
Together these make a checkpoint appear loadable while its effective model or
configuration may have changed.

Audit implication: checkpoint acceptance must be tested for semantic static
metadata, not only array shapes and module classes. The required invariant is
“loaded predictor has the saved static contract,” not merely “the leaves fit.”

### F2 — Runtime input domains and scaler invariants are under-validated (P1)

`check_bounds` validates the declared box and positivity of the *declared*
lower edge for `log`/`log10`, but `BoundScaler.__init__` does not validate
`temperature`, `logit_eps`, or the shape of a per-component temperature
([`base.py`](../../src/jaxhybridmodels/predictors/base.py#L210-L236)). The runtime
input is then passed through the selected warp before the soft inverse
([`base.py`](../../src/jaxhybridmodels/predictors/base.py#L250-L281)). For a
positive-only warp, a state-derived input of zero or less is still possible
even when the declared bounds are valid. In the installed environment:

```text
BoundScaler(bounds=((1, 100),), warp="log10").to_latent([0.0])  -> [-inf]
BoundScaler(bounds=((1, 100),), warp="log10").to_latent([-1.0])  -> [nan]
BoundScaler(bounds=((0, 1),), temperature=0).from_latent([0.0])  -> [nan]
```

This matters more inside an ODE than at an ordinary feed-forward boundary:
one invalid trajectory can contaminate a vmapped bucket. Equinox recommends
using `__check_init__` for intrinsic shape/range/cross-field invariants and
rejecting invalid state before the first traced call
([Equinox fields and invariant checks](https://docs.kidger.site/equinox/api/module/advanced_fields/)).

The current soft-inverse design is correct for finite values that are outside
the normalized box, and the latent saturation penalty is the right place to
recover output-side gradients. It does not repair an undefined upstream warp.
The audit must therefore distinguish “outside a linear box” from “outside the
domain of a nonlinear warp.”

### F3 — Optax and Evosax do not optimize the same bucket-reduced objective (P1)

The Optax loop sums each bucket loss and divides by the number of buckets; it
also averages the bucket gradients
([`optax.py`](../../src/jaxhybridmodels/training/optax.py#L399-L424)). The Evosax
single-individual evaluator instead sums bucket losses directly
([`evosax.py`](../../src/jaxhybridmodels/training/evosax.py#L254-L284)). This is
not just a reporting difference: the Optax gradient scale changes with the
number of buckets, while Evosax’s fitness scale changes with that same count
and with the bucket contents.

Within each bucket, the shipped `masked_*` losses normalize differently from
the trajectory penalties. In particular, `trajectory_saturation_penalty`
sums over time, while `masked_mse`/`bal_mse` are means; the embedded trajectory
path also charges a terminal time-integral. Thus a longer trajectory, a larger
bucket, or a different number of buckets changes the effective regularization
weight unless that is explicitly intended
([`penalties.py`](../../src/jaxhybridmodels/penalties.py#L355-L383),
[`losses.py`](../../src/jaxhybridmodels/losses.py#L70-L148)).

Optax defines a gradient transformation as operating on candidate gradients;
the caller owns loss reduction and optimizer-state sequencing
([Optax getting started](https://optax.readthedocs.io/en/stable/getting_started.html),
[Optax transformations](https://optax.readthedocs.io/en/stable/api/transformations.html)).
The package therefore needs one explicit canonical weighting policy before
comparing Optax histories with Evosax fitness or treating a penalty weight as
portable across datasets.

### F4 — Precision is caller/process state, not a library-level invariant (P2)

The current environment has `jax_enable_x64=False`. JAX defaults to 32-bit
creation and can downcast requested 64-bit values when X64 is disabled
([JAX default dtypes](https://docs.jax.dev/en/latest/default_dtypes.html)).
The package resolves NumPy dtypes at data construction, which is good, but
`ChannelObs`, covariates, `y0`, `BoundScaler.temperature`, and solver
tolerances are still created through independent `jnp.asarray` paths
([`data.py`](../../src/jaxhybridmodels/data.py#L86-L105),
[`data.py`](../../src/jaxhybridmodels/data.py#L223-L255),
[`solver.py`](../../src/jaxhybridmodels/solver.py#L149-L174)). The examples that
need double precision enable it themselves; the core API does not establish or
check a run-wide dtype policy.

For hybrid ODE training this can alter solver error control, latent scaling,
small variances, and gradient/penalty magnitudes. The audit should exercise
the same model under X64 on/off and mixed input dtypes, and treat any change
beyond an explicitly accepted tolerance as a precision contract failure.

### F5 — Data-boundary validation leaves silent loss-of-observation paths (P2)

`ChannelObs` documents that a channel must not repeat a timestamp, but the
constructor does not enforce that. `_per_experiment_arrays` converts timestamps
to a set and then scatters them, so duplicate timestamps collapse to one union
location and the later value overwrites the earlier one
([`data.py`](../../src/jaxhybridmodels/data.py#L65-L67),
[`data.py`](../../src/jaxhybridmodels/data.py#L309-L349)). This is silent data
loss before the ODE or loss kernel sees the experiment.

Similarly, Gaussian MLE losses replace every masked variance with `1.0` and
then clamp every active variance below `1e-12`
([`losses.py`](../../src/jaxhybridmodels/losses.py#L70-L91)). That is useful for
finite traced arithmetic, but it means zero or negative active variances are
treated as tiny positive variances rather than rejected as invalid measurement
metadata. The boundary contract should make this distinction observable.

The masking implementation itself is sound: it sanitizes potentially invalid
values before applying `jnp.where`. JAX explicitly warns that NaNs in either
branch of `where` can propagate through reverse-mode gradients
([JAX `where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html),
[JAX FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where)).

### F6 — Generic tournament reinitialization conflates parameters with float state (P2)

The fallback branch of `reinitialize_with_key` replaces every inexact-array
leaf with a standard-normal sample
([`base.py`](../../src/jaxhybridmodels/predictors/base.py#L452-L490)). Equinox’s
filtered gradients likewise treat all floating-point array leaves in the first
argument as differentiable unless filtered
([Equinox filtered transformations](https://docs.kidger.site/equinox/api/transformations/)).
That is consistent mechanically, but not semantically: a custom predictor may
contain floating-point buffers, normalization statistics, or fixed physical
constants that are not initialization parameters. Unless it implements
`initialized_with_key`, a tournament restart mutates those values too.

The same distinction applies to custom solver/adjoint objects placed in
`SolverConfig`: all config fields are `static=True`. Equinox warns that static
fields do not participate in transforms and that making JAX arrays static is
usually a bug ([Equinox static fields](https://docs.kidger.site/equinox/api/module/advanced_fields/)).
The built-in solver instances are configuration-like, but extension tests need
to reject or deliberately handle array-bearing custom solver state.

### F7 — User callbacks are transformed without a boundary contract check (P2)

`simulate_fn`, `state_to_output`, and custom loss/penalty callbacks are invoked
inside `vmap`, `filter_jit`, and (for training) reverse-mode differentiation
([`kernels.py`](../../src/jaxhybridmodels/training/kernels.py#L78-L116),
[`kernels.py`](../../src/jaxhybridmodels/training/kernels.py#L148-L175)). JAX
requires jitted functions to be pure and accepts array/scalar or nested
standard-container arguments; `jax.grad` requires a scalar output
([JAX `jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html),
[JAX `grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)).

The package relies on users honoring this, but does not preflight the callback
contract. Python value-dependent branches, host side effects, dynamic-shape
indexing, or a penalty returning a non-scalar will fail only after tracing (or
produce transform-specific behavior). The same applies to runtime errors:
Equinox documents that `error_if` under `vmap` raises if *any* batch element
matches, so one bad experiment can fail a whole bucket
([Equinox runtime errors](https://docs.kidger.site/equinox/api/errors/)). This
is compatible with the shared-tournament design, but it must be treated as a
whole-bucket failure boundary, not per-experiment recovery.

## Verified-safe patterns

- `simulate_bucket` uses explicit `in_axes=(0, 0, 0)` and leaves the shared
  predictors and solver unmapped, matching the JAX `vmap` contract
  ([JAX `vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)).
- The training kernels partition predictors before filtered differentiation,
  and Optax receives only the filtered parameter tree. This matches Equinox’s
  PyTree/filtered-gradient model ([Equinox transformations](https://docs.kidger.site/equinox/api/transformations/)).
- Named root-key folds are compatible with JAX’s explicit, pure PRNG model;
  `fold_in` is deterministic, and JAX recommends deriving per-step keys from a
  common parent rather than building a long dependent chain
  ([JAX `fold_in`](https://docs.jax.dev/en/latest/_autosummary/jax.random.fold_in.html),
  [JAX pseudorandom numbers](https://docs.jax.dev/en/latest/101/random.html)).
- The latent-side saturation penalty avoids relying on a saturated physical
  output derivative. The repository’s own bound-penalty tests verify gradient
  reachability to inner predictor weights; this is also consistent with the
  JAX/Equinox requirement to validate AD behavior, not only primal outputs.
- Python bucket loops and host-side validation stay outside the compiled
  kernels. This matches JAX’s rule that static values affect compilation and
  that changing static arguments can trigger recompilation
  ([JAX `jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)).

## Inputs reviewed

- [Supplied `jax-equinox-numerics` skill](https://github.com/mancusolab/coding-skills/tree/main/plugins/coding-skills/skills/jax-equinox-numerics)
- [Supplied `jax-project-engineering` skill](https://github.com/mancusolab/coding-skills/tree/main/plugins/coding-skills/skills/jax-project-engineering)
- [Official Equinox API: transformations](https://docs.kidger.site/equinox/api/transformations/)
- [Official Equinox API: fields and invariant checks](https://docs.kidger.site/equinox/api/module/advanced_fields/)
- [Official Equinox API: serialization](https://docs.kidger.site/equinox/api/serialisation/)
- [Official JAX API: `jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)
- [Official JAX API: `vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
- [Official JAX docs: PyTrees](https://docs.jax.dev/en/latest/pytrees.html)
- [Official JAX docs: pseudorandom numbers](https://docs.jax.dev/en/latest/101/random.html)
- [Official JAX docs: default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html)
- [Official JAX API: `where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html)
- [Official Optax getting started](https://optax.readthedocs.io/en/stable/getting_started.html)
- [Official Optax transformations](https://optax.readthedocs.io/en/stable/api/transformations.html)
