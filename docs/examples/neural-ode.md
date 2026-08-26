# Neural ODEs and hybrid composition

Two scripts under `examples/neural_ode/`, sharing one data module. The first is a pure neural ODE on regularly sampled data. The second keeps the physics it knows, learns the two pieces it does not, and runs on data that is irregular per channel.

```bash
uv run python examples/neural_ode/train_neural_ode.py
uv run python examples/neural_ode/train_hybrid_ode.py
```

Both datasets are ports of the generators in the [Equinox](https://docs.kidger.site/equinox/examples/neural_ode/) and [diffrax](https://docs.kidger.site/diffrax/examples/latent_ode/) galleries, so the difference between reading those files and reading these is the framework, not the problem.

## Two data shapes

`make_dataset` groups experiments by the length of their union timestamp axis. That is the only thing it groups on, and it is what decides how many times training compiles.

**Rectangular.** Every experiment sampled on `linspace(0, 8, 40)`, both coordinates observed at every sample:

```
1 bucket(s)
  bucket 0: N=16  T= 40  D=2  mask density 1.00
```

One bucket, mask all `True`. This is a `[16, 40, 2]` array with a mask attached. There is no separate code path for it. Regular data is the degenerate case of bucketing, which is worth seeing once so the general case does not look like overhead.

**Irregular, thinned per channel.** Each experiment gets its own end time in $[6, 9]$, its own randomly drawn sample times, and each of the two channels is thinned independently:

```
3 bucket(s)
  bucket 0: N= 5  T= 15  D=2  mask density 0.53
  bucket 1: N=11  T= 19  D=2  mask density 0.53
  bucket 2: N= 8  T= 23  D=2  mask density 0.52
```

Three distinct union lengths, so three buckets and three compilations. Nothing is padded to a common length and nothing is dropped. The mask runs about half full because a timestamp that exists for `y1` usually does not exist for `y2`, which is the normal situation when two instruments sample at their own rates.

Both generators keep $t = 0$ in every channel, so `y0_fn` reads the initial state off the first observation of each channel. Dropping that assumption is the latent-ODE problem and needs an encoder.

## Where the network goes

The hybrid model has two networks on opposite sides of the solver. The ground truth is

$$
\frac{dy}{dt} = \begin{bmatrix} -k & \omega \\ -\omega & -k \end{bmatrix} y + C y^{3}, \qquad \omega = 1, \quad k = k(T)
$$

The model keeps the rotation and knows $\omega$. It learns $k$ from a temperature covariate and learns $C y^3$ from the trajectories.

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    rate_net, residual_net = predictors

    # Outside the solve. One call per experiment.
    k = rate_net(covariates).reshape(())
    rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]])

    def vector_field(t, y, args):
        # Inside the solve. One call per solver step, on the tape.
        return rotation @ y + residual_net(y)

    ...
```

The covariate does not change during a trajectory, so `rate_net` is hoisted above `diffeqsolve`. Its cost is then independent of how many steps the solver takes and it never appears on the adjoint tape. `residual_net` has to run inside, and that is what makes the adjoint choice matter:

```python
solver = SolverConfig(..., adjoint=diffrax.RecursiveCheckpointAdjoint())
```

The framework's default is `DirectAdjoint`, which stores the whole forward tape. With a network in the vector field that is usually the memory bottleneck. See [`ADJOINT_REGISTRY`](/api/solver) for the four shipped choices and what each one trades.

The two networks travel as a plain tuple and are unpacked at the top of `simulate_fn`. The framework never inspects the container ([ADR-0006](https://github.com/DanielePessina/jax-hybridmodels/blob/main/docs/adr/0006-predictors-as-pytree.md)); a dict or a NamedTuple works the same. `--mechanistic-only` drops the second entry and the tuple becomes length one, with no other change.

## Bounding a learned vector field

Both networks are wrapped in `BoundedPredictor`, so neither sees a bound and neither can produce a value outside one.

```python
BoundedPredictor(
    input_keys=("y1", "y2"),
    in_scaler=BoundScaler(bounds=((-2.0, 2.0), (-2.0, 2.0)), transform="sigmoid"),
    inner=MLPPredictor(in_size=2, out_size=2, width_size=32, depth=2, key=key),
    out_scaler=BoundScaler(bounds=((-3.0, 3.0), (-3.0, 3.0)), transform="softsign", warp="symlog"),
)
```

The output squash is `softsign`, not `sigmoid`. A vector field visits the edges of its derivative box during the early, badly fitted part of training, and sigmoid's gradient underflows to exactly `0.0` past a latent of about 15. Softsign decays polynomially instead, so a pinned component can still recover. The measured decay rates are in [`hybridmodels.transforms`](/api/transforms).

## Rates that span decades

`rate_net` maps temperature to $k$, and the true rates run from 0.0063 to 0.276. Its output box is `(1e-3, 1)` with `warp="log10"`:

```python
out_scaler=BoundScaler(bounds=((1e-3, 1.0),), transform="sigmoid", warp="log10")
```

Under a linear warp the box midpoint is 0.5, and every rate in the data sits in the bottom 3% of the box where the network has to produce large negative latents to reach anything. Under `log10` the midpoint is 0.032 and the data covers the middle of the box. The warp changes what "halfway between the bounds" means; it does not change which physical values are reachable.

The check that this worked is the fitted rate against the Arrhenius law it never saw:

| temperature (K) | true $k$ | fitted $k$ | ratio |
|---|---|---|---|
| 280 | 0.00629 | 0.00885 | 1.41 |
| 292 | 0.01516 | 0.01918 | 1.26 |
| 304 | 0.03412 | 0.03809 | 1.12 |
| 316 | 0.07221 | 0.07526 | 1.04 |
| 328 | 0.14463 | 0.14932 | 1.03 |
| 340 | 0.27583 | 0.29543 | 1.07 |

The network is fitted on trajectories alone and recovers 1.6 decades of temperature dependence. The worst point is the coldest, where the trajectory barely decays over the observation window and $k$ is therefore the least identifiable.

## Registering your own warp

The residual box straddles zero, so `log10` is unusable, and a linear box spends its resolution on large corrections that should never happen. The script registers a signed-logarithmic warp instead, without touching the package:

```python
from hybridmodels import Warp, register_warp

register_warp(
    "symlog",
    Warp(
        forward=lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x) / SYMLOG_EPS),
        inverse=lambda w: jnp.sign(w) * SYMLOG_EPS * jnp.expm1(jnp.abs(w)),
        requires_positive=False,
    ),
)
```

`forward(0) = 0`, so the box midpoint stays at zero and a freshly initialised residual network produces a correction near zero rather than at some arbitrary interior point. [`register_bound_transform`](/api/transforms) is the same idea on the other axis, for a squash with a different saturating profile.

A scaler stores the warp by name, so a custom warp has to be registered before a saved model that references it can be loaded.

## The saturation penalty

Bounds hold by construction here, so a violation cannot be represented and there is nothing to clip. What can go wrong is the other end: a network pinned against a bound, where the squash derivative has decayed and the gradient that would pull it back has gone. The penalty is charged on the latent rather than on the physical output, which is what keeps it working after the squash gradient has died. See [ADR-0007](https://github.com/DanielePessina/jax-hybridmodels/blob/main/docs/adr/0007-collocation-bound-penalty.md).

```python
config = OptaxTrainingConfig(
    ...,
    penalty_weight=(1e-3,),   # length 1 broadcasts across phases
    penalty_grid_points=7,
)
```

The script prints the end-of-run value:

```
  end-of-run saturation penalty: 3.055e-03
```

Zero means no predictor is pinned anywhere in its declared input box. Split by leaf, all of that 3e-3 belongs to the rate network and the residual network sits at exactly zero.

That split is the penalty doing its job. `bound_penalty` sweeps a collocation grid over each predictor's whole declared input box, not over the states a trajectory happened to visit, so it reports the rate network pushing against the top of its $k$ box out at the warm end of `TEMPERATURE_BOUNDS`, past any experiment in the data. That is a fact about the fitted model worth knowing before anyone runs it at a temperature you did not measure, and it is invisible to the training loss. The trade is the other direction: a trajectory-blind penalty cannot tell you whether a solve pushed an input out of range, which needs the penalty evaluated where the state actually goes.

Run with `--penalty-weight 0` to see the term switched off.

## Swapping the architecture

`--inner kan` replaces both MLPs with KANs. The only part of the script that changes is the one function that builds the inner network:

```python
def _inner(kind, *, in_size, out_size, width, key):
    if kind == "mlp":
        return MLPPredictor(in_size=in_size, out_size=out_size, width_size=width, depth=2, key=key)
    if kind == "kan":
        return KANPredictor(in_size=in_size, out_size=out_size, hidden_widths=(width,), key=key)
```

`BoundedPredictor` holds a `Predictor` rather than subclassing one, so everything downstream sees only the abstract type.

## Results

Default settings, one CPU, seeds fixed.

| model | data | $R^2$ (y1) | $R^2$ (y2) |
|---|---|---|---|
| pure neural ODE | rectangular, 1 bucket | 0.883 | 0.838 |
| mechanistic only, no residual | irregular, 3 buckets | 0.662 | 0.702 |
| hybrid, rate net plus residual | irregular, 3 buckets | **0.985** | **0.983** |
| the same with `--inner kan` | irregular, 3 buckets | 0.990 | 0.992 |

The middle row is the one worth reading twice. Dropping the residual does not just cost fit. The rate network absorbs the missing cubic term into $k$, and its recovered rate goes from 1.41 times the truth at 280 K to 32 times. An unmodelled term corrupts the mechanistic parameter you wanted to measure, which is the case for hybrid models rather than for either extreme.

The KAN row fits the trajectories slightly better and recovers the residual better (37% relative RMS against the MLP's 48%), and it recovers $k$ considerably worse: 6.4 times the truth at 280 K rather than 1.41. A more expressive residual absorbs part of the damping, and the mechanistic parameter pays for it. If the fitted parameter is what you came for, the residual wants to be the least expressive thing that closes the gap.

The residual network reaches an RMS error of 0.12 against a true residual RMS of 0.25 on a grid over the data range. It is only identifiable where trajectories went, and the grid includes corners none of them visited, so the grid figure overstates the error the trajectory fit sees.

## Which script to copy

Start from `train_hybrid_ode.py` if you have a mechanistic model with unknown parameters or an unknown term. Start from `train_neural_ode.py` if you have no mechanistic model at all and want the bound and bucketing machinery around a learned field.
