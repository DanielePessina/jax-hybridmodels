# Uncertainty

The package does no sampling itself. `hm.inference.make_problem` turns a model
and its data into a log-density, a pure JAX function of a pytree position, and a
sampler such as [BlackJAX](https://blackjax-devs.github.io/blackjax/) takes it
from there. Install both with `pip install jax-hybridmodels[bayes]`.

The same problem also answers questions that need no sampler: the maximum
likelihood, standard errors from the Fisher information, which parameters the data
cannot tell apart, and a Laplace approximation to the posterior. See
[Without sampling](#without-sampling).

For a first example where the uncertainty is the answer, read
[shelf life](/examples/shelf-life): a drug's label claim set from an accelerated
stability study, where the best estimate would overstate it.

## What every interval assumes

- The model is right. An interval measures the noise, not a wrong law or a
  missing term.
- The noise is independent and Gaussian, with the variances in the data or the
  ones inferred. `log_likelihood=` replaces that, see [Noise](#noise).
- Wald intervals, `derived`, `FisherResult.sample` and `laplace` assume the
  estimate is close to Gaussian. The profile assumes the chi-square limit of the
  likelihood ratio, which holds better with few data.
- A posterior or an estimate conditions on everything held fixed: solver settings,
  `y0`, other layers of a network, and any value pinned at its bound.
- An extrapolation is only as good as the law where it is made. A shelf life at
  25 °C from data at 40 to 70 °C trusts the Arrhenius form below the data.

## A first posterior

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#problem

`log_prior` is written on physical values. `noise_sd="data"` trusts the variances in
your data. Neither has a default, because a silent choice in either changes the
answer. `flat_prior` is available for a parameter with finite bounds, but a flat
prior is not neutral under a warp.

The sampler then runs as BlackJAX documents it:

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#sample

Start from a fitted model, because `problem.position` holds its current values.
[Making it fast](#making-it-fast) explains the threads and `preconditioned()`.

`problem.log_density` differs from the training loss in three ways:

- it is the likelihood itself, where the trainers minimise `Objective.loss`; the
  two agree only for `loss="mle"` with the data's variances, since `"mse"` weighs
  every measurement the same whatever its variance. Bucketing changes neither,
  so `max_buckets` is safe here;
- it adds the change-of-variables term for free `Parameter`s, so the prior you
  wrote is the prior you get, although the sampler moves in latent space;
- a failed solve, or a non-positive input to a log-warped scaler, gives `-inf` with a
  zero gradient, which a sampler counts as a rejected proposal, instead of an
  exception that ends the chain.

A `simulate_fn` that calls `diffrax.diffeqsolve` itself must pass
`throw=solver.throw`, or a failed solve still raises.

## What a posterior conditions on

A posterior is only over what is free. Everything else is treated as exactly
known, and its width does not account for it. `problem.conditional_on` lists the
trainable leaves left fixed, and `make_problem` warns when there are any.

| Model and method | What it gives | What it conditions on |
| --- | --- | --- |
| `Parameter`s only, default `estimate=` | Full posterior over the model's fitted values | Solver settings, `y0` |
| `Parameter`s and a network, default `estimate=` | `p(theta \| weights, data)` | The fitted weights, which can make it too narrow and off centre |
| `estimate=` naming a last layer | Posterior over that layer | Every other layer at its fitted value, so usually narrower than the network's full uncertainty, though not a bound on it |
| `fit_ensemble` | Spread across independent fits | Nothing is sampled: members are not posterior draws |

A model with only networks has no `Parameter`, so the default raises. Name a layer
with `estimate=`, or use [ensembles](/guide/ensembles):

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#last-layer

Network weights have no physical meaning, and they have no change-of-variables
term, so write their prior in `log_prior`. `flat_prior` is refused for them
because it is improper.

### Checking a mixed model

A network can absorb what the physics should explain, and a conditional posterior
cannot show this. An ensemble can. Fit several members, run a conditional posterior
on a few of them, and compare the two sources of spread:

$$
\operatorname{Var}(\theta) \approx
\underbrace{\mathbb{E}_w[\operatorname{Var}(\theta \mid w)]}_{\text{within a member}} +
\underbrace{\operatorname{Var}_w(\mathbb{E}[\theta \mid w])}_{\text{between members}}
$$

If the between-member term is not small next to the within-member one, the
conditional posterior is misleading. Members are not draws from a posterior over
the weights, so read this as a diagnostic and not as an estimate.

### Initial states

Some models read `y0` from the first measurement with `y0_fn`. A posterior treats
that noisy value as exact, which matters most when the early dynamics are fast.

## Noise

`noise_sd="data"` uses each observation's `variance`, and warns for a channel left at
the default of 1.0, because that sets the posterior's width. To infer a standard
deviation per channel, name a `Parameter` for each:

```python
problem = hm.inference.make_problem(
    model, data,
    log_prior=log_prior,  # receives noise["conc"], the physical value
    noise_sd={"conc": hm.Parameter(0.05, bounds=(1e-4, 1.0), warp="log10")},
)
```

With few measurements per inferred noise, Wald and Laplace intervals run a little
narrow: they use the normal quantile where a t quantile would apply, and the
maximum-likelihood noise is biased low by a factor of about `sqrt((n - p) / n)`. In
a 12-point decay fit, a 95% interval covered about 92% of the time. The profile
likelihood held its 95% there.

Independent Gaussian residuals are often wrong for a densely sampled trajectory,
and treating correlated residuals as independent makes a posterior overconfident.
`log_likelihood=` replaces the Gaussian. It is called once per bucket and summed,
and `noise` keys are then free-form, so they can hold its hyperparameters. An AR(1)
likelihood for one channel on a grid that is regular and fully measured:

```python
def ar1(pred_obs, bp, noise):
    sigma, rho = noise["sigma"], noise["rho"]
    residual = bp.observed[..., 0] - pred_obs[..., 0]
    innovation = residual[:, 1:] - rho * residual[:, :-1]
    variance = sigma**2 * (1 - rho**2)
    first = jstats.norm.logpdf(residual[:, 0], 0.0, sigma)
    rest = jstats.norm.logpdf(innovation, 0.0, jnp.sqrt(variance))
    return first.sum() + rest.sum()
```

## Predictions

`problem.predict` takes the sampler's output, chooses `n_draws` at random, and
reuses `predict_ensemble`. A draw whose solve fails on your grid is dropped with a
warning. `to_model` gives one draw as an ordinary model.

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#predict

`outputs` holds the mean over draws. `std` and `quantiles` describe the noiseless
trajectory, a credible band. With `observation_noise=True` they describe a new
measurement, using each draw's inferred noise; this needs a `noise` mapping with one
entry per channel, because the variances in data exist only at measured cells.

## Diagnostics

`problem.physical(positions)` converts sampler output to physical values, keeping
any leading axes, so a recipe needs no ArviZ:

```python
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction

draws = problem.physical(positions)      # each value has shape [chain, draw, ...]
print(draws["logA"].mean(), draws["logA"].std())
print(potential_scale_reduction(draws["logA"]), effective_sample_size(draws["logA"]))
```

`problem.to_arviz(positions)` takes positions with leading `[chain, draw]` axes and
returns an ArviZ `DataTree` of physical values named by path, with `conditional_on`
in its attributes. It needs Python 3.12 or later. `blackjax.diagnostics` also has
`potential_scale_reduction` and `effective_sample_size`, which is what the example
uses.

## Making it fast

A sampler's cost is its number of gradients, and a run can need tens of thousands
where a fit needs hundreds. A gradient costs about what a training step does, so
the number to reduce is the count. In order of effect on the tubular reactor:

- **Precondition.** `problem.preconditioned()` whitens the position with the
  Gauss-Newton curvature at the fit. It is a constant linear map, so the posterior
  does not change. On the last-layer problem, whose raw Gauss-Newton curvature (not
  the correlation form `fisher()` reports) has a condition number of about 1e6, it took the step size from 0.0097 to 0.67 and warmup from the 63-leapfrog
  depth cap on nearly every iteration to about 7. It needs a fitted model to help,
  and the Gaussian likelihood unless you pass `precision=`.
  Skipping it is not wrong, only slower: in one run on the Arrhenius posterior, with
  a correlation of 0.999 on `logA` and `E`, NUTS mixed well either way, at 47 s
  without it and 5 s with it.
- **Run chains on threads.** Compiled code releases the GIL, so threads gave 3.4
  times the speed of running 4 chains in turn. Vmapping chains makes each one wait
  for the deepest NUTS tree in the batch.
- **Use forward mode for few parameters.** `ad_mode="auto"` does this up to 8 free
  values: 8 ms against 25 ms at 2 parameters, but 368 ms against 187 ms at 17.
- **Tune the solver last.** This tube needs about 200 steps for stability, not
  accuracy, so loosening `rtol` by 100 times saved 5 steps. Damped PID gains
  (`pcoeff=0.3, icoeff=0.4`) saved about 12% of each gradient by rejecting fewer steps.

Together these took the two-parameter posterior from 527 s to about 20 s on 8 CPU cores,
with a larger effective sample size. The 17-parameter last layer took 683 s for 4
chains of 450 iterations each, with R-hat below 1.01.

## Precision and solver tolerance

x64 is the safe default, and `make_problem` warns without it. On the tubular
reactor's two parameters, float32 showed no sign of trouble: no divergences and an
acceptance rate like x64's. That is one easy, well-specified problem, so it does
not show float32 is safe in general. Tightening `rtol` from 1e-5 to 1e-7 did not
move that posterior beyond Monte Carlo error.

Every method here trusts the model's gradient as the likelihood's own. A model
that uses `clip_ste` has a deliberately wrong gradient wherever the clip is
active, so use `softclip` there instead. Through `SaveAt(ts)`, diffrax's `Dopri8`
gave gradients about 1e-5 off in a check where `Tsit5` stayed near 1e-9, so keep
to `Tsit5` or a stiff solver for Fisher, Laplace and profile work.

## Without sampling

`log_prior=None`, the default, gives a problem with only the likelihood, for
frequentist work.
`optimize` finds its maximum with Optimistix's BFGS. A training fit is close to it
but not at it: Adam stops after a finite budget, and a loss other than `"mle"` has
a different minimum when the variances differ. The result's
`converged` says whether BFGS met its tolerances, and `gradient` gives the
remaining slope per value, with `gradient_norm` its largest entry.
`fisher()` then gives the information at that point.

The example fits first-order decay at seven temperatures, with the rate written two
ways: the textbook `A exp(-E / T)`, and the same law centred at a reference
temperature.

<<< @/../examples/identifiability/arrhenius.py#model

<<< @/../examples/identifiability/arrhenius.py#fisher

```text
Fisher information
            estimate    std error  95% interval
  E             5988        29.72  [5930, 6046]
  log_A         17.1      0.09661  [16.91, 17.29]
  Strongest correlations
    E and log_A: +0.999
  Condition number 1.83e+03
```

A profile likelihood needs no Gaussian assumption. `profile(name)` holds one
value fixed at points stepping away from the maximum, re-optimises all the others
with BFGS at each, and closes a side where twice the drop in log-likelihood passes
the chi-square quantile for the confidence level. It is local: it follows the
likelihood from the maximum you start at, so a second mode, such as an aliased
frequency, can hold parameter values the profile never visits. It warns when a
point it reaches fits better than its start. `profiles()` runs every free
`Parameter`:

<<< @/../examples/identifiability/arrhenius.py#profile

```text
E: 95% profile interval [5930, 6047]
log_A: 95% profile interval [16.91, 17.29]
```

Here they match the Wald intervals, because this likelihood is close to Gaussian.
Where it is not, the profile is the one to trust, and it does not depend on how the
model is parametrised. A side that does not close is reported as `None`, and
`profile.stop_reasons` says why: the value reached its bound, so the data do not bound
it; the walk ran out of `max_points`; the model stopped solving; or BFGS did not
converge where the ratio crossed. Only a converged crossing closes a side.
`profile.others` holds every other value along the profile, which shows what
compensates.

The threshold is the $\chi^2_1$ quantile. When the true value lies on a bound, the
likelihood ratio follows $\tfrac12\chi^2_0 + \tfrac12\chi^2_1$ instead (Self and
Liang, 1987), so the side that closes has about 97.5% coverage rather than 95%:
the interval is conservative there.

The information is computed in the latent coordinates the problem uses, and
reported in physical units. Standard errors use the delta method. A Wald interval
is built in the latent and mapped back, so it leans the way a log warp does and
stays inside a `Parameter`'s bounds. `fisher` and `laplace` both default to
`curvature="gauss_newton"`, `J^T J` of the whitened residuals: the expected
information. `curvature="exact"` uses the negative Hessian instead, the observed
information, and is the one that works with a custom `log_likelihood`; with one,
pass it explicitly. The condition number and the
not-identified test read the information scaled to a unit diagonal, its
correlation form, so changing a bound or a warp does not move them.

A correlation of 0.999 says the data fix a combination of the two. `log A` and `E`
trade off along the rates' temperature dependence. Centring the law at a reference
temperature removes it. The Fisher covariance gives the temperature at which
`log k` and `E` are uncorrelated, and it lies inside the data's range. Centring at
the condition you care about instead, such as a storage temperature below the
data, leaves the correlation high: the [shelf life](/examples/shelf-life) example's
25 °C form still has -0.989.

<<< @/../examples/identifiability/arrhenius.py#centred

The centred fit's correlation is essentially zero, and the condition number, read
in correlation form, falls from about 1.8e3 to 1. When the data cannot separate
parameters at all,
`fisher().unidentified()` names the directions it cannot see, such as two rates
whose product is all a model uses. Those parameters' standard errors are infinite,
and the summary lists them under "Not identified". Their Wald interval runs to
their bounds, or to infinity without one: that is the box you declared, not
information from the data. A profile side that closes only because another value
was pinned at its bound is likewise conditional on that bound.

`laplace()` is a Gaussian approximation to the posterior at its mode, so it needs a
prior. `sample` draws positions that `physical`, `predict` and `to_arviz` read as
they read a sampler's output:

<<< @/../examples/identifiability/arrhenius.py#laplace

On the Arrhenius data of the [custom predictor](/examples/custom-predictor) example
with inferred noise, the Laplace standard deviations (measured with
`curvature="exact"`, the default before 0.3.0) were within 4% of NUTS. The
means of `log A` and `E` agreed within 0.02 standard deviations, and the noise's
within 0.16, as its posterior is skewed. A Laplace approximation is only as good
as the posterior is Gaussian in latent space, so check it against a sampler before
relying on it for a new model.

`fisher` and `laplace` warn when the position is not at the maximum, since both
describe a maximum.

`FisherResult.sample` draws from the estimate's asymptotic sampling distribution,
for a frequentist band on any function of the parameters, such as a prediction at
new conditions. The [shelf life](/examples/shelf-life) example uses it for a potency
band at a temperature the study never ran. It refuses when a direction is not
identified, since that spread is unbounded.

### Derived quantities and one-sided bounds

`derived(fn)` gives the estimate and delta-method standard error of any function
of the values. `fn` takes the dict that `physical` returns and must use
`jax.numpy`, since its gradient comes from `jax.jacobian`. A shelf life from the
rate at 25 °C, in the [shelf life](/examples/shelf-life) example's model, whose
`log_k25` is that rate's log:

```python
life = fisher.derived(lambda values: jnp.log(1 / 0.9) / jnp.exp(values["log_k25"]))
low, high = life.interval(confidence=0.95, side="lower")  # high is inf
```

It assumes the quantity is close to Gaussian. For a bound that does not, profile
a parameter that is the quantity itself, or that the quantity depends on
monotonically. A two-sided interval at `confidence=0.90` has a one-sided 95% bound
at each end. To make a quantity at new conditions a parameter, rewrite the model
around those conditions, as the [shelf life](/examples/shelf-life#at-another-storage-temperature)
example does for a storage temperature.

### Values at a bound

A value is pinned when the data would push it past a bound of its `Parameter`, so
the bound, not the data, holds it there. Its latent is then far out in the
squash's flat tail, and the curvature there says nothing about how far the data
would let the value move. Being close to a bound is not enough: a value near the
end of a wide or log-warped box that the data put there is reported as usual.
`fisher` and `laplace` therefore report `nan` for its standard error and Wald
interval, mark the row "at bound", and warn. A `derived` quantity that uses it
directly is `nan` too. Profile the value to see whether the data bound it from
the other side.

The other estimates are the best fit with the pinned value held at its bound, so
they, their standard errors and any `derived` quantity built from them are
conditional on that bound. One correlated with the pinned value is biased when
the bound is wrong, and its interval can exclude the truth while looking normal.
An activation energy capped below its true value pulls the rate constant with it.
Widen the bound and refit, or profile, before trusting the rest.

A profile of another value re-optimises the pinned one at each point, but cannot
move it past its bound. If the data trade the two off, the profile then drops
faster than it would with the bound widened, and closes sooner. The interval is
right for the bounds as set. `profile.others_pinned` names every value that was pinned
along the walk, the summary lists them, and a warning says so. If such a bound is
a numerical guard rather than physics, widen it and profile again.

BFGS and the solves inside it can print `Algebraic simplifier is likely stuck` to
stderr while they compile. This is harmless; see
[troubleshooting](/guide/troubleshooting#algebraic-simplifier-is-likely-stuck-in-stderr).

## Worked example

`examples/tubular_reactor/sample_tubular_reactor.py` samples the Arrhenius constants
of the [tubular reactor](/examples/tubular-reactor), and with `--last-layer` the last
layer of its hybrid model. The mechanistic part takes under a minute on 8 CPU cores,
and the last layer about ten minutes.
