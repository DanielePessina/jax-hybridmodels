# Uncertainty

The package does no sampling itself. `hm.inference.make_problem` turns a model
and its data into a log-density, a pure JAX function of a pytree position, and a
sampler such as [BlackJAX](https://blackjax-devs.github.io/blackjax/) takes it
from there. Install both with `pip install jax-hybridmodels[bayes]`.

## A first posterior

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#problem

`log_prior` is written on physical values. `noise="data"` trusts the variances in
your data. Neither has a default, because a silent choice in either changes the
answer. `flat_prior` is available for a parameter with finite bounds, but a flat
prior is not neutral under a warp.

The sampler then runs as BlackJAX documents it:

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#sample

Start from a fitted model, because `problem.position` holds its current values.
[Making it fast](#making-it-fast) explains the threads and `preconditioned()`.

`problem.log_density` differs from the training loss in three ways:

- it sums over buckets where the trainers average them, so the grouping of your
  experiments does not change the posterior, and `max_buckets` is safe here;
- it adds the change-of-variables term for free `Parameter`s, so the prior you
  wrote is the prior you get, although the sampler moves in latent space;
- a failed solve gives `-inf` with a zero gradient, which a sampler counts as a
  rejected proposal, instead of an exception that ends the chain.

A `simulate_fn` that calls `diffrax.diffeqsolve` itself must pass
`throw=solver.throw`, or a failed solve still raises.

## What a posterior conditions on

A posterior is only over what is free. Everything else is treated as exactly
known, and its width does not account for it. `problem.conditional_on` lists the
trainable leaves left fixed, and `make_problem` warns when there are any.

| Model and method | What it gives | What it conditions on |
| --- | --- | --- |
| `Parameter`s only, default `free=` | Full posterior over the model's fitted values | Solver settings, `y0` |
| `Parameter`s and a network, default `free=` | `p(theta \| weights, data)` | The fitted weights, which can make it too narrow and off centre |
| `free=` naming a last layer | Posterior over that layer | Every other layer, so a lower bound on the network's uncertainty |
| `fit_ensemble` | Spread across independent fits | Nothing is sampled: members are not posterior draws |

A model with only networks has no `Parameter`, so the default raises. Name a layer
with `free=`, or use [ensembles](/guide/ensembles):

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

`noise="data"` uses each observation's `variance`, and warns for a channel left at
the default of 1.0, because that sets the posterior's width. To infer a standard
deviation per channel, name a `Parameter` for each:

```python
problem = hm.inference.make_problem(
    model, data,
    log_prior=log_prior,  # receives noise["conc"], the physical value
    noise={"conc": hm.Parameter(0.05, bounds=(1e-4, 1.0), warp="log10")},
)
```

Independent Gaussian residuals are often wrong for a densely sampled trajectory,
and treating correlated residuals as independent makes a posterior overconfident.
`log_likelihood=` replaces the Gaussian. It is called once per bucket and summed,
and `noise` keys are then free-form, so they can hold its hyperparameters. An AR(1)
likelihood for one channel on a grid that is regular and fully measured:

```python
def ar1(pred_obs, bp, noise):
    sigma, rho = noise["sigma"], noise["rho"]
    residual = bp.y_observed[..., 0] - pred_obs[..., 0]
    innovation = residual[:, 1:] - rho * residual[:, :-1]
    variance = sigma**2 * (1 - rho**2)
    first = jstats.norm.logpdf(residual[:, 0], 0.0, sigma)
    rest = jstats.norm.logpdf(innovation, 0.0, jnp.sqrt(variance))
    return first.sum() + rest.sum()
```

## Predictions

`problem.predict` takes the sampler's output, chooses `n_draws` at random, and
reuses `predict_ensemble`. `to_model` gives one draw as an ordinary model.

<<< @/../examples/tubular_reactor/sample_tubular_reactor.py#predict

`outputs` holds the mean over draws. `std` and `quantiles` describe the noiseless
trajectory, a credible band. With `observation_noise=True` they describe a new
measurement, using each draw's inferred noise; this needs a `noise` mapping with one
entry per channel, because the variances in data exist only at measured cells.

## Diagnostics

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
  does not change. On the last-layer problem, whose curvature has a condition number
  of 1e6, it took the step size from 0.0097 to 0.67 and warmup from the 63-leapfrog
  depth cap on nearly every iteration to about 7. It needs a fitted model to help,
  and the Gaussian likelihood unless you pass `precision=`.
- **Run chains on threads.** Compiled code releases the GIL, so threads gave 3.4
  times the speed of running 4 chains in turn. Vmapping chains makes each one wait
  for the deepest NUTS tree in the batch.
- **Use forward mode for few parameters.** `gradient="auto"` does this up to 8 free
  values: 8 ms against 25 ms at 2 parameters, but 368 ms against 187 ms at 17.
- **Tune the solver last.** This tube needs about 200 steps for stability, not
  accuracy, so loosening `rtol` by 100 times saved 5 steps. Damped PID gains
  (`pcoeff=0.3, icoeff=0.4`) saved about 12% of each gradient by rejecting fewer steps.

Together these took the two-parameter posterior from 527 s to 18 s on 8 CPU cores,
with a larger effective sample size. The 17-parameter last layer took 559 s for 4
chains of 450 iterations each, with R-hat below 1.01.

## Precision and solver tolerance

x64 is the safe default, and `make_problem` warns without it. On the tubular
reactor's two parameters, float32 showed no sign of trouble: no divergences and an
acceptance rate like x64's. That is one easy, well-specified problem, so it does
not show float32 is safe in general. Tightening `rtol` from 1e-5 to 1e-7 did not
move that posterior beyond Monte Carlo error.

## Worked example

`examples/tubular_reactor/sample_tubular_reactor.py` samples the Arrhenius constants
of the [tubular reactor](/examples/tubular-reactor), and with `--last-layer` the last
layer of its hybrid model. The mechanistic part takes under a minute on 8 CPU cores,
and the last layer about ten minutes.
