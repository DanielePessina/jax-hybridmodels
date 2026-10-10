# Neural polynomial kinetics

Learns a nucleation rate with `NeuralNPolynomial`, which writes each output
as a polynomial whose coefficients come from an inner network. Read it when
you expect a power law and want that prior in place of a black-box MLP.

## Run it

```bash
uv run python examples/supersaturation_poly/train_supersaturation_poly.py
```

About a minute on CPU. Prints the buckets, metrics and the learned rate at
each supersaturation; writes `parity.png` and `recovered_rate.png` to
`examples/supersaturation_poly/figures/`. `--steps 5 --no-plot` is a quick
smoke run.

## The model

Two moments of the crystal size distribution:

```
d mu0/dt = J          (nucleation)
d mu1/dt = G * mu0    (growth)
```

Growth `G` is known. The nucleation rate `J(S)` is a polynomial with
exponents 2, 3 and 4 whose coefficients come from a small MLP, wrapped in a
`BoundedPredictor` so it stays inside `(0.01, 2.0)`. Only `mu0` is
measured. The data is synthetic, `J_true = 0.02 + 0.15 (S - 1)^4` plus
noise, at five supersaturations.

The polynomial basis is latent. `NeuralNPolynomial` evaluates its
polynomial at `sum(x)`, where `x` is the input after the
`BoundedPredictor`'s input scaler, so the basis is a latent coordinate of
`S - 1`, not `S - 1` itself. For a polynomial in one variable with
coefficients conditioned on another, for example
`sum_i c_i(T) * (S - 1)^p_i`, compose two predictors in the vector field.

## Code

`J` depends only on the constant supersaturation, so `simulate_fn`
evaluates it once and integrates at that rate:

```python
def simulate_fn(predictors, ts, covariates, y0, solver):
    # S is constant within an experiment, so J is evaluated once per solve
    # instead of at every solver step.
    j_rate = predictors["J"]({"super_sat": covariates["super_sat"]})[0]
    return moments(j_rate, ts, y0, solver)
```

The predictor:

```python
# The coefficient network outputs one coefficient per exponent.
coeff_net = hm.MLPPredictor(
    in_size=1,
    out_size=len(EXPONENTS),
    width_size=16,
    depth=2,
    activation_name="tanh",
    key=k_model,
)
rate = hm.BoundedPredictor(
    inner=hm.NeuralNPolynomial(coeff_net=coeff_net, exponents=EXPONENTS, in_size=1, out_size=1),
    input_keys=("super_sat",),
    input_bounds=(-0.2, 2.0),
    # A log warp because J spans two decades inside its bounds.
    out_scaler=hm.BoundScaler(bounds=(J_BOUNDS,), warp="log"),
)
```

## Results

Seed 0, 400 steps. The `mu0` series fit to the noise floor
(`R^2 = 0.998`).

![Predicted against observed nucleation moment](assets/supersaturation-poly/parity.png)

![Learned nucleation rate against the truth](assets/supersaturation-poly/recovered_rate.png)

The learned rate matches the truth at the five training supersaturations
and wanders between them, because the data says nothing about `J` there.
More conditions, or fewer free coefficients, would tighten the curve.

## Things to try

- Change `EXPONENTS` to `(4.0,)`, the true power, and compare the curve
  between training points.
- Swap the polynomial for a plain `MLPPredictor` with the same bounds.
- Add supersaturations to `SUPERSATURATIONS` and watch the gaps close.
