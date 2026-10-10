"""Write your own predictor family and train it like any other.

The library ships ``MLPPredictor`` and ``KANPredictor``, but a predictor
is only an ``eqx.Module`` mapping an array to an array. This script
writes a third family from scratch, a random-Fourier-features
regressor, and trains it inside a hybrid ODE without the framework
knowing or caring that it is not an MLP.

The physics
-----------
First-order decay whose rate depends on temperature::

    dy/dt = -k(T) y,     k(T) = A exp(-Ea / (R T))

Each experiment is one isothermal run: a temperature, a decay curve
sampled with Gaussian noise, and the same initial concentration. The
Arrhenius form is the answer, not an input. The predictor sees only the
temperature covariate and has to learn the rate.

What this example is for
------------------------
**A predictor with a fixed part and a trained part.** Random Fourier
features project the input through a random bank of cosines and fit
linear weights on top. The bank is drawn once and held still; only the
output weights and bias learn. That splits the module's arrays into two
groups and forces every question the contract asks:

- Which fields are trainable arrays, and which are static
  hyperparameters? Static fields are not saved, so the load site has to
  be able to rebuild them.
- How does the module re-initialise itself? The restart tournament calls
  ``initialized_with_key``. Without it, every float array is resampled
  from a standard normal, which would destroy the feature bank's scale.
- How do you keep the optimiser off the fixed arrays? With a
  trainability mask, not a field flag. ``freeze_paths`` names the two
  bank leaves, and ``HybridModel.fit(trainable=...)`` takes the mask.
- Does it survive a save and reload? ``save_predictors`` and
  ``load_predictors`` round-trip through a template built by the same
  code path.

The narrative walkthrough is in ``docs/guide/custom-predictors.md``.

Run::

    uv run python examples/custom_predictor/train_custom_predictor.py
    uv run python examples/custom_predictor/train_custom_predictor.py --train-bank
    uv run python examples/custom_predictor/train_custom_predictor.py --n-features 128

Prints the learned rate against the Arrhenius truth before and after
training, the save/load drift, and per-channel metrics. Writes
``figures/parity.png`` and ``figures/trajectories.png`` unless
``--no-plot`` is given.
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float

import jaxhybridmodels as hm
from jaxhybridmodels.prediction import evaluate_predictor

# ``examples/_shared`` is a sibling directory; put it on sys.path so the
# helpers import without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    parity_diagnostics,
    parity_plot,
    trajectory_plot,
)

# Arrhenius constants. Ea/R = 6000 K and A chosen so k(320 K) = 0.2 puts
# the true rates between 0.029 and 1.0 over the temperatures below: 1.5
# decades, which is why the output scaler warps logarithmically.
EA_OVER_R: float = 6000.0
PREFACTOR: float = 2.78e7

TEMPERATURES: tuple[float, ...] = (290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0)

# The input box is wider than the sampled temperatures so the scaler stays
# clear of its saturating ends. The output box is loose on purpose: under
# a log10 warp, headroom costs almost no resolution.
TEMPERATURE_BOUNDS: tuple[tuple[float, float], ...] = ((280.0, 360.0),)
K_BOUNDS: tuple[tuple[float, float], ...] = ((1e-2, 3.0),)

Y0: float = 1.0
T_MAX: float = 25.0
N_TIMESTEPS: int = 15
NOISE_STD: float = 0.01
OUTPUT_CHANNELS: tuple[str, ...] = ("concentration",)

# Dot-joined paths of the two fixed arrays inside the predictors dict: the
# "rate" entry, the BoundedPredictor's ``inner``, then the field.
BANK_PATHS: tuple[str, ...] = (
    "rate.inner.frequencies",
    "rate.inner.phases",
)


class RandomFourierPredictor(hm.Predictor):
    """Random Fourier features with a trained linear readout.

    Maps ``Float[Array, "in_size"] -> Float[Array, "out_size"]`` as::

        phi(x) = sqrt(2 / n_features) * cos(x @ frequencies + phases)
        f(x)   = phi(x) @ weights + bias

    ``frequencies`` and ``phases`` are the feature bank: drawn once at
    construction and never updated, which is what makes this random
    *features* rather than a one-hidden-layer network with a cosine
    activation. Holding them still is the caller's job, via the
    trainability mask built in :func:`_build_mask`; a module cannot
    declare a field untrainable, because trainability is a mask over the
    pytree rather than a property of a class.

    ``bandwidth`` is the standard deviation of the frequency draw. It
    sets the length scale the features can resolve: large values fit
    wiggly functions and overfit sparse data, small values underfit.
    Because ``BoundScaler`` hands the inner predictor a latent input of
    roughly unit scale, a bandwidth near 1 is the sane starting point.

    Attributes
    ----------
    weights : Float[Array, "n_features out_size"]
        Trainable readout weights.
    bias : Float[Array, " out_size"]
        Trainable readout offset.
    frequencies : Float[Array, "in_size n_features"]
        Fixed random projection.
    phases : Float[Array, " n_features"]
        Fixed random offsets, drawn on ``[0, 2 pi)``.
    in_size, out_size, n_features : int
        Static shape metadata.
    bandwidth : float
        Static frequency scale.
    """

    weights: Float[Array, "n_features out_size"]
    bias: Float[Array, " out_size"]
    frequencies: Float[Array, "in_size n_features"]
    phases: Float[Array, " n_features"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    bandwidth: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        n_features: int = 64,
        bandwidth: float = 1.0,
        key: Array,
    ) -> None:
        k_freq, k_phase, k_weight = jr.split(key, 3)
        self.in_size = in_size
        self.out_size = out_size
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.frequencies = bandwidth * jr.normal(k_freq, (in_size, n_features))
        self.phases = 2.0 * jnp.pi * jr.uniform(k_phase, (n_features,))
        # Scaling the readout by 1/sqrt(n_features) keeps the output
        # variance independent of the feature count, so changing
        # ``n_features`` does not silently change the effective learning
        # rate.
        self.weights = jr.normal(k_weight, (n_features, out_size)) / jnp.sqrt(n_features)
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: Float[Array, " in_size"]) -> Float[Array, " out_size"]:
        features = jnp.sqrt(2.0 / self.n_features) * jnp.cos(x @ self.frequencies + self.phases)
        return features @ self.weights + self.bias

    def initialized_with_key(self, key: Array) -> RandomFourierPredictor:
        """Re-init for the restart tournament: a fresh bank and readout.

        The default in ``reinitialize_with_key`` replaces every float
        leaf with a standard normal, which would give the bank the wrong
        scale and drop ``bandwidth`` on the floor. Rebuilding through
        ``__init__`` keeps the construction scheme in one place.

        Shapes are unchanged, so a trainability mask built before
        training still lines up after a restart.
        """
        return RandomFourierPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            n_features=self.n_features,
            bandwidth=self.bandwidth,
            key=key,
        )


def true_k(temperature: float | Array) -> Array:
    """Ground-truth Arrhenius rate constant, in reciprocal time units."""
    return PREFACTOR * jnp.exp(-EA_OVER_R / jnp.asarray(temperature))


def build_predictors(key: Array, *, n_features: int, bandwidth: float) -> dict[str, Any]:
    """Wrap the custom predictor so it reads a temperature and writes a rate.

    ``BoundedPredictor`` supplies the two things the inner module
    deliberately lacks: named inputs and physical units. The custom class
    never sees a bound.
    """
    rate = hm.BoundedPredictor(
        input_keys=("temperature",),
        in_scaler=hm.BoundScaler(bounds=TEMPERATURE_BOUNDS, transform="sigmoid"),
        inner=RandomFourierPredictor(
            in_size=1, out_size=1, n_features=n_features, bandwidth=bandwidth, key=key
        ),
        out_scaler=hm.BoundScaler(bounds=K_BOUNDS, transform="sigmoid", warp="log10"),
    )
    return {"rate": rate}


def simulate_fn(
    predictors: dict[str, Any],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 1"],
    solver: hm.SolverConfig,
) -> Float[Array, "T 1"]:
    """Integrate one isothermal run.

    The rate is constant along a trajectory, so the predictor is called
    once here rather than inside the vector field, and never touches the
    solver tape.
    """
    k = predictors["rate"](covariates).reshape(())

    def vector_field(t: ArrayLike, y: Float[Array, " 1"], args: object) -> Array:
        return -k * y

    return jnp.asarray(solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0).ys)


def state_to_output(state: Float[Array, "T 1"]) -> Float[Array, "T 1"]:
    """The single state component is the single observed channel."""
    return state


def build_dataset(key: Array) -> hm.Dataset:
    """One noisy decay curve per temperature, from the closed form.

    All runs share one time grid, so the dataset is a single fully observed
    bucket. Irregular sampling needs no change to the model code.
    """
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments = []
    for temperature in TEMPERATURES:
        key, k_noise = jr.split(key)
        clean = Y0 * jnp.exp(-true_k(temperature) * ts)
        noisy = clean + NOISE_STD * jr.normal(k_noise, ts.shape)
        experiments.append(
            hm.make_experiment(
                covariates={"temperature": temperature},
                channels={
                    "concentration": hm.ChannelObs(
                        ts=ts, values=noisy, variance=jnp.full(ts.shape, NOISE_STD**2)
                    )
                },
                y0=[Y0],
                exp_id=f"decay_T={temperature:.0f}",
            )
        )
    return hm.make_dataset(experiments, output_channel_names=OUTPUT_CHANNELS)


def print_rate_table(predictors: dict[str, Any], header: str) -> None:
    """Learned rate against the Arrhenius truth at every sampled temperature."""
    print(header)
    print(f"    {'T [K]':>7}  {'k_true':>9}  {'k_learned':>9}  {'rel. err':>9}")
    for temperature in TEMPERATURES:
        truth = float(true_k(temperature))
        learned = evaluate_predictor(predictors["rate"], {"temperature": temperature})
        rel = abs(learned - truth) / truth
        print(f"    {temperature:7.0f}  {truth:9.4f}  {learned:9.4f}  {rel:9.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=64, help="Size of the feature bank.")
    parser.add_argument("--bandwidth", type=float, default=1.0, help="Frequency draw scale.")
    parser.add_argument(
        "--train-bank",
        action="store_true",
        help="Train the frequency bank too, instead of freezing it.",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the figures.")
    args = parser.parse_args()

    apply_default_style()
    k_data, k_init, k_train = jr.split(jr.key(args.seed), 3)

    dataset = build_dataset(k_data)
    print(hm.describe_buckets(dataset))
    print(
        f"true k spans {float(true_k(TEMPERATURES[0])):.4f} to "
        f"{float(true_k(TEMPERATURES[-1])):.4f}"
    )

    model = hm.HybridModel(
        predictors=build_predictors(k_init, n_features=args.n_features, bandwidth=args.bandwidth),
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
        solver=hm.SolverConfig(
            solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=10_000, dt0=0.1
        ),
    )
    # ``trainable_mask`` marks every array of the custom predictor
    # trainable, the bank included. ``freeze_paths`` turns the bank off by
    # name, and raises on a path that matches nothing, so a renamed field
    # fails loudly instead of training an array you believed was fixed.
    mask = hm.trainable_mask(model.predictors)
    if not args.train_bank:
        mask = hm.freeze_paths(mask, BANK_PATHS)
    frozen = "none (--train-bank)" if args.train_bank else ", ".join(BANK_PATHS)
    print(f"\nRandomFourierPredictor: {args.n_features} features, bandwidth {args.bandwidth}")
    print(f"frozen leaves: {frozen}")
    print_rate_table(model.predictors, "rates before training:")

    # The tournament restarts the predictor three times through
    # ``initialized_with_key`` and keeps the best start.
    config = hm.OptaxTrainingConfig(
        steps=args.steps,
        lr=args.lr,
        loss="mse",
        tournament_attempts=3,
        tournament_steps=40,
        tournament_lr=args.lr,
        verbose=False,
    )
    result = model.fit(dataset, config, key=k_train, trainable=mask)
    trained = result.model.predictors
    print(f"\n{len(result.history.loss)} steps; final loss {result.history.loss[-1]:.6e}")
    print_rate_table(trained, "rates after training:")

    # Static fields are not saved. ``load_predictors`` takes them from a
    # template, built here by the same function with any key.
    template = build_predictors(jr.key(0), n_features=args.n_features, bandwidth=args.bandwidth)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "custom_predictor.eqx"
        hm.save_predictors(path, trained)
        reloaded = hm.load_predictors(path, template)
    drift = max(
        abs(
            evaluate_predictor(reloaded["rate"], {"temperature": t})
            - evaluate_predictor(trained["rate"], {"temperature": t})
        )
        for t in TEMPERATURES
    )
    print(f"\nlargest rate difference after save and load: {drift:.3e}\n")

    predictions = result.model.predict(dataset).buckets
    hm.print_metrics(hm.compute_metrics(predictions, dataset))

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            parity_diagnostics(predictions, dataset),
            title="Custom predictor parity",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained,
            simulate_fn=simulate_fn,
            state_to_output=state_to_output,
            solver=model.solver,
            max_experiments=len(TEMPERATURES),
            title="Custom predictor trajectories",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\nfigures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
