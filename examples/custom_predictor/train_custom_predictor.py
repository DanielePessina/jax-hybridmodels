"""Write your own predictor family and train it like any other.

A predictor is only an ``eqx.Module`` mapping an array to an array. This
script writes a random-Fourier-features regressor from scratch and learns
an Arrhenius rate ``k(T)`` inside first-order decay,
``dy/dt = -k(T) y``, from noisy isothermal runs. The feature bank is drawn
once and held fixed; only the output weights and bias train.

Shows:

- a custom ``Predictor`` with static fields and ``initialized_with_key``
- ``HybridModel.fit(freeze=...)`` with dotted paths to hold the bank
- the restart tournament (``tournament_attempts``)
- ``hm.save`` / ``hm.load(like=...)`` round-trip through a template model

Run::

    uv run python examples/custom_predictor/train_custom_predictor.py
    uv run python examples/custom_predictor/train_custom_predictor.py --train-bank
    uv run python examples/custom_predictor/train_custom_predictor.py --steps 5 --no-plot

Prints the learned rate against the truth before and after training, the
save/load drift and metrics; writes ``figures/parity.png`` and
``figures/trajectories.png``. About a minute on CPU.
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
TEMPERATURE_BOUNDS: tuple[float, float] = (280.0, 360.0)
K_BOUNDS: tuple[float, float] = (1e-2, 3.0)

Y0: float = 1.0
T_MAX: float = 25.0
N_TIMESTEPS: int = 15
NOISE_STD: float = 0.01
OUTPUT_CHANNELS: tuple[str, ...] = ("concentration",)

# region bank-paths
# Dot-joined paths of the two fixed arrays inside the predictors dict: the
# "rate" entry, the BoundedPredictor's ``inner``, then the field.
BANK_PATHS: tuple[str, ...] = (
    "rate.inner.frequencies",
    "rate.inner.phases",
)
# endregion bank-paths


class RandomFourierPredictor(hm.Predictor):
    """Random Fourier features with a trained linear readout.

    Maps ``Float[Array, "in_size"] -> Float[Array, "out_size"]`` as::

        phi(x) = sqrt(2 / n_features) * cos(x @ frequencies + phases)
        f(x)   = phi(x) @ weights + bias

    ``frequencies`` and ``phases`` are the feature bank: drawn once at
    construction and never updated, which is what makes this random
    *features* rather than a one-hidden-layer network with a cosine
    activation. Holding them still is the caller's job, via the
    ``freeze=`` argument to ``fit`` in ``main``; a module cannot declare a field
    untrainable, because trainability is a mask over the pytree rather
    than a property of a class.

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

    # region predictor
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

        Shapes are unchanged, so the trainability mask built from
        ``freeze=`` still lines up after a restart.
        """
        return RandomFourierPredictor(
            in_size=self.in_size,
            out_size=self.out_size,
            n_features=self.n_features,
            bandwidth=self.bandwidth,
            key=key,
        )

    # endregion predictor


def true_k(temperature: float | Array) -> Array:
    """Ground-truth Arrhenius rate constant, in reciprocal time units."""
    return PREFACTOR * jnp.exp(-EA_OVER_R / jnp.asarray(temperature))


def build_predictors(key: Array, *, n_features: int, bandwidth: float) -> dict[str, Any]:
    """Wrap the custom predictor so it reads a temperature and writes a rate.

    ``BoundedPredictor`` supplies the two things the inner module
    deliberately lacks: named inputs and physical units. The custom class
    never sees a bound.
    """
    # region bounds
    rate = hm.BoundedPredictor(
        input_keys=("temperature",),
        input_bounds=TEMPERATURE_BOUNDS,  # default sigmoid input scaler
        inner=RandomFourierPredictor(
            in_size=1, out_size=1, n_features=n_features, bandwidth=bandwidth, key=key
        ),
        out_scaler=hm.BoundScaler(bounds=K_BOUNDS, transform="sigmoid", warp="log10"),
    )
    return {"rate": rate}
    # endregion bounds


def simulate_fn(
    predictors: dict[str, Any],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 1"],
    solver: hm.SolverConfig,
) -> diffrax.Solution:
    """Integrate one isothermal run; the framework reads the state from ``.ys``.

    The rate is constant along a trajectory, so the predictor is called
    once here rather than inside the vector field, and never touches the
    solver tape.
    """
    k = predictors["rate"](covariates)  # one output, so a scalar

    def vector_field(t: ArrayLike, y: Float[Array, " 1"], args: object) -> Array:
        return -k * y

    return solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0)


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
                    "concentration": hm.ChannelObs(ts=ts, values=noisy, variance=NOISE_STD**2)
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
        learned = float(hm.evaluate_predictor(predictors["rate"], {"temperature": temperature}))
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
    print("[data]")
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
        # Tighter tolerances and a larger step budget than the defaults.
        solver=hm.SolverConfig(rtol=1e-6, atol=1e-8, max_steps=10_000, dt0=0.1),
    )
    # region mask
    # By default every array of the custom predictor trains, the bank
    # included. ``fit(freeze=...)`` turns the bank off by name, and raises
    # on a path that matches nothing, so a renamed field fails loudly
    # instead of training an array you believed was fixed.
    freeze = None if args.train_bank else BANK_PATHS
    # endregion mask
    frozen = "none (--train-bank)" if args.train_bank else ", ".join(BANK_PATHS)
    print("\n[fit]")
    print(f"RandomFourierPredictor: {args.n_features} features, bandwidth {args.bandwidth}")
    print(f"frozen leaves: {frozen}")
    print_rate_table(model.predictors, "rates before training:")

    # region fit
    # The tournament restarts the predictor three times through
    # ``initialized_with_key`` and keeps the best start.
    config = hm.OptaxTrainingConfig(
        steps=args.steps,
        lr=args.lr,
        tournament_attempts=3,
        tournament_steps=40,
        tournament_lr=args.lr,
        verbose=False,
    )
    result = model.fit(dataset, config, key=k_train, freeze=freeze)
    # endregion fit
    trained = result.model.predictors
    print("\n[results]")
    print(f"{len(result.history.loss)} steps; final loss {result.history.loss[-1]:.6e}")
    print_rate_table(trained, "rates after training:")

    # Static fields and simulate_fn are not saved. ``load`` takes them from
    # ``like``, a model built the same way: the untrained ``model`` is one.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "custom_predictor"
        hm.save(path, result, config=config)
        run = hm.load(path, like=model)
    reloaded = run.model.predictors
    assert len(run.history.loss) == len(result.history.loss), "history should round-trip"
    grid = {"temperature": jnp.asarray(TEMPERATURES)}
    k_reloaded = hm.evaluate_predictor(reloaded["rate"], grid)
    k_trained = hm.evaluate_predictor(trained["rate"], grid)
    drift = float(jnp.max(jnp.abs(k_reloaded - k_trained)))
    print(f"\nlargest rate difference after save and load: {drift:.3e}\n")

    predictions = result.model.predict(dataset)
    hm.print_metrics(predictions)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            parity_diagnostics(predictions),
            title="Custom predictor parity",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            result.model,
            dataset,
            predictions=predictions,
            max_experiments=len(TEMPERATURES),
            title="Custom predictor trajectories",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
