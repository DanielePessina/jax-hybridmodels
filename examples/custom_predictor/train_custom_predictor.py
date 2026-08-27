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
  hyperparameters? Static fields must be JSON-encodable, because they
  travel in the checkpoint sidecar rather than the binary.
- How does the module re-initialise itself? The restart tournament calls
  ``initialized_with_key``. Without it, every float array is resampled
  from a standard normal, which would destroy the feature bank's scale.
- How do you keep the optimiser off the fixed arrays? With a
  trainability mask, not a field flag. ``freeze_paths`` names the two
  bank leaves.
- Does it survive a save and reload? ``save_predictors`` and
  ``load_predictors`` round-trip through a template built by the same
  code path.

The narrative walkthrough is in ``docs/guide/custom-predictors.md``.

Run::

    uv run python examples/custom_predictor/train_custom_predictor.py
    uv run python examples/custom_predictor/train_custom_predictor.py --train-bank
    uv run python examples/custom_predictor/train_custom_predictor.py --n-features 128
"""

# ruff: noqa: F722

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float

from hybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    Experiment,
    SolverConfig,
    freeze_paths,
    load_predictors,
    make_dataset,
    make_experiment,
    predict_dataset,
    save_predictors,
    trainable_mask,
)
from hybridmodels.predictors.base import Predictor
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

# ``examples/_shared`` is a sibling directory; put it on sys.path so the
# helpers import without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    apply_default_style,
    compute_diagnostics,
    parity_plot,
    print_diagnostics,
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

# Dot-joined paths of the two fixed arrays inside the predictors pytree:
# element 0 of the tuple, the BoundedPredictor's ``inner``, then the field.
BANK_PATHS: tuple[str, ...] = (
    "0.inner.frequencies",
    "0.inner.phases",
)


class RandomFourierPredictor(Predictor):
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
    pytree rather than a property of a class (ADR-0003).

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


def _build_predictor(key: Array, *, n_features: int, bandwidth: float) -> BoundedPredictor:
    """Wrap the custom predictor so it reads a temperature and writes a rate.

    ``BoundedPredictor`` supplies the two things the inner module
    deliberately lacks: named inputs and physical units. The custom class
    never sees a bound.
    """
    return BoundedPredictor(
        input_keys=("temperature",),
        in_scaler=BoundScaler(bounds=TEMPERATURE_BOUNDS, transform="sigmoid"),
        inner=RandomFourierPredictor(
            in_size=1,
            out_size=1,
            n_features=n_features,
            bandwidth=bandwidth,
            key=key,
        ),
        out_scaler=BoundScaler(bounds=K_BOUNDS, transform="sigmoid", warp="log10"),
    )


def _build_mask(predictors: tuple[BoundedPredictor, ...], *, train_bank: bool):
    """Boolean mask over ``predictors``: everything trains except the bank.

    ``trainable_mask`` marks every floating-point array trainable, which
    includes ``frequencies`` and ``phases``. ``freeze_paths`` turns those
    two off by name. A path matching no leaf raises rather than passing
    silently, so a renamed field fails loudly instead of training an
    array you believed was fixed.
    """
    mask = trainable_mask(predictors)
    if train_bank:
        return mask
    return freeze_paths(mask, BANK_PATHS)


def _y0_fn(_cov: dict[str, Array], _channels: dict[str, ChannelObs]) -> Array:
    """Every run starts at the same concentration."""
    return jnp.asarray([Y0], dtype=jnp.float32)


def _state_to_output(state: Float[Array, "T 1"]) -> Float[Array, "T 1"]:
    """The single state component is the single observed channel."""
    return state


def _simulate_fn(
    predictors: tuple[BoundedPredictor, ...],
    ts: Float[Array, " T"],
    covariates: dict[str, Array],
    y0: Float[Array, " 1"],
    solver: SolverConfig,
) -> Float[Array, "T 1"]:
    """Integrate one isothermal run.

    The rate is constant along a trajectory, so the predictor is called
    once here rather than inside the vector field, and never touches the
    solver tape.
    """
    (rate,) = predictors
    k = rate(covariates).reshape(())

    def vector_field(t: Array, y: Float[Array, " 1"], args: object) -> Array:
        return -k * y

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver.solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=solver.dt0 if solver.dt0 is not None else 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=solver.rtol, atol=solver.atol),
        max_steps=solver.max_steps,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.asarray(sol.ys)


def _build_experiments(noise_key: Array) -> list[Experiment]:
    """One noisy decay curve per temperature, from the closed form.

    All runs share the same timestamp grid, so the dataset collapses to a
    single fully observed bucket. Irregular sampling is the general case
    and needs no change to the model code.
    """
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments: list[Experiment] = []
    rng = noise_key
    for temperature in TEMPERATURES:
        rng, k_noise = jr.split(rng)
        clean = Y0 * jnp.exp(-true_k(temperature) * ts)
        noisy = clean + NOISE_STD * jr.normal(k_noise, ts.shape)
        experiments.append(
            make_experiment(
                covariates={"temperature": temperature},
                channels={
                    "concentration": ChannelObs(
                        ts=ts,
                        values=noisy,
                        variance=jnp.full(ts.shape, NOISE_STD**2),
                    )
                },
                y0_fn=_y0_fn,
                exp_id=f"decay_T={temperature:.0f}",
            )
        )
    return experiments


def _read_k(predictors: tuple[BoundedPredictor, ...], temperature: float) -> float:
    """Evaluate the learned rate at one temperature, in physical units."""
    (rate,) = predictors
    out = rate({"temperature": jnp.asarray(temperature)})
    return float(jnp.asarray(out).reshape(()))


def _print_rate_table(predictors: tuple[BoundedPredictor, ...], header: str) -> None:
    """Learned rate against the Arrhenius truth at every sampled temperature."""
    print(header)
    print(f"    {'T [K]':>7}  {'k_true':>9}  {'k_learned':>9}  {'rel. err':>9}")
    for temperature in TEMPERATURES:
        truth = float(true_k(temperature))
        learned = _read_k(predictors, temperature)
        rel = abs(learned - truth) / truth
        print(f"    {temperature:7.0f}  {truth:9.4f}  {learned:9.4f}  {rel:9.2%}")


def _check_roundtrip(
    predictors: tuple[BoundedPredictor, ...], *, n_features: int, bandwidth: float
) -> tuple[BoundedPredictor, ...]:
    """Save the trained predictors and load them back through a template.

    ``load_predictors`` needs a template with the right structure, which
    means building the same predictor again with any key. Every static
    field must be JSON-encodable for this to work, which is why
    ``bandwidth`` is a plain float and the shapes are plain ints.
    """
    template = (_build_predictor(jr.PRNGKey(0), n_features=n_features, bandwidth=bandwidth),)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "custom_predictor.eqx"
        save_predictors(path, predictors)
        return load_predictors(path, template)


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
        "--plot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to write parity + trajectory PNGs into.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (still prints diagnostics).",
    )
    args = parser.parse_args()

    apply_default_style()

    root_key = jr.PRNGKey(args.seed)
    k_data, k_init, k_train = jr.split(root_key, 3)

    print("[build] synthetic Arrhenius decay dataset")
    experiments = _build_experiments(k_data)
    dataset = make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=OUTPUT_CHANNELS,
    )
    print(f"  {len(experiments)} experiments, {len(dataset.bucket_payloads)} bucket(s)")
    print(
        f"  true k spans {float(true_k(TEMPERATURES[0])):.4f} to "
        f"{float(true_k(TEMPERATURES[-1])):.4f}"
    )

    print("\n[build] solver + custom predictor")
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-6,
        atol=1e-8,
        max_steps=10_000,
        dt0=0.1,
    )
    predictors = (_build_predictor(k_init, n_features=args.n_features, bandwidth=args.bandwidth),)
    mask = _build_mask(predictors, train_bank=args.train_bank)
    frozen = "none (--train-bank)" if args.train_bank else ", ".join(BANK_PATHS)
    print(f"  RandomFourierPredictor: {args.n_features} features, bandwidth {args.bandwidth}")
    print(f"  frozen leaves: {frozen}")
    _print_rate_table(predictors, "  rates before training:")

    print("\n[train] optax, 3-attempt tournament then one phase")
    config = OptaxTrainingConfig(
        steps=(args.steps,),
        lr=(args.lr,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        loss="mse",
        tournament_attempts=3,
        tournament_steps=40,
        tournament_lr=args.lr,
        verbose=False,
    )
    history, trained = train_with_optax(
        predictors,
        dataset,
        config,
        simulate_fn=_simulate_fn,
        solver=solver,
        trainable=mask,
        key=k_train,
    )
    print(f"  {len(history)} steps; final loss {history[-1]:.6e}")

    _print_rate_table(trained, "\n  rates after training:")

    print("\n[serialise] save and reload the trained predictor")
    reloaded = _check_roundtrip(trained, n_features=args.n_features, bandwidth=args.bandwidth)
    drift = max(
        abs(_read_k(reloaded, temperature) - _read_k(trained, temperature))
        for temperature in TEMPERATURES
    )
    print(f"  largest rate difference after round-trip: {drift:.3e}")

    print("\n[diagnostics] per-channel parity stats over the training set")
    predictions = predict_dataset(trained, dataset, simulate_fn=_simulate_fn, solver=solver)
    diag = compute_diagnostics(predictions, dataset)
    print_diagnostics(diag)

    if not args.no_plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        parity_plot(
            diag,
            title="Custom predictor parity (trained model)",
            save_path=args.plot_dir / "parity.png",
        )
        trajectory_plot(
            predictions,
            dataset,
            predictors=trained,
            simulate_fn=_simulate_fn,
            solver=solver,
            max_experiments=len(experiments),
            title="Custom predictor trajectories (trained model)",
            save_path=args.plot_dir / "trajectories.png",
        )
        print(f"\n[plot] figures written to {args.plot_dir}")


if __name__ == "__main__":
    main()
