"""The shortest end-to-end use of ``HybridModel``: fit, predict, simulate.

One undamped oscillator, ``x(t) = cos(t)``, measured in position only. The
frequency starts at 1.3 and is fitted back to 1.0; the damping is a
``Constant`` and stays at zero. Shows:

- ``Parameter`` and ``Constant`` read through ``.value`` in ``simulate_fn``
- an initial state at ``t0`` before the first measurement
- a two-phase ``OptaxTrainingConfig`` built from ``TrainingPhase``
- ``FitResult.best_step`` and ``FitResult.stop_reason``
- ``predict`` results indexed by experiment and channel name
- ``simulate`` for a new initial state with no dataset

Run::

    uv run python examples/pendulum/train_model_api.py

Prints the original and fitted omega, the run summary, the predicted
positions, and the end point of a new trajectory. Writes nothing.
"""

import diffrax
import jax.numpy as jnp
import jax.random as jr

import jaxhybridmodels as hm


def simulate_fn(predictors, ts, covariates, y0, solver):
    omega = predictors["omega"].value
    damping = predictors["damping"].value

    def rhs(t, y, args):
        return jnp.stack((y[1], -(omega**2) * y[0] - damping * y[1]))

    return solver.diffeqsolve(diffrax.ODETerm(rhs), ts, y0).ys


def main():
    # The state is known at t0 = 0; the first measurement comes later.
    ts = jnp.linspace(0.2, 5.0, 20)
    experiment = hm.make_experiment(
        exp_id="oscillator",
        t0=0.0,
        y0=[1.0, 0.0],
        channels={"position": hm.ChannelObs(ts=ts, values=jnp.cos(ts))},
    )
    dataset = hm.make_dataset([experiment], output_channel_names=("position",))
    model = hm.HybridModel(
        predictors={
            "omega": hm.Parameter(1.3, bounds=(0.5, 2.0)),
            "damping": hm.Constant(0.0),
        },
        simulate_fn=simulate_fn,
        state_to_output=lambda state: state[..., :1],
        output_channel_names=("position",),
        solver=hm.SolverConfig(
            solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, dt0=0.05, max_steps=4096
        ),
    )
    # A fast phase to get close, then a slower one to settle.
    config = hm.OptaxTrainingConfig(
        phases=(
            hm.TrainingPhase(steps=100, lr=0.03),
            hm.TrainingPhase(steps=100, lr=0.005),
        ),
        verbose=False,
    )
    result = model.fit(dataset, config, key=jr.key(0))
    print(f"Original omega: {model.predictors['omega'].value:.4f}")
    print(f"Fitted omega: {result.model.predictors['omega'].value:.4f}")
    print(f"Best step: {result.best_step}; stop reason: {result.stop_reason}")
    # One entry per experiment, one array per channel. ``times`` includes
    # t0, where nothing was measured.
    prediction = result.model.predict(dataset)["oscillator"]
    for t, x in zip(prediction.times[:3], prediction.outputs["position"][:3], strict=True):
        print(f"t = {float(t):.2f}: predicted position {float(x):.4f}")
    # A new initial condition, past the end of the training data.
    trajectory = result.model.simulate(
        ts=jnp.linspace(1.0, 8.0, 100),
        t0=0.0,
        y0=[0.5, 0.0],
    )
    print(f"New trajectory, position at t = 8: {float(trajectory.outputs['position'][-1]):.4f}")


if __name__ == "__main__":
    main()
