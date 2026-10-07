"""Fit a physical frequency through .value, predict by name, and simulate new conditions.

Run: uv run python examples/pendulum/train_model_api.py
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
    ts = jnp.linspace(0.2, 5.0, 20)  # First measurement is after the initial state.
    experiment = hm.make_experiment(
        exp_id="oscillator",
        covariates={},
        t0=0.0,
        y0=[1.0, 0.0],
        channels={"position": hm.ChannelObs(ts=ts, values=jnp.cos(ts))},
    )
    dataset = hm.make_dataset([experiment], output_channel_names=("position",))
    model = hm.HybridModel(
        predictors={
            "omega": hm.BoundedPredictor(value=1.3, bounds=(0.5, 2.0)),
            "damping": hm.ConstantPredictor(value=0.0),
        },
        simulate_fn=simulate_fn,
        state_to_output=lambda state: state[..., :1],
        output_channels=("position",),
        solver=hm.SolverConfig(
            solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8, dt0=0.05, max_steps=4096
        ),
    )
    config = hm.OptaxTrainingConfig(
        phases=[
            hm.TrainingPhase(steps=100, lr=0.03),
            hm.TrainingPhase(steps=100, lr=0.005),
        ],
        verbose=False,
    )
    result = model.fit(dataset, config, key=jr.key(0))
    print(f"Original omega: {model.predictors['omega'].value:.4f}")
    print(f"Fitted omega: {result.model.predictors['omega'].value:.4f}")
    print(f"Best step: {result.best_step}; stop reason: {result.stop_reason}")
    predictions = result.model.predict(dataset)
    print("Predicted position:", predictions["oscillator"]["position"])
    trajectory = result.model.simulate(
        ts=jnp.linspace(1.0, 8.0, 100),
        t0=0.0,
        y0=[0.5, 0.0],
    )
    print("New trajectory final position:", trajectory.outputs["position"][-1])


if __name__ == "__main__":
    main()
