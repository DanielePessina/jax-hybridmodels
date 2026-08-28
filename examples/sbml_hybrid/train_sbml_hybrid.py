"""Hybrid kinetics: a mechanistic SBML-style core with a neural rate.

Shows how an *external* mechanistic model — the kind you'd load from an
SBML file — plugs into a ``hybridmodels`` fit, with one of its rate
parameters supplied by a neural ``BoundedPredictor``.

The mechanistic core here is a hand-written Michaelis-Menten system that
stands in for an external kinetic model. With **jaxkineticmodel**
installed, the swap is:

    from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
    kmodel = SBMLModel("model.xml").get_kinetic_model()
    # kmodel(ts, y0, params) -> ys : a full solver wrapper, carrying its
    # own diffrax solver, tolerances, and adjoint.

The external model **replaces the whole integration block**, not just the
vector field: it owns its solver, so inside ``simulate_fn`` you call it
directly instead of ``solver.diffeqsolve``. To make a rate neural, inject
the predictor's output into the params dict you hand it:

    def simulate_fn(predictors, ts, covariates, y0, solver):
        params = {**kmodel.parameters, "Vmax": predictors[0](covariates).reshape(())}
        return kmodel(ts, y0, params)   # hybridmodels' ``solver`` is unused

That is exactly what the hand-written ``simulate_fn`` below does, except it
uses ``solver.diffeqsolve`` because the core here is a bare vector field.
(Caveat: jaxkineticmodel currently pins ``jax==0.4.35`` and
``optax==0.2.3``, which conflict with ``hybridmodels``' ``jax>=0.10`` /
``optax>=0.2.8``, so run it in a separate environment; the recipe is what
matters.)

The hybrid twist: the true maximum velocity ``Vmax`` depends on the enzyme
level ``E`` via ``Vmax(E) = 2 E``. We give the model a fixed mechanistic
form (Michaelis-Menten with known ``Km``) and let an MLP
``BoundedPredictor`` *learn* ``Vmax`` as a function of ``E`` from noisy
S/P time-series across experiments at different enzyme levels. This is the
crystallisation pattern — known physics for the structure, a network for
the unknown rate law — applied to a standard systems-biology model.

Run:
    uv run python examples/sbml_hybrid/train_sbml_hybrid.py
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
from jax import Array

import hybridmodels as hm
from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from hybridmodels.solver import SolverConfig

# ---------------------------------------------------------------------------
# The mechanistic core (would come from an SBML file via jaxkineticmodel).
# ---------------------------------------------------------------------------

KM_TRUE: float = 1.0
# The unknown rate law the network must recover.
TRUE_VMAX_SLOPE: float = 2.0
ENZYME_LEVELS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
N_TIMESTEPS: int = 40
T_MAX: float = 8.0
NOISE_STD: float = 0.02
S0: float = 2.0


def mechanistic_core(t: Array, y: Array, args) -> Array:
    """A bare kinetic vector field: ``(t, y, (params, ...)) -> dy/dt``.

    This is the shape a hand-written core takes. jaxkineticmodel's public
    ``get_kinetic_model()`` instead returns a full solver wrapper
    ``(ts, y0, params) -> ys`` — the swap point is the integration block,
    as documented in the module docstring.
    """
    params, _boundary, _assignments = args
    vmax, km = params["Vmax"], params["Km"]
    s = y[0]
    rate = vmax * s / (km + s)  # Michaelis-Menten
    return jnp.stack([-rate, rate])


def build_dataset() -> Dataset:
    """One experiment per enzyme level; S and P observed on a shared grid.

    The true Vmax(E) = 2E drives the data; the network never sees E's role
    beyond the covariate it is asked to map to Vmax.
    """
    key = jax.random.PRNGKey(0)
    ts = jnp.linspace(0.0, T_MAX, N_TIMESTEPS)
    experiments = []
    for e in ENZYME_LEVELS:
        vmax_true = TRUE_VMAX_SLOPE * e
        y0 = jnp.asarray([S0, 0.0], dtype=jnp.float32)

        def vector_field(t, y, args):
            params, _bc, _ar = args
            vmax, km = params["Vmax"], params["Km"]
            rate = vmax * y[0] / (km + y[0])
            return jnp.stack([-rate, rate])


        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.05,
            y0=y0,
            args=({"Vmax": vmax_true, "Km": KM_TRUE}, None, None),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=4096,
        )
        ys = jnp.asarray(sol.ys)
        s_obs = ys[:, 0] + NOISE_STD * jax.random.normal(key, ts.shape)
        p_obs = ys[:, 1] + NOISE_STD * jax.random.normal(key, ts.shape)
        experiments.append(
            make_experiment(
                covariates={"enzyme": float(e)},
                channels={
                    "S": ChannelObs(ts=ts, values=s_obs),
                    "P": ChannelObs(ts=ts, values=p_obs),
                },
                y0_fn=lambda c, ch: jnp.asarray([S0, 0.0], dtype=jnp.float32),
                exp_id=f"E_{e:g}",
            )
        )
    return make_dataset(experiments, output_channel_names=("S", "P"))


# ---------------------------------------------------------------------------
# The hybrid model: mechanistic core + a neural rate parameter.
# ---------------------------------------------------------------------------


def simulate_fn(predictors, ts, covariates, y0, solver):
    """Integrate with ``Vmax`` supplied by the network, everything else fixed."""

    def vector_field(t, y, args):
        # The network maps the enzyme covariate to a bounded Vmax.
        vmax_neural = predictors[0]({"enzyme": covariates["enzyme"]}).reshape(())
        params, bc, ar = args
        params = {**params, "Vmax": vmax_neural}
        return mechanistic_core(t, y, (params, bc, ar))


    params = {"Vmax": 1.0, "Km": KM_TRUE}
    sol = solver.diffeqsolve(diffrax.ODETerm(vector_field), ts, y0, args=(params, None, None))
    return jnp.asarray(sol.ys)


def state_to_output(state: Array) -> Array:
    """Both species are measured."""
    return state


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------


def main() -> None:
    ds = build_dataset()
    print(hm.describe_buckets(ds))

    # An MLP with one bounded output: Vmax in [0.5, 10].
    predictor = hm.BoundedPredictor(
        input_keys=("enzyme",),
        in_scaler=hm.BoundScaler(bounds=((0.0, 3.0),)),
        inner=hm.MLPPredictor(
            in_size=1, out_size=1, width_size=16, depth=2,
            activation_name="tanh", key=jax.random.PRNGKey(0),
        ),
        out_scaler=hm.BoundScaler(bounds=((0.5, 10.0),)),
    )
    predictors = (predictor,)
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-7,
        atol=1e-9,
        max_steps=4096,
        dt0=0.05,
    )
    mask = hm.trainable_mask(predictors)

    history, trained = hm.train_with_optax(
        predictors,
        ds,
        hm.OptaxTrainingConfig(
            steps=(120,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            penalty_weight=(1e-2,),
        ),
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        trainable=mask,
        key=jax.random.PRNGKey(0),
    )

    # Diagnostics on the held-in data.
    preds = hm.predict_dataset(
        trained, ds, simulate_fn=simulate_fn,
        state_to_output=state_to_output, solver=solver,
    )
    hm.print_metrics(
        hm.compute_metrics(preds, ds), header=f"fit (final data loss {history[-1]:.3e})"
    )

    # The recovered rate law vs truth: Vmax(E) should be ~2E.
    print("\nrecovered Vmax(E) vs truth 2E:")
    for e in ENZYME_LEVELS:
        learned = hm.evaluate_predictor(trained[0], {"enzyme": float(e)})
        print(f"  E={e:4.1f}  learned Vmax={learned:6.3f}  truth={TRUE_VMAX_SLOPE * e:6.3f}")


if __name__ == "__main__":
    main()
