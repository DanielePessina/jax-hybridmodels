"""Hybrid ODE modelling with jaxhybridmodels, walked through end to end.

Keeps a mechanistic model's known structure, learns the two parts of it
that have no formula, and holds every physical quantity inside a declared
range while doing so. Runs on data where each measured channel has its own
timestamps.

Self-contained: the data generator, the physics and the models are defined
here rather than imported, so it reads top to bottom. The script version is
``train_hybrid_ode.py`` in the same directory.

Run as script:     ``uv run python examples/hybrid_ode/notebook.py``

# Fitting a hybrid ODE model

## The situation this library is for

You have measurements of something that changes over time, and a
differential equation you partly believe:

$$
\frac{dy}{dt} = f(y, t; \theta)
$$

Some of $f$ comes from conservation laws or a mechanism you trust. Some
does not: a rate constant varies with temperature in a way nobody has
written down, or a term is missing altogether. You want the missing
parts learned from data without discarding the parts you know.

That is a **hybrid model**: mechanistic structure in closed form,
unknown pieces replaced by trainable networks. It is worth the trouble
because a model that keeps its structure extrapolates, and because its
parameters keep their physical meaning after fitting. The last section
shows what a fitted rate constant becomes when you skip the correction
and let the mechanistic parameter absorb the error instead.

## What this library provides

Three things, all awkward by hand.

**Physical quantities stay inside their ranges.** A rate constant cannot
be negative; a mole fraction lives in $[0, 1]$. An optimiser that does
not know this proposes values that make the solver diverge, and clipping
is not the fix: a clip has zero derivative outside the range, so it
destroys the gradient that would pull the parameter back, exactly when
it is needed. This library reparameterises instead, so an out-of-range
value cannot be represented at all.

**Irregular measurements need no padding or interpolation.** Two
instruments rarely agree on when they sampled, and solvers want
rectangular arrays. The library takes a per-experiment union of
timestamps, marks the holes with a mask, and groups experiments by
length so each group is one compiled solve.

**Trainable pieces compose with mechanistic terms.** A network can sit
above the solver, inside the vector field, or both, and the training
loop is never told which.

## What this library is not

Not a neural ODE library. A trainable network inside a vector field is
one thing you can build here, and
[diffrax](https://docs.kidger.site/diffrax/) and
[Equinox](https://docs.kidger.site/equinox/) already document that
technique. This notebook uses it in one line without explaining it.

Assumed: Python, and some experience fitting models to data. Not
assumed: JAX, Equinox, diffrax, or the chemistry.
"""

# ruff: noqa: F722

from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from jaxhybridmodels import (
    BoundedPredictor,
    BoundScaler,
    ChannelObs,
    MLPPredictor,
    SolverConfig,
    Warp,
    make_dataset,
    make_experiment,
    predict_dataset,
    register_warp,
)
from jaxhybridmodels.penalties import bound_penalty, box_grid
from jaxhybridmodels.training.optax import OptaxTrainingConfig, train_with_optax


def main() -> None:
    Path("examples/hybrid_ode/figures").mkdir(parents=True, exist_ok=True)

    # 1. The system, and the two gaps in the model
    # --------------------------------------------
    # A two-dimensional system: a rotation at fixed frequency w, damped at
    # rate k, with a cubic coupling on top.
    #
    #     dy/dt = [[-k, w], [-w, -k]] y  +  C y^3        w = 1
    #
    # The model keeps the rotation and treats w as known. Two things it will
    # not know.
    #
    # The damping rate depends on temperature. Each experiment runs at one
    # of six temperature levels, and the true rate follows Arrhenius:
    #
    #     k(T) = k_ref * exp( -(Ea/R) * (1/T - 1/T_ref) )
    #
    # Across the six levels k runs from 0.0063 to 0.276, a factor of 44.
    # Remember that number: it is why k needs a logarithmic axis in section 3.
    #
    # The cubic coupling is missing entirely. C y^3 has no counterpart in
    # the model, so a network has to reproduce its effect from trajectories
    # alone. In a real problem you would know neither; here they are
    # synthesised so there is something to check the fit against.

    OMEGA_TRUE = 1.0
    COUPLING = jnp.array([[0.0, 0.6], [-0.6, 0.0]])
    K_REF = 0.05
    EA_OVER_R = 6000.0
    T_REF = 310.0
    TEMPERATURES = (280.0, 292.0, 304.0, 316.0, 328.0, 340.0)
    CHANNELS = ("y1", "y2")

    def true_k(temperature):
        """Arrhenius decay rate. Ground truth; never shown to the model."""
        t = jnp.asarray(temperature)
        return K_REF * jnp.exp(-EA_OVER_R * (1.0 / t - 1.0 / T_REF))

    def true_field(k, y):
        """Damped rotation plus cubic coupling.

        Split the way the model will split it. The first term is what the
        hybrid keeps in closed form up to the unknown `k`; the second is
        what the residual network has to learn.
        """
        rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]]) @ y
        return rotation + COUPLING @ y**3

    def solve_reference(field, ts, y0):
        """Integrate `field` from `y0` and sample at `ts`.

        Used only to manufacture ground truth, so the tolerances are far
        tighter than anything training will use.
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(field),
            diffrax.Tsit5(),
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=100_000,
        )
        return jnp.asarray(sol.ys)

    print("true k across the temperature levels:")
    for _t in TEMPERATURES:
        print(f"  {_t:6.1f} K   k = {float(true_k(_t)):.5f}")
    print(f"  ratio hottest / coldest = {float(true_k(340.0) / true_k(280.0)):.1f}")

    # 2. Getting data into the library
    # ----------------------------------------------------------
    #
    # Data enters as one `Experiment` per run, holding covariates (the
    # temperature), one `ChannelObs` per measured quantity (each with its
    # own timestamps, values and variances), and a `y0_fn` returning the
    # full state at t = 0. `make_dataset` then takes the union of each
    # experiment's channel timestamps, writes each channel's values into
    # the rows where it was measured, records a boolean mask over the real
    # cells, and stacks experiments whose union has the same length into
    # one bucket. One bucket is one vectorised compiled solve.
    #
    # The same physics is sampled two ways. Rectangular: every experiment
    # on the same 20-point grid, both channels measured every time.
    # Irregular: each experiment gets its own end time in [6, 9], its own
    # drawn sample times, and each channel thinned independently to 8 or
    # 12 samples. The union length is n1 + n2 - 1, so three possible
    # lengths appear: 15, 19 and 23.

    # JAX has no global RNG: every random draw takes an explicit key, and
    # splitting one root key is how a run stays reproducible. Split the
    # same way train_hybrid_ode.py does, so this notebook reproduces the
    # numbers quoted on the documentation page.
    k_data, k_rate_init, k_res_init, k_train = jr.split(jr.PRNGKey(0), 4)

    NOISE_STD = 0.03

    def y0_from_first_samples(covariates, channels):
        """Initial state, read off the t=0 sample of each channel."""
        return jnp.stack([channels[name].values[0] for name in CHANNELS])

    def build_experiment(ts_per_channel, values_per_channel, temperature, exp_id):
        channels = {
            name: ChannelObs(
                ts=ts, values=vals, variance=jnp.full(ts.shape, NOISE_STD**2)
            )
            for name, ts, vals in zip(CHANNELS, ts_per_channel, values_per_channel, strict=True)
        }
        return make_experiment(
            covariates={"temperature": float(temperature)},
            channels=channels,
            y0_fn=y0_from_first_samples,
            exp_id=exp_id,
        )

    def state_to_output(state):
        """Both state components are observed, in order, so this is the identity.

        In a problem with hidden states this would select or combine.
        The solver integrates the full state; this maps it to predicted
        observations, column order matching `output_channel_names`.
        """
        return state

    def build_dataset(experiments):
        return make_dataset(
            experiments,
            output_channel_names=CHANNELS,
        )

    def describe_buckets(dataset):
        lines = [f"{len(dataset.bucket_payloads)} bucket(s)"]
        for i, bp in enumerate(dataset.bucket_payloads):
            n, t, d = bp.y_observed.shape
            lines.append(
                f"  bucket {i}: N={n:2d} experiments, T={t:3d} timestamps, "
                f"D={d} channels, mask {float(bp.mask.mean()):.2f} full"
            )
        return "\n".join(lines)

    N_EXPERIMENTS = 24

    rect_ts = jnp.linspace(0.0, 8.0, 20)
    _y0_key, _noise_key = jr.split(k_data)
    _y0s = jr.uniform(_y0_key, (N_EXPERIMENTS, 2), minval=-0.6, maxval=1.0)
    rect_temps = jnp.asarray([TEMPERATURES[i % len(TEMPERATURES)] for i in range(N_EXPERIMENTS)])
    _clean = jax.vmap(
        lambda y0, temp: solve_reference(
            lambda t, y, args: true_field(true_k(temp), y), rect_ts, y0
        )
    )(_y0s, rect_temps)
    rect_observed = _clean + NOISE_STD * jr.normal(_noise_key, _clean.shape)

    rect_dataset = build_dataset(
        [
            build_experiment(
                (rect_ts, rect_ts),
                (rect_observed[i, :, 0], rect_observed[i, :, 1]),
                rect_temps[i],
                f"rect_{i:02d}",
            )
            for i in range(N_EXPERIMENTS)
        ]
    )
    print("rectangular sampling")
    print(describe_buckets(rect_dataset))

    SAMPLES_PER_CHANNEL = (8, 12)

    irr_experiments = []
    for _i, _exp_key in enumerate(jr.split(k_data, N_EXPERIMENTS)):
        _k_end, _k_y0, _k_n, _k_t1, _k_t2, _k_noise = jr.split(_exp_key, 6)
        _temperature = TEMPERATURES[_i % len(TEMPERATURES)]
        _k_decay = true_k(_temperature)
        _end = jr.uniform(_k_end, (), minval=6.0, maxval=9.0)

        # A sample count per channel, then that many times in (0, end],
        # with t=0 prepended so y0_fn has something to read.
        _counts = jr.choice(_k_n, jnp.asarray(SAMPLES_PER_CHANNEL), shape=(2,))
        _ts_channels = []
        for _count_key, _count in zip((_k_t1, _k_t2), _counts, strict=True):
            _interior = jr.uniform(_count_key, (int(_count) - 1,)) * _end
            _ts_channels.append(jnp.concatenate([jnp.zeros((1,)), jnp.sort(_interior)]))

        # One dense solve on the union of both channels' times, then read
        # each channel off it. Solving twice would give the two channels
        # independently truncated views of the same system.
        _union = jnp.unique(jnp.concatenate(_ts_channels), size=int(_counts.sum()) - 1)
        _y0 = jr.uniform(_k_y0, (2,), minval=-0.6, maxval=1.0)
        _ys = solve_reference(lambda t, y, args, _k=_k_decay: true_field(_k, y), _union, _y0)

        _noise_keys = jr.split(_k_noise, 2)
        _values = []
        for _d, (_ts_c, _nkey) in enumerate(zip(_ts_channels, _noise_keys, strict=True)):
            _clean_c = _ys[jnp.searchsorted(_union, _ts_c), _d]
            _values.append(_clean_c + NOISE_STD * jr.normal(_nkey, _clean_c.shape))

        irr_experiments.append(
            build_experiment(
                (_ts_channels[0], _ts_channels[1]),
                (_values[0], _values[1]),
                _temperature,
                f"irr_{_i:02d}_T{int(_temperature)}",
            )
        )

    irr_dataset = build_dataset(irr_experiments)
    print("irregular sampling, channels thinned separately")
    print(describe_buckets(irr_dataset))
    print()
    print("first experiment, samples per channel:")
    for _name in CHANNELS:
        print(f"  {_name}: {irr_experiments[0].channels[_name].ts.shape[0]}")

    # One bucket against three, and a mask that goes from completely full
    # to about half full. The figure below makes the second case concrete:
    # black cells are real measurements, white cells are positions on the
    # union axis where the *other* channel was measured and this one was
    # not. Nothing is interpolated to fill them; the loss skips them.
    fig_mask, axes_mask = plt.subplots(4, 1, figsize=(9, 5.2))
    _panels = [
        (rect_dataset.bucket_payloads[0], 0, "rectangular, y1"),
        (rect_dataset.bucket_payloads[0], 1, "rectangular, y2"),
        (irr_dataset.bucket_payloads[1], 0, "irregular bucket 1, y1"),
        (irr_dataset.bucket_payloads[1], 1, "irregular bucket 1, y2"),
    ]
    for _ax, (_bp, _d, _label) in zip(axes_mask, _panels, strict=True):
        _ax.imshow(
            np.asarray(_bp.mask[:, :, _d]),
            aspect="auto",
            cmap="binary",
            interpolation="nearest",
        )
        _ax.set_ylabel(_label, fontsize=7)
        _ax.set_xticks([])
        _ax.set_yticks([])
    axes_mask[-1].set_xlabel("position on the union timestamp axis")
    fig_mask.suptitle("Mask: black is measured, white is a hole")
    fig_mask.tight_layout()
    fig_mask.savefig("examples/hybrid_ode/figures/mask_layout.png")
    plt.close(fig_mask)

    fig_raw, axes_raw = plt.subplots(1, 4, figsize=(13, 3.0))
    for _i in range(rect_observed.shape[0]):
        axes_raw[0].plot(rect_ts, rect_observed[_i, :, 0], lw=0.8, alpha=0.6)
    axes_raw[0].set(xlabel="t", ylabel="y1", title="rectangular, all experiments")
    for _ax, _exp in zip(axes_raw[1:], irr_experiments[:3], strict=True):
        for _name, _marker in (("y1", "o"), ("y2", "s")):
            _obs = _exp.channels[_name]
            _ax.plot(_obs.ts, _obs.values, _marker, ms=4, alpha=0.85, label=_name)
        _ax.set(xlabel="t", title=_exp.exp_id)
    axes_raw[1].legend(fontsize=8)
    fig_raw.suptitle("Left: one shared grid. Right: the two channels rarely share a timestamp.")
    fig_raw.tight_layout()
    fig_raw.savefig("examples/hybrid_ode/figures/raw_data.png")
    plt.close(fig_raw)

    # 3. Keeping physical quantities in range
    # ---------------------------------------------------
    #
    # Bounds hold by reparameterisation, not by clipping: `BoundScaler`
    # wraps a network so the output is `x = lo + (hi - lo) * sigma(z / T)`,
    # and an out-of-range value is not representable. `to_latent` is the
    # exact inverse, continued linearly outside a narrow band so the logit
    # poles at the box ends do not kill the gradient.
    #
    # Reparameterising is not free: the derivative carries sigma'(z/T),
    # which for a logistic sigmoid in float32 underflows to zero past
    # |z/T| ~ 15. The choice of squash trades that:
    #
    #     squash     tail of |du/dz|    dead at
    #     sigmoid    e^-|z|             z ~ 17
    #     algebraic  |z|^-3 / 2         z ~ 3e3
    #     softsign   |z|^-2 / 2         z ~ 1e7
    #
    # Below, the left panel is what the model can express, the right is
    # whether it can still learn once it is out there. Note the log scale.
    fig_sq, axes_sq = plt.subplots(1, 2, figsize=(10, 3.4))
    _z = jnp.linspace(-25.0, 25.0, 801)
    for _name in ("sigmoid", "algebraic", "softsign"):
        _scaler = BoundScaler(bounds=((0.0, 10.0),), transform=_name)
        _value = _scaler.from_latent(_z[:, None])[:, 0]
        _grad = jax.vmap(jax.grad(lambda v, s=_scaler: s.from_latent(v[None])[0]))(_z)
        axes_sq[0].plot(_z, _value, label=_name)
        axes_sq[1].plot(_z, np.abs(np.asarray(_grad)) + 1e-30, label=_name)
    axes_sq[0].set(xlabel="latent z", ylabel="physical value", title="Squash into (0, 10)")
    axes_sq[1].set(
        xlabel="latent z",
        ylabel="|d(physical)/dz|",
        yscale="log",
        ylim=(1e-12, 1e2),
        title="Surviving gradient",
    )
    axes_sq[1].legend(fontsize=8)
    fig_sq.tight_layout()
    fig_sq.savefig("examples/hybrid_ode/figures/squash_survival.png")
    plt.close(fig_sq)

    # Choosing the axis: warps
    #
    # The rates here run from 0.0063 to 0.276. A linear box over
    # (1e-3, 1) has midpoint 0.5, so every rate in the data sits in the
    # bottom 3% of the range, where the sigmoid is steepest and only large
    # negative latents reach. A warp reparameterises the physical axis
    # before normalising: under warp="log10" the midpoint is 1e-2 and the
    # data covers the middle of the box. Warp and squash are independent.
    _linear = BoundScaler(bounds=((1e-3, 1.0),), transform="sigmoid")
    _log10 = BoundScaler(bounds=((1e-3, 1.0),), transform="sigmoid", warp="log10")
    _z = jnp.linspace(-6.0, 6.0, 400)[:, None]

    fig_warp, ax_warp = plt.subplots(figsize=(7, 3.4))
    ax_warp.plot(np.asarray(_z[:, 0]), np.asarray(_linear.from_latent(_z))[:, 0], label="linear")
    ax_warp.plot(np.asarray(_z[:, 0]), np.asarray(_log10.from_latent(_z))[:, 0], label="log10")
    ax_warp.axhspan(0.0063, 0.276, color="0.85", zorder=0, label="rates in the data")
    ax_warp.set(
        xlabel="latent z",
        ylabel="k",
        yscale="log",
        title="Same box (1e-3, 1), two warps. Latent 0 maps to 0.5 or to 0.01.",
    )
    ax_warp.legend(fontsize=8)
    fig_warp.tight_layout()
    fig_warp.savefig("examples/hybrid_ode/figures/warp_comparison.png")
    plt.close(fig_warp)

    # Registering a warp of your own
    #
    # The residual's output box straddles zero, so log10 is unusable, and
    # a linear box spends resolution evenly, including on large corrections
    # that should never happen if the mechanistic part is any good. What is
    # wanted is an axis linear near zero and logarithmic in the tails, so
    # register one: warps and squashes live in name-keyed registries.
    #
    #     forward(x) = sign(x) log(1 + |x| / eps)
    #
    # forward(0) = 0, so the box midpoint stays at zero and a fresh residual
    # network starts near no correction. A scaler stores its warp by name,
    # which keeps a saved model to a JSON sidecar; a custom warp must be
    # registered before a model referencing it can load.
    SYMLOG_EPS = 0.25

    register_warp(
        "symlog",
        Warp(
            forward=lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x) / SYMLOG_EPS),
            inverse=lambda w: jnp.sign(w) * SYMLOG_EPS * jnp.expm1(jnp.abs(w)),
            requires_positive=False,
        ),
    )
    print("registered warp 'symlog'")

    # 4. The model, and where each network sits
    # -----------------------------------------------------
    #
    # Two gaps, two networks, on opposite sides of the solver.
    #
    #     rate_net:     T -> k             once per experiment, off the tape
    #     residual_net: y -> correction    once per solver step, on the tape
    #
    # A covariate does not change during a trajectory, so anything
    # depending only on covariates is computed before the solve and closed
    # over as a constant. The library is never told which is which; both
    # are ordinary calls placed by writing the code.
    #
    # `BoundedPredictor` bundles each network with its two scalers, and
    # `input_keys` says how the covariate dict becomes a vector. That
    # declared order travels with the saved model.
    TEMPERATURE_BOUNDS = ((270.0, 350.0),)
    K_BOUNDS = ((1e-3, 1.0),)
    STATE_BOUNDS = ((-2.0, 2.0), (-2.0, 2.0))
    RESIDUAL_BOUNDS = ((-3.0, 3.0), (-3.0, 3.0))

    rate_net = BoundedPredictor(
        input_keys=("temperature",),
        in_scaler=BoundScaler(bounds=TEMPERATURE_BOUNDS, transform="sigmoid"),
        inner=MLPPredictor(
            in_size=1,
            out_size=1,
            width_size=16,
            depth=2,
            activation_name="softplus",
            key=k_rate_init,
        ),
        # log10, because the true rates span 1.6 decades.
        out_scaler=BoundScaler(bounds=K_BOUNDS, transform="sigmoid", warp="log10"),
    )

    residual_net = BoundedPredictor(
        input_keys=("y1", "y2"),
        in_scaler=BoundScaler(bounds=STATE_BOUNDS, transform="sigmoid"),
        out_scaler=BoundScaler(bounds=RESIDUAL_BOUNDS, transform="softsign", warp="symlog"),
        inner=MLPPredictor(
            in_size=2,
            out_size=2,
            width_size=32,
            depth=2,
            activation_name="softplus",
            key=k_res_init,
        ),
    )

    # The convention for the trainable object is a tuple. Any pytree the
    # library can walk is accepted; it never inspects the container, which
    # is why dropping the residual later is a one-line change.
    predictors = (rate_net, residual_net)
    print(
        f"rate_net expects {rate_net.input_keys}, "
        f"residual_net expects {residual_net.input_keys}"
    )

    # simulate_fn: the one function you always write yourself. Its contract
    # is fixed:
    #
    #     simulate_fn(predictors, ts, covariates, y0, solver) -> [T, S]
    #
    # Everything inside is yours; the library vectorises, compiles and
    # differentiates this call. The two placements are visible in the first
    # four lines. `adjoint` picks how gradients come back through the solve;
    # RecursiveCheckpointAdjoint trades recomputation for O(log n) storage.
    def simulate_fn(predictors_, ts, covariates, y0, solver):
        rate_net_ = predictors_[0]
        residual_net_ = predictors_[1] if len(predictors_) > 1 else None

        # Outside the solve. One evaluation, closed over as a constant.
        k = rate_net_(covariates).reshape(())
        rotation = jnp.array([[-k, OMEGA_TRUE], [-OMEGA_TRUE, -k]])

        def vector_field(t, y, args):
            mechanistic = rotation @ y
            if residual_net_ is None:
                return mechanistic
            # Inside the solve. One evaluation per solver step.
            return mechanistic + residual_net_(y)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=solver.stepsize_controller(),
            max_steps=solver.max_steps,
            adjoint=solver.adjoint,
        )
        return jnp.asarray(sol.ys)

    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=1e-6,
        max_steps=4096,
        dt0=0.1,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    print(solver)

    # 5. Training
    # -----------------------------------------------------------
    #
    # A two-phase curriculum: fit the first part of every trajectory, then
    # all of it (`length_schedule`), with a lower learning rate in phase
    # two and a fresh minimum for `restore_best` at the phase boundary.
    #
    # The saturation penalty is charged on the latent, not the physical
    # output, and evaluated at the measured points (plus any user-supplied
    # penalty-only points) rather than along the trajectories. The residual
    # network reads the ODE state, so the dataset resolves no measured
    # points for it; both leaves are covered with a warp-uniform box sweep
    # (`box_grid`) -- the collocation-as-extension recipe.
    # `penalty_weight` is a length-1 tuple and broadcasts across both
    # phases.
    #
    # In the original interactive version these two values were UI controls
    # (a slider and a dropdown); a script fixes them at the defaults.
    steps = 400
    penalty_weight = 1e-3

    config = OptaxTrainingConfig(
        steps=(steps // 2, steps),
        lr=(5e-3, 1e-3),
        optimizer=("adamw", "adamw"),
        reset_optimiser_state=(False, False),
        length_schedule=(0.4, 1.0),
        penalty_weight=(penalty_weight,),
        penalty_points=tuple(box_grid(leaf.in_scaler, n_per_dim=7) for leaf in predictors),
        loss="mse",
        verbose=False,
    )
    history, trained = train_with_optax(
        predictors,
        irr_dataset,
        config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        key=k_train,
    )
    print(f"{len(history)} steps, final data loss {history[-1]:.5f}")

    fig_loss, ax_loss = plt.subplots(figsize=(7, 3.2))
    ax_loss.plot(history, lw=1.0)
    ax_loss.axvline(steps // 2, color="0.6", ls="--", lw=0.8)
    ax_loss.set(
        xlabel="step",
        ylabel="data loss (masked MSE)",
        yscale="log",
        title="The step at the dashed line is the horizon widening, not the fit degrading",
    )
    fig_loss.tight_layout()
    fig_loss.savefig("examples/hybrid_ode/figures/loss_curve.png")
    plt.close(fig_loss)

    # `loss_history` from the Optax trainer is the raw per-step data loss
    # and can go up, as it does at the phase boundary. The Evosax trainer
    # returns best-so-far, which cannot. The penalty is excluded from this
    # series, so runs with different penalty weights stay comparable.
    print("saturation penalty at the end of the run, by leaf:")
    for _name, _leaf in zip(("rate_net", "residual_net"), trained, strict=True):
        print(f"  {_name:13s} {float(bound_penalty((_leaf,), (box_grid(_leaf.in_scaler),))):.4e}")

    # With the default settings the residual network reads exactly zero and
    # the rate network reads a small non-zero value. That split is the
    # penalty working rather than a problem: `rate_net` is declared valid
    # over 270 to 350 K, the data only reaches 340 K, and the fitted
    # network extrapolates hard enough at the warm end to press against the
    # top of its k box. Set the weight to zero to see what the term held
    # back.

    # 6. Did it recover the physics?
    # ---------------------------------------------------
    #
    # `rate_net` never sees k directly. It sees trajectories and a
    # temperature label. If the fitted k(T) tracks the Arrhenius law across
    # the levels, the covariate dependence was genuinely recovered rather
    # than memorised per experiment.
    fitted_rates = []
    print("  temperature   true k     fitted k    ratio")
    for _t in TEMPERATURES:
        _truth = float(true_k(_t))
        _fitted = float(trained[0]({"temperature": jnp.asarray(_t)}).reshape(()))
        fitted_rates.append(_fitted)
        print(f"    {_t:6.1f}    {_truth:.5f}    {_fitted:.5f}    {_fitted / _truth:5.2f}")

    _dense = np.linspace(270.0, 350.0, 120)
    _curve = [float(trained[0]({"temperature": jnp.asarray(t)}).reshape(())) for t in _dense]

    fig_rate, ax_rate = plt.subplots(figsize=(7, 3.6))
    ax_rate.plot(_dense, [float(true_k(t)) for t in _dense], "k--", lw=1.2, label="true Arrhenius")
    ax_rate.plot(_dense, _curve, lw=1.6, label="fitted rate_net")
    ax_rate.plot(TEMPERATURES, fitted_rates, "o", ms=6, label="temperatures in the data")
    ax_rate.axvspan(270.0, 280.0, color="0.9", zorder=0)
    ax_rate.axvspan(340.0, 350.0, color="0.9", zorder=0, label="declared box, no data")
    ax_rate.set(xlabel="temperature (K)", ylabel="k", yscale="log", title="Recovered rate law")
    ax_rate.legend(fontsize=8)
    fig_rate.tight_layout()
    fig_rate.savefig("examples/hybrid_ode/figures/recovered_rate_law.png")
    plt.close(fig_rate)

    # The network inside the solver, compared against C y^3 on a grid over
    # the region the trajectories occupy. The residual is only identifiable
    # where data went.
    _axis = jnp.linspace(-0.8, 1.0, 9)
    _grid = jnp.stack(jnp.meshgrid(_axis, _axis, indexing="ij"), axis=-1).reshape(-1, 2)
    _truth = (COUPLING @ (_grid**3).T).T
    _fitted = jnp.stack([trained[1](point) for point in _grid])
    _rms_error = float(jnp.sqrt(jnp.mean((_fitted - _truth) ** 2)))
    _rms_truth = float(jnp.sqrt(jnp.mean(_truth**2)))
    print(f"residual RMS error {_rms_error:.4f} against a true RMS of {_rms_truth:.4f}")
    print(f"relative {_rms_error / _rms_truth:.1%}")

    fig_res, axes_res = plt.subplots(1, 2, figsize=(9, 3.6))
    for _d, _ax in enumerate(axes_res):
        _ax.plot(np.asarray(_truth[:, _d]), np.asarray(_fitted[:, _d]), "o", ms=4, alpha=0.7)
        _lim = float(jnp.max(jnp.abs(_truth[:, _d]))) * 1.15
        _ax.plot([-_lim, _lim], [-_lim, _lim], "k--", lw=0.8)
        _ax.set(xlabel="true correction", ylabel="fitted", title=f"component {_d + 1}")
    fig_res.suptitle("Residual network against the cubic coupling it never saw")
    fig_res.tight_layout()
    fig_res.savefig("examples/hybrid_ode/figures/residual_vs_cubic.png")
    plt.close(fig_res)

    irr_predictions = predict_dataset(
        trained,
        irr_dataset,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
    )
    print(f"one array per bucket: {[p.shape for p in irr_predictions]}")

    _bp = irr_dataset.bucket_payloads[1]
    _pred = np.asarray(irr_predictions[1])
    _obs = np.asarray(_bp.y_observed)
    _mask = np.asarray(_bp.mask)
    _ts = np.asarray(_bp.ts)

    fig_traj, axes_traj = plt.subplots(2, 3, figsize=(11, 5), sharex="col")
    for _col in range(3):
        for _row, _channel in enumerate(("y1", "y2")):
            _ax = axes_traj[_row, _col]
            _sel = _mask[_col, :, _row]
            _ax.plot(_ts[_col][_sel], _obs[_col, _sel, _row], "o", ms=5, label="observed")
            _ax.plot(_ts[_col], _pred[_col, :, _row], lw=1.4, label="hybrid")
            _ax.set_ylabel(_channel if _col == 0 else "")
            if _row == 1:
                _ax.set_xlabel("t")
        axes_traj[0, _col].set_title(f"experiment {_col}")
    axes_traj[0, 0].legend(fontsize=8)
    fig_traj.suptitle("Hybrid fit on irregular data. Only masked-in points are scored.")
    fig_traj.tight_layout()
    fig_traj.savefig("examples/hybrid_ode/figures/hybrid_trajectories.png")
    plt.close(fig_traj)

    # 7. What the structure bought, and what the layout cost
    # -------------------------------------------------------------
    #
    # Two more fits on the same model code. "Mechanistic only" drops the
    # residual from the tuple and nothing else changes. "Rectangular data"
    # runs the full hybrid model on the one-bucket dataset from section 2.
    # Dropping the residual is a one-line change: the tuple gets shorter.
    # simulate_fn already handles a length-one tuple. The penalty points
    # are positional per leaf, so the mechanistic config rebuilds them for
    # the single-leaf tuple.
    from dataclasses import replace

    mech_config = replace(
        config,
        penalty_points=tuple(box_grid(leaf.in_scaler, n_per_dim=7) for leaf in (predictors[0],)),
    )
    _mech_history, mech_trained = train_with_optax(
        (predictors[0],),
        irr_dataset,
        mech_config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        key=k_train,
    )
    print(f"mechanistic-only, irregular: final data loss {_mech_history[-1]:.5f}")

    _rect_history, rect_trained = train_with_optax(
        predictors,
        rect_dataset,
        config,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
        key=k_train,
    )
    print(f"hybrid, rectangular:         final data loss {_rect_history[-1]:.5f}")

    mech_predictions = predict_dataset(
        mech_trained,
        irr_dataset,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
    )
    rect_predictions = predict_dataset(
        rect_trained,
        rect_dataset,
        simulate_fn=simulate_fn,
        state_to_output=state_to_output,
        solver=solver,
    )

    print(f"{'model':38s} {'R2 y1':>8s} {'R2 y2':>8s}")
    for _label, _preds, _ds in (
        ("hybrid, irregular (3 buckets)", irr_predictions, irr_dataset),
        ("hybrid, rectangular (1 bucket)", rect_predictions, rect_dataset),
        ("mechanistic only, irregular", mech_predictions, irr_dataset),
    ):
        # R^2 per channel over masked-in cells only.
        _r2 = []
        for _d in range(_ds.bucket_payloads[0].y_observed.shape[-1]):
            _obs, _pred = [], []
            for _arr, _bp in zip(_preds, _ds.bucket_payloads, strict=True):
                _sel = np.asarray(_bp.mask[:, :, _d])
                _obs.append(np.asarray(_bp.y_observed[:, :, _d])[_sel])
                _pred.append(np.asarray(_arr[:, :, _d])[_sel])
            _obs, _pred = np.concatenate(_obs), np.concatenate(_pred)
            _r2.append(1.0 - np.sum((_obs - _pred) ** 2) / np.sum((_obs - _obs.mean()) ** 2))
        print(f"{_label:38s} {_r2[0]:8.3f} {_r2[1]:8.3f}")

    print("recovered k, hybrid against mechanistic-only")
    print(f"  {'T (K)':>7s} {'true':>9s} {'hybrid':>9s} {'mech-only':>10s}")
    for _t in TEMPERATURES:
        _truth = float(true_k(_t))
        _hyb = float(trained[0]({"temperature": jnp.asarray(_t)}).reshape(()))
        _mech = float(mech_trained[0]({"temperature": jnp.asarray(_t)}).reshape(()))
        print(f"  {_t:7.1f} {_truth:9.5f} {_hyb:9.5f} {_mech:10.5f}")

    # Reading the two tables
    # -------------------------------------------------------------
    #
    # The fit numbers say the expected thing: a model that keeps its known
    # structure and corrects it beats one that keeps the structure and
    # cannot. Both data layouts reach the same trajectory accuracy; their
    # recovered rate laws do differ at the cold end, which is about which
    # trajectories were sampled, not the machinery.
    #
    # The rate table matters more. Removing the residual costs more than
    # accuracy: the rate network is then the only flexible thing left, so
    # it absorbs the missing cubic term into k and the recovered rate law
    # comes out badly wrong, worst at the cold end. An unmodelled term does
    # not stay in its own residual; it contaminates whichever parameter is
    # flexible enough to absorb it, usually the one you built the
    # experiment to measure.
    #
    # Make the residual *more* expressive, swapping `MLPPredictor` for
    # `KANPredictor`, and it fits the trajectories slightly better while
    # recovering k noticeably worse, because it can absorb part of the
    # damping as well. If a fitted parameter is what you came for, the
    # residual wants to be the least expressive thing that closes the gap.
    #
    # Where to go next:
    # - `train_hybrid_ode.py` in this directory is the script version.
    # - `examples/crystallisation/notebook.py` uses the same machinery on
    #   real experimental data with a population balance.
    # - `examples/batch_reactor/notebook.py` shows a two-stage fit that
    #   starts with a global search before switching to gradients.


if __name__ == "__main__":
    main()