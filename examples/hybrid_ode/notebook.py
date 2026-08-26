"""Hybrid ODE modelling with hybridmodels, walked through end to end.

Keeps a mechanistic model's known structure, learns the two parts of it
that have no formula, and holds every physical quantity inside a declared
range while doing so. Runs on data where each measured channel has its own
timestamps.

Self-contained: the data generator, the physics and the models are defined
here rather than imported, so it reads top to bottom. The script version is
``train_hybrid_ode.py`` in the same directory.

Run interactively: ``uv run marimo edit examples/hybrid_ode/notebook.py``
Run as script:     ``uv run python examples/hybrid_ode/notebook.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Fitting a hybrid ODE model

    ## The situation this library is for

    You have measurements of something that changes over time, and a
    differential equation you partly believe:

    $$
    \frac{dy}{dt} = f(y, t; \theta)
    $$

    Some of $f$ comes from conservation laws or a reaction mechanism you
    trust. Some of it does not. A rate constant varies with temperature
    in a way nobody has written down. A term is missing altogether. You
    want to learn the missing parts from data without throwing away the
    parts you already know.

    That is a **hybrid model**: mechanistic structure kept in closed
    form, unknown pieces replaced by trainable networks. It is worth the
    trouble because a model that keeps its structure extrapolates, and
    because the parameters inside that structure keep their physical
    meaning after fitting. The last section of this notebook shows what
    happens to a fitted rate constant when you skip the correction and
    let the mechanistic parameter absorb the error instead.

    ## What this library provides

    Three things, and they are all awkward to do by hand.

    **Physical quantities stay inside their ranges.** A rate constant
    cannot be negative. A mole fraction lives in $[0, 1]$. An optimiser
    that does not know this proposes values that make the solver
    diverge. Clipping the output looks like the fix and is not: a clip
    has zero derivative outside the range, so it destroys the gradient
    that would pull the parameter back, exactly when that gradient is
    needed. This library instead reparameterises, so an out-of-range
    value cannot be represented at all.

    **Irregular measurements are handled without padding or
    interpolation.** Two instruments sampling one experiment rarely
    agree on when. Solvers want rectangular arrays. The library builds
    a per-experiment union of timestamps, marks the holes with a mask,
    and groups experiments by length so each group is one compiled
    solve.

    **Trainable pieces compose with mechanistic terms.** A network can
    sit above the solver, or inside the vector field, or both, and the
    training loop does not need to be told which.

    ## What this library is not

    It is not a neural ODE library. A trainable network inside a vector
    field is one of the things you can build here, and
    [diffrax](https://docs.kidger.site/diffrax/) and
    [Equinox](https://docs.kidger.site/equinox/) already document that
    technique well. This notebook uses it in one line and does not
    explain it.

    Assumed: Python, and some experience fitting models to data. Not
    assumed: JAX, Equinox, diffrax, or the chemistry.
    """)
    return


@app.cell
def _imports():
    import diffrax
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from hybridmodels import (
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
    from hybridmodels.penalties import bound_penalty, collocation_grids
    from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax

    return (
        BoundScaler,
        BoundedPredictor,
        ChannelObs,
        MLPPredictor,
        OptaxTrainingConfig,
        SolverConfig,
        Warp,
        bound_penalty,
        collocation_grids,
        diffrax,
        jax,
        jnp,
        jr,
        make_dataset,
        make_experiment,
        mo,
        np,
        plt,
        predict_dataset,
        register_warp,
        train_with_optax,
    )


@app.cell(hide_code=True)
def _system_md(mo):
    mo.md(r"""
    ---

    # 1. The system, and the two gaps in the model

    A two-dimensional system: a rotation at fixed frequency $\omega$,
    damped at rate $k$, with a cubic coupling on top.

    $$
    \frac{dy}{dt} =
    \begin{bmatrix} -k & \omega \\ -\omega & -k \end{bmatrix} y
    \;+\; C y^{3},
    \qquad \omega = 1
    $$

    The model will keep the rotation and treat $\omega$ as known. Two
    things it will not know:

    **The damping rate depends on temperature.** Each experiment runs at
    one of six temperature levels, and the true rate follows an
    Arrhenius law:

    $$
    k(T) = k_{\mathrm{ref}} \exp\!\left(
      -\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\mathrm{ref}}}\right)
    \right)
    $$

    Across the six levels this makes $k$ run from 0.0063 to 0.276, a
    factor of 44. Remember that number; it is why the model needs a
    logarithmic axis for $k$ in section 3.

    **The cubic coupling is missing entirely.** $C y^3$ has no
    counterpart in the model, and a network has to reproduce its effect
    from trajectories alone.

    In a real problem you would not know either of these. Here they are
    synthesised so there is something to check the fit against.
    """)
    return


@app.cell
def _physics(diffrax, jnp):
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
    return (
        CHANNELS,
        COUPLING,
        OMEGA_TRUE,
        TEMPERATURES,
        solve_reference,
        true_field,
        true_k,
    )


@app.cell(hide_code=True)
def _data_md(mo):
    mo.md(r"""
    ---

    # 2. Getting data into the library

    ## The containers

    Data enters as one `Experiment` per run. An `Experiment` holds:

    - **covariates**: constants describing that run. Here, the
      temperature it was held at.
    - **channels**: one `ChannelObs` per measured quantity, each with
      its own timestamps, values and variances. Channels are allowed to
      disagree about when they were measured.
    - **y0_fn**: a function returning the full state at $t = 0$. The
      full state can be larger than what you measure. Here both
      components are measured, so it reads them off the first sample.

    `make_dataset` then does the bookkeeping this library is built
    around. For each experiment it takes the **union** of its channels'
    timestamps, writes each channel's values into the rows of that union
    where it was actually measured, and records a boolean **mask**
    marking which cells are real. Experiments whose union has the same
    length are stacked into one **bucket**.

    Buckets are what the solver sees. One bucket is one vectorised,
    compiled solve. JAX compiles a function once per distinct input
    shape, so the number of distinct union lengths in your data is the
    number of times training compiles. Grouping by length avoids padding
    everything out to the longest experiment, and nothing is
    interpolated to fill a hole.

    ## Two sampling layouts

    The same physics, sampled two ways, so the difference the library
    actually makes is visible.

    **Rectangular.** Every experiment on the same 20-point grid, both
    channels measured every time.

    **Irregular.** Every experiment gets its own end time in $[6, 9]$,
    its own randomly drawn sample times, and each channel thinned
    independently to 8 or 12 samples. Two instruments sampling at their
    own rates is the normal situation in a laboratory.

    Since the union length is `n1 + n2 - 1` (the shared $t = 0$ counts
    once) and each count is 8 or 12, there are three possible lengths:
    15, 19 and 23.

    $t = 0$ stays in every channel in both layouts, so `y0_fn` can read
    the initial state off the first observation. Relaxing that needs an
    encoder, which is the latent-ODE problem rather than this one.
    """)
    return


@app.cell
def _keys(jr):
    # JAX has no global RNG: every random draw takes an explicit key, and
    # splitting one root key is how a run stays reproducible. Split the
    # same way train_hybrid_ode.py does, so this notebook reproduces the
    # numbers quoted on the documentation page.
    k_data, k_rate_init, k_res_init, k_train = jr.split(jr.PRNGKey(0), 4)
    return k_data, k_rate_init, k_res_init, k_train


@app.cell
def _builders(CHANNELS, ChannelObs, jnp, make_dataset, make_experiment):
    NOISE_STD = 0.03

    def y0_from_first_samples(covariates, channels):
        """Initial state, read off the t=0 sample of each channel."""
        return jnp.stack([channels[name].values[0] for name in CHANNELS])

    def build_experiment(ts_per_channel, values_per_channel, temperature, exp_id):
        channels = {
            name: ChannelObs(ts=ts, values=vals, variance=jnp.full(ts.shape, NOISE_STD**2))
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
            state_to_output=state_to_output,
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

    return NOISE_STD, build_dataset, build_experiment, describe_buckets


@app.cell
def _rect_data(
    NOISE_STD,
    TEMPERATURES,
    build_dataset,
    build_experiment,
    describe_buckets,
    jax,
    jnp,
    jr,
    k_data,
    solve_reference,
    true_field,
    true_k,
):
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
    return N_EXPERIMENTS, rect_dataset, rect_observed, rect_ts


@app.cell
def _irr_data(
    CHANNELS,
    NOISE_STD,
    N_EXPERIMENTS,
    TEMPERATURES,
    build_dataset,
    build_experiment,
    describe_buckets,
    jnp,
    jr,
    k_data,
    solve_reference,
    true_field,
    true_k,
):
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
    return irr_dataset, irr_experiments


@app.cell(hide_code=True)
def _mask_md(mo):
    mo.md(r"""
    One bucket against three, and a mask that goes from completely full
    to about half full. The figure below makes the second case concrete:
    black cells are real measurements, white cells are positions on the
    union axis where *the other* channel was measured and this one was
    not. Nothing is interpolated to fill them; the loss skips them.
    """)
    return


@app.cell
def _mask_plot(irr_dataset, np, plt, rect_dataset):
    fig_mask, axes_mask = plt.subplots(4, 1, figsize=(9, 5.2))
    _panels = [
        (rect_dataset.bucket_payloads[0], 0, "rectangular, y1"),
        (rect_dataset.bucket_payloads[0], 1, "rectangular, y2"),
        (irr_dataset.bucket_payloads[1], 0, "irregular bucket 1, y1"),
        (irr_dataset.bucket_payloads[1], 1, "irregular bucket 1, y2"),
    ]
    for _ax, (_bp, _d, _label) in zip(axes_mask, _panels, strict=True):
        _ax.imshow(
            np.asarray(_bp.mask[:, :, _d]), aspect="auto", cmap="binary", interpolation="nearest"
        )
        _ax.set_ylabel(_label, fontsize=7)
        _ax.set_xticks([])
        _ax.set_yticks([])
    axes_mask[-1].set_xlabel("position on the union timestamp axis")
    fig_mask.suptitle("Mask: black is measured, white is a hole")
    fig_mask.tight_layout()
    fig_mask
    return


@app.cell
def _raw_plot(irr_experiments, plt, rect_observed, rect_ts):
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
    fig_raw
    return


@app.cell(hide_code=True)
def _bounds_md(mo):
    mo.md(r"""
    ---

    # 3. Keeping physical quantities in range

    ## Why not just clip

    Suppose you constrain a rate constant to $(10^{-3}, 1)$ by clipping
    whatever the network emits. The forward pass is now correct. The
    backward pass is broken: a clip has derivative zero outside the
    range, so once the network proposes 1.5 the gradient telling it to
    come down is multiplied by zero. The parameter is stuck at its
    bound, and nothing raises.

    `BoundScaler` reparameterises instead. It composes with a network
    like this:

    ```
    physical input -> to_latent -> network -> from_latent -> physical output
    ```

    The network in the middle sees normalised, unbounded numbers, has no
    idea a bound exists, and never has to clamp itself. `from_latent`
    maps an unbounded $z$ into $[\ell, u]$ by squashing:

    $$
    x = \ell + (u - \ell)\,\sigma(z / T)
    $$

    Since $\sigma$ lands in $(0, 1)$, an out-of-range value is not
    representable. There is nothing to clip and nothing to check.

    `to_latent` is the inverse, used on inputs. Its logit has poles at
    the ends of the box, so a value at or past a bound would give
    infinity. Clipping there would reintroduce the original disease, so
    the library continues the logit linearly outside a narrow band
    instead. Values stay finite, the gradient stays non-zero and points
    the right way, and the join is smooth enough that an adaptive solver
    notices nothing.

    ## The cost, and the choice of squash

    Reparameterising is not free. The derivative of `from_latent`
    carries a factor $\sigma'(z/T)$, and for a logistic sigmoid in
    float32 that factor **underflows to exactly zero** past
    $|z/T| \approx 15$. A network pushed hard against a bound stops
    receiving any signal to come back, permanently.

    The library therefore offers a choice of squash, by name:

    | name | tail of $\lvert du/dz \rvert$ | dead at |
    |---|---|---|
    | `sigmoid` | $e^{-\lvert z \rvert}$ | $z \approx 17$ |
    | `algebraic` | $\lvert z \rvert^{-3}/2$ | $z \approx 3 \times 10^{3}$ |
    | `softsign` | $\lvert z \rvert^{-2}/2$ | $z \approx 10^{7}$ |

    Polynomial decay does not make saturation free. Escaping from
    $z = 100$ under a $c/z^2$ gradient takes $10^6$ times as long as
    from $z = 1$. It turns an impossible recovery into a slow one.

    The left panel below is what the model can express, the right panel
    is whether it can still learn once it is out there. Note the log
    scale.
    """)
    return


@app.cell
def _squash_plot(BoundScaler, jax, jnp, np, plt):
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
    fig_sq
    return


@app.cell(hide_code=True)
def _warp_md(mo):
    mo.md(r"""
    ## Choosing the axis: warps

    The rates in this problem run from 0.0063 to 0.276. Give the rate
    network an output box of $(10^{-3}, 1)$ with the default linear
    normalisation and the midpoint of that box is $0.5$. Every rate in
    the data then sits in the bottom 3% of the range, where the sigmoid
    is steepest and where the network must emit large negative latents
    to reach anything at all.

    A **warp** reparameterises the physical axis before normalising. It
    changes what "halfway between the bounds" means without changing
    which physical values are reachable. With `warp="log10"` the
    midpoint becomes $10^{-2}$ and the data covers the middle of the
    box.

    Warp and squash are independent. The warp decides how the box is
    laid out; the squash decides how the model behaves near an edge.
    Shipped warps are `linear`, `log` and `log10`.
    """)
    return


@app.cell
def _warp_plot(BoundScaler, jnp, np, plt):
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
    fig_warp
    return


@app.cell(hide_code=True)
def _symlog_md(mo):
    mo.md(r"""
    ## Registering a warp of your own

    The residual's output box straddles zero, so `log10` is unusable. A
    linear box works but spends resolution evenly, including on large
    corrections that should never happen: if the mechanistic part is any
    good, the residual is small.

    What is wanted is an axis linear near zero and logarithmic in the
    tails. That is not shipped, so register it. Warps and squashes both
    live in name-keyed registries and adding one is a function call.

    $$
    \mathrm{forward}(x) = \operatorname{sign}(x)\,
      \log\!\left(1 + \frac{|x|}{\varepsilon}\right)
    $$

    `forward(0) = 0`, so the box midpoint stays at zero and a freshly
    initialised residual network produces a correction near zero rather
    than at some arbitrary interior point.

    A scaler stores its warp **by name**, which is what keeps a saved
    model a small JSON sidecar plus an array file. The consequence: a
    custom warp must be registered before a model referencing it can be
    loaded. `register_bound_transform` is the same idea for squashes.
    """)
    return


@app.cell
def _register_symlog(Warp, jnp, register_warp):
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
    return


@app.cell(hide_code=True)
def _model_md(mo):
    mo.md(r"""
    ---

    # 4. The model, and where each network sits

    Two gaps, two networks, on opposite sides of the solver.

    | | learns | called | on the solver tape |
    |---|---|---|---|
    | `rate_net` | $T \mapsto k$ | once per experiment | no |
    | `residual_net` | $y \mapsto \text{correction}$ | once per solver step | yes |

    A covariate does not change during a trajectory, so anything
    depending only on covariates can be computed **before** the solve
    and closed over as a constant. Its cost is then independent of how
    many solver steps run, and it never appears on the tape the backward
    pass walks. A term depending on the state has no choice but to run
    inside. (A trainable network inside a vector field is a neural ODE.
    diffrax and Equinox document that technique; this notebook uses it
    and moves on.)

    The library is not told which is which. Both are ordinary calls; you
    place them by writing the code.

    `BoundedPredictor` bundles each network with its two scalers.
    `input_keys` says how the covariate dict becomes a vector, in a
    declared order that travels with the saved model, so a reload site
    knows what the predictor expects without consulting the code that
    built it.
    """)
    return


@app.cell
def _models(
    BoundScaler,
    BoundedPredictor,
    MLPPredictor,
    k_rate_init,
    k_res_init,
):
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
        inner=MLPPredictor(
            in_size=2,
            out_size=2,
            width_size=32,
            depth=2,
            activation_name="softplus",
            key=k_res_init,
        ),
        # softsign, because a correction term visits the edge of its box
        # early in training and sigmoid's gradient would be gone by then.
        out_scaler=BoundScaler(bounds=RESIDUAL_BOUNDS, transform="softsign", warp="symlog"),
    )

    # The convention for the trainable object is a tuple. Any pytree the
    # library can walk is accepted: a dict, a NamedTuple, a bare module.
    # It never inspects the container, which is why dropping the residual
    # later is a one-line change.
    predictors = (rate_net, residual_net)
    print(f"rate_net expects {rate_net.input_keys}, residual_net expects {residual_net.input_keys}")
    return (predictors,)


@app.cell(hide_code=True)
def _simulate_md(mo):
    mo.md(r"""
    ## simulate_fn

    The one function you always write yourself. Its contract is fixed:

    ```
    simulate_fn(predictors, ts, covariates, y0, solver) -> [T, S]
    ```

    Given the trainable object, one experiment's timestamps, its
    covariates and its initial state, return the full state at every
    timestamp. Everything inside is yours. The library never inspects
    the physics; it vectorises, compiles and differentiates this call.

    The two placements are visible in the first four lines.

    `SolverConfig` collects the numerical choices so they can be saved
    with the model. `adjoint` picks how gradients are taken through the
    solve. `DirectAdjoint`, the library default, stores the whole
    forward trajectory. With a network in the vector field that is
    usually the memory bottleneck, so this example uses
    `RecursiveCheckpointAdjoint`, which stores $O(\log n)$ checkpoints
    and recomputes the rest.
    """)
    return


@app.cell
def _simulate(OMEGA_TRUE, diffrax, jnp):
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

    return (simulate_fn,)


@app.cell
def _solver_cfg(SolverConfig, diffrax):
    solver = SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-4,
        atol=1e-6,
        max_steps=4096,
        dt0=0.1,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    print(solver)
    return (solver,)


@app.cell(hide_code=True)
def _train_md(mo):
    mo.md(r"""
    ---

    # 5. Training

    ## Phases

    Fitting a trajectory model over a long window is hard for a reason
    unrelated to the optimiser. Early on the model is wrong, so the
    predicted trajectory leaves the data almost immediately, and the
    loss at late times is dominated by that divergence rather than by
    anything the parameters can fix locally.

    The fix is a curriculum: fit the first part of every trajectory,
    then all of it. Here that is `length_schedule`, one entry per phase.

    ```python
    OptaxTrainingConfig(
        steps=(200, 400),
        lr=(5e-3, 1e-3),
        optimizer=("adamw", "adamw"),
        reset_optimiser_state=(False, False),
        length_schedule=(0.4, 1.0),
    )
    ```

    Every phase-keyed field is a tuple of the same length, spelled out
    rather than broadcast, so a schedule cannot be silently truncated.

    `length_schedule` masks the **loss** to the first fraction of each
    trajectory. It does not shorten the integration, so an early step
    costs the same as a late one. What changes is which residuals the
    optimiser is allowed to see.

    One consequence worth knowing. A phase at 0.4 and a phase at 1.0
    measure different quantities, so their loss values are not
    comparable. `restore_best`, which returns the best model rather than
    the last one, resets its running minimum whenever `length_schedule`
    changes. Without that reset the minimum lands in the shortest phase
    every time and you get back the least-trained point in the run.

    ## The saturation penalty

    Bounds hold by construction, so a violation cannot be represented.
    The failure mode that remains is the opposite one: a network
    **pinned** against a bound, where the squash derivative has decayed
    and the gradient that would pull it back has gone.

    Two details make the penalty work.

    It is charged on the **latent**, not the physical output. A penalty
    written against the physical value would inherit the same
    $\sigma'(z/T)$ factor on its backward pass and die exactly where
    saturation is worst. Reading $|z|/T$ gives a gradient linear in the
    overshoot that never underflows.

    It is evaluated on a **collocation grid** over each predictor's
    declared input box, not along the trajectories. That makes it
    trajectory-blind, which cuts both ways. It reports saturation
    anywhere in the box the model claims to be valid on, including
    regions no training run visited, so it catches extrapolation trouble
    before deployment. It cannot tell you whether one particular solve
    pushed an input out of range.

    `penalty_weight` is a length-1 tuple, which broadcasts across every
    phase. Give it one entry per phase to ramp it.

    Move the controls and everything downstream recomputes.
    """)
    return


@app.cell
def _controls(mo):
    steps = mo.ui.slider(start=100, stop=1200, step=100, value=400, label="final-phase steps")
    penalty_weight = mo.ui.dropdown(
        options={"0 (off)": 0.0, "1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2},
        value="1e-3",
        label="saturation penalty weight",
    )
    mo.vstack([steps, penalty_weight])
    return penalty_weight, steps


@app.cell
def _train_irregular(
    OptaxTrainingConfig,
    irr_dataset,
    k_train,
    penalty_weight,
    predictors,
    simulate_fn,
    solver,
    steps,
    train_with_optax,
):
    config = OptaxTrainingConfig(
        steps=(steps.value // 2, steps.value),
        lr=(5e-3, 1e-3),
        optimizer=("adamw", "adamw"),
        reset_optimiser_state=(False, False),
        length_schedule=(0.4, 1.0),
        penalty_weight=(penalty_weight.value,),
        penalty_grid_points=7,
        loss="mse",
        verbose=False,
    )
    history, trained = train_with_optax(
        predictors,
        irr_dataset,
        config,
        simulate_fn=simulate_fn,
        solver=solver,
        key=k_train,
    )
    print(f"{len(history)} steps, final data loss {history[-1]:.5f}")
    return config, history, trained


@app.cell
def _loss_plot(history, plt, steps):
    fig_loss, ax_loss = plt.subplots(figsize=(7, 3.2))
    ax_loss.plot(history, lw=1.0)
    ax_loss.axvline(steps.value // 2, color="0.6", ls="--", lw=0.8)
    ax_loss.set(
        xlabel="step",
        ylabel="data loss (masked MSE)",
        yscale="log",
        title="The step at the dashed line is the horizon widening, not the fit degrading",
    )
    fig_loss.tight_layout()
    fig_loss
    return


@app.cell(hide_code=True)
def _loss_note(mo):
    mo.md(r"""
    `loss_history` from the Optax trainer is the raw per-step data loss
    and can go up, as it does at the phase boundary. The Evosax trainer
    returns best-so-far, which cannot. Same type, same position in the
    return tuple, different meaning.

    The penalty is excluded from this series. Including it would move
    "best" whenever only the weight changed and would make runs with
    different weights incomparable.
    """)
    return


@app.cell
def _penalty_readout(bound_penalty, collocation_grids, trained):
    print("saturation penalty at the end of the run, by leaf:")
    for _name, _leaf in zip(("rate_net", "residual_net"), trained, strict=True):
        print(f"  {_name:13s} {float(bound_penalty((_leaf,), collocation_grids((_leaf,)))):.4e}")
    return


@app.cell(hide_code=True)
def _penalty_note(mo):
    mo.md(r"""
    With the default settings the residual network reads exactly zero
    and the rate network reads a small non-zero value. That split is the
    penalty working rather than a problem. `rate_net` is declared valid
    over 270 to 350 K, the data only reaches 340 K, and the fitted
    network extrapolates hard enough at the warm end to press against
    the top of its $k$ box. The training loss cannot see that, because
    no experiment is there.

    Set the weight to zero above and re-read the numbers to see what the
    term was holding back.
    """)
    return


@app.cell(hide_code=True)
def _rate_md(mo):
    mo.md(r"""
    ---

    # 6. Did it recover the physics?

    ## The network above the solver

    `rate_net` never sees $k$. It sees trajectories and a temperature
    label. If the fitted $k(T)$ tracks the Arrhenius law across the
    levels, the covariate dependence was genuinely recovered rather than
    memorised per experiment.
    """)
    return


@app.cell
def _rate_table(TEMPERATURES, jnp, trained, true_k):
    fitted_rates = []
    print("  temperature   true k     fitted k    ratio")
    for _t in TEMPERATURES:
        _truth = float(true_k(_t))
        _fitted = float(trained[0]({"temperature": jnp.asarray(_t)}).reshape(()))
        fitted_rates.append(_fitted)
        print(f"    {_t:6.1f}    {_truth:.5f}    {_fitted:.5f}    {_fitted / _truth:5.2f}")
    return (fitted_rates,)


@app.cell
def _rate_plot(TEMPERATURES, fitted_rates, jnp, np, plt, trained, true_k):
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
    fig_rate
    return


@app.cell(hide_code=True)
def _residual_md(mo):
    mo.md(r"""
    ## The network inside the solver

    Compared against $C y^3$ on a grid over the region the trajectories
    occupy. The residual is only identifiable where data went.
    """)
    return


@app.cell
def _residual_check(COUPLING, jnp, np, plt, trained):
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
    fig_res
    return


@app.cell
def _predictions(irr_dataset, predict_dataset, simulate_fn, solver, trained):
    irr_predictions = predict_dataset(trained, irr_dataset, simulate_fn=simulate_fn, solver=solver)
    print(f"one array per bucket: {[p.shape for p in irr_predictions]}")
    return (irr_predictions,)


@app.cell
def _traj_plot(irr_dataset, irr_predictions, np, plt):
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
    fig_traj
    return


@app.cell(hide_code=True)
def _compare_md(mo):
    mo.md(r"""
    ---

    # 7. What the structure bought, and what the layout cost

    Two more fits on the same model code.

    **Mechanistic only.** The residual is dropped from the tuple and
    nothing else changes. Identical data, identical rate network, one
    term deleted.

    **Rectangular data.** The full hybrid model on the one-bucket
    dataset from section 2. Same builders, same `simulate_fn`, same
    config.
    """)
    return


@app.cell
def _extra_fits(
    config,
    irr_dataset,
    k_train,
    predictors,
    rect_dataset,
    simulate_fn,
    solver,
    train_with_optax,
):
    # Dropping the residual is a one-line change: the tuple gets shorter.
    # simulate_fn already handles a length-one tuple.
    _mech_history, mech_trained = train_with_optax(
        (predictors[0],),
        irr_dataset,
        config,
        simulate_fn=simulate_fn,
        solver=solver,
        key=k_train,
    )
    print(f"mechanistic-only, irregular: final data loss {_mech_history[-1]:.5f}")

    _rect_history, rect_trained = train_with_optax(
        predictors,
        rect_dataset,
        config,
        simulate_fn=simulate_fn,
        solver=solver,
        key=k_train,
    )
    print(f"hybrid, rectangular:         final data loss {_rect_history[-1]:.5f}")
    return mech_trained, rect_trained


@app.cell
def _r2_table(
    irr_dataset,
    irr_predictions,
    mech_trained,
    np,
    predict_dataset,
    rect_dataset,
    rect_trained,
    simulate_fn,
    solver,
):
    def r2_per_channel(predictions, dataset):
        """R^2 per channel over masked-in cells only."""
        out = []
        for d in range(dataset.bucket_payloads[0].y_observed.shape[-1]):
            obs, pred = [], []
            for arr, bp in zip(predictions, dataset.bucket_payloads, strict=True):
                sel = np.asarray(bp.mask[:, :, d])
                obs.append(np.asarray(bp.y_observed[:, :, d])[sel])
                pred.append(np.asarray(arr[:, :, d])[sel])
            obs, pred = np.concatenate(obs), np.concatenate(pred)
            out.append(1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))
        return out

    mech_predictions = predict_dataset(
        mech_trained, irr_dataset, simulate_fn=simulate_fn, solver=solver
    )
    rect_predictions = predict_dataset(
        rect_trained, rect_dataset, simulate_fn=simulate_fn, solver=solver
    )

    print(f"{'model':38s} {'R2 y1':>8s} {'R2 y2':>8s}")
    for _label, _preds, _ds in (
        ("hybrid, irregular (3 buckets)", irr_predictions, irr_dataset),
        ("hybrid, rectangular (1 bucket)", rect_predictions, rect_dataset),
        ("mechanistic only, irregular", mech_predictions, irr_dataset),
    ):
        _r2 = r2_per_channel(_preds, _ds)
        print(f"{_label:38s} {_r2[0]:8.3f} {_r2[1]:8.3f}")
    return


@app.cell
def _mech_rate_table(TEMPERATURES, jnp, mech_trained, trained, true_k):
    print("recovered k, hybrid against mechanistic-only")
    print(f"  {'T (K)':>7s} {'true':>9s} {'hybrid':>9s} {'mech-only':>10s}")
    for _t in TEMPERATURES:
        _truth = float(true_k(_t))
        _hyb = float(trained[0]({"temperature": jnp.asarray(_t)}).reshape(()))
        _mech = float(mech_trained[0]({"temperature": jnp.asarray(_t)}).reshape(()))
        print(f"  {_t:7.1f} {_truth:9.5f} {_hyb:9.5f} {_mech:10.5f}")
    return


@app.cell(hide_code=True)
def _closing_md(mo):
    mo.md(r"""
    ## Reading the two tables

    The fit numbers say the expected thing: a model that keeps its known
    structure and corrects it beats one that keeps the structure and
    cannot correct it.

    The two data layouts reach the same trajectory accuracy, which is
    the point of the bucketing. Half a mask is not a handicap. Their
    recovered rate laws do differ at the cold end, and that is about
    which trajectories were sampled rather than about the machinery: the
    irregular set draws end times up to 9 while the rectangular set
    stops at 8, and a cold experiment barely decays inside either
    window, so an extra unit of time is worth more there than extra
    points are.

    The rate table is the more important one. Removing the residual
    costs more than accuracy. The rate network is now the only flexible
    thing left in the model, so it absorbs the missing cubic term into
    $k$ and the recovered rate law comes out badly wrong. It is worst at
    the cold end, where the true damping is smallest and the cubic term
    is proportionally largest.

    That is the case for hybrid models stated precisely. An unmodelled
    term does not stay politely in its own residual. It contaminates
    whichever parameter is flexible enough to absorb it, which is
    usually the one you built the experiment to measure.

    The same effect runs the other way. Make the residual *more*
    expressive (swap `MLPPredictor` for `KANPredictor` in the two
    builders above, nothing else) and it fits the trajectories slightly
    better while recovering $k$ noticeably worse, because a more
    flexible residual can absorb part of the damping too. If a fitted
    parameter is what you came for, the residual wants to be the least
    expressive thing that closes the gap.

    ## Where to go next

    - `train_hybrid_ode.py` in this directory is the script version,
      with flags for the variants above.
    - `examples/crystallisation/notebook.py` puts the same machinery on
      real experimental data with a population balance.
    - `examples/batch_reactor/notebook.py` shows a two-stage fit that
      starts with a global search before switching to gradients.
    """)
    return


if __name__ == "__main__":
    app.run()
