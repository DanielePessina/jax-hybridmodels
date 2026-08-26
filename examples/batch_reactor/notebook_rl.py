"""Batch reactor, part two: a bounded parameter controller trained by PPO.

Narrative version of ``examples/batch_reactor/train_rl_deactivation.py``. The
catalyst calibrated on the first batch reactor page fouls as the batch runs, and
a reinforcement learning policy recovers the missing activity without ever
differentiating the ODE.

The plumbing (aged data generation, the rollout, the PPO update) is imported
from that script so this file stays readable. Everything that carries the
argument is defined and run in cells here.

Run interactively: ``uv run marimo edit examples/batch_reactor/notebook_rl.py``
Run as script:     ``uv run python examples/batch_reactor/notebook_rl.py``
"""

# ruff: noqa: F722

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(r"""
    # Kinetic parameters as control actions

    ## Where this picks up

    The batch reactor page fitted a rate constant $k(T, \mathrm{pH})$
    for a first-order reaction $A \to B$: a known Arrhenius
    temperature law, plus a small bounded network for the pH
    dependence, which has no first-principles form.

    Those were commissioning runs on **fresh catalyst**. This page
    takes that fitted model, freezes it, and points it at a reactor
    whose catalyst **fouls while the batch runs**. The frozen model
    cannot express that, and it fails badly.

    ## The reframing

    Mowbray, Wu, Rogers, Del Rio-Chanona and Zhang (2023) propose
    treating kinetic parameter estimation as a *control* problem.
    Parameters become actions. A policy reads the current model
    state and emits a bounded parameter value; the mechanistic model
    integrates one interval holding it fixed; the reward is how
    close the result lands to the next measurement.

    $$
    \max_\pi \sum_{t=0}^{T-1} R(x^m_t, p_t, x^m_{t+1}),
    \qquad x^m_{t+1} = f(x^m_t, p_t),
    \qquad p_t = \pi(x^m_t) \in \mathcal{P}
    $$

    Two details matter. The rollout is **free-running**: $\pi$ reads
    the state its own earlier actions produced, not the measurement,
    so errors compound and the trained policy is a model you can run
    forward on a fresh batch. And $\mathcal{P}$ is a **hard
    constraint** on the parameter.

    ## What this page is about

    Two claims, in order of how much they are worth.

    **Bounds by reparameterisation, inside an RL actor.** Continuous
    control normally bounds actions by squashing a Gaussian through
    $\tanh$, which then owes a log-determinant correction in the
    log-probability. Here the policy distribution lives in the
    *latent* space and `BoundScaler` carries the latent into the
    physical box as part of the dynamics. No correction term, no
    clip, and the bound holds structurally.

    **No gradient through the solver.** PPO differentiates the
    policy and the value head. The ODE rollout produces rewards and
    is never differentiated. That is the honest reason to reach for
    RL on a problem whose gradient is perfectly well behaved, and
    this page reports a gradient-trained version of the identical
    model to stay honest about it.
    """)
    return


@app.cell
def _imports():
    import sys
    from pathlib import Path

    import diffrax
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np

    _here = Path(__file__).resolve().parent
    for _path in (_here, _here.parent):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    from _shared import compute_diagnostics

    from hybridmodels import SolverConfig, make_dataset, predict_dataset

    return (
        SolverConfig,
        compute_diagnostics,
        diffrax,
        jax,
        jnp,
        jr,
        make_dataset,
        np,
        plt,
        predict_dataset,
    )


@app.cell
def _load_example():
    # The script is the reference implementation. Importing it keeps this
    # notebook from carrying a second, drifting copy of the same physics.
    import train_rl_deactivation as ex

    return (ex,)


@app.cell(hide_code=True)
def _system_md(mo):
    mo.md(r"""
    ## The system

    Fresh catalyst obeys the truth the earlier page fitted. Aged
    catalyst multiplies it by an activity factor:

    $$
    k(T, \mathrm{pH}, t) = a(t, \mathrm{pH}) \cdot
    k_\mathrm{fresh}(T, \mathrm{pH}),
    \qquad
    a(t, \mathrm{pH}) = \frac{1}{1 + (t / \tau(\mathrm{pH}))^{3}}
    $$

    with $\tau$ shortening at low pH, so acid attacks the catalyst
    sooner.

    The cubic exponent is a deliberate choice. First-order
    deactivation $a(t) = e^{-k_d t}$ would make
    $\mathrm{d}C_A/\mathrm{d}t = -k e^{-k_d t} C_A$ solvable in
    closed form, and one extra trainable scalar would fit it
    exactly, leaving a time-varying policy with nothing to do. A
    cubic Hill curve has a plateau followed by a sharp fall and no
    such shortcut.
    """)
    return


@app.cell
def _activity_plot(ex, jnp, plt):
    _fig, _ax = plt.subplots(figsize=(6.4, 3.8))
    _t = jnp.linspace(0.0, ex.T_MAX, 300)
    for _ph in (4.5, 6.0, 7.5):
        _ax.plot(_t, ex._activity_true(_t, _ph), lw=2, label=f"pH {_ph}")
        # The exponential that agrees with the truth at both ends. It is the
        # best a one-scalar decay law can do, and it misses the plateau.
        _end = float(ex._activity_true(ex.T_MAX, _ph))
        _kd = -float(jnp.log(_end)) / ex.T_MAX
        _ax.plot(_t, jnp.exp(-_kd * _t), lw=1.0, ls=":", color="grey")
    _ax.set_xlabel("time")
    _ax.set_ylabel("catalyst activity")
    _ax.set_title("Truth (solid) against the best endpoint-matched exponential (dotted)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _stall_md(mo):
    mo.md(r"""
    The fouling is severe on purpose. A starting model that is
    already good makes the whole exercise decorative, so the batch
    is made to **stall** at roughly 55% conversion where the
    fresh-catalyst model predicts it runs nearly to completion.
    """)
    return


@app.cell
def _stall_table(ex, jnp, mo):
    _grid = jnp.linspace(0.0, ex.T_MAX, 4001)
    _rows = []
    for _ph in (4.5, 6.0, 7.5):
        _k = float(ex._k_true(25.0, _ph))
        _integral = float(jnp.trapezoid(ex._activity_true(_grid, _ph), _grid))
        _rows.append(
            f"| {_ph} | {float(ex._activity_true(ex.T_MAX, _ph)):.4f} "
            f"| {float(jnp.exp(-_k * _integral)):.3f} "
            f"| {float(jnp.exp(-_k * ex.T_MAX)):.3f} |"
        )
    mo.md(
        "| pH | activity at end | $C_A$ end, aged | $C_A$ end, no fouling |\n"
        "|---|---|---|---|\n" + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def _data_md(mo):
    mo.md(r"""
    ## Data and the frozen trunk

    Nine aged runs at the same Latin hypercube $(T, \mathrm{pH})$
    points the fresh calibration used, so activity is the only
    difference between the two datasets, plus two off-grid runs held
    out for validation. Only $C_A$ is observed, twelve times per
    run, with the same heteroscedastic noise model.

    The trunk comes off disk. `load_predictors` rebuilds a saved
    pytree against a template, matching static configuration on
    every leaf, which is why `ArrheniusKinetics` lives in a shared
    `_model.py` rather than being defined twice.
    """)
    return


@app.cell
def _build_data(SolverConfig, diffrax, ex, jr, make_dataset):
    k_noise, k_template, k_agent, k_grad, k_exp, k_ppo = jr.split(jr.PRNGKey(0), 6)

    train_experiments, val_experiments = ex._build_aged_datasets(doe_seed=0, noise_key=k_noise)
    train_dataset = make_dataset(
        train_experiments,
        state_to_output=ex.state_to_output,
        output_channel_names=ex.OUTPUT_CHANNELS,
    )
    val_dataset = make_dataset(
        val_experiments,
        state_to_output=ex.state_to_output,
        output_channel_names=ex.OUTPUT_CHANNELS,
    )

    trunk = ex.load_trunk(
        ex.Path(ex.__file__).parent / "artefacts" / "trunk_fresh.eqx", key=k_template
    )
    solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05)
    episodes = ex.episodes_from_experiments(train_experiments, trunk)
    val_episodes = ex.episodes_from_experiments(val_experiments, trunk)
    return (
        episodes,
        k_agent,
        k_exp,
        k_grad,
        k_ppo,
        solver,
        train_dataset,
        trunk,
        val_dataset,
        val_episodes,
        val_experiments,
    )


@app.cell(hide_code=True)
def _hold_md(mo):
    mo.md(r"""
    ## The zero-order hold has an exact target

    Activity is held constant across each of the eleven intervals
    between measurements. That sounds like an approximation of a
    continuous curve, and for the *state* it would be. It is not.

    For $\mathrm{d}C_A/\mathrm{d}t = -k\,a(t)\,C_A$ the solution
    over one interval is

    $$
    C_A(t_1) = C_A(t_0)\,\exp\!\left(-k \int_{t_0}^{t_1} a(t)\,
    \mathrm{d}t\right)
    $$

    Only the *integral* of $a$ enters. So holding activity at its
    **interval mean** reproduces the continuous truth exactly, while
    holding it at the left endpoint systematically overestimates on
    a falling curve.

    This matters twice. It fixes what a converged policy should be
    compared against, and it says the step function the policy emits
    is not a sampled $a(t_i)$: it is the piecewise-constant activity
    that best represents each interval, and that target is exact.
    """)
    return


@app.cell
def _hold_demo(ex, jnp, mo):
    _t0, _t1, _ph = 1.0, 1.5, 5.5
    _grid = jnp.linspace(_t0, _t1, 20_001)
    _exact = float(jnp.trapezoid(ex._activity_true(_grid, _ph), _grid) / (_t1 - _t0))
    mo.md(
        f"Over $[{_t0}, {_t1}]$ at pH {_ph}: left endpoint "
        f"**{float(ex._activity_true(_t0, _ph)):.4f}**, interval mean "
        f"**{_exact:.4f}**. The endpoint is {100 * (float(ex._activity_true(_t0, _ph)) / _exact - 1):.1f}% high."
    )
    return


@app.cell(hide_code=True)
def _reward_md(mo):
    mo.md(r"""
    ## Reward, and why it has no ceiling of 11

    $$
    r_t = \exp\!\left(-\frac{(C_A^{m} - C_A^{d})^2}{\sigma^2}\right)
    - w \cdot \mathrm{saturation}(z_t)
    $$

    The first term is the paper's bounded form. The weight is
    $1/\sigma^2$ taken from `ChannelObs.variance`, which the dataset
    already carries, rather than an arbitrary constant.

    The second reads the **latent**, not the physical activity.
    `from_latent`'s derivative carries a $\sigma'(z)$ factor that
    underflows to exactly zero past $|z| \approx 15$, so a penalty
    written against the output would die precisely where saturation
    is worst.

    The return is undiscounted. It is a fit criterion over a finite
    horizon, not a control return.

    Now the part that is easy to get wrong. With residuals
    $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ at the *true* model,

    $$
    \mathbb{E}\!\left[e^{-\varepsilon^2/\sigma^2}\right]
    = \frac{1}{\sqrt{3}} \approx 0.577
    $$

    so a perfect model scores about $0.577 \times 11 = 6.35$, not
    11. And that is not a ceiling either: a model scoring *above*
    the true law is fitting observation noise.
    """)
    return


@app.cell
def _references(episodes, ex, mo, solver):
    truth_return, static_return = ex.reference_returns(episodes, solver)
    noise_optimum = ex.HORIZON / 3.0**0.5
    mo.md(
        f"| reference | undiscounted return |\n|---|---|\n"
        f"| the true deactivation law | **{truth_return:.3f}** |\n"
        f"| no deactivation at all | {static_return:.3f} |\n"
        f"| pure-noise optimum, $11/\\sqrt3$ | {noise_optimum:.3f} |\n"
        f"| naive ceiling, wrong | {ex.HORIZON} |"
    )
    return static_return, truth_return


@app.cell(hide_code=True)
def _policy_md(mo):
    mo.md(r"""
    ## The policy is an ordinary `BoundedPredictor`

    Nothing about it is RL-specific. It maps
    $(C_A, T, \mathrm{pH})$ to an activity in $(0, 1.05]$.

    The covariates are in the observation out of necessity, not
    convenience. Only $C_A$ is measured and $C_{A,0}$ is fixed, so
    the model state is a single scalar: conversion. A policy on
    conversion alone would return the same action for a hot acidic
    batch and a cool neutral one at equal conversion, and could not
    fit the data even in principle. Mowbray's systems had no
    covariates, which is why state-only was complete there.

    The upper bound is 1.05, not 1.0. A fresh catalyst has activity
    exactly 1 and a sigmoid only approaches its edge asymptotically,
    so a hard ceiling would force the policy to saturate at $t = 0$
    and set the saturation term fighting the fit.
    """)
    return


@app.cell
def _build_policy(ex, k_agent, mo):
    agent0 = ex.build_agent(key=k_agent)
    mo.md(
        f"Observation box: `{ex.OBS_KEYS}` over `{ex.OBS_BOUNDS}`  \n"
        f"Action box: `{ex.ACTIVITY_BOUNDS}`, squash "
        f"`{agent0.policy.out_scaler.transform}`, knee at "
        f"±{float(agent0.policy.out_scaler.z_knee):.3f}"
    )
    return (agent0,)


@app.cell(hide_code=True)
def _split_md(mo):
    mo.md(r"""
    ## Sampling in the latent space

    Here is the whole trick. The policy is split at `out_scaler`:

    - the input scaler and inner network produce the Gaussian's mean
      over the **latent** $z$;
    - $z \sim \mathcal{N}(\mu_z, \sigma)$ is the action, so its
      log-probability is a plain diagonal Gaussian with no
      correction term;
    - `out_scaler.from_latent(z)` maps $z$ into the physical box
      *inside the rollout*, as part of the dynamics.

    A $\tanh$-squashed actor would owe a log-determinant term at the
    log-probability. This one owes nothing, because nothing was
    squashed inside the policy.

    Recombined, the two halves are exactly
    `BoundedPredictor.__call__`, so the trained artefact is a plain
    predictor that drops into `predict_dataset`. Worth checking
    rather than asserting:
    """)
    return


@app.cell
def _split_check(agent0, ex, jnp, mo):
    _obs = jnp.array([0.6, 25.0, 6.0])
    _via_split = agent0.policy.out_scaler.from_latent(ex._policy_mean(agent0, _obs))
    _via_call = agent0.policy({"Ca": _obs[0], "temperature_C": _obs[1], "pH": _obs[2]})
    mo.md(
        f"`from_latent(inner(in_scaler(obs)))` = `{float(_via_split[0]):.10f}`  \n"
        f"`policy(obs)` = `{float(_via_call[0]):.10f}`  \n"
        f"bit-identical: **{bool(jnp.all(_via_split == _via_call))}**"
    )
    return


@app.cell(hide_code=True)
def _structural_md(mo):
    mo.md(r"""
    The bound is structural, not enforced. Turning the exploration
    spread up to something absurd cannot push a single sampled
    action outside the box, because no finite $z$ maps outside it.
    A clip would instead pile probability mass onto the edges.
    """)
    return


@app.cell
def _bound_check(episodes, ex, jnp, jr, mo, solver):
    _wild = ex.build_agent(key=jr.PRNGKey(4))._replace(log_std=jnp.full((1,), 3.0))
    _rollout = ex.rollout_batch(
        _wild, episodes, jr.PRNGKey(5), solver=solver, penalty_weight=0.0, n_samples=16
    )
    _low, _high = ex.ACTIVITY_BOUNDS[0]
    mo.md(
        f"With `log_std = 3.0` (spread {float(jnp.exp(jnp.asarray(3.0))):.1f} in latent units), "
        f"{_rollout.activity.size} sampled actions span "
        f"**[{float(jnp.min(_rollout.activity)):.5f}, "
        f"{float(jnp.max(_rollout.activity)):.5f}]** inside the declared "
        f"`({_low}, {_high})`."
    )
    return


@app.cell(hide_code=True)
def _nograd_md(mo):
    mo.md(r"""
    ## PPO never differentiates the ODE

    The rollout is data collection. It calls `diffeqsolve` eleven
    times per episode and returns observations, sampled latents,
    log-probabilities and rewards. Nothing is differentiated.

    The update then recomputes log-probabilities from the *stored*
    observations and latents and never re-enters the solver.
    `rlax` supplies the two pieces of PPO maths:
    `truncated_generalized_advantage_estimation` and
    `clipped_surrogate_pg_loss`.

    So `SolverConfig.adjoint` is irrelevant to this training loop.
    The solver could be stiff, nonsmooth, or a black box.

    (One `rlax` function is not used. `entropy_loss` is categorical,
    reducing a softmax entropy over logits. A diagonal Gaussian's
    differential entropy is a closed form in `log_std`.)

    ## Two baselines to keep it honest

    A one-scalar exponential decay law, and the *identical*
    `BoundedPredictor` trained by ordinary backpropagation through
    the solve. Both use the same zero-order-hold simulator, so the
    discretisation is not a handicap applied only to the policy.
    """)
    return


@app.cell
def _optax_baselines(ex, k_exp, k_grad, mo, solver, train_dataset, trunk):
    from hybridmodels.training import OptaxTrainingConfig, train_with_optax

    def fit(predictors, kind, key):
        """Train one activity model with optax, freezing the scalers."""
        mask = ex.freeze_modules_of_type(ex.trainable_mask(predictors), predictors, ex.BoundScaler)
        return train_with_optax(
            predictors,
            train_dataset,
            OptaxTrainingConfig(
                steps=(600,),
                lr=(3e-3,),
                optimizer=("adamw",),
                reset_optimiser_state=(False,),
                length_schedule=(1.0,),
                loss="mse",
                verbose=False,
            ),
            simulate_fn=ex.make_simulate_fn(trunk, kind=kind),
            solver=solver,
            trainable=mask,
            key=key,
        )

    hist_exp, exponential = fit(ex.ExponentialDecay(key=k_exp), "exponential", k_exp)
    hist_grad, gradient_policy = fit(ex.build_policy(key=k_grad), "policy", k_grad)
    mo.md(
        f"Exponential decay: final loss {hist_exp[-1]:.6f}, "
        f"fitted $k_d$ = {float(exponential.rate()):.4f}  \n"
        f"Gradient-trained policy: final loss {hist_grad[-1]:.6f}"
    )
    return exponential, gradient_policy


@app.cell(hide_code=True)
def _train_md(mo):
    mo.md(r"""
    ## Training the policy
    """)
    return


@app.cell
def _updates_slider(mo):
    updates = mo.ui.slider(
        start=100, stop=800, step=50, value=600, label="PPO updates", show_value=True
    )
    updates
    return (updates,)


@app.cell
def _train(
    agent0,
    episodes,
    ex,
    k_ppo,
    solver,
    truth_return,
    updates,
    val_episodes,
):
    agent, val_agent, returns, val_returns, losses = ex.train_ppo(
        agent0,
        episodes,
        solver=solver,
        n_updates=updates.value,
        n_samples=64,
        lr=3e-4,
        penalty_weight=1e-3,
        truth_return=truth_return,
        val_episodes=val_episodes,
        key=k_ppo,
    )
    return agent, returns, val_agent, val_returns


@app.cell(hide_code=True)
def _overfit_md(mo):
    mo.md(r"""
    ## Over-parameterisation, shown rather than described

    Eleven free actions per run against twelve noisy observations
    leaves room to fit the noise, and the policy takes it. The
    training return climbs past what the true law scores on the same
    data, which is the tell.

    Selecting on training return, the only signal an honest RL loop
    has, therefore picks an overfitted iterate. Both agents are kept
    below, one selected each way.

    Watch where the validation curve peaks: almost exactly at the
    true law's own score, which is where theory says it should.
    """)
    return


@app.cell
def _curves(plt, returns, static_return, truth_return, val_returns):
    _fig, _ax = plt.subplots(figsize=(6.8, 3.8))
    _ax.plot(returns, color="tab:red", lw=1.4, label="training")
    _ax.plot(val_returns, color="tab:blue", lw=1.4, label="validation")
    _ax.axhline(
        truth_return, color="black", ls="--", lw=1.0, label=f"true law scores {truth_return:.2f}"
    )
    _ax.axhline(
        static_return,
        color="tab:grey",
        ls=":",
        lw=1.0,
        label=f"no deactivation = {static_return:.2f}",
    )
    _ax.set_xlabel("update")
    _ax.set_ylabel("mean deterministic return")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _results_md(mo):
    mo.md(r"""
    ## Results on the held-out aged runs
    """)
    return


@app.cell
def _results(
    agent,
    compute_diagnostics,
    ex,
    exponential,
    gradient_policy,
    mo,
    np,
    predict_dataset,
    solver,
    train_dataset,
    trunk,
    val_agent,
    val_dataset,
    val_experiments,
):
    models = {
        "frozen trunk": (None, "frozen"),
        "exp decay": (exponential, "exponential"),
        "optax policy": (gradient_policy, "policy"),
        "PPO train-sel": (agent.policy, "policy"),
        "PPO val-sel": (val_agent.policy, "policy"),
    }
    rows = []
    val_predictions = {}
    for _name, (_predictors, _kind) in models.items():
        _sim = ex.make_simulate_fn(trunk, kind=_kind)
        _vp = predict_dataset(_predictors, val_dataset, simulate_fn=_sim, solver=solver)
        _tp = predict_dataset(_predictors, train_dataset, simulate_fn=_sim, solver=solver)
        _dv = compute_diagnostics(_vp, val_dataset)["Ca"]
        _dt = compute_diagnostics(_tp, train_dataset)["Ca"]
        rows.append(
            f"| {_name} | {_dt.r2:.4f} | {_dv.r2:.4f} | {_dv.rmse:.4f} | {_dt.r2 - _dv.r2:+.4f} |"
        )
        val_predictions[_name] = [np.asarray(_vp[0][i, :, 0]) for i in range(len(val_experiments))]
    mo.md(
        "| model | train $R^2$ | val $R^2$ | val RMSE | gap |\n|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    return (val_predictions,)


@app.cell
def _trajectories(np, plt, val_experiments, val_predictions):
    _fig, _axes = plt.subplots(1, len(val_experiments), figsize=(5.4 * len(val_experiments), 3.8))
    _colours = {
        "frozen trunk": "tab:grey",
        "exp decay": "tab:green",
        "optax policy": "tab:blue",
        "PPO train-sel": "tab:orange",
        "PPO val-sel": "tab:red",
    }
    for _i, (_ax, _exp) in enumerate(zip(_axes, val_experiments, strict=True)):
        _ts = np.asarray(_exp.channels["Ca"].ts)
        _ax.plot(
            _ts, np.asarray(_exp.channels["Ca"].values), "o", ms=4, color="black", label="data"
        )
        for _name, _series in val_predictions.items():
            _ax.plot(_ts, _series[_i], lw=1.6, color=_colours[_name], label=_name)
        _ax.set_title(
            f"T = {float(_exp.covariates['temperature_C']):.1f} °C, "
            f"pH = {float(_exp.covariates['pH']):.2f}",
            fontsize=10,
        )
        _ax.set_xlabel("time")
        _ax.set_ylabel("Ca")
        if _i == 0:
            _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _recovery_md(mo):
    mo.md(r"""
    ## Did it recover the physics

    The step function against the hidden truth, and against the
    exact hold target from earlier. This is the plot that says
    whether the policy learned the deactivation or merely learned to
    fit $C_A$.
    """)
    return


@app.cell
def _recovery(
    ex,
    jax,
    jnp,
    jr,
    np,
    plt,
    solver,
    val_agent,
    val_episodes,
    val_experiments,
):
    _fig, _axes = plt.subplots(1, len(val_experiments), figsize=(5.4 * len(val_experiments), 3.8))
    _dense = jnp.linspace(0.0, ex.T_MAX, 300)
    for _j, (_ax, _exp) in enumerate(zip(_axes, val_experiments, strict=True)):
        _ph = float(_exp.covariates["pH"])
        _ts = np.asarray(_exp.channels["Ca"].ts)
        _ax.plot(_dense, ex._activity_true(_dense, _ph), color="black", lw=2, label="truth")
        _ax.step(
            _ts[:-1],
            [float(ex._activity_interval_mean(_ts[i], _ts[i + 1], _ph)) for i in range(ex.HORIZON)],
            where="post",
            color="black",
            ls="--",
            lw=1.2,
            label="exact hold target",
        )
        _episode = jax.tree.map(lambda x, _j=_j: x[_j][None], val_episodes)
        _roll = ex.rollout_batch(
            val_agent,
            _episode,
            jr.PRNGKey(0),
            solver=solver,
            penalty_weight=0.0,
            n_samples=1,
            deterministic=True,
        )
        _ax.step(
            _ts[:-1],
            np.asarray(_roll.activity[0]),
            where="post",
            color="tab:red",
            lw=1.8,
            label="PPO policy",
        )
        _ax.set_title(f"pH = {_ph:.2f}", fontsize=10)
        _ax.set_xlabel("time")
        _ax.set_ylabel("catalyst activity")
        _ax.set_ylim(-0.05, 1.15)
        if _j == 0:
            _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _closing_md(mo):
    mo.md(r"""
    ## What the structure bought

    The policy is an ordinary `BoundedPredictor`. It was trained by
    an algorithm the library knows nothing about, using a
    third-party RL library, and it still comes back as a predictor
    that `predict_dataset` accepts. That is what "the container is
    never inspected" buys in practice.

    Bounds came from `BoundScaler` rather than from a $\tanh$ bolted
    onto an actor, which removed the log-determinant correction from
    the log-probability and made the box a property of the model
    instead of a property of the training code. The saturation term
    was already there, and reading the latent rather than the output
    is the same reason it is written that way everywhere else in the
    library.

    And the gradient-trained baseline wins on validation fit. It
    should, on a problem this smooth. The result worth keeping is
    that PPO reached essentially the same model without ever
    differentiating the solver, which is what makes it available
    when the adjoint is not.
    """)
    return


@app.cell
def _mo():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
