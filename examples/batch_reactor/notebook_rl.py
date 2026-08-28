"""Batch reactor, part two: a bounded parameter controller trained by PPO.

Narrative version of ``examples/batch_reactor/train_rl_deactivation.py``. The
catalyst calibrated on the first batch reactor page fouls as the batch runs, and
a reinforcement learning policy recovers the missing activity without ever
differentiating the ODE.

The plumbing (aged data generation, the rollout, the PPO update) is imported
from that script so this file stays readable. Everything that carries the
argument is defined and run in cells here.

Run: ``uv run python examples/batch_reactor/notebook_rl.py``

---

## Kinetic parameters as control actions

The batch reactor page fitted a rate constant ``k(T, pH)`` for a first-order
reaction ``A -> B``: a known Arrhenius temperature law, plus a small bounded
network for the pH dependence, which has no first-principles form.

Those were commissioning runs on **fresh catalyst**. This page takes that
fitted model, freezes it, and points it at a reactor whose catalyst **fouls
while the batch runs**. The frozen model cannot express that, and it fails
badly.

## The reframing

Mowbray, Wu, Rogers, Del Rio-Chanona and Zhang (2023) propose treating
kinetic parameter estimation as a *control* problem. Parameters become
actions. A policy reads the current model state and emits a bounded parameter
value; the mechanistic model integrates one interval holding it fixed; the
reward is how close the result lands to the next measurement.

    max_pi sum_{t=0}^{T-1} R(x^m_t, p_t, x^m_{t+1}),
    x^m_{t+1} = f(x^m_t, p_t),     p_t = pi(x^m_t) in P

Two details matter. The rollout is **free-running**: the policy reads the
state its own earlier actions produced, not the measurement, so errors
compound and the trained policy is a model you can run forward on a fresh
batch. And P is a **hard constraint** on the parameter.

## What this page is about

Two claims, in order of how much they are worth.

**Bounds by reparameterisation, inside an RL actor.** Continuous control
normally bounds actions by squashing a Gaussian through ``tanh``, which then
owes a log-determinant correction in the log-probability. Here the policy
distribution lives in the *latent* space and ``BoundScaler`` carries the
latent into the physical box as part of the dynamics. No correction term, no
clip, and the bound holds structurally.

**No gradient through the solver.** PPO differentiates the policy and the
value head. The ODE rollout produces rewards and is never differentiated.
That is the honest reason to reach for RL on a problem whose gradient is
perfectly well behaved, and this page reports a gradient-trained version of
the identical policy to stay honest about it.
"""

# ruff: noqa: F722

import sys
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from hybridmodels import SolverConfig, make_dataset, predict_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_rl_deactivation as ex  # noqa: E402
from _shared import compute_diagnostics  # noqa: E402


def main() -> None:
    # ## The system
    #
    # Fresh catalyst obeys the truth the earlier page fitted. Aged catalyst
    # multiplies it by an activity factor:
    #
    #   k(T, pH, t) = a(t, pH) * k_fresh(T, pH),   a(t, pH) = 1 / (1 + (t/tau(pH))^3)
    #
    # with tau shortening at low pH, so acid attacks the catalyst sooner. The
    # cubic exponent is deliberate: first-order deactivation a(t) = exp(-k_d t)
    # is solvable in closed form and one extra trainable scalar would fit it
    # exactly, leaving a time-varying policy with nothing to do. A cubic Hill
    # curve has a plateau then a sharp fall, and no such shortcut.

    fig_dir = Path(__file__).resolve().parent / "figures_rl"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Activity truth against the best endpoint-matched exponential.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    t = jnp.linspace(0.0, ex.T_MAX, 300)
    for ph in (4.5, 6.0, 7.5):
        ax.plot(t, ex._activity_true(t, ph), lw=2, label=f"pH {ph}")
        # The exponential that agrees with the truth at both ends. It is the
        # best a one-scalar decay law can do, and it misses the plateau.
        end = float(ex._activity_true(ex.T_MAX, ph))
        kd = -float(jnp.log(end)) / ex.T_MAX
        ax.plot(t, jnp.exp(-kd * t), lw=1.0, ls=":", color="grey")
    ax.set_xlabel("time")
    ax.set_ylabel("catalyst activity")
    ax.set_title("Truth (solid) against the best endpoint-matched exponential (dotted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "activity_truth.png", dpi=150)
    plt.close(fig)

    # The fouling is severe on purpose: a starting model that is already good
    # makes the whole exercise decorative. The batch is made to stall at
    # roughly 55% conversion where the fresh-catalyst model predicts it runs
    # nearly to completion.
    print("| pH | activity at end | C_A end, aged | C_A end, no fouling |")
    print("|---|---|---|---|")
    grid = jnp.linspace(0.0, ex.T_MAX, 4001)
    for ph in (4.5, 6.0, 7.5):
        k = float(ex._k_true(25.0, ph))
        integral = float(jnp.trapezoid(ex._activity_true(grid, ph), grid))
        print(
            f"| {ph} | {float(ex._activity_true(ex.T_MAX, ph)):.4f} "
            f"| {float(jnp.exp(-k * integral)):.3f} "
            f"| {float(jnp.exp(-k * ex.T_MAX)):.3f} |"
        )

    # ## Data and the frozen trunk
    #
    # Nine aged runs at the same Latin hypercube (T, pH) points the fresh
    # calibration used, so activity is the only difference between the two
    # datasets, plus two off-grid runs held out for validation. Only C_A is
    # observed, twelve times per run, with the same heteroscedastic noise
    # model.
    k_noise, k_template, k_agent, k_exp, k_grad, k_ppo = jr.split(jr.PRNGKey(0), 6)

    train_experiments, val_experiments = ex._build_aged_datasets(doe_seed=0, noise_key=k_noise)
    train_dataset = make_dataset(train_experiments, output_channel_names=ex.OUTPUT_CHANNELS)
    val_dataset = make_dataset(val_experiments, output_channel_names=ex.OUTPUT_CHANNELS)

    trunk = ex.load_trunk(
        ex.Path(ex.__file__).parent / "artefacts" / "trunk_fresh.eqx", key=k_template
    )
    solver = SolverConfig(solver=diffrax.Tsit5(), rtol=1e-5, atol=1e-7, max_steps=10_000, dt0=0.05)
    episodes = ex.episodes_from_experiments(train_experiments, trunk)
    val_episodes = ex.episodes_from_experiments(val_experiments, trunk)

    # ## The zero-order hold has an exact target
    #
    # Activity is held constant across each of the eleven intervals between
    # measurements. For dC_A/dt = -k a(t) C_A the solution over one interval is
    # C_A(t1) = C_A(t0) exp(-k * integral of a): only the integral of a enters,
    # so holding activity at its interval mean reproduces the continuous truth
    # exactly. The step function the policy emits is the piecewise-constant
    # activity that best represents each interval, and that target is exact.
    t0, t1, ph = 1.0, 1.5, 5.5
    grid = jnp.linspace(t0, t1, 20_001)
    exact = float(jnp.trapezoid(ex._activity_true(grid, ph), grid) / (t1 - t0))
    print(
        f"Over [{t0}, {t1}] at pH {ph}: left endpoint "
        f"{float(ex._activity_true(t0, ph)):.4f}, interval mean "
        f"{exact:.4f}. The endpoint is "
        f"{100 * (float(ex._activity_true(t0, ph)) / exact - 1):.1f}% high."
    )

    # ## Reward, and why it has no ceiling of 11
    #
    #   r_t = exp(-(C_A^m - C_A^d)^2 / sigma^2) - w * saturation(z_t)
    #
    # The weight is taken from ChannelObs.variance, which the dataset already
    # carries. The penalty reads the latent, not the physical activity, because
    # from_latent's derivative underflows to exactly zero past |z| ~ 15. The
    # return is undiscounted: a fit criterion over a finite horizon. With
    # residuals N(0, sigma^2) at the true model,
    # E[exp(-eps^2/sigma^2)] = 1/sqrt(3) = 0.577, so a perfect model scores
    # about 0.577 * 11 = 6.35, not 11.
    truth_return, static_return = ex.reference_returns(episodes, solver)
    noise_optimum = ex.HORIZON / 3.0**0.5
    print("| reference | undiscounted return |")
    print("|---|---|")
    print(f"| the true deactivation law | {truth_return:.3f} |")
    print(f"| no deactivation at all | {static_return:.3f} |")
    print(f"| pure-noise optimum, 11/sqrt3 | {noise_optimum:.3f} |")
    print(f"| naive ceiling, wrong | {ex.HORIZON} |")

    # ## The policy is an ordinary BoundedPredictor
    #
    # Nothing about it is RL-specific. It maps (C_A, T, pH) to an activity in
    # (0, 1.05]. The covariates are in the observation out of necessity: only
    # C_A is measured and C_A,0 is fixed, so a policy on conversion alone would
    # return the same action for a hot acidic batch and a cool neutral one. The
    # upper bound is 1.05, not 1.0: a fresh catalyst has activity exactly 1 and
    # a sigmoid only approaches its edge asymptotically, so a hard ceiling
    # would force the policy to saturate at t = 0.
    agent0 = ex.build_agent(key=k_agent)
    print(
        f"Observation box: `{ex.OBS_KEYS}` over `{ex.OBS_BOUNDS}`  \n"
        f"Action box: `{ex.ACTIVITY_BOUNDS}`, squash "
        f"`{agent0.policy.out_scaler.transform}`, knee at "
        f"\u00b1{float(agent0.policy.out_scaler.z_knee):.3f}"
    )

    # ## Sampling in the latent space
    #
    # The policy is split at out_scaler: the input scaler and inner network
    # produce the Gaussian's mean over the latent z; z ~ N(mu_z, sigma) is the
    # action, so its log-probability is a plain diagonal Gaussian with no
    # correction term; out_scaler.from_latent(z) maps z into the physical box
    # inside the rollout. Recombined, the two halves are exactly
    # BoundedPredictor.__call__, so the trained artefact is a plain predictor
    # that drops into predict_dataset.
    obs = jnp.array([0.6, 25.0, 6.0])
    via_split = agent0.policy.out_scaler.from_latent(ex._policy_mean(agent0, obs))
    via_call = agent0.policy({"Ca": obs[0], "temperature_C": obs[1], "pH": obs[2]})
    print(
        f"`from_latent(inner(in_scaler(obs)))` = `{float(via_split[0]):.10f}`  \n"
        f"`policy(obs)` = `{float(via_call[0]):.10f}`  \n"
        f"bit-identical: **{bool(jnp.all(via_split == via_call))}**"
    )

    # The bound is structural, not enforced: no finite z maps outside the box,
    # so turning the exploration spread up absurdly cannot push a single
    # sampled action outside it.
    wild = ex.build_agent(key=jr.PRNGKey(4))._replace(log_std=jnp.full((1,), 3.0))
    rollout = ex.rollout_batch(
        wild, episodes, jr.PRNGKey(5), solver=solver, penalty_weight=0.0, n_samples=16
    )
    low, high = ex.ACTIVITY_BOUNDS[0]
    print(
        f"With `log_std = 3.0` (spread {float(jnp.exp(jnp.asarray(3.0))):.1f} in latent units), "
        f"{rollout.activity.size} sampled actions span "
        f"**[{float(jnp.min(rollout.activity)):.5f}, "
        f"{float(jnp.max(rollout.activity)):.5f}]** inside the declared "
        f"`({low}, {high})`."
    )

    # ## PPO never differentiates the ODE
    #
    # The rollout is data collection: it calls diffeqsolve eleven times per
    # episode and returns observations, sampled latents, log-probabilities and
    # rewards. Nothing is differentiated. The update recomputes
    # log-probabilities from the stored observations and latents and never
    # re-enters the solver. SolverConfig.adjoint is irrelevant to this loop.
    #
    # Two baselines keep it honest: a one-scalar exponential decay law, and the
    # identical BoundedPredictor trained by ordinary backpropagation through
    # the solve. Both use the same zero-order-hold simulator, so the
    # discretisation is not a handicap applied only to the policy.
    from hybridmodels.training import OptaxTrainingConfig, train_with_optax

    def fit(predictors, kind, key):
        """Train one activity model with optax, freezing the scalers."""
        mask = ex.frozen_default_mask(predictors, ex.BoundScaler)
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
            state_to_output=ex.state_to_output,
            solver=solver,
            trainable=mask,
            key=key,
        )

    hist_exp, exponential = fit(ex.ExponentialDecay(key=k_exp), "exponential", k_exp)
    hist_grad, gradient_policy = fit(ex.build_policy(key=k_grad), "policy", k_grad)
    print(
        f"Exponential decay: final loss {hist_exp[-1]:.6f}, "
        f"fitted k_d = {float(exponential.rate()):.4f}  \n"
        f"Gradient-trained policy: final loss {hist_grad[-1]:.6f}"
    )

    # ## Training the policy
    #
    # Eleven free actions per run against twelve noisy observations leaves room
    # to fit the noise, and the policy takes it: the training return climbs
    # past what the true law scores on the same data. Both agents are kept
    # below, one selected on training return (the only signal an honest RL loop
    # has) and one selected on validation return.
    updates = 600  # the notebook's default slider position
    agent, val_agent, returns, val_returns, losses = ex.train_ppo(
        agent0,
        episodes,
        solver=solver,
        n_updates=updates,
        n_samples=64,
        lr=3e-4,
        penalty_weight=1e-3,
        truth_return=truth_return,
        val_episodes=val_episodes,
        key=k_ppo,
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(returns, color="tab:red", lw=1.4, label="training")
    ax.plot(val_returns, color="tab:blue", lw=1.4, label="validation")
    ax.axhline(
        truth_return, color="black", ls="--", lw=1.0, label=f"true law scores {truth_return:.2f}"
    )
    ax.axhline(
        static_return,
        color="tab:grey",
        ls=":",
        lw=1.0,
        label=f"no deactivation = {static_return:.2f}",
    )
    ax.set_xlabel("update")
    ax.set_ylabel("mean deterministic return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "ppo_learning_curves.png", dpi=150)
    plt.close(fig)

    # ## Results on the held-out aged runs
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
        _vp = predict_dataset(
            _predictors, val_dataset, simulate_fn=_sim, state_to_output=ex.state_to_output,
            solver=solver,
        )
        _tp = predict_dataset(
            _predictors, train_dataset, simulate_fn=_sim, state_to_output=ex.state_to_output,
            solver=solver,
        )
        _dv = compute_diagnostics(_vp, val_dataset)["Ca"]
        _dt = compute_diagnostics(_tp, train_dataset)["Ca"]
        rows.append(
            f"| {_name} | {_dt.r2:.4f} | {_dv.r2:.4f} | {_dv.rmse:.4f} | {_dt.r2 - _dv.r2:+.4f} |"
        )
        val_predictions[_name] = [np.asarray(_vp[0][i, :, 0]) for i in range(len(val_experiments))]
    print("| model | train R2 | val R2 | val RMSE | gap |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))

    fig, axes = plt.subplots(1, len(val_experiments), figsize=(5.4 * len(val_experiments), 3.8))
    colours = {
        "frozen trunk": "tab:grey",
        "exp decay": "tab:green",
        "optax policy": "tab:blue",
        "PPO train-sel": "tab:orange",
        "PPO val-sel": "tab:red",
    }
    for _i, (_ax, _exp) in enumerate(zip(axes, val_experiments, strict=True)):
        _ts = np.asarray(_exp.channels["Ca"].ts)
        _ax.plot(_ts, np.asarray(_exp.channels["Ca"].values), "o", ms=4, color="black",
                 label="data")
        for _name, _series in val_predictions.items():
            _ax.plot(_ts, _series[_i], lw=1.6, color=colours[_name], label=_name)
        _ax.set_title(
            f"T = {float(_exp.covariates['temperature_C']):.1f} \u00b0C, "
            f"pH = {float(_exp.covariates['pH']):.2f}",
            fontsize=10,
        )
        _ax.set_xlabel("time")
        _ax.set_ylabel("Ca")
        if _i == 0:
            _ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "val_trajectories.png", dpi=150)
    plt.close(fig)

    # ## Did it recover the physics
    #
    # The step function against the hidden truth, and against the exact hold
    # target from earlier. This is the plot that says whether the policy
    # learned the deactivation or merely learned to fit C_A.
    fig, axes = plt.subplots(1, len(val_experiments), figsize=(5.4 * len(val_experiments), 3.8))
    dense = jnp.linspace(0.0, ex.T_MAX, 300)
    for _j, (_ax, _exp) in enumerate(zip(axes, val_experiments, strict=True)):
        _ph = float(_exp.covariates["pH"])
        _ts = np.asarray(_exp.channels["Ca"].ts)
        _ax.plot(dense, ex._activity_true(dense, _ph), color="black", lw=2, label="truth")
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
    fig.tight_layout()
    fig.savefig(fig_dir / "activity_recovery.png", dpi=150)
    plt.close(fig)

    # ## What the structure bought
    #
    # The policy is an ordinary BoundedPredictor trained by an algorithm the
    # library knows nothing about, and it still comes back as a predictor that
    # predict_dataset accepts. Bounds came from BoundScaler rather than a tanh
    # bolted onto an actor, removing the log-determinant correction and making
    # the box a property of the model. And the gradient-trained baseline wins
    # on validation fit -- it should, on a problem this smooth. The result
    # worth keeping is that PPO reached essentially the same model without ever
    # differentiating the solver.


if __name__ == "__main__":
    main()