"""Tests for ``RichTrainingUI`` (SPEC §5.9 / R-U1, R-U2, R-U3, R-U4).

Rendered output is verified via ``Console(record=True, force_terminal=False)``
so the dashboard can be exercised in CI without an interactive terminal.
``Live`` in non-terminal mode flushes its final renderable once on ``stop()``,
which is exactly what we want for assertion-friendly captures.
"""

from __future__ import annotations

import re
from io import StringIO

import diffrax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from rich.console import Console

from hybridmodels.data import ChannelObs, Dataset, make_dataset, make_experiment
from hybridmodels.predictors.base import Predictor
from hybridmodels.solver import SolverConfig
from hybridmodels.training.optax import OptaxTrainingConfig, train_with_optax
from hybridmodels.ui import RichTrainingUI, TrainingUI

# Significant-figure pattern: ``\d\.\d{4,}`` matches things like 0.5000, 0.16667.
# Used both to gate on "loss values rendered to >= 4 sig figs" and to count the
# rows of the loss table in the throttling test.
_SIG4 = re.compile(r"\d\.\d{4,}")


def _make_console() -> Console:
    """Recording console with no TTY so ``Live`` prints once at stop()."""
    return Console(record=True, force_terminal=False, width=120, file=StringIO())


# ---------------------------------------------------------------------------
# Test 1: protocol satisfaction.
# ---------------------------------------------------------------------------


def test_rich_training_ui_satisfies_protocol() -> None:
    ui = RichTrainingUI(console=_make_console())
    assert isinstance(ui, TrainingUI)


# ---------------------------------------------------------------------------
# Test 2: rendered-output content (R-U1, R-U3, R-U4).
# ---------------------------------------------------------------------------


def test_rendered_output_contains_phase_loss_optimizer_and_final() -> None:
    console = _make_console()
    ui = RichTrainingUI(console=console)

    ui.on_run_start(total_steps=10, num_phases=2)

    ui.on_compile_start(bucket_idx=0, bucket_shape=(4, 8))
    ui.on_compile_progress(bucket_idx=0, total_buckets=1)
    ui.on_compile_done(bucket_idx=0)

    ui.on_phase_start(phase_idx=0, phase_steps=5, lr=0.01, optimizer="adamw")
    for i in range(3):
        ui.on_step_end(step_idx=i, phase_idx=0, loss=0.5 / (i + 1))
    ui.on_phase_end(phase_idx=0)

    ui.on_phase_start(phase_idx=1, phase_steps=5, lr=0.005, optimizer="adabelief")
    for i in range(3, 5):
        ui.on_step_end(step_idx=i, phase_idx=1, loss=0.05 / (i + 1))
    ui.on_phase_end(phase_idx=1)

    ui.on_run_end(final_loss=0.001)

    output = console.export_text()
    lower = output.lower()

    assert "phase" in lower, "rendered output must mention phase somewhere"
    assert "adamw" in lower, "rendered output must surface the active optimiser name"
    # 'final' should appear near the end (post-run summary). Check overall and
    # the back third of the buffer for stricter ordering.
    assert "final" in lower
    tail = lower[-len(lower) // 3 :]
    assert "final" in tail, "expected 'final' to appear in the post-run section"

    assert _SIG4.search(output) is not None, (
        "expected at least one loss rendered to >= 4 significant figures"
    )


# ---------------------------------------------------------------------------
# Test 3: defensive event ordering.
# ---------------------------------------------------------------------------


def test_out_of_order_events_do_not_crash() -> None:
    ui = RichTrainingUI(console=_make_console())
    ui.on_run_start(total_steps=1, num_phases=1)
    # phase_end before any phase_start: should be a no-op, not an error.
    ui.on_phase_end(phase_idx=0)
    # step_end with no active phase: also harmless.
    ui.on_step_end(step_idx=0, phase_idx=0, loss=0.0)
    ui.on_run_end(final_loss=0.0)


# ---------------------------------------------------------------------------
# Test 4: log_every throttling of the loss table.
# ---------------------------------------------------------------------------


def test_log_every_throttles_loss_table_rows() -> None:
    console = _make_console()
    ui = RichTrainingUI(console=console, log_every=3, recent_losses=10)

    ui.on_run_start(total_steps=3, num_phases=1)
    ui.on_phase_start(phase_idx=0, phase_steps=3, lr=0.01, optimizer="adamw")

    # Three distinct loss values that won't collide with the final-loss summary
    # when later rendered to four-significant-figure precision.
    losses = [0.7771, 0.6661, 0.5551]
    for i, loss in enumerate(losses):
        ui.on_step_end(step_idx=i, phase_idx=0, loss=loss)
    ui.on_phase_end(phase_idx=0)

    # Final loss is set to a unique value so we can distinguish summary text
    # from logged step rows.
    ui.on_run_end(final_loss=0.4441)

    output = console.export_text()

    expected_rows = len([i for i in range(len(losses)) if i % 3 == 0])
    assert expected_rows == 1

    # Every loss in the throttled set should appear; the others must not show
    # up as table rows. The final-loss summary value is excluded by design.
    appearance = {round(loss, 4): f"{loss:.4f}" in output for loss in losses}
    assert appearance == {0.7771: True, 0.6661: False, 0.5551: False}

    # Cross-check by counting per-line matches of the four-sig-fig pattern that
    # are not part of the run header / final-loss summary. We restrict to the
    # block that contains "Recent losses" if our render labels the panel that
    # way; otherwise count all >=4-sig-fig occurrences of the throttled losses.
    visible_step_losses = [
        m for m in _SIG4.findall(output) if m.startswith("0.7771")
    ]
    assert len(visible_step_losses) == expected_rows


# ---------------------------------------------------------------------------
# Tests 5 & 6: integration with train_with_optax (R-U2).
#
# A minimal harmonic-oscillator fixture, mirroring tests/test_train_optax.py
# but trimmed to the bare minimum needed to exercise the verbose=True /
# verbose=False branches of UI selection.
# ---------------------------------------------------------------------------


class _OmegaPredictor(Predictor):
    omega: Array

    def __init__(self, omega: float) -> None:
        self.omega = jnp.asarray(omega, dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:  # type: ignore[override]
        return self.omega


def _y0_fn_factory(y0: Array):
    def _y0_fn(_cov, _channels):
        return y0

    return _y0_fn


def _solver_config() -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
        max_steps=4096,
        dt0=0.05,
    )


def _state_to_output(state: Array) -> Array:
    return state[..., 0:1]


def _make_oscillator_dataset() -> Dataset:
    ts = jnp.linspace(0.0, 5.0, 10)
    experiments = []
    for i, (x0, v0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        x_obs = x0 * jnp.cos(ts) + v0 * jnp.sin(ts)
        y0 = jnp.asarray([x0, v0])
        experiments.append(
            make_experiment(
                covariates={"id": float(i)},
                channels={"position": ChannelObs(ts=ts, values=x_obs)},
                y0_fn=_y0_fn_factory(y0),
                exp_id=f"exp_{i}",
            )
        )
    return make_dataset(
        experiments,
        state_to_output=_state_to_output,
        output_channel_names=("position",),
    )


def _make_simulate_fn():
    def simulate_fn(predictor, ts, covariates, y0, solver):
        omega = predictor.omega

        def vector_field(t, y, args):
            return jnp.stack([y[1], -(omega**2) * y[0]])

        term = diffrax.ODETerm(vector_field)
        controller = diffrax.PIDController(rtol=solver.rtol, atol=solver.atol)
        sol = diffrax.diffeqsolve(
            term,
            solver.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=solver.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=solver.max_steps,
        )
        return sol.ys

    return simulate_fn


def _trivial_config(verbose: bool) -> OptaxTrainingConfig:
    return OptaxTrainingConfig(
        steps=(2,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        length_schedule=(1.0,),
        verbose=verbose,
    )


def test_train_with_optax_verbose_true_runs_with_rich_ui_default(monkeypatch) -> None:
    """verbose=True with ui=None must select RichTrainingUI and run end-to-end.

    We patch the module-level ``RichTrainingUI`` symbol used by
    ``train_with_optax`` to inject a recording console so the test verifies
    both that selection happened and that rendering reached the recording
    surface.
    """
    import hybridmodels.training.optax as optax_module

    record_console = _make_console()

    def _factory():
        return RichTrainingUI(console=record_console)

    monkeypatch.setattr(optax_module, "RichTrainingUI", _factory)

    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()
    history, trained = train_with_optax(
        pred,
        ds,
        _trivial_config(verbose=True),
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )

    assert trained is not None
    assert len(history) == 2
    output = record_console.export_text()
    # Confirm the Rich UI actually saw events: we expect the optimiser name
    # and the final-loss summary in the rendered text.
    assert "adamw" in output.lower()
    assert "final" in output.lower()


def test_train_with_optax_verbose_false_silent_on_stdout(capsys) -> None:
    pred = _OmegaPredictor(omega=2.0)
    ds = _make_oscillator_dataset()
    train_with_optax(
        pred,
        ds,
        _trivial_config(verbose=False),
        simulate_fn=_make_simulate_fn(),
        solver=_solver_config(),
        key=jr.PRNGKey(0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# Quiet ruff F401: pytest is imported for type checking the fixtures.
_ = pytest
