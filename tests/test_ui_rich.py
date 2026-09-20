"""Tests for ``RichTrainingUI``.

Rendered output is verified via
``Console(record=True, force_terminal=False)`` so the dashboard can be
exercised in CI without an interactive terminal. In non-terminal mode
``rich.live.Live`` flushes its final renderable once on ``stop()``,
which is exactly what we want for assertion-friendly captures.
"""

from __future__ import annotations

import re

import jax.random as jr
import pytest
from _harness import (
    OmegaPredictor,
    make_oscillator_dataset,
    make_oscillator_simulate_fn,
    oscillator_state_to_output,
    recording_console,
    solver_config,
)

from jaxhybridmodels.training.optax import OptaxTrainingConfig, train_with_optax
from jaxhybridmodels.ui import RichTrainingUI, TrainingUI

# Significant-figure pattern: ``\d\.\d{4,}`` matches things like 0.5000, 0.16667.
# Used both to gate on "loss values rendered to >= 4 sig figs" and to count the
# rows of the loss table in the throttling test.
_SIG4 = re.compile(r"\d\.\d{4,}")


# ---------------------------------------------------------------------------
# Test 1: protocol satisfaction.
# ---------------------------------------------------------------------------


def test_rich_training_ui_satisfies_protocol() -> None:
    ui = RichTrainingUI(console=recording_console())
    assert isinstance(ui, TrainingUI)


# ---------------------------------------------------------------------------
# Test 2: rendered-output content.
# ---------------------------------------------------------------------------


def test_rendered_output_contains_phase_loss_optimizer_and_final() -> None:
    console = recording_console()
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
    ui = RichTrainingUI(console=recording_console())
    ui.on_run_start(total_steps=1, num_phases=1)
    # phase_end before any phase_start: should be a no-op, not an error.
    ui.on_phase_end(phase_idx=0)
    # step_end with no active phase: also harmless.
    ui.on_step_end(step_idx=0, phase_idx=0, loss=0.0)
    ui.on_run_end(final_loss=0.0)


# ---------------------------------------------------------------------------
# Tests 4 & 5: integration with ``train_with_optax``.
#
# The harmonic oscillator comes from ``tests/_harness.py``, cut to two
# experiments here: these tests only need the ``verbose=True`` /
# ``verbose=False`` branches of UI selection to run end-to-end.
# ---------------------------------------------------------------------------


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
    import jaxhybridmodels.training.optax as optax_module

    record_console = recording_console()

    def _factory():
        return RichTrainingUI(console=record_console)

    monkeypatch.setattr(optax_module, "RichTrainingUI", _factory)

    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset(initial_states=((1.0, 0.0), (0.0, 1.0)), t_max=5.0, n_timesteps=10)
    history, trained = train_with_optax(
        pred,
        ds,
        _trivial_config(verbose=True),
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
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
    pred = OmegaPredictor(omega=2.0)
    ds = make_oscillator_dataset(initial_states=((1.0, 0.0), (0.0, 1.0)), t_max=5.0, n_timesteps=10)
    train_with_optax(
        pred,
        ds,
        _trivial_config(verbose=False),
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# Quiet ruff F401: pytest is imported for type checking the fixtures.
_ = pytest
