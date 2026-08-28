"""Tests for ``RichEvosaxUI``.

Mirrors ``tests/test_ui_rich.py`` but targets the evosax UI surface:
instead of phases and per-step losses there are generations with
best/mean fitness. The test discipline is otherwise identical — a
non-TTY recording console captures the dashboard so assertions can be
written against ``console.export_text()``.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import jax.random as jr
import pytest
from _harness import (
    N_DIM,
    QuadraticPredictor,
    quadratic_dataset,
    quadratic_simulate_fn,
    quadratic_state_to_output,
    recording_console,
    solver_config,
)

from hybridmodels.training.evosax import EvosaxTrainingConfig, train_with_evosax
from hybridmodels.ui import EvosaxUI, RichEvosaxUI

# Significant-figure pattern: ``\d\.\d{4,}`` matches things like 0.5000, 0.16667.
# Used to assert "fitness rendered to >= 4 sig figs" and to count throttled
# rows in the recent-generation table.
_SIG4 = re.compile(r"\d\.\d{4,}")


# ---------------------------------------------------------------------------
# Test 1: protocol satisfaction.
# ---------------------------------------------------------------------------


def test_rich_evosax_ui_satisfies_protocol() -> None:
    ui = RichEvosaxUI(console=recording_console())
    assert isinstance(ui, EvosaxUI)


# ---------------------------------------------------------------------------
# Test 2: rendered-output content.
# ---------------------------------------------------------------------------


def test_rendered_output_contains_population_generation_fitness_and_final() -> None:
    console = recording_console()
    ui = RichEvosaxUI(console=console)

    ui.on_run_start(num_generations=5, population_size=12)
    ui.on_compile_start(bucket_idx=0, bucket_shape=(1, 4))
    ui.on_compile_progress(bucket_idx=0, total_buckets=1)
    ui.on_compile_done(bucket_idx=0)
    for i in range(5):
        ui.on_generation_end(
            gen_idx=i,
            best_fitness=1.0 / (i + 1),
            mean_fitness=2.0 / (i + 1),
        )
    ui.on_run_end(best_fitness=0.05)

    output = console.export_text()
    lower = output.lower()

    # Population size from on_run_start should be visible somewhere in the dashboard.
    assert "12" in output, "rendered output must surface the population size"
    # Either "generation" or the abbreviation "gen" must appear.
    assert "generation" in lower or "gen" in lower, (
        "rendered output must mention generation/gen somewhere"
    )
    # At least one fitness rendered to >= 4 significant figures.
    assert _SIG4.search(output) is not None, (
        "expected at least one fitness rendered to >= 4 significant figures"
    )
    # 'final' should appear near the end (post-run summary).
    assert "final" in lower
    tail = lower[-len(lower) // 3 :]
    assert "final" in tail, "expected 'final' to appear in the post-run section"


# ---------------------------------------------------------------------------
# Test 3: defensive event ordering.
# ---------------------------------------------------------------------------


def test_out_of_order_events_do_not_crash() -> None:
    ui = RichEvosaxUI(console=recording_console())
    # on_run_end immediately after on_run_start: no generations in between.
    ui.on_run_start(num_generations=3, population_size=4)
    ui.on_run_end(best_fitness=0.0)


# ---------------------------------------------------------------------------
# Test 4: log_every throttling of the recent-generation table.
# ---------------------------------------------------------------------------


def test_log_every_throttles_generation_table_rows() -> None:
    console = recording_console()
    # recent_generations large enough to hold all expected rows so throttling
    # alone determines what shows up.
    ui = RichEvosaxUI(console=console, log_every=2, recent_generations=10)

    ui.on_run_start(num_generations=5, population_size=4)

    # Distinct best-fitness values per generation, picked so the four-sig-fig
    # rendering disambiguates them: 0.7771, 0.6661, 0.5551, 0.4441, 0.3331.
    bests = [0.7771, 0.6661, 0.5551, 0.4441, 0.3331]
    means = [b + 0.1 for b in bests]
    for i, (b, m) in enumerate(zip(bests, means, strict=True)):
        ui.on_generation_end(gen_idx=i, best_fitness=b, mean_fitness=m)

    # Final fitness picked unique so it does not collide with any of the above.
    ui.on_run_end(best_fitness=0.2221)

    output = console.export_text()

    # log_every=2 means gen_idx in {0, 2, 4} are recorded; {1, 3} are not.
    expected_present = {0: True, 1: False, 2: True, 3: False, 4: True}
    appearance = {i: f"{bests[i]:.4f}" in output for i in range(len(bests))}
    assert appearance == expected_present


# ---------------------------------------------------------------------------
# Tests 5 & 6: integration with ``train_with_evosax``.
#
# The 4-D quadratic problem comes from ``tests/_harness.py``, shared with
# ``tests/test_train_evosax.py``. It needs no ODE solve, so UI selection is
# exercised without paying for numerics.
# ---------------------------------------------------------------------------


def test_train_with_evosax_verbose_true_runs_with_rich_ui_default(monkeypatch) -> None:
    """verbose=True with ui=None must select RichEvosaxUI and run end-to-end.

    Variant implemented: monkeypatch the module-level ``RichEvosaxUI`` symbol
    in ``hybridmodels.training.evosax`` so the trainer instantiates a
    recording-console-wired UI, then assert the recording captured generation
    output and a fitness number.
    """
    import hybridmodels.training.evosax as evosax_module

    record_console = recording_console()

    def _factory(log_every: int | None = None):
        return RichEvosaxUI(console=record_console)

    monkeypatch.setattr(evosax_module, "RichEvosaxUI", _factory)

    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=8,
        num_generations=2,
        init="warm",
        sigma_init=0.5,
        verbose=True,
    )
    history, trained = train_with_evosax(
        pred,
        ds,
        config,
        simulate_fn=quadratic_simulate_fn,
        state_to_output=quadratic_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )

    assert trained is not None
    assert len(history) == 2
    output = record_console.export_text()
    lower = output.lower()
    assert "generation" in lower or "gen" in lower
    assert _SIG4.search(output) is not None, (
        "expected at least one fitness number rendered to >= 4 sig figs"
    )


def test_train_with_evosax_verbose_false_silent_on_stdout(capsys) -> None:
    pred = QuadraticPredictor(theta=jnp.zeros(N_DIM))
    ds = quadratic_dataset()
    config = EvosaxTrainingConfig(
        algorithm="CMA_ES",
        population_size=8,
        num_generations=2,
        init="warm",
        sigma_init=0.5,
        verbose=False,
    )
    train_with_evosax(
        pred,
        ds,
        config,
        simulate_fn=quadratic_simulate_fn,
        state_to_output=quadratic_state_to_output,
        solver=solver_config(),
        key=jr.PRNGKey(0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# Quiet ruff F401: pytest is imported for fixture typing.
_ = pytest
