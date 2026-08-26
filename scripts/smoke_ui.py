"""Smoke run for the UI callback protocols.

Run with:

    uv run python scripts/smoke_ui.py
"""

from __future__ import annotations

import io
import sys
from typing import Any

from hybridmodels import EvosaxUI, SilentUI, TrainingUI
from hybridmodels.ui.testing import RecordingUI


def fire_representative_events(ui: Any) -> None:
    ui.on_run_start(total_steps=20, num_phases=2)
    ui.on_compile_start(bucket_idx=0, bucket_shape=(4, 8))
    ui.on_compile_progress(bucket_idx=0, total_buckets=2)
    ui.on_compile_done(bucket_idx=0)
    ui.on_phase_start(phase_idx=0, phase_steps=10, lr=1e-3, optimizer="adamw")
    ui.on_step_end(step_idx=4, phase_idx=0, loss=0.42)
    ui.on_phase_end(phase_idx=0)
    ui.on_generation_end(gen_idx=0, best_fitness=0.2, mean_fitness=0.5)
    ui.on_run_end(final_loss=0.05)
    ui.on_message(level="info", text="finished")


def main() -> None:
    recorder = RecordingUI()
    fire_representative_events(recorder)

    print(f"RecordingUI captured {len(recorder.events)} events:")
    for i, (name, kwargs) in enumerate(recorder.events, start=1):
        print(f"  {i:>2}. {name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")

    silent = SilentUI()
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = captured_out, captured_err
    try:
        fire_representative_events(silent)
    finally:
        sys.stdout, sys.stderr = real_out, real_err

    out_text = captured_out.getvalue()
    err_text = captured_err.getvalue()
    if out_text or err_text:
        raise SystemExit(f"SilentUI emitted output. stdout={out_text!r}, stderr={err_text!r}")

    print(f"SilentUI silent on stdout/stderr: stdout={out_text!r}, stderr={err_text!r}")
    print(f"isinstance(SilentUI(), TrainingUI) = {isinstance(silent, TrainingUI)}")
    print(f"isinstance(SilentUI(), EvosaxUI)   = {isinstance(silent, EvosaxUI)}")


if __name__ == "__main__":
    main()
