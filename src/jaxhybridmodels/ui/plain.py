"""Plain progress lines, for output that is not a terminal.

A live dashboard redraws itself, which only makes sense on a terminal. In a
log file, a CI job or under ``nohup`` these UIs print a line about every tenth
of the run instead, so a long job shows it is alive and its log stays
greppable. ``fit`` picks them when ``verbose=True`` and no display can redraw.
"""

import math
import time
from typing import Any

from rich.console import Console

_LINES_PER_RUN = 10


def live_display_available() -> bool:
    """Whether output can redraw in place: a terminal or a Jupyter notebook.

    Jupyter is checked separately because Rich can report it as not a terminal.
    """
    console = Console()
    return console.is_terminal or console.is_jupyter


class _PlainUI:
    """What the Optax and Evosax lines share: timing, the compile notice and messages."""

    _total = 0
    _done = 0
    _best = math.inf
    _started = 0.0
    _compiling = False

    def _begin(self, total: int, opening: str) -> None:
        self._total, self._done, self._best = total, 0, math.inf
        self._started = time.monotonic()
        self._compiling = False
        print(opening, flush=True)

    def _is_due(self) -> bool:
        self._done += 1
        every = max(1, self._total // _LINES_PER_RUN)
        return self._done % every == 0 or self._done == self._total

    def on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None:
        # Printed once: it exists so a long first compile does not look hung.
        if not self._compiling:
            self._compiling = True
            print("compiling, which can take a while before the first step", flush=True)

    def on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None:
        pass

    def on_compile_done(self, *, bucket_idx: int) -> None:
        pass

    def on_run_end(self, *, returned_loss: float) -> None:
        elapsed = time.monotonic() - self._started
        if math.isnan(returned_loss):
            print(f"stopped after {elapsed:.0f} s with no finite loss", flush=True)
        else:
            print(f"done: final loss {returned_loss:.2e} in {elapsed:.0f} s", flush=True)

    def on_message(self, *, level: str, text: str) -> None:
        print(f"{level}: {text}", flush=True)


class PlainTrainingUI(_PlainUI):
    """:class:`~jaxhybridmodels.ui.TrainingUI` that prints plain lines."""

    def on_run_start(self, *, total_steps: int, num_phases: int) -> None:
        phases = "phase" if num_phases == 1 else "phases"
        self._begin(total_steps, f"training: {total_steps} steps in {num_phases} {phases}")

    def on_phase_start(
        self, *, phase_idx: int, phase_steps: int, lr: float, optimizer: Any
    ) -> None:
        pass

    def on_phase_end(self, *, phase_idx: int) -> None:
        pass

    def on_step_end(
        self, *, step_idx: int, phase_idx: int, loss: float, penalty: float = 0.0
    ) -> None:
        self._best = min(self._best, loss)
        if self._is_due():
            print(
                f"step {self._done}/{self._total}  loss {loss:.2e}  best {self._best:.2e}",
                flush=True,
            )


class PlainEvosaxUI(_PlainUI):
    """:class:`~jaxhybridmodels.ui.EvosaxUI` that prints plain lines."""

    def on_run_start(self, *, total_steps: int, population_size: int) -> None:
        self._begin(
            total_steps, f"searching: {total_steps} generations of {population_size} candidates"
        )

    def on_generation_end(self, *, gen_idx: int, best_loss: float, mean_loss: float) -> None:
        self._best = min(self._best, best_loss)
        if self._is_due():
            print(
                f"generation {self._done}/{self._total}  mean {mean_loss:.2e}  "
                f"best {self._best:.2e}",
                flush=True,
            )
