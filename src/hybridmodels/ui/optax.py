"""Rich-based ``TrainingUI`` for ``train_with_optax``.

A single :class:`rich.live.Live` is started in
:meth:`RichTrainingUI.on_run_start` and stopped in
:meth:`RichTrainingUI.on_run_end`. The live renderable is a
:class:`rich.console.Group` that swaps panels as the run progresses:

* a header :class:`rich.panel.Panel` with run-level info (total steps,
  number of phases, elapsed wall-clock);
* a per-phase :class:`rich.progress.Progress` bar showing per-step
  advancement inside the active phase (rebuilt on every
  ``on_phase_start`` so the bar resets cleanly between phases);
* a compile-progress panel shown only between ``on_compile_start`` and
  the matching ``on_compile_done``; once the *first* bucket finishes
  compiling this slot is replaced by the phase progress bar
  permanently;
* a message-log panel showing the most recent ``recent_messages``
  lines fired through :meth:`RichTrainingUI.on_message`.

The class is deliberately defensive about event ordering. An event that
arrives before the state it references, ``on_phase_end`` with no active
phase for instance, becomes a no-op rather than an assertion. Tournament
restarts and graceful-abort paths can legitimately fire events out of
order, and a rendering failure must never take down a training run.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


def _format_loss(loss: float) -> str:
    """Format a loss value to four decimal places."""
    return f"{float(loss):.4f}"


class RichTrainingUI:
    """Live Rich dashboard satisfying ``hybridmodels.ui.base.TrainingUI``.

    Parameters
    ----------
    console:
        Optional :class:`rich.console.Console`. ``None`` constructs a default
        ``Console()``. Tests typically pass a recording console
        (``Console(record=True, force_terminal=False, ...)``) so the
        rendered final state can be asserted on.
    recent_messages:
        Maximum number of lines the message-log panel holds.

    Notes
    -----
    The instance carries one :class:`rich.live.Live` between
    ``on_run_start`` and ``on_run_end``. Any state mutation outside that
    window updates the model only; the next ``on_run_start`` rebuilds Live
    afresh, so a single ``RichTrainingUI`` instance can be reused for
    sequential runs (used in tests).
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        recent_messages: int = 5,
    ) -> None:
        self._console: Console = console if console is not None else Console()
        # The deque maxlen pins the visible history without unbounded growth.
        self._messages: deque[tuple[str, str]] = deque(maxlen=recent_messages)

        # Run-level state (set in on_run_start, read by _render).
        self._run_start_wallclock: float | None = None
        self._total_steps: int = 0
        self._num_phases: int = 0
        self._final_loss: float | None = None
        self._run_active: bool = False

        # Phase-level state.
        self._phase_idx: int | None = None
        self._phase_steps: int = 0
        self._phase_progress_bar: Progress | None = None
        self._phase_task_id: TaskID | None = None
        self._phase_lr: float | None = None
        self._phase_optimizer: str | None = None
        # Cumulative phase summary (phase_idx, optimizer, lr). Shown in the
        # header so the final frame of a multi-phase run names every
        # optimiser it used, rather than only the last one.
        self._phase_history: list[tuple[int, str, float]] = []

        # Compile-panel state. The compile panel occupies the same
        # vertical slot as the phase-progress bar; once any bucket has
        # finished compiling we leave that slot to the phase-progress
        # rendering for the rest of the run.
        self._compile_active: bool = False
        self._compile_bucket_idx: int | None = None
        self._compile_bucket_shape: tuple[int, ...] | None = None
        self._compile_total_buckets: int | None = None
        self._compile_first_done: bool = False

        self._live: Live | None = None

    # ------------------------------------------------------------------
    # TrainingUI protocol implementation.
    # ------------------------------------------------------------------

    def on_run_start(self, *, total_steps: int, num_phases: int) -> None:
        self._run_start_wallclock = time.monotonic()
        self._total_steps = int(total_steps)
        self._num_phases = int(num_phases)
        self._final_loss = None
        self._run_active = True
        # Reset per-phase / compile / message state so a reused instance
        # does not bleed prior-run rows into the new run.
        self._phase_idx = None
        self._phase_progress_bar = None
        self._phase_task_id = None
        self._phase_history = []
        self._compile_active = False
        self._compile_first_done = False
        self._messages.clear()

        # Stop any stale Live (defensive: shouldn't happen but cheap to guard).
        self._safe_stop_live()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None:
        self._compile_active = True
        self._compile_bucket_idx = int(bucket_idx)
        self._compile_bucket_shape = tuple(int(d) for d in bucket_shape)
        self._refresh()

    def on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None:
        # Best-effort heartbeat: just refresh the panel so any animated
        # spinners step forward. We also retain ``total_buckets`` for the
        # progress label.
        self._compile_total_buckets = int(total_buckets)
        self._refresh()

    def on_compile_done(self, *, bucket_idx: int) -> None:
        self._compile_active = False
        self._compile_first_done = True
        self._refresh()

    def on_phase_start(
        self, *, phase_idx: int, phase_steps: int, lr: float, optimizer: str
    ) -> None:
        self._phase_idx = int(phase_idx)
        self._phase_steps = int(phase_steps)
        self._phase_lr = float(lr)
        self._phase_optimizer = str(optimizer)
        self._phase_history.append((self._phase_idx, self._phase_optimizer, self._phase_lr))

        # Build a fresh Progress so the bar starts from zero on every
        # phase. Reusing one Progress across phases would either show
        # cumulative step counts or require manual reset gymnastics; a
        # new Progress per phase is cheaper and cleaner.
        #
        # The trailing ``loss`` column is fed through ``task.fields`` and
        # updated from ``on_step_end``; before the first step of a phase
        # there is no loss yet, so the field starts as a placeholder
        # rather than a synthetic zero (which would mislead readers).
        self._phase_progress_bar = Progress(
            TextColumn("[bold]phase {task.fields[phase_idx]}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} steps"),
            TimeElapsedColumn(),
            TextColumn("loss [bold]{task.fields[loss]}"),
            console=self._console,
            transient=False,
            auto_refresh=False,
        )
        self._phase_task_id = self._phase_progress_bar.add_task(
            description="phase",
            total=max(self._phase_steps, 1),
            phase_idx=self._phase_idx,
            loss="-",
        )
        self._refresh()

    def on_phase_end(self, *, phase_idx: int) -> None:
        # The progress bar object is kept around so the final rendered state
        # still shows the completed phase; nothing else to do.
        self._refresh()

    def on_step_end(
        self, *, step_idx: int, phase_idx: int, loss: float, penalty: float = 0.0
    ) -> None:
        # Advance the phase progress bar and push the latest loss into the
        # bar's ``loss`` field, so the trailing column renders a live value
        # next to the step and elapsed columns. Guarded because an
        # out-of-order run can call on_step_end with no preceding
        # on_phase_start, leaving nothing to update.
        if self._phase_progress_bar is not None and self._phase_task_id is not None:
            try:
                # Penalty is appended only when it is actually charged,
                # so runs that never enable it read exactly as before.
                text = _format_loss(loss)
                if penalty > 0.0:
                    text = f"{text} +pen {_format_loss(penalty)}"
                self._phase_progress_bar.update(
                    self._phase_task_id,
                    advance=1,
                    loss=text,
                )
            except Exception:
                pass
        self._refresh()

    def on_run_end(self, *, final_loss: float) -> None:
        self._final_loss = float(final_loss)
        self._run_active = False
        # Push the terminal frame into Live, then stop. With a non-TTY
        # console (tests) Live's stop() flushes the renderable into the
        # recording buffer once, which is how export_text() ends up
        # containing the final dashboard.
        self._refresh()
        self._safe_stop_live()

    def on_message(self, *, level: str, text: str) -> None:
        self._messages.append((str(level), str(text)))
        self._refresh()

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _safe_stop_live(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:
                pass
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def _refresh(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:
                # Rendering errors must never propagate into the training
                # loop. UI is best-effort.
                pass

    def _render(self) -> RenderableType:
        children: list[RenderableType] = [
            self._render_header(),
            self._render_compile_or_progress(),
            self._render_message_log(),
        ]
        # The footer is only emitted post-run so the word "final" lands in
        # the trailing region of the buffer (the rendered-output test pins
        # on this ordering).
        if self._final_loss is not None:
            children.append(self._render_footer())
        # Constrain to half the current terminal width. Rich panels default
        # to ``expand=True`` and the dashboard otherwise sprawls across the
        # full terminal, which both wastes horizontal space and overflows
        # the buffer when the terminal is resized narrower mid-run. Wrapping
        # in ``Align.left`` with an explicit width pins the dashboard to
        # half-width regardless of terminal size; the value is recomputed on
        # each refresh so it tracks live resizes. ``max(40, …)`` keeps
        # contents legible on very narrow terminals.
        target_width = max(40, self._console.width // 2)
        return Align.left(Group(*children), width=target_width)

    def _render_header(self) -> Panel:
        elapsed = 0.0
        if self._run_start_wallclock is not None:
            elapsed = max(0.0, time.monotonic() - self._run_start_wallclock)

        body = Table.grid(padding=(0, 2))
        body.add_column(style="bold", no_wrap=True)
        body.add_column()
        body.add_row("total steps", str(self._total_steps))
        body.add_row("num phases", str(self._num_phases))
        body.add_row("elapsed", f"{elapsed:.2f}s")
        if self._phase_history:
            phase_summary = ", ".join(
                f"{idx}:{opt}@{lr:.2e}" for idx, opt, lr in self._phase_history
            )
            body.add_row("phases seen", phase_summary)
        if self._final_loss is not None:
            # The header shows the final-loss summary once the run is over.
            # The word "final" here is what the rendered-output test pins on.
            body.add_row("final loss", _format_loss(self._final_loss))

        title = "training run" if self._run_active else "training run (finished)"
        return Panel(body, title=title, border_style="cyan")

    def _render_compile_or_progress(self) -> RenderableType:
        # Until any bucket has finished compiling we keep the compile panel
        # in this slot; afterwards we always show the per-phase progress
        # bar (or a placeholder if no phase is active yet).
        if self._compile_active or not self._compile_first_done:
            return self._render_compile_panel()
        return self._render_phase_progress()

    def _render_compile_panel(self) -> Panel:
        if self._compile_bucket_idx is None:
            body: RenderableType = Text("waiting for first compile…", style="dim")
        else:
            shape_repr = (
                "x".join(str(d) for d in self._compile_bucket_shape)
                if self._compile_bucket_shape is not None
                else "?"
            )
            status = "compiling" if self._compile_active else "compiled"
            label = f"{status} bucket {self._compile_bucket_idx} (shape {shape_repr})"
            if self._compile_total_buckets is not None:
                label += f" of {self._compile_total_buckets}"
            body = Text(label)
        return Panel(body, title="compile", border_style="magenta")

    def _render_phase_progress(self) -> RenderableType:
        if self._phase_progress_bar is None:
            placeholder = Text("no active phase", style="dim")
            return Panel(placeholder, title="phase progress", border_style="green")

        # Show lr and optimizer next to the bar, so the rendered-output test
        # can pin "adamw" without a separate panel.
        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="bold")
        meta.add_column()
        meta.add_row("optimizer", str(self._phase_optimizer))
        meta.add_row("lr", f"{self._phase_lr:.2e}" if self._phase_lr is not None else "-")

        return Panel(
            Group(self._phase_progress_bar, meta),
            title=f"phase {self._phase_idx} progress",
            border_style="green",
        )

    def _render_footer(self) -> Panel:
        # Final-loss summary at the bottom of the dashboard.
        final = _format_loss(self._final_loss) if self._final_loss is not None else "-"
        body = Text.assemble(("final loss ", "bold"), final)
        return Panel(body, title="run summary", border_style="cyan")

    def _render_message_log(self) -> Panel:
        if not self._messages:
            body: RenderableType = Text("(no messages)", style="dim")
        else:
            lines = Text()
            for i, (level, text) in enumerate(self._messages):
                if i:
                    lines.append("\n")
                lines.append(f"[{level}] ", style="bold")
                lines.append(text)
            body = lines
        return Panel(body, title="messages", border_style="blue")

    # Defensive cleanup: if the user discards the instance mid-run we don't
    # want the Live thread to hang the interpreter. ``__del__`` is best
    # effort only.
    def __del__(self) -> None:  # pragma: no cover - GC-timing dependent
        try:
            self._safe_stop_live()
        except Exception:
            pass


# ``Any`` is imported for forward-compat type annotations on stub methods.
_ = Any
