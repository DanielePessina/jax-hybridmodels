"""Rich-based ``TrainingUI`` for ``train_with_optax``.

One :class:`rich.live.Live` runs between
:meth:`RichTrainingUI.on_run_start` and
:meth:`RichTrainingUI.on_run_end`, rendering a
:class:`rich.console.Group` that swaps panels as the run progresses:

* a header panel with total steps, phase count, and elapsed wall-clock;
* a per-phase progress bar, rebuilt on every ``on_phase_start`` so it
  resets cleanly between phases;
* a compile-progress panel, shown between ``on_compile_start`` and
  ``on_compile_done`` and replaced permanently by the phase bar once the
  first bucket finishes compiling;
* a message-log panel with the most recent ``recent_messages`` lines.

Event ordering is handled defensively: an event arriving before the state
it references, ``on_phase_end`` with no active phase say, is a no-op
rather than an assertion. Tournament restarts and graceful aborts fire
events out of order, and a rendering failure must never take down a run.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

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

from jaxhybridmodels.ui._rich import (
    format_value,
    half_width,
    render_compile_panel,
    render_footer,
    render_message_log,
    safe_stop,
    safe_update,
)


class RichTrainingUI:
    """Live Rich dashboard satisfying ``jaxhybridmodels.ui.base.TrainingUI``.

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
        # does not bleed prior-run rows into the new run. ``_compile_first_done``
        # is deliberately kept: an ensemble runs one training loop per member
        # through the same instance, and once a compile has completed the
        # phase-progress slot must stay live for the later members instead of
        # reverting to the compile panel. A genuinely new compile still shows
        # the panel, because ``on_compile_start`` raises ``_compile_active``.
        self._phase_idx = None
        self._phase_progress_bar = None
        self._phase_task_id = None
        self._phase_history = []
        self._compile_active = False
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
        self, *, phase_idx: int, phase_steps: int, lr: float, optimizer: Any
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
                text = format_value(loss)
                if penalty > 0.0:
                    text = f"{text} +pen {format_value(penalty)}"
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
        safe_stop(self._live, self._render)
        self._live = None

    def _refresh(self) -> None:
        safe_update(self._live, self._render)

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
        return half_width(self._console, children)

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
            body.add_row("final loss", format_value(self._final_loss))

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
        return render_compile_panel(
            bucket_idx=self._compile_bucket_idx,
            bucket_shape=self._compile_bucket_shape,
            total_buckets=self._compile_total_buckets,
            active=self._compile_active,
        )

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
        return render_footer("final loss", self._final_loss)

    def _render_message_log(self) -> Panel:
        return render_message_log(self._messages)

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
