"""Rich-based ``EvosaxUI`` for ``train_with_evosax``.

Mirrors :class:`hybridmodels.ui.optax.RichTrainingUI` but with the panel
set adjusted for an evolutionary outer loop: phases and per-step
losses are replaced by generations and population statistics. A single
:class:`rich.live.Live` is started in :meth:`RichEvosaxUI.on_run_start`
and stopped in :meth:`RichEvosaxUI.on_run_end`. The live renderable is
a :class:`rich.console.Group` that swaps panels as the run progresses:

* a header :class:`rich.panel.Panel` with run-level info (generations,
  population size, elapsed wall-clock, and post-run final best
  fitness);
* a compile-progress panel shown only between ``on_compile_start`` and
  the matching ``on_compile_done``; once any bucket has finished
  compiling that slot is replaced by a
  :class:`rich.progress.Progress` bar tracking generations completed /
  total;
* a small :class:`rich.table.Table` of the most recent
  ``recent_generations`` generations with columns
  ``gen | best_fitness | mean_fitness | best-so-far``, throttled by
  ``log_every``;
* a message-log panel with the most recent ``recent_messages`` lines.

The class is defensive about event ordering. An event that arrives
before the state it references, ``on_run_end`` with no generations seen
for instance, becomes a no-op rather than an assertion. Graceful-abort
paths and recording-UI tests can legitimately fire events out of order.

This class deliberately does **not** subclass ``RichTrainingUI``. The
two share a few small rendering helpers, but the bodies are short enough
that duplication costs less than a shared base. Extracting one would
force both UIs to negotiate every future panel change through a single
supertype.
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


def _format_fitness(value: float) -> str:
    """Format a fitness value to four decimal places.

    Same convention as ``hybridmodels.ui.optax._format_loss``, so the
    rendered-output tests for both UIs can share one ``\\d\\.\\d{4,}``
    regex.
    """
    return f"{float(value):.4f}"


class RichEvosaxUI:
    """Live Rich dashboard satisfying ``hybridmodels.ui.base.EvosaxUI``.

    Parameters
    ----------
    console:
        Optional :class:`rich.console.Console`. ``None`` constructs a default
        ``Console()``. Tests typically pass a recording console
        (``Console(record=True, force_terminal=False, ...)``) so the
        rendered final state can be asserted on.
    log_every:
        Throttle for the recent-generation table. ``log_every=k`` records
        generations where ``gen_idx % k == 0``. ``None`` (the default) defers
        the choice to :meth:`on_run_start`, which sets it to
        ``max(1, num_generations // 5)`` so the table accumulates to exactly
        five rows over the run instead of sliding past a constantly-changing
        last-five window. An explicit integer always overrides the
        auto-scale.
    recent_generations:
        Maximum number of rows the recent-generation table holds.
    recent_messages:
        Maximum number of lines the message-log panel holds.

    Notes
    -----
    The instance carries one :class:`rich.live.Live` between
    ``on_run_start`` and ``on_run_end``. State is reset in
    ``on_run_start`` so a single instance can be reused across sequential
    runs (used in tests).
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        log_every: int | None = None,
        recent_generations: int = 5,
        recent_messages: int = 5,
    ) -> None:
        if log_every is not None and log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        self._console: Console = console if console is not None else Console()
        # ``None`` means "auto-scale at on_run_start". A concrete int is the
        # user's override and is locked in immediately. on_run_start reads
        # the flag, so once an explicit value is set the auto-scale path is
        # off for the life of this instance.
        self._auto_log_every: bool = log_every is None
        self._log_every: int = 1 if log_every is None else int(log_every)
        # Each row is (gen_idx, best_fitness, mean_fitness, best_so_far).
        # The deque maxlen pins visible history without unbounded growth.
        self._generations: deque[tuple[int, float, float, float]] = deque(maxlen=recent_generations)
        self._messages: deque[tuple[str, str]] = deque(maxlen=recent_messages)

        # Run-level state (set in on_run_start, read by _render).
        self._run_start_wallclock: float | None = None
        self._num_generations: int = 0
        self._population_size: int = 0
        self._final_fitness: float | None = None
        self._run_active: bool = False

        # Generation-progress state. The bar is rebuilt on every
        # ``on_run_start`` so a reused instance does not show a stale bar
        # advancing past its old total.
        self._gen_progress_bar: Progress | None = None
        self._gen_task_id: TaskID | None = None
        # Best-so-far tracker, shown in both the recent-generation table
        # and the header summary.
        self._best_so_far: float | None = None

        # Compile-panel state. Same slot semantics as ``RichTrainingUI``:
        # the compile panel and the generation-progress bar share the
        # vertical position; once any bucket finishes compiling we
        # leave that slot to the progress bar for the rest of the run.
        self._compile_active: bool = False
        self._compile_bucket_idx: int | None = None
        self._compile_bucket_shape: tuple[int, ...] | None = None
        self._compile_total_buckets: int | None = None
        self._compile_first_done: bool = False

        self._live: Live | None = None

    # ------------------------------------------------------------------
    # EvosaxUI protocol implementation.
    # ------------------------------------------------------------------

    def on_run_start(self, *, num_generations: int, population_size: int) -> None:
        self._run_start_wallclock = time.monotonic()
        self._num_generations = int(num_generations)
        self._population_size = int(population_size)
        self._final_fitness = None
        self._run_active = True
        # Auto-scale ``log_every`` so the recent-generation table grows to
        # exactly five rows over the run rather than constantly overwriting a
        # sliding window. ``recent_generations`` is the deque cap (default 5);
        # spacing log_every at ``num_generations // 5`` makes the deque fill
        # exactly once across the whole run. An explicit log_every passed at
        # construction time disables this and is honoured verbatim.
        if self._auto_log_every:
            self._log_every = max(1, self._num_generations // 5)
        # Reset per-generation / compile / message state so a reused instance
        # does not bleed prior-run rows into the new run.
        self._generations.clear()
        self._messages.clear()
        self._best_so_far = None
        self._compile_active = False
        self._compile_first_done = False
        self._compile_bucket_idx = None
        self._compile_bucket_shape = None
        self._compile_total_buckets = None

        # Build a fresh Progress so the bar starts from zero on every
        # run. Reusing one Progress across runs would either show a
        # stale total or require manual reset gymnastics; a new
        # Progress per run is cheaper and cleaner.
        self._gen_progress_bar = Progress(
            TextColumn("[bold]generation"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
            auto_refresh=False,
        )
        self._gen_task_id = self._gen_progress_bar.add_task(
            description="generation",
            total=max(self._num_generations, 1),
        )

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
        # Best-effort heartbeat; we retain ``total_buckets`` for the label.
        self._compile_total_buckets = int(total_buckets)
        self._refresh()

    def on_compile_done(self, *, bucket_idx: int) -> None:
        self._compile_active = False
        self._compile_first_done = True
        self._refresh()

    def on_generation_end(self, *, gen_idx: int, best_fitness: float, mean_fitness: float) -> None:
        idx = int(gen_idx)
        best = float(best_fitness)
        mean = float(mean_fitness)

        # Best-so-far is monotone non-increasing. Update it before
        # recording the row, so the table shows the post-update value.
        # This mirrors the best-ever bookkeeping the training loop keeps.
        if self._best_so_far is None or best < self._best_so_far:
            self._best_so_far = best

        # Advance the generation progress bar. Guarded because an
        # out-of-order run can call on_generation_end with no preceding
        # on_run_start.
        if self._gen_progress_bar is not None and self._gen_task_id is not None:
            try:
                self._gen_progress_bar.advance(self._gen_task_id, advance=1)
            except Exception:
                pass

        # Throttle: only record rows where ``gen_idx % log_every == 0``.
        if idx % self._log_every == 0:
            self._generations.append((idx, best, mean, float(self._best_so_far)))
        self._refresh()

    def on_run_end(self, *, best_fitness: float) -> None:
        self._final_fitness = float(best_fitness)
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
            self._render_generation_table(),
            self._render_message_log(),
        ]
        # The footer is only emitted post-run so the word "final" lands in
        # the trailing region of the buffer (the rendered-output test pins
        # on this ordering).
        if self._final_fitness is not None:
            children.append(self._render_footer())
        # Constrain to half the current terminal width. The matching
        # comment in ``hybridmodels.ui.optax.RichTrainingUI._render`` has
        # the reasoning: sprawling panels and resize-time overflow.
        target_width = max(40, self._console.width // 2)
        return Align.left(Group(*children), width=target_width)

    def _render_header(self) -> Panel:
        elapsed = 0.0
        if self._run_start_wallclock is not None:
            elapsed = max(0.0, time.monotonic() - self._run_start_wallclock)

        body = Table.grid(padding=(0, 2))
        body.add_column(style="bold", no_wrap=True)
        body.add_column()
        body.add_row("generations", str(self._num_generations))
        body.add_row("population", str(self._population_size))
        body.add_row("elapsed", f"{elapsed:.2f}s")
        if self._best_so_far is not None:
            body.add_row("best so far", _format_fitness(self._best_so_far))
        if self._final_fitness is not None:
            # The header shows the final fitness once the run is over. The
            # word "final" here is what the rendered-output test pins on.
            body.add_row("final fitness", _format_fitness(self._final_fitness))

        title = "evosax run" if self._run_active else "evosax run (finished)"
        return Panel(body, title=title, border_style="cyan")

    def _render_compile_or_progress(self) -> RenderableType:
        # Until any bucket has finished compiling we keep the compile panel
        # in this slot; afterwards we always show the generation-progress
        # bar (or a placeholder if the run hasn't started yet).
        if self._compile_active or not self._compile_first_done:
            return self._render_compile_panel()
        return self._render_generation_progress()

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

    def _render_generation_progress(self) -> RenderableType:
        if self._gen_progress_bar is None:
            placeholder = Text("no active run", style="dim")
            return Panel(placeholder, title="generation progress", border_style="green")
        return Panel(
            self._gen_progress_bar,
            title="generation progress",
            border_style="green",
        )

    def _render_generation_table(self) -> Panel:
        table = Table(show_header=True, expand=True)
        table.add_column("gen", justify="right", no_wrap=True)
        table.add_column("best_fitness", justify="right", no_wrap=True)
        table.add_column("mean_fitness", justify="right", no_wrap=True)
        table.add_column("best-so-far", justify="right", no_wrap=True)
        for gen_idx, best, mean, best_so_far in self._generations:
            table.add_row(
                str(gen_idx),
                _format_fitness(best),
                _format_fitness(mean),
                _format_fitness(best_so_far),
            )
        if not self._generations:
            table.add_row("-", "-", "-", "-")
        return Panel(table, title="recent generations", border_style="yellow")

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

    def _render_footer(self) -> Panel:
        # Final-fitness summary at the bottom of the dashboard.
        final = _format_fitness(self._final_fitness) if self._final_fitness is not None else "-"
        body = Text.assemble(("final fitness ", "bold"), final)
        return Panel(body, title="run summary", border_style="cyan")

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
