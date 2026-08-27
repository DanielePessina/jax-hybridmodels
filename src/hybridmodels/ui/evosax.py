"""Rich-based ``EvosaxUI`` for ``train_with_evosax``.

Mirrors :class:`hybridmodels.ui.optax.RichTrainingUI` with phases and
per-step losses replaced by generations and population statistics. One
:class:`rich.live.Live` runs between :meth:`RichEvosaxUI.on_run_start`
and :meth:`RichEvosaxUI.on_run_end`, rendering:

* a header panel with generations, population size, elapsed wall-clock,
  and the final best fitness after the run;
* a compile-progress panel, replaced by a generations-completed progress
  bar once a bucket has finished compiling;
* a table of the most recent ``recent_generations`` generations
  (``gen | best_fitness | mean_fitness | best-so-far``), throttled by
  ``log_every``;
* a message-log panel with the most recent ``recent_messages`` lines.

Event ordering is handled defensively: an event arriving before the state
it references is a no-op rather than an assertion.

This class deliberately does not subclass ``RichTrainingUI``. The two
share a few small rendering helpers, and a shared base would force both
to negotiate every future panel change through one supertype.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from rich.console import Console, RenderableType
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

from hybridmodels.ui._rich import (
    format_value,
    half_width,
    render_compile_panel,
    render_footer,
    render_message_log,
    safe_stop,
    safe_update,
)


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
        safe_stop(self._live, self._render)
        self._live = None

    def _refresh(self) -> None:
        safe_update(self._live, self._render)

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
        return half_width(self._console, children)

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
            body.add_row("best so far", format_value(self._best_so_far))
        if self._final_fitness is not None:
            # The header shows the final fitness once the run is over. The
            # word "final" here is what the rendered-output test pins on.
            body.add_row("final fitness", format_value(self._final_fitness))

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
        return render_compile_panel(
            bucket_idx=self._compile_bucket_idx,
            bucket_shape=self._compile_bucket_shape,
            total_buckets=self._compile_total_buckets,
            active=self._compile_active,
        )

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
                format_value(best),
                format_value(mean),
                format_value(best_so_far),
            )
        if not self._generations:
            table.add_row("-", "-", "-", "-")
        return Panel(table, title="recent generations", border_style="yellow")

    def _render_message_log(self) -> Panel:
        return render_message_log(self._messages)

    def _render_footer(self) -> Panel:
        return render_footer("final fitness", self._final_fitness)

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
