"""Rendering pieces shared by the two Rich dashboards.

``RichTrainingUI`` and ``RichEvosaxUI`` show different things, phases and
per-step losses against generations and population statistics, but they draw
several panels identically and guard their ``rich.live.Live`` identically.
Those pieces live here.

Free functions, not a base class. ``jaxhybridmodels.ui.evosax`` explains why the
two dashboards do not share a supertype: it would force both to negotiate
every future panel change through one type. Sharing widgets does not,
because a dashboard that wants a different compile panel just stops calling
this one.

The Live guards matter more than they look. A rendering failure must never
propagate into a training loop, so every call into ``Live`` here is wrapped.
A broken dashboard is an annoyance; a run that dies at step 4000 because a
panel could not lay out is not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


def format_value(value: float) -> str:
    """Format a loss or fitness to four decimal places.

    One convention for both dashboards, so the rendered-output tests can
    share a single ``\\d\\.\\d{4,}`` regex.
    """
    return f"{float(value):.4f}"


def safe_update(live: Live | None, render: Callable[[], RenderableType]) -> None:
    """Push a fresh frame into ``live``, swallowing any rendering error.

    ``render`` is passed as a callable rather than a renderable so a failure
    while *building* the frame is caught too, not just a failure inside
    ``Live.update``.
    """
    if live is None:
        return
    try:
        live.update(render())
    except Exception:
        # UI is best-effort. Never propagate into the training loop.
        pass


def safe_stop(live: Live | None, render: Callable[[], RenderableType]) -> None:
    """Push a terminal frame into ``live`` and stop it, swallowing errors.

    The final update is what makes the dashboard testable: against a non-TTY
    console, ``Live.stop()`` flushes the current renderable into the
    recording buffer exactly once, which is what ``export_text()`` returns.

    The caller still owns its ``_live`` attribute and must clear it.
    """
    if live is None:
        return
    try:
        live.update(render())
    except Exception:
        pass
    try:
        live.stop()
    except Exception:
        pass


def half_width(console: Console, children: Sequence[RenderableType]) -> RenderableType:
    """Stack ``children`` and pin the result to half the terminal width.

    Rich panels default to ``expand=True``, so an unconstrained dashboard
    sprawls across the full terminal and overflows the buffer when the
    terminal is resized narrower mid-run. The width is recomputed per frame
    so it tracks live resizes, and floored at 40 columns to stay legible on
    a very narrow terminal.
    """
    return Align.left(Group(*children), width=max(40, console.width // 2))


def render_compile_panel(
    *,
    bucket_idx: int | None,
    bucket_shape: tuple[int, ...] | None,
    total_buckets: int | None,
    active: bool,
) -> Panel:
    """The compile-progress panel, identical in both dashboards.

    Compilation runs before any step or generation, so without this the
    progress bar would sit at zero through the slowest part of a run and
    make it look hung.
    """
    if bucket_idx is None:
        body: RenderableType = Text("waiting for first compile…", style="dim")
    else:
        shape_repr = "x".join(str(d) for d in bucket_shape) if bucket_shape is not None else "?"
        status = "compiling" if active else "compiled"
        label = f"{status} bucket {bucket_idx} (shape {shape_repr})"
        if total_buckets is not None:
            label += f" of {total_buckets}"
        body = Text(label)
    return Panel(body, title="compile", border_style="magenta")


def render_message_log(messages: Sequence[tuple[str, str]]) -> Panel:
    """The ``on_message`` log panel, identical in both dashboards."""
    if not messages:
        body: RenderableType = Text("(no messages)", style="dim")
    else:
        lines = Text()
        for i, (level, text) in enumerate(messages):
            if i:
                lines.append("\n")
            lines.append(f"[{level}] ", style="bold")
            lines.append(text)
        body = lines
    return Panel(body, title="messages", border_style="blue")


def render_footer(label: str, value: float | None) -> Panel:
    """Post-run summary panel. ``label`` is ``"final loss"`` or ``"final fitness"``.

    Emitted only after the run so the word "final" lands in the trailing
    region of the recorded buffer, which the rendered-output tests pin on.
    """
    body = Text.assemble((f"{label} ", "bold"), format_value(value) if value is not None else "-")
    return Panel(body, title="run summary", border_style="cyan")
