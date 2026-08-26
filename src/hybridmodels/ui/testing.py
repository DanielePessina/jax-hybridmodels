"""Spy UI for tests: records every lifecycle call instead of rendering it.

Used by ``tests/test_ui_callbacks.py`` and any training test that needs
to assert on which events fired and in what order. It lives under
``hybridmodels.ui.testing`` so it stays out of what ``hybridmodels.ui``
re-exports.
"""

from typing import Any


class RecordingUI:
    """Test spy implementing both ``TrainingUI`` and ``EvosaxUI``.

    Every lifecycle method appends ``(name, kwargs_copy)`` to ``self.events``.
    Tests can then assert on the sequence of names, the count of a particular
    event, or the kwargs of a specific call. ``kwargs`` are shallow-copied so
    later mutation of the caller's argument dict does not retroactively
    change recorded values.

    Attributes
    ----------
    events : list[tuple[str, dict[str, Any]]]
        Ordered log of ``(method_name, kwargs)`` pairs, one per fired event.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.events.append((name, dict(kwargs)))

    def on_run_start(self, **kwargs: Any) -> None:
        self._record("on_run_start", kwargs)

    def on_compile_start(self, **kwargs: Any) -> None:
        self._record("on_compile_start", kwargs)

    def on_compile_progress(self, **kwargs: Any) -> None:
        self._record("on_compile_progress", kwargs)

    def on_compile_done(self, **kwargs: Any) -> None:
        self._record("on_compile_done", kwargs)

    def on_phase_start(self, **kwargs: Any) -> None:
        self._record("on_phase_start", kwargs)

    def on_phase_end(self, **kwargs: Any) -> None:
        self._record("on_phase_end", kwargs)

    def on_step_end(self, **kwargs: Any) -> None:
        self._record("on_step_end", kwargs)

    def on_generation_end(self, **kwargs: Any) -> None:
        self._record("on_generation_end", kwargs)

    def on_run_end(self, **kwargs: Any) -> None:
        self._record("on_run_end", kwargs)

    def on_message(self, **kwargs: Any) -> None:
        self._record("on_message", kwargs)
