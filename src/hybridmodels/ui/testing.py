from typing import Any


class RecordingUI:
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
