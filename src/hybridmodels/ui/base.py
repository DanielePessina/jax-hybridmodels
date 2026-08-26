"""UI lifecycle protocols and the no-op ``SilentUI``.

The training loops drive a UI by calling lifecycle methods at
well-defined points. Two protocols are defined here — ``TrainingUI``
(consumed by the Optax loop) and ``EvosaxUI`` (consumed by the Evosax
loop). They share several event names but the arguments differ enough
that a merged supertype would just be confusing, so the two are kept
separate and a UI implementation that wants to support both training
loops simply implements both protocols.

UI selection
------------
Each training config has a ``verbose: bool`` field defaulting to
``True`` (the Rich live-dashboard UI); ``False`` selects ``SilentUI``.
An explicit ``ui=...`` argument to ``train_with_optax`` /
``train_with_evosax`` always overrides both.

Compile events
--------------
Per-bucket-shape JIT compilation is the dominant cost of the first
epoch. ``on_compile_start`` / ``on_compile_progress`` /
``on_compile_done`` exist so users have visible feedback during that
latency; without them, a Rich progress bar would hang on the first
bucket while the kernel compiles and look like an apparent hang.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainingUI(Protocol):
    """Callback protocol for ``train_with_optax``.

    Implementations are duck-typed (``@runtime_checkable``) so users can
    define their own UI without inheriting from this class. The default
    Rich and Silent implementations live in ``ui/optax.py`` and
    ``ui/base.py`` respectively.

    Event order during a typical run::

        on_run_start
        for each bucket shape encountered first time:
            on_compile_start  -> on_compile_progress*  -> on_compile_done
        for phase in phases:
            on_phase_start
                on_step_end (one per training step)
            on_phase_end
        on_run_end

    ``on_message`` may fire at any point for log lines (e.g. tournament
    fallback warnings).
    """

    def on_run_start(self, *, total_steps: int, num_phases: int) -> None:
        """Fires once before the first phase. ``total_steps`` is the sum across phases."""
        ...

    def on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None:
        """A bucket of shape ``bucket_shape`` is about to be JIT-compiled for the first time."""
        ...

    def on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None:
        """Periodic heartbeat during long compiles (best-effort, may not fire)."""
        ...

    def on_compile_done(self, *, bucket_idx: int) -> None:
        """The bucket at ``bucket_idx`` finished compiling."""
        ...

    def on_phase_start(
        self, *, phase_idx: int, phase_steps: int, lr: float, optimizer: str
    ) -> None:
        """Fires at the start of each phase; ``phase_steps`` is the per-phase step budget."""
        ...

    def on_phase_end(self, *, phase_idx: int) -> None:
        """Fires after the last step of a phase, before any optimiser reset."""
        ...

    def on_step_end(
        self, *, step_idx: int, phase_idx: int, loss: float, penalty: float = 0.0
    ) -> None:
        """Fires after each training step. ``step_idx`` is global; ``phase_idx`` localises it.

        ``loss`` is the **data** term only, never the combined objective:
        it is the series ``restore_best`` and early stopping act on, and
        mixing in a penalty whose weight ramps between phases would make
        successive values incomparable. ``penalty`` reports the unweighted
        bound penalty alongside it, and defaults to ``0.0`` so UIs written
        against the earlier signature keep satisfying this protocol.
        """
        ...

    def on_run_end(self, *, final_loss: float) -> None:
        """Fires once after every phase has completed (or training was aborted gracefully)."""
        ...

    def on_message(self, *, level: str, text: str) -> None:
        """Free-form log line. ``level`` is one of ``"info"``, ``"warning"``, ``"error"``."""
        ...


@runtime_checkable
class EvosaxUI(Protocol):
    """Callback protocol for ``train_with_evosax``.

    Differs from ``TrainingUI`` because evosax does not have phases or
    per-step gradient losses; instead each generation reports best/mean
    fitness across the population.
    """

    def on_run_start(self, *, num_generations: int, population_size: int) -> None:
        """Fires once before the first generation."""
        ...

    def on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None:
        """A bucket of shape ``bucket_shape`` is about to be JIT-compiled."""
        ...

    def on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None:
        """Periodic compile-time heartbeat (best-effort)."""
        ...

    def on_compile_done(self, *, bucket_idx: int) -> None:
        """The bucket finished compiling."""
        ...

    def on_generation_end(self, *, gen_idx: int, best_fitness: float, mean_fitness: float) -> None:
        """Fires once per generation with population statistics."""
        ...

    def on_run_end(self, *, best_fitness: float) -> None:
        """Fires once after the last generation."""
        ...

    def on_message(self, *, level: str, text: str) -> None:
        """Free-form log line; same level set as ``TrainingUI.on_message``."""
        ...


class SilentUI:
    """No-op UI satisfying both ``TrainingUI`` and ``EvosaxUI``.

    Selected when ``config.verbose=False`` and used in tests where stdout
    output would pollute captured logs. Every method accepts ``**kwargs``
    and returns ``None``, so it tolerates protocol drift without raising.
    """

    def on_run_start(self, **kwargs: Any) -> None:
        pass

    def on_compile_start(self, **kwargs: Any) -> None:
        pass

    def on_compile_progress(self, **kwargs: Any) -> None:
        pass

    def on_compile_done(self, **kwargs: Any) -> None:
        pass

    def on_phase_start(self, **kwargs: Any) -> None:
        pass

    def on_phase_end(self, **kwargs: Any) -> None:
        pass

    def on_step_end(self, **kwargs: Any) -> None:
        pass

    def on_generation_end(self, **kwargs: Any) -> None:
        pass

    def on_run_end(self, **kwargs: Any) -> None:
        pass

    def on_message(self, **kwargs: Any) -> None:
        pass
