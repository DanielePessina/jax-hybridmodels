"""UI lifecycle protocols and the no-op ``SilentUI``.

Progress reporting is callback-based, not log-based. A training loop
calls lifecycle methods on the UI object it was handed, and the UI
decides what to do with them. Writing your own means implementing the
methods below; nothing inherits from anything.

``TrainingUI`` is what the Optax loop calls, ``EvosaxUI`` what the Evosax
loop calls. Several event names are shared, but the arguments differ
enough that a merged supertype would only confuse. A UI serving both
loops implements both protocols, as ``SilentUI`` does.

UI selection
------------
Each training config has a ``verbose: bool`` field defaulting to
``True`` (the Rich live-dashboard UI); ``False`` selects ``SilentUI``.
An explicit ``ui=...`` argument to ``train_with_optax`` /
``train_with_evosax`` always overrides both.

Compile events
--------------
JIT compilation, once per bucket shape, dominates the first epoch and
can take tens of seconds. ``on_compile_start``, ``on_compile_progress``
and ``on_compile_done`` exist so a progress bar does not sit at zero
through it and make the run look hung.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainingUI(Protocol):
    """Callback protocol for ``train_with_optax``.

    Duck-typed (``@runtime_checkable``), so a custom UI just defines the
    methods. Shipped: ``RichTrainingUI`` in ``ui/optax.py`` and
    ``SilentUI`` below.

    Event order during a typical run::

        on_run_start
        for each bucket shape encountered first time:
            on_compile_start  -> on_compile_progress*  -> on_compile_done
        for phase in phases:
            on_phase_start
                on_step_end (one per training step)
            on_phase_end
        on_run_end

    ``on_message`` can fire at any point, for log lines such as the
    tournament's fallback warning.
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
        """Fires after each training step. ``step_idx`` counts within the phase.

        ``loss`` is the **data** term alone, the series ``restore_best``
        and early stopping act on; a penalty whose weight ramps between
        phases would make successive values incomparable. ``penalty``
        reports the unweighted bound penalty next to it, defaulting to
        ``0.0`` so a UI written against the earlier signature still
        satisfies this protocol.
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

    Differs from ``TrainingUI`` because the evosax loop has no phases and
    no per-step gradient loss. Each generation reports the best and mean
    fitness across its population instead.

    Event order during a typical run::

        on_run_start
        on_compile_start -> on_compile_progress* -> on_compile_done
        on_generation_end (one per generation)
        on_run_end
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

    Selected when ``config.verbose=False``, and used in tests where stdout
    would pollute captured logs. Every method takes ``**kwargs``, so a new
    event argument never breaks it.
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
