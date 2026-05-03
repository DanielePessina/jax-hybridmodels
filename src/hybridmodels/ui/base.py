from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainingUI(Protocol):
    def on_run_start(self, *, total_steps: int, num_phases: int) -> None: ...
    def on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None: ...
    def on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None: ...
    def on_compile_done(self, *, bucket_idx: int) -> None: ...
    def on_phase_start(
        self, *, phase_idx: int, phase_steps: int, lr: float, optimizer: str
    ) -> None: ...
    def on_phase_end(self, *, phase_idx: int) -> None: ...
    def on_step_end(self, *, step_idx: int, phase_idx: int, loss: float) -> None: ...
    def on_run_end(self, *, final_loss: float) -> None: ...
    def on_message(self, *, level: str, text: str) -> None: ...


@runtime_checkable
class EvosaxUI(Protocol):
    def on_run_start(self, *, num_generations: int, population_size: int) -> None: ...
    def on_compile_start(self, *, bucket_idx: int, bucket_shape: tuple[int, ...]) -> None: ...
    def on_compile_progress(self, *, bucket_idx: int, total_buckets: int) -> None: ...
    def on_compile_done(self, *, bucket_idx: int) -> None: ...
    def on_generation_end(
        self, *, gen_idx: int, best_fitness: float, mean_fitness: float
    ) -> None: ...
    def on_run_end(self, *, best_fitness: float) -> None: ...
    def on_message(self, *, level: str, text: str) -> None: ...


class SilentUI:
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
