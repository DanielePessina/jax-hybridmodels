from hybridmodels.ui import EvosaxUI, SilentUI, TrainingUI
from hybridmodels.ui.testing import RecordingUI


def test_silent_ui_produces_no_output(capsys) -> None:
    ui = SilentUI()
    ui.on_run_start(total_steps=10, num_phases=1)
    ui.on_compile_start(bucket_idx=0, bucket_shape=(4, 8))
    ui.on_compile_progress(bucket_idx=0, total_buckets=2)
    ui.on_compile_done(bucket_idx=0)
    ui.on_phase_start(phase_idx=0, phase_steps=10, lr=1e-3, optimizer="adamw")
    ui.on_step_end(step_idx=0, phase_idx=0, loss=0.5)
    ui.on_phase_end(phase_idx=0)
    ui.on_run_end(final_loss=0.1)
    ui.on_message(level="info", text="done")
    ui.on_generation_end(gen_idx=0, best_fitness=0.2, mean_fitness=0.5)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_silent_ui_satisfies_training_protocol() -> None:
    assert isinstance(SilentUI(), TrainingUI)


def test_silent_ui_satisfies_evosax_protocol() -> None:
    assert isinstance(SilentUI(), EvosaxUI)


def test_recording_ui_satisfies_training_protocol() -> None:
    assert isinstance(RecordingUI(), TrainingUI)


def test_recording_ui_satisfies_evosax_protocol() -> None:
    assert isinstance(RecordingUI(), EvosaxUI)


def test_recording_ui_starts_with_empty_events() -> None:
    ui = RecordingUI()
    assert ui.events == []


def test_recording_ui_records_training_lifecycle() -> None:
    ui = RecordingUI()
    ui.on_run_start(total_steps=20, num_phases=2)
    assert len(ui.events) == 1
    assert ui.events[-1] == ("on_run_start", {"total_steps": 20, "num_phases": 2})

    ui.on_compile_start(bucket_idx=1, bucket_shape=(3, 7))
    assert len(ui.events) == 2
    assert ui.events[-1] == ("on_compile_start", {"bucket_idx": 1, "bucket_shape": (3, 7)})

    ui.on_compile_progress(bucket_idx=1, total_buckets=3)
    assert len(ui.events) == 3
    assert ui.events[-1] == ("on_compile_progress", {"bucket_idx": 1, "total_buckets": 3})

    ui.on_compile_done(bucket_idx=1)
    assert len(ui.events) == 4
    assert ui.events[-1] == ("on_compile_done", {"bucket_idx": 1})

    ui.on_phase_start(phase_idx=0, phase_steps=10, lr=1e-3, optimizer="adamw")
    assert len(ui.events) == 5
    assert ui.events[-1] == (
        "on_phase_start",
        {"phase_idx": 0, "phase_steps": 10, "lr": 1e-3, "optimizer": "adamw"},
    )

    ui.on_step_end(step_idx=4, phase_idx=0, loss=0.42)
    assert len(ui.events) == 6
    assert ui.events[-1] == ("on_step_end", {"step_idx": 4, "phase_idx": 0, "loss": 0.42})

    ui.on_phase_end(phase_idx=0)
    assert len(ui.events) == 7
    assert ui.events[-1] == ("on_phase_end", {"phase_idx": 0})

    ui.on_run_end(final_loss=0.05)
    assert len(ui.events) == 8
    assert ui.events[-1] == ("on_run_end", {"final_loss": 0.05})

    ui.on_message(level="warning", text="diffrax retry")
    assert len(ui.events) == 9
    assert ui.events[-1] == ("on_message", {"level": "warning", "text": "diffrax retry"})


def test_recording_ui_records_evosax_lifecycle() -> None:
    ui = RecordingUI()
    ui.on_run_start(num_generations=50, population_size=32)
    assert ui.events[-1] == ("on_run_start", {"num_generations": 50, "population_size": 32})

    ui.on_generation_end(gen_idx=3, best_fitness=0.1, mean_fitness=0.3)
    assert ui.events[-1] == (
        "on_generation_end",
        {"gen_idx": 3, "best_fitness": 0.1, "mean_fitness": 0.3},
    )

    ui.on_run_end(best_fitness=0.05)
    assert ui.events[-1] == ("on_run_end", {"best_fitness": 0.05})


def test_recording_ui_each_event_appends_exactly_one_entry() -> None:
    ui = RecordingUI()
    calls = [
        ("on_run_start", {"total_steps": 1, "num_phases": 1}),
        ("on_compile_start", {"bucket_idx": 0, "bucket_shape": (1, 1)}),
        ("on_compile_progress", {"bucket_idx": 0, "total_buckets": 1}),
        ("on_compile_done", {"bucket_idx": 0}),
        ("on_phase_start", {"phase_idx": 0, "phase_steps": 1, "lr": 1e-3, "optimizer": "adamw"}),
        ("on_step_end", {"step_idx": 0, "phase_idx": 0, "loss": 0.0}),
        ("on_phase_end", {"phase_idx": 0}),
        ("on_generation_end", {"gen_idx": 0, "best_fitness": 0.0, "mean_fitness": 0.0}),
        ("on_run_end", {"final_loss": 0.0}),
        ("on_message", {"level": "info", "text": "hi"}),
    ]
    for i, (method, kwargs) in enumerate(calls, start=1):
        getattr(ui, method)(**kwargs)
        assert len(ui.events) == i
