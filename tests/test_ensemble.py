"""Integration tests for the ensemble helpers.

Covers the four pieces added for bagging-style and seed-style ensembles:

- ``make_bootstrap_dataset`` — resample experiments with replacement,
  re-bucketing the irregular data.
- ``train_seed_ensemble`` — rank warm-start seeds with the tournament,
  fully train the top-k, return them ranked by loss.
- ``ensemble_predictions`` — average member predictions per bucket.
- ``predict_dense`` — evaluate a trained model on a fine per-experiment
  time grid without the diffraxtra dependency.

All drive the real public pipeline (dataset building, training,
prediction), not inner kernels.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import pytest
from _harness import (
    OmegaPredictor,
    make_oscillator_dataset,
    make_oscillator_simulate_fn,
    oscillator_state_to_output,
)

import hybridmodels as hm
from hybridmodels.data import Dataset, make_bootstrap_dataset
from hybridmodels.prediction import predict_dense
from hybridmodels.solver import SolverConfig
from hybridmodels.training.optax import (
    OptaxTrainingConfig,
    train_bootstrap_ensemble,
    train_seed_ensemble,
)


def solver() -> SolverConfig:
    return SolverConfig(
        solver=diffrax.Tsit5(),
        rtol=1e-6,
        atol=1e-8,
        max_steps=4096,
        dt0=0.05,
    )


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return make_oscillator_dataset()


class TestBootstrapDataset:
    def test_resamples_with_replacement_and_rebuckets(self, dataset):
        # m > N forces duplicates (pigeonhole), proving with-replacement.
        n_orig = len(dataset._experiments)
        boot = make_bootstrap_dataset(dataset, key=jax.random.PRNGKey(0), n_experiments=8)
        assert len(boot._experiments) == 8
        ids = [e.exp_id for e in boot._experiments]
        assert len(ids) == 8 and len(set(ids)) < 8  # a duplicate must exist
        # Re-bucketing produced valid payloads with 8 rows.
        assert sum(int(bp.y_observed.shape[0]) for bp in boot.bucket_payloads) == 8
        assert boot.output_channel_names == dataset.output_channel_names
        assert n_orig == 4  # sanity: the fixture really has 4 experiments

    def test_default_size_equals_source(self, dataset):
        boot = make_bootstrap_dataset(dataset, key=jax.random.PRNGKey(1))
        assert len(boot._experiments) == len(dataset._experiments)

    def test_deterministic_given_key(self, dataset):
        a = make_bootstrap_dataset(dataset, key=jax.random.PRNGKey(7))
        b = make_bootstrap_dataset(dataset, key=jax.random.PRNGKey(7))
        assert [e.exp_id for e in a._experiments] == [e.exp_id for e in b._experiments]
        c = make_bootstrap_dataset(dataset, key=jax.random.PRNGKey(8))
        assert [e.exp_id for e in a._experiments] != [e.exp_id for e in c._experiments]

    def test_requires_source_experiments(self, dataset):
        bare = Dataset(
            bucket_payloads=dataset.bucket_payloads,
            output_channel_names=dataset.output_channel_names,
            covariate_names=dataset.covariate_names,
        )
        with pytest.raises(ValueError, match="source experiments"):
            make_bootstrap_dataset(bare, key=jax.random.PRNGKey(0))

    def test_handles_irregular_per_channel_timestamps(self):
        # An experiment whose channels are measured on disjoint times must
        # still re-bucket correctly after resampling.
        from hybridmodels.data import ChannelObs, make_dataset, make_experiment

        ts_long = jnp.linspace(0.0, 5.0, 10)
        ts_short = ts_long[:5]
        experiments = []
        for i in range(4):
            experiments.append(
                make_experiment(
                    covariates={"id": float(i)},
                    channels={
                        "a": ChannelObs(ts=ts_long, values=jnp.cos(ts_long)),
                        "b": ChannelObs(ts=ts_short, values=jnp.sin(ts_short)),
                    },
                    y0_fn=lambda c, ch: jnp.zeros(2),
                    exp_id=f"e{i}",
                )
            )
        ds = make_dataset(experiments, output_channel_names=("a", "b"))
        boot = make_bootstrap_dataset(ds, key=jax.random.PRNGKey(3))
        for bp in boot.bucket_payloads:
            # Mask dimension must still line up with the union axis.
            assert bp.mask.shape == bp.y_observed.shape


class TestSeedEnsemble:
    def _config(self):
        return OptaxTrainingConfig(
            steps=(6,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            tournament_steps=2,
            restore_best=False,
        )

    def _data_loss(self, predictor, dataset) -> float:
        from hybridmodels.losses import masked_mse

        bp = dataset.bucket_payloads[0]
        pred = hm.predict_bucket(
            predictor,
            bp,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
        )
        return float(masked_mse(pred, bp))

    def _avg_data_loss(self, predictor, dataset) -> float:
        """Per-bucket average of the data loss, matching the trainer's step average."""
        from hybridmodels.losses import masked_mse

        total = 0.0
        for bp in dataset.bucket_payloads:
            pred = hm.predict_bucket(
                predictor,
                bp,
                simulate_fn=make_oscillator_simulate_fn(),
                state_to_output=oscillator_state_to_output,
                solver=solver(),
            )
            total += float(masked_mse(pred, bp))
        return total / len(dataset.bucket_payloads)

    def test_trains_k_best_members_ranked(self, dataset):
        start = OmegaPredictor(1.5)
        start_loss = self._data_loss(start, dataset)
        ranked = train_seed_ensemble(
            start,
            dataset,
            self._config(),
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            n_seeds=4,
            k_best=2,
            key=jax.random.PRNGKey(0),
        )
        assert len(ranked) == 2
        # Sorted ascending by final loss.
        losses = [loss_val for loss_val, _ in ranked]
        assert losses == sorted(losses)
        # Selection + training genuinely helped: the best member beats the
        # untrained input's loss (the oscillator landscape aliases, so we
        # assert improvement rather than convergence to a specific omega).
        assert losses[0] < start_loss

    def test_k_best_defaults_to_n_seeds(self, dataset):
        ranked = train_seed_ensemble(
            OmegaPredictor(1.5),
            dataset,
            self._config(),
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            n_seeds=3,
            key=jax.random.PRNGKey(0),
        )
        assert len(ranked) == 3

    def test_deterministic_given_key(self, dataset):
        cfg = self._config()

        def run(key):
            return [
                (loss_val, float(p.omega))
                for loss_val, p in train_seed_ensemble(
                    OmegaPredictor(1.5),
                    dataset,
                    cfg,
                    simulate_fn=make_oscillator_simulate_fn(),
                    state_to_output=oscillator_state_to_output,
                    solver=solver(),
                    n_seeds=2,
                    k_best=1,
                    key=key,
                )
            ]

        assert run(jax.random.PRNGKey(0)) == run(jax.random.PRNGKey(0))

    def test_reported_loss_labels_the_returned_member_under_restore_best(self, dataset):
        # With restore_best=True the members come back from their best step,
        # so the reported ``final_loss`` must be the loss measured at those
        # parameters — not the last step's loss, which describes a different
        # model. Re-measuring the returned member's data loss must agree.
        cfg = OptaxTrainingConfig(
            steps=(6,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            tournament_steps=2,
            restore_best=True,
        )
        ranked = train_seed_ensemble(
            OmegaPredictor(1.5),
            dataset,
            cfg,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            n_seeds=3,
            k_best=2,
            key=jax.random.PRNGKey(0),
        )
        assert len(ranked) == 2
        for final_loss, member in ranked:
            measured = self._avg_data_loss(member, dataset)
            assert abs(final_loss - measured) < 1e-3, (
                f"reported loss {final_loss:.6f} does not label the returned "
                f"member (measured {measured:.6f})"
            )


class _BracketingUI:
    """Counts run brackets; fails the test if any event escapes one.

    Pins the per-member UI lifecycle: every ensemble member is its own
    ``on_run_start``/``on_run_end`` bracket, and the warm-up compile lands
    inside the first member's bracket.
    """

    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.phase_starts = 0
        self._active = False
        self.invalid: list[str] = []

    def _check(self, event: str) -> None:
        if not self._active:
            self.invalid.append(f"{event} outside a run bracket")

    def on_run_start(self, *, total_steps: int, num_phases: int) -> None:
        if self._active:
            self.invalid.append("on_run_start during an active run")
        self._active = True
        self.starts += 1

    def on_compile_start(self, **kwargs) -> None:
        self._check("on_compile_start")

    def on_compile_progress(self, **kwargs) -> None:
        pass

    def on_compile_done(self, **kwargs) -> None:
        pass

    def on_phase_start(self, **kwargs) -> None:
        self._check("on_phase_start")
        self.phase_starts += 1

    def on_phase_end(self, **kwargs) -> None:
        self._check("on_phase_end")

    def on_step_end(self, **kwargs) -> None:
        self._check("on_step_end")

    def on_run_end(self, *, final_loss: float) -> None:
        self._check("on_run_end")
        if self._active:
            self._active = False
            self.ends += 1

    def on_message(self, **kwargs) -> None:
        pass


def test_each_member_is_its_own_ui_run(dataset):
    # Regression: _build_training_kernels used to fire on_run_start
    # exactly once while _run_phases fired on_run_end per member,
    # leaving the bracket unbalanced and a live Rich dashboard frozen
    # from member 2 onward. Every member must be bracketed, and the
    # warm-up compile must land inside the first member's bracket.
    ui = _BracketingUI()
    ranked = train_seed_ensemble(
        OmegaPredictor(1.5),
        dataset,
        OptaxTrainingConfig(
            steps=(6,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            tournament_steps=2,
        ),
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver(),
        n_seeds=3,
        k_best=2,
        key=jax.random.PRNGKey(0),
        ui=ui,
    )
    assert len(ranked) == 2
    assert ui.invalid == []
    assert ui.starts == 2 and ui.ends == 2
    assert ui.phase_starts == 2  # one phase per member run


def test_bootstrap_members_are_ui_bracketed(dataset):
    # The bootstrap loop brackets across resamples as well as members:
    # one on_run_start/on_run_end pair per trained member, never a
    # straddling or nested run.
    ui = _BracketingUI()
    train_bootstrap_ensemble(
        OmegaPredictor(1.5),
        dataset,
        OptaxTrainingConfig(
            steps=(4,),
            lr=(1e-2,),
            optimizer=("adamw",),
            reset_optimiser_state=(False,),
            tournament_steps=2,
        ),
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver(),
        n_bootstraps=2,
        n_seeds=1,
        key=jax.random.PRNGKey(0),
        ui=ui,
    )
    assert ui.invalid == []
    assert ui.starts == 2 and ui.ends == 2
    assert ui.phase_starts == 2


def test_all_seeds_failed_fallback_closes_the_ui_bracket(dataset):
    # When every tournament attempt fails (here: a simulate_fn that always
    # returns NaN, so every score is non-finite), the ensemble falls back
    # to the input predictors. The fallback must still close the run
    # bracket it opened, or a live dashboard hangs after it.
    ui = _BracketingUI()

    def nan_simulate_fn(predictors, ts, covariates, y0, solver):
        return jnp.full((ts.shape[0], 2), jnp.nan)

    with pytest.warns(RuntimeWarning, match="every seed failed"):
        members = train_seed_ensemble(
            OmegaPredictor(1.5),
            dataset,
            OptaxTrainingConfig(
                steps=(2,),
                lr=(1e-2,),
                optimizer=("adamw",),
                reset_optimiser_state=(False,),
                tournament_steps=2,
            ),
            simulate_fn=nan_simulate_fn,
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            n_seeds=3,
            k_best=1,
            key=jax.random.PRNGKey(0),
            ui=ui,
        )
    assert len(members) == 1 and members[0][0] == float("inf")
    assert ui.invalid == []
    assert ui.starts == 1 and ui.ends == 1  # bracket balanced on the fallback


class TestEnsemblePredictions:
    def test_averages_member_predictions(self, dataset):
        members = (OmegaPredictor(1.2), OmegaPredictor(1.8))
        ensemble = hm.ensemble_predictions(
            members,
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
        )
        assert len(ensemble) == len(dataset.bucket_payloads)
        per_bucket = [
            [
                hm.predict_bucket(
                    m,
                    bp,
                    simulate_fn=make_oscillator_simulate_fn(),
                    state_to_output=oscillator_state_to_output,
                    solver=solver(),
                )
                for m in members
            ]
            for bp in dataset.bucket_payloads
        ]
        for b, bp in enumerate(dataset.bucket_payloads):
            expected = jnp.mean(jnp.stack(per_bucket[b]), axis=0)
            assert ensemble[b].shape == (bp.y_observed.shape[0], bp.y_observed.shape[1], 1)
            assert jnp.allclose(ensemble[b], expected)


class TestPredictDense:
    def test_returns_dense_per_experiment_grid(self, dataset):
        dense = predict_dense(
            OmegaPredictor(1.0),
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            n_points=100,
        )
        assert len(dense) == len(dataset.bucket_payloads)
        for i, bp in enumerate(dataset.bucket_payloads):
            assert dense[i].shape == (bp.y_observed.shape[0], 100, 1)

    def test_ts_grid_matches_predict_dataset_at_measured_times(self, dataset):
        ts_measured = dataset.bucket_payloads[0].ts[0]
        dense = predict_dense(
            OmegaPredictor(1.0),
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            ts_grid=ts_measured,
        )
        direct = hm.predict_dataset(
            OmegaPredictor(1.0),
            dataset,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
        )
        for d, x in zip(dense, direct, strict=True):
            assert d.shape == x.shape
            assert jnp.allclose(d, x, atol=1e-4)


def test_ensemble_flow_end_to_end(dataset):
    """Bootstrap samples -> train a seed ensemble per sample -> average.

    A compact end-to-end of the whole ensemble story, so a regression in
    any piece shows up here.
    """
    cfg = OptaxTrainingConfig(
        steps=(4,),
        lr=(1e-2,),
        optimizer=("adamw",),
        reset_optimiser_state=(False,),
        tournament_steps=2,
        restore_best=False,
    )
    all_members = []
    for s in range(2):
        boot = make_bootstrap_dataset(dataset, key=jax.random.PRNGKey(s))
        ranked = train_seed_ensemble(
            OmegaPredictor(1.5),
            boot,
            cfg,
            simulate_fn=make_oscillator_simulate_fn(),
            state_to_output=oscillator_state_to_output,
            solver=solver(),
            n_seeds=3,
            k_best=2,
            key=jax.random.PRNGKey(s),
        )
        all_members.extend(p for _, p in ranked)
    assert len(all_members) == 4
    ensemble = hm.ensemble_predictions(
        tuple(all_members),
        dataset,
        simulate_fn=make_oscillator_simulate_fn(),
        state_to_output=oscillator_state_to_output,
        solver=solver(),
    )
    assert len(ensemble) == len(dataset.bucket_payloads)
