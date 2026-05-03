from __future__ import annotations

import json

import diffrax
import equinox as eqx
import jax
import pytest

from hybridmodels import SOLVER_REGISTRY, SolverConfig, register_solver


@pytest.fixture
def restore_registry():
    snapshot = dict(SOLVER_REGISTRY)
    yield
    SOLVER_REGISTRY.clear()
    SOLVER_REGISTRY.update(snapshot)


class TestSolverRegistry:
    def test_registry_has_required_builtins(self):
        assert SOLVER_REGISTRY["Tsit5"] is diffrax.Tsit5
        assert SOLVER_REGISTRY["Kvaerno3"] is diffrax.Kvaerno3
        assert SOLVER_REGISTRY["Dopri5"] is diffrax.Dopri5
        assert SOLVER_REGISTRY["Heun"] is diffrax.Heun

    def test_register_solver_extends_registry(self, restore_registry):
        class MyCustomSolver(diffrax.Heun):
            pass

        register_solver("MyCustomSolver", MyCustomSolver)
        assert SOLVER_REGISTRY["MyCustomSolver"] is MyCustomSolver

    def test_register_solver_round_trip_via_from_dict(self, restore_registry):
        class MyCustomSolver(diffrax.Heun):
            pass

        register_solver("MyCustomSolver", MyCustomSolver)
        d = {
            "solver": "MyCustomSolver",
            "rtol": 1e-3,
            "atol": 1e-4,
            "max_steps": 1_000,
            "dt0": None,
        }
        cfg = SolverConfig.from_dict(d)
        assert isinstance(cfg.solver, MyCustomSolver)


class TestSolverConfigConstruction:
    def test_basic_construction_and_field_access(self):
        cfg = SolverConfig(
            solver=diffrax.Tsit5(),
            rtol=1e-4,
            atol=1e-5,
            max_steps=10_000,
            dt0=None,
        )
        assert isinstance(cfg.solver, diffrax.Tsit5)
        assert cfg.rtol == 1e-4
        assert cfg.atol == 1e-5
        assert cfg.max_steps == 10_000
        assert cfg.dt0 is None

    def test_per_state_atol_tuple_preserved(self):
        atol = (1e-5, 1e-6, 1e-7)
        cfg = SolverConfig(
            solver=diffrax.Dopri5(),
            rtol=1e-3,
            atol=atol,
            max_steps=1_000,
            dt0=0.01,
        )
        assert cfg.atol == atol


class TestSolverConfigStatic:
    def test_no_dynamic_leaves(self):
        cfg = SolverConfig(
            solver=diffrax.Tsit5(),
            rtol=1e-4,
            atol=1e-5,
            max_steps=10_000,
            dt0=None,
        )
        leaves = jax.tree_util.tree_leaves(cfg)
        assert leaves == []

    def test_no_dynamic_leaves_with_tuple_atol(self):
        cfg = SolverConfig(
            solver=diffrax.Heun(),
            rtol=1e-3,
            atol=(1e-5, 1e-5, 1e-5),
            max_steps=1_000,
            dt0=0.05,
        )
        leaves = jax.tree_util.tree_leaves(cfg)
        assert leaves == []

    def test_filter_extracts_no_arrays(self):
        cfg = SolverConfig(
            solver=diffrax.Tsit5(),
            rtol=1e-4,
            atol=1e-5,
            max_steps=10_000,
            dt0=0.01,
        )
        # eqx.filter pulls dynamic-array leaves; SolverConfig must have none.
        arrays_only = eqx.filter(cfg, eqx.is_array)
        array_leaves = jax.tree_util.tree_leaves(arrays_only)
        assert array_leaves == []


class TestSolverConfigRoundTrip:
    def test_to_dict_scalar_atol(self):
        cfg = SolverConfig(
            solver=diffrax.Tsit5(),
            rtol=1e-4,
            atol=1e-5,
            max_steps=500_000,
            dt0=None,
        )
        d = cfg.to_dict()
        assert d["solver"] == "Tsit5"
        assert d["rtol"] == 1e-4
        assert d["atol"] == 1e-5
        assert d["max_steps"] == 500_000
        assert d["dt0"] is None

    def test_to_dict_tuple_atol_serialises_to_list(self):
        cfg = SolverConfig(
            solver=diffrax.Dopri5(),
            rtol=1e-3,
            atol=(1e-5, 1e-6, 1e-7),
            max_steps=1_000,
            dt0=0.01,
        )
        d = cfg.to_dict()
        # JSON has no tuples; tuple atol must serialise to a list.
        assert d["atol"] == [1e-5, 1e-6, 1e-7]
        assert d["solver"] == "Dopri5"

    def test_round_trip_scalar_atol(self):
        cfg = SolverConfig(
            solver=diffrax.Tsit5(),
            rtol=1e-4,
            atol=1e-5,
            max_steps=500_000,
            dt0=None,
        )
        cfg2 = SolverConfig.from_dict(cfg.to_dict())
        assert isinstance(cfg2.solver, diffrax.Tsit5)
        assert cfg2.rtol == cfg.rtol
        assert cfg2.atol == cfg.atol
        assert cfg2.max_steps == cfg.max_steps
        assert cfg2.dt0 == cfg.dt0

    def test_round_trip_tuple_atol(self):
        cfg = SolverConfig(
            solver=diffrax.Heun(),
            rtol=1e-3,
            atol=(1e-5, 1e-6, 1e-7),
            max_steps=1_000,
            dt0=0.01,
        )
        cfg2 = SolverConfig.from_dict(cfg.to_dict())
        assert isinstance(cfg2.solver, diffrax.Heun)
        # Round-trip restores the tuple shape (lists in JSON come back as tuple).
        assert cfg2.atol == (1e-5, 1e-6, 1e-7)
        assert cfg2.dt0 == 0.01

    def test_round_trip_through_real_json(self):
        cfg = SolverConfig(
            solver=diffrax.Kvaerno3(),
            rtol=1e-5,
            atol=(1e-7, 1e-8),
            max_steps=2_000,
            dt0=None,
        )
        text = json.dumps(cfg.to_dict())
        cfg2 = SolverConfig.from_dict(json.loads(text))
        assert isinstance(cfg2.solver, diffrax.Kvaerno3)
        assert cfg2.rtol == 1e-5
        assert cfg2.atol == (1e-7, 1e-8)
        assert cfg2.max_steps == 2_000
        assert cfg2.dt0 is None


class TestSolverConfigErrors:
    def test_from_dict_unknown_solver_raises(self):
        d = {
            "solver": "DoesNotExist",
            "rtol": 1e-4,
            "atol": 1e-5,
            "max_steps": 1_000,
            "dt0": None,
        }
        with pytest.raises(ValueError, match="(?i)solver"):
            SolverConfig.from_dict(d)

    def test_to_dict_unregistered_solver_class_raises(self):
        # A solver instance whose class is not in SOLVER_REGISTRY cannot be serialised.
        class UnregisteredSolver(diffrax.Heun):
            pass

        cfg = SolverConfig(
            solver=UnregisteredSolver(),
            rtol=1e-4,
            atol=1e-5,
            max_steps=1_000,
            dt0=None,
        )
        with pytest.raises(ValueError, match="(?i)registry|solver"):
            cfg.to_dict()


class TestPublicAPI:
    def test_lazy_imports(self):
        import hybridmodels

        assert hybridmodels.SolverConfig is SolverConfig
        assert hybridmodels.SOLVER_REGISTRY is SOLVER_REGISTRY
        assert hybridmodels.register_solver is register_solver
