from __future__ import annotations

import json

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hybridmodels import (
    ADJOINT_REGISTRY,
    SOLVER_REGISTRY,
    SolverConfig,
    register_adjoint,
    register_solver,
)


@pytest.fixture
def restore_registry():
    snapshot = dict(SOLVER_REGISTRY)
    yield
    SOLVER_REGISTRY.clear()
    SOLVER_REGISTRY.update(snapshot)


@pytest.fixture
def restore_adjoint_registry():
    snapshot = dict(ADJOINT_REGISTRY)
    yield
    ADJOINT_REGISTRY.clear()
    ADJOINT_REGISTRY.update(snapshot)


def _cfg(
    *,
    solver: diffrax.AbstractSolver | None = None,
    rtol: float = 1e-4,
    atol: float | tuple[float, ...] = 1e-6,
    max_steps: int = 10_000,
    dt0: float | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
) -> SolverConfig:
    """A valid SolverConfig with every new field left at its default."""
    return SolverConfig(
        solver=diffrax.Tsit5() if solver is None else solver,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        dt0=dt0,
        adjoint=diffrax.DirectAdjoint() if adjoint is None else adjoint,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
    )


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


class TestAdjointRegistry:
    def test_registry_has_the_diffrax_adjoints_worth_naming(self):
        for name in ("RecursiveCheckpoint", "Direct", "Backsolve", "ForwardMode"):
            assert name in ADJOINT_REGISTRY

    def test_default_adjoint_preserves_existing_example_behaviour(self):
        # Every example wrote adjoint=DirectAdjoint() by hand before this
        # field existed. Defaulting to anything else would silently change
        # how their gradients are computed.
        cfg = _cfg()
        assert isinstance(cfg.adjoint, diffrax.DirectAdjoint)

    def test_adjoint_round_trips_by_name(self):
        cfg = _cfg(adjoint=diffrax.RecursiveCheckpointAdjoint())
        assert cfg.to_dict()["adjoint"] == "RecursiveCheckpoint"
        assert isinstance(SolverConfig.from_dict(cfg.to_dict()).adjoint, type(cfg.adjoint))

    def test_unregistered_adjoint_class_raises_on_to_dict(self):
        class _Custom(diffrax.DirectAdjoint):
            pass

        with pytest.raises(ValueError, match="ADJOINT_REGISTRY"):
            _cfg(adjoint=_Custom()).to_dict()

    def test_register_adjoint_extends_the_registry(self, restore_adjoint_registry):
        class _Custom(diffrax.DirectAdjoint):
            pass

        register_adjoint("Custom", _Custom)
        assert _cfg(adjoint=_Custom()).to_dict()["adjoint"] == "Custom"

    def test_adjoint_is_static(self):
        assert not jax.tree_util.tree_leaves(
            eqx.filter(_cfg(adjoint=diffrax.RecursiveCheckpointAdjoint()), eqx.is_array)
        )


class TestStepsizeController:
    """``SolverConfig`` builds the controller so callers stop re-deriving it."""

    def test_returns_a_pid_controller_carrying_the_tolerances(self):
        controller = _cfg(rtol=1e-6, atol=1e-9).stepsize_controller()
        assert isinstance(controller, diffrax.PIDController)

    def test_tuple_atol_becomes_an_array(self):
        # The wart this method exists to remove: diffrax broadcasts atol
        # against the state pytree, and a Python tuple is not an array, so
        # every example that wanted per-state tolerances coerced it by hand.
        controller = _cfg(atol=(1e-5, 1e-6, 1e-7)).stepsize_controller()
        assert jnp.asarray(controller.atol).shape == (3,)

    def test_scalar_atol_stays_scalar(self):
        assert jnp.asarray(_cfg(atol=1e-8).stepsize_controller().atol).shape == ()

    def test_pid_coefficients_round_trip(self):
        cfg = _cfg(pcoeff=0.4, icoeff=0.3, dcoeff=0.0)
        assert SolverConfig.from_dict(cfg.to_dict()).pcoeff == 0.4
        controller = cfg.stepsize_controller()
        assert controller.pcoeff == 0.4 and controller.icoeff == 0.3

    def test_defaults_reproduce_the_plain_controller_examples_wrote(self):
        # pcoeff=icoeff=dcoeff=0 is diffrax's own default, i.e. plain
        # I-control, which is what PIDController(rtol=, atol=) gives.
        cfg = _cfg()
        assert (cfg.pcoeff, cfg.icoeff, cfg.dcoeff) == (0.0, 1.0, 0.0)

    def test_controller_is_usable_in_a_real_solve(self):
        cfg = _cfg(atol=(1e-8, 1e-8))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -y),
            cfg.solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=jnp.array([1.0, 2.0]),
            saveat=diffrax.SaveAt(ts=jnp.array([0.0, 1.0])),
            stepsize_controller=cfg.stepsize_controller(),
            max_steps=cfg.max_steps,
            adjoint=cfg.adjoint,
        )
        assert jnp.allclose(sol.ys[-1], jnp.array([1.0, 2.0]) * jnp.exp(-1.0), rtol=1e-4)
