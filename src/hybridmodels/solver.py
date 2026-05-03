"""SolverConfig and a name-keyed solver registry for JSON-serialisable configs.

Per SPEC §5.3 / R-S1 / R-S2: every field on SolverConfig is static so the
config carries no JAX-array leaves. Round-trip serialisation goes through
SOLVER_REGISTRY, which users extend via register_solver(name, cls).
"""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx

SOLVER_REGISTRY: dict[str, type[diffrax.AbstractSolver[Any]]] = {
    "Tsit5": diffrax.Tsit5,
    "Kvaerno3": diffrax.Kvaerno3,
    "Dopri5": diffrax.Dopri5,
    "Heun": diffrax.Heun,
}


def register_solver(name: str, cls: type[diffrax.AbstractSolver[Any]]) -> None:
    SOLVER_REGISTRY[name] = cls


class SolverConfig(eqx.Module):
    solver: diffrax.AbstractSolver[Any] = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float | tuple[float, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def to_dict(self) -> dict[str, Any]:
        solver_name: str | None = None
        for name, cls in SOLVER_REGISTRY.items():
            if type(self.solver) is cls:
                solver_name = name
                break
        if solver_name is None:
            raise ValueError(
                f"Solver class {type(self.solver).__name__!r} is not in SOLVER_REGISTRY; "
                "register it via register_solver(name, cls) before calling to_dict."
            )
        atol_serialised: float | list[float]
        if isinstance(self.atol, tuple):
            atol_serialised = list(self.atol)
        else:
            atol_serialised = self.atol
        return {
            "solver": solver_name,
            "rtol": self.rtol,
            "atol": atol_serialised,
            "max_steps": self.max_steps,
            "dt0": self.dt0,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SolverConfig:
        name = d["solver"]
        if name not in SOLVER_REGISTRY:
            raise ValueError(
                f"Unknown solver name {name!r}; "
                f"available: {sorted(SOLVER_REGISTRY.keys())}"
            )
        solver_cls = SOLVER_REGISTRY[name]
        atol_in = d["atol"]
        atol: float | tuple[float, ...]
        if isinstance(atol_in, list):
            atol = tuple(atol_in)
        else:
            atol = atol_in
        return cls(
            solver=solver_cls(),
            rtol=d["rtol"],
            atol=atol,
            max_steps=d["max_steps"],
            dt0=d["dt0"],
        )
