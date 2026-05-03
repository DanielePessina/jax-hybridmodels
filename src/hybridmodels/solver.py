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
"""Name → diffrax solver class. Extend via :func:`register_solver`."""


def register_solver(name: str, cls: type[diffrax.AbstractSolver[Any]]) -> None:
    """Register a custom diffrax solver class under ``name`` for round-trip serialisation.

    After registration, ``SolverConfig(solver=cls(), ...).to_dict()`` will
    emit ``{"solver": name, ...}`` and ``SolverConfig.from_dict`` will accept
    it. Re-registering an existing name overwrites silently — calling code
    is responsible for namespace hygiene.
    """
    SOLVER_REGISTRY[name] = cls


class SolverConfig(eqx.Module):
    """Static-only ``diffrax`` solver configuration (R-S1).

    Every field is ``eqx.field(static=True)`` so the config carries no JAX
    array leaves — it is closed over by jitted training/prediction functions
    and contributes to their static signature without re-tracing on value
    changes (changes do trigger a recompile, which is what we want).

    Attributes
    ----------
    solver : diffrax.AbstractSolver
        Concrete solver instance (e.g. ``diffrax.Tsit5()``); its class must
        appear in ``SOLVER_REGISTRY`` for ``to_dict`` to round-trip.
    rtol, atol : float | tuple[float, ...]
        Diffrax tolerances. ``atol`` may be per-state-component — a tuple
        whose length matches ``S`` from ``simulate_fn``.
    max_steps : int
        Diffrax ``max_steps`` budget.
    dt0 : float | None
        Initial step size; ``None`` lets diffrax pick.
    """

    solver: diffrax.AbstractSolver[Any] = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float | tuple[float, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict via ``SOLVER_REGISTRY`` (R-S2).

        The solver instance is replaced by its registered name; tuple ``atol``
        becomes a list (JSON has no tuple). Unknown solver classes raise so
        users register custom solvers explicitly via ``register_solver``.
        """
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
        """Reconstruct a ``SolverConfig`` from ``to_dict`` output.

        Looks ``d["solver"]`` up in ``SOLVER_REGISTRY`` and instantiates the
        class with no arguments. List-valued ``atol`` is coerced back to a
        tuple to match the static-field type. Unknown names raise with the
        currently-registered set.
        """
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
