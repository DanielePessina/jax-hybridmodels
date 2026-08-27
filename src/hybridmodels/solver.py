"""How the ODE gets integrated, in a form that can be saved to JSON.

``SolverConfig`` holds the settings handed to diffrax. Every field is
``eqx.field(static=True)``, so the config carries no JAX arrays: compiled
kernels close over it as configuration, and it round-trips through JSON.

JSON cannot hold a Python object, so the concrete diffrax class must be
findable by name. ``SOLVER_REGISTRY`` and ``ADJOINT_REGISTRY`` are those
name tables. Add a custom class with :func:`register_solver` or
:func:`register_adjoint` before saving a config that references it.
"""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax.numpy as jnp

SOLVER_REGISTRY: dict[str, type[diffrax.AbstractSolver[Any]]] = {
    "Tsit5": diffrax.Tsit5,
    "Kvaerno3": diffrax.Kvaerno3,
    "Dopri5": diffrax.Dopri5,
    "Heun": diffrax.Heun,
}
"""Name → diffrax solver class. Extend via :func:`register_solver`."""


ADJOINT_REGISTRY: dict[str, type[diffrax.AbstractAdjoint]] = {
    "RecursiveCheckpoint": diffrax.RecursiveCheckpointAdjoint,
    "Direct": diffrax.DirectAdjoint,
    "Backsolve": diffrax.BacksolveAdjoint,
    "ForwardMode": diffrax.ForwardMode,
}
"""Name to diffrax adjoint class. Extend via :func:`register_adjoint`.

An *adjoint* is the method used to get gradients back through an ODE
solve. The choices trade memory against recomputation and accuracy, and
for a neural ODE memory is usually the binding constraint.

- ``Direct`` stores the whole forward tape. Cheapest to differentiate,
  most memory. The default here because every example wrote it by hand
  before this field existed.
- ``RecursiveCheckpoint`` stores O(log n) checkpoints and recomputes the
  rest. Diffrax's own default and the right choice for long trajectories
  or a network inside the vector field.
- ``Backsolve`` re-integrates the adjoint ODE backwards in constant
  memory. Cheapest in memory, but the reversed solve accumulates its own
  error, so gradients can be wrong for stiff or chaotic systems.
- ``ForwardMode`` is for forward-mode differentiation, which wins only
  when parameters are far fewer than outputs.
"""


def register_adjoint(name: str, cls: type[diffrax.AbstractAdjoint]) -> None:
    """Register a diffrax adjoint class under ``name`` for round-trip serialisation.

    Same contract as :func:`register_solver`. Re-registering an existing
    name overwrites without warning.
    """
    ADJOINT_REGISTRY[name] = cls


def register_solver(name: str, cls: type[diffrax.AbstractSolver[Any]]) -> None:
    """Register a custom diffrax solver class under ``name`` for round-trip serialisation.

    After registration, ``SolverConfig(solver=cls(), ...).to_dict()`` emits
    ``{"solver": name, ...}`` and ``SolverConfig.from_dict`` accepts it.
    Re-registering an existing name overwrites without warning. Calling code
    owns the naming.
    """
    SOLVER_REGISTRY[name] = cls


class SolverConfig(eqx.Module):
    """Everything the ODE solve needs, held as static configuration.

    Every field is ``eqx.field(static=True)``, so the config carries no JAX
    array leaves. Compiled functions close over it, so changing a value
    recompiles rather than reusing the old kernel. That is intended: a
    tolerance change must change the compiled solve.

    Attributes
    ----------
    solver : diffrax.AbstractSolver
        Concrete solver instance, for example ``diffrax.Tsit5()``. Its class
        must appear in ``SOLVER_REGISTRY`` for ``to_dict`` to round-trip.
    rtol, atol : float | tuple[float, ...]
        Relative and absolute error tolerances for the adaptive step-size
        controller. ``atol`` may be set per state component, as a tuple
        whose length matches ``S``, the full state dimension the user's
        ``simulate_fn`` integrates.
    max_steps : int
        Upper limit on solver steps. The solve errors rather than running
        forever if it needs more.
    dt0 : float | None
        Initial step size. ``None`` lets diffrax pick one.
    adjoint : diffrax.AbstractAdjoint
        How gradients are taken back through the solve. See
        ``ADJOINT_REGISTRY`` for what each choice costs.
    pcoeff, icoeff, dcoeff : float
        Gains of the PID step-size controller. The defaults ``(0, 1, 0)``
        are diffrax's own and give plain I-control. See
        :meth:`stepsize_controller`.
    """

    solver: diffrax.AbstractSolver[Any] = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float | tuple[float, ...] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True, default_factory=diffrax.DirectAdjoint)
    pcoeff: float = eqx.field(static=True, default=0.0)
    icoeff: float = eqx.field(static=True, default=1.0)
    dcoeff: float = eqx.field(static=True, default=0.0)

    def stepsize_controller(self) -> diffrax.PIDController:
        """Build the adaptive step-size controller this config describes.

        Removes a coercion every caller had to remember: diffrax broadcasts
        ``atol`` against the state pytree, and a Python tuple is not an
        array, so per-state tolerances misbehaved unless the caller wrapped
        them in ``jnp.asarray`` first.

        The ``(0, 1, 0)`` coefficient defaults are diffrax's own plain
        I-control, so this reproduces the ``PIDController(rtol, atol)`` the
        examples wrote by hand. Raise ``pcoeff`` to 0.3 or 0.4 to damp
        step-size oscillation on stiff problems.
        """
        atol = jnp.asarray(self.atol) if isinstance(self.atol, tuple) else self.atol
        return diffrax.PIDController(
            rtol=self.rtol,
            atol=atol,
            pcoeff=self.pcoeff,
            icoeff=self.icoeff,
            dcoeff=self.dcoeff,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict via the two registries.

        Solver and adjoint instances become their registered names, and a
        tuple ``atol`` becomes a list. An unregistered class raises rather
        than being guessed at.
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
        adjoint_name: str | None = None
        for name, adjoint_cls in ADJOINT_REGISTRY.items():
            if type(self.adjoint) is adjoint_cls:
                adjoint_name = name
                break
        if adjoint_name is None:
            raise ValueError(
                f"Adjoint class {type(self.adjoint).__name__!r} is not in "
                "ADJOINT_REGISTRY; register it via register_adjoint(name, cls) "
                "before calling to_dict."
            )
        return {
            "solver": solver_name,
            "rtol": self.rtol,
            "atol": atol_serialised,
            "max_steps": self.max_steps,
            "dt0": self.dt0,
            "adjoint": adjoint_name,
            "pcoeff": self.pcoeff,
            "icoeff": self.icoeff,
            "dcoeff": self.dcoeff,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SolverConfig:
        """Reconstruct a ``SolverConfig`` from ``to_dict`` output.

        Instantiates the registered class with no arguments, so a solver
        that needs constructor arguments cannot round-trip this way. A
        list-valued ``atol`` is coerced back to a tuple. An unknown name
        raises, listing what is registered.
        """
        name = d["solver"]
        if name not in SOLVER_REGISTRY:
            raise ValueError(
                f"Unknown solver name {name!r}; available: {sorted(SOLVER_REGISTRY.keys())}"
            )
        solver_cls = SOLVER_REGISTRY[name]
        atol_in = d["atol"]
        atol: float | tuple[float, ...]
        if isinstance(atol_in, list):
            atol = tuple(atol_in)
        else:
            atol = atol_in
        # Absent keys fall back to the field defaults so configs written
        # before these fields existed still load.
        adjoint_name = d.get("adjoint", "Direct")
        if adjoint_name not in ADJOINT_REGISTRY:
            raise ValueError(
                f"Unknown adjoint name {adjoint_name!r}; "
                f"available: {sorted(ADJOINT_REGISTRY.keys())}"
            )
        return cls(
            solver=solver_cls(),
            rtol=d["rtol"],
            atol=atol,
            max_steps=d["max_steps"],
            dt0=d["dt0"],
            adjoint=ADJOINT_REGISTRY[adjoint_name](),
            pcoeff=d.get("pcoeff", 0.0),
            icoeff=d.get("icoeff", 1.0),
            dcoeff=d.get("dcoeff", 0.0),
        )
