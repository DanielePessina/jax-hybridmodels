# Solver: ODE Integration

[`SolverConfig`](#solverconfig) bundles a `diffrax` solver instance with its tolerances and step controls. Every field is static, so the config is closed over by jitted functions without re-tracing on value changes (a tolerance change does trigger a recompile, which is what we want).

Solvers are looked up by name through [`SOLVER_REGISTRY`](#solver_registry); [`register_solver`](#register_solver) extends the registry with custom implementations so saved configs round-trip cleanly.

## Quick links

- [`SolverConfig`](#solverconfig)
- [`SOLVER_REGISTRY`](#solver_registry)
- [`register_solver`](#register_solver)

---

<a id="solverconfig"></a>

### `SolverConfig`

<small>`from hybridmodels.solver import SolverConfig` &nbsp;·&nbsp; also re-exported as `hybridmodels.SolverConfig`</small>

```python
SolverConfig(
    solver: 'diffrax.AbstractSolver[Any]',
    rtol: 'float',
    atol: 'float | tuple[float, ...]',
    max_steps: 'int',
    dt0: 'float | None',
) -> None
```

Static-only ``diffrax`` solver configuration.

Every field is ``eqx.field(static=True)`` so the config carries no JAX
array leaves — it is closed over by jitted training/prediction functions
and contributes to their static signature without re-tracing on value
changes (changes do trigger a recompile, which is what we want).

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `solver` | `diffrax.AbstractSolver` | Concrete solver instance (e.g. ``diffrax.Tsit5()``); its class must appear in ``SOLVER_REGISTRY`` for ``to_dict`` to round-trip. |
| `rtol, atol` | `float | tuple[float, ...]` | Diffrax tolerances. ``atol`` may be per-state-component — a tuple whose length matches ``S`` (the full state dimension consumed by the user's ``simulate_fn``). |
| `max_steps` | `int` | Diffrax ``max_steps`` budget. |
| `dt0` | `float | None` | Initial step size; ``None`` lets diffrax pick. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L40)</small>

#### `SolverConfig.to_dict()`

```python
to_dict(self) -> 'dict[str, Any]'
```

Serialise to a JSON-compatible dict via ``SOLVER_REGISTRY``.

The solver instance is replaced by its registered name; tuple ``atol``
becomes a list (JSON has no tuple). Unknown solver classes raise so
users register custom solvers explicitly via ``register_solver``.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L69)</small>

---

<a id="solver_registry"></a>

### `SOLVER_REGISTRY`

<small>`from hybridmodels.solver import SOLVER_REGISTRY` &nbsp;·&nbsp; also re-exported as `hybridmodels.SOLVER_REGISTRY`</small>

```python
SOLVER_REGISTRY = {
  'Dopri5': _MetaAbstractSolver
  'Heun': _MetaAbstractSolver
  'Kvaerno3': _MetaAbstractSolver
  'Tsit5': _MetaAbstractSolver
}
```

dict() -> new empty dictionary

dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

---

<a id="register_solver"></a>

### `register_solver()`

<small>`from hybridmodels.solver import register_solver` &nbsp;·&nbsp; also re-exported as `hybridmodels.register_solver`</small>

```python
register_solver(name: 'str', cls: 'type[diffrax.AbstractSolver[Any]]') -> 'None'
```

Register a custom diffrax solver class under ``name`` for round-trip serialisation.

After registration, ``SolverConfig(solver=cls(), ...).to_dict()`` will
emit ``{"solver": name, ...}`` and ``SolverConfig.from_dict`` will accept
it. Re-registering an existing name overwrites silently — calling code
is responsible for namespace hygiene.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L29)</small>
