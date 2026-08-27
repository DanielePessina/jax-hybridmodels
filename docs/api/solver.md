# Solver: ODE Integration

[`SolverConfig`](#solverconfig) bundles a `diffrax` solver instance with its tolerances and step controls. Every field is static, so the config is closed over by jitted functions without re-tracing on value changes (a tolerance change does trigger a recompile, which is what we want).

Solvers are looked up by name through [`SOLVER_REGISTRY`](#solver_registry); [`register_solver`](#register_solver) extends the registry with custom implementations so saved configs round-trip cleanly.

## Quick links

- [`SolverConfig`](#solverconfig)
- [`SOLVER_REGISTRY`](#solver_registry)
- [`register_solver`](#register_solver)
- [`ADJOINT_REGISTRY`](#adjoint_registry)
- [`register_adjoint`](#register_adjoint)

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
    adjoint: 'diffrax.AbstractAdjoint' = <factory>,
    pcoeff: 'float' = 0.0,
    icoeff: 'float' = 1.0,
    dcoeff: 'float' = 0.0,
) -> None
```

Everything the ODE solve needs, held as static configuration.

Every field is ``eqx.field(static=True)``, so the config carries no JAX
array leaves. Compiled functions close over it, so changing a value
recompiles rather than reusing the old kernel. That is intended: a
tolerance change must change the compiled solve.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `solver` | `diffrax.AbstractSolver` | Concrete solver instance, for example ``diffrax.Tsit5()``. Its class must appear in ``SOLVER_REGISTRY`` for ``to_dict`` to round-trip. |
| `rtol, atol` | `float | tuple[float, ...]` | Relative and absolute error tolerances for the adaptive step-size controller. ``atol`` may be set per state component, as a tuple whose length matches ``S``, the full state dimension the user's ``simulate_fn`` integrates. |
| `max_steps` | `int` | Upper limit on solver steps. The solve errors rather than running forever if it needs more. |
| `dt0` | `float | None` | Initial step size. ``None`` lets diffrax pick one. |
| `adjoint` | `diffrax.AbstractAdjoint` | How gradients are taken back through the solve. See ``ADJOINT_REGISTRY`` for what each choice costs. |
| `pcoeff, icoeff, dcoeff` | `float` | Gains of the PID step-size controller. The defaults ``(0, 1, 0)`` are diffrax's own and give plain I-control. See :meth:`stepsize_controller`. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L76)</small>

#### `SolverConfig.stepsize_controller()`

```python
stepsize_controller(self) -> 'diffrax.PIDController'
```

Build the adaptive step-size controller this config describes.

Removes a coercion every caller had to remember: diffrax broadcasts
``atol`` against the state pytree, and a Python tuple is not an
array, so per-state tolerances misbehaved unless the caller wrapped
them in ``jnp.asarray`` first.

The ``(0, 1, 0)`` coefficient defaults are diffrax's own plain
I-control, so this reproduces the ``PIDController(rtol, atol)`` the
examples wrote by hand. Raise ``pcoeff`` to 0.3 or 0.4 to damp
step-size oscillation on stiff problems.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L118)</small>

#### `SolverConfig.to_dict()`

```python
to_dict(self) -> 'dict[str, Any]'
```

Serialise to a JSON-compatible dict via the two registries.

Solver and adjoint instances become their registered names, and a
tuple ``atol`` becomes a list. An unregistered class raises rather
than being guessed at.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L140)</small>

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

After registration, ``SolverConfig(solver=cls(), ...).to_dict()`` emits
``{"solver": name, ...}`` and ``SolverConfig.from_dict`` accepts it.
Re-registering an existing name overwrites without warning. Calling code
owns the naming.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L65)</small>

---

<a id="adjoint_registry"></a>

### `ADJOINT_REGISTRY`

<small>`from hybridmodels.solver import ADJOINT_REGISTRY` &nbsp;·&nbsp; also re-exported as `hybridmodels.ADJOINT_REGISTRY`</small>

```python
ADJOINT_REGISTRY = {
  'Backsolve': _ModuleMeta
  'Direct': _ModuleMeta
  'ForwardMode': _ModuleMeta
  'RecursiveCheckpoint': _ModuleMeta
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

<a id="register_adjoint"></a>

### `register_adjoint()`

<small>`from hybridmodels.solver import register_adjoint` &nbsp;·&nbsp; also re-exported as `hybridmodels.register_adjoint`</small>

```python
register_adjoint(name: 'str', cls: 'type[diffrax.AbstractAdjoint]') -> 'None'
```

Register a diffrax adjoint class under ``name`` for round-trip serialisation.

Same contract as :func:`register_solver`. Re-registering an existing
name overwrites without warning.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/solver.py#L56)</small>
