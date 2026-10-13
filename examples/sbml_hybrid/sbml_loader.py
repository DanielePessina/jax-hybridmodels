"""A minimal SBML to JAX kinetic model converter for the example.

Reads an SBML file with python-libsbml, converts each kinetic law from
MathML to a sympy expression, and lambdifies the species ODEs against JAX.
The result is a plain vector field ``(t, y, params) -> dy/dt`` to call
from a ``simulate_fn``. It handles no events or function definitions and
never touches the global JAX config.

Scope: SBML Level 2/3 with mass-action or rational kinetics. Supported
MathML: arithmetic (``+-*/^``), ``exp/ln/log/abs/min/max/floor/ceiling``,
``piecewise``, relational and logical operators, and the constants
``pi/e``. Anything else raises a ``RuntimeError`` naming the node.

This lives with the example rather than in the library: the framework
owns vmap, jit and grad, and the user owns the physics, here including
the converter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import diffrax
import jax.numpy as jnp
import libsbml
import sympy as sp
from jax import Array
from jax.typing import ArrayLike

# ---------------------------------------------------------------------------
# MathML AST -> sympy
# ---------------------------------------------------------------------------


def _ast_to_sympy(node: libsbml.ASTNode) -> Any:
    """Recursively convert a libsbml MathML ASTNode into a sympy expression."""
    t = node.getType()
    if t == libsbml.AST_NAME or t == libsbml.AST_NAME_AVOGADRO:
        return sp.Symbol(node.getName())
    if t == libsbml.AST_NAME_TIME:
        return sp.Symbol("t")
    if t == libsbml.AST_INTEGER:
        return sp.Integer(node.getInteger())
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        return sp.Float(node.getReal())
    if t == libsbml.AST_CONSTANT_PI:
        return sp.pi
    if t == libsbml.AST_CONSTANT_E:
        return sp.E
    if t == libsbml.AST_CONSTANT_TRUE:
        return sp.true
    if t == libsbml.AST_CONSTANT_FALSE:
        return sp.false

    # Any: SBML math mixes numeric expressions and boolean conditions, and
    # sympy types them as unrelated classes.
    children = [_ast_to_sympy(node.getChild(i)) for i in range(node.getNumChildren())]

    unary = {
        libsbml.AST_FUNCTION_EXP: sp.exp,
        libsbml.AST_FUNCTION_LN: sp.log,
        libsbml.AST_FUNCTION_ABS: sp.Abs,
        libsbml.AST_FUNCTION_FLOOR: sp.floor,
        libsbml.AST_FUNCTION_CEILING: sp.ceiling,
        libsbml.AST_FUNCTION_SIN: sp.sin,
        libsbml.AST_FUNCTION_COS: sp.cos,
        libsbml.AST_FUNCTION_TAN: sp.tan,
        libsbml.AST_FUNCTION_SINH: sp.sinh,
        libsbml.AST_FUNCTION_COSH: sp.cosh,
        libsbml.AST_FUNCTION_TANH: sp.tanh,
        libsbml.AST_LOGICAL_NOT: sp.Not,
    }
    if t in unary:
        return unary[t](*children)

    if t == libsbml.AST_PLUS:
        return sp.Add(*children)
    if t == libsbml.AST_MINUS:
        return -children[0] if len(children) == 1 else children[0] - children[1]
    if t == libsbml.AST_TIMES:
        return sp.Mul(*children)
    if t == libsbml.AST_DIVIDE:
        return children[0] / children[1]
    if t == libsbml.AST_POWER:
        return children[0] ** children[1]
    if t == libsbml.AST_FUNCTION_LOG:
        return sp.log(*children)  # 1-arg: ln; 2-arg: log base
    if t == libsbml.AST_FUNCTION_MIN:
        return sp.Min(*children)
    if t == libsbml.AST_FUNCTION_MAX:
        return sp.Max(*children)
    if t == libsbml.AST_FUNCTION_PIECEWISE:
        return _piecewise(children)
    if t in (
        libsbml.AST_RELATIONAL_EQ,
        libsbml.AST_RELATIONAL_NEQ,
        libsbml.AST_RELATIONAL_LT,
        libsbml.AST_RELATIONAL_LEQ,
        libsbml.AST_RELATIONAL_GT,
        libsbml.AST_RELATIONAL_GEQ,
    ):
        op = {
            libsbml.AST_RELATIONAL_EQ: sp.Eq,
            libsbml.AST_RELATIONAL_NEQ: sp.Ne,
            libsbml.AST_RELATIONAL_LT: sp.Lt,
            libsbml.AST_RELATIONAL_LEQ: sp.Le,
            libsbml.AST_RELATIONAL_GT: sp.Gt,
            libsbml.AST_RELATIONAL_GEQ: sp.Ge,
        }
        return op[t](children[0], children[1])
    if t in (libsbml.AST_LOGICAL_AND, libsbml.AST_LOGICAL_OR):
        return sp.And(*children) if t == libsbml.AST_LOGICAL_AND else sp.Or(*children)

    raise RuntimeError(f"unsupported MathML node type {libsbml.typeCodeToString(t)} ({t})")


def _piecewise(children: list[sp.Expr]) -> sp.Expr:
    """SBML piecewise nodes arrive as (expr, cond)* optionally (default,)."""
    args = []
    it = iter(children)
    for expr in it:
        try:
            cond = next(it)
        except StopIteration:
            args.append((expr, sp.true))  # trailing default
            break
        args.append((expr, cond))
    return sp.Piecewise(*args)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class SBMLKineticModel:
    """A parsed SBML model exposed as a JAX-compatible vector field.

    Attributes
    ----------
    species_names : list[str]
        Dynamic species, in state-vector order.
    params : dict[str, float]
        Global parameters, local parameters (namespaced per reaction), and
        constant assignment-rule results. Overridable per experiment.
    y0 : Array
        Initial amounts, with ``initialAssignment`` rules applied.
    reaction_names, stoich, compartment_sizes
        Reaction bookkeeping used to assemble the vector field.
    outputs : dict[str, sp.Expr]
        State-dependent assignment rules (e.g. scaled "total" readouts).
        Evaluate with :meth:`output_fn`.
    """

    species_names: list[str]
    params: dict[str, float]
    y0: Array
    reaction_names: list[str]
    stoich: Array
    compartment_sizes: Array
    outputs: dict[str, sp.Expr] = field(default_factory=dict)
    _fluxes: dict[str, sp.Expr] = field(default_factory=dict)
    _flux_extras: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _output_fn: Callable[..., Any] | None = field(default=None, init=False)
    _flux_fns: list = field(default_factory=list, init=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> SBMLKineticModel:
        document = libsbml.SBMLReader().readSBML(path)
        if document.getNumErrors() > 0:
            raise RuntimeError(document.printErrors())
        model = document.getModel()
        if model is None:
            raise RuntimeError(f"no SBML model in {path}")

        species_names, species_compartment = [], {}
        for s in model.getListOfSpecies():
            if s.getBoundaryCondition() or s.getConstant():
                continue
            species_names.append(s.getId())
            species_compartment[s.getId()] = s.getCompartment()
        compartments = {c.getId(): c.getSize() for c in model.getListOfCompartments()}

        params = {p.getId(): p.getValue() for p in model.getListOfParameters()}
        for reaction in model.getListOfReactions():
            law = reaction.getKineticLaw()
            if law is not None:
                for p in law.getListOfParameters():
                    params[f"lp.{reaction.getId()}.{p.getId()}"] = p.getValue()

        symbol = {name: sp.Symbol(name) for name in species_names}
        symbol.update({name: sp.Symbol(name) for name in params})

        initial = {
            ia.getSymbol(): _ast_to_sympy(ia.getMath())
            for ia in model.getListOfInitialAssignments()
        }
        assignment_rules = {}
        outputs: dict[str, sp.Expr] = {}
        species_symbols = {sp.Symbol(name) for name in species_names}
        for rule in model.getListOfRules():
            if not rule.isAssignment():
                continue
            expr = _ast_to_sympy(rule.getMath())
            if expr.free_symbols & species_symbols:
                outputs[rule.getVariable()] = expr
            else:
                assignment_rules[rule.getVariable()] = expr

        all_params = {**params, **assignment_rules}
        param_values = {k: float(v) for k, v in all_params.items() if not isinstance(v, sp.Expr)}
        for name, expr in assignment_rules.items():
            param_values[name] = float(
                expr.subs({sp.Symbol(k): v for k, v in param_values.items()})
            )

        y0 = []
        for name in species_names:
            if name in initial:
                val = initial[name].subs({sp.Symbol(k): v for k, v in param_values.items()})
                y0.append(float(val))
            else:
                y0.append(_species_amount(model, name))
        y0 = jnp.asarray(y0, dtype=jnp.float32)

        reactions = {r.getId(): r for r in model.getListOfReactions()}
        stoich = jnp.zeros((len(species_names), len(reactions)), dtype=jnp.float32)
        for j, rid in enumerate(reactions):
            r = reactions[rid]
            for ref in r.getListOfReactants():
                if ref.getSpecies() in species_names:
                    stoich = stoich.at[species_names.index(ref.getSpecies()), j].add(
                        -float(ref.getStoichiometry())
                    )
            for ref in r.getListOfProducts():
                if ref.getSpecies() in species_names:
                    stoich = stoich.at[species_names.index(ref.getSpecies()), j].add(
                        float(ref.getStoichiometry())
                    )

        fluxes = {
            rid: _ast_to_sympy(reactions[rid].getKineticLaw().getMath())
            for rid in reactions
            if reactions[rid].getKineticLaw() is not None
        }

        compartment_sizes = jnp.asarray(
            [compartments.get(species_compartment[name], 1.0) for name in species_names],
            dtype=jnp.float32,
        )
        km = cls(
            species_names,
            param_values,
            y0,
            list(reactions),
            stoich,
            compartment_sizes,
            outputs,
            fluxes,
        )
        km.params["Cell"] = float(compartments.get("Cell", 1.0))
        return km

    def make_flux(self, expr: sp.Expr, extra_params: tuple[str, ...] = ()) -> Callable[..., Array]:
        """Lambdify a sympy rate law against the model's state/param layout.

        The returned callable has signature ``(t, y, params) -> Array`` with
        ``params`` a plain dict; this is the same signature as the assembled
        vector field, so a single rate law can be swapped in and out.
        ``extra_params`` declares parameter names beyond ``self.params``
        (e.g. a neural ``Vmax`` injected by a ``simulate_fn``).
        """
        param_names = [*self.params.keys(), *extra_params]
        symbols = [*map(sp.Symbol, self.species_names), *map(sp.Symbol, param_names)]
        fn = sp.lambdify(symbols, expr, "jax")

        def flux(_t: Array, y: Array, params: dict) -> Array:
            return fn(*y, *(params[name] for name in param_names))

        return flux

    def replace_flux(
        self, reaction_id: str, expr: sp.Expr, extra_params: tuple[str, ...] = ()
    ) -> None:
        """Swap one reaction's rate law (e.g. for a neural rate parameter).

        ``extra_params`` names parameter symbols beyond ``self.params`` that
        the caller will inject into the ``params`` dict at call time.
        """
        self._fluxes[reaction_id] = expr
        self._flux_extras[reaction_id] = extra_params
        self._flux_fns = []

    def _all_fluxes(self) -> list[Callable[..., Array]]:
        if not self._flux_fns:
            self._flux_fns = [
                self.make_flux(expr, self._flux_extras.get(rid, ()))
                for rid, expr in self._fluxes.items()
            ]
        return self._flux_fns

    # -- simulation --------------------------------------------------------

    def vector_field(self, t: ArrayLike, y: Array, params: dict) -> Array:
        """``dy/dt`` with ``params`` a dict holding every name in ``self.params``."""
        v = jnp.stack([f(t, y, params) for f in self._all_fluxes()])
        return jnp.matmul(self.stoich, v) / self.compartment_sizes

    def integrate(self, solver, ts: Array, y0: Array, params: dict) -> Array:
        """Run the framework's diffrax solver over this model's vector field."""
        return solver.diffeqsolve(diffrax.ODETerm(self.vector_field), ts, y0, args=params).ys

    def output_fn(self, state: Array, params: dict) -> Array:
        """Evaluate the state-dependent assignment rules (e.g. scaled totals).

        ``state`` has shape ``[T, S]`` (or ``[S]``); returns ``[T, O]``.
        """
        if self._output_fn is None:
            symbols = [*map(sp.Symbol, self.species_names), *map(sp.Symbol, self.params)]
            exprs = list(self.outputs.values())
            self._output_fn = sp.lambdify(symbols, exprs, "jax")
        if state.ndim == 1:
            state = state[None, :]
        vals = self._output_fn(*state.T, *(params[name] for name in self.params))
        if isinstance(vals, (list, tuple)):
            return jnp.stack(vals, axis=-1)
        return jnp.asarray(vals)


def _species_amount(model, species_id: str) -> float:
    s = model.getSpecies(species_id)
    if s.isSetInitialAmount():
        return s.getInitialAmount()
    if s.isSetInitialConcentration():
        return s.getInitialConcentration()
    raise RuntimeError(f"species {species_id} has no initial amount or concentration")
