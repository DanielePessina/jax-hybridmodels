"""Generate VitePress API reference markdown from hybridmodels docstrings.

Walks ``hybridmodels.__all__``, groups public symbols by their declaring
submodule, and emits one markdown page per module under ``docs/api/``.

Each entry contains:

- An anchor heading (``### `name()```)
- The runtime signature, lifted via ``inspect.signature`` and wrapped in a
  Python code fence
- The parsed numpy-style docstring rendered as markdown — Parameters,
  Returns, Attributes, and Raises become two-column tables; Notes /
  Examples are passed through with their original formatting
- A source-code link pointing at the GitHub blob

Run via ``npm run docs:gen`` (which is ``uv run python scripts/gen_api_docs.py``)
or directly with ``uv run python scripts/gen_api_docs.py``. Passing
``--check`` exits non-zero if any generated file would change, suitable
for CI.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hybridmodels

# --------------------------------------------------------------------------- #
# Module groupings → output pages                                             #
# --------------------------------------------------------------------------- #

# Each output page maps to a list of public-API symbols (drawn from
# ``hybridmodels.__all__``) that should appear on it. Ordering inside a
# group controls the rendered order on the page; pages are emitted in the
# tuple order of ``PAGES``.
PAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "data",
        "Data: Experiments, Channels, Datasets",
        (
            "ChannelObs",
            "Experiment",
            "make_experiment",
            "BucketPayload",
            "Dataset",
            "make_dataset",
            "split_dataset",
        ),
    ),
    (
        "predictors",
        "Predictors: Trainable Components",
        (
            "Predictor",
            "BoundScaler",
            "BoundedPredictor",
            "MLPPredictor",
            "KANPredictor",
            "reinitialize_with_key",
            "reinitialize_pytree_with_key",
        ),
    ),
    (
        "solver",
        "Solver: ODE Integration",
        (
            "SolverConfig",
            "SOLVER_REGISTRY",
            "register_solver",
        ),
    ),
    (
        "training",
        "Training: Optax & Evosax Loops",
        (
            "OptaxTrainingConfig",
            "train_with_optax",
            "EvosaxTrainingConfig",
            "train_with_evosax",
        ),
    ),
    (
        "losses",
        "Losses: Masked & Balanced Objectives",
        (
            "masked_mse",
            "masked_mle",
            "bal_mse",
            "bal_mle",
            "LOSS_REGISTRY",
        ),
    ),
    (
        "trainable",
        "Trainable Masks: Freezing Leaves",
        (
            "default_trainable",
            "trainable_mask",
            "freeze_paths",
            "freeze_modules_of_type",
            "freeze_where",
        ),
    ),
    (
        "prediction",
        "Prediction: Forward Simulation",
        (
            "predict_bucket",
            "predict_dataset",
        ),
    ),
    (
        "serialise",
        "Serialise: Save & Load Runs",
        (
            "save_predictors",
            "load_predictors",
            "save_run",
            "load_run",
        ),
    ),
    (
        "ui",
        "UI: Training Dashboards",
        (
            "TrainingUI",
            "EvosaxUI",
            "SilentUI",
            "RichTrainingUI",
            "RichEvosaxUI",
        ),
    ),
    (
        "rng",
        "RNG: Named-Fold Keys",
        ("fold",),
    ),
)

# Repository GitHub URL prefix for source-code links.
GITHUB_BLOB = "https://github.com/DanielePessina/jax-hybridmodels/blob/main"


# --------------------------------------------------------------------------- #
# Numpy-style docstring parser                                                #
# --------------------------------------------------------------------------- #

# Section names treated as "parameter-like" (rendered as a table).
# Anything else is rendered as a plain `**Heading**` block followed by the
# verbatim body — Notes, Examples, See Also, Construction, Pipeline, etc.
_PARAM_SECTIONS = {"Parameters", "Returns", "Yields", "Attributes", "Raises", "Other Parameters"}

# Match a section header: a line whose next line is a row of dashes of the
# same length. Captures the header name. Multiline / non-greedy.
_SECTION_RE = re.compile(r"(?m)^(?P<title>[A-Z][A-Za-z ]+)\n-{3,}\n")


@dataclass
class ParsedDoc:
    summary: str
    body: str
    sections: list[tuple[str, str]]


def parse_docstring(doc: str | None) -> ParsedDoc:
    """Split a numpy-style docstring into summary, body, and named sections."""
    if not doc:
        return ParsedDoc(summary="", body="", sections=[])
    doc = inspect.cleandoc(doc)

    matches = list(_SECTION_RE.finditer(doc))
    if not matches:
        # No sections: the whole docstring is summary + body.
        lines = doc.split("\n", 1)
        summary = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        return ParsedDoc(summary=summary, body=body, sections=[])

    # Everything before the first section header is summary + body.
    head = doc[: matches[0].start()].strip()
    head_lines = head.split("\n", 1)
    summary = head_lines[0].strip()
    body = head_lines[1].strip() if len(head_lines) > 1 else ""

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group("title").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        section_body = doc[start:end].strip("\n")
        sections.append((title, section_body))

    return ParsedDoc(summary=summary, body=body, sections=sections)


def parse_param_section(body: str) -> list[tuple[str, str, str]]:
    """Parse a numpy-style parameter section into ``(name, type, desc)`` rows.

    Numpy convention is::

        name : type
            description (one or more indented lines)
        other_name : type
            description

    Names without a ``:`` separator are typeless (Returns/Yields with bare
    type-only entries also occur — those get reported with name = type and
    an empty type column).
    """
    rows: list[tuple[str, str, str]] = []
    if not body.strip():
        return rows

    # Tokenize: a new entry begins on any non-indented, non-blank line.
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith((" ", "\t")):
            i += 1
            continue
        # Header line: "name : type" or "name" or just "type".
        header = line.strip()
        if " : " in header:
            name, type_str = header.split(" : ", 1)
        else:
            name, type_str = header, ""
        # Collect indented description lines.
        desc_lines: list[str] = []
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
            desc_lines.append(lines[i])
            i += 1
        desc = textwrap.dedent("\n".join(desc_lines)).strip()
        rows.append((name.strip(), type_str.strip(), desc))
    return rows


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #


def _escape_table_cell(text: str) -> str:
    """Make a string safe to embed in a single markdown table cell.

    Pipes are escaped; newlines collapse to ``<br>`` so multi-line
    descriptions remain on one logical row. We deliberately avoid trying
    to preserve code fences inside cells — they don't render reliably; we
    inline backtick spans instead.
    """
    return text.replace("|", "\\|").replace("\n\n", "<br><br>").replace("\n", " ")


def render_param_table(rows: list[tuple[str, str, str]], header: str = "Parameter") -> str:
    if not rows:
        return ""
    out: list[str] = [
        f"| {header} | Type | Description |",
        "| --- | --- | --- |",
    ]
    for name, type_str, desc in rows:
        out.append(
            f"| `{name}` | {f'`{type_str}`' if type_str else ''} | {_escape_table_cell(desc)} |"
        )
    return "\n".join(out)


def _slugify(name: str) -> str:
    """Anchor slug used for ``[link](#slug)`` cross-references inside a page."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", name).lower()


def render_signature(name: str, obj: Any) -> str | None:
    """Best-effort signature extraction; returns None for non-callables."""
    if not callable(obj):
        return None
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    # Replace any callable defaults (whose ``repr`` includes an unstable
    # memory address) with a stable, qualified-name form so generator
    # output is deterministic across runs.
    sig = _stabilise_signature(sig)
    sig_text = f"{name}{sig}"
    if len(sig_text) > 88:
        sig_text = _wrap_signature(name, sig)
    return sig_text


def _stabilise_signature(sig: inspect.Signature) -> inspect.Signature:
    new_params: list[inspect.Parameter] = []
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty and callable(p.default):
            stable = _StableDefault(getattr(p.default, "__qualname__", repr(p.default)))
            new_params.append(p.replace(default=stable))
        else:
            new_params.append(p)
    return sig.replace(parameters=new_params)


class _StableDefault:
    """Stand-in for a callable default; ``repr`` returns the qualified name."""

    def __init__(self, qualname: str) -> None:
        self._qualname = qualname

    def __repr__(self) -> str:
        return f"<{self._qualname}>"


def _wrap_signature(name: str, sig: inspect.Signature) -> str:
    parts = [f"{p}" for p in sig.parameters.values()]
    indent = "    "
    body = (",\n" + indent).join(parts)
    ret = (
        ""
        if sig.return_annotation is inspect.Signature.empty
        else f" -> {_format_annotation(sig.return_annotation)}"
    )
    return f"{name}(\n{indent}{body},\n){ret}"


def _format_annotation(ann: Any) -> str:
    if isinstance(ann, str):
        return ann
    if hasattr(ann, "__qualname__"):
        return ann.__qualname__
    return str(ann)


def render_value_repr(name: str, obj: Any) -> str:
    """For non-callables (registries), render the public contents."""
    if isinstance(obj, dict):
        rows = "\n".join(f"  {k!r}: {type(v).__name__}" for k, v in sorted(obj.items()))
        return f"```python\n{name} = {{\n{rows}\n}}\n```"
    return f"```python\n{name} = {obj!r}\n```"


def get_source_link(obj: Any) -> str | None:
    """Return a ``GITHUB_BLOB`` URL to the symbol's source definition."""
    try:
        source_file = inspect.getsourcefile(obj)
        _, lineno = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None
    if source_file is None:
        return None
    src_path = Path(source_file).resolve()
    try:
        rel = src_path.relative_to(Path(__file__).resolve().parent.parent)
    except ValueError:
        return None
    return f"{GITHUB_BLOB}/{rel.as_posix()}#L{lineno}"


def render_entry(name: str, module_name: str) -> str:
    """Render a single API entry — heading, signature, parsed docstring, link."""
    obj = getattr(hybridmodels, name)
    parsed = parse_docstring(inspect.getdoc(obj))

    parts: list[str] = []
    parts.append(f'<a id="{_slugify(name)}"></a>')
    # Heading uses backticks for monospace; classes get no parens, callables get ().
    if inspect.isclass(obj):
        parts.append(f"### `{name}`")
    elif callable(obj):
        parts.append(f"### `{name}()`")
    else:
        parts.append(f"### `{name}`")

    # Module line (so users know the canonical import path).
    parts.append(
        f"<small>`from {module_name} import {name}` &nbsp;·&nbsp; "
        f"also re-exported as `hybridmodels.{name}`</small>"
    )

    # Signature or value block.
    sig = render_signature(name, obj)
    if sig is not None:
        parts.append(f"```python\n{sig}\n```")
    elif not callable(obj):
        parts.append(render_value_repr(name, obj))

    # Summary + body prose.
    if parsed.summary:
        parts.append(parsed.summary)
    if parsed.body:
        parts.append(parsed.body)

    # Sections.
    for title, body in parsed.sections:
        if title in _PARAM_SECTIONS:
            rows = parse_param_section(body)
            header = (
                "Parameter"
                if title == "Parameters"
                else "Field"
                if title == "Attributes"
                else "Item"
            )
            table = render_param_table(rows, header=header)
            parts.append(f"**{title}**\n\n{table}" if table else f"**{title}**\n\n{body}")
        else:
            parts.append(f"**{title}**\n\n{body}")

    # Source link.
    link = get_source_link(obj)
    if link is not None:
        parts.append(f"<small>[Source]({link})</small>")

    # For classes, emit a sub-entry per public method defined on the class
    # itself (not inherited). Keeps method docstrings — like
    # ``MLPPredictor.with_zero_final_head`` — inside the API page rather
    # than buried in source.
    if inspect.isclass(obj):
        for method_name, method_obj in _public_methods(obj):
            parts.append(_render_method_entry(name, method_name, method_obj))

    return "\n\n".join(parts)


def _public_methods(cls: type) -> list[tuple[str, Any]]:
    """Return ``(name, fn)`` pairs for documentable methods defined on ``cls``.

    A method is documentable when it satisfies all of:

    - public name (no leading underscore — dunder methods are noisy and
      typically restate framework conventions)
    - has a docstring (no point emitting a heading with nothing under it)
    - is defined on this class, not inherited (``__qualname__`` starts
      with the class name)
    """
    out: list[tuple[str, Any]] = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if not inspect.getdoc(member):
            continue
        qualname = getattr(member, "__qualname__", "")
        if not qualname.startswith(cls.__name__ + "."):
            continue
        out.append((name, member))
    return out


def _render_method_entry(class_name: str, method_name: str, method_obj: Any) -> str:
    """Render one ``#### ClassName.method()`` sub-entry inside a class entry."""
    parsed = parse_docstring(inspect.getdoc(method_obj))
    parts: list[str] = []
    parts.append(f"#### `{class_name}.{method_name}()`")

    sig = render_signature(method_name, method_obj)
    if sig is not None:
        parts.append(f"```python\n{sig}\n```")

    if parsed.summary:
        parts.append(parsed.summary)
    if parsed.body:
        parts.append(parsed.body)

    for title, body in parsed.sections:
        if title in _PARAM_SECTIONS:
            rows = parse_param_section(body)
            header = (
                "Parameter"
                if title == "Parameters"
                else "Field"
                if title == "Attributes"
                else "Item"
            )
            table = render_param_table(rows, header=header)
            parts.append(f"**{title}**\n\n{table}" if table else f"**{title}**\n\n{body}")
        else:
            parts.append(f"**{title}**\n\n{body}")

    link = get_source_link(method_obj)
    if link is not None:
        parts.append(f"<small>[Source]({link})</small>")

    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Page assembly                                                               #
# --------------------------------------------------------------------------- #

PAGE_INTROS: dict[str, str] = {
    "data": (
        "The data layer turns irregular, sparse experiment records into a "
        "JAX-traceable [`Dataset`](#dataset). Every observation channel can "
        "have its own timestamps; missing values are represented by a boolean "
        "mask, never by NaN sentinels. Experiments with the same number of "
        "union timestamps are stacked into one [`BucketPayload`](#bucketpayload), "
        "so each bucket compiles once and reuses its trace.\n\n"
        "**Typical flow:** build `ChannelObs` per channel → wrap in `Experiment` "
        "via [`make_experiment`](#make_experiment) → batch via "
        "[`make_dataset`](#make_dataset) → optionally partition with "
        "[`split_dataset`](#split_dataset)."
    ),
    "predictors": (
        "Predictors are the trainable components of a hybrid model — Equinox "
        "modules with a fixed `Array → Array` signature. The framework wraps "
        "them in a [`BoundedPredictor`](#boundedpredictor), which sigmoid-scales "
        "physical-unit inputs into a latent box, runs the inner predictor, and "
        "scales the output back to physical units. Inner predictors never see "
        "or enforce bounds.\n\n"
        "**Predictors pytree convention:** any pytree of `eqx.Module` leaves is "
        "accepted (single module, tuple, dict, NamedTuple). The single-predictor "
        "case is conventionally written as `(predictor,)` so the surrounding "
        "code never branches on container type."
    ),
    "solver": (
        "[`SolverConfig`](#solverconfig) bundles a `diffrax` solver instance "
        "with its tolerances and step controls. Every field is static, so the "
        "config is closed over by jitted functions without re-tracing on value "
        "changes (a tolerance change does trigger a recompile, which is what "
        "we want).\n\n"
        "Solvers are looked up by name through [`SOLVER_REGISTRY`](#solver_registry); "
        "[`register_solver`](#register_solver) extends the registry with custom "
        "implementations so saved configs round-trip cleanly."
    ),
    "training": (
        "Two training entry points share the same `(predictors, dataset, "
        "config, *, simulate_fn, solver, trainable, key, ui)` signature:\n\n"
        "- [`train_with_optax`](#train_with_optax) — gradient-based, multi-phase "
        "  schedule, optional shared tournament for warm-up restarts.\n"
        "- [`train_with_evosax`](#train_with_evosax) — population-based search "
        "  via evosax strategies; useful when the loss landscape is "
        "  non-differentiable or has many local minima.\n\n"
        "Both return `(loss_history, trained_predictors)`. Both require a "
        "`key` keyword-only argument so reproducibility never relies on an "
        "implicit default."
    ),
    "losses": (
        "Loss functions consume a model's predicted output `[N, T, D]` and "
        "the bucket payload, return a scalar, and respect the bucket mask. "
        "Two families: `masked_*` weights every observation equally; `bal_*` "
        "normalises per-experiment so duplication of one experiment can't "
        "dominate the gradient.\n\n"
        'Pass them by name via training-config `loss="mse"` (resolved through '
        "[`LOSS_REGISTRY`](#loss_registry)) or as a callable for custom losses."
    ),
    "trainable": (
        "Trainability is encoded as a boolean PyTree mask matching the "
        "predictors pytree's structure. The training loop calls "
        "`eqx.partition(predictors, mask)` once at start, optimises only the "
        "`True` leaves, and re-combines.\n\n"
        "[`trainable_mask`](#trainable_mask) builds the default mask "
        "(every inexact-array leaf trainable). The `freeze_*` helpers compose "
        "to zero out subsets — by path, by module type, or by arbitrary "
        "predicate."
    ),
    "prediction": (
        "Forward-simulate trained predictors against a dataset. "
        "[`predict_bucket`](#predict_bucket) is the single-bucket primitive "
        "(JIT-compiled, vmapped over the bucket's `N` axis); "
        "[`predict_dataset`](#predict_dataset) walks every bucket and returns "
        "one `[N, T, D]` array per bucket."
    ),
    "serialise": (
        "Serialisation uses Equinox's `tree_serialise_leaves` / "
        "`tree_deserialise_leaves` under the hood. "
        "[`save_predictors`](#save_predictors) / [`load_predictors`](#load_predictors) "
        "round-trip a single predictor tree; [`save_run`](#save_run) / "
        "[`load_run`](#load_run) bundle predictors, solver config, training "
        "config, and loss history into one directory.\n\n"
        "**Loading requires a template.** Equinox can't reconstruct module "
        "shapes from a binary blob, so you instantiate the same predictor "
        "structure (same shapes, same Module types) and the loader fills its "
        "leaves."
    ),
    "ui": (
        "Training UIs are protocols implemented by `RichTrainingUI` / "
        "`RichEvosaxUI` (live dashboards) and `SilentUI` (no-op). Pass via "
        "the `ui=` keyword on `train_with_optax` / `train_with_evosax`; the "
        "trainer calls lifecycle hooks (`on_phase_start`, `on_step_end`, "
        "`on_run_end`, ...) at the right points. Custom UIs implement the "
        "matching protocol — useful for piping training metrics into your "
        "own logger."
    ),
    "rng": (
        "Reproducibility is built on named folds: every random operation "
        "derives its key from a single root `key` via "
        "[`fold(root, name)`](#fold). Reordering operations or reorganising "
        "code doesn't change the keys downstream of unchanged names — "
        "compare to `jax.random.split`, which is positional and very "
        "fragile under refactors.\n\n"
        'Names used internally: `"init"`, `"tournament"`, `"phase_{i}"`, '
        '`"evosax_init"`, `"evosax_ask_{gen}"`. User code can fold its own '
        "names off the same root without collisions."
    ),
}


def render_page(slug: str, title: str, symbols: tuple[str, ...]) -> str:
    """Render one full API reference page as a markdown string."""
    intro = PAGE_INTROS.get(slug, "")
    parts: list[str] = [f"# {title}", ""]
    if intro:
        parts.append(intro)
        parts.append("")
    parts.append("## Quick links")
    parts.append("")
    for name in symbols:
        parts.append(f"- [`{name}`](#{_slugify(name)})")
    parts.append("")
    parts.append("---")
    parts.append("")

    # Resolve module path per symbol — used for the import line in each entry.
    exports = hybridmodels._EXPORTS
    for name in symbols:
        module_name = exports.get(name, "hybridmodels")
        parts.append(render_entry(name, module_name))
        parts.append("")
        parts.append("---")
        parts.append("")
    # Strip the trailing separator.
    while parts and parts[-1] in ("---", ""):
        parts.pop()
    return "\n".join(parts).rstrip() + "\n"


def render_index() -> str:
    """Render the API landing page (`docs/api/index.md`)."""
    parts = [
        "# API Reference",
        "",
        "The public surface is split across ten pages, grouped by concern. "
        "Every symbol below is also re-exported at the top level — "
        "`from hybridmodels import MLPPredictor` works exactly like "
        "`from hybridmodels.predictors import MLPPredictor`.",
        "",
    ]
    for slug, title, symbols in PAGES:
        parts.append(f"## [{title}](/api/{slug})")
        parts.append("")
        # Two-column inline list of symbols, comma-separated.
        items = ", ".join(f"[`{s}`](/api/{slug}#{_slugify(s)})" for s in symbols)
        parts.append(items)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Coverage check                                                              #
# --------------------------------------------------------------------------- #


def check_coverage() -> list[str]:
    """Return public-API symbols that appear in ``__all__`` but no PAGES group.

    A non-empty list means the docs are missing a symbol. CI fails if so.
    """
    documented: set[str] = set()
    for _, _, symbols in PAGES:
        documented.update(symbols)
    return sorted(set(hybridmodels.__all__) - documented)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "api",
        help="Output directory for generated markdown.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated content differs from on-disk files.",
    )
    args = parser.parse_args()

    missing = check_coverage()
    if missing:
        print(
            f"error: {len(missing)} public symbol(s) in __all__ have no docs entry: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: dict[Path, str] = {out_dir / "index.md": render_index()}
    for slug, title, symbols in PAGES:
        pages[out_dir / f"{slug}.md"] = render_page(slug, title, symbols)

    if args.check:
        diffs: list[Path] = []
        for path, content in pages.items():
            if not path.exists() or path.read_text() != content:
                diffs.append(path)
        if diffs:
            print(
                "error: API docs out of sync; run `npm run docs:gen`. "
                f"Diffs in: {', '.join(str(p) for p in diffs)}",
                file=sys.stderr,
            )
            return 1
        print("API docs are up to date.")
        return 0

    for path, content in pages.items():
        path.write_text(content)
        print(f"wrote {path.relative_to(out_dir.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
