# hybridmodels docs

VitePress documentation site for the `hybridmodels` package.

## Layout

```
docs/
├── index.md                      # Homepage (hero + features grid)
├── guide/                        # Hand-written narrative pages
│   ├── getting-started.md
│   ├── concepts.md
│   ├── training.md
│   └── recommendations.md
├── examples/                     # Hand-written walkthroughs
│   ├── crystallisation.md        # Featured first; canonical end-to-end
│   └── pendulum.md
├── api/                          # AUTO-GENERATED — do not edit by hand
│   ├── index.md
│   └── {data,predictors,solver,training,losses,trainable,prediction,serialise,ui,rng}.md
├── .vitepress/config.ts
└── package.json
```

The API reference under `docs/api/` is regenerated from docstrings by
`scripts/gen_api_docs.py`; edit the docstrings in `src/hybridmodels/`,
not the generated markdown.

## Local development

```bash
# One-time install of vitepress + math plugin.
npm install --prefix docs

# Regenerate API reference from docstrings.
npm --prefix docs run docs:gen

# Live-reloading dev server.
npm --prefix docs run docs:dev

# Production build (used by CI).
npm --prefix docs run docs:build

# Verify generated docs are committed (CI).
npm --prefix docs run docs:check
```

## How API generation works

`scripts/gen_api_docs.py` introspects `hybridmodels.__all__`, groups symbols
by output page (one per concern: data, predictors, training, ...), parses
each docstring as numpy-style, and emits markdown with:

- An anchor heading (`### `name()``)
- The runtime signature (callable defaults are stabilised so output is
  deterministic)
- Parameters / Returns / Attributes / Raises rendered as two-column tables
- Other sections (Notes, Examples, Construction, Pipeline, ...) passed
  through verbatim
- A source-code link pointing at the GitHub blob

To document a new public symbol:

1. Write its docstring in numpy style (Parameters / Returns / Attributes /
   Notes / Examples sections).
2. Add it to `hybridmodels.__all__` and `hybridmodels._EXPORTS`.
3. Add it to the appropriate `PAGES` group in `scripts/gen_api_docs.py`.
4. Run `npm --prefix docs run docs:gen` and commit the regenerated markdown.

The `--check` mode used by CI fails if any generated file would change,
catching either a docstring update that wasn't reflected in the docs or a
symbol that wasn't added to `PAGES`.
