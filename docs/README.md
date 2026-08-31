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
│   ├── custom-predictors.md
│   └── recommendations.md
├── examples/                     # Walkthroughs of the example scripts
│   ├── custom-loop.md            # Write your own training loop (kernels)
│   ├── crystallisation.md        # Canonical end-to-end
│   ├── hybrid-ode.md
│   ├── batch-reactor.md
│   ├── batch-reactor-rl.md
│   ├── sbml-hybrid.md            # External mechanistic model + neural rate
│   ├── custom-predictor.md
│   ├── supersaturation-poly.md   # NeuralNPolynomial rate law
│   └── pendulum.md
│   └── assets/                   # Committed figures the example pages embed
├── api/                          # AUTO-GENERATED, do not edit by hand
│   ├── index.md
│   └── {data,predictors,penalties,profiles,schedules,transforms,solver,training,kernels,losses,metrics,trainable,prediction,serialise,ui,rng}.md
├── .vitepress/config.ts
└── package.json
```

`scripts/gen_api_docs.py` regenerates the API reference under `docs/api/`
from docstrings. Edit the docstrings in `src/hybridmodels/`, never the
generated markdown.

`scripts/gen_example_figures.py` regenerates the committed figures under
`docs/examples/assets/` by running every example script with its
`--plot-dir` pointed there. Rerun it whenever an example's plotting
changes; `--quick` trades training budgets for iteration speed.

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
