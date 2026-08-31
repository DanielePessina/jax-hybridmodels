# jax-hybridmodels

A JAX/Equinox library for combining user-written ODE dynamics with trainable
predictors. It supports bounded physical quantities, regular and irregular
time-series experiments, and Optax or Evosax training. Crystallisation
kinetics is the canonical example, not the scope.

## Documentation

The documentation site is built with VitePress and deployed to GitHub Pages at
**<https://danielepessina.github.io/jax-hybridmodels/>**.

Site sections:

- Guide: Getting Started, Concepts, Training, Custom Predictors, Extending, Recommendations.
- Examples:
  - [Custom training loop](https://danielepessina.github.io/jax-hybridmodels/examples/custom-loop) — write your own loop against the public gradient kernels.
  - [Crystallisation walkthrough](https://danielepessina.github.io/jax-hybridmodels/examples/crystallisation) — canonical end-to-end.
  - [SBML hybrid kinetics](https://danielepessina.github.io/jax-hybridmodels/examples/sbml-hybrid) — an external mechanistic model with a neural rate.
  - [Neural polynomial kinetics](https://danielepessina.github.io/jax-hybridmodels/examples/supersaturation-poly) — a `NeuralNPolynomial` rate law in supersaturation.
  - Plus hybrid-ODE, batch reactor, RL, custom-predictor, and a [Harmonic Oscillator](https://danielepessina.github.io/jax-hybridmodels/examples/pendulum) sanity check with a known optimum.
- API Reference: per-module pages auto-generated from docstrings.

The full source tree for the site lives under [`docs/`](./docs). Start with
[Getting started](https://danielepessina.github.io/jax-hybridmodels/guide/getting-started),
then read [Concepts](https://danielepessina.github.io/jax-hybridmodels/guide/concepts).

### Examples

The examples under [`examples/`](./examples) are plain scripts (no notebook
runtime). Run one directly, or through the docs page that embeds it:

```bash
uv run python examples/custom_loop/train_custom_loop.py
uv run python examples/sbml_hybrid/train_sbml_hybrid.py
uv run python examples/pendulum/train_harmonic.py --no-plot
uv run python examples/supersaturation_poly/train_supersaturation_poly.py
```

A CI workflow (`.github/workflows/examples.yml`) compiles every example and
smoke-runs the fast ones on each push.

### Local preview

```bash
# One-time install of vitepress + the math plugin.
npm install --prefix docs

# Live-reloading dev server at http://localhost:5173/jax-hybridmodels/.
npm --prefix docs run docs:dev

# Production build (mirrors what CI runs).
npm --prefix docs run docs:build

# Verify generated API docs are committed and the build is clean.
npm --prefix docs run docs:check
```

### Build & deploy on CI, opt-in via commit message

The `.github/workflows/docs.yml` workflow only runs when the head commit
message contains one of two literal triggers. Routine pushes do not
redeploy the site, even ones that touch `docs/` or `src/`, so the public
Pages URL only updates when you explicitly mean it to.

| Trigger | When to use it | What CI does |
| --- | --- | --- |
| `[build docs]` | You ran `npm run docs:gen` locally and committed the result. | Runs `docs:check:api` first, failing fast if the in-repo `docs/api/` is out of sync with current docstrings. Builds and deploys. |
| `[regen docs]` | You only edited docstrings and didn't regenerate locally. | Runs `npm run docs:gen` on CI. If the API pages changed, commits them back to the branch with `[skip ci]` (so the auto-commit doesn't trigger another run). Builds and deploys. |

Use whichever feels right for the change you just made. `[build docs]` is
the safer default; it reports drift between docstrings and shipped docs
loudly. `[regen docs]` is the convenience option for "I just touched a
docstring, do the bookkeeping for me."

```bash
# I already ran `npm run docs:gen` and committed the result:
git commit -m "docs: rewrite the training guide [build docs]"

# I only changed a docstring; let CI regenerate the API page for me:
git commit -m "docs(predictors): clarify BoundedPredictor.input_keys [regen docs]"

git push
```

To deploy without a new commit (e.g. recovering from a failed run), use
the Run workflow button on the [Actions tab](https://github.com/DanielePessina/jax-hybridmodels/actions). `workflow_dispatch` bypasses the
commit-message gate and follows the `[build docs]` semantics (check,
don't regen).

> GitHub Pages must be enabled under Settings → Pages → Source:
> GitHub Actions before the first deploy will succeed. The
> `[regen docs]` path also requires the workflow's `contents: write`
> permission, which is set in `docs.yml`. No extra repo configuration
> is needed.

### Regenerate the API reference

The pages under `docs/api/` are auto-generated from public-API
docstrings by `scripts/gen_api_docs.py`. Edit the docstrings in
`src/hybridmodels/`, never the generated markdown.

```bash
# Regenerate every docs/api/*.md page from current docstrings.
npm --prefix docs run docs:gen

# Or directly (same effect):
uv run python scripts/gen_api_docs.py

# Verify on-disk docs match what the generator would produce (used by CI).
uv run python scripts/gen_api_docs.py --check
```

Adding a new public symbol takes four steps:

1. Write the docstring in numpy style (Parameters / Returns / Attributes /
   Notes / Examples sections, which the generator parses into tables).
2. Add the symbol to `hybridmodels.__all__` and `hybridmodels._EXPORTS`.
3. Add it to the appropriate `PAGES` group in `scripts/gen_api_docs.py`.
4. Run `npm --prefix docs run docs:gen` and commit the regenerated markdown
   along with your source change.

Removing a symbol is the same in reverse: drop it from `__all__`,
`_EXPORTS`, and `PAGES`, then regenerate. The generator's coverage check
fails CI if a public symbol exists in `__all__` but no `PAGES` group, so
new exports cannot ship undocumented.

## Installation

### From PyPI (once published)

```bash
uv add hybridmodels
```

### Development

```bash
uv sync
```

This installs `hybridmodels` in editable mode together with `jax`,
`equinox`, `diffrax`, `optax`, `evosax`, `jaxkan`, and the small CLI/UI
dependencies.

## Contributing

Run the verification checks before pushing:

```bash
uv run ruff check .        # lint
uv run ty check src        # typecheck
uv run pytest -q           # test suite
npm --prefix docs run docs:build   # docs site
```

## License

MIT. See [LICENSE](./LICENSE).
