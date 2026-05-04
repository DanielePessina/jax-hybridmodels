# jax-hybridmodels

A JAX/Equinox library for **hybrid models** — composing trainable function approximators (MLP, KAN, ...) with user-written ODE dynamics, trained on irregular time-series experiments. Crystallisation kinetics is the canonical example, not the scope.

## Documentation

The documentation site is built with VitePress and deployed to GitHub Pages at
**<https://danielepessina.github.io/jax-hybridmodels/>**.

Site sections:

- **Guide** — Getting Started, Concepts, Training, Recommendations.
- **Examples** — the [Crystallisation walkthrough](https://danielepessina.github.io/jax-hybridmodels/examples/crystallisation) (canonical end-to-end), and a [Harmonic Oscillator](https://danielepessina.github.io/jax-hybridmodels/examples/pendulum) sanity check with a known optimum.
- **API Reference** — per-module pages auto-generated from docstrings.

The full source tree for the site lives under [`docs/`](./docs).

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

### Build & deploy on CI — opt-in via commit message

The `.github/workflows/docs.yml` workflow only runs when the **head commit
message contains the literal trigger `[build docs]`**. Routine pushes — even
ones that touch `docs/` or `src/` — do not redeploy the site, so the
public Pages URL only updates when you explicitly mean it to.

To deploy:

```bash
git commit -m "docs: rewrite the training guide [build docs]"
git push
```

To deploy without a new commit (e.g. recovering from a failed run), use the
**Run workflow** button on the [Actions tab](https://github.com/DanielePessina/jax-hybridmodels/actions) — `workflow_dispatch` bypasses the commit-message gate.

The workflow runs the API-docs `--check` first and fails if the generated
markdown is out of sync with the source docstrings, so a `[build docs]`
commit that forgot `npm run docs:gen` will surface in CI rather than silently
deploy a stale API reference.

### Regenerate the API reference

The pages under `docs/api/` are **auto-generated** from public-API
docstrings by `scripts/gen_api_docs.py`. Edit the docstrings in
`src/hybridmodels/`, never the generated markdown.

```bash
# Regenerate every docs/api/*.md page from current docstrings.
npm --prefix docs run docs:gen

# Or directly (same effect):
uv run python scripts/gen_api_docs.py

# Verify on-disk docs match what the generator would produce — used by CI.
uv run python scripts/gen_api_docs.py --check
```

Adding a new public symbol takes four steps:

1. Write the docstring in numpy style (Parameters / Returns / Attributes /
   Notes / Examples sections — the generator parses these into tables).
2. Add the symbol to `hybridmodels.__all__` and `hybridmodels._EXPORTS`.
3. Add it to the appropriate `PAGES` group in `scripts/gen_api_docs.py`.
4. Run `npm --prefix docs run docs:gen` and commit the regenerated markdown
   along with your source change.

Removing a symbol is the same in reverse — drop it from `__all__`,
`_EXPORTS`, and `PAGES`, then regenerate. The generator's coverage check
fails CI if a public symbol exists in `__all__` but no `PAGES` group, so
new exports cannot ship undocumented.

## Installation (development)

```bash
uv sync
```

This installs `hybridmodels` in editable mode together with `jax`,
`equinox`, `diffrax`, `optax`, `evosax`, `jaxkan`, and the small CLI/UI
dependencies.

## License

MIT — see [LICENSE](./LICENSE).
