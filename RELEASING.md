# Releasing jax-hybridmodels

The repository currently prepares `0.2.0b1`. Citation and Zenodo metadata are
intentionally not part of this release.

## One-time GitHub setup

1. Make the repository public.
2. In repository Settings → Pages, select **GitHub Actions** as the source.
3. In Settings → Environments, create an environment named `pypi`. Add a
   required reviewer if releases should require approval.

The Pages workflow deploys on relevant pushes to `main`. The PyPI workflow is
`.github/workflows/release.yml` and publishes through Trusted Publishing.

## One-time PyPI setup

Create a pending publisher for the new project with:

- PyPI project: `jax-hybridmodels`
- GitHub owner: `DanielePessina`
- Repository: `jax-hybridmodels`
- Workflow filename: `release.yml`
- Environment: `pypi`

No long-lived PyPI token is required.

## Release

From a clean `main` checkout:

```bash
uv sync --frozen
uv run pytest -q
uv build
git tag v0.2.0b1
git push origin v0.2.0b1
```

The release workflow checks that the tag matches `pyproject.toml`, builds the
wheel and sdist, validates them, and publishes both files to PyPI.
