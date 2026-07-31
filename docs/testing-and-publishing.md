# Testing and publishing

This project uses GitHub Actions for continuous integration and PyPI
publishing. The PyPI distribution name is `ep-touche`; the import package and
CLI command remain `touche`.

## Continuous integration

CI is defined in `.github/workflows/ci.yml`.

It runs on:

- pull requests
- pushes to `main`

The CI matrix runs on Python 3.10 through 3.14. Each job:

1. installs `uv`
2. syncs development dependencies with `uv sync --dev --python <matrix-version>`
3. runs `uv run ruff check src tests`
4. runs `uv run pytest`
5. builds the package with `uv build`

Run the same checks locally from a development checkout:

```bash
uv sync --dev
uv run ruff check src tests
uv run pytest
uv build
```

## Publishing to PyPI

Publishing is defined in `.github/workflows/publish.yml`.

It runs on:

- published GitHub releases
- manual `workflow_dispatch`

The package uses Hatchling as its build backend. The workflow builds source and
wheel distributions with `uv build`, validates them with `twine check`, and
publishes to PyPI with trusted publishing.

## PyPI trusted publishing setup

Configure a trusted publisher for the `ep-touche` project on PyPI:

- owner: `adamyhe`
- repository: `touche`
- workflow name: `publish.yml`
- environment name: `pypi`

The GitHub workflow uses the `pypi` environment and requests the `id-token:
write` permission required for trusted publishing. No long-lived PyPI API token
is needed when trusted publishing is configured.

## Release checklist

Before creating a release:

1. update the version in `pyproject.toml`
2. update `uv.lock`
3. run the local checks listed above
4. push changes and confirm CI passes
5. create and publish a GitHub release
6. confirm the package appears at `https://pypi.org/project/ep-touche/`

After installation, users should run:

```bash
pip install ep-touche
touche --help
```
