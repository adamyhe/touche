# Contributing

Development is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/adamyhe/touche.git
cd touche/
uv sync --dev
uv run touche --help
```

## Running checks

```bash
uv run pytest                       # run the full test suite
uv run pytest tests/test_apa.py     # run one test file
uv run ruff check src tests         # lint (this is what CI runs)
uv build                            # build sdist/wheel
```

`mypy` is listed as a dev dependency but is not run in CI.

Numba is a core dependency (installed by `uv sync --dev`), so `backend="numba"`
code paths are always exercisable without an extra install step.

CI (`.github/workflows/ci.yml`) runs the same `ruff check`/`pytest`/`uv build`
steps above across the Python 3.10-3.14 support matrix on every pull request
and push to `main`. See [`docs/testing-and-publishing.md`](docs/testing-and-publishing.md)
for CI/release details, which are maintainer-facing rather than needed for a
typical contribution.

## Before opening a pull request

- Run `uv run ruff check src tests` and `uv run pytest` locally; both must
  pass in CI.
- Keep `docs/` current when changing CLI flags or public API signatures (see
  [`docs/README.md`](docs/README.md) for the docs layout).
- When in doubt about expected output shape or a threshold/default, check the
  corresponding reference script under `_reference/E-P_contacts/` (a local,
  gitignored clone of the [original Danko Lab reference workflows](https://github.com/Danko-Lab/E-P_contacts)
  `touche` refactors) rather than guessing -- see [`CITATION.cff`](CITATION.cff)
  for how to cite that original implementation and its associated paper.
- Implementation plans, experiment logs, and agent-facing notes belong under
  `notes/`, not `docs/`.
