# touche

[![PyPI](https://img.shields.io/pypi/v/ep-touche)](https://pypi.org/project/ep-touche/)
[![Tests](https://github.com/adamyhe/touche/actions/workflows/ci.yml/badge.svg)](https://github.com/adamyhe/touche/actions/workflows/ci.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ep-touche?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ep-touche)

Python API and CLI tools for analyzing enhancer-promoter contacts (touches) from high-resolution chromatin contact data, refactored from the [Danko Lab E-P_contacts reference workflows](https://github.com/Danko-Lab/E-P_contacts).

`touche` starts from processed pairs files. Raw FASTQ processing, alignment,
deduplication, and cooler generation should be handled by an external workflow
such as `distiller-nf`.

## Status

The current implementation includes:

- Micro-C pairs conversion, filtering, QC, and chromosome-sharded NPZ caches
- local-decay contact calling, pair-type assignment, and plotting
- APA aggregation and inter-sample APA comparison
- enhancer/promoter local-background counting and treatment comparison
- pipeline `run` wrappers that preserve intermediate outputs and write JSON
  manifests

## Installation

Install with `pip` from PyPI (package name is `ep-touche`):

```bash
pip install ep-touche
touche --help
```

`uv` can also install and run the package:

```bash
uv add ep-touche
uv run touche --help
```

## CLI Overview

```bash
touche preprocess --help
touche local-decay --help
touche apa --help
touche background --help
```

Available command groups:

- `touche preprocess`: convert/filter pairs, write QC summaries, and build NPZ
  caches.
- `touche local-decay`: call observed/expected contacts, assign pair types, plot
  distributions, or run the full local-decay workflow.
- `touche apa`: aggregate APA matrices, compare treatment/control APAs, or run a
  paired APA workflow.
- `touche background`: count EP/background contacts, compare treatment ratios,
  or run the full EP/background workflow.

See the [CLI reference](docs/cli.md) for examples, common options, and expected
outputs.

## Python API

For notebooks and custom scripts, import the provisional API surface:

```python
import touche.api as tt

indexes = tt.build_contact_indexes("sample.nodups_30_intra.pairs.gz", source="touche")
```

The API is organized around reading pairs and anchors once, running in-memory
compute functions such as `compute_apa`, `compute_local_decay`, and
`compute_ep_and_background`, then displaying or saving returned Matplotlib
figures as needed. Counting functions default to an accelerated Numba
backend (`backend="numba"`), with `backend="numpy"` available as the plain
NumPy reference implementation. Long-running CLI and API calls also support
optional progress bars and lightweight profiling. See the
[API guide](docs/api.md) for examples.

## Documentation

Detailed usage notes live under `docs/`:

- [Docs index](docs/README.md): human-facing guides and documentation
  conventions.
- [CLI reference](docs/cli.md): command groups, common options, examples,
  outputs, and run-wrapper manifests.
- [Micro-C preprocessing](docs/micro-c-preprocessing.md): distiller-nf boundary,
  pairs format expectations, filtering, QC, and cache building.
- [API](docs/api.md): provisional in-memory APIs for notebooks, interactive analyses,
  and custom scripts.
- [Reproducing reference plots](docs/reproducing-reference-plots.md): end-to-end
  commands for the reference local-decay, APA, and EP/background plots.
- [Testing and publishing](docs/testing-and-publishing.md): CI, local checks,
  and PyPI release workflow for the `ep-touche` distribution.

Implementation plans, experiment logs, and agent-facing notes belong in
`notes/`.

## Development

Development is managed with `uv`.

```bash
git clone https://github.com/adamyhe/touche.git
cd touche/
uv sync --dev
uv run touche --help
uv run pytest
```
