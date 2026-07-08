# touche

Python package and CLI tools for high-resolution chromatin contact analyses,
refactored from the Danko Lab E-P_contacts reference workflows.

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

## Development

Development is managed with `uv`.

```bash
git clone https://github.com/adamyhe/touche.git
cd touche/
uv sync --dev
uv run touche --help
uv run pytest
```

## Documentation

Detailed usage notes live under `docs/`:

- [Micro-C preprocessing](docs/micro-c-preprocessing.md): distiller-nf boundary,
  pairs format expectations, filtering, QC, and cache building.
- [Reproducing reference plots](docs/reproducing-reference-plots.md): end-to-end
  commands for the reference local-decay, APA, and EP/background plots.
- [Testing and publishing](docs/testing-and-publishing.md): CI, local checks,
  and PyPI release workflow for the `ep-touche` distribution.
- [Docs index](docs/README.md): human-facing guides and documentation
  conventions.

Implementation plans, experiment logs, and agent-facing notes belong in
`notes/`.

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

Use each command's `--help` output for the exact options.
