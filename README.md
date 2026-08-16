# touche

[![PyPI](https://img.shields.io/pypi/v/ep-touche)](https://pypi.org/project/ep-touche/)
[![Tests](https://github.com/adamyhe/touche/actions/workflows/ci.yml/badge.svg)](https://github.com/adamyhe/touche/actions/workflows/ci.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ep-touche?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ep-touche)

Python API and CLI tools for analyzing enhancer-promoter contacts (touches) from high-resolution chromatin contact data, refactored from the [Danko Lab E-P_contacts reference workflows](https://github.com/Danko-Lab/E-P_contacts).

`touche` starts from processed pairs files. Raw FASTQ processing, alignment,
deduplication, and cooler generation should be handled by an external workflow
such as [distiller-nf](https://github.com/open2c/distiller-nf).

## Status

The current implementation includes CLI tools+API for:

- Micro-C pairs conversion, filtering, QC, and chromosome-sharded NPZ caches
- local-decay contact calling, pair-type assignment, and plotting
- Aggregated peak analysis (APA) aggregation and inter-sample APA comparison
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

Or if you just want the CLI:

```bash
uv tool install ep-touche
touche --help
```

`local-decay` also supports a statsmodels-backed LOWESS path
(`lowess_backend="statsmodels"`) for exact reference comparisons
(at the cost of being much slower than our custom implementation).
Install the optional `legacy` extra only if you need that backend:

```bash
pip install ep-touche[legacy]
# or: uv sync --extra legacy
```

### Older CPUs and Rosetta

Polars publishes an `lts-cpu` wheel for older CPUs and for x86-64 Python on
Apple Silicon under Rosetta. Because `polars-lts-cpu` provides the same
`import polars` module but is a different Python distribution, it does not
automatically satisfy `ep-touche`'s normal `polars>=1.0` dependency. If the
standard Polars wheel does not run on your machine, install `ep-touche`, then
replace Polars in that environment:

```bash
pip install ep-touche
pip uninstall polars
pip install "polars-lts-cpu>=1.0"
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

## Typical workflow

Start from analysis-ready `.pairs` or `.pairs.gz` files produced by distiller-nf
or an equivalent workflow.

1. Filter or convert pairs with `touche preprocess`.
2. For real or repeated local-decay runs, build a position-only cache with
   `touche preprocess build-cache --no-metadata`.
3. Use the `run` wrappers for end-to-end analyses:
   `touche local-decay run`, `touche apa run`, and `touche background run`.
4. Use individual subcommands such as `local-decay call` or `background count`
   when debugging or replacing one stage.

Many of the computation-heavy steps use `numba` acceleration and parallelism.
Set `NUMBA_NUM_THREADS` before running to control CPU usage (default is to use
all threads on machine):

```bash
NUMBA_NUM_THREADS=8 touche background run ...
```

## Python API

For notebooks and custom scripts, import the provisional API surface:

```python
import touche.api as tt

indexes = tt.build_contact_indexes("sample.nodups_30_intra.pairs.gz", source="touche")
```

The API is organized around reading pairs and anchors once, running in-memory
compute functions such as `compute_apa`, `compute_local_decay`, and
`compute_ep_and_background`, then displaying or saving returned Matplotlib
figures as needed. Long-running CLI and API calls support optional progress
bars and lightweight profiling. See the [API guide](docs/api.md) for examples.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a development checkout,
running tests/lint, and pull request guidelines.

## Citation

If you use `touche`, please cite it -- and the original Danko Lab
E-P_contacts reference implementation and its associated paper, since
`touche` is a refactor of that reference workflow. See
[CITATION.cff](CITATION.cff).
