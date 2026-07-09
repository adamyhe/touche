# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`touche` (PyPI: `ep-touche`) is a Python package + CLI that refactors the
Danko Lab `E-P_contacts` R/shell reference workflows for high-resolution
chromatin contact (Micro-C) analysis into a dependency-light, faster
implementation. It starts from processed pairs files (from `distiller-nf` or
equivalent) — it does not do FASTQ alignment, dedup, or cooler generation.

The original reference implementation lives at `_reference/E-P_contacts/`
(R/awk/shell scripts). New Python implementations are expected to reproduce
its numeric behavior; when in doubt about expected output shape or a
threshold/default, check the corresponding reference script there before
changing behavior.

`touche` uses `polars` as its only dataframe library, both internally and at
the public `touche.api` surface (`pl.DataFrame`/`pl.LazyFrame`) — `pandas` is
not a dependency. CSV outputs have no implicit index column (polars has no
index concept); anywhere a row label matters (e.g. APA's pixel-bin matrices)
it's an explicit `bin_label` column instead.

## Commands

Development uses `uv`.

```bash
uv sync --dev                       # install package + dev deps
uv run touche --help                # run the CLI
uv run pytest                       # run the full test suite
uv run pytest tests/test_apa.py     # run one test file
uv run pytest tests/test_apa.py::test_name  # run one test
uv run ruff check src tests         # lint (this is what CI runs)
uv build                            # build sdist/wheel
```

`mypy` is listed as a dev dependency but is not run in CI (`ci.yml` only runs
`ruff check` and `pytest`).

Install the optional Numba acceleration extra with `pip install "ep-touche[fast]"`
or `uv sync --extra fast` — needed to exercise `backend="numba"` code paths.

CI (`.github/workflows/ci.yml`) runs on Python 3.10/3.11/3.12: `uv sync --dev`,
`ruff check src tests`, `pytest`, `uv build`. See `docs/testing-and-publishing.md`
for the release/publish process (trusted PyPI publishing via `publish.yml`).

## Architecture

### Layering

- `touche.io` — `scan_pairs` builds a lazy `polars.LazyFrame` over pairs files
  (canonical 9-column `touche` format vs. 10+ column `distiller`/pairtools
  format, `source="auto"` sniffs the layout from the first data row's field
  count). All downstream code reads pairs through this module.
- `touche.models` — shared frozen dataclasses: `ContactIndex` (numpy arrays
  for one chromosome: `pos_a`, `pos_b`, strand, mapq), `PairStats`,
  `NamedPath`/`NamedDepth` (used for `NAME=PATH` / `NAME=INTEGER` CLI
  arguments across treatment/control commands).
- `touche.contacts` — builds `ContactIndex` per chromosome from a pairs file
  and reads/writes chromosome-sharded NPZ caches (`build_contact_indexes`,
  `build_npz_cache`, `load_npz_cache`, `write_npz_cache`). This is the
  in-memory numeric representation almost every compute function operates on.
  `build_npz_cache`'s default (non-`"all"`) path spools the scanned pairs to a
  local Parquet file once, then materializes and writes one chromosome's rows
  at a time — peak memory is bounded by the largest single chromosome, not the
  whole file.
- `touche.preprocess` — pairs filtering/conversion (mapq + cis/trans filters,
  distiller→touche format conversion).
- `touche.pair_stats` — QC accumulation (row counts, cis/trans, mapq
  pass/fail, per-chromosome, distance histogram) shared by `preprocess qc`,
  `preprocess summarize`, and cache building (so a large file is scanned once
  for both the cache and its QC).
- Domain modules — `touche.local_decay`, `touche.apa`, `touche.background`:
  each exposes an in-memory `compute_*` function operating on `ContactIndex`
  objects/anchors, plus a file-driven wrapper (e.g. `call_local_decay`,
  `aggregate_apa`, `count_ep_and_background`) that reads inputs, calls the
  compute function, and writes output files.
- `touche.pipelines` — `run_*_pipeline` functions that chain the file-driven
  wrappers for a full command-group workflow (e.g. call → assign → plot for
  local-decay) and write a `manifest.json` (inputs, parameters, outputs,
  metrics, `touche` version, elapsed time, optional per-step timings).
- `touche.api` — the curated notebook-facing re-export surface (`import
  touche.api as tt`); update `__all__` here when adding a new public
  compute/plot function intended for interactive use.
- `touche.cli/` — one argparse module per command group (`preprocess.py`,
  `local_decay.py`, `background.py`, `apa.py`), wired together in `main.py`.
  Each subcommand's `func` callback maps CLI args to the corresponding
  domain/pipeline function and prints a JSON summary via `cli/utils.py`.
- `touche.backends` / `touche.numba_kernels` — optional Numba acceleration.
  Domain compute functions accept `backend="numpy"` (default) or
  `backend="numba"`; `validate_backend` raises a clear error if Numba isn't
  installed. Numba is never imported at package import time — only when a
  numba backend is actually requested. The pure NumPy/polars path is the
  canonical correctness path; Numba kernels must match it (see
  `notes/numba-implementation-plan.md`).
- `touche.instrumentation` — the shared `Instrumentation` dataclass
  (`progress`, `profile`) threaded through compute/pipeline functions as the
  `progress=` argument. `instrument.iter(...)` wraps iterables with `tqdm`
  when enabled; `instrument.step("name")` is a context manager that records
  step timings when `profile=True`. CLI flags: `--progress`/`--profile`.

### Contact indexing strategies

Local-decay (and similar chromosome-scoped work) supports three
`index_strategy` values, most relevant when touching caching or performance
code:

- `cache` (default): build/reuse chromosome-sharded NPZ caches under
  `--cache-dir` (or `contact_index_cache/` next to the output), loading one
  chromosome shard at a time — lowest memory.
- `all`: read the whole pairs file once, hold every chromosome in memory —
  fastest given enough RAM.
- `chromosome`: re-scan the pairs file once per bait chromosome, keeping only
  one chromosome resident — no persistent cache files, but slow on gzipped
  input.

### Pairs source formats

Most commands take `--source {auto,distiller,touche}`. `touche` format is the
canonical 9-column layout this package writes (`preprocess filter-pairs`/
`convert-pairs`); `distiller` is pairtools/distiller-nf's 10+ column layout
(leading read_id column). `auto` infers from field count — don't rely on
`auto` when column counts could plausibly overlap between formats.

### Repository conventions

- `docs/` — human-facing usage guides a `touche` user should read directly
  (CLI reference, API guide, preprocessing, reproducing reference plots,
  testing/publishing). Keep these current when changing CLI flags or public
  API signatures.
- `notes/` — agent-facing implementation plans, benchmark logs, and design
  sketches. Do not put user-facing documentation here, and don't move scratch
  planning into `docs/` unless it's rewritten for users.
- `_reference/E-P_contacts/` — the original reference workflows being ported;
  treat as read-only prior art, not something to modify.
- Manifests, cache directories, and other generated outputs (`.cache/`,
  `contact_index_cache/`, `results/`) are run artifacts, not source.
