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

Numba is a core dependency (installed by `uv sync --dev`), so `backend="numba"`
code paths are always exercisable without an extra install step.

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
- `touche.pairs` — the canonical bait/prey pair table (`PAIR_COLUMNS`), the
  deterministic `pair_id`, BEDPE I/O, and `split_pair_anchors` (the bridge
  from a pair table to the counting kernels' anchors-plus-index-vectors
  form). `pair_id` is built from anchor **centers**, sorted so it is
  anchor-order-invariant: centers because every analysis reduces an anchor
  to its center before counting, which is the only definition under which
  local-decay's point anchors and APA/background's BED intervals give the
  same id for the same biological pair. It is an explicit string, never a
  built-in hash, so it is stable across polars/Python versions and
  independent of chunking.
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
- Statistics modules — `touche.stats` holds pure numeric primitives
  (p-value adjustment, tail tests, effect sizes, cluster bootstrap);
  `touche.metadata` holds the `StatResult`/`MethodInfo` envelope and
  `METHOD_REGISTRY`; `touche.significance` (per-pair contact testing and
  calibration), `touche.compare` (group comparisons, correlations,
  covariate matching), `touche.apa_masks` (APA submatrix scores), and
  `touche.differential` (EP-versus-background odds ratios) are the analyses
  built on them. Every inferential function returns a `StatResult`, and
  writing one always writes a `.meta.json` sidecar -- a q-value without its
  test universe and inference class is not reproducible. Add new methods to
  `METHOD_REGISTRY` and document them in `docs/statistics.md`.
- `touche.adapters` — import/export for external callers (FitHiC2, MaxHiC,
  HiC-DC+, Mustache, Peakachu, Chromosight, HiCCUPS, generic BEDPE) plus
  `annotate_pairs`. Column resolution is header-name-driven with aliases and
  raises listing the real header on a mismatch; `touche` deliberately does
  not implement its own loop caller.
- `touche.api` — the curated notebook-facing re-export surface (`import
  touche.api as tt`); update `__all__` here when adding a new public
  compute/plot/statistics function intended for interactive use. A test
  asserts `__all__` and the module's public names stay in sync.
- `touche.cli/` — one argparse module per command group (`preprocess.py`,
  `pairs.py`, `local_decay.py`, `background.py`, `apa.py`), wired together in
  `main.py`.
  Each subcommand's `func` callback maps CLI args to the corresponding
  domain/pipeline function and prints a JSON summary via `cli/utils.py`.
- `touche.backends` / `touche.numba/` — Numba acceleration, a core
  dependency. `touche.numba` is a subpackage split by the domain that uses
  each kernel (`touche.numba.apa`, `.background`, `.local_decay`, `.stats`);
  numba is never imported at package import time — only when a
  numba-accelerated function is actually called, so every `from
  touche.numba.<domain> import ...` happens inside a function body, never at
  a domain module's own top level. Counting (APA matrix/1D signal,
  EP/background pairs, local-decay's observed-count helper) always uses its
  Numba kernel — each was verified exact-equivalent to the plain NumPy
  implementation it replaced (see `notes/numba-implementation-plan.md`), so
  there's no `backend` choice exposed for it. `lowess_backend`
  (`"numba"`/`"statsmodels"`) and `fisher_backend` (`"numba"`/`"scipy"`) are
  different: each numba path is a validated but inexact approximation of its
  alternative, so both choices remain exposed, defaulting to `"numba"`.
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

### Statistical compatibility rules

Every statistical addition is opt-in and no default numerical behavior has
changed. When touching these paths, keep it that way:

- `local-decay call` defaults to `method="binomial"` (the calibrated test)
  with `schema="legacy"`, so the *layout* is still the reference
  nine-column headerless TSV and only the p-value column's meaning differs.
  `schema="tidy"` is the opt-in canonical-schema output with
  `pair_id`/`n_trials`/`p_null`/`q_value`.
- A metadata sidecar is written whenever `method != "legacy_fisher"`, even
  under `schema="legacy"`: that layout is headerless and has no `q_value`
  column, so the sidecar is the only record of which null produced column
  five. `method="legacy_fisher"` writes none, keeping a
  reference-reproduction directory byte-identical -- don't change that.
- `legacy_fisher` is retained for reproducibility only, and is what
  `scripts/reference_replication.py` and
  `docs/reproducing-reference-plots.md` pass explicitly. Its 2x2 table uses
  a fitted expectation as an observed cell and a histogram-bin count as a
  trial total, so it is not a calibrated test and its registry entry says
  so. Do not present its q-values as an FDR-controlled discovery set. Any
  test that pins reference numbers, or that exercises `fisher_backend`,
  must pass it explicitly or it will silently stop testing what it names.
- `background compare` keeps its reference-reproducing zero filter
  (`zero_policy="drop"`) and CPB divisor (`scale="legacy"`, `depth / 1e10`
  -- contacts per *ten* billion despite the name). Inferential work belongs
  in `touche.differential`, which keeps zeros by default.
- Uncertainty always resamples a defensible cluster: chromosomes for APA
  pileups, the caller-supplied `cluster_by` for pair-level comparisons.
  Never resample pixels or treat pairs sharing an anchor as independent.
- A `--pairs-list` is the pair universe exactly as given -- never expanded
  to a product, never pruned by the distance window.

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
- `scripts/` — standalone, runnable tools promoted out of `notes/` once
  they're more than a one-off benchmark log (e.g. `reference_replication.py`,
  which downloads the upstream E-P_contacts example data and replicates its
  reference workflows end to end). Not part of the `touche` package build.
- `_reference/E-P_contacts/` — the original reference workflows being ported;
  treat as read-only prior art, not something to modify.
- Manifests, cache directories, and other generated outputs (`.cache/`,
  `contact_index_cache/`, `results/`) are run artifacts, not source.
