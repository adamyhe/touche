# touche implementation plan

## Goal

Refactor the contents of the E-P_contacts reference repository into `touche`: a Python package with stable CLI entrypoints, reusable APIs, no `rpy2`/R runtime dependency, and substantially lower runtime, memory use, and intermediate disk churn.

The reference checkout lives at `_reference/E-P_contacts/`. It contains three main analysis families plus preprocessing notes:

- Micro-C preprocessing configuration examples
- local contact normalization by distance decay
- APA and inter-sample APA
- enhancer/promoter contacts compared to local background

The first implementation target should preserve the reference behavior closely, then optimize behind that compatibility layer.

## Repository conventions

- `docs/` is for human-facing usage notes, tutorials, and reproducibility guides that a user of `touche` should be able to read directly.
- `notes/` is for agent-facing plans, implementation logs, experiments, design sketches, and migration notes.
- Do not put scratch experiments, temporary benchmark logs, or internal planning in `docs/`.
- If a note matures into user guidance, rewrite it for users before moving it from `notes/` to `docs/`.

## Proposed package layout

```text
touche/
  pyproject.toml
  README.md
  docs/
    micro-c-preprocessing.md
  src/
    touche/
      __init__.py
      cli/
        __init__.py
        __main__.py
        main.py
        utils.py
        preprocess.py
        local_decay.py
        background.py
        apa.py
      io.py
      contacts.py
      intervals.py
      stats.py
      plots.py
      models.py
      preprocess.py
      local_decay.py
      background.py
      apa.py
  tests/
    fixtures/
    test_io.py
    test_local_decay.py
    test_apa.py
    test_background.py
```

The public CLI entrypoint should be:

```bash
touche --help
```

Suggested subcommands:

```bash
touche preprocess filter-pairs
touche preprocess convert-pairs
touche preprocess qc
touche preprocess summarize
touche local-decay call
touche local-decay assign-pair-types
touche local-decay plot
touche local-decay run
touche apa aggregate
touche apa compare
touche apa run
touche background count
touche background compare
touche background run
```

Use stdlib `argparse` for now. Keep the top-level CLI package thin: `touche.cli.main` composes domain-specific CLI modules, and each domain module only translates command-line arguments into calls to the package API.

Keep individual subcommands for reproducibility and debugging, plus ergonomic `run` commands for full pipelines:

- `touche local-decay run`
- `touche apa run`
- `touche background run`

Each `run` command chains the relevant lower-level commands, writes a run manifest, and preserves all individual outputs.

## Current progress

Implemented so far:

- uv-managed Python package with `pyproject.toml`, `.python-version`, and `uv.lock`.
- `src/touche/cli/` package split by command group.
- `touche preprocess filter-pairs`, `convert-pairs`, `qc`, `summarize`, and `build-cache`.
- canonical 9-column contact format parsing and conversion.
- chromosome-sharded NPZ contact caches with a JSON manifest.
- `touche.stats.fisher_greater(...)` using `scipy.stats.fisher_exact` and preserving reference rounding.
- `touche local-decay call`, `assign-pair-types`, and `plot`.
- `touche local-decay run`, `background run`, and pairwise `apa run` wrappers with JSON manifests.
- distiller-nf handoff documentation in `docs/micro-c-preprocessing.md`.
- reference plot reproduction documentation in `docs/reproducing-reference-plots.md`.
- `touche background count`.
- `touche background compare`.
- `touche apa aggregate`.
- `touche apa compare`.
- shared dataclasses in `touche.models`.
- tests for preprocessing, contact caches, local-decay utilities, background comparison, APA comparison, stats, and version metadata.
- package metadata updated for PyPI distribution name `ep-touche`, with `touche` retained as the import package and CLI command.
- Hatchling build backend, GitHub Actions CI/publishing workflows, and Python 3.10 compatibility fix for `touche.pipelines`.
- first notebook-friendly API slice:
  - `touche.api` provisional exports.
  - in-memory `compute_apa`, `compute_ep_and_background`, and `compute_local_decay`.
  - `ApaResult` with `plot()` and `write()`.
  - plot functions return `Figure` and can run without output paths.
  - `docs/notebook-api.md`.

Still pending:

- richer result objects for local decay, background counts, background comparisons, and APA comparisons
- optional full-data reproduction scripts
- optional Numba kernels; detailed plan in `notes/numba-implementation-plan.md`
  - first EP/background counting backend implemented and benchmarked.
  - APA counting backend implemented and benchmarked.
  - local-decay observed-count backend implemented, but initial benchmark shows
    LOWESS dominates runtime.

## Phase 1: Establish compatibility targets

Before rewriting heavily, create small fixtures and regression tests for the current behavior.

Reference scripts to treat as behavior sources:

- `_reference/E-P_contacts/Contact_normalization_by_local_decay/ContactCaller_microC.py`
- `_reference/E-P_contacts/Contact_normalization_by_local_decay/Pair_type_assignment.py`
- `_reference/E-P_contacts/Contact_normalization_by_local_decay/Plotting_obs_over_exp_distribution_by_pair_type.py`
- `_reference/E-P_contacts/APA_and_inter-sample_APA/MicroC_Stranded_Aggregation_pipeline_with_1D_signal.bsh`
- `_reference/E-P_contacts/APA_and_inter-sample_APA/MicroC_Stranded_Aggregation_pipeline_get_bait_matrix.py`
- `_reference/E-P_contacts/APA_and_inter-sample_APA/MicroC_Stranded_Aggregation_pipeline_get_aggregated_matrix.py`
- `_reference/E-P_contacts/APA_and_inter-sample_APA/get_genome_wide_normalization_scores_by_search_window.py`
- `_reference/E-P_contacts/APA_and_inter-sample_APA/Change_calculation_and_visualization.py`
- `_reference/E-P_contacts/EP_contacts_compared_to_local_background/MicroC_EP_and_BG_contacts.bsh`
- `_reference/E-P_contacts/EP_contacts_compared_to_local_background/single_pair_contacts_and_background_calculation.py`
- `_reference/E-P_contacts/EP_contacts_compared_to_local_background/Compering_EP_contacts_between_treatments.py`
- `_reference/E-P_contacts/Micro-C_basic_processing/*.yml`

Define explicit input schemas:

- raw input manifest: sample ID, R1 FASTQ, R2 FASTQ, optional replicate/group metadata
- pairs: `chrA, posA, chrB, posB, strandA, strandB, read_type, mapqA, mapqB`
- BED anchors: `chrom, start, end, strand`, with `center` computed as `(start + end) // 2`
- local-decay anchors: `chrom, center`
- CRISPRi labels: preserve the Gasperini functional/nonfunctional CSV merge keys

Initial tests should include:

- smoke tests for every CLI command
- exact-count tests for synthetic contact data
- matrix-shape and matrix-value tests for APA
- p-value tolerance tests for local-decay Fisher tests
- output-column compatibility tests
- pair-filtering tests for cis/mapq/dedup assumptions
- pairs-format conversion tests

## Phase 2: Remove `rpy2`

The reference code uses `rpy2` only to call R's `fisher.test(..., alternative = "greater")` in `ContactCaller_microC.py`.

Replace it with:

```python
scipy.stats.fisher_exact(table, alternative="greater")
```

For compatibility, keep the reference behavior of rounding table values before computing the p-value:

```python
table = [
    [round(observed), round(expected)],
    [round(total_observed_background), round(total_expected_background)],
]
```

This removes:

- `rpy2`
- `r-base`
- an external R installation
- `LD_LIBRARY_PATH` setup in shell wrappers

Keep the replacement isolated in `touche.stats.fisher_greater(...)` so it is easy to test and easy to adjust if exact R compatibility requires small handling changes.

## Phase 3: Replace shell workflows with Python orchestration

The reference bash scripts are mostly orchestration: splitting by chromosome, creating per-anchor temporary files, gzipping local subsets, and launching many Python processes.

Replace them with package functions:

```python
touche.preprocess.filter_pairs(...)
touche.preprocess.convert_pairs(...)
touche.preprocess.summarize_pairs(...)
touche.local_decay.call_local_decay(...)
touche.apa.aggregate_apa(...)
touche.apa.compare_apa_change(...)
touche.background.count_ep_background(...)
touche.background.compare_background_ratios(...)
```

The CLI should call these APIs directly. For example:

```bash
touche local-decay call \
  --baits _reference/E-P_contacts/Input_files/Gasperini_dREG_based_TRE_baits_hg38.txt \
  --preys _reference/E-P_contacts/Input_files/Gasperini_dREG_based_promoter_preys_hg38.txt \
  --pairs GSE206131_K562_cis_mapq30_pairs.txt.gz \
  --out ContactCaller_microC_output.tsv \
  --threads 30 \
  --dist 1000000 \
  --cap 2000
```

## Phase 4: Micro-C preprocessing

Add Micro-C preprocessing as a first-class pipeline stage, but keep the boundary clear:

- `touche` should document how to run distiller-nf and what outputs/options downstream analyses expect.
- `touche preprocess` should start from distiller/pairtools-style `.pairs` outputs and own conversion, filtering, normalization, and QC summaries.
- Existing mapping/binning tools such as distiller-nf should remain external tools rather than package-managed execution dependencies.

Reference behavior:

- The reference repository stores distiller-nf YAML files in `_reference/E-P_contacts/Micro-C_basic_processing/`.
- Downstream analyses expect processed pairs files derived from distiller-nf using `parsing_options: '--add-columns mapq'` and `drop_readid: True`.
- The documented downstream filter keeps cis pairs with both sides `mapq >= 30` and drops the read ID column:

```bash
zcat prefix.pairs.gz \
  | awk 'BEGIN {OFS = "\t"} ; {if ($1 == "." && $2 == $4 && $9 >= 30 && $10 >= 30) {print $2, $3, $4, $5, $6, $7, $8, $9, $10}}' \
  > prefix.nodups_30_intra.pairs
```

Preprocessing CLI:

```bash
touche preprocess filter-pairs \
  --pairs sample.pairs.gz \
  --out sample.nodups_30_intra.pairs.gz \
  --min-mapq 30 \
  --cis-only
```

By default this writes canonical 9-column `touche` pairs and drops the read ID; use `--keep-read-id` only when preserving distiller-style 10-column rows is explicitly needed.

```bash
touche preprocess convert-pairs \
  --pairs sample.nodups_30_intra.pairs.gz \
  --from distiller \
  --to touche \
  --out sample.touche.pairs.gz
```

```bash
touche preprocess qc \
  --pairs sample.nodups_30_intra.pairs.gz \
  --out sample.qc.json
```

Implementation pieces:

- Add documentation pages with distiller-nf setup notes, required options, and example commands based on the reference YAMLs.
- Add `filter_pairs(...)` as a streaming transformation that handles plain text and gzip input/output.
- Add `convert_pairs(...)` for converting distiller/pairtools-style pairs into the canonical 9-column `touche` contact format.
- Add pair normalization options for chromosome naming, cis-only filtering, mapq thresholding, optional replicate concatenation, and output compression.
- Add QC metrics: total pairs, cis pairs, trans pairs, mapq-pass pairs, per-chromosome counts, distance-decay histogram, duplicate/read-type summaries when fields are present.
- Add provenance metadata next to filtered outputs: command, package version, source file checksums when feasible, filter settings, and timestamp.

Design caveats:

- Do not reimplement FASTQ alignment, duplicate marking, pairtools parsing, or cooler generation in pure Python.
- Do not make distiller-nf/Nextflow a runtime dependency of `touche`.
- Avoid a `touche preprocess run-distiller` wrapper unless users later ask for workflow orchestration.
- Keep raw FASTQ processing outside the package so users with existing `.pairs.gz` files can start at `filter-pairs`.

## Phase 5: Shared IO and indexing layer

Create a shared contact ingestion layer before porting all workflows. This is the core optimization.

`touche.io` should provide:

- transparent reading of plain text and `.gz`
- chunked pair iteration
- strict and permissive parsers for reference-compatible files
- pair filtering equivalent to the documented `awk` commands
- metadata/provenance sidecar writing for preprocessed pairs

`touche.contacts` should provide a chromosome-level index:

```python
ContactIndex(
    chrom: str,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    strand_a: np.ndarray | None,
    strand_b: np.ndarray | None,
)
```

Implementation notes:

- Store positions as compact integer arrays.
- Keep one index per chromosome.
- Sort by `pos_a` and `pos_b` where useful.
- Use `np.searchsorted` and vectorized comparisons for interval queries.
- Avoid loading all contact columns into pandas when only positions are needed.

Intermediate-file policy:

- Do not materialize per-anchor or per-pair contact files by default.
- Prefer streaming, chunked transforms, chromosome-level indexes, and direct aggregate accumulation.
- Write final user-facing outputs and compact provenance/QC sidecars.
- Keep optional `--write-intermediates` or `--debug-dir` switches for troubleshooting and reference comparison.
- If an intermediate is needed for memory safety or restartability, make it coarse-grained and reusable, for example per-chromosome compact arrays rather than thousands of per-locus BED files.
- Compress large reusable intermediates when writing them, and document whether they are safe to delete.
- Add memory guardrails for chunk size, chromosome batching, and worker count so avoiding disk intermediates does not create memory explosions.

Cache format policy:

- Start with versioned `.npz` cache files for internal numeric arrays because they require no dependency beyond NumPy.
- Keep `.npz` caches sharded by chromosome, and where useful by array role, so loading a cache does not require unpacking whole-genome data.
- Treat `.npz` as a reusable cache format, not a final user-facing interchange format.
- Include a small JSON sidecar or embedded metadata array with schema version, source checksum, filter settings, chromosome, row count, coordinate convention, and `touche` version.
- Prefer uncompressed `.npy` or lightly compressed `.npz` for very hot arrays when memory mapping or load speed matters more than disk size.
- Avoid monolithic compressed `.npz` files for whole-genome contact indexes, because unpacking them can become memory intensive.
- Keep final public outputs in TSV/CSV, optionally gzip-compressed, for compatibility and inspection.
- Revisit HDF5 or Zarr if caches become too large for comfortable per-chromosome `.npz` loading, if chunked random access becomes important, or if multi-sample matrix caches need richer metadata and partial reads.

The main anti-pattern to remove is:

```python
series.isin(list(range(start, stop + 1)))
```

Replace it with vectorized bounds checks:

```python
(series >= start) & (series <= stop)
```

or array equivalents.

Model/type boundary:

- Keep `touche.models` as a thin shared dataclass module.
- Keep workflow logic, parsing, filtering, plotting, cache writing, and statistics out of `models.py`.
- Small intrinsic methods such as `ContactPair.as_tsv()` and `ContactIndex.size` are acceptable.
- If model definitions grow substantially, split by concept later, for example `records.py`, `indexes.py`, and `cli_types.py`.

## Phase 6: Local-decay workflow

Port `ContactCaller_microC.py` into small functions:

```python
fit_zero_inflation_model(...)
fit_distance_decay(...)
call_bait_contacts(...)
call_local_decay(...)
```

Preserve initially:

- LOWESS via `statsmodels.nonparametric.lowess`
- default `dist=1000000`
- default `cap=2000`
- minimum prey distance behavior
- output columns and order

Optimization plan:

- Group baits and preys by chromosome once.
- For each bait, find candidate preys with sorted prey centers and `searchsorted`.
- Query contacts around the bait from the chromosome contact index.
- Avoid per-bait gzipped temp files.
- Run parallel work by chromosome or bait chunks using `concurrent.futures.ProcessPoolExecutor`.

CLI:

```bash
touche local-decay call
touche local-decay assign-pair-types
touche local-decay plot
touche local-decay run
```

`assign-pair-types` should replace the reference nested loops with dataframe joins on:

```text
target_site.chr
target_promoter.center
target_site.center
```

## Phase 7: APA and inter-sample APA workflow

Port the APA scripts into:

```python
aggregate_apa(...)
count_anchor_signal(...)
compare_apa_change(...)
plot_apa_heatmap(...)
```

Reference behavior to preserve:

- strand-aware orientation
- 5-prime read shifting by strand, currently 75 bp
- `winSize` and `pixNum` semantics
- aggregated matrix CSV output
- bait/prey 1D signal vector output
- inter-sample observed-over-expected change heatmap

Optimization plan:

- Shift contact positions during ingestion instead of writing shifted contact files.
- Build bait/prey candidate pairs from sorted centers by chromosome.
- Use `np.histogram2d` to build per-pair or per-chunk pileups.
- Accumulate directly into one aggregate matrix.
- Count 1D anchor signal with vectorized interval queries.
- Add `--write-intermediates` for debugging, but make no-intermediate execution the default.

CLI:

```bash
touche apa aggregate \
  --pairs sample.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --out-dir APA/DMSO
```

```bash
touche apa compare \
  --control-apa DMSO/AggMat.csv \
  --treatment-apa FLV/AggMat.csv \
  --control-baits DMSO/baits_genome_wide_contacts.csv \
  --control-preys DMSO/preys_genome_wide_contacts.csv \
  --treatment-baits FLV/baits_genome_wide_contacts.csv \
  --treatment-preys FLV/preys_genome_wide_contacts.csv \
  --bait-count 10530 \
  --prey-count 27900 \
  --out FLV_over_DMSO.svg
```

Eventually add:

```bash
touche apa run
```

The `run` command should orchestrate aggregate-control, aggregate-treatment, and compare steps from an explicit manifest/config, while still writing the same intermediate tables and figures as the individual subcommands.

## Phase 8: EP/background workflow

Port `MicroC_EP_and_BG_contacts.bsh` and `single_pair_contacts_and_background_calculation.py` into:

```python
count_ep_and_background(...)
compare_background_ratios(...)
```

Reference behavior to preserve:

- anchor-pair distance filtering
- `winSize`
- `minBGdis`
- `maxBGdis`
- no intra-anchor contacts in background counts
- output columns: `chr, promoter, enhancer, EP_contacts, BG_contacts`

Optimization plan:

- Build candidate bait/prey pairs once by chromosome.
- Query EP contacts directly from chromosome contact indexes.
- Query enhancer-in-promoter-background and promoter-in-enhancer-background without materialized per-anchor files.
- Write one final TSV from the parent process.
- Explicitly handle zero background counts.

CLI:

```bash
touche background count \
  --pairs sample.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 2500 \
  --min-bg-distance 10000 \
  --max-bg-distance 150000 \
  --out EP_and_BG_contacts.tsv
```

```bash
touche background compare \
  --control DMSO/EP_and_BG_contacts.tsv \
  --treatments FLV=FLV/EP_and_BG_contacts.tsv TRP=TRP/EP_and_BG_contacts.tsv \
  --depths DMSO=53226768 FLV=362862200 TRP=410040533 \
  --min-ep-cpb 8 \
  --out-dir plots/
```

Eventually add:

```bash
touche background run
```

The `run` command should orchestrate count and compare steps, emit a manifest, and avoid hiding the individual count tables.

## Phase 9: Plotting and reports

Centralize plotting in `touche.plots`.

Keep the reference plot types:

- local-decay violin plot
- APA aggregate heatmap
- APA inter-sample change heatmap
- EP/background treatment scatterplots

Add options for:

- output format: SVG, PNG, PDF
- figure size
- colormap
- title/labels
- log scaling where appropriate

Plotting commands should also be able to write the underlying merged/statistical table, not only images.

## Phase 10: Notebook-friendly API

Add a notebook-friendly API layer without weakening the CLI-oriented compatibility layer.

Design goals:

- Keep core compute functions usable from scripts, notebooks, and CLI wrappers.
- Separate compute, plotting, and writing where practical.
- Preserve current CLI behavior and filenames.
- Avoid forcing users to write temporary files just to inspect intermediate objects.
- Return ordinary Python objects: `pandas.DataFrame`, `pandas.Series`, `numpy.ndarray`, dataclasses, and `matplotlib.figure.Figure`.
- Keep batch pipeline functions in `touche.pipelines` focused on orchestration, manifests, and output paths.

Recommended API shape:

```python
from touche import api

indexes = api.build_contact_indexes("sample.pairs.gz")
calls = api.call_local_decay(...)
fig = api.plot_pair_type_distribution(assignments)

apa = api.compute_apa(...)
fig = apa.plot()
apa.write("out/")
```

Result-object pattern:

```python
@dataclass(slots=True)
class ApaResult:
    matrix: pd.DataFrame
    bait_signal: pd.DataFrame
    prey_signal: pd.DataFrame
    metadata: dict[str, object]

    def plot(self, *, reference_style: bool = True) -> Figure: ...
    def write(self, out_dir: str | Path) -> dict[str, Path]: ...
```

Candidate result objects:

- `LocalDecayResult`
  - `calls: pd.DataFrame`
  - optional `assignments: pd.DataFrame`
  - `plot(...) -> Figure`
  - `write(...)`
- `ApaResult`
  - `matrix: pd.DataFrame`
  - `bait_signal: pd.DataFrame`
  - `prey_signal: pd.DataFrame`
  - `plot(...) -> Figure`
  - `write(...)`
- `ApaComparisonResult`
  - `matrix: pd.DataFrame`
  - `plot(...) -> Figure`
  - `write(...)`
- `BackgroundCountResult`
  - `counts: pd.DataFrame`
  - `write(...)`
- `BackgroundComparisonResult`
  - `table: pd.DataFrame`
  - `plots(...) -> dict[str, Figure]`
  - `write(...)`

Plotting refactor:

- Make plotting functions return `matplotlib.figure.Figure`.
- Make `out`/`out_path` optional.
- Save only when an output path is provided.
- Avoid unconditional `plt.close(fig)` when returning figures for interactive use.
- Keep CLI wrappers responsible for closing figures if needed after saving.
- Move reusable plotting functions toward `touche.plots` once signatures are stable.

Compute/write split:

- Keep existing path-writing functions as compatibility wrappers for now.
- Add in-memory compute helpers underneath them.
- Prefer names like:
  - `compute_apa(...)`
  - `write_apa_result(...)`
  - `plot_raw_apa_heatmap(...)`
  - `compute_background_counts(...)`
  - `plot_background_scatter(...)`
- Existing CLI-facing functions can call these helpers and return the same current values until a major API cleanup.

Index reuse:

- Allow expensive workflows to accept prebuilt contact indexes in addition to pairs paths.
- Avoid rereading pairs files when users want to compare multiple anchor sets or conditions interactively.
- Keep file-path inputs as the CLI default.

Top-level exports:

- Create `touche.api` as the stable notebook-oriented import surface.
- Re-export a small, curated set of helpers from `touche.api`.
- Do not overload `touche.__init__` with many workflow functions yet.

First implementation slice:

1. Refactor plotting functions to return `Figure` and make output paths optional:
   - `plot_pair_type_distribution`
   - `plot_raw_apa_heatmap`
   - `plot_apa_change`
   - `plot_background_scatter`
   Done.
2. Update tests to assert figures are returned and files are written when paths are provided.
   Done.
3. Keep CLI behavior unchanged.
   Done.
4. Add `docs/notebook-api.md` with minimal examples after the first API slice lands.
   Done.

Second implementation slice:

1. Add `LocalDecayResult`, `BackgroundCountResult`, `BackgroundComparisonResult`, and `ApaComparisonResult`.
2. Add `.write()` helpers for each result object.
3. Add `.plot()` or `.plots()` helpers where applicable.
4. Keep existing dataframe-returning functions as compatibility wrappers until a deliberate API-breaking release.
5. Expand `docs/notebook-api.md` with result-object examples once this lands.

Acceptance criteria:

- Existing CLI tests continue to pass.
- Notebook users can generate plots without creating files.
- Returned figures can be displayed or saved by the caller.
- Plot-data tables remain accessible as dataframes.
- The API surface is documented as provisional until result objects are added.

## Phase 11: Reference figure reproduction

Add an explicit reproduction track for the figures demonstrated in the reference checkout. This should be treated as a high-level acceptance test for the refactor: `touche` should be able to start from the same processed pairs and input anchor files, generate equivalent intermediate tables, and render equivalent figures.

Reference figure targets:

- Local-decay normalized contacts by pair type:
  - Reference command: `Contact_normalization_by_local_decay/Plotting_obs_over_exp_distribution_by_pair_type.py`
  - Output: `Violinplot_for_normalized_contacts_by_pair_type.svg`
  - `touche` target: `touche local-decay plot`
- Raw APA aggregate heatmap:
  - Reference command: `APA_and_inter-sample_APA/MicroC_Stranded_Aggregation_pipeline_get_aggregated_matrix.py`
  - Output: `AggMat.csv` and `AggHeatmap.svg`
  - `touche` target: `touche apa aggregate`
- Inter-sample APA change heatmaps:
  - Reference command: `APA_and_inter-sample_APA/Change_calculation_and_visualization.py`
  - Outputs: `FLV_over_DMSO_1D_normalized_change_APA.svg`, `TRP_over_DMSO_1D_normalized_change_APA.svg`
  - `touche` target: `touche apa compare`
- EP/background treatment scatterplots:
  - Reference command: `EP_contacts_compared_to_local_background/Compering_EP_contacts_between_treatments.py`
  - Outputs: `FLV_vs_DMSO.svg`, `TRP_vs_DMSO.svg`, `FLV_vs_TRP.svg`
  - `touche` target: `touche background compare`

Implementation steps:

1. Create `docs/reproducing-reference-plots.md` with one section per figure family. Done.
2. Document required external data files, including which processed pairs files must be downloaded from GEO/FTP and which anchor/label files are already present under `_reference/E-P_contacts/Input_files`.
3. Provide a `touche` command translation for every reference command shown in `_reference/E-P_contacts/README.md`.
4. Add a `--reference-style` plotting option where useful to match figure size, palette, axis labels, heatmap colormaps, log scaling, and filtering thresholds used by the original scripts.
5. Ensure every plotting command can also write the exact table used for plotting, for example merged treatment ratios or obs/exp values after filtering.
6. Add small synthetic reproduction fixtures that exercise the full table-to-figure path without requiring large Micro-C files.
7. Add optional full-data reproduction scripts under `scripts/reproduce_reference_figures/` that are not run in normal tests.

Acceptance criteria:

- For small fixtures, generated count tables and matrices match reference-script outputs exactly where values are integer counts.
- For LOWESS-derived local-decay expected values, outputs match within a documented floating tolerance.
- For rendered figures, tests should compare plot data tables and basic SVG properties rather than brittle byte-for-byte SVG equality.
- For full data, maintain a manifest with command, input checksums where feasible, package version, wall time, peak memory, and output paths.
- If visual styling intentionally diverges from the reference, require a non-default style option and keep `--reference-style` as the compatibility path.

## Phase 12: Packaging and dependency policy

Initial `pyproject.toml`:

```toml
[project]
name = "touche"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pandas",
  "scipy",
  "statsmodels",
  "matplotlib",
  "seaborn",
]

[project.scripts]
touche = "touche.cli:main"
```

Optional extras:

```toml
[project.optional-dependencies]
fast = ["numba"]
io = ["polars", "pyarrow"]
```

Development dependencies are managed as a uv dependency group:

```toml
[dependency-groups]
dev = ["pytest", "ruff", "mypy"]
```

`__version__` should be read from package metadata with `importlib.metadata.version("ep-touche")`, so `pyproject.toml` remains the single source of truth for package versioning. The PyPI distribution name is `ep-touche`; the import package and CLI command remain `touche`.

Keep the initial skeleton correct and NumPy-based, but plan to add Numba kernels soon after the main CLI/API surface is in place. Numba is a reasonable optional acceleration dependency because the hot paths are pure counting kernels over numeric arrays. Avoid adding `polars`, `pyarrow`, or workflow-engine dependencies until there is a measured bottleneck or operational need that justifies them. Do not add `pyyaml` unless `touche` later starts parsing or rendering workflow configs.

**Update:** the measured bottleneck arrived — a benchmark showed the hand-rolled
pairs-file parser was 10-50x slower than a `polars`-based lazy reader. `touche`
has since migrated fully from `pandas` to `polars` as its only dataframe
library (internal and at the `touche.api` public surface); `pandas` is no
longer a dependency and `pyarrow` was never needed. See `CLAUDE.md` for the
current architecture.

Build backend:

- Use Hatchling for packaging.
- Keep `uv` as the development, lockfile, and build frontend.
- Publish `ep-touche` to PyPI with trusted publishing.

## Phase 13: Post-skeleton Numba kernels

After the main package skeleton and reference-compatible workflows are implemented, add optional Numba implementations for the counting-heavy kernels. See `notes/numba-implementation-plan.md` for the detailed staged plan.

Candidate kernels:

- EP/background contact counting for many bait/prey pairs.
- APA pileup matrix accumulation from contact arrays and oriented anchor pairs.
- 1D anchor signal vector counting around baits and preys.
- Local-decay observed-count and distance-histogram pieces, while keeping LOWESS and Fisher exact tests in Python.

Design:

- Keep pure NumPy implementations as the default correctness path.
- Add a backend switch where useful, starting with `backend="numpy"` and `backend="numba"` in in-memory compute functions and matching CLI `--backend` options.
- Put Numba-specific code behind optional imports so base installation remains simple.
- Use the same tests for NumPy and Numba backends, requiring exact equality for integer counts.
- Ensure accelerated kernels operate on compact arrays or bounded chunks rather than requiring whole-genome in-memory materialization.
- Benchmark on chromosome-scale data before changing defaults.
- Start with EP/background counts because the integer counting surface is the smallest and easiest to validate exactly, then add APA matrix/signal kernels, then local-decay count helpers.

## Phase 14: Validation and benchmarking

Validation levels:

1. Unit tests on synthetic data.
2. Golden-output tests against the reference scripts on tiny fixtures.
3. One chromosome-scale benchmark.
4. Full workflow benchmark when the real Micro-C pairs files are available.
5. Preprocessing tests on tiny synthetic pairs and format-conversion fixtures.

Track:

- wall time
- peak RSS memory
- per-step CLI profile timings from `--profile`
- temporary disk usage
- number, size, and reusability of intermediate files
- output equality or numerical tolerance
- number of contacts processed per second
- preprocessing throughput and gzip compression ratio

Current benchmark artifacts:

- `notes/benchmarks/benchmark_numba_kernels.py`: synthetic microbenchmarks for
  counting kernels and the LOWESS/Fisher backends.
- `scripts/reference_replication.py`: real reference-data benchmark pipeline
  that downloads the upstream example pairs and input files, then profiles
  preprocessing, local-decay, APA, and EP/background CLI steps with wall time
  and sampled peak RSS, rendering the same reference-comparable plots as the
  upstream README. Promoted out of `notes/benchmarks/` since it's a runnable
  replication tool, not just a benchmark log.
- `scripts/reference_replication.md`: usage notes for the real-data
  replication runner.

Suggested tolerance policy:

- exact equality for raw contact counts
- exact equality for candidate-pair lists
- exact equality between NumPy and Numba counting backends
- small floating tolerance for expected values and LOWESS-derived quantities
- p-value tolerance after matching rounding behavior
- exact equality for `filter-pairs` output against the documented `awk` filter on fixtures

## Recommended milestone order

1. Scaffold the `touche` package and CLI. Done.
2. Add `touche preprocess filter-pairs` and `touche preprocess qc`. Done.
3. Add `touche preprocess convert-pairs`. Done.
4. Add shared parsers, pair filtering, and chromosome contact indexes. Partially done.
5. Port local-decay Fisher tests and remove `rpy2`. Done for `touche.stats.fisher_greater`.
6. Implement `touche local-decay assign-pair-types`. Done.
7. Implement `touche local-decay plot`. Done.
8. Implement `touche background compare`. Done.
9. Implement `touche apa compare`. Done.
10. Refactor CLI into `src/touche/cli/` package. Done.
11. Document distiller-nf setup and required output options. Done.
12. Implement `touche local-decay call`. Done.
13. Implement `touche background count`. Done.
14. Implement `touche apa aggregate`. Done.
15. Add full pipeline `run` wrappers. Done.
16. Add reference plot reproduction documentation. Done.
17. Refactor plotting/results APIs for notebook-friendly use. First slice done.
18. Add provisional `touche.api` exports and notebook API docs. Done.
19. Add richer notebook result objects for remaining workflows.
20. Add post-skeleton Numba kernels for the hottest counting paths.
21. Add plotting polish, docs, and benchmark notes. In progress.
22. Run the real-data benchmark pipeline after explicitly approving the large
    downloads and expected runtime.

The best first working slice is:

```text
package scaffold -> preprocess filter-pairs -> preprocess convert-pairs -> preprocess qc -> local-decay call -> assign-pair-types
```

That slice proves the package/CLI shape, makes Micro-C pairs preparation reproducible, removes the R dependency, and exercises the most delicate statistical behavior before tackling the heavier APA and background optimizations.
