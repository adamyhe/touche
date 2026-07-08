# Reference real-data benchmark

This note documents the benchmark runner in
`notes/benchmarks/benchmark_reference_real_data.py`. It is intentionally
agent-facing and should not be part of normal tests or user-facing package docs.

The runner downloads the real example data named by the upstream
`Danko-Lab/E-P_contacts` README, then profiles `touche` subprocesses with wall
time and sampled peak RSS.

## Inputs

Large pairs files:

- `GSE206131_K562_cis_mapq30_pairs.txt.gz` from GEO.
- `mESCs_DMSO_30_intra.mm10.nodups.pairs.gz` from the Cornell FTP directory.
- `mESCs_FLV_30_intra.mm10.nodups.pairs.gz` from the Cornell FTP directory.
- `mESCs_TRP_30_intra.mm10.nodups.pairs.gz` from the Cornell FTP directory.

Small reference files:

- `Gasperini_dREG_based_TRE_baits_hg38.txt`
- `Gasperini_dREG_based_promoter_preys_hg38.txt`
- `Gasperini_dREG_based_functional.csv`
- `Gasperini_dREG_based_nonfunctional.csv`
- `dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed`
- `dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed`

The small files are downloaded from GitHub raw URLs so the benchmark can run
without relying on a local `_reference/` checkout.

## Planned steps

Preprocessing:

- `preprocess-qc-k562`
- `preprocess-cache-k562`
- `preprocess-qc-dmso`
- `preprocess-cache-dmso`
- `preprocess-qc-flv`
- `preprocess-cache-flv`
- `preprocess-qc-trp`
- `preprocess-cache-trp`

Local decay:

- `local-decay-call`
- `local-decay-assign-pair-types`
- `local-decay-plot`

APA:

- `apa-aggregate-dmso`
- `apa-aggregate-flv`
- `apa-aggregate-trp`
- `apa-compare-flv-vs-dmso`
- `apa-compare-trp-vs-dmso`

EP/background:

- `background-count-dmso`
- `background-count-flv`
- `background-count-trp`
- `background-compare`

## Usage

List downloads, commands, and output paths without downloading or running:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py --dry-run
```

Download only:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py --download-only
```

Run the full NumPy/statsmodels benchmark:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py
```

Run with optional accelerated kernels:

```bash
uv run --extra fast python notes/benchmarks/benchmark_reference_real_data.py \
  --backend numba \
  --lowess-backend numba
```

Run a subset:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py \
  --skip-download \
  --steps apa-aggregate-dmso apa-aggregate-flv apa-compare-flv-vs-dmso
```

Use `--keep-going` to continue after a failed step, and `--progress` to pass
`--progress` into the profiled `touche` commands.

Regenerate the human-readable report from an existing JSONL result file without
rerunning any benchmark steps:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py --plot-only
```

Use `--report-dir path/to/report` to place the report somewhere other than the
default work directory, or `--no-report` to skip report generation.

## Outputs

Default work directory:

```text
benchmark/reference-real-data/
```

The runner writes:

- `data/`: downloaded pairs and input files.
- `outputs/`: command outputs.
- `logs/*.stdout` and `logs/*.stderr`: per-step subprocess output.
- `benchmark-results.jsonl`: one result object per profiled step.
- `benchmark-manifest.json`: download records, checksums, command settings, and
  all step results.
- `report/`: human-readable summaries and plots.

Each result records:

- command arguments
- return code
- wall time
- sampled peak RSS in MiB
- expected output sizes
- parsed CLI JSON when stdout contains JSON

The report directory contains:

- `summary.md`: a Markdown table of wall time, sampled peak RSS, output size,
  return status, and rows where available.
- `summary.csv`: the same summary in a spreadsheet-friendly format.
- `command-timings.csv`: nested timings emitted by CLI `--profile` when present.
- `index.html`: a simple browser-readable report page.
- `wall-time.svg`: per-step wall-time bar chart.
- `peak-rss.svg`: per-step peak-memory bar chart.
- `output-size.svg`: per-step output-size bar chart.
- `command-timings.svg`: nested CLI timing chart when command timings exist.

## Notes

- Peak RSS is sampled with `ps` while each subprocess runs. This avoids adding a
  profiling dependency, but it is still an estimate and depends on poll interval.
- The default poll interval is `0.25` seconds. Lower values can catch sharper
  peaks at the cost of slightly more profiling overhead.
- The benchmark invokes `python -m touche` in a fresh subprocess per step. This
  includes Python import and CLI startup overhead, which is useful for end-user
  CLI performance but should be interpreted separately from in-process API
  microbenchmarks.
