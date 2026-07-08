# Reference real-data benchmark

This benchmark is agent-facing and is not run by normal tests. It downloads the
real example data named by the upstream `Danko-Lab/E-P_contacts` README and
profiles the full `touche` workflow on those inputs.

## What It Measures

The default run profiles these full-data steps:

- `preprocess-qc-k562`
- `preprocess-cache-k562`
- `preprocess-qc-dmso`
- `preprocess-cache-dmso`
- `preprocess-qc-flv`
- `preprocess-cache-flv`
- `preprocess-qc-trp`
- `preprocess-cache-trp`
- `local-decay-call`
- `local-decay-assign-pair-types`
- `local-decay-plot`
- `apa-aggregate-dmso`
- `apa-aggregate-flv`
- `apa-aggregate-trp`
- `apa-compare-flv-vs-dmso`
- `apa-compare-trp-vs-dmso`
- `background-count-dmso`
- `background-count-flv`
- `background-count-trp`
- `background-compare`

`local-decay-call` uses the K562 chromosome-sharded NPZ cache created by
`preprocess-cache-k562`. The benchmark passes `--require-cache`, so cache
construction is measured only in `preprocess-cache-k562` and is not hidden
inside the timed local-decay call.

## Inputs

Large pairs files:

- `GSE206131_K562_cis_mapq30_pairs.txt.gz` from GEO.
- `mESCs_DMSO_30_intra.mm10.nodups.pairs.gz` from the Cornell FTP directory.
- `mESCs_FLV_30_intra.mm10.nodups.pairs.gz` from the Cornell FTP directory.
- `mESCs_TRP_30_intra.mm10.nodups.pairs.gz` from the Cornell FTP directory.

Small input files are downloaded from GitHub raw URLs:

- `Gasperini_dREG_based_TRE_baits_hg38.txt`
- `Gasperini_dREG_based_promoter_preys_hg38.txt`
- `Gasperini_dREG_based_functional.csv`
- `Gasperini_dREG_based_nonfunctional.csv`
- `dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed`
- `dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed`

## How To Run

Preview the full benchmark without downloading or running anything:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py --dry-run
```

Download inputs only:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py --download-only
```

Run the full benchmark:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py \
  --skip-download \
  --fail-on-missing-output
```

Run the full benchmark with optional accelerated kernels:

```bash
uv run --extra fast python notes/benchmarks/benchmark_reference_real_data.py \
  --skip-download \
  --backend numba \
  --lowess-backend numba \
  --fail-on-missing-output
```

Regenerate plots and tables from an existing result file:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py --plot-only
```

## Outputs

Default work directory:

```text
benchmark/reference-real-data/
```

The runner writes:

- `data/`: downloaded input files.
- `outputs/`: command outputs and NPZ caches.
- `logs/*.stdout` and `logs/*.stderr`: per-step subprocess output.
- `benchmark-results.jsonl`: one profiled result per step.
- `benchmark-manifest.json`: downloads, checksums, settings, and all step results.
- `report/summary.md`: human-readable result table.
- `report/summary.csv`: spreadsheet-friendly result table.
- `report/index.html`: browser-readable report page.
- `report/wall-time.svg`: per-step wall-time chart.
- `report/peak-rss.svg`: per-step peak-memory chart.
- `report/output-size.svg`: per-step output-size chart.
- `report/command-timings.*`: nested CLI `--profile` timings when available.

Each result records command arguments, return code, signal name for negative
return codes such as `SIGKILL`, wall time, sampled peak RSS, expected output
sizes, and parsed CLI JSON when stdout contains JSON.

## Interpreting Failures

Return code `-9` is `SIGKILL`. On local machines and schedulers this usually
means the subprocess exceeded available memory or a job limit.

If `local-decay-call` fails before running, check that `preprocess-cache-k562`
completed and wrote:

```text
benchmark/reference-real-data/outputs/caches/k562/k562.manifest.json
```

The full benchmark includes `preprocess-cache-k562` before `local-decay-call`.
If you use `--steps` for debugging, include dependent steps yourself.

## Optional Debugging

`--steps` exists for rerunning or debugging full benchmark stages, not for
defining the main benchmark. For example, to rerun just cache construction and
cache-backed local-decay after inputs are already downloaded:

```bash
uv run python notes/benchmarks/benchmark_reference_real_data.py \
  --skip-download \
  --steps preprocess-cache-k562 local-decay-call \
  --fail-on-missing-output
```

Use `--keep-going` to continue after failed steps, `--progress` to pass
`--progress` to profiled `touche` commands, `--report-dir` to change report
location, and `--no-report` to skip report generation.

Peak RSS is sampled with `ps` while each subprocess runs. The default poll
interval is `0.25` seconds.
