# Reference real-data benchmark

This benchmark is agent-facing and is not run by normal tests. It downloads the
real example data named by the upstream `Danko-Lab/E-P_contacts` README and
profiles the full `touche` workflow on those inputs, generating the same
reference-comparable plots along the way.

## What It Measures

The benchmark profiles the `preprocess-cache-*` steps once, then every
downstream step once. `--lowess-backend` and `--fisher-backend` remain real
choices for `local-decay-call` (default `numba` for both; pass
`statsmodels`/`scipy` for the conservative reference-comparison path).

Full step list:

- `preprocess-cache-k562`, `preprocess-cache-dmso`, `preprocess-cache-flv`,
  `preprocess-cache-trp`
- `local-decay-call`, `local-decay-assign-pair-types`, `local-decay-plot`
- `apa-aggregate-dmso`, `apa-aggregate-flv`, `apa-aggregate-trp`
- `apa-compare-flv-vs-dmso`, `apa-compare-trp-vs-dmso`
- `background-count-dmso`, `background-count-flv`, `background-count-trp`
- `background-compare`

`local-decay-call` uses the K562 chromosome-sharded NPZ cache created by
`preprocess-cache-k562`. The benchmark passes `--require-cache`, so cache
construction is measured only in `preprocess-cache-k562` and is not hidden
inside the timed local-decay call.

Each `preprocess-cache-*` step also writes the default sample QC JSON beside the
cache manifest, so preprocessing scans each compressed pairs file once instead
of running separate QC and cache passes.

The K562 cache is built with `--no-metadata` because the benchmark consumes it
only through `local-decay-call`, which loads position arrays and ignores strand
and MAPQ cache arrays.

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
uv run python scripts/reference_replication.py --dry-run
```

Download inputs only:

```bash
uv run python scripts/reference_replication.py --download-only
```

Run the full benchmark:

```bash
uv run python scripts/reference_replication.py \
  --skip-download \
  --progress \
  --fail-on-missing-output
```

Run the exact, statsmodels/scipy-backed reference-parity path instead (requires
the optional `legacy` extra -- `uv sync --extra legacy`):

```bash
uv run python scripts/reference_replication.py \
  --skip-download \
  --lowess-backend statsmodels \
  --fisher-backend scipy \
  --progress \
  --fail-on-missing-output
```

Regenerate plots and tables from an existing result file:

```bash
uv run python scripts/reference_replication.py --plot-only
```

## Outputs

Default work directory:

```text
benchmark/reference-real-data/
```

The runner writes:

- `data/`: downloaded input files.
- `outputs/`: command outputs and NPZ caches, including the shared caches
  under `outputs/caches/`.
- `logs/*.stdout` and `logs/*.stderr`: per-step subprocess output.
- `benchmark-results.jsonl`: one profiled result per step.
- `benchmark-manifest.json`: downloads, checksums, settings, and all step results.
- `report/summary.md`: human-readable result table.
- `report/summary.csv`: spreadsheet-friendly table.
- `report/index.html`: browser-readable report page, including a **Generated
  Plots** gallery embedding every SVG a step produced (APA heatmaps, the
  local-decay violin plot, background scatter plots) grouped by workflow --
  open this to visually compare touche's output against the published
  Danko-lab reference figures.
- `report/wall-time.svg`, `report/peak-rss.svg`, `report/cpu-percent.svg`,
  `report/output-size.svg`: per-step charts.
- `report/command-timings.*`: nested CLI `--profile` timings when available.

Each result records command arguments, return code, signal name for negative
return codes such as `SIGKILL`, wall time, sampled peak RSS, CPU time/percent
(via `resource.getrusage(RUSAGE_CHILDREN)` deltas), expected output sizes, and
parsed CLI JSON when stdout contains JSON.

## Interpreting Failures

Return code `-9` is `SIGKILL`. On local machines and schedulers this usually
means the subprocess exceeded available memory or a job limit.

Downloads retry automatically on HTTP 429/500/502/503/504 and on connection
errors, with exponential backoff (honoring a `Retry-After` response header
when present). This is common on shared HPC egress IPs hitting GitHub's or
NCBI's per-IP rate limits. Retries are logged to stderr as `[download] ...
retrying in Ns (attempt X/Y)`. Tune with `--download-retries` (default 6,
use `0` to disable) and `--download-retry-backoff` (default 2.0s base
delay). If you still see a persistent `HTTP Error 429` after retries are
exhausted, wait a bit and re-run -- already-downloaded files are skipped via
the existing-file check, so re-running only fetches what's missing -- or pass
a higher `--download-retries`/`--download-retry-backoff` if run alongside
other jobs contending for the same node's egress IP.

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
uv run python scripts/reference_replication.py \
  --skip-download \
  --steps preprocess-cache-k562 local-decay-call \
  --fail-on-missing-output
```

Use `--keep-going` to continue after failed steps. Use `--progress` to pass
`--progress` to profiled `touche` commands and stream their stderr progress bars
live in the terminal while still saving per-step stderr logs. Use `--report-dir`
to change report location, and `--no-report` to skip report generation.

Peak RSS is sampled with `ps` while each subprocess runs; CPU time/percent is
computed from `resource.getrusage(RUSAGE_CHILDREN)` deltas around each
subprocess. The default RSS poll interval is `0.25` seconds.
