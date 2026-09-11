# CLI reference

The `touche` command is installed by the `ep-touche` Python package:

```bash
pip install ep-touche
touche --help
```

or

```bash
uv add ep-touche
uv run touche --help
```

The examples below use `uv run touche` so they work from a checkout. If the
package is installed with `pip`, drop the `uv run` prefix.

### Older CPUs and Rosetta

If the standard Polars wheel does not run on your CPU, use Polars' LTS CPU
build. It provides the same `import polars` module, but it is distributed as
`polars-lts-cpu`, so install `ep-touche` first and then replace the default
Polars wheel in that environment:

```bash
pip install ep-touche
pip uninstall polars
pip install "polars-lts-cpu>=1.0"
touche --help
```

## Command groups

```bash
uv run touche preprocess --help
uv run touche pairs --help
uv run touche local-decay --help
uv run touche background --help
uv run touche apa --help
```

Use `--help` on any subcommand to see all options and defaults:

```bash
uv run touche local-decay call --help
```

Most commands accept `--source auto`, `--source distiller`, or `--source touche`
when reading pairs files. `auto` is the default and infers the input shape from
the row fields. Use `distiller` for pairtools/distiller-style pairs and `touche`
for the canonical 9-column format written by `touche preprocess`.

## Main workflow

For routine analyses, prefer the `run` wrappers. They run the expected stages,
preserve intermediate tables and figures, and write a `manifest.json` with
inputs, parameters, outputs, metrics, and timings when `--profile` is enabled.

Typical command order:

```bash
# Optional, but useful when starting from distiller/pairtools pairs.
uv run touche preprocess filter-pairs \
  --pairs sample.pairs.gz \
  --out sample.nodups_30_intra.pairs.gz \
  --min-mapq 30 \
  --cis-only

# Recommended before real or repeated local-decay runs.
uv run touche preprocess build-cache \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/sample \
  --prefix sample \
  --no-metadata

uv run touche local-decay run ...
uv run touche apa run ...
uv run touche background run ...
```

Use individual commands such as `local-decay call`, `apa aggregate`, or
`background count` when debugging, benchmarking, or replacing one stage of a
larger workflow.

### Statistics and interoperability commands

These are opt-in additions; none of them change what the commands above
produce by default. Each is documented in full, with its null hypothesis,
unit of replication, and FDR family, in the
[statistics guide](statistics.md).

```bash
# Materialize the bait/prey pair universe as a BEDPE pair list
uv run touche pairs build ...
# Import FitHiC2 / MaxHiC / HiC-DC+ / Mustache / Peakachu / Chromosight /
# HiCCUPS calls, and attach them to a pair list by anchor overlap
uv run touche pairs import-calls ...
uv run touche pairs annotate ...

# Calibrated per-pair significance with BH q-values, and a calibration check
uv run touche local-decay call --method binomial --schema tidy ...
uv run touche local-decay test ...
uv run touche local-decay calibration ...
# Effect size, rank test, and clustered bootstrap interval between groups
uv run touche local-decay compare-groups ...

# Quantitative APA mask scores
uv run touche apa aggregate --summary-out ... --bootstrap 1000 ...
uv run touche apa summarize ...

# Zero-safe EP-versus-background comparison between two libraries
uv run touche background diff ...
```

Every statistical command writes a `.meta.json` sidecar next to its table
recording the method, hypothesis, null, test universe, FDR family, unit of
analysis, clustering unit, seed, and any assumption warnings.

## Pairs

`touche pairs` manages the explicit bait/prey pair list that `apa` and
`background` can analyze in place of the implicit "every bait crossed with
every prey inside a distance window" universe.

```bash
uv run touche pairs build \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 20000 \
  --max-distance 1000000 \
  --out pairs.bedpe
```

The written BEDPE carries a stable `pair_id` in its name column. Filter,
subset, or annotate it, then pass it back with `--pairs-list`:

```bash
uv run touche background count --pairs sample.pairs.gz --pairs-list pairs.bedpe ...
uv run touche apa aggregate     --pairs sample.pairs.gz --pairs-list pairs.bedpe ...
```

The list is the pair universe exactly as given. The distance window does not
prune it, and it is never expanded back to a product.

### Import and attach external calls

```bash
uv run touche pairs import-calls \
  --input fithic2.significances.txt \
  --format fithic2 \
  --out calls.tsv \
  --bedpe-out calls.bedpe

uv run touche pairs annotate \
  --pairs pairs.bedpe \
  --calls calls.bedpe \
  --format bedpe \
  --slop 2000 \
  --out annotated.tsv
```

Supported `--format` values: `fithic2`, `hicdcplus`, `maxhic`, `mustache`,
`chromosight`, `peakachu`, `hiccups`, `bedpe`. Use `--column FIELD=COLUMN`
(repeatable) when a tool's header differs from the built-in mapping; a
mismatch fails with the header it actually found rather than mapping the
wrong column. Chromosome naming is not translated, so harmonize `chr1`
versus `1` before annotating.

## CPU usage

Compute-heavy commands use Numba kernels. Control their core budget with
`NUMBA_NUM_THREADS` before running `touche`:

```bash
NUMBA_NUM_THREADS=8 uv run touche background run ...
NUMBA_NUM_THREADS=16 uv run touche apa aggregate ...
```

If `NUMBA_NUM_THREADS` is unset, Numba chooses its default thread count for the
machine. On shared servers and schedulers, set it explicitly to the cores
allocated to the job. For `local-decay`, keep `--jobs 1` when Numba is already
using the available cores; increasing `--jobs` only helps if profiling shows
substantial idle CPU outside the kernels (though setting `--jobs n_cpus` is
generally safe).

## Progress and Profiling

Long-running counting commands accept:

- `--progress`: show `tqdm` progress bars on stderr.
- `--profile`: collect lightweight step timings.

The JSON summaries still print to stdout, so they can be redirected or parsed
while progress bars remain separate. For full `run` workflows, `--profile`
also writes the timing records into `manifest.json`.

Examples:

```bash
uv run touche local-decay call ... --progress --profile
uv run touche background run ... --progress --profile
uv run touche apa aggregate ... --progress --profile
```

Progress bars are currently available for `local-decay call`, `local-decay run`,
`background count`, `background run`, `apa aggregate`, and `apa run`.

## Preprocess

`touche preprocess` prepares pairs files for downstream analysis. Raw FASTQ
alignment, duplicate handling, and cooler generation should happen upstream with
[distiller-nf](https://github.com/open2c/distiller-nf) or an equivalent
pairs-producing workflow.

### Filter pairs

Use `filter-pairs` to convert a distiller/pairtools-style pairs file into the
canonical `touche` format while applying mapq and cis/trans filters:

```bash
uv run touche preprocess filter-pairs \
  --pairs sample.pairs.gz \
  --out sample.nodups_30_intra.pairs.gz \
  --min-mapq 30 \
  --cis-only
```

By default this writes cis pairs where both sides pass `--min-mapq`. Add
`--no-cis-only` to keep trans pairs, or `--keep-read-id` to retain read IDs when
available.

### Convert pairs

Use `convert-pairs` when the input is already filtered and only needs format
conversion:

```bash
uv run touche preprocess convert-pairs \
  --pairs sample.pairs.gz \
  --from distiller \
  --to touche \
  --out sample.touche.pairs.gz
```

The current output target is `touche`.

### QC and summary

Write a QC JSON file:

```bash
uv run touche preprocess qc \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche \
  --out sample.qc.json
```

Print the same style of summary JSON to stdout:

```bash
uv run touche preprocess summarize \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche
```

The summary includes parsed rows, written rows where relevant, cis/trans counts,
mapq pass/fail counts, per-chromosome counts, and a coarse cis-distance
histogram.

For large files that will also be cached, use `build-cache`; it writes QC during
cache construction by default so the compressed pairs file is scanned once.

### Build NPZ caches

Build chromosome-sharded NPZ caches for real or repeated local-decay analysis:

```bash
uv run touche preprocess build-cache \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/sample \
  --prefix sample
```

Add `--compressed` to write compressed NPZ shards. The default is uncompressed
NPZ, which is usually faster to load and avoids turning one cache into a large
monolithic archive.

The default cache builder emits chromosome-sharded NPZ files and
`.cache/touche/sample/sample.qc.json` from one streaming pass over the input. Use
`--qc-out PATH` to choose another QC location, or `--no-qc` to suppress QC
output. Use `--no-metadata` for faster, smaller position-only caches when the
downstream command does not need strand or MAPQ arrays.

## Local Decay

`touche local-decay` calls bait-prey contacts normalized by local distance decay,
then can assign functional/nonfunctional labels and plot observed/expected
contact distributions.

Inputs:

- `--baits`: BED-like bait anchors.
- `--preys`: BED-like prey anchors.
- `--pairs`: analysis-ready pairs file.
- `--functional`: two-column or BED-like list of functional bait/prey pairs for
  assignment.
- `--nonfunctional`: matching list of nonfunctional pairs.

### Run the full local decay workflow

Use `local-decay run` when you want contact calling, assignment, plotting, and a
manifest in one step:

```bash
uv run touche local-decay run \
  --baits baits.bed \
  --preys preys.bed \
  --pairs sample.nodups_30_intra.pairs.gz \
  --functional functional_pairs.tsv \
  --nonfunctional nonfunctional_pairs.tsv \
  --out-dir results/local-decay \
  --source touche
```

The run wrapper writes:

- `ContactCaller_microC_output.tsv`
- `ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv`
- `Violinplot_for_normalized_contacts_by_pair_type.tsv`
- `Violinplot_for_normalized_contacts_by_pair_type.svg`
- `manifest.json`

Useful tuning options include `--dist`, `--cap`, `--min-distance`,
`--lowess-window`, `--lowess-delta`, `--plot-min-contacts`, and
`--plot-min-distance`.

### Run individual local-decay steps

Call contacts:

```bash
uv run touche local-decay call \
  --baits baits.bed \
  --preys preys.bed \
  --pairs sample.nodups_30_intra.pairs.gz \
  --out results/local-decay/ContactCaller_microC_output.tsv \
  --source touche
```

Assign pair types:

```bash
uv run touche local-decay assign-pair-types \
  --contacts results/local-decay/ContactCaller_microC_output.tsv \
  --functional functional_pairs.tsv \
  --nonfunctional nonfunctional_pairs.tsv \
  --out results/local-decay/pair_assignments.tsv
```

Plot assigned pairs:

```bash
uv run touche local-decay plot \
  --assignments results/local-decay/pair_assignments.tsv \
  --out results/local-decay/pair_type_distribution.svg \
  --plot-table-out results/local-decay/pair_type_distribution.tsv
```

Use `--no-reference-style` on `run` or `plot` to use the package's standard
Matplotlib styling instead of the reference-style plot appearance.

`--lowess-backend` and `--fisher-backend` control the local-decay smoothing and
Fisher exact-test implementations. Both default to `numba`; use
`--lowess-backend statsmodels` and `--fisher-backend scipy` for the most
conservative reference-comparison path.

`local-decay call` and `local-decay run` default to `--index-strategy cache`.
For real data, treat this as the normal path: build the cache once, then reuse
it for parameter sweeps, reruns, and benchmarks. If `--cache-dir` is omitted,
the cache is created under `contact_index_cache/` next to the local-decay output
table.

For repeated local-decay runs, build the cache explicitly and reuse it:

```bash
uv run touche preprocess build-cache \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/sample \
  --prefix sample \
  --no-metadata

uv run touche local-decay call \
  --baits baits.bed \
  --preys preys.bed \
  --pairs sample.nodups_30_intra.pairs.gz \
  --out results/local-decay/ContactCaller_microC_output.tsv \
  --index-strategy cache \
  --cache-dir .cache/touche/sample \
  --cache-prefix sample \
  --require-cache
```

Use `--require-cache` when you want reproducible timing or workflow behavior:
the command fails if the cache is missing instead of building one implicitly.

Alternative strategies are kept for small data and diagnostics:

- `--index-strategy all`: read the pairs file once and hold every chromosome in
  memory. This is useful for tests, notebooks, and small inputs.
- `--index-strategy chromosome`: scan the pairs file once per bait chromosome
  and keep only that chromosome in memory. This avoids persistent cache files,
  but can be slow for gzipped real-data pairs.

`--lowess-backend statsmodels` requires the optional `legacy` extra
(`pip install ep-touche[legacy]` / `uv sync --extra legacy`). Use
`--lowess-iterations` to change the number of robust residual reweighting
passes; lower values are faster but can change expected-contact estimates.

`--jobs`/`-j` (default 1) processes that many baits concurrently. Leave it at
1 when `NUMBA_NUM_THREADS` already saturates the available cores. Try a higher
value only after checking `--profile` timings and CPU utilization.

### Calibrated significance and q-values

`local-decay call` defaults to `--method legacy_fisher --schema legacy`,
which writes the reference nine-column headerless TSV with the reference
workflow's numbers. Opt in to the calibrated path with:

```bash
uv run touche local-decay call \
  --baits baits.tsv --preys preys.tsv --pairs sample.pairs.gz \
  --out results/calls.tsv \
  --method binomial \
  --schema tidy \
  --fdr bh
```

`--schema tidy` writes a headed table on the canonical pair schema with
`pair_id`, `n_trials`, `p_null`, `log2_oe`, and `q_value`, plus a
`.meta.json` sidecar. `--method binomial` tests the observed count against
the model's own trial total and null probability instead of the legacy
Fisher table, which is retained for reproducibility but is not a calibrated
test. `--fdr-scope COLUMN` (repeatable) declares a stratified FDR family
instead of one global family.

Retest an existing tidy table without recounting contacts, and check whether
its p-values are actually uniform under the null:

```bash
uv run touche local-decay test --calls results/calls.tsv --out retested.tsv --method poisson
uv run touche local-decay calibration --calls null_pairs.tsv --strata chrom
```

Run `calibration` on a null set — distance- and coverage-matched shifted or
random pairs — not on real enhancer-promoter pairs.

### Compare groups of pairs

```bash
uv run touche local-decay compare-groups \
  --table results/assignments.tsv \
  --value-col log2_oe \
  --group-col PosNeg \
  --cluster bait_id \
  --bootstrap 1000 \
  --out comparison.tsv
```

Reports both group medians, their difference with a bootstrap confidence
interval, Cliff's delta, and a Mann-Whitney U p-value. Pass `--cluster` with
the column pairs actually share (a promoter or enhancer id): without it the
interval treats every pair as independent evidence and will be too narrow.
`--correlate-with COLUMN` switches to a Spearman or Pearson correlation
instead.

## Background

`touche background` counts enhancer-promoter contacts and local background
contacts, then compares normalized EP/background ratios across samples.

Inputs:

- `--baits`: BED-like promoter or bait anchors.
- `--preys`: BED-like enhancer or prey anchors.
- `--control`: `NAME=PATH`.
- `--treatments`: one or more `NAME=PATH` values.
- `--depths`: sequencing depth values as `NAME=INTEGER`.

### Run the full background workflow

Use `background run` to count each sample, compare ratios, write plots, and
write a manifest:

```bash
uv run touche background run \
  --control DMSO=dmso.nodups_30_intra.pairs.gz \
  --treatments FLV=flv.nodups_30_intra.pairs.gz TRP=trp.nodups_30_intra.pairs.gz \
  --depths DMSO=1881564360 FLV=1623244357 TRP=1718104230 \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --min-bg-distance 10000 \
  --max-bg-distance 150000 \
  --out-dir results/background \
  --source touche
```

The run wrapper writes per-sample counts under `counts/`, comparison plots under
`plots/`, a merged `background_comparison.tsv`, and `manifest.json`.

### Run individual background steps

Count one sample:

```bash
uv run touche background count \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --min-bg-distance 10000 \
  --max-bg-distance 150000 \
  --out results/background/counts/DMSO_EP_and_BG_contacts.tsv \
  --source touche
```

Run `background count` for each treatment, then compare the count tables:

```bash
uv run touche background compare \
  --control DMSO=results/background/counts/DMSO_EP_and_BG_contacts.tsv \
  --treatments FLV=results/background/counts/FLV_EP_and_BG_contacts.tsv TRP=results/background/counts/TRP_EP_and_BG_contacts.tsv \
  --depths DMSO=1881564360 FLV=1623244357 TRP=1718104230 \
  --min-ep-cpb 8 \
  --out-dir results/background/plots \
  --table-out results/background/background_comparison.tsv
```

The `--min-ep-cpb` threshold filters pairs by control EP contacts per billion
contacts before plotting comparisons.

`background count`/`background run` default to `--index-strategy all` (parse
`--pairs` directly, matching prior behavior). Pass `--index-strategy cache` to
read a persistent NPZ `ContactIndex` cache instead -- useful when `apa
aggregate`/`apa run` also runs against the same sample, since both would
otherwise each re-parse the same pairs file:

```bash
uv run touche preprocess build-cache \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/dmso \
  --prefix dmso

uv run touche background count \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --min-bg-distance 10000 \
  --max-bg-distance 150000 \
  --out results/background/counts/DMSO_EP_and_BG_contacts.tsv \
  --index-strategy cache \
  --cache-dir .cache/touche/dmso \
  --cache-prefix dmso \
  --require-cache
```

`background run`'s `--cache-dir` is a base directory namespaced per sample
(`--cache-dir/DMSO`, `--cache-dir/FLV`, ...) rather than a single shared
manifest, since each sample's cache is distinct. If omitted, each sample's
cache defaults to a `contact_index_cache/` directory next to that sample's own
count output. `--require-cache` fails the command instead of building a
missing cache implicitly.

`background count` accepts `--pairs-list pairs.bedpe` in place of
`--baits`/`--preys` to count an explicit pair list; see [Pairs](#pairs).

### Test the EP-versus-background change between two libraries

```bash
uv run touche background diff \
  --control DMSO=results/background/counts/DMSO_EP_and_BG_contacts.tsv \
  --treatment FLV=results/background/counts/FLV_EP_and_BG_contacts.tsv \
  --out results/background/DMSO_vs_FLV.tsv
```

Per pair, this is the log odds ratio of the 2x2 table of (EP, background) by
(control, treatment), with BH-adjusted q-values. Testing EP relative to its
own local background is what makes the comparison robust to a difference in
library depth.

Zeros are kept: a pair with contacts in one library and none in the other is
a complete gain or loss, and the raw `log2_ratio_change` stays infinite for
it. A separate `log2_ratio_change_display` column applies
`--display-pseudocount` purely so a plot has something finite to draw.

With one library per condition this is inference about read sampling
conditional on those two libraries, not evidence of biological variation
between conditions. The result's metadata says so.

`background compare` is the plotting path and keeps its reference-reproducing
defaults, including the filter that requires positive EP signal in every
sample. `--zero-policy keep` turns that filter off, and `--scale per_billion`
switches the `EP_CPB_*` divisor from the historical `depth / 1e10` to
`depth / 1e9`.

## APA

`touche apa` runs aggregate peak analysis for bait/prey pairs and can compare
1D-normalized APA signal between a control and treatment sample.

Inputs:

- `--baits`: BED-like bait anchors.
- `--preys`: BED-like prey anchors.
- `--control`: `NAME=PAIRS`.
- `--treatment`: `NAME=PAIRS`.

### Run the full APA workflow

Use `apa run` to aggregate APA matrices for a control and treatment sample,
compare them, and write a manifest:

```bash
uv run touche apa run \
  --control DMSO=dmso.nodups_30_intra.pairs.gz \
  --treatment FLV=flv.nodups_30_intra.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --out-dir results/apa/FLV_vs_DMSO \
  --source touche
```

Each sample directory contains:

- `AggMat.csv`
- `AggHeatmap.svg`
- `baits_genome_wide_contacts.csv`
- `preys_genome_wide_contacts.csv`

The comparison directory contains `ObsOverExp.csv` and `ObsOverExp.svg`, and the
top-level output directory contains `manifest.json`.

### Run individual APA steps

Aggregate one sample:

```bash
uv run touche apa aggregate \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --out-dir results/apa/DMSO \
  --source touche
```

Aggregate the treatment sample the same way, then compare:

```bash
uv run touche apa compare \
  --control-apa results/apa/DMSO/AggMat.csv \
  --treatment-apa results/apa/FLV/AggMat.csv \
  --control-baits results/apa/DMSO/baits_genome_wide_contacts.csv \
  --control-preys results/apa/DMSO/preys_genome_wide_contacts.csv \
  --treatment-baits results/apa/FLV/baits_genome_wide_contacts.csv \
  --treatment-preys results/apa/FLV/preys_genome_wide_contacts.csv \
  --bait-count 1000 \
  --prey-count 1000 \
  --out results/apa/FLV_over_DMSO_1D_normalized_change_APA.svg \
  --matrix-out results/apa/FLV_over_DMSO_1D_normalized_change_APA.csv
```

`apa run` can infer `--bait-count` and `--prey-count` from the anchor files. The
standalone `apa compare` command requires them because it only receives the
already aggregated APA and signal files.

`apa aggregate`/`apa run` default to `--index-strategy all` (parse `--pairs`
directly, matching prior behavior). Pass `--index-strategy cache` to read a
persistent NPZ `ContactIndex` cache instead -- built with strand metadata
(not `--no-metadata`), since APA needs it. This cache can be shared with
`background count`/`background run` on the same sample:

```bash
uv run touche preprocess build-cache \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/dmso \
  --prefix dmso

uv run touche apa aggregate \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --out-dir results/apa/DMSO \
  --index-strategy cache \
  --cache-dir .cache/touche/dmso \
  --cache-prefix dmso \
  --require-cache
```

`apa run`'s `--cache-dir` is namespaced per sample the same way `background
run`'s is (`--cache-dir/DMSO`, `--cache-dir/FLV`, ...).

### Quantitative APA scores

`apa aggregate` can write a tidy score table alongside the matrix:

```bash
uv run touche apa aggregate \
  --pairs dmso.nodups_30_intra.pairs.gz \
  --baits promoters.bed \
  --preys enhancers.bed \
  --min-distance 25000 --max-distance 150000 \
  --window 10000 --pixels 50 \
  --out-dir results/apa/DMSO \
  --summary-out results/apa/DMSO/summary.tsv \
  --bootstrap 1000
```

The summary reports central enrichment, P2LL, P2M, promoter- and
enhancer-stripe enrichment, ring enrichment, and dot-to-stripe ratios, each
with its own numerator, denominator, and pixel counts. Masks are defined in
base pairs of offset, so the same `--masks` file means the same thing at any
resolution.

`--bootstrap` resamples whole **chromosomes**, not pixels — pixels within one
pileup are not independent observations — so it keeps one pileup per
chromosome in memory for the run. To score a matrix that was already written,
use:

```bash
uv run touche apa summarize --matrix results/apa/DMSO/AggMat.csv --out summary.tsv
```

That path has no per-chromosome pileups, so it reports scores without
confidence intervals.

`apa aggregate` also accepts `--pairs-list pairs.bedpe` in place of
`--baits`/`--preys`; see [Pairs](#pairs).

## Output and manifests

Commands print compact JSON summaries to stdout. The `run` wrappers also write a
`manifest.json` containing inputs, parameters, output paths, metrics, the
`touche` version, and elapsed runtime. Prefer the `run` wrappers for reproducible
end-to-end analysis, and the individual commands when debugging or replacing one
stage of a workflow.
