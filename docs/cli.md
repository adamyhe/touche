# CLI reference

The `touche` command is installed by the `ep-touche` Python package:

```bash
pip install ep-touche
touche --help
```

From a development checkout, run the same commands through `uv`:

```bash
uv run touche --help
```

The examples below use `uv run touche` so they work from a checkout. If the
package is installed with `pip`, drop the `uv run` prefix.

## Command groups

```bash
uv run touche preprocess --help
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

For large files that will also be cached, write QC during cache construction
with `build-cache --qc-out` so the compressed pairs file is scanned once.

### Build NPZ caches

Build chromosome-sharded NPZ caches for repeated analysis:

```bash
uv run touche preprocess build-cache \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/sample \
  --prefix sample \
  --qc-out sample.qc.json
```

Add `--compressed` to write compressed NPZ shards. The default is uncompressed
NPZ, which is usually faster to load and avoids turning one cache into a large
monolithic archive.

The default cache builder emits chromosome-sharded NPZ files from one streaming
pass over the input. The optional QC file is computed during that same pass.

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

### Run the full workflow

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
  --source touche \
  --backend numpy
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
  --source touche \
  --backend numpy
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

`local-decay call` and `local-decay run` accept `--backend numba`, but the
current acceleration only covers observed-count helpers. LOWESS fitting remains
in Python and usually dominates runtime.

`local-decay call` and `local-decay run` default to `--index-strategy cache`.
The cache strategy builds or reuses chromosome-sharded NPZ contact indexes, then
loads one chromosome shard at a time. If `--cache-dir` is omitted, the cache is
created under `contact_index_cache/` next to the local-decay output table.

For repeated local-decay runs, build the cache explicitly and reuse it:

```bash
uv run touche preprocess build-cache \
  --pairs sample.nodups_30_intra.pairs.gz \
  --source touche \
  --cache-dir .cache/touche/sample \
  --prefix sample

uv run touche local-decay call \
  --baits baits.bed \
  --preys preys.bed \
  --pairs sample.nodups_30_intra.pairs.gz \
  --out results/local-decay/ContactCaller_microC_output.tsv \
  --index-strategy cache \
  --cache-dir .cache/touche/sample \
  --cache-prefix sample
```

Alternative strategies are available for diagnostics:

- `--index-strategy all`: read the pairs file once and hold every chromosome in
  memory. This is fastest when enough memory is available.
- `--index-strategy chromosome`: scan the pairs file once per bait chromosome
  and keep only that chromosome in memory. This avoids persistent cache files,
  but can be slow for gzipped pairs.

They also expose `--lowess-backend numba` as an experimental smoother. This is
faster on small synthetic benchmarks and matches the current chunked
local-decay smoothing wrappers on regression fixtures, but keep the default
`statsmodels` backend for the most conservative reference-compatible runs. Use
`--lowess-iterations` to change the number of robust residual reweighting passes;
lower values are faster but can change expected-contact estimates.

## Background

`touche background` counts enhancer-promoter contacts and local background
contacts, then compares normalized EP/background ratios across samples.

Inputs:

- `--baits`: BED-like promoter or bait anchors.
- `--preys`: BED-like enhancer or prey anchors.
- `--control`: `NAME=PATH`.
- `--treatments`: one or more `NAME=PATH` values.
- `--depths`: sequencing depth values as `NAME=INTEGER`.

### Run the full workflow

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
  --source touche \
  --backend numpy
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
  --source touche \
  --backend numpy
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

`background count` and `background run` support `--backend numba` for optional
accelerated EP/background counting. Install the speed extra first:

```bash
pip install "ep-touche[fast]"
```

The default remains `--backend numpy`.

## APA

`touche apa` runs aggregate peak analysis for bait/prey pairs and can compare
1D-normalized APA signal between a control and treatment sample.

Inputs:

- `--baits`: BED-like bait anchors.
- `--preys`: BED-like prey anchors.
- `--control`: `NAME=PAIRS`.
- `--treatment`: `NAME=PAIRS`.

### Run the full workflow

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
  --source touche \
  --backend numpy
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
  --source touche \
  --backend numpy
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

`apa aggregate` and `apa run` support `--backend numba` for optional accelerated
APA matrix and 1D signal counting.

## Output and manifests

Commands print compact JSON summaries to stdout. The `run` wrappers also write a
`manifest.json` containing inputs, parameters, output paths, metrics, the
`touche` version, and elapsed runtime. Prefer the `run` wrappers for reproducible
end-to-end analysis, and the individual commands when debugging or replacing one
stage of a workflow.
