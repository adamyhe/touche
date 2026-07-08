# Reproducing reference plots

This guide maps the figure-producing workflows in
[Danko-Lab/E-P_contacts](https://github.com/Danko-Lab/E-P_contacts) to
`touche` commands.

The commands below are documentation examples. They are intended to reproduce
the same analysis shape and output files as the reference workflows after the
large Micro-C pairs files have been downloaded or generated.

## Inputs

The reference workflows use two categories of inputs:

- processed Micro-C pairs files, either downloaded from the Danko lab data
  location described by the
  [E-P_contacts README](https://github.com/Danko-Lab/E-P_contacts#readme) or
  generated from FASTQ files with
  [distiller-nf](https://github.com/open2c/distiller-nf)
- small anchor and annotation files stored in the upstream
  [`Input_files/`](https://github.com/Danko-Lab/E-P_contacts/tree/main/Input_files)
  directory

For the examples below, set a shell variable pointing to the upstream
`Input_files` directory:

```bash
EP_CONTACTS_INPUTS=/path/to/Input_files
```

For `touche`, first convert or filter each downloaded/generated pairs file into
the canonical analysis-ready format:

```bash
uv run touche preprocess filter-pairs \
  --pairs sample.pairs.gz \
  --out sample.nodups_30_intra.pairs.gz \
  --min-mapq 30 \
  --cis-only
```

See [micro-c-preprocessing.md](micro-c-preprocessing.md) for the distiller-nf
boundary and pairs format expectations.

## Local-decay violin plot

Reference target:

- workflow: `Contact_normalization_by_local_decay`
- final figure:
  `Contact_normalization_by_local_decay/Violinplot_for_normalized_contacts_by_pair_type.svg`
- reference inputs:
  - `GSE206131_K562_cis_mapq30_pairs.txt.gz`
  - `Input_files/Gasperini_dREG_based_TRE_baits_hg38.txt`
  - `Input_files/Gasperini_dREG_based_promoter_preys_hg38.txt`
  - `Input_files/Gasperini_dREG_based_functional.csv`
  - `Input_files/Gasperini_dREG_based_nonfunctional.csv`

The original workflow runs `ContactCaller_microC.bsh`, concatenates per-bait
outputs, assigns functional/nonfunctional/other pair labels, and plots
observed/expected contacts.

With `touche`, the one-command pipeline is:

```bash
uv run touche local-decay run \
  --baits "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_TRE_baits_hg38.txt" \
  --preys "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_promoter_preys_hg38.txt" \
  --pairs GSE206131_K562_cis_mapq30_pairs.txt.gz \
  --functional "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_functional.csv" \
  --nonfunctional "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_nonfunctional.csv" \
  --dist 1000000 \
  --cap 2000 \
  --plot-min-contacts 1 \
  --plot-min-distance 15000 \
  --out-dir results/reference-plots/local-decay
```

Expected outputs:

- `results/reference-plots/local-decay/ContactCaller_microC_output.tsv`
- `results/reference-plots/local-decay/ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv`
- `results/reference-plots/local-decay/Violinplot_for_normalized_contacts_by_pair_type.svg`
- `results/reference-plots/local-decay/manifest.json`

Equivalent lower-level commands are available when debugging:

```bash
uv run touche local-decay call \
  --baits "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_TRE_baits_hg38.txt" \
  --preys "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_promoter_preys_hg38.txt" \
  --pairs GSE206131_K562_cis_mapq30_pairs.txt.gz \
  --dist 1000000 \
  --cap 2000 \
  --out results/reference-plots/local-decay/ContactCaller_microC_output.tsv

uv run touche local-decay assign-pair-types \
  --contacts results/reference-plots/local-decay/ContactCaller_microC_output.tsv \
  --functional "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_functional.csv" \
  --nonfunctional "$EP_CONTACTS_INPUTS/Gasperini_dREG_based_nonfunctional.csv" \
  --out results/reference-plots/local-decay/ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv

uv run touche local-decay plot \
  --assignments results/reference-plots/local-decay/ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv \
  --min-contacts 1 \
  --min-distance 15000 \
  --out results/reference-plots/local-decay/Violinplot_for_normalized_contacts_by_pair_type.svg
```

## APA change heatmaps

Reference target:

- workflow: `APA_and_inter-sample_APA`
- final figures:
  - `APA_and_inter-sample_APA/FLV_over_DMSO_1D_normalized_change_APA.svg`
  - `APA_and_inter-sample_APA/TRP_over_DMSO_1D_normalized_change_APA.svg`
- reference inputs:
  - `mESCs_DMSO_30_intra.mm10.nodups.pairs.gz`
  - `mESCs_FLV_30_intra.mm10.nodups.pairs.gz`
  - `mESCs_TRP_30_intra.mm10.nodups.pairs.gz`
  - `Input_files/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed`
  - `Input_files/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed`

The reference commands use enhancer-promoter pairs within 25-150 kb, a 10 kb
half-window, and 50 pixels per half-window. They compare treatment APAs to DMSO
with 10,530 promoter baits and 27,900 enhancer preys.

Run the FLV comparison:

```bash
uv run touche apa run \
  --control DMSO=mESCs_DMSO_30_intra.mm10.nodups.pairs.gz \
  --treatment FLV=mESCs_FLV_30_intra.mm10.nodups.pairs.gz \
  --baits "$EP_CONTACTS_INPUTS/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --preys "$EP_CONTACTS_INPUTS/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --bait-count 10530 \
  --prey-count 27900 \
  --out-dir results/reference-plots/apa/FLV_vs_DMSO
```

Run the TRP comparison:

```bash
uv run touche apa run \
  --control DMSO=mESCs_DMSO_30_intra.mm10.nodups.pairs.gz \
  --treatment TRP=mESCs_TRP_30_intra.mm10.nodups.pairs.gz \
  --baits "$EP_CONTACTS_INPUTS/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --preys "$EP_CONTACTS_INPUTS/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --bait-count 10530 \
  --prey-count 27900 \
  --out-dir results/reference-plots/apa/TRP_vs_DMSO
```

Each `apa run` writes:

- raw aggregate APA outputs for the control sample
- raw aggregate APA outputs for the treatment sample
- `ObsOverExp.csv`
- `ObsOverExp.svg`
- `manifest.json`

For example, the FLV heatmap is:

```text
results/reference-plots/apa/FLV_vs_DMSO/FLV_vs_DMSO/ObsOverExp.svg
```

Use lower-level commands if you want to reuse one DMSO aggregate for both
treatment comparisons:

```bash
uv run touche apa aggregate \
  --pairs mESCs_DMSO_30_intra.mm10.nodups.pairs.gz \
  --baits "$EP_CONTACTS_INPUTS/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --preys "$EP_CONTACTS_INPUTS/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --out-dir results/reference-plots/apa/DMSO

uv run touche apa aggregate \
  --pairs mESCs_FLV_30_intra.mm10.nodups.pairs.gz \
  --baits "$EP_CONTACTS_INPUTS/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --preys "$EP_CONTACTS_INPUTS/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 10000 \
  --pixels 50 \
  --out-dir results/reference-plots/apa/FLV

uv run touche apa compare \
  --control-apa results/reference-plots/apa/DMSO/AggMat.csv \
  --treatment-apa results/reference-plots/apa/FLV/AggMat.csv \
  --control-baits results/reference-plots/apa/DMSO/baits_genome_wide_contacts.csv \
  --control-preys results/reference-plots/apa/DMSO/preys_genome_wide_contacts.csv \
  --treatment-baits results/reference-plots/apa/FLV/baits_genome_wide_contacts.csv \
  --treatment-preys results/reference-plots/apa/FLV/preys_genome_wide_contacts.csv \
  --bait-count 10530 \
  --prey-count 27900 \
  --window 10000 \
  --pixels 50 \
  --out results/reference-plots/apa/FLV_over_DMSO_1D_normalized_change_APA.svg \
  --matrix-out results/reference-plots/apa/FLV_over_DMSO_1D_normalized_change_APA.csv
```

## EP/background scatterplots

Reference target:

- workflow: `EP_contacts_compared_to_local_background`
- final figures:
  - `EP_contacts_compared_to_local_background/FLV_vs_DMSO.svg`
  - `EP_contacts_compared_to_local_background/TRP_vs_DMSO.svg`
  - `EP_contacts_compared_to_local_background/FLV_vs_TRP.svg`
- reference inputs:
  - `mESCs_DMSO_30_intra.mm10.nodups.pairs.gz`
  - `mESCs_FLV_30_intra.mm10.nodups.pairs.gz`
  - `mESCs_TRP_30_intra.mm10.nodups.pairs.gz`
  - the same promoter and TRE BED files used by the APA workflow

The reference commands count contacts between 5 kb windows around anchors,
screen pairs within 25-150 kb, and count local background contacts 10-150 kb
from the opposite anchor. The comparison filters pairs with at least 8 CPB in
one condition and uses these sequencing depths:

```text
DMSO  53226768
FLV   362862200
TRP   410040533
```

Run the full pipeline:

```bash
uv run touche background run \
  --control DMSO=mESCs_DMSO_30_intra.mm10.nodups.pairs.gz \
  --treatments FLV=mESCs_FLV_30_intra.mm10.nodups.pairs.gz TRP=mESCs_TRP_30_intra.mm10.nodups.pairs.gz \
  --depths DMSO=53226768 FLV=362862200 TRP=410040533 \
  --baits "$EP_CONTACTS_INPUTS/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --preys "$EP_CONTACTS_INPUTS/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 2500 \
  --min-bg-distance 10000 \
  --max-bg-distance 150000 \
  --min-ep-cpb 8 \
  --out-dir results/reference-plots/background
```

Expected outputs:

- `results/reference-plots/background/counts/DMSO_EP_and_BG_contacts.tsv`
- `results/reference-plots/background/counts/FLV_EP_and_BG_contacts.tsv`
- `results/reference-plots/background/counts/TRP_EP_and_BG_contacts.tsv`
- `results/reference-plots/background/background_comparison.tsv`
- `results/reference-plots/background/plots/FLV_vs_DMSO.svg`
- `results/reference-plots/background/plots/TRP_vs_DMSO.svg`
- `results/reference-plots/background/plots/TRP_vs_FLV.svg`
- `results/reference-plots/background/manifest.json`

The pairwise treatment plot name follows the order generated by `touche`
(`TRP_vs_FLV.svg`). The reference README names the same comparison
`FLV_vs_TRP.svg`.

Equivalent lower-level commands:

```bash
uv run touche background count \
  --pairs mESCs_DMSO_30_intra.mm10.nodups.pairs.gz \
  --baits "$EP_CONTACTS_INPUTS/dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --preys "$EP_CONTACTS_INPUTS/dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed" \
  --min-distance 25000 \
  --max-distance 150000 \
  --window 2500 \
  --min-bg-distance 10000 \
  --max-bg-distance 150000 \
  --out results/reference-plots/background/counts/DMSO_EP_and_BG_contacts.tsv

uv run touche background compare \
  --control DMSO=results/reference-plots/background/counts/DMSO_EP_and_BG_contacts.tsv \
  --treatments FLV=results/reference-plots/background/counts/FLV_EP_and_BG_contacts.tsv TRP=results/reference-plots/background/counts/TRP_EP_and_BG_contacts.tsv \
  --depths DMSO=53226768 FLV=362862200 TRP=410040533 \
  --min-ep-cpb 8 \
  --out-dir results/reference-plots/background/plots \
  --table-out results/reference-plots/background/background_comparison.tsv
```

Run `background count` for FLV and TRP before the `background compare` command
when using the lower-level form.

## Expected differences from the reference scripts

The `touche` commands are designed to reproduce the reference workflow outputs,
but they deliberately avoid some implementation details of the original code:

- `touche` does not run raw FASTQ processing; use distiller-nf or an equivalent
  workflow first.
- `touche` writes one final output table per command rather than concatenating
  many temporary per-anchor files.
- `touche` replaces the `rpy2` call to R's Fisher exact test with SciPy while
  preserving the reference rounding behavior before the test.
- File names are regularized in a few places, especially pipeline `manifest.json`
  files and the `TRP_vs_FLV.svg` comparison name noted above.
