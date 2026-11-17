# Roadmap

What `touche` plans to add, and what it deliberately leaves out. Items are
roughly ordered by priority within each section; none has a committed date.
For what has already shipped, see the [changelog](CHANGELOG.md). For the
statistical caveats of what exists today, see
[Not yet implemented](docs/statistics.md#not-yet-implemented) in the
statistics guide.

## Next

- **Replicate-aware differential contacts.** A validated sample/design-table
  parser and one replicate-aware GLM backend behind a stable interface,
  supporting batch/block covariates, offsets, and condition-by-region
  contrasts. This is what makes a `biological` inference class reachable;
  today every condition comparison is conditional on the libraries observed.
- **APA null controls.** Distance- and chromosome-matched random-shift
  controls and per-diagonal expected normalization, so APA scores are
  enrichments against a matched null rather than against masks of the same
  pileup.
- **Real-data benchmarking.** Run external callers (via the existing
  adapters) through `scripts/gasperini_benchmark.py` alongside the native
  methods.

## Later

- **Reproducibility and saturation QC.** P(s) curves, distance-stratified
  replicate concordance, HiCRep SCC, and deterministic downsampling
  stability curves.
- **Simulation suite** for type-I error, power, zero inflation, and
  overdispersion.
- **Cross-validated null fitting.** `p_null` is currently fitted from the
  same bait window the pair is tested in; the reuse is recorded in the
  metadata but not removed.
- **A faster decay fit.** Replace the chunked LOWESS that dominates
  local-decay's runtime with a spline or isotonic fit to the binned
  histogram, at the same accuracy as `decay_model="anchored"`.

## Exploratory

- An experimental AbLE-compatible local-quantification module.
- Optional ABC scoring.
- Optional promoter-centric graph and stripe summaries.

## Out of scope

- **FASTQ alignment, deduplication, and cooler generation.** Use
  [distiller-nf](https://github.com/open2c/distiller-nf) or equivalent;
  `touche` starts from processed pairs.
- **A native loop caller.** `touche` imports calls from FitHiC2, MaxHiC,
  HiC-DC+, Mustache, Peakachu, Chromosight, and HiCCUPS instead.
- **Capture Hi-C inference** (CHiCAGO, Chicdiff), unless `touche` formally
  expands to capture assays.
- **Changing the reference-reproduction path.** `method="legacy_fisher"` with
  `decay_model="legacy"` will keep reproducing the reference output exactly,
  defects included.
