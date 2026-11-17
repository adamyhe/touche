# Roadmap

What `touche` plans to add, and what it deliberately leaves out. Items are
roughly ordered by priority within each section; none has a committed date.
For what has already shipped, see the [changelog](CHANGELOG.md).

## In progress

Statistical validity of the analyses ported from the reference workflows:

- **Calibrated per-pair contact significance** with FDR control. The
  reference's Fisher score is not a calibrated test, so `local-decay call` is
  expected to default to a calibrated null, with the reference score kept for
  reproduction.
- **A corrected local-decay background model.** The reference decay fit
  integrates to ~0.54 rather than 1 on sparse histograms and is tilted by the
  window's inclusion geometry, biasing every expected count. The port
  reproduces this faithfully today; a corrected model is planned, with the
  reference model kept for reproduction.
- **An overdispersed null**, since real contact counts are ~2.7x
  overdispersed relative to binomial.
- **Self-describing results**: a metadata sidecar recording each result's
  method, null, test universe, and inference class.
- **Effect sizes and uncertainty**: group comparisons, cluster bootstrap
  intervals, quantitative APA scores, and zero-safe differential
  EP/background testing.
- **Interoperability**: explicit BEDPE pair lists, a stable `pair_id`, and
  import/export adapters for FitHiC2, MaxHiC, HiC-DC+, Mustache, Peakachu,
  Chromosight, and HiCCUPS.
- **A Gasperini K562 benchmark** for null calibration and functional
  prediction.

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
- **Benchmarking external callers** through the adapters alongside the
  native methods.

## Later

- **Reproducibility and saturation QC.** P(s) curves, distance-stratified
  replicate concordance, HiCRep SCC, and deterministic downsampling
  stability curves.
- **Simulation suite** for type-I error, power, zero inflation, and
  overdispersion.
- **Cross-validated null fitting**, so the null is not fitted from the same
  bait window the pair is tested in.
- **A faster decay fit.** Replace the chunked LOWESS that dominates
  local-decay's runtime with a spline or isotonic fit to the binned
  histogram.

## Exploratory

- An experimental AbLE-compatible local-quantification module.
- Optional ABC scoring.
- Optional promoter-centric graph and stripe summaries.

## Out of scope

- **FASTQ alignment, deduplication, and cooler generation.** Use
  [distiller-nf](https://github.com/open2c/distiller-nf) or equivalent;
  `touche` starts from processed pairs.
- **A native loop caller.** `touche` will import calls from established
  callers instead.
- **Capture Hi-C inference** (CHiCAGO, Chicdiff), unless `touche` formally
  expands to capture assays.
- **Changing the reference-reproduction path.** Corrected methods will be
  added alongside the reference behavior, which will remain available to
  reproduce the reference output exactly, defects included.
