# Statistics guide

This guide documents every inferential method `touche` provides: what it
tests, what it assumes, what its q-values are corrected over, and — most
importantly — what kind of claim its result supports.

One default did change: `local-decay call` now tests with the calibrated
binomial null instead of the reference workflow's Fisher score. The output
*layout* is unchanged, so only the p-value column moves, and
`--method legacy_fisher` reproduces the reference numbers byte for byte.
Everything else here is opt-in. See [Compatibility](#compatibility).

## Contents

- [The three inference classes](#the-three-inference-classes)
- [Result metadata](#result-metadata)
- [The pair universe](#the-pair-universe)
- [Per-pair contact significance](#per-pair-contact-significance)
- [Multiple testing](#multiple-testing)
- [Group comparisons and effect sizes](#group-comparisons-and-effect-sizes)
- [Covariate matching](#covariate-matching)
- [Quantitative APA scores](#quantitative-apa-scores)
- [Differential EP versus background](#differential-ep-versus-background)
- [Zeros](#zeros)
- [Importing external calls](#importing-external-calls)
- [Compatibility](#compatibility)

## The three inference classes

Every `touche` statistical result declares an `inference_class`. Read it
first; it determines what sentence you are allowed to write about the result.

| Class | What it supports | What it does **not** support |
| --- | --- | --- |
| `descriptive` | A summary of the data in hand. "Functional pairs have a higher median log2 O/E than non-functional pairs in this dataset." | Any claim about a population, biological or otherwise. |
| `technical` | Inference about read sampling, conditional on the libraries you sequenced. "Given these two libraries, the EP/background odds differ more than read sampling alone explains." | A claim that the condition causes the difference. Between-replicate biological variation was never sampled, so it cannot be ruled out. |
| `biological` | Inference that generalizes across biological replicates. | Nothing in `touche` currently returns this class. It requires replicate libraries as the unit of analysis; see [Not yet implemented](#not-yet-implemented). |

The single most common error in contact analysis is treating a `technical`
result as `biological`. A per-pair p-value of 1e-30 from one control and one
treatment library says the read counts differ; it says nothing about whether
a second pair of libraries would agree.

The second most common error is **pseudoreplication**: treating things that
are not independent observations as if they were. In this domain these are
all *not* independent replicates:

- multiple pairs sharing a promoter,
- multiple pairs sharing an enhancer,
- neighbouring contact bins,
- pixels within one APA window,
- read pairs from one library,
- multiple windows from one biological sample.

`touche` handles this by taking a `cluster_by`/`--cluster` argument on every
comparison and resampling whole clusters for its confidence intervals. The
nominal rank-test p-values still assume independent rows — which is exactly
why those results are labelled `descriptive`.

## Result metadata

Every inferential function returns a `StatResult`: a polars table plus a
`MethodInfo`. Writing one always writes a `.meta.json` sidecar beside the
table.

```python
import touche.api as tt

result = tt.test_contacts(calls, method="binomial")
result.info.inference_class   # "technical"
result.info.fdr_family        # "all called pairs"
result.info.warnings          # assumption violations, untestable rows, ...
result.write("calls.tested.tsv")   # also writes calls.tested.tsv.meta.json
```

The sidecar records the method and its version, hypothesis, null,
alternative, input and tested counts, FDR method and family, unit of
analysis, clustering unit, filtered counts with reasons, random seed, and
warnings. A q-value column whose family lives only in a terminal log is not
reproducible, so the sidecar is not optional.

`touche.metadata.METHOD_REGISTRY` holds the fixed documentation for each
method name; `tt.describe_method("binomial")` prints it.

## The pair universe

`touche.pairs` defines one canonical bait/prey pair table used by every
module. Its `pair_id` is:

- **center-based** — built from the two anchor *centers*, because every
  `touche` analysis reduces an anchor to its center before counting anything.
  This is what lets `local-decay` (which takes point center anchors) and
  `apa`/`background` (which take BED intervals) produce the same id for the
  same biological pair.
- **anchor-order-invariant** — the two anchors are sorted before the id is
  formed, so a BEDPE row and its anchor-swapped twin join.
- **deterministic** — built from an explicit string, not a built-in hash, so
  it is stable across `polars` and Python versions, row order, and chunking.

Two anchors sharing a center are one anchor as far as `touche` is concerned.
Full interval coordinates are retained in the table as provenance.

### Explicit pair lists

`apa aggregate` and `background count` analyze an implicit universe by
default: every bait crossed with every prey inside `[--min-distance,
--max-distance]`. `--pairs-list` replaces that with a BEDPE you supply.

```bash
# Materialize the implicit universe so it can be inspected and filtered
uv run touche pairs build \
  --baits promoters.bed --preys enhancers.bed \
  --min-distance 20000 --max-distance 1000000 \
  --out pairs.bedpe

# Analyze exactly that list
uv run touche background count --pairs sample.pairs.gz --pairs-list pairs.bedpe ...
uv run touche apa aggregate     --pairs sample.pairs.gz --pairs-list pairs.bedpe ...
```

The list is honoured exactly as given. No pair is added to it, and the
distance window does **not** prune it — a listed pair closer than
`--min-distance` is still counted. Silently replacing an explicit list with
a Cartesian product, or quietly filtering it, changes the hypothesis being
tested.

`read_bedpe` documents its handling of 0-based half-open coordinates, anchor
role assignment (`--bait-anchor first|second|left|right`), trans pairs (null
`chrom` and `distance`, never a nonsense distance), duplicates, invalid
intervals, and extra annotation columns, which are carried through untouched
so per-pair covariates can travel in the same file.

## Per-pair contact significance

### `binomial` (default) — technical

The local-decay model already computes the two quantities a calibrated test
needs. They are now emitted as columns:

- `n_trials` — the number of contacts anchored at the bait window
  (`bait_center ± cap`) heading in the prey's direction. An auditable count
  from the experiment, not a derived quantity.
- `p_null` — the probability that such a contact lands in the prey window,
  integrated from the bait's own fitted LOWESS distance-decay density over
  `[|d| - cap, |d| + cap]`.

`expected` is their product, which is what the legacy table already used.

The test is then

> K_i ~ Binomial(N_i, p0_i), one-sided upper tail.

- Null: the bait's contacts are distributed according to its fitted local
  distance decay alone.
- Alternative: greater.
- Unit of analysis: bait-prey pair. This is *technical* inference — it is
  about contact sampling within one library.
- Assumptions: contacts anchored at the bait are independent draws;
  `p0_i` is correctly specified.
- Expected/offset source: the per-bait LOWESS fit, described above.
- Failure modes: `n_trials = 0` makes a pair untestable (reported as NaN in
  the output, excluded from the FDR family, never reported as
  non-significant); small `n_trials` makes the discrete p-value coarse, so
  the achievable minimum p-value can exceed your alpha — this is warned
  about when the median `n_trials` is below 10; overdispersion relative to
  a binomial makes the test anticonservative.
- Caveat: `p0_i` is fitted from the same bait window the pair is tested in.
  That reuse is recorded in method metadata. Because the fit is a smooth
  function of distance over the whole ±`dist` window and each pair
  contributes a vanishing fraction of it, the leakage is small, but it is
  not zero — check calibration before quoting an FDR.

### `legacy_fisher` — descriptive, for reproduction only

The reference `ContactCaller_microC` workflow builds this 2x2 table per pair:

```text
              observed          fitted expected
  background  bins - observed   bins - fitted expected
```

where `bins` is the number of one-base distance-histogram bins, and applies
a one-sided Fisher exact test. Two things about it are not a generative
null: a *fitted expectation* appears where an observed count belongs, and
the trial total is a count of histogram positions rather than a contact
total.

It is retained under this name because reproducing the reference workflow's
published numbers matters, and it is what `scripts/reference_replication.py`
and [Reproducing reference plots](reproducing-reference-plots.md) pass. It
should be read as a reproducibility-preserving score, not a calibrated
p-value, and **adding BH correction to it does not fix that** — you must
calibrate a null before you can control an error rate over it.

- Null: not well defined (see above).
- Alternative: greater.
- Unit of analysis: bait-prey pair.
- Failure mode: p-values are not uniform under any null you can construct;
  `q < 0.05` language is not supportable.

### `poisson` — technical

The rare-event limit, `K_i ~ Poisson(N_i · p0_i)`. Useful because it is the
form most external callers report, so native and imported scores are
comparable. Testable from a legacy call table, since it needs only
`expected`; `binomial` is not, and raises rather than inventing a trial
total.

### Checking calibration

Never take calibration on faith. `assess_calibration` reports, per stratum,
a Kolmogorov-Smirnov test against `Uniform(0, 1)` and the empirical
rejection rate at several nominal levels:

```bash
uv run touche local-decay calibration --calls null_pairs.tidy.tsv --strata distance_bin
```

Run it on a **null** set — distance-, chromosome-, and coverage-matched
shifted or random pairs — not on real enhancer-promoter pairs, which are not
expected to be null. Stratify by distance and coverage: a method can look
uniform overall while being badly anticonservative for short-range or
low-coverage pairs, which are exactly the pairs you care about.

`touche`'s own test suite simulates from the declared binomial null and
asserts that type-I error is controlled at 0.01 and 0.05 within Monte Carlo
error, and that BH controls the false discovery proportion over a wholly
null family. That checks the *test*; it does not check the expected-count
model the test consumes. For that, and for whether any of this improves
functional prediction, run `scripts/gasperini_benchmark.py` -- see
[its guide](../scripts/gasperini_benchmark.md) and the warning under
[Known issues](#known-issues).

### Running it

```bash
# The default: calibrated binomial p-values in the reference layout
uv run touche local-decay call \
  --baits promoters.tsv --preys enhancers.tsv --pairs sample.pairs.gz \
  --out calls.tsv

# Add --schema tidy for q-values and the canonical pair schema
uv run touche local-decay call \
  --baits promoters.tsv --preys enhancers.tsv --pairs sample.pairs.gz \
  --out calls.tsv --schema tidy

# Reproduce the reference workflow's numbers exactly
uv run touche local-decay call \
  --baits promoters.tsv --preys enhancers.tsv --pairs sample.pairs.gz \
  --out calls.tsv --method legacy_fisher

# Or retest an existing tidy table under a different null
uv run touche local-decay test --calls calls.tsv --out retested.tsv --method poisson
```

`--schema legacy` (still the default) writes the reference nine-column
headerless layout and no q-values — there is no column to put them in, and
appending one would break every existing reader of that format. Because that
layout also cannot record which null produced its p-value column, a
non-`legacy_fisher` run writes a `.meta.json` sidecar beside it; a
`legacy_fisher` run writes none, leaving a reference-reproduction directory
byte-identical.

`--schema tidy` writes the canonical pair schema with `pair_id`,
`n_trials`, `p_null`, `log2_oe`, `q_value`, and the sidecar. **Use it if you
want q-values**: the default combination of a calibrated test and the legacy
layout gives you calibrated p-values with no FDR control.

## Multiple testing

`adjust_pvalues` implements Benjamini-Hochberg, Benjamini-Yekutieli, Holm,
and Bonferroni, matching R's `p.adjust` (BH/BY are verified against
`scipy.stats.false_discovery_control`, including ties and NaNs). Tied
p-values always receive identical adjusted values.

NaN p-values are **not tested**: they pass through as NaN and are excluded
from the family size `m`, so an untestable pair cannot inflate the correction
applied to the pairs that were tested.

The FDR family is explicit and recorded:

- `--fdr-scope` omitted — one global family over every called pair. This is
  what a bare `q_value` means.
- `--fdr-scope chrom` (repeatable) — a stratified family, corrected
  independently within each stratum. Only valid if the strata were chosen
  *before* looking at the p-values.

## Group comparisons and effect sizes

`compare_groups` reports, between two labelled groups of pairs: both group
summaries, the difference in median (or mean) with a bootstrap confidence
interval, Cliff's delta as a rank effect size, and a Mann-Whitney U p-value.
`compare_paired` is the matched-pairs equivalent (Wilcoxon signed-rank plus
rank-biserial correlation). `correlate` gives Pearson or Spearman with a
clustered interval.

```bash
uv run touche local-decay compare-groups \
  --table assignments.tsv --value-col log2_oe --group-col PosNeg \
  --cluster bait_id --bootstrap 1000 --out comparison.tsv
```

- Inference class: `descriptive`. A pair-level rank test is not a
  biological-replicate test however small its p-value.
- Unit of analysis: bait-prey pair.
- Clustering unit: whatever `--cluster` names. Choose the promoter when many
  enhancers share a promoter, the enhancer when one enhancer serves many
  promoters, the connected component when both, and the library for
  cross-sample questions.
- The bootstrap interval respects `--cluster`; the nominal rank-test p-value
  does not. When the two disagree in width, trust the interval.
- Omitting `--cluster` emits a warning saying the interval treats every pair
  as independent evidence and is therefore probably too narrow.
- Non-finite values (including the infinities a zero-denominator log ratio
  produces) are excluded from the statistics and **counted in metadata**,
  never silently dropped. Exact zeros are counted per group too.

Bootstrap intervals default to 1,000 resamples with a fixed seed, so they
reproduce exactly. `bootstrap_ci(method="bca")` gives bias-corrected and
accelerated bounds, jackknifing over whole clusters.

## Covariate matching

Pairs classified by functional outcome usually differ in genomic distance,
enhancer activity, promoter transcription, accessibility, and coverage. An
unmatched comparison attributes those differences to regulatory function.

```python
matched = tt.match_pairs(
    pairs,
    group_col="PosNeg",
    covariates={"log_distance": 0.1, "coverage": 0.25},  # column -> caliper
    groups=("positive", "negative"),
)
tt.balance_table(pairs,   group_col="PosNeg", covariates=["log_distance", "coverage"])  # before
tt.balance_table(matched, group_col="PosNeg", covariates=["log_distance", "coverage"])  # after
```

Matching is greedy, without replacement, in a seeded random case order.
Controls are ranked by caliper-scaled distance, so "nearest" means the same
thing whether a covariate is in bases or reads. A case with no control inside
every caliper is dropped rather than matched to a distant one.

Use **log** distance, not raw distance: contact frequency is roughly linear
in log distance, so a 5 kb caliper means something very different at 20 kb
than at 500 kb.

`balance_table` reports the standardized mean difference per covariate.
Above roughly 0.1 in absolute value is the conventional signal that a
covariate is still imbalanced and can still confound.

## Quantitative APA scores

`summarize_apa` turns a pileup into numbers. Masks are defined in **base
pairs of offset from the anchors**, not pixel indices, so one mask file means
the same thing at 200 bp and 2 kb and summaries are comparable across
resolutions.

Default masks, scaled to `--window`:

| Mask | Region |
| --- | --- |
| `all` | every pixel |
| `dot` | central ±10% in both directions |
| `promoter_stripe` | central column band, excluding the dot |
| `enhancer_stripe` | central row band, excluding the dot |
| `ring` | ±20% square, with the stripe cross removed (a donut) |
| `lower_left` | outer lower-left corner block |
| `global_background` | everything outside the cross and ring |

Reported scores: `central_enrichment`, `p2ll`, `p2m`,
`promoter_stripe_enrichment`, `enhancer_stripe_enrichment`,
`ring_enrichment`, `dot_to_promoter_stripe`, `dot_to_enhancer_stripe`. Every
row carries its own numerator, denominator, and both pixel counts, so no
ratio has to be reverse-engineered, and each mask's own mean and sum are
reported alongside.

```bash
uv run touche apa aggregate ... --summary-out apa/summary.tsv --bootstrap 1000
uv run touche apa summarize --matrix apa/AggMat.csv --out summary.tsv --masks masks.yaml
```

Masks round-trip through JSON or YAML (`tt.read_masks` / `tt.write_masks`),
so a custom mask set travels with an analysis.

**Confidence intervals resample chromosomes, never pixels.** Pixels within
one pileup are not independent and are not a sample of anything, so an
interval built from them is meaningless however tight it looks.
`--bootstrap` therefore requires per-chromosome pileups
(`compute_apa(keep_chromosomes=True)`, which `apa aggregate --bootstrap`
enables for you); that costs a few megabytes, not one matrix per pair.
Asking for a bootstrap without them returns null bounds and a warning saying
why, rather than silently substituting a pixel bootstrap. Fewer than ten
chromosomes also warns.

### Ratio of aggregates versus aggregate of ratios

These are not equal, and `touche` reports the components so you can tell
which you have:

```text
  sum(x_i) / sum(y_i)   !=   mean(x_i / y_i)
```

APA mask scores are **ratios of aggregates**: the numerator and denominator
are each summed over their mask first. `compare_apa_change`'s matrix is a
ratio of per-pixel aggregates. Where a pair-level ratio is wanted instead,
compute it from the retained raw numerator and denominator columns.

## Differential EP versus background

`background diff` compares one treatment library against one control on the
EP-versus-background contrast. For each pair:

```text
                enhancer-promoter    matched background
  control             EP_C                  BG_C
  treatment           EP_T                  BG_T
```

and the estimand is the log odds ratio of that table. Testing EP *relative
to its own local background* is what makes the comparison robust to a global
difference in depth or coverage, which a bare EP fold change is not.

```bash
uv run touche background diff \
  --control DMSO=counts/DMSO.tsv --treatment FLV=counts/FLV.tsv \
  --out diff.tsv --method wald
```

- Inference class: **technical**. With one library per condition this cannot
  be anything else: the only randomness a per-pair 2x2 table can speak to is
  read sampling within the libraries observed. It cannot separate a condition
  effect from between-animal, between-passage, or between-prep variation,
  because those were never sampled. The result says so in its warnings.
- Null: log odds ratio = 0, conditional on the observed libraries.
- Alternative: two-sided.
- Unit of analysis: bait-prey pair within one library pair.
- `--method wald` uses the Woolf standard error (vectorized, the default);
  `--method fisher` is an exact two-sided conditional test per pair, better
  behaved at small counts and substantially slower (budget roughly a minute
  per million pairs).
- `--correction` is the Haldane-Anscombe continuity correction applied to the
  *tested* table. The reported raw `log2_ratio_change` is uncorrected.
- FDR family: all pairs shared by both libraries. Pairs present in only one
  input are not tested — absence from a counts file means the pair was
  outside that sample's candidate universe, which is not the same as zero
  contacts — and their count is reported.

## Zeros

A pair with contacts in one library and none in the other is a **complete
gain or loss**, usually the largest real effect in the dataset. Filtering it
because a ratio is undefined removes exactly the strongest observations.

`touche`'s rules:

- Raw estimates keep their infinities. `log2_fold_change` with the default
  `pseudocount=0` returns `+inf` for a complete gain, `-inf` for a complete
  loss, and `NaN` for `0/0`.
- Pseudocounts are for display only. `background diff` reports both
  `log2_ratio_change` (raw) and `log2_ratio_change_display` (finite), and
  records the pseudocount used. The ranking a pseudocount induces among
  zero-containing pairs depends entirely on its size, so it is never applied
  silently.
- The four zero classes — zero/zero, zero/positive, positive/zero,
  positive/positive — are counted in metadata.
- Zeros stay in the inferential population, so the FDR family matches the
  declared pair universe.
- Plots filter for finite positive values at the plotting step, not in the
  analysis table.

`background compare`'s requirement that every sample have positive EP signal
is a **visualization** filter. It remains the default there so the reference
plot reproduces, but `--zero-policy keep` turns it off, and anything
inferential should use `background diff`, which keeps the full universe by
default.

## Importing external calls

`touche` does not implement a loop caller. FitHiC2, MaxHiC, HiC-DC+,
Mustache, Peakachu, Chromosight, and HiCCUPS are mature and make different
modelling assumptions; the useful thing is to interoperate with them.

```bash
uv run touche pairs import-calls --input fithic2.significances.txt \
  --format fithic2 --out calls.tsv

uv run touche pairs annotate --pairs pairs.bedpe --calls calls.tsv \
  --format bedpe --out annotated.tsv --slop 2000
```

Notes that matter in practice:

- Column resolution is by header name with aliases, matched without regard
  to case or punctuation. A mismatch raises an error listing the header
  actually found and pointing at `--column FIELD=COLUMN`, rather than
  silently mapping the wrong column — these tools rename output columns
  between releases.
- Negative-log significance scales are converted back to probabilities on
  import.
- Method-specific scores are **not** collapsed into one column named
  `probability`. Peakachu's classifier posterior, Chromosight's pattern
  correlation, and FitHiC2's binomial p-value stay distinct, with `method`
  and `source_format` attached.
- Genome build and chromosome naming are **not** translated. If your anchors
  say `chr1` and the imported calls say `1`, nothing will overlap; harmonize
  them first.
- Matching is by anchor containment in both anchor orders, not `pair_id`
  equality — `touche` anchors are points and external callers report bins.
  `--slop` widens each call anchor. Ties resolve to the best q-value, so the
  annotation is deterministic.

`write_loop_calls` exports BEDPE or a FitHiC2-style significance table for
downstream tools.

## Compatibility

One default numerical behaviour changed: the per-pair null. The output
*layout* did not, so a reader of the nine-column table still finds nine
columns in the same order — column five now holds a binomial p-value rather
than a Fisher score. `--method legacy_fisher` restores the reference numbers
exactly, and is what the reference-replication script and guide pass.

Untestable pairs (`n_trials = 0`) write `NaN` in that column rather than a
misleading 1.0. Standard parsers read it as a floating-point NaN.

Everything else is unchanged:

| Behaviour | Default | Opt out |
| --- | --- | --- |
| `local-decay call` p-values | **calibrated `binomial`** (changed) | `--method legacy_fisher` |
| `local-decay call` layout | reference nine-column headerless TSV | `--schema tidy` |
| `background compare` filter | drops pairs without positive EP signal in every sample | `--zero-policy keep` |
| `EP_CPB_*` divisor | `depth / 1e10` | `--scale per_billion` |
| APA pileup universe | distance-filtered bait x prey product | `--pairs-list` |

### The CPB unit

The historical `EP_CPB_*` divisor is `depth / 1e10` — contacts per *ten*
billion, despite the name. This is the divisor under which the reference
workflow's published `--min-ep-cpb 8` threshold reproduces its plots, so it
remains the default as `--scale legacy`. `--scale per_billion` uses
`depth / 1e9`, the conventionally named unit, for new analyses. Both are
recorded in the command's JSON summary.

## Known issues

### The fitted decay density loses mass

`touche.local_decay.fit_distance_decay_model` returns a background density
that integrates to roughly **0.54** rather than 1, while the raw distance
histogram it smooths integrates to ~1. Both LOWESS backends agree, so this
is the chunked smooth-and-merge itself, not a backend approximation.

The consequence is that `p_null` -- and therefore `expected` -- is roughly
a factor of two too small. On distance-preserving null pairs the benchmark
measures `observed / expected` at about **1.8**, roughly uniformly across
distance strata.

This predates the calibrated tests: `legacy_fisher` consumed the same
`expected`. It mattered less there only because that test placed `expected`
in a contingency table against a background of hundreds of thousands of
histogram bins, where a factor of two barely moved a p-value that was
pinned near 0.5 regardless. `binomial` and `poisson` use `expected`
directly, so they inherit the bias and reject roughly twice as often as
their nominal level.

**Consequence for use:** treat `q_value` from `binomial`/`poisson` as a
ranking, not as a calibrated FDR, until this is resolved. The *ranking* is
unaffected -- a near-uniform multiplicative bias in `p_null` is monotone,
so it does not reorder pairs -- which is why the benchmark's AUPRC
comparison is still meaningful.

It is not yet known whether the reference R implementation has the same
behaviour (in which case `touche` is faithfully reproducing it) or whether
this is a porting defect in the chunked merge. Resolving that needs the
reference sources, and fixing it would change `expected` in the reference
nine-column output, so it is deliberately not changed here.

Reproduce with `scripts/gasperini_benchmark.py --demo`, which reports
`observed_over_expected` in `expected_bias.tsv` and flags anticonservative
methods in its summary verdict.

## Not yet implemented

These are real gaps, not oversights. They are documented here so a result is
not mistaken for something it is not:

- **Replicate-aware differential contacts.** No negative-binomial GLM over a
  sample/design table yet, so no `biological` inference class is reachable.
  Until then, a condition comparison in `touche` is conditional on the
  libraries observed.
- **APA null controls.** No distance- and chromosome-matched random-shift
  null and no per-diagonal expected normalization. APA scores are therefore
  enrichments against masks of the same pileup, not against a matched null.
- **Reproducibility and saturation QC.** No P(s) curves, distance-stratified
  replicate concordance, HiCRep SCC, or downsampling stability curves.
- **Cross-validated null fitting.** `p_null` is fitted from the same bait
  window the pair is tested in; the reuse is recorded but not removed.
