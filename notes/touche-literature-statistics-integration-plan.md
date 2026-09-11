# Touché: literature and statistics integration plan

> Engineering handoff for extending the enhancer–promoter contact-analysis toolkit.
>
> Repository audited: [`Danko-Lab/touche`](https://github.com/Danko-Lab/touche), which currently redirects to [`adamyhe/touche`](https://github.com/adamyhe/touche).  
> Audited revision: [`3a8ef67af8110f32e87522ee90e16f2e1db7600d`](https://github.com/adamyhe/touche/tree/3a8ef67af8110f32e87522ee90e16f2e1db7600d), dated 2026-08-29.  
> Package version at audit: 0.1.4.

## Task objective

Extend Touché with literature-supported analyses and statistics that improve inferential validity, reproducibility, and interoperability while preserving its high-performance core and existing outputs.

Implement this as a sequence of small, backward-compatible pull requests. Do **not** attempt to add every method in one change. Statistical correctness, explicit assumptions, and stable schemas take precedence over maximizing the number of features.

## Executive recommendation

Touché already covers fast preprocessing, local distance-decay modeling, pair-type summaries, enhancer–promoter/background aggregation, and aggregate peak analysis (APA). The largest immediate opportunity is **statistical completeness**, not another de novo loop caller.

The recommended order is:

1. Add explicit pair identities, multiple-testing correction, effect sizes, confidence intervals, and zero-safe inference.
2. Add replicate-aware differential testing in which the biological library is the experimental unit.
3. Expand APA with quantitative subregion scores, clustered uncertainty, and matched null controls.
4. Add reproducibility and saturation QC.
5. Add adapters for established significance and loop-calling tools rather than reimplementing them.
6. Treat AbLE-style absolute contact estimation, ABC scoring, and graph-level analyses as optional advanced modules.

The most important statistical design principle is:

> Contact pairs, pixels, and APA cells are not biological replicates. Preserve biological libraries and perform inference at the library level, with clustering or resampling that respects shared promoters, shared enhancers, and local chromatin structure.

## Scope of the audit

This is an engineering-focused review of methods with clear relevance to enhancer–promoter contact analysis. It is not a formal systematic review or a claim that every published method has been enumerated.

The audited code currently includes:

- preprocessing, quality control, and caching;
- per-bait local distance-decay estimation using LOWESS, including zero inflation;
- a raw one-sided Fisher-style significance calculation;
- assignment of pair types and violin plots of log2 observed/expected signal;
- strand-aware APA, raw aggregation, and 1D-normalized intersample ratios;
- enhancer–promoter/background counts and ratio plots.

Relevant implementation locations:

- [Project README](https://github.com/adamyhe/touche/blob/3a8ef67af8110f32e87522ee90e16f2e1db7600d/README.md)
- [Local-decay significance calculation](https://github.com/adamyhe/touche/blob/3a8ef67af8110f32e87522ee90e16f2e1db7600d/src/touche/local_decay.py#L755-L790)
- [Statistical helpers](https://github.com/adamyhe/touche/blob/3a8ef67af8110f32e87522ee90e16f2e1db7600d/src/touche/stats.py#L11-L50)
- [Background analysis](https://github.com/adamyhe/touche/blob/3a8ef67af8110f32e87522ee90e16f2e1db7600d/src/touche/background.py#L284-L301)
- [APA aggregation and normalization](https://github.com/adamyhe/touche/blob/3a8ef67af8110f32e87522ee90e16f2e1db7600d/src/touche/apa.py#L410-L459)

Important gaps at the audited revision include:

- no global false-discovery-rate-adjusted `q_value` output;
- few group-level inferential tests or confidence intervals;
- no biological-replicate-aware differential-contact model;
- no general explicit BEDPE/pair-list input shared by APA and background analyses;
- limited support for external covariates and matched controls;
- no standard expected-matrix or random-shift APA null;
- no common import layer for external loop/significance calls;
- potential loss of complete gains or losses when background ratios contain zeros.

## Prioritized integration matrix

| Capability | Literature precedent | Value to Touché | Recommendation | Priority |
| --- | --- | --- | --- | --- |
| Effect sizes, rank tests, correlations, bootstrap intervals, APA subregion summaries | Analyses used in the Touché-associated study; common functional-genomics practice | Makes published comparisons reproducible and reports uncertainty, not only plots | Implement natively | P0 |
| Calibrated per-pair significance plus Benjamini–Hochberg FDR | FitHiC2 and related contact-significance methods | Converts nominal hits into a declared, auditable discovery set | Implement natively; retain legacy mode | P0 |
| Replicate-aware differential enhancer–promoter enrichment | diffHic; multiHiCcompare; count-GLM practice | Separates biological variation from read-sampling variation | Implement a stable interface with one validated backend first | P0 |
| Explicit BEDPE/pair-list mode and stable pair IDs | Standard interchange across 3D-genomics tools | Enables restricted hypotheses, provenance, interoperability, and correct resampling | Implement natively | P0 |
| Covariate-matched or weighted functional comparisons | Enhancer-prediction evaluation practice | Reduces confounding by distance, activity, coverage, accessibility, and transcription | Implement reusable matching/weighting utilities | P0 |
| Zero-safe foreground/background inference | General count-model practice | Preserves biologically important complete gains and losses | Correct current behavior | P0 |
| Quantitative APA scores and clustered bootstrap | coolpup.py and common pileup analyses | Turns heatmaps into comparable estimates with uncertainty | Implement natively | P1 |
| Random-shift and diagonal-expected APA nulls | coolpup.py and pileup literature | Distinguishes focal enrichment from distance decay and coverage | Implement natively or through a cooler adapter | P1 |
| P(s), replicate concordance, depth curves, and HiCRep SCC | HiCRep and standard 3D-genome QC | Detects unstable results and insufficient depth | Implement core QC plus optional adapter | P1 |
| External contact-significance backends | FitHiC2, MaxHiC, HiC-DC+ | Avoids maintaining several complex statistical callers | Add import/export adapters after benchmarking | P1 |
| AbLE-style local 2D loop quantification | AbLE | Potential absolute or locally calibrated contact interpretation | Experimental optional module with strict calibration metadata | P2 |
| ABC/contact component and promoter-centric graph or stripe summaries | ABC model; promoter–enhancer stripe studies | Connects contact signal to regulatory prioritization and multi-edge organization | Optional downstream modules | P2 |
| Native de novo loop calling | Mustache, Peakachu, Chromosight, HiCCUPS | Useful, but already served by mature tools | Do not reimplement; import calls | Non-goal |
| Capture-specific inference | CHiCAGO, Chicdiff | Strong methods for capture Hi-C designs | Keep outside core unless Touché formally expands to capture assays | Deferred |

## P0: statistical foundation

### 1. Stable pair schema and explicit pair-list mode

Create one canonical internal pair table used by local-decay, background, annotation, and APA modules. Every result must retain a stable `pair_id` and source-row provenance.

Recommended long-form columns:

```text
pair_id
bait_id
prey_id
chrom
distance
sample
condition
replicate
observed
expected
background
foreground_exposure
background_exposure
depth
log2_oe
log2fc
p_value
q_value
method
```

Allow additional annotation and model-diagnostic columns without breaking the core schema.

Add BEDPE or an equivalent explicit-pair input with documented handling of:

- 0-based half-open coordinates;
- anchor assignment and anchor swapping;
- strand-aware orientation;
- cis versus trans pairs;
- duplicate pairs;
- multiple features mapping to one anchor;
- out-of-range or missing bins;
- pair-level labels and covariates;
- deterministic `pair_id` generation;
- provenance for generated versus user-supplied pairs.

The same pair list must be usable for APA, foreground/background extraction, annotation, and export. Do not silently replace an explicit list with the Cartesian product of feature sets.

### 2. Calibrated per-pair significance and FDR

The current Fisher-style calculation should be retained only as a named compatibility mode, for example `legacy_fisher`. It appears to construct a nonstandard contingency table using a rounded fitted expectation as a cell and a complement related to the number of one-base histogram bins rather than an observed contact-count total. That null distribution is not clearly generated by the data-generating process.

Required changes:

1. Document the hypothesis, test universe, null, sidedness, and sample-size definition for every method.
2. Add a calibrated native test, initially one of:
   - a binomial test when an actual trial total and null contact probability are defensible;
   - a beta-binomial test when overdispersion can be estimated;
   - an empirical p-value from chromosome- and distance-matched shifted/null pairs.
3. Add global Benjamini–Hochberg correction and emit both `p_value` and `q_value`.
4. Make the FDR family explicit: all tested pairs, all pairs per sample, or a documented stratified family.
5. Never use `q < 0.05` language for a method whose nominal p-values fail null-calibration checks.

For a binomial formulation, the intended model is conceptually

\[
K_i \sim \operatorname{Binomial}(N_i, p_{0i}),
\]

where `K_i` is the observed contact count, `N_i` is an auditable read/contact trial total from the actual experiment, and `p0_i` is the distance- and bias-adjusted null probability. The number of histogram positions is not a substitute for `N_i`.

If the same observations are used to fit `p0_i` and test the pair, implement leave-one-out, cross-fitted, held-out, or empirical calibration where practical. At minimum, expose that reuse in method metadata.

FitHiC2 is the most directly relevant precedent for distance-aware expected contact modeling, binomial significance testing, and FDR control. Its implementation need not be copied, but Touché's output contract should be compatible with importing its results.

### 3. Effect sizes, uncertainty, and paper-parity summaries

Add reusable statistical functions for:

- medians and median differences;
- log2 fold change with an explicitly configurable plotting pseudocount;
- odds ratios or log odds ratios for foreground/background comparisons;
- Mann–Whitney U for independent descriptive groups;
- paired Wilcoxon signed-rank tests for genuinely paired observations;
- Pearson and Spearman correlations;
- bootstrap percentile or BCa confidence intervals, with 1,000 resamples as a default compatible with the associated analysis;
- sample sizes, missingness, zero counts, and the unit of analysis in every result.

These tests are useful for reproducing published analyses, but a nominal pair-level rank test is not automatically a valid biological-replicate test. Label descriptive tests clearly and offer clustered resampling by promoter, enhancer, or connected component when pairs share regulatory elements.

### 4. Replicate-aware differential enhancer–promoter enrichment

Do not pool biological replicates before inference. Add a sample/design table, for example:

```tsv
sample	condition	replicate	batch	path
control_r1	control	1	batch1	/path/control_r1.cool
control_r2	control	2	batch1	/path/control_r2.cool
treatment_r1	treatment	1	batch1	/path/treatment_r1.cool
treatment_r2	treatment	2	batch1	/path/treatment_r2.cool
```

The scientific target should be the condition-by-region interaction: does enhancer–promoter signal change more than its matched local background?

For pair `i`, sample `s`, and region type `r` (`EP` or `BG`), a count model may be expressed as:

\[
Y_{isr} \sim \operatorname{NB}(\mu_{isr}, \phi),
\]

\[
\log \mu_{isr} = \log E_{isr} + \beta_0 + \beta_C C_s + \beta_R R_r
                    + \beta_{CR}(C_s R_r) + \text{batch/sample/pair terms},
\]

where `E_isr` is a declared exposure or offset, such as usable library depth multiplied by foreground/background opportunity. The primary test is `H0: beta_CR = 0`.

Implementation requirements:

- preserve individual libraries through extraction and modeling;
- support batch/block covariates;
- retain foreground and background counts and exposures separately;
- return effect estimate, standard error, test statistic, p-value, q-value, convergence state, and method metadata;
- distinguish technical read-sampling inference from biological inference;
- reject or downgrade biological claims when a condition has no biological replication;
- include simulations and null contrasts to verify type-I error.

For an unreplicated exploratory comparison, a per-pair 2x2 table is acceptable only as technical-sampling inference:

```text
                enhancer-promoter    matched background
control              EP_C                   BG_C
treatment            EP_T                   BG_T
```

Report the log odds ratio and BH-adjusted p-value, but label the result as conditional on the observed libraries, not evidence of population-level biological variation.

Candidate backends include:

- a negative-binomial/quasi-likelihood model inspired by edgeR/diffHic;
- a distance-aware GLM inspired by multiHiCcompare;
- a beta-binomial model for foreground fractions when its assumptions fit the extracted data.

Use a backend interface so a validated method can be selected without changing result schemas. New heavy dependencies should remain optional.

### 5. Matched and clustered functional comparisons

Enhancer–promoter pairs classified by functional outcome often differ in genomic distance, enhancer activity, promoter transcription, accessibility, mappability, and contact coverage. An unmatched comparison can therefore attribute covariate effects to regulatory function.

Add matching or weighting on user-selected covariates, initially:

- genomic distance, preferably exact bins or calipers on log distance;
- baseline contact coverage or expected contact;
- enhancer accessibility/activity;
- promoter transcription;
- GC/mappability or other technical covariates when supplied.

Outputs must include balance diagnostics before and after matching/weighting, retained sample counts, effective sample size, and the estimand. Support chromosome-held-out evaluation to reduce leakage from locally correlated pairs.

For uncertainty, resample at a defensible cluster level:

- promoter when many enhancers share a promoter;
- enhancer when one enhancer maps to many promoters;
- connected component in the enhancer–promoter graph when both occur;
- biological library for cross-sample inference.

### 6. Zero-safe background analysis

Do not drop pairs solely because one condition has zero enhancer–promoter or background signal. Complete gains and losses are often the effects of greatest interest.

Requirements:

- preserve raw zeros in data and inferential models;
- use pseudocounts only for display transforms unless a statistical method explicitly requires and documents them;
- report infinite raw ratios rather than silently filtering them;
- provide a finite plotting value separately, with the pseudocount recorded;
- expose numbers of zero/zero, zero/positive, positive/zero, and positive/positive pairs;
- test all eligible pairs and apply FDR to the full declared universe.

The current `positive_all`-style filtering should remain only as an explicit visualization filter, not the default inferential population.

## P1: quantitative APA and quality control

### 7. APA subregion statistics

APA should return a tidy summary table in addition to a heatmap. Define named, configurable masks such as:

- `dot`: central pixel or central 3x3 window;
- `promoter_stripe`: central column excluding the dot;
- `enhancer_stripe`: central row excluding the dot;
- `edge` or `ring`: local outer neighborhood;
- `lower_left`: conventional lower-left background window;
- `global_background`: user-selected outer cells.

Report at least:

- central enrichment;
- peak-to-lower-left (`P2LL`);
- peak-to-mean-background (`P2M`);
- promoter-stripe enrichment;
- enhancer-stripe enrichment;
- edge/ring enrichment;
- dot-to-stripe ratio;
- the numerator and denominator for every ratio.

Mask definitions must be stored in result metadata and scale predictably with resolution/window size.

For uncertainty, resample pairs or biological units, not individual matrix pixels. Offer cluster bootstrap by promoter or enhancer and a biological-replicate bootstrap when multiple libraries exist. Return confidence intervals and the effective number of independent clusters.

### 8. APA null controls

Add two complementary null strategies:

1. **Distance- and chromosome-matched random shifts or random pairs.** Preserve cis distance, chromosome, coverage strata, and blacklist constraints while excluding known target windows.
2. **Per-diagonal expected normalization.** Divide or residualize each observed value against a distance-specific expected value computed from an appropriate matrix or imported expected track.

Use deterministic random seeds and record the matching strata, number of null replicates, exclusions, and fallback behavior. Permit separate nulls for dot, promoter stripe, and enhancer stripe because those structures need not share the same background distribution.

The coolpup.py model is a useful precedent for scalable pileups, expected normalization, and controls. Prefer interoperability with `.cool`/`.mcool` expected tables rather than inventing a private format.

### 9. Reproducibility, depth, and saturation QC

Add machine-readable and plotted QC for:

- contact probability versus genomic distance, `P(s)`;
- usable cis contacts and library depth;
- distance-stratified correlations between biological replicates;
- deterministic downsampling/rarefaction curves;
- stability of pair scores, significant sets, and APA scores across depths;
- overlap/Jaccard or rank concordance for calls at matched depth;
- optional HiCRep stratum-adjusted correlation coefficient through an adapter.

Downsampling must operate on counts or raw contact records when valid, not simply scale already normalized matrices. Record the seed and target depth. A result should be marked unstable when the chosen depth lies before its score/call curve has reasonably plateaued.

### 10. External significance and loop-call adapters

Define a narrow adapter contract rather than reimplementing mature callers. Import/export should support at least:

- BEDPE-like anchors;
- sample and method identifiers;
- observed and expected counts where available;
- p-values, q-values, scores, and model-specific diagnostics;
- resolution and coordinate conventions;
- reference genome and chromosome naming;
- stable mapping back to `pair_id`.

Prioritize adapters for:

- FitHiC2;
- MaxHiC;
- HiC-DC+.

Benchmark them against Touché's native local model before selecting a default. Different methods make different assumptions about count distributions, genomic distance, bias covariates, and local versus global backgrounds; do not collapse all method-specific scores into a column named simply `probability`.

Import standardized loop calls from tools such as Mustache, Peakachu, Chromosight, or HiCCUPS for downstream Touché analysis. Native de novo loop detection is outside the recommended core scope.

## P2: optional advanced modules

### 11. AbLE-style local 2D quantification

AbLE provides a relevant recent direction for converting a local 2D contact pattern into a loop-strength estimate and, under explicit calibration assumptions, an absolute contact probability. A Touché integration could be valuable for comparing regulatory contacts across perturbations.

Treat this as experimental because absolute interpretation depends on normalization, calibration, matrix balancing, resolution, and assay characteristics. The module should:

- accept a local contact window and declared normalization;
- return local background and focal enrichment components;
- separate relative enrichment from any absolute-probability estimate;
- require and preserve calibration metadata before labeling a result absolute;
- include reference examples against the authors' implementation;
- never imply absolute molecular contact frequency from an uncalibrated bulk matrix.

### 12. ABC and promoter-centric network summaries

An optional Activity-by-Contact module can combine an externally supplied activity term with Touché contact estimates:

\[
\operatorname{ABC}_{e,p} =
\frac{A_e C_{e,p}}{\sum_{e' \in W_p} A_{e'} C_{e',p}}.
\]

The implementation must expose the promoter window, activity definition, contact normalization, candidate enhancer universe, denominator, and any threshold. It should not bundle a particular accessibility or histone-mark pipeline into the core.

Promoter-centric graph summaries may include degree, weighted degree, entropy/concentration of enhancer contacts, connected components, and stripe/dot composition. These are useful descriptors, but pairwise bulk contacts do **not** establish simultaneous multiway hubs. Name outputs accordingly and avoid causal or multiway claims.

## Correctness and compatibility issues to resolve

### Legacy Fisher calculation

Keep the current result available only under an explicit `legacy_fisher` method name. Add warnings or metadata explaining that it is retained for reproducibility and should not be interpreted as a calibrated test until validated by simulation and matched-null data.

Adding BH correction cannot repair an invalid or miscalibrated nominal p-value. Calibrate the null first, then control FDR.

### Count-per-billion scaling

The audited code appears to use a divisor based on `depth / 1e10` for a count-per-billion-like transformation, although CPB conventionally uses `1e9`. Verify the intended unit against fixtures, historical outputs, and documentation before changing it.

If it is a defect:

- preserve the old behavior under an explicit compatibility mode;
- add a corrected mode or rename the historical unit;
- version the output metadata;
- include regression tests showing the exact factor difference;
- avoid silently changing published numerical outputs.

### Ratio-of-aggregates versus aggregate-of-ratios

For APA and foreground/background comparisons, make the estimand explicit. These quantities are not generally equal:

\[
\frac{\sum_i x_i}{\sum_i y_i}
\neq
\frac{1}{n}\sum_i \frac{x_i}{y_i}.
\]

Return raw numerators, raw denominators, pair-level ratios when defined, and aggregate ratios. Name each result unambiguously.

### Pseudoreplication

Avoid treating the following as independent biological observations:

- multiple pairs sharing a promoter;
- multiple pairs sharing an enhancer;
- neighboring contact bins;
- pixels within one APA window;
- technical read pairs from one library;
- multiple windows extracted from the same biological sample.

Use cluster-aware uncertainty for descriptive pair analyses and library-level replication for biological claims.

## Proposed CLI and API surface

Names are illustrative; align them with the project's existing command style.

```bash
# Recompute or adjust pair-level significance
touche local-decay test \
  --method binomial \
  --alternative greater \
  --fdr bh \
  --fdr-scope sample

# Compare annotated groups with effect sizes and clustered uncertainty
touche local-decay compare-groups \
  --group-col functional_class \
  --cluster promoter_id \
  --bootstrap 1000

# Replicate-aware enhancer-promoter versus background interaction
touche background diff \
  --samples samples.tsv \
  --pairs pairs.bedpe \
  --method nb-glm \
  --design '~ batch + condition * region_type'

# Quantitative APA summaries
touche apa summarize \
  --pairs pairs.bedpe \
  --masks masks.yaml \
  --bootstrap 1000 \
  --cluster promoter_id \
  --null random-shift

# Attach imported loop/significance calls
touche annotate loops \
  --input fithic2_results.tsv \
  --format fithic2

# Reproducibility and depth QC
touche qc replicate --samples samples.tsv --method distance-stratified
touche qc saturation --sample sample.cool --depths 0.25,0.5,0.75,1.0

# Optional downstream score
touche score abc --activity activity.tsv --contacts pair_scores.tsv
```

Suggested Python-level contracts:

```python
adjust_pvalues(table, p_col="p_value", method="bh", groupby=None)

test_contacts(table, method="binomial", alternative="greater", **method_options)

compare_groups(
    table,
    value_col,
    group_col,
    paired_by=None,
    cluster_by=None,
    bootstrap=1000,
    seed=0,
)

fit_differential_contacts(
    counts,
    sample_table,
    design,
    contrast,
    method="nb_glm",
    offset_cols=("depth", "exposure"),
)

summarize_apa(
    matrices,
    masks,
    cluster_by=None,
    bootstrap=1000,
    null=None,
    seed=0,
)
```

Every statistical result should include:

- `method` and method version;
- input count and number tested;
- hypothesis and alternative;
- effect estimate and direction;
- p-value and q-value, when applicable;
- confidence interval, when applicable;
- unit of analysis and clustering unit;
- missing/filtered counts and reasons;
- random seed for stochastic procedures;
- warnings for violated assumptions or inadequate replication.

## Implementation sequence

### Release A: statistical correctness and shared schema

Implement as separate reviewable pull requests:

- [ ] Define the canonical pair schema and stable `pair_id`.
- [ ] Add explicit BEDPE/pair-list input shared by APA and background analysis.
- [ ] Add BH-adjusted `q_value` with a documented test universe.
- [ ] Rename current significance output to `legacy_fisher` without breaking old APIs.
- [ ] Add one calibrated native significance mode and null-calibration tests.
- [ ] Add effect sizes, rank tests, correlations, and bootstrap confidence intervals.
- [ ] Add configurable APA masks and tidy summary output.
- [ ] Preserve zeros in background inference; separate display transforms from modeling.
- [ ] Verify CPB scaling and add a compatibility-preserving correction if warranted.
- [ ] Add method/configuration metadata to all statistical outputs.

Release A should avoid new mandatory dependencies beyond the existing NumPy/SciPy-style stack where feasible.

### Release B: biological replication and reproducibility

- [ ] Add a validated sample/design-table parser.
- [ ] Add one replicate-aware differential backend behind a stable interface.
- [ ] Support batch/block covariates, offsets, and condition-by-region contrasts.
- [ ] Add cluster bootstrap for shared promoters/enhancers/components.
- [ ] Add matched/weighted control construction with balance diagnostics.
- [ ] Add distance-matched random-shift APA controls and expected normalization.
- [ ] Add `P(s)`, replicate concordance, and deterministic depth/saturation QC.
- [ ] Add simulations for type-I error, power, zero inflation, and overdispersion.

### Release C: interoperability and advanced analyses

- [ ] Add FitHiC2, MaxHiC, and HiC-DC+ import/export adapters.
- [ ] Add generic BEDPE loop imports for Mustache, Peakachu, Chromosight, and HiCCUPS.
- [ ] Benchmark native and external significance methods.
- [ ] Add an experimental AbLE-compatible local-quantification module.
- [ ] Add optional ABC scoring.
- [ ] Add optional promoter-centric graph and stripe summaries.

## Benchmark plan

Use at least the current Gasperini K562 functional enhancer–promoter set and the mESC treatment data already associated with the project, subject to their licensing and provenance requirements.

### Methods to compare

- current legacy local-decay/Fisher output;
- corrected native local binomial or empirical-null model;
- FitHiC2;
- MaxHiC;
- HiC-DC+;
- any later native replicate-aware model.

### Null calibration

Construct held-out negative sets using chromosome-, distance-, and coverage-matched shifted or random pairs. Evaluate:

- p-value histograms and QQ plots;
- empirical type-I error at several nominal thresholds;
- empirical false-discovery proportion where labels permit;
- calibration by distance and coverage stratum;
- failure modes near zero counts and chromosome boundaries.

### Functional prediction

With positive and negative functional labels, evaluate:

- area under the precision–recall curve as the primary metric for imbalanced outcomes;
- ROC AUC as a secondary metric;
- calibration of scores or probabilities when claimed;
- chromosome-held-out validation;
- matched evaluation controlling distance, accessibility, transcription, and coverage;
- uncertainty across held-out chromosomes or biological samples.

### Reproducibility and computational performance

Evaluate:

- replicate rank correlation and distance-stratified concordance;
- significant-set stability at matched sequencing depth;
- APA score and confidence-interval stability under downsampling;
- runtime, peak memory, and disk footprint;
- deterministic results for fixed seeds;
- scalability with number of pairs, chromosomes, and samples.

Do not select the default method based on the number of calls. Prefer calibrated error control, biological reproducibility, robustness to depth, and computational tractability.

## Testing and acceptance criteria

### Unit tests

- BH correction matches a trusted implementation, including ties, NaNs, and empty inputs.
- Pair IDs are deterministic under chunking and parallel execution.
- Coordinate conversion and anchor swapping are lossless and documented.
- Zero counts survive extraction, modeling, serialization, and plotting.
- APA masks select exactly the intended pixels for odd and even window sizes.
- Cluster bootstrap samples whole clusters rather than rows or pixels.
- Seeds reproduce null pairs, bootstrap intervals, and downsampling results.
- Scaling tests distinguish historical and corrected CPB behavior.

### Simulation tests

- Under the declared null, p-values are approximately uniform after stratifying by distance and coverage.
- Type-I error is controlled at declared levels.
- BH controls empirical FDR in representative simulations.
- Differential models recover known condition-by-region interactions.
- Overdispersion and zero inflation do not produce systematic anticonservative results.
- Unreplicated mode emits an explicit technical-inference warning.

### Integration tests

- Existing commands reproduce historical outputs when compatibility flags are used.
- A BEDPE pair list yields consistent `pair_id` values across local decay, background, APA, and export.
- External calls round-trip without changing coordinates or significance fields.
- Replicates remain separate from input through final result.
- Chunked and non-chunked execution agree within numerical tolerances.

### Documentation acceptance

For every inferential method, documentation must state:

- the null and alternative hypotheses;
- the biological or technical unit of replication;
- the data distribution and independence assumptions;
- how expected contact and offsets are obtained;
- the multiple-testing family;
- the meaning of effect sizes and confidence intervals;
- recommended and minimum replication;
- known failure modes;
- whether the result is descriptive, technical-sampling, or biological inference.

## Non-goals and guardrails

- Do not implement a new native general-purpose loop caller in the near term.
- Do not silently pool biological replicates.
- Do not use pair count or pixel count as the replicate count.
- Do not discard zero-containing pairs from inferential datasets by default.
- Do not describe a score as a probability unless it is calibrated and validated as one.
- Do not claim simultaneous multiway enhancer hubs from bulk pairwise matrices.
- Do not add capture-specific CHiCAGO/Chicdiff logic to the core unless supported assays and data contracts explicitly expand.
- Do not add single-cell-specific models to the bulk analysis path without a separate design.
- Do not change historical numerical behavior silently; add named compatibility modes and versioned metadata.
- Do not make heavy R/Python modeling packages mandatory for users who only need the high-performance core.

## Literature and software references

### Directly relevant statistics and enhancer–promoter analysis

1. [Barshad et al. (2023), Touché-associated enhancer–promoter contact analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10714922/) — precedent for rank-based comparisons, correlations, bootstrap confidence intervals, and decomposing focal versus stripe-like contact structure.
2. [FitHiC2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7451401/) — distance-aware spline modeling, contact significance, and multiple-testing correction.
3. [diffHic](https://link.springer.com/article/10.1186/s12859-015-0683-0) — replicate-aware differential interaction analysis using count-model methodology.
4. [multiHiCcompare](https://academic.oup.com/bioinformatics/article/35/17/2916/5298730) — joint normalization and differential comparison across Hi-C datasets with genomic-distance awareness.
5. [MaxHiC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9262194/) — significant-interaction detection applicable to high-resolution Hi-C/Micro-C and enhancer–promoter evaluation.
6. [HiC-DC+](https://www.nature.com/articles/s41467-021-23749-x) — statistically modeled significant interactions with distance and bias covariates.

### APA, reproducibility, and quantitative contact interpretation

7. [coolpup.py](https://academic.oup.com/bioinformatics/article/36/10/2980/5719023) — scalable pileups, expected normalization, and control construction.
8. [HiCRep](https://genome.cshlp.org/content/27/11/1939) — stratum-adjusted correlation for Hi-C reproducibility.
9. [AbLE](https://www.nature.com/articles/s41594-026-01819-2) and [authors' analysis code](https://github.com/ahansenlab/AbsQuant_analysis_code) — local 2D loop quantification and calibrated absolute-contact interpretation.
10. [Activity-by-Contact model](https://www.nature.com/articles/s41588-019-0538-0) — combining enhancer activity and contact to prioritize enhancer–gene links.
11. [Promoter–enhancer stripe analysis](https://www.nature.com/articles/s41588-025-02247-6) — precedent for promoter/enhancer stripe structure beyond focal loop dots.

### Loop-call interoperability

12. [Mustache](https://pmc.ncbi.nlm.nih.gov/articles/PMC7528378/) — multi-scale loop detection.
13. [Peakachu](https://www.nature.com/articles/s41467-020-17239-9) — supervised chromatin-loop detection.
14. [Chromosight](https://www.nature.com/articles/s41467-020-19562-7) — pattern-based detection of chromatin features.

### Assay-specific methods to keep optional or deferred

15. [CHiCAGO](https://link.springer.com/article/10.1186/s13059-016-0992-2) — capture Hi-C interaction calling.
16. [Chicdiff](https://academic.oup.com/bioinformatics/article/35/22/4764/5514042) — differential analysis for capture Hi-C.

## Definition of done for the first implementation milestone

Release A is complete when a user can provide a stable explicit pair list, obtain raw effect estimates plus calibrated nominal p-values and BH-adjusted q-values, receive bootstrap uncertainty with a declared cluster unit, retain zero-containing pairs, generate quantitative APA mask summaries, and reproduce historical behavior through explicit compatibility options. All outputs must carry sufficient metadata to identify the statistical method, test universe, assumptions, and software version.

