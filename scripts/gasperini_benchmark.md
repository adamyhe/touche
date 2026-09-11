# Gasperini benchmark

Does a calibrated test rank validated enhancer–promoter pairs better than
the reference workflow's Fisher score — and does either beat the trivial
baselines? This script answers that on the Gasperini K562 functional set,
the dataset the reference workflow itself ships with.

It exists because `local-decay call` was switched to `--method binomial` on
statistical-validity grounds without downstream evidence. This is the
evidence.

## Running it

```bash
# Synthetic data with a planted signal. Seconds, no downloads.
uv run python scripts/gasperini_benchmark.py --demo

# The real thing. Downloads several GB on a first run.
uv run python scripts/gasperini_benchmark.py \
  --data-dir benchmark/gasperini/data \
  --work-dir benchmark/gasperini/work \
  --out-dir  benchmark/gasperini/report
```

Useful flags: `--skip-download` (fail instead of fetching), `--jobs`,
`--shifts` (the null offsets), `--null-exclusion`, `--bootstrap`,
`--no-plots`. `--help` lists them all.

Everything is importable, so a notebook can drive the stages directly:

```python
import sys; sys.path.insert(0, "scripts")
from gasperini_benchmark import build_demo_inputs, call_tidy, score_all_methods, attach_labels
```

## What it measures

**Null calibration.** Both anchor sets are translated by the same offset,
which preserves every pair's genomic distance exactly while moving it off
its real locus. A correctly specified test produces uniform p-values on
those pairs. Reported as a KS statistic against `Uniform(0, 1)` plus the
empirical rejection rate at several nominal levels, overall and stratified
by distance and coverage quintiles — a method can look uniform overall and
be badly anticonservative for exactly the short-range, low-coverage pairs
an enhancer–promoter study cares about.

Shifted pairs whose anchors land back within `--null-exclusion` bases of
real candidate anchors are dropped. Without that step a shift commensurate
with structure in the anchor set can map it onto itself, and the "null"
silently becomes the real data again. The run reports how many were dropped;
if that number is most of them, choose different `--shifts`.

**Is `expected` unbiased?** Every method consumes the same `expected`, so
this table is about the distance-decay model rather than the choice of null.
On null pairs, `observed_over_expected` should be ~1. A ratio of `r` means
the expectation is `r` times too small and any test built on it rejects
roughly `r` times too often. **Read this table before the calibration
table** — it says whether a calibration failure is the test's fault or the
model's.

**Functional prediction.** AUPRC is the primary metric because the labels
are heavily imbalanced; compare it against `baseline_auprc` (the
prevalence), never against 0.5. ROC AUC is secondary and stays deceptively
high for a score that is useless at the top of the ranking.
`held_out_auprc_mean` averages per-chromosome AUPRC with a bootstrap
interval over chromosomes, which is the honest uncertainty here: pairs
within a chromosome share local chromatin structure and are not independent.

**Matched prediction.** The same, after matching positives to negatives on
log distance and log coverage, so a method is not credited for
rediscovering that functional pairs are closer together. `balance.tsv`
gives standardized differences before and after; above ~0.1 means a
covariate is still imbalanced.

## Baselines are the point

`observed` (raw contact count) and `neg_log10_distance` are scored
alongside the methods deliberately. Genomic distance alone predicts
functional enhancer–promoter pairs well. A contact score that does not beat
`neg_log10_distance`, and does not beat raw `observed`, has not been shown
to add anything — no matter how good its p-values look.

Untestable pairs are ranked last rather than dropped (`--nan-policy`
equivalent in `touche.evaluate`), so a method that returns NaN where it
struggles is not quietly evaluated on an easier subset than its rivals.

## Outputs

| File | Contents |
| --- | --- |
| `summary.md` | Headline tables, an automatic calibration verdict, and the caveats |
| `expected_bias.tsv` | `observed_over_expected` per distance stratum |
| `calibration.tsv` | KS and rejection rates per method × stratification |
| `prediction.tsv` | AUPRC/AUROC per score, with held-out-chromosome intervals |
| `prediction_matched.tsv` | The same on distance- and coverage-matched pairs |
| `balance.tsv` | Standardized covariate differences before and after matching |
| `scored_pairs.tsv` | Labelled real pairs with every method's p, q, and score |
| `null_pairs.tsv` | The surviving shifted pairs, scored the same way |
| `null_pvalue_qq.svg` | QQ plot against uniform, per method |
| `precision_recall.svg` | PR curves for every score, with the no-skill line |
| `manifest.json` | Parameters, counts, and output paths |

## Reading the demo

`--demo` is a pipeline check, not evidence about real data. It plants a
modest focal signal in half the candidate pairs and gives every pair the
same anchor separation, so a correct run shows high AUPRC for every contact
score and exactly chance for `neg_log10_distance`. It cannot rank the
methods against each other — the planted signal is too clean and too
uniform for that. Only the real run can.

## Known limitations

- A shifted pair can land on a genuine contact by chance, which inflates
  the extreme tail slightly. The calibration check is therefore mildly
  pessimistic rather than optimistic.
- `p_null` is fitted from the same bait window each pair is tested in. The
  leakage is small but not zero, and this benchmark does not remove it.
- External callers (FitHiC2, MaxHiC, HiC-DC+) are not run here. Import
  their output with `touche pairs import-calls` and join it on with
  `touche pairs annotate` to put them in the same comparison.
- Everything measured is technical inference from one library. It says
  nothing about biological reproducibility across replicates.
