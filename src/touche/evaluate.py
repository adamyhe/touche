"""Ranking-quality evaluation of contact scores against functional labels.

Public API: `roc_auc`, `average_precision`, `precision_recall_curve` (pure
numpy, tie-correct), and `evaluate_scores` (compare several score columns on
one labelled pair table, optionally held out by chromosome).

How imbalanced the evaluation is depends on a design choice, and it changes
which metric to lead with -- so `evaluate_scores` reports both and the
no-skill AUPRC baseline alongside them.

`evaluate_scores` evaluates only *labelled* rows. A CRISPRi screen tests
specific pairs, so an unlabelled candidate means "never tested", not
"non-functional"; calling it a negative would invent data. On the
Gasperini K562 set that leaves 623 functional against 745 non-functional --
near balanced, with an AUPRC no-skill line around 0.46, and ROC AUC
perfectly readable.

Treat the untested candidates as negatives instead and the same set becomes
623 positives in 12,801 pairs, about 5% prevalence. *There* AUPRC is
clearly primary and ROC AUC misleads, because it is dominated by an
enormous negative class and stays high for a score that is useless at the
top of the ranking.

Always read `auprc` against `baseline_auprc` in the same row rather than
against any remembered number.

Two things this module insists on, because both are easy to get wrong and
both flatter a method that does not deserve it:

- **Untestable pairs are not dropped by default.** A method that returns NaN
  for the pairs it cannot handle would otherwise be evaluated on an easier
  subset than its competitors. `nan_policy="worst"` ranks them last, which
  is what "no evidence" means, and keeps every method on the same rows.
- **Baselines belong in the comparison.** Genomic distance alone predicts
  functional enhancer-promoter pairs well. A contact score that does not
  beat `-distance` has not been shown to add anything, so
  `evaluate_scores` is designed to score several columns at once and is
  meant to be given the trivial baselines alongside the real methods.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from touche.metadata import StatResult, method_info
from touche.stats import bootstrap_ci

NAN_POLICIES = {"worst", "drop"}


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed from ranks so ties are handled exactly.

    Equivalent to the probability that a randomly chosen positive outranks a
    randomly chosen negative, with ties counting a half. Returns NaN when
    either class is empty, since the quantity is undefined there.
    """

    labels, scores = _finite_pairs(labels, scores)
    positives = labels.sum()
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, and threshold arrays, with tied scores collapsed to one point.

    Collapsing ties matters: a score with many equal values (a discrete
    p-value that saturates, say) cannot separate the pairs inside a tie, and
    treating them as separately rankable would credit it with precision it
    cannot deliver.
    """

    labels, scores = _finite_pairs(labels, scores)
    if labels.size == 0 or labels.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    # One cut point per distinct score, at the last index of each tie group.
    distinct = np.flatnonzero(np.diff(sorted_scores)) if sorted_scores.size > 1 else np.array([], dtype=int)
    cuts = np.append(distinct, sorted_labels.size - 1)

    true_positives = np.cumsum(sorted_labels)[cuts]
    predicted = cuts + 1
    return true_positives / predicted, true_positives / labels.sum(), sorted_scores[cuts]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: the step-wise area under the precision-recall curve.

    Sums `(R_n - R_{n-1}) * P_n` over thresholds, matching
    `sklearn.metrics.average_precision_score`. This is the primary metric for
    imbalanced functional labels; its no-skill baseline is the positive-class
    prevalence, so always read it against that rather than against 0.5.
    """

    precision, recall, _ = precision_recall_curve(labels, scores)
    if precision.size == 0:
        return float("nan")
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def evaluate_scores(
    table: pl.DataFrame,
    *,
    label_col: str,
    score_cols: list[str],
    positive_label: object = True,
    held_out_col: str | None = None,
    nan_policy: str = "worst",
    min_group_size: int = 20,
    bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> StatResult:
    """Compare several score columns at discriminating labelled pairs.

    Returns one row per entry in `score_cols` with AUPRC, ROC AUC, the
    no-skill AUPRC baseline (the prevalence), and the counts behind them.

    `held_out_col` (typically `"chrom"`) additionally evaluates each group
    separately and reports the mean across groups with a bootstrap interval
    over them. That is the honest uncertainty for this design: pairs within
    a chromosome share local chromatin structure and are not independent, so
    an interval computed over pairs would be far too narrow. Groups with
    fewer than `min_group_size` labelled pairs, or with only one class, are
    skipped and counted.

    Rows whose label is neither `positive_label` nor a valid negative are
    excluded -- in practice the `"other"` class of unlabelled candidates.
    """

    if nan_policy not in NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of: {', '.join(sorted(NAN_POLICIES))}")
    missing = [name for name in [label_col, *score_cols] if name not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if held_out_col is not None and held_out_col not in table.columns:
        raise ValueError(f"held_out_col {held_out_col!r} is not a column of the input table")

    labelled = table.filter(pl.col(label_col).is_not_null())
    labels = (labelled[label_col] == positive_label).to_numpy()
    rows = []
    for name in score_cols:
        scores = labelled[name].cast(pl.Float64).to_numpy()
        row = _score_row(name, labels, scores, nan_policy)
        if held_out_col is not None:
            row.update(
                _held_out_summary(
                    labelled, labels, scores, held_out_col, nan_policy,
                    min_group_size=min_group_size, bootstrap=bootstrap,
                    confidence=confidence, seed=seed,
                )
            )
        rows.append(row)

    result = pl.DataFrame(rows)
    prevalence = float(labels.mean()) if labels.size else float("nan")
    warnings = []
    if labels.sum() < 20:
        warnings.append(
            f"Only {int(labels.sum())} positive pairs; AUPRC is very noisy at this count and "
            "differences between methods should not be read as meaningful."
        )
    if held_out_col is not None:
        groups_used = min(int(row["n_groups"]) for row in rows if row.get("n_groups") is not None)
        groups_skipped = max(int(row["n_groups_skipped"]) for row in rows if row.get("n_groups_skipped") is not None)
        if groups_used < 10:
            warnings.append(
                f"The held-out bootstrap resampled only {groups_used} groups; intervals from this "
                "few independent units are wide and unreliable, and differences between scores "
                "inside them are not evidence. Lower min_group_size or pool groups."
            )
        if groups_skipped >= groups_used:
            warnings.append(
                f"{groups_skipped} of {groups_used + groups_skipped} groups were skipped for "
                f"having fewer than {min_group_size} labelled pairs or only one class, so the "
                "held-out mean describes the better-covered groups rather than the whole "
                "dataset."
            )
    if held_out_col is None:
        warnings.append(
            "No held_out_col: metrics are computed over all pairs pooled, with no uncertainty. "
            "Pairs within a chromosome are not independent, so a pooled point estimate "
            "overstates how reproducible the ranking is."
        )
    info = method_info(
        "ranking_evaluation",
        alternative="two-sided",
        unit_of_analysis="labelled bait-prey pair",
        cluster_unit=held_out_col,
        n_input=table.height,
        n_tested=int(labels.size),
        seed=seed,
        parameters={
            "label_col": label_col,
            "score_cols": list(score_cols),
            "positive_label": str(positive_label),
            "held_out_col": held_out_col,
            "nan_policy": nan_policy,
            "prevalence": prevalence,
            "n_positive": int(labels.sum()),
            "n_negative": int(labels.size - labels.sum()),
            "bootstrap": bootstrap,
        },
        warnings=warnings,
    )
    return StatResult(table=result, info=info)


def _finite_pairs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean labels and float scores restricted to rows where the score is finite."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    usable = np.isfinite(scores)
    return labels[usable], scores[usable]


def _apply_nan_policy(scores: np.ndarray, nan_policy: str) -> np.ndarray:
    """Rank non-finite scores last, or leave them to be dropped downstream."""
    if nan_policy == "drop":
        return scores
    scores = np.asarray(scores, dtype=np.float64).copy()
    unusable = ~np.isfinite(scores)
    if unusable.any():
        finite = scores[~unusable]
        # Strictly below every real score, so an untestable pair never
        # outranks a tested one and never ties with the worst of them.
        scores[unusable] = (finite.min() - 1.0) if finite.size else 0.0
    return scores


def _score_row(name: str, labels: np.ndarray, scores: np.ndarray, nan_policy: str) -> dict[str, object]:
    """Pooled AUPRC/AUROC for one score column, plus the counts they rest on."""
    ranked = _apply_nan_policy(scores, nan_policy)
    usable = np.isfinite(ranked)
    return {
        "score": name,
        "auprc": average_precision(labels[usable], ranked[usable]),
        "auroc": roc_auc(labels[usable], ranked[usable]),
        "baseline_auprc": float(labels[usable].mean()) if usable.any() else float("nan"),
        "n": int(usable.sum()),
        "n_positive": int(labels[usable].sum()),
        "n_missing_score": int((~np.isfinite(scores)).sum()),
        "nan_policy": nan_policy,
    }


def _held_out_summary(
    labelled: pl.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    held_out_col: str,
    nan_policy: str,
    *,
    min_group_size: int,
    bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Per-group AUPRC averaged across held-out groups, with a bootstrap interval over groups."""
    groups = labelled[held_out_col].to_numpy()
    ranked = _apply_nan_policy(scores, nan_policy)
    per_group = []
    skipped = 0
    for group in np.unique(groups):
        selected = groups == group
        group_labels = labels[selected]
        if group_labels.size < min_group_size or group_labels.sum() == 0 or group_labels.all():
            skipped += 1
            continue
        usable = np.isfinite(ranked[selected])
        per_group.append(average_precision(group_labels[usable], ranked[selected][usable]))

    values = np.array(per_group, dtype=np.float64)
    interval = bootstrap_ci(values, statistic=np.mean, n_resamples=bootstrap, confidence=confidence, seed=seed)
    return {
        "held_out_auprc_mean": interval["estimate"],
        "held_out_ci_low": interval["ci_low"],
        "held_out_ci_high": interval["ci_high"],
        "n_groups": int(values.size),
        "n_groups_skipped": skipped,
    }
