"""Group comparisons, correlations, and covariate matching for pair-level tables.

Public API: `compare_groups` (effect size + rank test + clustered bootstrap
interval between two labelled groups of pairs), `compare_paired`
(matched-pairs version), `correlate` (Pearson/Spearman with a clustered
interval), `match_pairs` (covariate-matched control construction), and
`balance_table` (standardized-difference diagnostics before and after
matching).

The unit of analysis here is the bait-prey pair, and that is a deliberate
limit. Pairs sharing a promoter, sharing an enhancer, or sitting in the same
locus are not independent observations, so every function takes a
`cluster_by` column and resamples whole clusters for its confidence
interval. The nominal rank-test p-values do *not* get that treatment -- they
assume independent rows, which is why every result from this module is
labelled `descriptive`. A pair-level rank test is not a biological-replicate
test no matter how small its p-value; for that, the unit must be the library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from touche.metadata import StatResult, method_info
from touche.stats import (
    bootstrap_ci,
    bootstrap_difference_ci,
    cliffs_delta,
    median_difference,
    rank_biserial,
)

STATISTICS = {"median": np.median, "mean": np.mean}


def compare_groups(
    table: pl.DataFrame,
    *,
    value_col: str,
    group_col: str,
    groups: tuple[str, str] | None = None,
    cluster_by: str | None = None,
    statistic: str = "median",
    alternative: str = "two-sided",
    bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> StatResult:
    """Compare `value_col` between two levels of `group_col`.

    Returns a one-row table with both group summaries, the difference in
    `statistic` with a bootstrap confidence interval, Cliff's delta as a
    rank effect size, and a two-sample Mann-Whitney U p-value.

    `groups` picks which two levels to compare (defaults to the two most
    frequent, in sorted order); other levels are reported as filtered rather
    than silently pooled. `cluster_by` names the column holding the
    resampling unit -- `promoter`/`bait_id` when many enhancers share a
    promoter, `enhancer`/`prey_id` when one enhancer serves many promoters.
    Without it the interval treats every pair as independent evidence and
    will be too narrow whenever they are not.

    Non-finite values (including the infinities a zero-denominator log ratio
    produces) are excluded from the statistics and counted in the metadata,
    never silently dropped.
    """

    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of: {', '.join(sorted(STATISTICS))}")
    _require_columns(table, [value_col, group_col] + ([cluster_by] if cluster_by else []))
    left, right = _resolve_groups(table, group_col, groups)

    parts = {name: table.filter(pl.col(group_col) == name) for name in (left, right)}
    values = {name: part[value_col].cast(pl.Float64).to_numpy() for name, part in parts.items()}
    clusters = {
        name: (part[cluster_by].to_numpy() if cluster_by else None) for name, part in parts.items()
    }
    finite = {name: np.isfinite(array) for name, array in values.items()}

    func = STATISTICS[statistic]
    interval = bootstrap_difference_ci(
        values[left],
        values[right],
        statistic=func,
        x_clusters=clusters[left],
        y_clusters=clusters[right],
        n_resamples=bootstrap,
        confidence=confidence,
        seed=seed,
    )
    test = _mannwhitney(values[left][finite[left]], values[right][finite[right]], alternative)

    row = {
        "group_x": left,
        "group_y": right,
        "value_col": value_col,
        "statistic": statistic,
        f"{statistic}_x": _summary(func, values[left]),
        f"{statistic}_y": _summary(func, values[right]),
        "difference": interval["estimate"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "median_difference": median_difference(values[left], values[right]),
        "cliffs_delta": cliffs_delta(values[left], values[right]),
        "u_statistic": test["statistic"],
        "p_value": test["p_value"],
        "n_x": int(finite[left].sum()),
        "n_y": int(finite[right].sum()),
        "n_clusters_x": interval["n_clusters_x"],
        "n_clusters_y": interval["n_clusters_y"],
        "n_zero_x": _count_zeros(values[left]),
        "n_zero_y": _count_zeros(values[right]),
    }
    filtered = {
        "non_finite_x": int((~finite[left]).sum()),
        "non_finite_y": int((~finite[right]).sum()),
        "other_groups": int(table.height - parts[left].height - parts[right].height),
    }
    info = method_info(
        "mannwhitney",
        alternative=alternative,
        cluster_unit=cluster_by or "row (pairs treated as independent)",
        n_input=table.height,
        n_tested=row["n_x"] + row["n_y"],
        seed=seed,
        parameters={
            "value_col": value_col,
            "group_col": group_col,
            "statistic": statistic,
            "bootstrap": bootstrap,
            "confidence": confidence,
        },
        filtered=filtered,
        warnings=_cluster_warnings(cluster_by, row["n_clusters_x"], row["n_clusters_y"]),
    )
    return StatResult(table=pl.DataFrame([row]), info=info)


def compare_paired(
    table: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    cluster_by: str | None = None,
    bootstrap: int = 1000,
    confidence: float = 0.95,
    alternative: str = "two-sided",
    seed: int = 0,
) -> StatResult:
    """Wilcoxon signed-rank comparison of two measurements on the same pairs.

    Use this only when rows are genuinely matched -- the same bait-prey pair
    measured in two conditions, for instance. Two independent groups of pairs
    that merely have the same length are not paired; use `compare_groups`.
    Rows where either value is non-finite are dropped pairwise and reported.
    """

    _require_columns(table, [x_col, y_col] + ([cluster_by] if cluster_by else []))
    x = table[x_col].cast(pl.Float64).to_numpy()
    y = table[y_col].cast(pl.Float64).to_numpy()
    usable = np.isfinite(x) & np.isfinite(y)
    differences = x[usable] - y[usable]
    clusters = table[cluster_by].to_numpy()[usable] if cluster_by else None

    interval = bootstrap_ci(
        differences,
        statistic=np.median,
        clusters=clusters,
        n_resamples=bootstrap,
        confidence=confidence,
        seed=seed,
    )
    test = _wilcoxon(differences, alternative)
    row = {
        "x_col": x_col,
        "y_col": y_col,
        "median_x": _summary(np.median, x[usable]),
        "median_y": _summary(np.median, y[usable]),
        "median_difference": interval["estimate"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "rank_biserial": rank_biserial(differences),
        "w_statistic": test["statistic"],
        "p_value": test["p_value"],
        "n_pairs": int(usable.sum()),
        "n_clusters": interval["n_clusters"],
        "n_ties": int((differences == 0).sum()),
    }
    info = method_info(
        "wilcoxon",
        alternative=alternative,
        unit_of_analysis="matched pair of observations",
        cluster_unit=cluster_by or "row (pairs treated as independent)",
        n_input=table.height,
        n_tested=row["n_pairs"],
        seed=seed,
        parameters={"x_col": x_col, "y_col": y_col, "bootstrap": bootstrap, "confidence": confidence},
        filtered={"non_finite": int((~usable).sum())},
        warnings=_cluster_warnings(cluster_by, row["n_clusters"], row["n_clusters"]),
    )
    return StatResult(table=pl.DataFrame([row]), info=info)


def correlate(
    table: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    method: str = "spearman",
    cluster_by: str | None = None,
    bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> StatResult:
    """Pearson or Spearman correlation with a clustered bootstrap interval.

    The reported p-value is the standard one, which assumes independent
    observations; the bootstrap interval is the part that respects
    `cluster_by`. When the two disagree in width, trust the interval.
    """

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be one of: pearson, spearman")
    _require_columns(table, [x_col, y_col] + ([cluster_by] if cluster_by else []))
    from scipy.stats import pearsonr, spearmanr

    x = table[x_col].cast(pl.Float64).to_numpy()
    y = table[y_col].cast(pl.Float64).to_numpy()
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    clusters = table[cluster_by].to_numpy()[usable] if cluster_by else None

    if x.size < 3:
        estimate, p_value = float("nan"), float("nan")
    elif method == "pearson":
        result = pearsonr(x, y)
        estimate, p_value = float(result.statistic), float(result.pvalue)
    else:
        result = spearmanr(x, y)
        estimate, p_value = float(result.statistic), float(result.pvalue)

    # Bootstrap the correlation itself by resampling index positions (as
    # clusters of rows), so x and y stay paired inside each resample.
    positions = np.arange(x.size, dtype=np.float64)
    correlation = (
        (lambda picked: _correlation_of(x, y, picked.astype(np.int64), method)) if x.size >= 3 else None
    )
    interval = (
        bootstrap_ci(
            positions,
            statistic=correlation,
            clusters=clusters,
            n_resamples=bootstrap,
            confidence=confidence,
            seed=seed,
        )
        if correlation is not None
        else {"ci_low": float("nan"), "ci_high": float("nan"), "n_clusters": 0}
    )

    row = {
        "x_col": x_col,
        "y_col": y_col,
        "correlation": estimate,
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "p_value": p_value,
        "n": int(x.size),
        "n_clusters": interval["n_clusters"],
    }
    info = method_info(
        method,
        alternative="two-sided",
        cluster_unit=cluster_by or "row (pairs treated as independent)",
        n_input=table.height,
        n_tested=int(x.size),
        seed=seed,
        parameters={"x_col": x_col, "y_col": y_col, "bootstrap": bootstrap, "confidence": confidence},
        filtered={"non_finite": int((~usable).sum())},
    )
    return StatResult(table=pl.DataFrame([row]), info=info)


def match_pairs(
    table: pl.DataFrame,
    *,
    group_col: str,
    covariates: dict[str, float],
    groups: tuple[str, str] | None = None,
    ratio: int = 1,
    seed: int = 0,
) -> pl.DataFrame:
    """Nearest-neighbour match control pairs to case pairs on `covariates`.

    `covariates` maps a column name to a caliper -- the maximum absolute
    difference allowed on that covariate for a match. Use log distance rather
    than raw distance: enhancer-promoter contact frequency is roughly linear
    in log distance, so a caliper of 5 kb means something very different at
    20 kb than at 500 kb.

    Matching is greedy, without replacement, in a seeded random case order,
    and a case with no control inside every caliper is dropped rather than
    matched to a distant one. The result adds `match_id` (shared by a case
    and its controls) and `match_role`; feed it to `balance_table` to check
    the covariates actually balanced before interpreting anything.
    """

    if ratio < 1:
        raise ValueError("ratio must be positive")
    if not covariates:
        raise ValueError("At least one covariate with a caliper is required")
    _require_columns(table, [group_col, *covariates])
    case_label, control_label = _resolve_groups(table, group_col, groups)

    names = list(covariates)
    calipers = np.array([covariates[name] for name in names], dtype=np.float64)
    if np.any(calipers <= 0):
        raise ValueError("Every caliper must be positive")

    cases = table.filter(pl.col(group_col) == case_label).with_row_index("_row")
    controls = table.filter(pl.col(group_col) == control_label).with_row_index("_row")
    case_values = cases.select(names).to_numpy().astype(np.float64)
    control_values = controls.select(names).to_numpy().astype(np.float64)

    available = np.ones(control_values.shape[0], dtype=bool)
    order = np.random.default_rng(seed).permutation(case_values.shape[0])
    matched_cases: list[int] = []
    matched_controls: list[int] = []
    case_ids: list[int] = []
    control_ids: list[int] = []
    for match_id, case_index in enumerate(order):
        deltas = np.abs(control_values - case_values[case_index])
        eligible = np.flatnonzero(available & np.all(deltas <= calipers, axis=1))
        if eligible.size < ratio:
            continue
        # Rank eligible controls by caliper-scaled distance so "nearest" means
        # the same thing whether a covariate is measured in bases or in reads.
        scores = np.sum(deltas[eligible] / calipers, axis=1)
        picked = eligible[np.argsort(scores, kind="mergesort")[:ratio]]
        available[picked] = False
        matched_cases.append(int(case_index))
        matched_controls.extend(int(index) for index in picked)
        case_ids.append(match_id)
        control_ids.extend([match_id] * ratio)

    annotated = table.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("match_id"), pl.lit(None, dtype=pl.Utf8).alias("match_role")
    )
    if not matched_cases:
        return annotated.clear()
    case_rows = cases[matched_cases].with_columns(
        pl.Series("match_id", case_ids, dtype=pl.Int64), pl.lit("case").alias("match_role")
    )
    control_rows = controls[matched_controls].with_columns(
        pl.Series("match_id", control_ids, dtype=pl.Int64), pl.lit("control").alias("match_role")
    )
    return pl.concat([case_rows, control_rows]).drop("_row").sort(["match_id", "match_role"])


def balance_table(
    table: pl.DataFrame, *, group_col: str, covariates: list[str], groups: tuple[str, str] | None = None
) -> pl.DataFrame:
    """Standardized mean difference per covariate between two groups.

    Report this before and after `match_pairs`. A standardized difference
    above roughly 0.1 in absolute value is the conventional signal that a
    covariate is still imbalanced and can still confound the comparison.
    """

    _require_columns(table, [group_col, *covariates])
    left, right = _resolve_groups(table, group_col, groups)
    rows = []
    for name in covariates:
        x = _finite_column(table, group_col, left, name)
        y = _finite_column(table, group_col, right, name)
        pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2) if x.size > 1 and y.size > 1 else np.nan
        rows.append(
            {
                "covariate": name,
                "group_x": left,
                "group_y": right,
                "mean_x": float(np.mean(x)) if x.size else float("nan"),
                "mean_y": float(np.mean(y)) if y.size else float("nan"),
                "n_x": int(x.size),
                "n_y": int(y.size),
                "standardized_difference": (
                    float((np.mean(x) - np.mean(y)) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else float("nan")
                ),
            }
        )
    return pl.DataFrame(rows)


def _cluster_warnings(cluster_by: str | None, n_clusters_x: int, n_clusters_y: int) -> list[str]:
    """Flag the two ways a clustered interval silently stops being trustworthy."""
    warnings: list[str] = []
    if cluster_by is None:
        warnings.append(
            "No cluster_by given: the bootstrap interval treats every pair as an independent "
            "observation. Pairs sharing a promoter, enhancer, or locus are not independent, so "
            "the interval is likely too narrow. This is a descriptive comparison, not a "
            "biological-replicate test."
        )
    elif min(n_clusters_x, n_clusters_y) < 10:
        warnings.append(
            f"Only {min(n_clusters_x, n_clusters_y)} clusters in the smaller group; a cluster "
            "bootstrap with this few independent units gives an unreliable interval."
        )
    return warnings


def _require_columns(table: pl.DataFrame, columns: list[str]) -> None:
    """Raise naming every requested column the frame is missing, not just the first."""
    missing = [name for name in columns if name not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _resolve_groups(
    table: pl.DataFrame, group_col: str, groups: tuple[str, str] | None
) -> tuple[str, str]:
    """Pick the two group levels to compare, defaulting to the two most populated."""
    if groups is not None:
        present = set(table[group_col].unique().to_list())
        missing = [name for name in groups if name not in present]
        if missing:
            raise ValueError(f"{group_col!r} has no rows for: {', '.join(map(str, missing))}")
        return groups
    counts = table[group_col].value_counts().sort(["count", group_col], descending=[True, False])
    levels = counts[group_col].to_list()
    if len(levels) < 2:
        raise ValueError(f"{group_col!r} needs at least two levels to compare, found {len(levels)}")
    return tuple(sorted(levels[:2]))


def _summary(func: Any, values: np.ndarray) -> float:
    """Apply a summary statistic to the finite entries, NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(func(finite)) if finite.size else float("nan")


def _count_zeros(values: np.ndarray) -> int:
    """How many values are exactly zero -- reported so complete-absence pairs stay visible."""
    return int(np.sum(values == 0))


def _finite_column(table: pl.DataFrame, group_col: str, group: str, column: str) -> np.ndarray:
    """Finite values of `column` for one level of `group_col`."""
    values = table.filter(pl.col(group_col) == group)[column].cast(pl.Float64).to_numpy()
    return values[np.isfinite(values)]


def _correlation_of(x: np.ndarray, y: np.ndarray, picked: np.ndarray, method: str) -> float:
    """Correlation of a resampled index selection, used as the bootstrap statistic."""
    from scipy.stats import pearsonr, spearmanr

    if picked.size < 3:
        return float("nan")
    sample_x, sample_y = x[picked], y[picked]
    if np.all(sample_x == sample_x[0]) or np.all(sample_y == sample_y[0]):
        return float("nan")
    func = pearsonr if method == "pearson" else spearmanr
    return float(func(sample_x, sample_y).statistic)


def _mannwhitney(x: np.ndarray, y: np.ndarray, alternative: str) -> dict[str, float]:
    """Two-sample Mann-Whitney U, returning NaN rather than raising on an empty group."""
    from scipy.stats import mannwhitneyu

    if x.size == 0 or y.size == 0:
        return {"statistic": float("nan"), "p_value": float("nan")}
    result = mannwhitneyu(x, y, alternative=alternative)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def _wilcoxon(differences: np.ndarray, alternative: str) -> dict[str, float]:
    """Wilcoxon signed-rank test, returning NaN when every difference is an exact tie."""
    from scipy.stats import wilcoxon

    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return {"statistic": float("nan"), "p_value": float("nan")}
    result = wilcoxon(nonzero, alternative=alternative)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
