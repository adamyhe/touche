"""Numeric statistical primitives shared by every `touche` inferential module.

Public API: `fisher_greater_batch` (the legacy R-compatible one-sided Fisher
exact test), the calibrated upper-tail tests `binom_sf_greater`/
`poisson_sf_greater`, multiple-testing correction (`adjust_pvalues`,
`adjust_pvalue_column`), effect sizes (`log2_fold_change`,
`log_odds_ratio`, `cliffs_delta`, `rank_biserial`, `median_difference`),
and resampling uncertainty (`bootstrap_ci`, `bootstrap_difference_ci`).

Everything here takes and returns numpy arrays or plain floats -- no schema,
no metadata, no file I/O. The self-describing result envelopes those feed
into live in `touche.metadata`, and the analyses that use them live in
`touche.significance` and `touche.compare`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import binom, hypergeom, poisson

from touche.backends import DEFAULT_FISHER_BACKEND

ADJUST_METHODS = {"bh", "by", "holm", "bonferroni", "none"}
BOOTSTRAP_METHODS = {"percentile", "bca"}


def fisher_greater_batch(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    *,
    backend: str = DEFAULT_FISHER_BACKEND,
) -> np.ndarray:
    """R-compatible one-sided Fisher exact p-values used by legacy local-decay calls.

    The reference implementation rounds values before calling R's
    ``fisher.test(..., alternative = "greater")`` (via rpy2); this function
    replaces that call and keeps the same rounding so results match exactly.

    This backs `method="legacy_fisher"` only. Its 2x2 table treats a fitted
    expected count as an observed cell and the number of one-base distance
    histogram bins as a trial total, which is not the process that generated
    the counts -- see `touche.significance` for the calibrated alternatives
    and `docs/statistics.md` for why the legacy null is not interpretable as
    a calibrated p-value.

    `backend="numba"` (the default) uses a `prange`-parallel hypergeometric
    survival function instead of `scipy.stats.hypergeom.sf`, matching it to
    within ~1e-8 absolute error (see notes/numba-implementation-plan.md) in
    exchange for using more than one core. `backend="scipy"` is exact.
    """

    a = np.round(a1)
    b = np.round(a2)
    c = np.round(b1)
    d = np.round(b2)
    if min(a.min(), b.min(), c.min(), d.min()) < 0:
        raise ValueError("Fisher exact table entries must be non-negative")

    total = a + b + c + d
    row_1 = a + b
    col_1 = a + c
    if backend == "numba":
        from touche.numba.stats import hypergeom_sf_numba

        return hypergeom_sf_numba(
            (a - 1).astype(np.int64),
            total.astype(np.int64),
            row_1.astype(np.int64),
            col_1.astype(np.int64),
        )
    return hypergeom.sf(a - 1, total, row_1, col_1)


def binom_sf_greater(successes: np.ndarray, trials: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Upper-tail binomial p-values, `P(K >= k)` under `K ~ Binomial(n, p)`.

    This is the calibrated per-pair test: `k` is the observed contact count,
    `n` the auditable number of contacts anchored at the bait in the relevant
    direction, and `p` the distance-decay null probability that such a
    contact lands in the prey window. Pairs with `n == 0` have no trials and
    therefore no test; they return NaN rather than a misleading `p = 1`.
    Probabilities are clipped into `[0, 1]` because a smoothed density can
    integrate marginally outside it.
    """

    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    testable = (trials > 0) & np.isfinite(probability) & np.isfinite(successes)
    result = np.full(successes.shape, np.nan, dtype=np.float64)
    if not testable.any():
        return result
    result[testable] = binom.sf(
        successes[testable] - 1, trials[testable], probability[testable]
    )
    return np.clip(result, 0.0, 1.0)


def poisson_sf_greater(successes: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Upper-tail Poisson p-values, `P(K >= k)` under `K ~ Poisson(mu)`.

    The large-`n`, small-`p` limit of `binom_sf_greater`, and the form most
    external contact callers (FitHiC2 among them) report. Pairs with a
    non-positive or non-finite expectation return NaN.
    """

    successes = np.asarray(successes, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    testable = (expected > 0) & np.isfinite(expected) & np.isfinite(successes)
    result = np.full(successes.shape, np.nan, dtype=np.float64)
    if testable.any():
        result[testable] = poisson.sf(successes[testable] - 1, expected[testable])
    return np.clip(result, 0.0, 1.0)


def adjust_pvalues(p_values: np.ndarray, *, method: str = "bh") -> np.ndarray:
    """Multiple-testing correction matching R's `p.adjust`.

    `method` is one of `bh` (Benjamini-Hochberg FDR), `by`
    (Benjamini-Yekutieli, valid under arbitrary dependence), `holm`,
    `bonferroni`, or `none`. NaN p-values are *not* tested: they pass through
    as NaN and are excluded from the family size `m`, so an untestable pair
    cannot inflate the correction applied to the pairs that were tested.
    Tied p-values always receive identical adjusted values.
    """

    if method not in ADJUST_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(ADJUST_METHODS))}")
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    tested = np.flatnonzero(np.isfinite(values))
    m = tested.size
    if m == 0:
        return adjusted
    if method == "none":
        adjusted[tested] = values[tested]
        return adjusted

    finite = values[tested]
    if method == "bonferroni":
        adjusted[tested] = np.minimum(1.0, finite * m)
        return adjusted

    if method == "holm":
        order = np.argsort(finite, kind="mergesort")
        ranked = finite[order] * (m - np.arange(m))
        adjusted[tested[order]] = np.minimum(1.0, np.maximum.accumulate(ranked))
        return adjusted

    # BH/BY: step-up from the largest p-value, so the running minimum enforces
    # monotonicity and gives tied p-values the same q-value by construction.
    scale = 1.0 if method == "bh" else float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(finite, kind="mergesort")[::-1]
    ranks = np.arange(m, 0, -1, dtype=np.float64)
    ranked = finite[order] * scale * m / ranks
    adjusted[tested[order]] = np.minimum(1.0, np.minimum.accumulate(ranked))
    return adjusted


def adjust_pvalue_column(
    table: pl.DataFrame,
    *,
    p_col: str = "p_value",
    q_col: str = "q_value",
    method: str = "bh",
    groupby: str | list[str] | None = None,
) -> pl.DataFrame:
    """Add an adjusted-p column to a polars frame, optionally per stratum.

    `groupby=None` corrects over every row in `table` as one family -- the
    default, and the family a `q_value` means unless documented otherwise.
    Passing `groupby` (e.g. `"sample"`, or `["sample", "chrom"]`) declares a
    stratified family instead: each group is corrected independently, which
    is only valid if the strata were chosen before looking at the p-values.
    The chosen family is recorded by callers in result metadata; this
    function just does the arithmetic.
    """

    if p_col not in table.columns:
        raise ValueError(f"{p_col!r} is not a column of the input table")
    if table.is_empty():
        return table.with_columns(pl.lit(None, dtype=pl.Float64).alias(q_col))
    if groupby is None:
        return table.with_columns(
            pl.Series(q_col, adjust_pvalues(table[p_col].to_numpy(), method=method))
        )
    keys = [groupby] if isinstance(groupby, str) else list(groupby)
    return table.with_columns(
        pl.col(p_col)
        .map_batches(
            lambda values: pl.Series(adjust_pvalues(values.to_numpy(), method=method)),
            return_dtype=pl.Float64,
        )
        .over(keys)
        .alias(q_col)
    )


def log2_fold_change(
    numerator: np.ndarray, denominator: np.ndarray, *, pseudocount: float = 0.0
) -> np.ndarray:
    """`log2((numerator + c) / (denominator + c))`.

    `pseudocount=0` (the default) is the raw estimand: a complete gain is
    `+inf`, a complete loss is `-inf`, and `0/0` is NaN. Those are real,
    interpretable outcomes and are preserved rather than filtered. Pass a
    positive `pseudocount` only to produce a finite value for display, and
    record which one you used -- the ranking it induces among zero-containing
    pairs depends entirely on its size.
    """

    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    numerator = np.asarray(numerator, dtype=np.float64) + pseudocount
    denominator = np.asarray(denominator, dtype=np.float64) + pseudocount
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(numerator / denominator)


def log_odds_ratio(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, *, correction: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Log odds ratio `log((a*d)/(b*c))` and its Woolf standard error.

    Rows are `[[a, b], [c, d]]`, e.g. treatment/control by
    enhancer-promoter/background counts. `correction=0.5` applies the
    Haldane-Anscombe continuity correction, which makes zero-containing
    tables finite and the standard error defined; `correction=0` keeps the
    raw infinite estimate for a complete gain or loss. The standard error is
    always NaN where any raw cell is zero and no correction was applied.
    """

    if correction < 0:
        raise ValueError("correction must be non-negative")
    cells = [np.asarray(x, dtype=np.float64) + correction for x in (a, b, c, d)]
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = np.log((cells[0] * cells[3]) / (cells[1] * cells[2]))
        standard_error = np.sqrt(sum(1.0 / cell for cell in cells))
    return estimate, standard_error


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta, the rank effect size matching a Mann-Whitney U test.

    `+1` means every `x` exceeds every `y`, `-1` the reverse, `0` complete
    overlap. Derived from U so ties are handled the same way the test is.
    """

    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return float("nan")
    from scipy.stats import mannwhitneyu

    statistic = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * statistic / (x.size * y.size) - 1.0)


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the effect size for a signed-rank test.

    Computed from the nonzero paired differences as
    `(W+ - W-) / (W+ + W-)`, so it ranges over `[-1, 1]` and is invariant to
    the number of exact ties, which contribute no signed rank.
    """

    differences = _finite(differences)
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def median_difference(x: np.ndarray, y: np.ndarray) -> float:
    """Difference of medians, `median(x) - median(y)`, ignoring non-finite values."""
    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.median(x) - np.median(y))


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    clusters: np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    method: str = "percentile",
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap confidence interval for a one-sample statistic.

    Pass `clusters` to resample whole clusters with replacement instead of
    individual rows. That is the correct unit whenever rows are not
    independent -- pairs sharing a promoter, pairs sharing an enhancer,
    pixels from one APA window, or windows from one library. Resampling rows
    in those settings understates the interval, sometimes by a lot, because
    it treats correlated rows as independent evidence.

    Returns the point estimate, interval bounds, the number of effective
    independent units (clusters, or rows when unclustered), and the seed, so
    the interval can be reproduced and its replication unit audited.
    """

    if method not in BOOTSTRAP_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(BOOTSTRAP_METHODS))}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    values = np.asarray(values, dtype=np.float64)
    groups = _cluster_groups(values.size, clusters)
    observed = _safe_statistic(statistic, values)
    result: dict[str, Any] = {
        "estimate": observed,
        "n": int(values.size),
        "n_clusters": len(groups),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "bootstrap_method": method,
        "cluster_unit": "row" if clusters is None else "cluster",
        "seed": int(seed),
    }
    if len(groups) < 2 or not np.isfinite(observed):
        result.update(ci_low=float("nan"), ci_high=float("nan"))
        return result

    rng = np.random.default_rng(seed)
    replicates = _bootstrap_replicates(values, groups, statistic, n_resamples, rng)
    alpha = 1.0 - confidence
    if method == "percentile":
        low, high = np.nanpercentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        low, high = _bca_bounds(values, groups, statistic, replicates, observed, alpha)
    result.update(ci_low=float(low), ci_high=float(high))
    return result


def bootstrap_difference_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    x_clusters: np.ndarray | None = None,
    y_clusters: np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap interval for `statistic(x) - statistic(y)`.

    Each group is resampled independently (clusters within group, when
    given), which is the right structure for two independent groups of pairs.
    For genuinely paired observations, bootstrap the paired differences with
    `bootstrap_ci` instead.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_groups = _cluster_groups(x.size, x_clusters)
    y_groups = _cluster_groups(y.size, y_clusters)
    observed = _safe_statistic(statistic, x) - _safe_statistic(statistic, y)
    result: dict[str, Any] = {
        "estimate": observed,
        "n_x": int(x.size),
        "n_y": int(y.size),
        "n_clusters_x": len(x_groups),
        "n_clusters_y": len(y_groups),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "bootstrap_method": "percentile",
        "cluster_unit": "row" if x_clusters is None and y_clusters is None else "cluster",
        "seed": int(seed),
    }
    if len(x_groups) < 2 or len(y_groups) < 2 or not np.isfinite(observed):
        result.update(ci_low=float("nan"), ci_high=float("nan"))
        return result

    rng = np.random.default_rng(seed)
    replicates = _bootstrap_replicates(x, x_groups, statistic, n_resamples, rng) - _bootstrap_replicates(
        y, y_groups, statistic, n_resamples, rng
    )
    alpha = 1.0 - confidence
    low, high = np.nanpercentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    result.update(ci_low=float(low), ci_high=float(high))
    return result


def _finite(values: np.ndarray) -> np.ndarray:
    """Drop non-finite entries; every descriptive statistic here ignores them."""
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _safe_statistic(statistic: Callable[[np.ndarray], float], values: np.ndarray) -> float:
    """Apply `statistic` to the finite entries of `values`, returning NaN when none remain."""
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    return float(statistic(finite))


def _cluster_groups(size: int, clusters: np.ndarray | None) -> list[np.ndarray]:
    """Row indexes grouped by cluster label, or one singleton group per row when unclustered."""
    if clusters is None:
        return [np.array([i]) for i in range(size)]
    clusters = np.asarray(clusters)
    if clusters.shape[0] != size:
        raise ValueError("clusters must have one label per value")
    _, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries) if order.size else []


def _bootstrap_replicates(
    values: np.ndarray,
    groups: list[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample whole clusters with replacement `n_resamples` times and evaluate `statistic`."""
    n_groups = len(groups)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        picked = rng.integers(0, n_groups, n_groups)
        sample = np.concatenate([groups[j] for j in picked])
        replicates[i] = _safe_statistic(statistic, values[sample])
    return replicates


def _bca_bounds(
    values: np.ndarray,
    groups: list[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    replicates: np.ndarray,
    observed: float,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval bounds, jackknifing over whole clusters."""
    from scipy.stats import norm

    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        return float("nan"), float("nan")
    proportion = float(np.mean(finite < observed))
    if proportion <= 0.0 or proportion >= 1.0:
        # Degenerate bias correction; fall back to the percentile interval
        # rather than emitting a bound at +/- infinity.
        return tuple(np.nanpercentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    bias = norm.ppf(proportion)

    all_indexes = np.arange(values.size)
    jackknife = np.array(
        [
            _safe_statistic(statistic, values[np.setdiff1d(all_indexes, group, assume_unique=False)])
            for group in groups
        ]
    )
    centered = np.nanmean(jackknife) - jackknife
    denominator = 6.0 * float(np.nansum(centered**2)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.nansum(centered**3)) / denominator

    quantiles = []
    for tail in (alpha / 2, 1 - alpha / 2):
        z = bias + norm.ppf(tail)
        quantiles.append(norm.cdf(bias + z / (1 - acceleration * z)))
    return tuple(np.nanpercentile(replicates, [100 * q for q in quantiles]))
