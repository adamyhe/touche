from __future__ import annotations

from scipy import stats as scipy_stats


def fisher_greater(a1: float, a2: float, b1: float, b2: float) -> float:
    """R-compatible one-sided Fisher exact p-value used by local decay calls.

    The reference implementation rounds values before calling R's
    ``fisher.test(..., alternative = "greater")`` through rpy2. Keep the
    rounding here so this can replace rpy2 without changing expected behavior.
    """

    table = [
        [round(a1), round(a2)],
        [round(b1), round(b2)],
    ]
    result = scipy_stats.fisher_exact(table, alternative="greater")
    return float(result.pvalue)
