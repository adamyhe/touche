from __future__ import annotations

from scipy.stats import hypergeom


def fisher_greater(a1: float, a2: float, b1: float, b2: float) -> float:
    """R-compatible one-sided Fisher exact p-value used by local decay calls.

    The reference implementation rounds values before calling R's
    ``fisher.test(..., alternative = "greater")`` through rpy2. Keep the
    rounding here so this can replace rpy2 without changing expected behavior.
    """

    a = round(a1)
    b = round(a2)
    c = round(b1)
    d = round(b2)
    if min(a, b, c, d) < 0:
        raise ValueError("Fisher exact table entries must be non-negative")

    total = a + b + c + d
    row_1 = a + b
    col_1 = a + c
    return float(hypergeom.sf(a - 1, total, row_1, col_1))
