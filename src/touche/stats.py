"""One-sided Fisher exact test helpers. Public API: `fisher_greater_batch`."""

from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

from touche.backends import DEFAULT_FISHER_BACKEND


def fisher_greater_batch(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    *,
    backend: str = DEFAULT_FISHER_BACKEND,
) -> np.ndarray:
    """R-compatible one-sided Fisher exact p-values used by local decay calls.

    The reference implementation rounds values before calling R's
    ``fisher.test(..., alternative = "greater")`` (via rpy2); this function
    replaces that call and keeps the same rounding so results match exactly.

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
