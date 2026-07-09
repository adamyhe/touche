from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

from touche.backends import DEFAULT_FISHER_BACKEND


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


def fisher_greater_batch(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    *,
    backend: str = DEFAULT_FISHER_BACKEND,
) -> np.ndarray:
    """Vectorized sibling of `fisher_greater` -- `hypergeom.sf` broadcasts over arrays.

    `backend="numba"` uses a `prange`-parallel hypergeometric survival
    function instead of `scipy.stats.hypergeom.sf`, matching it to within
    ~1e-8 absolute error (see notes/numba-implementation-plan.md) in exchange
    for using more than one core. `backend="scipy"` (the default) is exact.
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
        from touche.numba_kernels import hypergeom_sf_numba

        return hypergeom_sf_numba(
            (a - 1).astype(np.int64),
            total.astype(np.int64),
            row_1.astype(np.int64),
            col_1.astype(np.int64),
        )
    return hypergeom.sf(a - 1, total, row_1, col_1)
