from __future__ import annotations

from math import exp as _exp
from math import lgamma as _lgamma

import numpy as np

from touche.numba._kernel_imports import njit, prange


@njit(cache=True)
def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return _lgamma(n + 1.0) - _lgamma(k + 1.0) - _lgamma(n - k + 1.0)


@njit(cache=True)
def _hypergeom_sf_scalar(k: int, total: int, successes: int, draws: int) -> float:
    """P(X > k) for X ~ Hypergeometric(total, successes, draws), in log-space.

    Matches `scipy.stats.hypergeom.sf(k, total, successes, draws)` to within
    ~1e-8 absolute error (validated against scipy across randomized table
    shapes, including local decay's actual shape: `total`/`draws` in the
    millions with `successes` in the tens -- see
    notes/numba-implementation-plan.md). Degenerate `total <= 0` tables
    (unreachable from `_call_bait_contacts`, whose histograms always have at
    least one bin) return nan, matching scipy's convention.
    """

    if total <= 0:
        return np.nan
    support_lo = max(0, draws - (total - successes))
    support_hi = min(successes, draws)
    start = max(k + 1, support_lo)
    if start > support_hi:
        return 0.0
    if start <= support_lo:
        return 1.0

    log_denom = _log_binom(total, draws)
    # Sum whichever side of the split is smaller -- summing the larger side
    # directly loses precision when the true answer is close to 1.
    upper_count = support_hi - start + 1
    lower_count = start - support_lo
    if upper_count <= lower_count:
        total_prob = 0.0
        for i in range(start, support_hi + 1):
            log_pmf = _log_binom(successes, i) + _log_binom(total - successes, draws - i) - log_denom
            total_prob += _exp(log_pmf)
        return min(1.0, total_prob)
    total_prob = 0.0
    for i in range(support_lo, start):
        log_pmf = _log_binom(successes, i) + _log_binom(total - successes, draws - i) - log_denom
        total_prob += _exp(log_pmf)
    return max(0.0, 1.0 - total_prob)


@njit(cache=True, parallel=True)
def hypergeom_sf_numba(
    k: np.ndarray, total: np.ndarray, successes: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    out = np.empty(k.shape[0], dtype=np.float64)
    for i in prange(k.shape[0]):
        out[i] = _hypergeom_sf_scalar(k[i], total[i], successes[i], draws[i])
    return out
