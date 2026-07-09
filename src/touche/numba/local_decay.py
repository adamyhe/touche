"""Numba kernels for local-decay: observed-contact counting and LOWESS smoothing.

`local_decay_observed_counts_numba` is always used by
`touche.local_decay._call_bait_contacts` -- there's no alternate counting
path. The `lowess_*` kernels back `lowess_backend="numba"`, a validated
approximation of the `statsmodels` alternative (see
`notes/numba-implementation-plan.md`), not a bit-exact equivalent.
"""

from __future__ import annotations

import numpy as np

from touche.numba._kernel_imports import njit, prange


@njit(cache=True, parallel=True)
def local_decay_observed_counts_numba(
    plus: np.ndarray,
    minus: np.ndarray,
    bait_center: int,
    prey_centers: np.ndarray,
    cap: int,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed contact count, directional distance, and background pool size for each prey."""
    observed = np.zeros(prey_centers.shape[0], dtype=np.int64)
    directional_distances = np.zeros(prey_centers.shape[0], dtype=np.int64)
    contact_counts = np.zeros(prey_centers.shape[0], dtype=np.int64)

    for prey_index in prange(prey_centers.shape[0]):
        prey_center = prey_centers[prey_index]
        directional_distance = prey_center - bait_center
        directional_distances[prey_index] = directional_distance
        if abs(directional_distance) <= min_distance:
            contact_counts[prey_index] = -1
            continue

        prey_start = prey_center - cap
        prey_stop = prey_center + cap
        count = 0
        if directional_distance > 0:
            contact_counts[prey_index] = plus.shape[0]
            for contact_index in range(plus.shape[0]):
                if prey_start <= plus[contact_index] <= prey_stop:
                    count += 1
        else:
            contact_counts[prey_index] = minus.shape[0]
            for contact_index in range(minus.shape[0]):
                if prey_start <= minus[contact_index] <= prey_stop:
                    count += 1
        observed[prey_index] = count

    return observed, directional_distances, contact_counts


@njit(cache=True, parallel=True)
def lowess_evenly_spaced_numba(
    endog: np.ndarray,
    frac: float,
    iterations: int,
    delta: float,
) -> np.ndarray:
    """Robust LOWESS fit assuming implicitly evenly-spaced positions `0..n-1` (no explicit `exog`).

    `delta`-spaced anchor points are fit directly (`_lowess_anchor_indexes`)
    and the rest interpolated (`_interpolate_lowess_anchors`), mirroring
    `statsmodels.nonparametric.smoothers_lowess.lowess`'s own `delta`
    short-circuit for evenly-spaced input.
    """
    n = endog.shape[0]
    fitted = np.empty(n, dtype=np.float64)
    if n <= 2:
        for i in prange(n):
            fitted[i] = endog[i]
        return fitted

    window = int(np.ceil(frac * n))
    if window < 2:
        window = 2
    if window > n:
        window = n

    anchor_indexes = _lowess_anchor_indexes(n, delta)
    anchor_fitted = np.empty(anchor_indexes.shape[0], dtype=np.float64)
    robust = np.ones(n, dtype=np.float64)
    for iteration in range(iterations + 1):
        for anchor_pos in prange(anchor_indexes.shape[0]):
            i = anchor_indexes[anchor_pos]
            left = i - (window // 2)
            if left < 0:
                left = 0
            if left + window > n:
                left = n - window
            right = left + window
            max_dist = i - left
            right_dist = right - 1 - i
            if right_dist > max_dist:
                max_dist = right_dist
            if max_dist <= 0:
                fitted[i] = endog[i]
                continue

            sw = 0.0
            swx = 0.0
            swy = 0.0
            swxx = 0.0
            swxy = 0.0
            for j in range(left, right):
                scaled = abs(j - i) / max_dist
                base = 1.0 - scaled * scaled * scaled
                weight = base * base * base * robust[j]
                x = j - i
                y = endog[j]
                sw += weight
                swx += weight * x
                swy += weight * y
                swxx += weight * x * x
                swxy += weight * x * y

            denom = (sw * swxx) - (swx * swx)
            if abs(denom) < 1e-12:
                anchor_fitted[anchor_pos] = swy / sw if sw > 0 else endog[i]
            else:
                anchor_fitted[anchor_pos] = ((swy * swxx) - (swx * swxy)) / denom

        _interpolate_lowess_anchors(anchor_indexes, anchor_fitted, fitted)

        if iteration == iterations:
            break

        residuals = np.empty(n, dtype=np.float64)
        for i in prange(n):
            residuals[i] = abs(endog[i] - fitted[i])
        median_residual = np.median(residuals)
        if median_residual <= 0:
            break
        cutoff = 6.0 * median_residual
        for i in prange(n):
            scaled_residual = residuals[i] / cutoff
            if scaled_residual >= 1.0:
                robust[i] = 0.0
            else:
                weight = 1.0 - scaled_residual * scaled_residual
                robust[i] = weight * weight

    return fitted


@njit(cache=True)
def _lowess_fit_chunk_sequential(
    endog: np.ndarray, frac: float, iterations: int, delta: float, fitted: np.ndarray
) -> None:
    """Fit one LOWESS chunk into `fitted`, sequentially (no `prange`).

    Same anchor/residual/robustness-weight formulas as
    `lowess_evenly_spaced_numba`, just with `range` instead of `prange` --
    every one of those loops only ever reads/writes its own array slot (no
    cross-anchor or cross-element accumulation), so this cannot change any
    computed value versus the parallel version; it only changes who is
    allowed to run it concurrently. Meant to be called once per chunk from
    inside `lowess_evenly_spaced_batched_numba`'s outer `prange` -- per-chunk
    work is already the unit of parallelism there, and nesting another
    `prange` inside it would just be silently serialized by numba anyway.
    """
    n = endog.shape[0]
    if n <= 2:
        for i in range(n):
            fitted[i] = endog[i]
        return

    window = int(np.ceil(frac * n))
    if window < 2:
        window = 2
    if window > n:
        window = n

    anchor_indexes = _lowess_anchor_indexes(n, delta)
    anchor_fitted = np.empty(anchor_indexes.shape[0], dtype=np.float64)
    robust = np.ones(n, dtype=np.float64)
    for iteration in range(iterations + 1):
        for anchor_pos in range(anchor_indexes.shape[0]):
            i = anchor_indexes[anchor_pos]
            left = i - (window // 2)
            if left < 0:
                left = 0
            if left + window > n:
                left = n - window
            right = left + window
            max_dist = i - left
            right_dist = right - 1 - i
            if right_dist > max_dist:
                max_dist = right_dist
            if max_dist <= 0:
                fitted[i] = endog[i]
                continue

            sw = 0.0
            swx = 0.0
            swy = 0.0
            swxx = 0.0
            swxy = 0.0
            for j in range(left, right):
                scaled = abs(j - i) / max_dist
                base = 1.0 - scaled * scaled * scaled
                weight = base * base * base * robust[j]
                x = j - i
                y = endog[j]
                sw += weight
                swx += weight * x
                swy += weight * y
                swxx += weight * x * x
                swxy += weight * x * y

            denom = (sw * swxx) - (swx * swx)
            if abs(denom) < 1e-12:
                anchor_fitted[anchor_pos] = swy / sw if sw > 0 else endog[i]
            else:
                anchor_fitted[anchor_pos] = ((swy * swxx) - (swx * swxy)) / denom

        _interpolate_lowess_anchors(anchor_indexes, anchor_fitted, fitted)

        if iteration == iterations:
            break

        residuals = np.empty(n, dtype=np.float64)
        for i in range(n):
            residuals[i] = abs(endog[i] - fitted[i])
        median_residual = np.median(residuals)
        if median_residual <= 0:
            break
        cutoff = 6.0 * median_residual
        for i in range(n):
            scaled_residual = residuals[i] / cutoff
            if scaled_residual >= 1.0:
                robust[i] = 0.0
            else:
                weight = 1.0 - scaled_residual * scaled_residual
                robust[i] = weight * weight


@njit(cache=True, parallel=True)
def lowess_evenly_spaced_batched_numba(
    endog_flat: np.ndarray,
    offsets: np.ndarray,
    frac: float,
    iterations: int,
    delta: float,
) -> np.ndarray:
    """Fit LOWESS independently on each `endog_flat[offsets[c]:offsets[c+1]]` chunk.

    Numerically identical to calling `lowess_evenly_spaced_numba` once per
    chunk -- each chunk's fit has no cross-chunk dependency, so this only
    changes the parallelism granularity (one `prange` launch over chunks,
    instead of one small `prange` launch per chunk over that chunk's
    anchors). Exists because local-decay's LOWESS chunking (`lowess_window`,
    default 5,000 of a `dist`-sized array) makes each individual chunk's
    anchor count too small for `prange` launch overhead to pay for itself --
    see notes/numba-implementation-plan.md.
    """
    fitted_flat = np.empty(endog_flat.shape[0], dtype=np.float64)
    n_chunks = offsets.shape[0] - 1
    for chunk_index in prange(n_chunks):
        start = offsets[chunk_index]
        stop = offsets[chunk_index + 1]
        _lowess_fit_chunk_sequential(
            endog_flat[start:stop], frac, iterations, delta, fitted_flat[start:stop]
        )
    return fitted_flat


@njit(cache=True)
def _lowess_anchor_indexes(n: int, delta: float) -> np.ndarray:
    """Indexes to fit directly: every `delta`-th position, plus `0` and `n - 1`. All `n` if `delta <= 0`."""
    if delta <= 0:
        anchors = np.empty(n, dtype=np.int64)
        for i in range(n):
            anchors[i] = i
        return anchors

    scratch = np.empty(n, dtype=np.int64)
    count = 0
    current = 0
    while current < n:
        scratch[count] = current
        count += 1
        next_index = current + int(np.floor(delta))
        if next_index <= current:
            next_index = current + 1
        if next_index >= n:
            break
        current = next_index

    if scratch[count - 1] != n - 1:
        scratch[count] = n - 1
        count += 1

    return scratch[:count].copy()


@njit(cache=True)
def _interpolate_lowess_anchors(
    anchor_indexes: np.ndarray,
    anchor_fitted: np.ndarray,
    fitted: np.ndarray,
) -> None:
    """Linearly interpolate `fitted` between consecutive anchor points, in place."""
    for anchor_pos in range(anchor_indexes.shape[0] - 1):
        left = anchor_indexes[anchor_pos]
        right = anchor_indexes[anchor_pos + 1]
        left_value = anchor_fitted[anchor_pos]
        right_value = anchor_fitted[anchor_pos + 1]
        fitted[left] = left_value
        span = right - left
        if span <= 0:
            continue
        for i in range(left + 1, right):
            frac = (i - left) / span
            fitted[i] = left_value + frac * (right_value - left_value)
    fitted[anchor_indexes[-1]] = anchor_fitted[-1]
