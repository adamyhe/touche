from __future__ import annotations

from math import exp as _exp
from math import lgamma as _lgamma

import numpy as np

try:
    from numba import get_thread_id, njit, prange
except ImportError as exc:  # pragma: no cover - exercised through backend helper
    raise RuntimeError(
        "Numba kernels require numba, which is a core dependency of ep-touche. "
        "If this error occurs, your installation may be incomplete -- try "
        "`uv sync --dev` or reinstalling `ep-touche`."
    ) from exc


@njit(cache=True, parallel=True)
def count_ep_background_pairs_eager_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    pair_bait_index: np.ndarray,
    pair_prey_index: np.ndarray,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    ep_counts = np.zeros(pair_bait_index.shape[0], dtype=np.int64)
    bg_counts = np.zeros(pair_bait_index.shape[0], dtype=np.int64)

    for pair_index in prange(pair_bait_index.shape[0]):
        bait_center = bait_centers[pair_bait_index[pair_index]]
        prey_center = prey_centers[pair_prey_index[pair_index]]

        bait_start = bait_center - window
        bait_end = bait_center + window
        prey_start = prey_center - window
        prey_end = prey_center + window

        prey_bg_left_start = prey_center - max_bg_distance
        prey_bg_left_end = prey_center - min_bg_distance
        prey_bg_right_start = prey_center + min_bg_distance
        prey_bg_right_end = prey_center + max_bg_distance

        bait_bg_left_start = bait_center - max_bg_distance
        bait_bg_left_end = bait_center - min_bg_distance
        bait_bg_right_start = bait_center + min_bg_distance
        bait_bg_right_end = bait_center + max_bg_distance

        ep = 0
        bg = 0
        for contact_index in range(pos_a.shape[0]):
            a = pos_a[contact_index]
            b = pos_b[contact_index]

            a_in_bait = bait_start <= a <= bait_end
            b_in_bait = bait_start <= b <= bait_end
            a_in_prey = prey_start <= a <= prey_end
            b_in_prey = prey_start <= b <= prey_end

            if (a_in_bait and b_in_prey) or (b_in_bait and a_in_prey):
                ep += 1

            a_in_prey_bg = (prey_bg_left_start <= a <= prey_bg_left_end) or (
                prey_bg_right_start <= a <= prey_bg_right_end
            )
            b_in_prey_bg = (prey_bg_left_start <= b <= prey_bg_left_end) or (
                prey_bg_right_start <= b <= prey_bg_right_end
            )
            a_in_bait_bg = (bait_bg_left_start <= a <= bait_bg_left_end) or (
                bait_bg_right_start <= a <= bait_bg_right_end
            )
            b_in_bait_bg = (bait_bg_left_start <= b <= bait_bg_left_end) or (
                bait_bg_right_start <= b <= bait_bg_right_end
            )

            if (a_in_bait and b_in_prey_bg) or (b_in_bait and a_in_prey_bg):
                bg += 1
            if (a_in_prey and b_in_bait_bg) or (b_in_prey and a_in_bait_bg):
                bg += 1

        ep_counts[pair_index] = ep
        bg_counts[pair_index] = bg

    return ep_counts, bg_counts


@njit(cache=True, parallel=True)
def count_ep_background_pairs_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    pair_bait_index: np.ndarray,
    pair_prey_index: np.ndarray,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    ep_counts = np.zeros(pair_bait_index.shape[0], dtype=np.int64)
    bg_counts = np.zeros(pair_bait_index.shape[0], dtype=np.int64)

    for pair_index in prange(pair_bait_index.shape[0]):
        bait_center = bait_centers[pair_bait_index[pair_index]]
        prey_center = prey_centers[pair_prey_index[pair_index]]

        bait_start = bait_center - window
        bait_end = bait_center + window
        prey_start = prey_center - window
        prey_end = prey_center + window

        prey_bg_left_start = prey_center - max_bg_distance
        prey_bg_left_end = prey_center - min_bg_distance
        prey_bg_right_start = prey_center + min_bg_distance
        prey_bg_right_end = prey_center + max_bg_distance

        bait_bg_left_start = bait_center - max_bg_distance
        bait_bg_left_end = bait_center - min_bg_distance
        bait_bg_right_start = bait_center + min_bg_distance
        bait_bg_right_end = bait_center + max_bg_distance

        ep = 0
        bg = 0
        for contact_index in range(pos_a.shape[0]):
            a = pos_a[contact_index]
            b = pos_b[contact_index]

            a_in_bait = bait_start <= a <= bait_end
            b_in_bait = bait_start <= b <= bait_end
            a_in_prey = prey_start <= a <= prey_end
            b_in_prey = prey_start <= b <= prey_end

            if (a_in_bait and b_in_prey) or (b_in_bait and a_in_prey):
                ep += 1

            bait_to_prey_bg = False
            if a_in_bait and (
                (prey_bg_left_start <= b <= prey_bg_left_end)
                or (prey_bg_right_start <= b <= prey_bg_right_end)
            ):
                bait_to_prey_bg = True
            if b_in_bait and (
                (prey_bg_left_start <= a <= prey_bg_left_end)
                or (prey_bg_right_start <= a <= prey_bg_right_end)
            ):
                bait_to_prey_bg = True
            if bait_to_prey_bg:
                bg += 1

            prey_to_bait_bg = False
            if a_in_prey and (
                (bait_bg_left_start <= b <= bait_bg_left_end)
                or (bait_bg_right_start <= b <= bait_bg_right_end)
            ):
                prey_to_bait_bg = True
            if b_in_prey and (
                (bait_bg_left_start <= a <= bait_bg_left_end)
                or (bait_bg_right_start <= a <= bait_bg_right_end)
            ):
                prey_to_bait_bg = True
            if prey_to_bait_bg:
                bg += 1

        ep_counts[pair_index] = ep
        bg_counts[pair_index] = bg

    return ep_counts, bg_counts


@njit(cache=True, parallel=True)
def apa_anchor_signal_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    centers: np.ndarray,
    strand_codes: np.ndarray,
    contact_mask: np.ndarray,
    window: int,
    pixels: int,
) -> np.ndarray:
    bins = pixels * 2
    step = window // pixels
    signals = np.zeros((centers.shape[0], bins), dtype=np.int64)

    for center_index in prange(centers.shape[0]):
        center = centers[center_index]
        strand_code = strand_codes[center_index]
        for bin_index in range(bins):
            if strand_code < 0:
                end = center + window - (bin_index * step)
                start = end - step
            else:
                start = center - window + (bin_index * step)
                end = start + step

            count = 0
            for contact_index in range(pos_a.shape[0]):
                if not contact_mask[contact_index]:
                    continue
                a = pos_a[contact_index]
                b = pos_b[contact_index]
                if (start <= a <= end) or (start <= b <= end):
                    count += 1
            signals[center_index, bin_index] = count

    return signals


@njit(cache=True, parallel=True)
def apa_matrix_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    bait_centers: np.ndarray,
    bait_strands: np.ndarray,
    prey_centers: np.ndarray,
    prey_strands: np.ndarray,
    pair_bait_index: np.ndarray,
    pair_prey_index: np.ndarray,
    long_range: np.ndarray,
    window: int,
    pixels: int,
    n_threads: int,
) -> np.ndarray:
    bins = pixels * 2
    step = window // pixels
    # Different pairs can land in the same (prey_bin, bait_bin) cell, so a
    # plain prange over pair_index would race on `matrix[...] +=`. Give each
    # thread its own exclusive matrix slice and reduce after the loop.
    # n_threads is passed in (rather than read via get_num_threads() here)
    # because calling get_num_threads() inside the kernel makes numba treat
    # it as a dynamic global and silently disables on-disk caching.
    thread_matrices = np.zeros((n_threads, bins, bins), dtype=np.int64)

    for pair_index in prange(pair_bait_index.shape[0]):
        tid = get_thread_id()
        bait_center = bait_centers[pair_bait_index[pair_index]]
        bait_strand = bait_strands[pair_bait_index[pair_index]]
        prey_center = prey_centers[pair_prey_index[pair_index]]
        prey_strand = prey_strands[pair_prey_index[pair_index]]
        bait_window_start = bait_center - window
        bait_window_end = bait_center + window

        for contact_index in range(pos_a.shape[0]):
            if not long_range[contact_index]:
                continue
            a = pos_a[contact_index]
            b = pos_b[contact_index]
            if not (
                (bait_window_start <= a <= bait_window_end)
                or (bait_window_start <= b <= bait_window_end)
            ):
                continue

            for bait_bin in range(bins):
                if bait_strand < 0:
                    bait_end = bait_center + window - (bait_bin * step)
                    bait_start = bait_end - step
                else:
                    bait_start = bait_center - window + (bait_bin * step)
                    bait_end = bait_start + step
                a_in_bait = bait_start <= a <= bait_end
                b_in_bait = bait_start <= b <= bait_end
                if not (a_in_bait or b_in_bait):
                    continue

                for prey_bin in range(bins):
                    if prey_strand < 0:
                        prey_end = prey_center + window - (prey_bin * step)
                        prey_start = prey_end - step
                    else:
                        prey_start = prey_center - window + (prey_bin * step)
                        prey_end = prey_start + step
                    a_in_prey = prey_start <= a <= prey_end
                    b_in_prey = prey_start <= b <= prey_end
                    if (a_in_bait and b_in_prey) or (b_in_bait and a_in_prey):
                        thread_matrices[tid, prey_bin, bait_bin] += 1

    return thread_matrices.sum(axis=0)


@njit(cache=True, parallel=True)
def local_decay_observed_counts_numba(
    plus: np.ndarray,
    minus: np.ndarray,
    bait_center: int,
    prey_centers: np.ndarray,
    cap: int,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def _lowess_anchor_indexes(n: int, delta: float) -> np.ndarray:
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
