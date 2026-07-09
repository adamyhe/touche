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
    sorted_pos_a: np.ndarray,
    order_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    order_b: np.ndarray,
    centers: np.ndarray,
    strand_codes: np.ndarray,
    contact_mask: np.ndarray,
    window: int,
    pixels: int,
) -> np.ndarray:
    bins = pixels * 2
    step = window // pixels
    signals = np.zeros((centers.shape[0], bins), dtype=np.int64)

    # Binary search sorted_pos_a/sorted_pos_b (sorted once per chromosome,
    # with order_a/order_b mapping back to contact indices) for the contacts
    # within this anchor's window, instead of rescanning every contact for
    # every one of its bins. That turned this kernel into an O(anchors *
    # bins * contacts) scan with no window filtering at all.
    for center_index in prange(centers.shape[0]):
        center = centers[center_index]
        strand_code = strand_codes[center_index]
        window_start = center - window
        window_end = center + window

        lo_a = np.searchsorted(sorted_pos_a, window_start, side="left")
        hi_a = np.searchsorted(sorted_pos_a, window_end, side="right")
        lo_b = np.searchsorted(sorted_pos_b, window_start, side="left")
        hi_b = np.searchsorted(sorted_pos_b, window_end, side="right")

        n_a = hi_a - lo_a
        n_b = hi_b - lo_b
        candidates = np.empty(n_a + n_b, dtype=np.int64)
        candidates[:n_a] = order_a[lo_a:hi_a]
        candidates[n_a:] = order_b[lo_b:hi_b]
        candidates.sort()

        prev = np.int64(-1)
        for k in range(candidates.shape[0]):
            contact_index = candidates[k]
            if contact_index == prev:
                continue
            prev = contact_index
            if not contact_mask[contact_index]:
                continue
            a = pos_a[contact_index]
            b = pos_b[contact_index]
            for bin_index in range(bins):
                if strand_code < 0:
                    end = center + window - (bin_index * step)
                    start = end - step
                else:
                    start = center - window + (bin_index * step)
                    end = start + step
                if (start <= a <= end) or (start <= b <= end):
                    signals[center_index, bin_index] += 1

    return signals


@njit(cache=True, parallel=True)
def apa_matrix_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    sorted_pos_a: np.ndarray,
    order_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    order_b: np.ndarray,
    bait_centers: np.ndarray,
    bait_strands: np.ndarray,
    prey_centers: np.ndarray,
    prey_strands: np.ndarray,
    group_bait_index: np.ndarray,
    group_start: np.ndarray,
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

    # pair_prey_index is grouped contiguously by bait (group_start marks each
    # bait's slice), so the contact-window filter and the bait-side bin
    # classification -- both independent of which prey we pair the bait with
    # -- are done once per bait group here rather than once per (bait, prey)
    # pair. The window filter itself uses binary search against pos_a/pos_b
    # sorted once per chromosome (sorted_pos_a/sorted_pos_b + the order_*
    # permutations that map back to contact indices), instead of a linear
    # scan of every contact per bait, so a bait's cost is bounded by the
    # local contact density around it rather than the whole chromosome.
    for group_index in prange(group_bait_index.shape[0]):
        tid = get_thread_id()
        bait_index = group_bait_index[group_index]
        bait_center = bait_centers[bait_index]
        bait_strand = bait_strands[bait_index]
        bait_window_start = bait_center - window
        bait_window_end = bait_center + window

        lo_a = np.searchsorted(sorted_pos_a, bait_window_start, side="left")
        hi_a = np.searchsorted(sorted_pos_a, bait_window_end, side="right")
        lo_b = np.searchsorted(sorted_pos_b, bait_window_start, side="left")
        hi_b = np.searchsorted(sorted_pos_b, bait_window_end, side="right")

        n_a = hi_a - lo_a
        n_b = hi_b - lo_b
        candidates = np.empty(n_a + n_b, dtype=np.int64)
        candidates[:n_a] = order_a[lo_a:hi_a]
        candidates[n_a:] = order_b[lo_b:hi_b]
        candidates.sort()

        # A contact can satisfy the window on both the a-side and the
        # b-side, landing in both slices above -- dedupe via adjacency on
        # the sorted candidate list rather than an O(n_contacts) seen set.
        count = 0
        prev = np.int64(-1)
        for k in range(candidates.shape[0]):
            contact_index = candidates[k]
            if contact_index == prev:
                continue
            prev = contact_index
            if long_range[contact_index]:
                count += 1

        local_a = np.empty(count, dtype=np.int64)
        local_b = np.empty(count, dtype=np.int64)
        a_in_bait = np.zeros((count, bins), dtype=np.bool_)
        b_in_bait = np.zeros((count, bins), dtype=np.bool_)

        idx = 0
        prev = np.int64(-1)
        for k in range(candidates.shape[0]):
            contact_index = candidates[k]
            if contact_index == prev:
                continue
            prev = contact_index
            if not long_range[contact_index]:
                continue
            a = pos_a[contact_index]
            b = pos_b[contact_index]
            local_a[idx] = a
            local_b[idx] = b
            for bait_bin in range(bins):
                if bait_strand < 0:
                    bait_end = bait_center + window - (bait_bin * step)
                    bait_start = bait_end - step
                else:
                    bait_start = bait_center - window + (bait_bin * step)
                    bait_end = bait_start + step
                if bait_start <= a <= bait_end:
                    a_in_bait[idx, bait_bin] = True
                if bait_start <= b <= bait_end:
                    b_in_bait[idx, bait_bin] = True
            idx += 1

        start = group_start[group_index]
        end = group_start[group_index + 1]
        for pair_index in range(start, end):
            prey_center = prey_centers[pair_prey_index[pair_index]]
            prey_strand = prey_strands[pair_prey_index[pair_index]]

            for contact_index in range(count):
                a = local_a[contact_index]
                b = local_b[contact_index]

                for bait_bin in range(bins):
                    a_in_this_bait = a_in_bait[contact_index, bait_bin]
                    b_in_this_bait = b_in_bait[contact_index, bait_bin]
                    if not (a_in_this_bait or b_in_this_bait):
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
                        if (a_in_this_bait and b_in_prey) or (b_in_this_bait and a_in_prey):
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
