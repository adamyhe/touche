"""Numba counting kernels for APA: the 1D anchor signal and the pairwise pixel matrix.

Both are always used by `touche.apa.compute_apa` -- there's no alternate
counting path to choose between. Wrapped by `touche.apa._apa_anchor_signal_numba`/
`_apa_matrix_numba`, which handle dtype casting before calling in here.
"""

from __future__ import annotations

import numpy as np

from touche.numba._kernel_imports import get_thread_id, njit, prange


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
    """1D pixel-binned contact signal around each anchor in `centers` (bait or prey side)."""
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
    """Pairwise pixel-binned contact matrix (prey bin x bait bin) for the given bait/prey groups."""
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
