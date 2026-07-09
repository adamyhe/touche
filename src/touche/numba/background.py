"""Numba counting kernels for EP/background pair counting.

`count_ep_background_pairs_numba` is the one `touche.background.compute_ep_and_background`
always uses -- wrapped by `touche.background._count_ep_background_pairs_numba`,
which handles dtype casting and sorting first.
"""

from __future__ import annotations

import numpy as np

from touche.numba._kernel_imports import njit, prange


@njit(cache=True, parallel=True)
def count_ep_background_pairs_numba(
    sorted_pos_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    pair_bait_index: np.ndarray,
    pair_prey_index: np.ndarray,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimized EP/background pair counter; always used by `touche.background.compute_ep_and_background`.

    Numerically exact-equivalent to the reference tests in
    `tests/test_background_count.py`.
    `sorted_pos_a` must be ascending, and `sorted_pos_b` must be reordered by
    the same permutation; this lets each pair scan only contacts whose first
    endpoint is in a relevant foreground/background window instead of scanning
    the whole chromosome.
    """
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

        if (
            _windows_overlap(prey_bg_left_start, prey_bg_left_end, prey_bg_right_start, prey_bg_right_end)
            or _windows_overlap(bait_bg_left_start, bait_bg_left_end, bait_bg_right_start, bait_bg_right_end)
        ):
            ep, bg = _count_pair_eager(
                sorted_pos_a,
                sorted_pos_b,
                bait_start,
                bait_end,
                prey_start,
                prey_end,
                prey_bg_left_start,
                prey_bg_left_end,
                prey_bg_right_start,
                prey_bg_right_end,
                bait_bg_left_start,
                bait_bg_left_end,
                bait_bg_right_start,
                bait_bg_right_end,
            )
            ep_counts[pair_index] = ep
            bg_counts[pair_index] = bg
        else:
            ep = _count_between_either_orientation(
                sorted_pos_a,
                sorted_pos_b,
                bait_start,
                bait_end,
                prey_start,
                prey_end,
            )
            bait_to_prey_bg = _count_between_either_orientation_disjoint_right(
                sorted_pos_a,
                sorted_pos_b,
                bait_start,
                bait_end,
                prey_bg_left_start,
                prey_bg_left_end,
                prey_bg_right_start,
                prey_bg_right_end,
            )
            prey_to_bait_bg = _count_between_either_orientation_disjoint_right(
                sorted_pos_a,
                sorted_pos_b,
                prey_start,
                prey_end,
                bait_bg_left_start,
                bait_bg_left_end,
                bait_bg_right_start,
                bait_bg_right_end,
            )

            ep_counts[pair_index] = ep
            bg_counts[pair_index] = bait_to_prey_bg + prey_to_bait_bg

    return ep_counts, bg_counts


@njit(cache=True)
def _windows_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


@njit(cache=True)
def _count_pair_eager(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    bait_start: int,
    bait_end: int,
    prey_start: int,
    prey_end: int,
    prey_bg_left_start: int,
    prey_bg_left_end: int,
    prey_bg_right_start: int,
    prey_bg_right_end: int,
    bait_bg_left_start: int,
    bait_bg_left_end: int,
    bait_bg_right_start: int,
    bait_bg_right_end: int,
) -> tuple[int, int]:
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

    return ep, bg


@njit(cache=True)
def _count_between_either_orientation(
    sorted_pos_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    forward = _count_a_to_b(sorted_pos_a, sorted_pos_b, left_start, left_end, right_start, right_end)
    reverse = _count_a_to_b(sorted_pos_a, sorted_pos_b, right_start, right_end, left_start, left_end)
    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    if overlap_start <= overlap_end:
        duplicated = _count_a_to_b(
            sorted_pos_a,
            sorted_pos_b,
            overlap_start,
            overlap_end,
            overlap_start,
            overlap_end,
        )
        return forward + reverse - duplicated
    return forward + reverse


@njit(cache=True)
def _count_between_either_orientation_disjoint_right(
    sorted_pos_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    left_start: int,
    left_end: int,
    right_left_start: int,
    right_left_end: int,
    right_right_start: int,
    right_right_end: int,
) -> int:
    left_to_right = _count_a_to_b(
        sorted_pos_a,
        sorted_pos_b,
        left_start,
        left_end,
        right_left_start,
        right_left_end,
    ) + _count_a_to_b(
        sorted_pos_a,
        sorted_pos_b,
        left_start,
        left_end,
        right_right_start,
        right_right_end,
    )
    right_to_left = _count_a_to_b(
        sorted_pos_a,
        sorted_pos_b,
        right_left_start,
        right_left_end,
        left_start,
        left_end,
    ) + _count_a_to_b(
        sorted_pos_a,
        sorted_pos_b,
        right_right_start,
        right_right_end,
        left_start,
        left_end,
    )

    overlap_left_start = max(left_start, right_left_start)
    overlap_left_end = min(left_end, right_left_end)
    duplicated = 0
    if overlap_left_start <= overlap_left_end:
        duplicated += _count_a_to_b(
            sorted_pos_a,
            sorted_pos_b,
            overlap_left_start,
            overlap_left_end,
            overlap_left_start,
            overlap_left_end,
        )

    overlap_right_start = max(left_start, right_right_start)
    overlap_right_end = min(left_end, right_right_end)
    if overlap_right_start <= overlap_right_end:
        duplicated += _count_a_to_b(
            sorted_pos_a,
            sorted_pos_b,
            overlap_right_start,
            overlap_right_end,
            overlap_right_start,
            overlap_right_end,
        )

    if overlap_left_start <= overlap_left_end and overlap_right_start <= overlap_right_end:
        duplicated += _count_a_to_b(
            sorted_pos_a,
            sorted_pos_b,
            overlap_left_start,
            overlap_left_end,
            overlap_right_start,
            overlap_right_end,
        )
        duplicated += _count_a_to_b(
            sorted_pos_a,
            sorted_pos_b,
            overlap_right_start,
            overlap_right_end,
            overlap_left_start,
            overlap_left_end,
        )

    return left_to_right + right_to_left - duplicated


@njit(cache=True)
def _count_a_to_b(
    sorted_pos_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> int:
    left = np.searchsorted(sorted_pos_a, a_start, side="left")
    right = np.searchsorted(sorted_pos_a, a_end, side="right")
    count = 0
    for contact_index in range(left, right):
        b = sorted_pos_b[contact_index]
        if b_start <= b <= b_end:
            count += 1
    return count
