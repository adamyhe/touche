from __future__ import annotations

import numpy as np

from touche.numba._kernel_imports import njit, prange


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
