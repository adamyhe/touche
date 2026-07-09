from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from touche.anchors import read_bed_anchors
from touche.backends import has_numba
from touche.background import compute_ep_and_background, count_ep_and_background
from touche.models import ContactIndex
from touche.contacts import build_contact_indexes


class BackgroundCountTests(unittest.TestCase):
    def test_count_ep_and_background_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            out = tmp_path / "EP_and_BG_contacts.tsv"

            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t100\tchr1\t300\t+\t-\tUU\t30\t30",  # EP
                        "chr1\t100\tchr1\t250\t+\t-\tUU\t30\t30",  # bait to prey background
                        "chr1\t150\tchr1\t300\t+\t-\tUU\t30\t30",  # prey to bait background
                        "chr1\t100\tchr1\t101\t+\t-\tUU\t30\t30",  # intra bait, ignored
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")

            result = count_ep_and_background(
                pairs,
                baits,
                preys,
                out,
                min_distance=150,
                max_distance=250,
                window=10,
                min_bg_distance=40,
                max_bg_distance=60,
                source="touche",
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0, "EP_contacts"], 1)
            self.assertEqual(result[0, "BG_contacts"], 2)
            written = pl.read_csv(
                out, separator="\t", has_header=False, new_columns=["chr", "bait", "prey", "ep", "bg"]
            )
            self.assertEqual(written[0, "ep"], 1)
            self.assertEqual(written[0, "bg"], 2)

    def test_compute_ep_and_background_accepts_in_memory_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            pairs.write_text(
                "chr1\t100\tchr1\t300\t+\t-\tUU\t30\t30\n",
                encoding="utf-8",
            )
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")

            result = compute_ep_and_background(
                build_contact_indexes(pairs, source="touche"),
                read_bed_anchors(baits),
                read_bed_anchors(preys),
                min_distance=150,
                max_distance=250,
                window=10,
                min_bg_distance=40,
                max_bg_distance=60,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0, "EP_contacts"], 1)

    @unittest.skipUnless(has_numba(), "numba is not installed")
    def test_compute_ep_and_background_matches_expected_counts(self) -> None:
        indexes = {
            "chr1": ContactIndex(
                chrom="chr1",
                pos_a=np.asarray([100, 100, 150, 100, 510, 500, 560], dtype=np.int64),
                pos_b=np.asarray([300, 250, 300, 101, 700, 650, 700], dtype=np.int64),
                strand_a=np.asarray(["+"] * 7),
                strand_b=np.asarray(["-"] * 7),
                mapq_a=np.asarray([30] * 7, dtype=np.int64),
                mapq_b=np.asarray([30] * 7, dtype=np.int64),
            )
        }
        baits = pl.DataFrame(
            {
                "chr": ["chr1", "chr1"],
                "start": [95, 495],
                "end": [105, 505],
                "strand": ["+", "+"],
                "center": [100, 500],
            }
        )
        preys = pl.DataFrame(
            {
                "chr": ["chr1", "chr1"],
                "start": [295, 695],
                "end": [305, 705],
                "strand": ["+", "+"],
                "center": [300, 700],
            }
        )
        kwargs = {
            "min_distance": 150,
            "max_distance": 250,
            "window": 10,
            "min_bg_distance": 40,
            "max_bg_distance": 60,
        }

        result = compute_ep_and_background(indexes, baits, preys, **kwargs)

        expected = pl.DataFrame(
            {
                "chr": ["chr1", "chr1", "chr1"],
                "promoter": [100, 500, 500],
                "enhancer": [300, 300, 700],
                "EP_contacts": [1, 0, 1],
                "BG_contacts": [2, 0, 2],
            },
            schema={
                "chr": pl.Utf8,
                "promoter": pl.Int64,
                "enhancer": pl.Int64,
                "EP_contacts": pl.Int64,
                "BG_contacts": pl.Int64,
            },
        )
        assert_frame_equal(result, expected)

    @unittest.skipUnless(has_numba(), "numba is not installed")
    def test_optimized_background_kernel_matches_reference_with_overlapping_windows(self) -> None:
        from touche.numba.background import count_ep_background_pairs_numba

        pos_a = np.asarray([90, 100, 105, 125, 135, 160, 170, 200], dtype=np.int64)
        pos_b = np.asarray([120, 135, 170, 95, 145, 100, 130, 110], dtype=np.int64)
        bait_centers = np.asarray([100, 150], dtype=np.int64)
        prey_centers = np.asarray([125, 170], dtype=np.int64)
        pair_bait_index = np.asarray([0, 0, 1, 1], dtype=np.int64)
        pair_prey_index = np.asarray([0, 1, 0, 1], dtype=np.int64)
        order = np.argsort(pos_a, kind="mergesort")

        expected_ep, expected_bg = _reference_ep_background_counts(
            pos_a,
            pos_b,
            bait_centers,
            prey_centers,
            pair_bait_index,
            pair_prey_index,
            20,
            10,
            50,
        )
        opt_ep, opt_bg = count_ep_background_pairs_numba(
            pos_a[order],
            pos_b[order],
            bait_centers,
            prey_centers,
            pair_bait_index,
            pair_prey_index,
            20,
            10,
            50,
        )

        np.testing.assert_array_equal(opt_ep, expected_ep)
        np.testing.assert_array_equal(opt_bg, expected_bg)


def _reference_ep_background_counts(
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

    for pair_index, (bait_index, prey_index) in enumerate(
        zip(pair_bait_index, pair_prey_index, strict=True)
    ):
        bait_center = bait_centers[bait_index]
        prey_center = prey_centers[prey_index]

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

        for a, b in zip(pos_a, pos_b, strict=True):
            a_in_bait = bait_start <= a <= bait_end
            b_in_bait = bait_start <= b <= bait_end
            a_in_prey = prey_start <= a <= prey_end
            b_in_prey = prey_start <= b <= prey_end

            if (a_in_bait and b_in_prey) or (b_in_bait and a_in_prey):
                ep_counts[pair_index] += 1

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
                bg_counts[pair_index] += 1
            if (a_in_prey and b_in_bait_bg) or (b_in_prey and a_in_bait_bg):
                bg_counts[pair_index] += 1

    return ep_counts, bg_counts


if __name__ == "__main__":
    unittest.main()
