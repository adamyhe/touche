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
    def test_numba_background_backend_matches_numpy_counts(self) -> None:
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

        numpy_result = compute_ep_and_background(indexes, baits, preys, backend="numpy", **kwargs)
        numba_result = compute_ep_and_background(indexes, baits, preys, backend="numba", **kwargs)

        assert_frame_equal(numba_result, numpy_result)

    def test_invalid_background_backend_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend"):
            compute_ep_and_background(
                {},
                pl.DataFrame(schema=["chr", "center"]),
                pl.DataFrame(schema=["chr", "center"]),
                min_distance=150,
                max_distance=250,
                window=10,
                min_bg_distance=40,
                max_bg_distance=60,
                backend="bad",
            )


if __name__ == "__main__":
    unittest.main()
