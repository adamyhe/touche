from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from touche.pairs import (
    PAIR_COLUMNS,
    attach_pair_ids,
    build_pair_table,
    pair_table_from_centers,
    read_bedpe,
    write_bedpe,
)


def _anchors(chroms: list[str], starts: list[int], ends: list[int], strands: list[str] | None = None) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "chr": chroms,
            "start": starts,
            "end": ends,
            "strand": strands if strands is not None else ["."] * len(chroms),
        }
    )
    return frame.with_columns(((pl.col("start") + pl.col("end")) // 2).alias("center"))


class BuildPairTableTests(unittest.TestCase):
    def test_expands_same_chromosome_pairs_within_distance_window(self) -> None:
        baits = _anchors(["chr1", "chr2"], [1000, 1000], [1200, 1200])
        preys = _anchors(["chr1", "chr1", "chr2"], [20000, 900000, 3000], [20200, 900200, 3200])

        pairs = build_pair_table(baits, preys, min_distance=5_000, max_distance=100_000)

        self.assertEqual(pairs.columns, PAIR_COLUMNS)
        self.assertEqual(pairs["prey_center"].to_list(), [20100])
        self.assertEqual(pairs["distance"].to_list(), [19000])
        self.assertTrue(pairs["is_cis"].all())

    def test_empty_anchor_input_returns_empty_frame_with_schema(self) -> None:
        empty = build_pair_table(_anchors([], [], []), _anchors(["chr1"], [0], [10]))
        self.assertEqual(empty.height, 0)
        self.assertEqual(empty.columns, PAIR_COLUMNS)

    def test_rejects_inverted_distance_window(self) -> None:
        with self.assertRaises(ValueError):
            build_pair_table(_anchors(["chr1"], [0], [10]), _anchors(["chr1"], [0], [10]), min_distance=10, max_distance=5)


class PairIdTests(unittest.TestCase):
    def test_pair_id_is_invariant_to_anchor_order(self) -> None:
        forward = pl.DataFrame(
            {"bait_chrom": ["chr1"], "bait_center": [1000], "prey_chrom": ["chr1"], "prey_center": [50000]}
        )
        reversed_ = pl.DataFrame(
            {"bait_chrom": ["chr1"], "bait_center": [50000], "prey_chrom": ["chr1"], "prey_center": [1000]}
        )

        self.assertEqual(
            attach_pair_ids(forward)["pair_id"].to_list(), attach_pair_ids(reversed_)["pair_id"].to_list()
        )

    def test_pair_id_is_independent_of_chunking(self) -> None:
        baits = _anchors(["chr1"] * 50, list(range(0, 500_000, 10_000)), list(range(200, 500_200, 10_000)))
        preys = _anchors(["chr1"] * 50, list(range(5_000, 505_000, 10_000)), list(range(5_200, 505_200, 10_000)))
        whole = build_pair_table(baits, preys, min_distance=1_000)

        halves = pl.concat(
            [
                build_pair_table(baits[:25], preys, min_distance=1_000),
                build_pair_table(baits[25:], preys, min_distance=1_000),
            ]
        )

        self.assertEqual(sorted(whole["pair_id"].to_list()), sorted(halves["pair_id"].to_list()))

    def test_digest_style_is_stable_and_fixed_width(self) -> None:
        frame = pl.DataFrame(
            {"bait_chrom": ["chr1"], "bait_center": [1000], "prey_chrom": ["chr1"], "prey_center": [50000]}
        )
        first = attach_pair_ids(frame, style="digest")["pair_id"][0]
        second = attach_pair_ids(frame, style="digest")["pair_id"][0]

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertNotEqual(first, attach_pair_ids(frame)["pair_id"][0])

    def test_center_anchored_and_interval_anchored_pairs_share_ids(self) -> None:
        interval = build_pair_table(
            _anchors(["chr1"], [900], [1100]), _anchors(["chr1"], [49900], [50100]), min_distance=1
        )
        centers = pair_table_from_centers(["chr1"], [1000], [50000])

        self.assertEqual(interval["pair_id"].to_list(), centers["pair_id"].to_list())

    def test_attach_pair_ids_reports_missing_anchor_columns(self) -> None:
        with self.assertRaises(ValueError):
            attach_pair_ids(pl.DataFrame({"bait_chrom": ["chr1"]}))


class BedpeTests(unittest.TestCase):
    def _write(self, tmp: Path, text: str) -> Path:
        path = tmp / "pairs.bedpe"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_core_columns_and_derives_centers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                "# comment\nchr1\t1000\t1200\tchr1\t50000\t50200\tpairA\t3.5\t+\t-\n",
            )
            pairs = read_bedpe(path)

        self.assertEqual(pairs["bait_center"].to_list(), [1100])
        self.assertEqual(pairs["prey_center"].to_list(), [50100])
        self.assertEqual(pairs["distance"].to_list(), [49000])
        self.assertEqual(pairs["bait_strand"].to_list(), ["+"])
        self.assertEqual(pairs["name"].to_list(), ["pairA"])

    def test_carries_extra_columns_through_as_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp), "chr1\t1000\t1200\tchr1\t50000\t50200\tp\t0\t+\t-\tpositive\t12.5\n"
            )
            pairs = read_bedpe(path)

        self.assertEqual(pairs["extra_0"].to_list(), ["positive"])
        self.assertEqual(pairs["extra_1"].to_list(), [12.5])

    def test_trans_pairs_have_null_chrom_and_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), "chr1\t1000\t1200\tchr2\t50000\t50200\n")
            pairs = read_bedpe(path)
            self.assertEqual(read_bedpe(path, cis_only=True).height, 0)

        self.assertFalse(pairs["is_cis"][0])
        self.assertIsNone(pairs["chrom"][0])
        self.assertIsNone(pairs["distance"][0])

    def test_bait_anchor_left_right_assigns_by_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), "chr1\t50000\t50200\tchr1\t1000\t1200\n")
            left = read_bedpe(path, bait_anchor="left")
            right = read_bedpe(path, bait_anchor="right")

        self.assertEqual(left["bait_center"].to_list(), [1100])
        self.assertEqual(right["bait_center"].to_list(), [50100])
        self.assertEqual(left["pair_id"].to_list(), right["pair_id"].to_list())

    def test_duplicate_policies(self) -> None:
        row = "chr1\t1000\t1200\tchr1\t50000\t50200\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), row * 3)
            self.assertEqual(read_bedpe(path, on_duplicate="first").height, 1)
            self.assertEqual(read_bedpe(path, on_duplicate="keep").height, 3)
            with self.assertRaises(ValueError):
                read_bedpe(path, on_duplicate="error")

    def test_rejects_invalid_intervals_and_short_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(ValueError):
                read_bedpe(self._write(tmp_path, "chr1\t1200\t1000\tchr1\t50000\t50200\n"))
            with self.assertRaises(ValueError):
                read_bedpe(self._write(tmp_path, "chr1\t1000\t1200\n"))

    def test_round_trips_through_write_bedpe(self) -> None:
        baits = _anchors(["chr1"], [1000], [1200], ["+"])
        preys = _anchors(["chr1"], [50000], [50200], ["-"])
        pairs = build_pair_table(baits, preys, min_distance=1_000)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.bedpe"
            write_bedpe(pairs, out)
            reloaded = read_bedpe(out)

        for column in ("pair_id", "bait_chrom", "bait_start", "bait_end", "prey_start", "distance"):
            self.assertEqual(reloaded[column].to_list(), pairs[column].to_list(), column)


if __name__ == "__main__":
    unittest.main()
