from __future__ import annotations

import unittest

import polars as pl

from touche.pair_stats import _distance_bin_expr, compute_pair_stats, distance_bin


class DistanceBinParityTests(unittest.TestCase):
    def test_distance_bin_expr_matches_python_at_every_boundary(self) -> None:
        distances = [
            0,
            1,
            999,
            1_000,
            1_001,
            9_999,
            10_000,
            10_001,
            99_999,
            100_000,
            100_001,
            999_999,
            1_000_000,
            1_000_001,
            9_999_999,
            10_000_000,
            10_000_001,
            50_000_000,
        ]
        expected = [distance_bin(d) for d in distances]

        frame = pl.DataFrame({"distance": distances}).with_columns(
            _distance_bin_expr(pl.col("distance")).alias("bin")
        )
        actual = frame["bin"].to_list()

        self.assertEqual(actual, expected)


class ComputePairStatsTests(unittest.TestCase):
    def test_matches_hand_counted_stats(self) -> None:
        lf = pl.LazyFrame(
            {
                "chrom_a": ["chr1", "chr1", "chr1", "chr2"],
                "chrom_b": ["chr1", "chr1", "chr2", "chr2"],
                "pos_a": [10, 20, 10, 5],
                "pos_b": [30, 40, 30, 15],
                "mapq_a": [30, 29, 30, 30],
                "mapq_b": [31, 31, 31, 31],
            }
        )

        stats = compute_pair_stats(lf, min_mapq=30, written_rows=2)

        self.assertEqual(stats.total_rows, 4)
        self.assertEqual(stats.cis_rows, 3)
        self.assertEqual(stats.trans_rows, 1)
        self.assertEqual(stats.mapq_pass_rows, 3)
        self.assertEqual(stats.mapq_fail_rows, 1)
        self.assertEqual(stats.written_rows, 2)
        self.assertEqual(stats.per_chromosome, {"chr1": 2, "chr2": 1})
        self.assertEqual(stats.distance_histogram, {"<1kb": 3})


if __name__ == "__main__":
    unittest.main()
