"""A BEDPE pair list must define the pair universe exactly, in both APA and background."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from touche.apa import aggregate_apa, compute_apa
from touche.background import compute_ep_and_background, count_ep_and_background
from touche.models import ContactIndex
from touche.pairs import build_pair_table, write_bedpe

WINDOW_KWARGS = dict(min_distance=50_000, max_distance=200_000)
BACKGROUND_KWARGS = dict(window=1_000, min_bg_distance=20_000, max_bg_distance=100_000, **WINDOW_KWARGS)


def _indexes(chroms: tuple[str, ...], seed: int = 0) -> dict[str, ContactIndex]:
    rng = np.random.default_rng(seed)
    indexes = {}
    for chrom in chroms:
        n = 20_000
        pos_a = rng.integers(0, 600_000, n)
        indexes[chrom] = ContactIndex(
            chrom=chrom,
            pos_a=pos_a,
            pos_b=pos_a + rng.integers(40_000, 200_000, n),
            strand_a=np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8),
            strand_b=np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8),
            mapq_a=np.full(n, 30, dtype=np.int16),
            mapq_b=np.full(n, 30, dtype=np.int16),
        )
    return indexes


def _anchors(chroms: tuple[str, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    starts = list(range(0, 500_000, 25_000))
    baits = pl.DataFrame(
        {"chr": [c for c in chroms for _ in starts], "start": starts * len(chroms)}
    ).with_columns(
        (pl.col("start") + 1_000).alias("end"),
        pl.lit("+").alias("strand"),
        (pl.col("start") + 500).alias("center"),
    )
    preys = baits.with_columns(
        pl.lit("-").alias("strand"),
        (pl.col("start") + 12_000).alias("start"),
        (pl.col("end") + 12_000).alias("end"),
        (pl.col("center") + 12_000).alias("center"),
    )
    return baits, preys


class ExplicitPairUniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chroms = ("chr1", "chr2")
        self.baits, self.preys = _anchors(self.chroms)
        self.pairs = build_pair_table(self.baits, self.preys, **WINDOW_KWARGS)
        self.empty = pl.DataFrame()

    def test_apa_over_the_implicit_universe_written_out_reproduces_it_exactly(self) -> None:
        implicit = compute_apa(
            _indexes(self.chroms), self.baits, self.preys, window=5_000, pixels=10, **WINDOW_KWARGS
        )
        explicit = compute_apa(
            _indexes(self.chroms), self.empty, self.empty, pairs=self.pairs,
            window=5_000, pixels=10, **WINDOW_KWARGS,
        )

        for attribute in ("matrix", "bait_signal", "prey_signal"):
            np.testing.assert_array_equal(
                getattr(implicit, attribute).drop("bin_label").to_numpy(),
                getattr(explicit, attribute).drop("bin_label").to_numpy(),
                attribute,
            )

    def test_background_over_the_implicit_universe_reproduces_it_exactly(self) -> None:
        order = ["chr", "promoter", "enhancer"]
        implicit = compute_ep_and_background(
            _indexes(self.chroms), self.baits, self.preys, **BACKGROUND_KWARGS
        ).sort(order)
        explicit = compute_ep_and_background(
            _indexes(self.chroms), self.empty, self.empty, pairs=self.pairs, **BACKGROUND_KWARGS
        ).sort(order)

        self.assertTrue(implicit.equals(explicit))

    def test_a_restricted_pair_list_is_not_expanded_back_to_the_product(self) -> None:
        subset = self.pairs.head(7)

        counts = compute_ep_and_background(
            _indexes(self.chroms), self.empty, self.empty, pairs=subset, **BACKGROUND_KWARGS
        )

        self.assertEqual(counts.height, 7)
        self.assertEqual(counts["promoter"].to_list(), subset["bait_center"].to_list())
        self.assertEqual(counts["enhancer"].to_list(), subset["prey_center"].to_list())

    def test_pairs_outside_the_distance_window_are_still_counted_when_listed(self) -> None:
        # The distance filter builds the implicit universe; it must not also
        # silently prune an explicit one.
        close = build_pair_table(self.baits, self.preys, min_distance=0, max_distance=20_000)
        self.assertGreater(close.height, 0)

        counts = compute_ep_and_background(
            _indexes(self.chroms), self.empty, self.empty, pairs=close, **BACKGROUND_KWARGS
        )

        self.assertEqual(counts.height, close.height)

    def test_trans_pairs_are_dropped_from_a_cis_only_analysis(self) -> None:
        trans = self.pairs.head(2).with_columns(
            pl.lit("chr2").alias("prey_chrom"), pl.lit(False).alias("is_cis")
        )

        counts = compute_ep_and_background(
            _indexes(self.chroms), self.empty, self.empty,
            pairs=pl.concat([self.pairs, trans]), **BACKGROUND_KWARGS,
        )

        self.assertEqual(counts.height, self.pairs.height)


class PairListFileTests(unittest.TestCase):
    def _pairs_file(self, tmp: Path) -> tuple[Path, pl.DataFrame]:
        baits, preys = _anchors(("chr1",))
        pairs = build_pair_table(baits, preys, **WINDOW_KWARGS)
        path = write_bedpe(pairs, tmp / "pairs.bedpe")
        return path, pairs

    def _pairs_input(self, tmp: Path) -> Path:
        pairs_file = tmp / "contacts.pairs"
        rng = np.random.default_rng(0)
        pos_a = np.sort(rng.integers(0, 600_000, 3_000))
        rows = [
            f"chr1\t{a}\tchr1\t{a + d}\t+\t-\tUU\t30\t30"
            for a, d in zip(pos_a, rng.integers(40_000, 200_000, pos_a.size))
        ]
        pairs_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return pairs_file

    def test_background_count_accepts_a_bedpe_pair_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_list, pairs = self._pairs_file(tmp_path)
            out = tmp_path / "counts.tsv"

            counts = count_ep_and_background(
                self._pairs_input(tmp_path), baits_path=None, preys_path=None,
                out_path=out, pairs_list=pairs_list, **BACKGROUND_KWARGS,
            )

        self.assertEqual(counts.height, pairs.height)

    def test_apa_aggregate_accepts_a_bedpe_pair_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_list, _ = self._pairs_file(tmp_path)

            outputs = aggregate_apa(
                self._pairs_input(tmp_path), baits_path=None, preys_path=None,
                out_dir=tmp_path / "apa", pairs_list=pairs_list,
                window=5_000, pixels=10, **WINDOW_KWARGS,
            )
            matrix = pl.read_csv(outputs["matrix"])

        self.assertEqual(matrix.height, 20)


if __name__ == "__main__":
    unittest.main()
