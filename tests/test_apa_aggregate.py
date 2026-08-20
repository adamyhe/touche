from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from touche.anchors import read_bed_anchors
from touche.apa import aggregate_apa, compute_apa
from touche.backends import has_numba
from touche.contacts import build_contact_indexes, build_npz_cache


class ApaAggregateTests(unittest.TestCase):
    def test_aggregate_apa_writes_reference_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            out_dir = tmp_path / "apa"

            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t95\tchr1\t305\t+\t+\tUU\t30\t30",
                        "chr1\t95\tchr1\t315\t+\t+\tUU\t30\t30",
                        "chr1\t85\tchr1\t305\t+\t+\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")

            outputs = aggregate_apa(
                pairs,
                baits,
                preys,
                out_dir,
                min_distance=150,
                max_distance=250,
                window=20,
                pixels=2,
                source="touche",
                shift=0,
            )

            self.assertEqual(set(outputs), {"matrix", "heatmap", "baits_signal", "preys_signal"})
            for path in outputs.values():
                self.assertTrue(path.exists())
            matrix = pl.read_csv(outputs["matrix"])
            self.assertEqual(int(matrix.drop("bin_label").to_numpy().sum()), 3)
            bait_signal = pl.read_csv(outputs["baits_signal"])
            prey_signal = pl.read_csv(outputs["preys_signal"])
            self.assertEqual(int(bait_signal["contacts"].sum()), 3)
            self.assertEqual(int(prey_signal["contacts"].sum()), 3)

    def test_compute_apa_accepts_in_memory_indexes_and_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            pairs.write_text(
                "chr1\t95\tchr1\t305\t+\t+\tUU\t30\t30\n",
                encoding="utf-8",
            )
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")

            result = compute_apa(
                build_contact_indexes(pairs, source="touche"),
                read_bed_anchors(baits),
                read_bed_anchors(preys),
                min_distance=150,
                max_distance=250,
                window=20,
                pixels=2,
                shift=0,
            )
            fig = result.plot()

            self.assertEqual(int(result.matrix.drop("bin_label").to_numpy().sum()), 1)
            self.assertEqual(int(result.bait_signal["contacts"].sum()), 1)
            self.assertTrue(hasattr(fig, "savefig"))

            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_aggregate_apa_cache_index_strategy_matches_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            cache_dir = tmp_path / "cache"
            all_out = tmp_path / "all"
            cache_out = tmp_path / "cache_run"

            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t95\tchr1\t305\t+\t+\tUU\t30\t30",
                        "chr1\t95\tchr1\t315\t+\t+\tUU\t30\t30",
                        "chr1\t85\tchr1\t305\t+\t+\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")
            build_npz_cache(pairs, cache_dir, source="touche", prefix="sample", include_metadata=True)

            all_outputs = aggregate_apa(
                pairs,
                baits,
                preys,
                all_out,
                min_distance=150,
                max_distance=250,
                window=20,
                pixels=2,
                source="touche",
                shift=0,
                index_strategy="all",
            )
            cache_outputs = aggregate_apa(
                pairs,
                baits,
                preys,
                cache_out,
                min_distance=150,
                max_distance=250,
                window=20,
                pixels=2,
                source="touche",
                shift=0,
                index_strategy="cache",
                cache_dir=cache_dir,
                cache_prefix="sample",
            )

            assert_frame_equal(
                pl.read_csv(cache_outputs["matrix"]), pl.read_csv(all_outputs["matrix"])
            )

    def test_aggregate_apa_require_cache_rejects_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            out_dir = tmp_path / "apa"
            pairs.write_text("chr1\t95\tchr1\t305\t+\t+\tUU\t30\t30\n", encoding="utf-8")
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                aggregate_apa(
                    pairs,
                    baits,
                    preys,
                    out_dir,
                    min_distance=150,
                    max_distance=250,
                    window=20,
                    pixels=2,
                    source="touche",
                    shift=0,
                    index_strategy="cache",
                    cache_dir=tmp_path / "missing-cache",
                    require_cache=True,
                )

    @unittest.skipUnless(has_numba(), "numba is not installed")
    def test_compute_apa_matches_expected_matrix_and_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t95\tchr1\t305\t+\t+\tUU\t30\t30",
                        "chr1\t95\tchr1\t315\t+\t+\tUU\t30\t30",
                        "chr1\t85\tchr1\t305\t+\t+\tUU\t30\t30",
                        "chr1\t505\tchr1\t705\t+\t-\tUU\t30\t30",
                        "chr1\t515\tchr1\t695\t-\t+\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            baits.write_text(
                "chr1\t95\t105\t+\nchr1\t495\t505\t-\n",
                encoding="utf-8",
            )
            preys.write_text(
                "chr1\t295\t305\t+\nchr1\t695\t705\t-\n",
                encoding="utf-8",
            )
            indexes = build_contact_indexes(pairs, source="touche")
            bait_anchors = read_bed_anchors(baits)
            prey_anchors = read_bed_anchors(preys)

            result = compute_apa(
                indexes,
                bait_anchors,
                prey_anchors,
                min_distance=150,
                max_distance=250,
                window=20,
                pixels=2,
                shift=0,
            )

            expected_matrix = pl.DataFrame(
                {
                    "bin_label": [20, 10, -10, -20],
                    "-20": [0, 2, 0, 0],
                    "-10": [1, 1, 1, 0],
                    "10": [0, 0, 0, 0],
                    "20": [0, 0, 0, 0],
                }
            )
            expected_bait_signal = pl.DataFrame(
                {"bin_label": [-20, -10, 10, 20], "contacts": [2, 3, 0, 0]}
            )
            expected_prey_signal = pl.DataFrame(
                {"bin_label": [-20, -10, 10, 20], "contacts": [0, 1, 3, 1]}
            )

            assert_frame_equal(result.matrix, expected_matrix)
            assert_frame_equal(result.bait_signal, expected_bait_signal)
            assert_frame_equal(result.prey_signal, expected_prey_signal)


if __name__ == "__main__":
    unittest.main()
