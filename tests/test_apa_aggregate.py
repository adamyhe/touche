from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from touche.anchors import read_bed_anchors
from touche.apa import aggregate_apa, compute_apa
from touche.backends import has_numba
from touche.contacts import build_contact_indexes


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

    @unittest.skipUnless(has_numba(), "numba extra is not installed")
    def test_numba_compute_apa_matches_numpy(self) -> None:
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
            kwargs = {
                "min_distance": 150,
                "max_distance": 250,
                "window": 20,
                "pixels": 2,
                "shift": 0,
            }

            numpy_result = compute_apa(
                indexes, bait_anchors, prey_anchors, backend="numpy", **kwargs
            )
            numba_result = compute_apa(
                indexes, bait_anchors, prey_anchors, backend="numba", **kwargs
            )

            assert_frame_equal(numba_result.matrix, numpy_result.matrix)
            assert_frame_equal(numba_result.bait_signal, numpy_result.bait_signal)
            assert_frame_equal(numba_result.prey_signal, numpy_result.prey_signal)
            self.assertEqual(
                numba_result.matrix.drop("bin_label").to_numpy().sum(),
                numpy_result.matrix.drop("bin_label").to_numpy().sum(),
            )


if __name__ == "__main__":
    unittest.main()
