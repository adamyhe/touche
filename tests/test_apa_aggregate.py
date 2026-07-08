from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from touche.apa import aggregate_apa


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
            matrix = pd.read_csv(outputs["matrix"], index_col=0)
            self.assertEqual(int(matrix.to_numpy().sum()), 3)
            bait_signal = pd.read_csv(outputs["baits_signal"], index_col=0)
            prey_signal = pd.read_csv(outputs["preys_signal"], index_col=0)
            self.assertEqual(int(bait_signal["contacts"].sum()), 3)
            self.assertEqual(int(prey_signal["contacts"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
