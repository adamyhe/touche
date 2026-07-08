from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from touche.local_decay import assign_pair_types, call_local_decay, plot_pair_type_distribution


class LocalDecayTests(unittest.TestCase):
    def test_call_local_decay_writes_reference_columns_and_counts_observed_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            out = tmp_path / "calls.tsv"
            baits.write_text("chr1\t10000\n", encoding="utf-8")
            preys.write_text("chr1\t6000\nchr1\t14000\n", encoding="utf-8")
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t9950\tchr1\t14020\t+\t-\tUU\t30\t30",
                        "chr1\t9990\tchr1\t14080\t+\t-\tUU\t30\t30",
                        "chr1\t6050\tchr1\t10020\t+\t-\tUU\t30\t30",
                        "chr1\t3000\tchr1\t7000\t+\t-\tUU\t30\t30",
                        "chr1\t12000\tchr1\t17000\t+\t-\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            calls = call_local_decay(
                baits,
                preys,
                pairs,
                out,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
            )

            self.assertEqual(calls.shape, (2, 9))
            self.assertEqual(calls["chr"].tolist(), ["chr1", "chr1"])
            self.assertEqual(calls["observed"].tolist(), [1, 2])
            self.assertTrue((calls["expected"] >= 0).all())
            self.assertTrue(out.exists())
            written = pd.read_csv(out, sep="\t", header=None)
            self.assertEqual(written.shape, (2, 9))

    def test_assign_pair_types_uses_functional_and_nonfunctional_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contacts = tmp_path / "contacts.tsv"
            functional = tmp_path / "functional.csv"
            nonfunctional = tmp_path / "nonfunctional.csv"
            out = tmp_path / "assigned.tsv"
            contacts.write_text(
                "\n".join(
                    [
                        "chr1\t100\t200\t100\t0.01\t5\t2\t0\t0",
                        "chr1\t110\t210\t100\t0.02\t4\t2\t0\t0",
                        "chr1\t120\t220\t100\t0.03\t3\t2\t0\t0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            functional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,200,100\n",
                encoding="utf-8",
            )
            nonfunctional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,210,110\n",
                encoding="utf-8",
            )

            assigned = assign_pair_types(contacts, functional, nonfunctional, out)

            self.assertEqual(assigned["PosNeg"].tolist(), ["positive", "negative", "other"])
            written = pd.read_csv(out, sep="\t", index_col=0)
            self.assertEqual(written["distance"].tolist(), [100, 100, 100])

    def test_plot_pair_type_distribution_writes_plot_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            assignments = tmp_path / "assigned.tsv"
            figure = tmp_path / "plot.svg"
            table = tmp_path / "plot-data.tsv"
            pd.DataFrame(
                {
                    "target_site.chr": ["chr1", "chr1", "chr1"],
                    "target_site.center": [100, 110, 120],
                    "target_promoter.center": [200, 210, 220],
                    "directional_distance": [100, 100, 100],
                    "observed": [8, 4, 6],
                    "expected": [2, 2, 3],
                    "distance": [20_000, 20_000, 10_000],
                    "PosNeg": ["positive", "negative", "other"],
                }
            ).to_csv(assignments, sep="\t")

            plot_data = plot_pair_type_distribution(
                assignments,
                figure,
                min_contacts=1,
                min_distance=15_000,
                plot_table_out=table,
            )

            self.assertEqual(len(plot_data), 2)
            self.assertTrue(figure.exists())
            self.assertTrue(table.exists())


if __name__ == "__main__":
    unittest.main()
