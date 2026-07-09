from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from touche.apa import compare_apa_change, plot_apa_change


class ApaCompareTests(unittest.TestCase):
    def test_compare_apa_change_writes_matrix_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control_apa = tmp_path / "control_apa.csv"
            treatment_apa = tmp_path / "treatment_apa.csv"
            control_baits = tmp_path / "control_baits.csv"
            control_preys = tmp_path / "control_preys.csv"
            treatment_baits = tmp_path / "treatment_baits.csv"
            treatment_preys = tmp_path / "treatment_preys.csv"
            out = tmp_path / "change.svg"
            matrix_out = tmp_path / "change.csv"

            pl.DataFrame({"bin_label": [-1, 1], "-1": [10, 30], "1": [20, 40]}).write_csv(
                control_apa
            )
            pl.DataFrame({"bin_label": [-1, 1], "-1": [20, 60], "1": [40, 80]}).write_csv(
                treatment_apa
            )
            pl.DataFrame({"bin_label": [-1, 1], "contacts": [10, 10]}).write_csv(control_baits)
            pl.DataFrame({"bin_label": [-1, 1], "contacts": [10, 10]}).write_csv(control_preys)
            pl.DataFrame({"bin_label": [-1, 1], "contacts": [20, 20]}).write_csv(treatment_baits)
            pl.DataFrame({"bin_label": [-1, 1], "contacts": [20, 20]}).write_csv(treatment_preys)

            matrix = compare_apa_change(
                control_apa,
                treatment_apa,
                control_baits,
                control_preys,
                treatment_baits,
                treatment_preys,
                bait_count=1,
                prey_count=1,
                out=out,
                matrix_out=matrix_out,
                window=1_000,
                pixels=1,
            )

            values = matrix.drop("bin_label").to_numpy()
            self.assertEqual(values.shape, (2, 2))
            self.assertTrue((values == 1.0).all())
            self.assertTrue(out.exists())
            self.assertTrue(matrix_out.exists())

    def test_plot_apa_change_returns_figure_without_writing(self) -> None:
        matrix = pl.DataFrame({"bin_label": [-1, 1], "-1": [1.0, 2.0], "1": [2.0, 1.0]})

        fig = plot_apa_change(matrix, window=1_000, pixels=1)

        self.assertTrue(hasattr(fig, "savefig"))
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
