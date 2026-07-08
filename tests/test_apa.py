from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

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

            pd.DataFrame([[10, 20], [30, 40]], index=[-1, 1], columns=[-1, 1]).to_csv(control_apa)
            pd.DataFrame([[20, 40], [60, 80]], index=[-1, 1], columns=[-1, 1]).to_csv(
                treatment_apa
            )
            pd.DataFrame({"contacts": [10, 10]}, index=[-1, 1]).to_csv(control_baits)
            pd.DataFrame({"contacts": [10, 10]}, index=[-1, 1]).to_csv(control_preys)
            pd.DataFrame({"contacts": [20, 20]}, index=[-1, 1]).to_csv(treatment_baits)
            pd.DataFrame({"contacts": [20, 20]}, index=[-1, 1]).to_csv(treatment_preys)

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

            self.assertEqual(matrix.shape, (2, 2))
            self.assertTrue((matrix == 1.0).all().all())
            self.assertTrue(out.exists())
            self.assertTrue(matrix_out.exists())

    def test_plot_apa_change_returns_figure_without_writing(self) -> None:
        matrix = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=[-1, 1], columns=[-1, 1])

        fig = plot_apa_change(matrix, window=1_000, pixels=1)

        self.assertTrue(hasattr(fig, "savefig"))
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
