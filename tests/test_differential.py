from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from touche.differential import test_background_change, zero_class_counts
from touche.models import NamedPath
from touche.stats import adjust_pvalues

test_background_change.__test__ = False


def _counts_file(tmp: Path, name: str, rows: list[tuple[str, int, int, int, int]]) -> NamedPath:
    path = tmp / f"{name}.tsv"
    path.write_text("\n".join("\t".join(str(v) for v in row) for row in rows) + "\n", encoding="utf-8")
    return NamedPath(name=name, path=path)


class ZeroClassTests(unittest.TestCase):
    def test_counts_every_zero_combination(self) -> None:
        counts = zero_class_counts(np.array([0, 0, 5, 5]), np.array([0, 3, 0, 7]))

        self.assertEqual(
            counts,
            {"zero_zero": 1, "zero_positive": 1, "positive_zero": 1, "positive_positive": 1},
        )


class TestBackgroundChangeTests(unittest.TestCase):
    def _pair(self, tmp: Path) -> tuple[NamedPath, NamedPath]:
        control = _counts_file(
            tmp,
            "dmso",
            [
                ("chr1", 1000, 20000, 10, 100),
                ("chr1", 1000, 40000, 0, 100),
                ("chr1", 1000, 60000, 50, 100),
                ("chr1", 1000, 80000, 0, 100),
            ],
        )
        treatment = _counts_file(
            tmp,
            "flv",
            [
                ("chr1", 1000, 20000, 60, 100),
                ("chr1", 1000, 40000, 40, 100),
                ("chr1", 1000, 60000, 0, 100),
                ("chr1", 1000, 80000, 0, 100),
            ],
        )
        return control, treatment

    def test_complete_gains_and_losses_survive_into_the_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            result = test_background_change(control, treatment)

        self.assertEqual(result.table.height, 4)
        changes = result.table["log2_ratio_change"].to_list()
        self.assertTrue(np.isposinf(changes[1]), "complete gain must stay +inf, not be dropped")
        self.assertTrue(np.isneginf(changes[2]), "complete loss must stay -inf, not be dropped")
        self.assertTrue(np.isnan(changes[3]), "zero/zero is undefined, not zero")

    def test_display_column_is_finite_where_the_raw_column_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            result = test_background_change(control, treatment, display_pseudocount=1.0)

        self.assertTrue(np.isfinite(result.table["log2_ratio_change_display"].to_numpy()).all())
        self.assertEqual(result.info.parameters["display_pseudocount"], 1.0)

    def test_zero_classes_are_reported_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            result = test_background_change(control, treatment)

        self.assertEqual(result.info.parameters["n_zero_zero"], 1)
        self.assertEqual(result.info.parameters["n_zero_positive"], 1)
        self.assertEqual(result.info.parameters["n_positive_zero"], 1)
        self.assertEqual(result.info.parameters["n_positive_positive"], 1)

    def test_result_is_labelled_technical_inference_with_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            result = test_background_change(control, treatment)

        self.assertEqual(result.info.inference_class, "technical")
        self.assertTrue(any("not evidence of biological variation" in w for w in result.info.warnings))

    def test_odds_ratio_direction_and_q_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            result = test_background_change(control, treatment)

        row = result.table.row(0, named=True)
        self.assertGreater(row["log_odds_ratio"], 0.0, "EP up in treatment should give a positive log OR")
        self.assertLess(row["p_value"], 0.01)
        np.testing.assert_allclose(
            result.table["q_value"].to_numpy(),
            adjust_pvalues(result.table["p_value"].to_numpy()),
            equal_nan=True,
        )

    def test_no_change_relative_to_background_gives_an_odds_ratio_near_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control = _counts_file(tmp_path, "a", [("chr1", 1000, 20000, 10, 100)])
            # Both counts doubled: EP rose, but so did its own background.
            treatment = _counts_file(tmp_path, "b", [("chr1", 1000, 20000, 20, 200)])
            row = test_background_change(control, treatment).table.row(0, named=True)

        self.assertAlmostEqual(row["log_odds_ratio"], 0.0, places=1)
        self.assertGreater(row["p_value"], 0.5)

    def test_fisher_and_wald_agree_in_direction_and_broadly_in_magnitude(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            wald = test_background_change(control, treatment, method="wald").table
            fisher = test_background_change(control, treatment, method="fisher").table

        self.assertEqual(fisher["method"].unique().to_list(), ["odds_ratio_fisher"])
        both = np.isfinite(wald["p_value"].to_numpy()) & np.isfinite(fisher["p_value"].to_numpy())
        self.assertTrue(((wald["p_value"].to_numpy() < 0.05) == (fisher["p_value"].to_numpy() < 0.05))[both].all())

    def test_pairs_missing_from_one_input_are_reported_not_filled_with_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control = _counts_file(
                tmp_path, "a", [("chr1", 1000, 20000, 10, 100), ("chr1", 1000, 90000, 5, 100)]
            )
            treatment = _counts_file(tmp_path, "b", [("chr1", 1000, 20000, 20, 100)])
            result = test_background_change(control, treatment)

        self.assertEqual(result.table.height, 1)
        self.assertEqual(result.info.filtered["not_shared"], 1)
        self.assertTrue(any("only one input" in w for w in result.info.warnings))

    def test_writes_a_table_and_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control, treatment = self._pair(tmp_path)
            out = tmp_path / "diff.tsv"
            test_background_change(control, treatment, out_path=out)

            written = pl.read_csv(out, separator="\t")
            self.assertTrue((tmp_path / "diff.tsv.meta.json").exists())

        self.assertIn("q_value", written.columns)
        self.assertIn("pair_id", written.columns)

    def test_rejects_bad_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control, treatment = self._pair(Path(tmp))
            with self.assertRaises(ValueError):
                test_background_change(control, treatment, method="glm")
            with self.assertRaises(ValueError):
                test_background_change(control, treatment, fdr="storey")
            with self.assertRaises(ValueError):
                test_background_change(control, treatment, correction=-1.0)


if __name__ == "__main__":
    unittest.main()
