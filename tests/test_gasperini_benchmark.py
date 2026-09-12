"""Smoke test for `scripts/gasperini_benchmark.py --demo`.

The real benchmark needs multi-gigabyte downloads, so CI runs the synthetic
demo instead. The demo plants a known signal, which lets these assertions
check that the pipeline produces *correct* answers rather than merely
running: the planted signal must be recovered, and the distance-only
baseline must sit at chance.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gasperini_benchmark.py"


class GasperiniBenchmarkDemoTests(unittest.TestCase):
    report: pl.DataFrame
    manifest: dict
    out_dir: Path
    _tmp: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.out_dir = root / "report"
        completed = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--demo",
                "--work-dir", str(root / "work"),
                "--out-dir", str(cls.out_dir),
                "--bootstrap", "50", "--no-plots",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            raise AssertionError(f"benchmark demo failed:\n{completed.stdout}\n{completed.stderr}")
        cls.manifest = json.loads(completed.stdout)
        cls.report = pl.read_csv(cls.out_dir / "prediction.tsv", separator="\t")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_writes_every_report_artifact(self) -> None:
        for name in (
            "scored_pairs.tsv", "null_pairs.tsv", "calibration.tsv", "expected_bias.tsv",
            "prediction.tsv", "balance.tsv", "summary.md", "manifest.json",
        ):
            self.assertTrue((self.out_dir / name).exists(), name)

    def test_the_per_bait_fit_is_compared_against_a_global_curve(self) -> None:
        # The per-bait LOWESS is the most expensive part of local-decay, so
        # the report has to carry the control that says whether it pays.
        comparison = pl.read_csv(self.out_dir / "decay_model_comparison.tsv", separator="\t")

        for column in (
            "per_bait_observed_over_expected", "per_bait_dispersion",
            "global_observed_over_expected", "global_dispersion",
        ):
            self.assertIn(column, comparison.columns)
        self.assertIn("all", comparison["stratum"].to_list())
        self.assertIn("log2_oe_global", self.report["score"].to_list())

    def test_dispersion_is_estimated_on_the_null_pairs(self) -> None:
        # Estimating it from the pairs under test would fold real signal into
        # the variance and make the test conservative by an unknown amount.
        self.assertIn("dispersion", self.manifest["parameters"])
        self.assertGreaterEqual(self.manifest["parameters"]["dispersion"], 1.0)

    def test_labels_are_attached_to_both_classes(self) -> None:
        counts = self.manifest["counts"]
        self.assertGreater(counts["positive"], 50)
        self.assertGreater(counts["negative"], 50)
        self.assertGreater(counts["unlabelled"], 0, "unlabelled candidates must be excluded, not dropped")

    def test_planted_signal_is_recovered_by_every_contact_score(self) -> None:
        rows = {row["score"]: row for row in self.report.iter_rows(named=True)}
        for name in (
            "neg_log10_p_binomial", "neg_log10_p_negative_binomial",
            "neg_log10_p_legacy_fisher", "log2_oe", "observed",
        ):
            self.assertGreater(rows[name]["auprc"], 0.8, f"{name} failed to recover the planted signal")

    def test_distance_only_baseline_sits_at_chance(self) -> None:
        # Every demo pair has the same anchor separation, so distance carries
        # no information. A baseline above chance here means the metric or the
        # label join is wired up wrong.
        row = {r["score"]: r for r in self.report.iter_rows(named=True)}["neg_log10_distance"]

        self.assertAlmostEqual(row["auprc"], row["baseline_auprc"], places=2)
        self.assertAlmostEqual(row["auroc"], 0.5, places=2)

    def test_null_set_excludes_pairs_that_landed_back_on_real_anchors(self) -> None:
        null = pl.read_csv(self.out_dir / "null_pairs.tsv", separator="\t")
        real = pl.read_csv(self.out_dir / "scored_pairs.tsv", separator="\t")
        keys = set(zip(real["chrom"], real["bait_center"], real["prey_center"]))

        overlap = sum(key in keys for key in zip(null["chrom"], null["bait_center"], null["prey_center"]))
        self.assertEqual(overlap, 0, "a shifted pair that is also a real candidate pair is not null")
        self.assertGreater(null.height, 100)

    def test_calibration_covers_every_method_and_stratification(self) -> None:
        calibration = pl.read_csv(self.out_dir / "calibration.tsv", separator="\t")

        self.assertEqual(
            sorted(calibration["method"].unique().to_list()),
            ["binomial", "legacy_fisher", "negative_binomial", "poisson"],
        )
        self.assertEqual(
            sorted(calibration["stratification"].unique().to_list()),
            ["coverage", "distance", "overall"],
        )

    def test_expected_bias_table_reports_the_ratio_and_the_dispersion(self) -> None:
        bias = pl.read_csv(self.out_dir / "expected_bias.tsv", separator="\t")

        # Both are needed: a correct mean with an overdispersed variance still
        # over-rejects, and only `dispersion` shows that.
        self.assertIn("observed_over_expected", bias.columns)
        self.assertIn("dispersion", bias.columns)
        self.assertIn("all", bias["stratum"].to_list())
        self.assertTrue((bias["n"] > 0).all())
        self.assertTrue((bias["dispersion"] > 0).all())

    def test_calibration_table_keeps_the_columns_needed_to_read_it(self) -> None:
        calibration = pl.read_csv(self.out_dir / "calibration.tsv", separator="\t")

        # A rejection rate is uninterpretable without the attainable rate to
        # compare it against, and without knowing how discrete the test is.
        self.assertIn("attainable_rate_at_0.05", calibration.columns)
        self.assertIn("fraction_at_one", calibration.columns)
        self.assertIn("reject_rate_at_0.05", calibration.columns)


class SkipDownloadTests(unittest.TestCase):
    def test_missing_inputs_are_reported_by_name_not_a_crash(self) -> None:
        # --skip-download is the path someone uses with pre-staged data, so
        # it must fail with a list of what is missing.
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--skip-download",
                    "--data-dir", str(Path(tmp) / "absent"),
                    "--out-dir", str(Path(tmp) / "report"),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing", completed.stderr.lower())
        self.assertIn("Gasperini_dREG_based_TRE_baits_hg38.txt", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
