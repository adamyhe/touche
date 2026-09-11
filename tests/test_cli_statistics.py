"""End-to-end CLI coverage for the statistics and interoperability subcommands."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl

from touche.cli.main import build_parser


def _run(argv: list[str]) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        args.func(args)
    return json.loads(buffer.getvalue())


def _fixture(tmp: Path) -> None:
    rng = np.random.default_rng(0)
    positions = np.sort(rng.integers(0, 600_000, 20_000))
    for name, seed in (("contacts.pairs", 1), ("contacts2.pairs", 2)):
        distances = np.random.default_rng(seed).integers(20_000, 200_000, positions.size)
        (tmp / name).write_text(
            "\n".join(
                f"chr1\t{a}\tchr1\t{a + d}\t+\t-\tUU\t30\t30" for a, d in zip(positions, distances)
            )
            + "\n",
            encoding="utf-8",
        )
    (tmp / "baits.bed").write_text(
        "".join(f"chr1\t{s}\t{s + 1000}\t+\n" for s in range(0, 500_000, 25_000)), encoding="utf-8"
    )
    (tmp / "preys.bed").write_text(
        "".join(f"chr1\t{s}\t{s + 1000}\t-\n" for s in range(12_000, 512_000, 25_000)),
        encoding="utf-8",
    )
    (tmp / "baits.tsv").write_text(
        "".join(f"chr1\t{s + 500}\n" for s in range(0, 500_000, 25_000)), encoding="utf-8"
    )
    (tmp / "preys.tsv").write_text(
        "".join(f"chr1\t{s + 500}\n" for s in range(12_000, 512_000, 25_000)), encoding="utf-8"
    )


BACKGROUND_ARGS = [
    "--min-distance", "50000", "--max-distance", "200000", "--window", "1000",
    "--min-bg-distance", "20000", "--max-bg-distance", "100000",
]


class PairsCommandTests(unittest.TestCase):
    def test_build_import_and_annotate_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)

            built = _run(
                ["pairs", "build", "--baits", str(tmp_path / "baits.bed"),
                 "--preys", str(tmp_path / "preys.bed"), "--out", str(tmp_path / "pairs.bedpe"),
                 "--min-distance", "50000", "--max-distance", "200000"]
            )
            (tmp_path / "fithic.txt").write_text(
                "chr1\tfragmentMid1\tchr2\tfragmentMid2\tcontactCount\tp-value\tq-value\n"
                "chr1\t500\tchr1\t62500\t42\t1e-8\t1e-06\n",
                encoding="utf-8",
            )
            imported = _run(
                ["pairs", "import-calls", "--input", str(tmp_path / "fithic.txt"),
                 "--format", "fithic2", "--out", str(tmp_path / "calls.tsv"),
                 "--bedpe-out", str(tmp_path / "calls.bedpe")]
            )
            annotated = _run(
                ["pairs", "annotate", "--pairs", str(tmp_path / "pairs.bedpe"),
                 "--calls", str(tmp_path / "calls.bedpe"), "--format", "bedpe",
                 "--out", str(tmp_path / "annotated.tsv"), "--slop", "1000"]
            )
            table = pl.read_csv(tmp_path / "annotated.tsv", separator="\t")

        self.assertEqual(built["pairs"], 180)
        self.assertEqual(imported["calls"], 1)
        self.assertEqual(imported["with_q_value"], 1)
        self.assertEqual(annotated["matched"], 1)
        self.assertIn("loop_matched", table.columns)

    def test_column_override_is_parsed_from_the_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "odd.txt").write_text(
                "seq1\tmid1\tseq2\tmid2\tn\nchr1\t500\tchr1\t62500\t42\n", encoding="utf-8"
            )
            summary = _run(
                ["pairs", "import-calls", "--input", str(tmp_path / "odd.txt"),
                 "--format", "fithic2", "--out", str(tmp_path / "calls.tsv"),
                 "--column", "bait_chrom=seq1", "--column", "bait_start=mid1",
                 "--column", "prey_chrom=seq2", "--column", "prey_start=mid2",
                 "--column", "observed=n", "--method", "custom_run"]
            )
            table = pl.read_csv(tmp_path / "calls.tsv", separator="\t")

        self.assertEqual(summary["calls"], 1)
        self.assertEqual(table["method"].to_list(), ["custom_run"])


class LocalDecayStatisticsCommandTests(unittest.TestCase):
    def _call(self, tmp: Path, *extra: str) -> dict:
        return _run(
            ["local-decay", "call", "--baits", str(tmp / "baits.tsv"), "--preys", str(tmp / "preys.tsv"),
             "--pairs", str(tmp / "contacts.pairs"), "--out", str(tmp / "ld.tsv"),
             "--dist", "300000", "--cap", "500", "--min-distance", "20000", *extra]
        )

    def test_tidy_schema_and_binomial_method_reach_the_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)

            summary = self._call(tmp_path, "--method", "binomial", "--schema", "tidy")
            table = pl.read_csv(tmp_path / "ld.tsv", separator="\t")
            metadata = json.loads((tmp_path / "ld.tsv.meta.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["method"], "binomial")
        self.assertEqual(summary["schema"], "tidy")
        self.assertIn("q_value", table.columns)
        self.assertEqual(metadata["method"], "binomial")

    def test_default_call_still_writes_the_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)

            summary = self._call(tmp_path)
            table = pl.read_csv(tmp_path / "ld.tsv", separator="\t", has_header=False)

        self.assertEqual(summary["method"], "legacy_fisher")
        self.assertEqual(table.width, 9)
        self.assertFalse((tmp_path / "ld.tsv.meta.json").exists())

    def test_test_subcommand_retests_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            self._call(tmp_path, "--method", "binomial", "--schema", "tidy")

            summary = _run(
                ["local-decay", "test", "--calls", str(tmp_path / "ld.tsv"),
                 "--out", str(tmp_path / "retested.tsv"), "--method", "poisson"]
            )
            table = pl.read_csv(tmp_path / "retested.tsv", separator="\t")
            self.assertTrue((tmp_path / "retested.tsv.meta.json").exists())

        self.assertEqual(summary["method"], "poisson")
        self.assertEqual(table["method"].unique().to_list(), ["poisson"])

    def test_calibration_subcommand_reports_rejection_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            self._call(tmp_path, "--method", "binomial", "--schema", "tidy")

            summary = _run(
                ["local-decay", "calibration", "--calls", str(tmp_path / "ld.tsv"),
                 "--out", str(tmp_path / "calib.tsv")]
            )

        stratum = summary["strata"][0]
        self.assertGreater(stratum["n_tested"], 0)
        self.assertIn("reject_rate_at_0.05", stratum)

    def test_compare_groups_reports_an_effect_size_and_clustered_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            self._call(tmp_path, "--method", "binomial", "--schema", "tidy")
            table = pl.read_csv(tmp_path / "ld.tsv", separator="\t")
            labels = np.where(np.random.default_rng(0).random(table.height) < 0.3, "positive", "negative")
            table.with_columns(pl.Series("PosNeg", labels)).write_csv(
                tmp_path / "labelled.tsv", separator="\t"
            )

            summary = _run(
                ["local-decay", "compare-groups", "--table", str(tmp_path / "labelled.tsv"),
                 "--value-col", "log2_oe", "--group-col", "PosNeg", "--cluster", "bait_id",
                 "--bootstrap", "100", "--out", str(tmp_path / "cmp.tsv")]
            )

        self.assertEqual(summary["cluster_unit"], "bait_id")
        self.assertIn("cliffs_delta", summary["result"])
        self.assertIn("ci_low", summary["result"])


class ApaAndBackgroundCommandTests(unittest.TestCase):
    def test_apa_aggregate_with_a_pair_list_writes_a_bootstrapped_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            _run(
                ["pairs", "build", "--baits", str(tmp_path / "baits.bed"),
                 "--preys", str(tmp_path / "preys.bed"), "--out", str(tmp_path / "pairs.bedpe"),
                 "--min-distance", "50000", "--max-distance", "200000"]
            )

            outputs = _run(
                ["apa", "aggregate", "--pairs", str(tmp_path / "contacts.pairs"),
                 "--pairs-list", str(tmp_path / "pairs.bedpe"), "--min-distance", "50000",
                 "--max-distance", "200000", "--window", "5000", "--pixels", "10",
                 "--out-dir", str(tmp_path / "apa"), "--summary-out", str(tmp_path / "apa/summary.tsv"),
                 "--bootstrap", "25"]
            )
            summary = pl.read_csv(outputs["summary"], separator="\t")
            self.assertTrue(Path(outputs["summary_metadata"]).exists())

        self.assertIn("p2ll", summary["metric"].to_list())

    def test_apa_summarize_scores_a_written_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            _run(
                ["apa", "aggregate", "--pairs", str(tmp_path / "contacts.pairs"),
                 "--baits", str(tmp_path / "baits.bed"), "--preys", str(tmp_path / "preys.bed"),
                 "--min-distance", "50000", "--max-distance", "200000", "--window", "5000",
                 "--pixels", "10", "--out-dir", str(tmp_path / "apa")]
            )

            summary = _run(
                ["apa", "summarize", "--matrix", str(tmp_path / "apa/AggMat.csv"),
                 "--out", str(tmp_path / "summary.tsv")]
            )

        self.assertIn("central_enrichment", summary["scores"])
        self.assertEqual(summary["method"], "apa_mask")

    def test_background_count_requires_anchors_or_a_pair_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)

            with self.assertRaises(SystemExit):
                _run(
                    ["background", "count", "--pairs", str(tmp_path / "contacts.pairs"),
                     "--out", str(tmp_path / "bg.tsv"), *BACKGROUND_ARGS]
                )

    def test_background_diff_writes_q_values_and_zero_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            _run(
                ["pairs", "build", "--baits", str(tmp_path / "baits.bed"),
                 "--preys", str(tmp_path / "preys.bed"), "--out", str(tmp_path / "pairs.bedpe"),
                 "--min-distance", "50000", "--max-distance", "200000"]
            )
            for source, out in (("contacts.pairs", "bg1.tsv"), ("contacts2.pairs", "bg2.tsv")):
                _run(
                    ["background", "count", "--pairs", str(tmp_path / source),
                     "--pairs-list", str(tmp_path / "pairs.bedpe"), "--out", str(tmp_path / out),
                     *BACKGROUND_ARGS]
                )

            summary = _run(
                ["background", "diff", "--control", f"dmso={tmp_path / 'bg1.tsv'}",
                 "--treatment", f"flv={tmp_path / 'bg2.tsv'}", "--out", str(tmp_path / "diff.tsv")]
            )
            table = pl.read_csv(tmp_path / "diff.tsv", separator="\t")

        self.assertEqual(summary["inference_class"], "technical")
        self.assertIn("n_zero_positive", summary["parameters"])
        self.assertIn("q_value", table.columns)

    def test_background_compare_scale_and_zero_policy_reach_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _fixture(tmp_path)
            _run(
                ["pairs", "build", "--baits", str(tmp_path / "baits.bed"),
                 "--preys", str(tmp_path / "preys.bed"), "--out", str(tmp_path / "pairs.bedpe"),
                 "--min-distance", "50000", "--max-distance", "200000"]
            )
            for source, out in (("contacts.pairs", "bg1.tsv"), ("contacts2.pairs", "bg2.tsv")):
                _run(
                    ["background", "count", "--pairs", str(tmp_path / source),
                     "--pairs-list", str(tmp_path / "pairs.bedpe"), "--out", str(tmp_path / out),
                     *BACKGROUND_ARGS]
                )

            kept = _run(
                ["background", "compare", "--control", f"dmso={tmp_path / 'bg1.tsv'}",
                 "--treatments", f"flv={tmp_path / 'bg2.tsv'}", "--depths", "dmso=1000000000",
                 "flv=1000000000", "--min-ep-cpb", "0", "--zero-policy", "keep",
                 "--scale", "per_billion", "--table-out", str(tmp_path / "kept.tsv")]
            )
            dropped = _run(
                ["background", "compare", "--control", f"dmso={tmp_path / 'bg1.tsv'}",
                 "--treatments", f"flv={tmp_path / 'bg2.tsv'}", "--depths", "dmso=1000000000",
                 "flv=1000000000", "--min-ep-cpb", "0", "--table-out", str(tmp_path / "dropped.tsv")]
            )

        self.assertEqual(kept["zero_policy"], "keep")
        self.assertEqual(kept["scale"], "per_billion")
        self.assertGreater(kept["rows"], dropped["rows"], "keeping zeros must retain more pairs")


if __name__ == "__main__":
    unittest.main()
