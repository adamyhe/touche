from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from touche.adapters import (
    CALL_COLUMNS,
    LOOP_FORMATS,
    annotate_pairs,
    read_loop_calls,
    write_loop_calls,
)
from touche.pairs import build_pair_table

FITHIC = (
    "chr1\tfragmentMid1\tchr2\tfragmentMid2\tcontactCount\tp-value\tq-value\tbias1\tbias2\n"
    "chr1\t1100\tchr1\t50100\t42\t1e-8\t1e-06\t1.0\t1.0\n"
    "chr1\t1100\tchr1\t90100\t5\t0.3\t0.6\t1.0\t1.0\n"
)
BEDPE = "chr1\t900\t1300\tchr1\t49900\t50300\tL1\t0.9\t+\t-\n"


def _write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def _pairs() -> pl.DataFrame:
    baits = pl.DataFrame({"chr": ["chr1"], "start": [1000], "end": [1200], "strand": ["+"]})
    preys = pl.DataFrame(
        {"chr": ["chr1", "chr1"], "start": [50000, 90000], "end": [50200, 90200], "strand": [".", "."]}
    )
    add_center = ((pl.col("start") + pl.col("end")) // 2).alias("center")
    return build_pair_table(baits.with_columns(add_center), preys.with_columns(add_center), min_distance=1_000)


class FormatRegistryTests(unittest.TestCase):
    def test_every_format_declares_the_required_anchor_columns(self) -> None:
        for name, spec in LOOP_FORMATS.items():
            for field in spec.required:
                self.assertIn(field, spec.columns, f"{name} cannot locate {field}")

    def test_unknown_format_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "x.bedpe", BEDPE)
            with self.assertRaises(ValueError):
                read_loop_calls(path, format="hiccups2")


class ReadLoopCallsTests(unittest.TestCase):
    def test_fithic2_point_anchors_land_on_the_reported_midpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "f.txt", FITHIC), format="fithic2")

        self.assertEqual(calls.columns, CALL_COLUMNS)
        self.assertEqual(calls["bait_center"].to_list(), [1100, 1100])
        self.assertEqual(calls["prey_center"].to_list(), [50100, 90100])
        self.assertEqual(calls["observed"].to_list(), [42.0, 5.0])
        self.assertAlmostEqual(calls["p_value"][0], 1e-8)
        self.assertEqual(calls["source_format"].unique().to_list(), ["fithic2"])

    def test_bedpe_reads_positionally_and_keeps_its_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "l.bedpe", BEDPE), format="bedpe")

        self.assertEqual(calls["bait_start"].to_list(), [900])
        self.assertEqual(calls["bait_end"].to_list(), [1300])
        self.assertEqual(calls["score"].to_list(), [0.9])
        self.assertTrue(calls["p_value"].is_null().all())

    def test_peakachu_probability_is_kept_as_a_score_not_a_p_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "p.bedpe", "chr1\t900\t1300\tchr1\t49900\t50300\t0.93\n")
            calls = read_loop_calls(path, format="peakachu")

        self.assertEqual(calls["score"].to_list(), [0.93])
        self.assertTrue(calls["p_value"].is_null().all())
        self.assertTrue(calls["q_value"].is_null().all())

    def test_chromosight_comma_separated_output(self) -> None:
        text = (
            "chrom1,start1,end1,chrom2,start2,end2,score,pvalue,qvalue\n"
            "chr1,900,1300,chr1,49900,50300,0.7,0.001,0.01\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "c.csv", text), format="chromosight")

        self.assertEqual(calls["score"].to_list(), [0.7])
        self.assertAlmostEqual(calls["q_value"][0], 0.01)

    def test_mustache_header_names(self) -> None:
        text = (
            "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n"
            "chr1\t900\t1300\tchr1\t49900\t50300\t0.002\t5\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "m.tsv", text), format="mustache")

        self.assertAlmostEqual(calls["q_value"][0], 0.002)
        self.assertEqual(calls["score"].to_list(), [5.0])

    def test_hiccups_header_names_are_matched_case_insensitively(self) -> None:
        text = (
            "chr1\tx1\tx2\tchr2\ty1\ty2\tcolor\to\texpectedDonut\tfdrDonut\n"
            "chr1\t900\t1300\tchr1\t49900\t50300\t0,0,0\t30\t4.0\t0.001\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "h.bedpe", text), format="hiccups")

        self.assertEqual(calls["observed"].to_list(), [30.0])
        self.assertEqual(calls["expected"].to_list(), [4.0])
        self.assertAlmostEqual(calls["q_value"][0], 0.001)

    def test_negative_log_significance_is_converted_back_to_probabilities(self) -> None:
        text = (
            "chr1\tstart1\tend1\tchr2\tstart2\tend2\tobserved_interactions\texp_interactions"
            "\tneg_ln_p_val\tneg_ln_q_val\n"
            "chr1\t900\t1300\tchr1\t49900\t50300\t42\t5.0\t11.5129\t6.9078\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "mx.txt", text), format="maxhic")

        self.assertAlmostEqual(calls["p_value"][0], 1e-5, places=8)
        self.assertAlmostEqual(calls["q_value"][0], 1e-3, places=6)

    def test_missing_columns_raise_an_error_naming_the_actual_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "bad.txt", "a\tb\n1\t2\n")
            with self.assertRaises(ValueError) as raised:
                read_loop_calls(path, format="fithic2")

        message = str(raised.exception)
        self.assertIn("Header found: a, b", message)
        self.assertIn("columns=", message)

    def test_column_override_rescues_an_unexpected_header(self) -> None:
        text = "seq1\tmid1\tseq2\tmid2\tn\n" "chr1\t1100\tchr1\t50100\t42\n"
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(
                _write(Path(tmp), "odd.txt", text),
                format="fithic2",
                columns={
                    "bait_chrom": "seq1", "bait_start": "mid1",
                    "prey_chrom": "seq2", "prey_start": "mid2", "observed": "n",
                },
            )

        self.assertEqual(calls["observed"].to_list(), [42.0])

    def test_non_numeric_score_column_raises_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "l.bedpe", "chr1\t900\t1300\tchr1\t49900\t50300\tname\n")
            with self.assertRaises(ValueError) as raised:
                read_loop_calls(path, format="peakachu")

        self.assertIn("not numeric", str(raised.exception))

    def test_trans_calls_can_be_excluded(self) -> None:
        text = BEDPE + "chr1\t900\t1300\tchr2\t49900\t50300\tL2\t0.5\t+\t-\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "t.bedpe", text)
            self.assertEqual(read_loop_calls(path, format="bedpe").height, 2)
            self.assertEqual(read_loop_calls(path, format="bedpe", cis_only=True).height, 1)

    def test_method_label_can_be_overridden_for_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(
                _write(Path(tmp), "l.bedpe", BEDPE), format="bedpe", method="mustache_run3"
            )

        self.assertEqual(calls["method"].to_list(), ["mustache_run3"])
        self.assertEqual(calls["source_format"].to_list(), ["bedpe"])


class AnnotatePairsTests(unittest.TestCase):
    def test_binned_calls_match_point_anchored_pairs_by_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "l.bedpe", BEDPE), format="bedpe")
            annotated = annotate_pairs(_pairs(), calls)

        self.assertEqual(annotated["loop_matched"].to_list(), [True, False])
        self.assertEqual(annotated["loop_score"].to_list(), [0.9, None])

    def test_anchor_order_does_not_affect_matching(self) -> None:
        swapped = "chr1\t49900\t50300\tchr1\t900\t1300\tL1\t0.9\t-\t+\n"
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "s.bedpe", swapped), format="bedpe")
            annotated = annotate_pairs(_pairs(), calls)

        self.assertEqual(annotated["loop_matched"].to_list(), [True, False])

    def test_slop_widens_the_match_window(self) -> None:
        near_miss = "chr1\t1300\t1400\tchr1\t49900\t50300\tL1\t0.9\t+\t-\n"
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "n.bedpe", near_miss), format="bedpe")
            pairs = _pairs()

            self.assertEqual(annotate_pairs(pairs, calls)["loop_matched"].to_list(), [False, False])
            self.assertEqual(
                annotate_pairs(pairs, calls, slop=500)["loop_matched"].to_list(), [True, False]
            )

    def test_overlapping_calls_resolve_deterministically_to_the_best_q_value(self) -> None:
        text = (
            "chr1\tfragmentMid1\tchr2\tfragmentMid2\tcontactCount\tp-value\tq-value\n"
            "chr1\t1100\tchr1\t50100\t10\t0.01\t0.10\n"
            "chr1\t1100\tchr1\t50100\t42\t0.001\t0.01\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            calls = read_loop_calls(_write(Path(tmp), "f.txt", text), format="fithic2")
            annotated = annotate_pairs(_pairs(), calls)

        self.assertAlmostEqual(annotated["loop_q_value"][0], 0.01)
        self.assertEqual(annotated["loop_observed"][0], 42.0)

    def test_chromosome_naming_mismatch_matches_nothing_rather_than_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "l.bedpe", BEDPE.replace("chr1", "1"))
            annotated = annotate_pairs(_pairs(), read_loop_calls(path, format="bedpe"))

        self.assertFalse(any(annotated["loop_matched"].to_list()))

    def test_empty_inputs_are_handled(self) -> None:
        pairs = _pairs()
        empty = pairs.head(0).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("score"), pl.lit(None, dtype=pl.Utf8).alias("method")
        )

        self.assertFalse(any(annotate_pairs(pairs, empty)["loop_matched"].to_list()))
        self.assertEqual(annotate_pairs(pairs.head(0), empty).height, 0)

    def test_rejects_negative_slop(self) -> None:
        with self.assertRaises(ValueError):
            annotate_pairs(_pairs(), _pairs(), slop=-1)


class WriteLoopCallsTests(unittest.TestCase):
    def test_bedpe_export_round_trips_coordinates_and_pair_ids(self) -> None:
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_loop_calls(pairs, Path(tmp) / "out.bedpe")
            reloaded = read_loop_calls(path, format="bedpe")

        self.assertEqual(reloaded["pair_id"].to_list(), pairs["pair_id"].to_list())
        self.assertEqual(reloaded["bait_start"].to_list(), pairs["bait_start"].to_list())

    def test_fithic2_export_round_trips_through_the_fithic2_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calls = read_loop_calls(_write(tmp_path, "f.txt", FITHIC), format="fithic2")
            path = write_loop_calls(calls, tmp_path / "out.tsv", format="fithic2")
            reloaded = read_loop_calls(path, format="fithic2")

        self.assertEqual(reloaded["pair_id"].to_list(), calls["pair_id"].to_list())
        np.testing.assert_allclose(reloaded["p_value"].to_numpy(), calls["p_value"].to_numpy())
        np.testing.assert_allclose(reloaded["observed"].to_numpy(), calls["observed"].to_numpy())

    def test_unknown_export_format_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_loop_calls(_pairs(), Path(tmp) / "out.cool", format="cool")


if __name__ == "__main__":
    unittest.main()
