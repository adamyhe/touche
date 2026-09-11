from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from touche.apa import ApaResult, compute_apa
from touche.apa_masks import (
    MaskSpec,
    build_masks,
    default_masks,
    read_masks,
    summarize_apa,
    write_masks,
)
from touche.models import ContactIndex


def _apa_result(values: np.ndarray, window: int = 1000, pixels: int = 5) -> ApaResult:
    labels = list(range(-window, 0, window // pixels)) + list(range(window // pixels, window + 1, window // pixels))
    rows = list(reversed(labels))
    frame = pl.DataFrame({"bin_label": rows}).with_columns(
        [pl.Series(str(label), values[:, i]) for i, label in enumerate(labels)]
    )
    signal = pl.DataFrame({"bin_label": labels, "contacts": [0] * len(labels)})
    return ApaResult(matrix=frame, bait_signal=signal, prey_signal=signal, window=window, pixels=pixels)


class MaskGeometryTests(unittest.TestCase):
    def test_masks_are_disjoint_where_they_exclude_each_other_and_cover_the_matrix(self) -> None:
        offsets = np.array([-1000.0, -500.0, -100.0, 100.0, 500.0, 1000.0])
        masks = build_masks(offsets, offsets, default_masks(1000))

        cross_and_ring = masks["dot"] | masks["promoter_stripe"] | masks["enhancer_stripe"] | masks["ring"]
        self.assertFalse((masks["dot"] & masks["promoter_stripe"]).any())
        self.assertFalse((masks["dot"] & masks["enhancer_stripe"]).any())
        self.assertFalse((masks["ring"] & cross_and_ring & ~masks["ring"]).any())
        np.testing.assert_array_equal(masks["all"], cross_and_ring | masks["global_background"])

    def test_dot_selects_the_central_block_for_even_and_odd_matrices(self) -> None:
        for n in (6, 7):
            offsets = np.linspace(-1000.0, 1000.0, n)
            masks = build_masks(offsets, offsets, default_masks(1000, dot_fraction=0.25, ring_fraction=0.4))
            rows, cols = np.where(masks["dot"])
            self.assertTrue(np.all(np.abs(offsets[rows]) <= 250.0), n)
            self.assertTrue(np.all(np.abs(offsets[cols]) <= 250.0), n)

    def test_stripes_run_the_full_length_of_the_matrix(self) -> None:
        offsets = np.linspace(-1000.0, 1000.0, 21)
        masks = build_masks(offsets, offsets, default_masks(1000))

        # Every row is represented in the promoter (column-band) stripe except
        # the rows the dot occupies.
        stripe_rows = set(np.where(masks["promoter_stripe"].any(axis=1))[0])
        dot_rows = set(np.where(masks["dot"].any(axis=1))[0])
        self.assertEqual(stripe_rows | dot_rows, set(range(21)))

    def test_lower_left_is_the_negative_negative_corner(self) -> None:
        offsets = np.linspace(-1000.0, 1000.0, 21)
        masks = build_masks(offsets, offsets, default_masks(1000))
        rows, cols = np.where(masks["lower_left"])

        self.assertTrue(np.all(offsets[rows] <= -800.0))
        self.assertTrue(np.all(offsets[cols] <= -800.0))

    def test_mask_definitions_scale_with_the_window(self) -> None:
        small = build_masks(np.linspace(-1000, 1000, 21), np.linspace(-1000, 1000, 21), default_masks(1000))
        large = build_masks(np.linspace(-10000, 10000, 21), np.linspace(-10000, 10000, 21), default_masks(10000))

        for name in small:
            np.testing.assert_array_equal(small[name], large[name], name)

    def test_excluding_an_undefined_mask_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_masks(np.array([0.0]), np.array([0.0]), [MaskSpec("a", exclude=("missing",))])

    def test_rejects_out_of_range_fractions(self) -> None:
        with self.assertRaises(ValueError):
            default_masks(1000, dot_fraction=0.0)
        with self.assertRaises(ValueError):
            default_masks(1000, dot_fraction=0.3, ring_fraction=0.2)


class SummarizeApaTests(unittest.TestCase):
    def test_focal_signal_produces_enrichment_above_one(self) -> None:
        values = np.ones((10, 10))
        values[4:6, 4:6] = 20.0
        table = summarize_apa(_apa_result(values)).table
        values = dict(zip(table["metric"], table["value"]))

        self.assertGreater(values["central_enrichment"], 1.0)
        self.assertGreater(values["p2ll"], 1.0)
        self.assertGreater(values["p2m"], 1.0)
        self.assertAlmostEqual(values["ring_enrichment"], 1.0)

    def test_stripe_signal_is_separated_from_focal_signal(self) -> None:
        values = np.ones((10, 10))
        values[:, 4:6] = 10.0
        table = summarize_apa(_apa_result(values)).table
        values = dict(zip(table["metric"], table["value"]))

        self.assertAlmostEqual(values["promoter_stripe_enrichment"], 10.0)
        self.assertAlmostEqual(values["enhancer_stripe_enrichment"], 1.0)
        self.assertAlmostEqual(values["dot_to_promoter_stripe"], 1.0)

    def test_every_ratio_reports_its_own_numerator_and_denominator(self) -> None:
        values = np.arange(100, dtype=float).reshape(10, 10)
        table = summarize_apa(_apa_result(values)).table

        ratios = table.filter(pl.col("denominator_mask").is_not_null())
        self.assertGreater(ratios.height, 0)
        for row in ratios.iter_rows(named=True):
            self.assertAlmostEqual(row["value"], row["numerator"] / row["denominator"], places=9)
            self.assertGreater(row["n_pixels_numerator"], 0)
            self.assertGreater(row["n_pixels_denominator"], 0)

    def test_mask_rows_report_sums_and_pixel_counts(self) -> None:
        values = np.ones((10, 10))
        table = summarize_apa(_apa_result(values)).table.filter(pl.col("metric") == "all_mean")

        self.assertEqual(table["numerator"][0], 100.0)
        self.assertEqual(table["n_pixels_numerator"][0], 100)
        self.assertEqual(table["value"][0], 1.0)

    def test_custom_masks_are_recorded_in_metadata(self) -> None:
        specs = [MaskSpec("all"), MaskSpec("tiny", (-100.0, 100.0), (-100.0, 100.0))]
        result = summarize_apa(_apa_result(np.ones((10, 10))), masks=specs, scores=())

        self.assertEqual(
            [row["metric"] for row in result.table.iter_rows(named=True)], ["all_mean", "tiny_mean"]
        )
        self.assertEqual([m["name"] for m in result.info.parameters["masks"]], ["all", "tiny"])

    def test_empty_mask_is_reported_not_raised(self) -> None:
        specs = [MaskSpec("all"), MaskSpec("offscreen", (1e9, 2e9), (1e9, 2e9))]
        result = summarize_apa(_apa_result(np.ones((10, 10))), masks=specs, scores=())

        self.assertEqual(result.table.filter(pl.col("metric") == "offscreen_mean")["n_pixels_numerator"][0], 0)
        self.assertTrue(any("no pixels" in warning for warning in result.info.warnings))

    def test_bootstrap_without_per_chromosome_matrices_warns_and_returns_null_bounds(self) -> None:
        result = summarize_apa(_apa_result(np.ones((10, 10))), bootstrap=100)

        self.assertTrue(result.table["ci_low"].is_null().all())
        self.assertTrue(any("keep_chromosomes" in warning for warning in result.info.warnings))


class ChromosomeBootstrapTests(unittest.TestCase):
    def _indexes(self, chroms: tuple[str, ...], seed: int = 0) -> dict[str, ContactIndex]:
        rng = np.random.default_rng(seed)
        indexes = {}
        for chrom in chroms:
            n = 5000
            pos_a = rng.integers(0, 400_000, n)
            indexes[chrom] = ContactIndex(
                chrom=chrom,
                pos_a=pos_a,
                pos_b=pos_a + rng.integers(50_000, 150_000, n),
                strand_a=np.ones(n, dtype=np.int8),
                strand_b=-np.ones(n, dtype=np.int8),
                mapq_a=np.full(n, 30, dtype=np.int16),
                mapq_b=np.full(n, 30, dtype=np.int16),
            )
        return indexes

    def _anchors(self, chroms: tuple[str, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
        starts = list(range(0, 300_000, 25_000))
        baits = pl.DataFrame(
            {"chr": [c for c in chroms for _ in starts], "start": starts * len(chroms)}
        ).with_columns(
            (pl.col("start") + 1000).alias("end"),
            pl.lit("+").alias("strand"),
            (pl.col("start") + 500).alias("center"),
        )
        preys = baits.with_columns((pl.col("center") + 80_000).alias("center"))
        return baits, preys

    def test_per_chromosome_matrices_sum_to_the_genome_wide_matrix(self) -> None:
        chroms = ("chr1", "chr2", "chr3")
        baits, preys = self._anchors(chroms)
        kwargs = dict(min_distance=50_000, max_distance=120_000, window=2_000, pixels=5)

        pooled = compute_apa(self._indexes(chroms), baits, preys, **kwargs)
        split = compute_apa(self._indexes(chroms), baits, preys, keep_chromosomes=True, **kwargs)

        np.testing.assert_array_equal(
            pooled.matrix.drop("bin_label").to_numpy(), split.matrix.drop("bin_label").to_numpy()
        )
        np.testing.assert_array_equal(
            split.matrix.drop("bin_label").to_numpy(), sum(split.chrom_matrices.values())
        )

    def test_bootstrap_produces_reproducible_intervals_bracketing_the_estimate(self) -> None:
        chroms = tuple(f"chr{i}" for i in range(1, 13))
        baits, preys = self._anchors(chroms)
        result = compute_apa(
            self._indexes(chroms), baits, preys, min_distance=50_000, max_distance=120_000,
            window=2_000, pixels=5, keep_chromosomes=True,
        )

        first = summarize_apa(result, bootstrap=200, seed=1).table
        second = summarize_apa(result, bootstrap=200, seed=1).table

        self.assertEqual(first.to_dicts(), second.to_dicts())
        central = first.filter(pl.col("metric") == "central_enrichment").row(0, named=True)
        self.assertLessEqual(central["ci_low"], central["value"])
        self.assertGreaterEqual(central["ci_high"], central["value"])
        self.assertEqual(summarize_apa(result, bootstrap=200).info.cluster_unit, "chromosome")


class MaskFileTests(unittest.TestCase):
    def test_masks_round_trip_through_json_and_yaml(self) -> None:
        specs = default_masks(10_000)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("masks.json", "masks.yaml"):
                path = write_masks(specs, Path(tmp) / name)
                self.assertEqual(read_masks(path), specs)

    def test_mask_file_may_wrap_the_list_in_a_masks_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masks.json"
            path.write_text('{"masks": [{"name": "all"}]}', encoding="utf-8")
            self.assertEqual(read_masks(path), [MaskSpec("all")])

    def test_malformed_mask_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masks.json"
            path.write_text('{"nope": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_masks(path)


if __name__ == "__main__":
    unittest.main()
