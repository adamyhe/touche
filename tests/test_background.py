from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from touche.background import (
    _safe_kde,
    compare_background_ratios,
    parse_named_depth,
    parse_named_path,
    plot_background_scatter,
)
from touche.models import NamedPath


class BackgroundCompareTests(unittest.TestCase):
    def test_parse_named_values(self) -> None:
        named_path = parse_named_path("DMSO=/tmp/dmso.tsv")
        named_depth = parse_named_depth("DMSO=123")
        self.assertEqual(named_path.name, "DMSO")
        self.assertEqual(str(named_path.path), "/tmp/dmso.tsv")
        self.assertEqual(named_depth.depth, 123)

    def test_compare_background_ratios_filters_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dmso = tmp_path / "dmso.tsv"
            flv = tmp_path / "flv.tsv"
            trp = tmp_path / "trp.tsv"
            out_dir = tmp_path / "plots"
            table = tmp_path / "merged.tsv"
            dmso.write_text(
                "chr1\t100\t200\t10\t5\n"
                "chr1\t110\t210\t0\t5\n"
                "chr1\t120\t220\t2\t0\n",
                encoding="utf-8",
            )
            flv.write_text(
                "chr1\t100\t200\t20\t5\n"
                "chr1\t110\t210\t1\t5\n"
                "chr1\t120\t220\t2\t1\n",
                encoding="utf-8",
            )
            trp.write_text(
                "chr1\t100\t200\t30\t5\n"
                "chr1\t110\t210\t1\t5\n"
                "chr1\t120\t220\t2\t1\n",
                encoding="utf-8",
            )

            merged, plots = compare_background_ratios(
                NamedPath("DMSO", dmso),
                [NamedPath("FLV", flv), NamedPath("TRP", trp)],
                {"DMSO": 10_000_000_000, "FLV": 10_000_000_000, "TRP": 10_000_000_000},
                min_ep_cpb=8,
                out_dir=out_dir,
                table_out=table,
            )

            self.assertEqual(len(merged), 1)
            self.assertEqual(set(plots), {"FLV_vs_DMSO", "TRP_vs_DMSO", "TRP_vs_FLV"})
            self.assertTrue(table.exists())
            for path in plots.values():
                self.assertTrue(path.exists())

    def test_plot_background_scatter_returns_figure_without_writing(self) -> None:
        data = pl.DataFrame(
            {
                "ratio_DMSO": [1.0, 2.0],
                "ratio_FLV": [1.5, 2.5],
            }
        )

        fig = plot_background_scatter(data, x_sample="DMSO", y_sample="FLV")

        self.assertTrue(hasattr(fig, "savefig"))
        import matplotlib.pyplot as plt

        plt.close(fig)


class SafeKdeTests(unittest.TestCase):
    def test_safe_kde_degenerate_input_returns_ones(self) -> None:
        np.testing.assert_array_equal(_safe_kde(np.zeros((2, 0))), np.ones(0))
        np.testing.assert_array_equal(_safe_kde(np.zeros((2, 1))), np.ones(1))

    def test_safe_kde_below_threshold_matches_unsubsampled_gaussian_kde(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=(2, 300))

        result = _safe_kde(values, max_fit_points=5000)
        expected = gaussian_kde(values)(values)

        np.testing.assert_allclose(result, expected)

    def test_safe_kde_above_threshold_is_deterministic_and_close_to_full_fit(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(size=(2, 2000))

        subsampled = _safe_kde(values, max_fit_points=200)
        repeat = _safe_kde(values, max_fit_points=200)
        full = gaussian_kde(values)(values)

        self.assertEqual(subsampled.shape, (2000,))
        self.assertTrue(np.all(np.isfinite(subsampled)))
        np.testing.assert_array_equal(subsampled, repeat)
        correlation = np.corrcoef(subsampled, full)[0, 1]
        self.assertGreater(correlation, 0.9)

    def test_safe_kde_different_seeds_still_correlate_closely(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(size=(2, 2000))

        seed_a = _safe_kde(values, max_fit_points=200, seed=0)
        seed_b = _safe_kde(values, max_fit_points=200, seed=1)

        correlation = np.corrcoef(seed_a, seed_b)[0, 1]
        self.assertGreater(correlation, 0.9)


if __name__ == "__main__":
    unittest.main()
