from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from touche.background import (
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
        data = pd.DataFrame(
            {
                "ratio_DMSO": [1.0, 2.0],
                "ratio_FLV": [1.5, 2.5],
            }
        )

        fig = plot_background_scatter(data, x_sample="DMSO", y_sample="FLV")

        self.assertTrue(hasattr(fig, "savefig"))
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
