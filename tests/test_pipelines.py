from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from touche.background import parse_named_depth, parse_named_path
from touche.pipelines import run_background_pipeline, run_local_decay_pipeline


class PipelineTests(unittest.TestCase):
    def test_run_local_decay_pipeline_writes_outputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            functional = tmp_path / "functional.csv"
            nonfunctional = tmp_path / "nonfunctional.csv"
            out_dir = tmp_path / "local-decay"
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
            functional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,6000,10000\n",
                encoding="utf-8",
            )
            nonfunctional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,14000,10000\n",
                encoding="utf-8",
            )

            manifest = run_local_decay_pipeline(
                baits,
                preys,
                pairs,
                functional,
                nonfunctional,
                out_dir,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
                plot_min_contacts=0,
                plot_min_distance=0,
            )

            self.assertEqual(manifest["command"], "local-decay run")
            self.assertEqual(manifest["metrics"]["called_rows"], 2)
            self.assertTrue(Path(manifest["outputs"]["contacts"]).exists())
            self.assertTrue(Path(manifest["outputs"]["assignments"]).exists())
            self.assertTrue(Path(manifest["outputs"]["figure"]).exists())
            written_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(written_manifest["outputs"]["contacts"], manifest["outputs"]["contacts"])

    def test_run_background_pipeline_writes_counts_compare_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            out_dir = tmp_path / "background"
            baits.write_text("chr1\t90\t110\t+\n", encoding="utf-8")
            preys.write_text("chr1\t240\t260\t+\n", encoding="utf-8")
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t100\tchr1\t250\t+\t-\tUU\t30\t30",
                        "chr1\t100\tchr1\t320\t+\t-\tUU\t30\t30",
                        "chr1\t180\tchr1\t250\t+\t-\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            control = parse_named_path(f"ctrl={pairs}")
            treatments = [parse_named_path(f"trt={pairs}")]
            depths = {named.name: named.depth for named in [parse_named_depth("ctrl=100"), parse_named_depth("trt=100")]}
            manifest = run_background_pipeline(
                control,
                treatments,
                depths,
                baits,
                preys,
                out_dir,
                min_distance=50,
                max_distance=300,
                window=20,
                min_bg_distance=50,
                max_bg_distance=100,
                min_ep_cpb=0,
            )

            self.assertEqual(manifest["command"], "background run")
            self.assertEqual(manifest["metrics"]["comparison_rows"], 1)
            self.assertTrue(Path(manifest["outputs"]["counts"]["ctrl"]).exists())
            self.assertTrue(Path(manifest["outputs"]["counts"]["trt"]).exists())
            self.assertTrue(Path(manifest["outputs"]["comparison_table"]).exists())


if __name__ == "__main__":
    unittest.main()
