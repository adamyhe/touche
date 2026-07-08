from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from touche.contacts import build_npz_cache, load_npz_cache
from touche.preprocess import write_qc


class ContactCacheTests(unittest.TestCase):
    def test_builds_chromosome_sharded_npz_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            cache_dir = tmp_path / "cache"
            pairs.write_text(
                "\n".join(
                    [
                        "chr2\t50\tchr2\t70\t+\t-\tUU\t30\t31",
                        "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31",
                        "chr1\t20\tchr1\t40\t-\t+\tUU\t32\t33",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            written = build_npz_cache(pairs, cache_dir, source="touche", prefix="sample")

            self.assertEqual(len(written), 4)
            manifest = json.loads((cache_dir / "sample.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["chromosomes"]["chr1"]["rows"], 2)
            self.assertTrue((cache_dir / "sample.qc.json").exists())
            chr1 = load_npz_cache(cache_dir / "sample.chr1.npz")
            self.assertEqual(chr1.chrom, "chr1")
            self.assertEqual(chr1.pos_a.tolist(), [10, 20])

    def test_build_cache_can_write_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            cache_dir = tmp_path / "cache"
            qc_out = tmp_path / "nested" / "qc" / "sample.qc.json"
            standalone_qc = tmp_path / "standalone.qc.json"
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31",
                        "chr1\t10\tchr2\t30\t+\t-\tUU\t30\t31",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            written = build_npz_cache(
                pairs,
                cache_dir,
                source="touche",
                prefix="sample",
                qc_out=qc_out,
            )
            write_qc(pairs, standalone_qc, source="touche")

            self.assertIn(qc_out, written)
            self.assertTrue(qc_out.exists())
            payload = json.loads(qc_out.read_text(encoding="utf-8"))
            standalone_payload = json.loads(standalone_qc.read_text(encoding="utf-8"))
            self.assertEqual(payload["stats"]["parsed_rows"], 2)
            self.assertEqual(payload["stats"]["cis_rows"], 1)
            self.assertEqual(payload["stats"]["trans_rows"], 1)
            self.assertEqual(payload["stats"], standalone_payload["stats"])

    def test_build_cache_can_disable_default_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            cache_dir = tmp_path / "cache"
            pairs.write_text("chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n", encoding="utf-8")

            written = build_npz_cache(
                pairs,
                cache_dir,
                source="touche",
                prefix="sample",
                write_qc=False,
            )

            self.assertNotIn(cache_dir / "sample.qc.json", written)
            self.assertFalse((cache_dir / "sample.qc.json").exists())

    def test_default_and_all_cache_strategies_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            default_cache = tmp_path / "default-cache"
            all_cache = tmp_path / "all-cache"
            pairs.write_text(
                "\n".join(
                    [
                        "chr2\t50\tchr2\t70\t+\t-\tUU\t30\t31",
                        "chr1\t20\tchr1\t40\t-\t+\tUU\t32\t33",
                        "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            build_npz_cache(pairs, default_cache, source="touche", prefix="sample")
            build_npz_cache(
                pairs,
                all_cache,
                source="touche",
                prefix="sample",
                index_strategy="all",
            )

            default_chr1 = load_npz_cache(default_cache / "sample.chr1.npz")
            all_chr1 = load_npz_cache(all_cache / "sample.chr1.npz")
            self.assertEqual(default_chr1.pos_a.tolist(), all_chr1.pos_a.tolist())
            self.assertEqual(default_chr1.pos_b.tolist(), all_chr1.pos_b.tolist())
            self.assertEqual(default_chr1.strand_a.tolist(), all_chr1.strand_a.tolist())
            self.assertEqual(default_chr1.mapq_a.tolist(), all_chr1.mapq_a.tolist())

    def test_build_cache_without_metadata_is_position_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            cache_dir = tmp_path / "cache"
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31",
                        "chr1\t20\tchr1\t40\t-\t+\tUU\t32\t33",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            build_npz_cache(
                pairs,
                cache_dir,
                source="touche",
                prefix="sample",
                include_metadata=False,
            )

            chr1 = load_npz_cache(cache_dir / "sample.chr1.npz", include_metadata=False)
            self.assertEqual(chr1.pos_a.tolist(), [10, 20])
            self.assertEqual(chr1.strand_a.tolist(), [0, 0])
            self.assertEqual(chr1.mapq_a.tolist(), [0, 0])
            with np.load(cache_dir / "sample.chr1.npz", allow_pickle=False) as data:
                self.assertNotIn("strand_a", data.files)
                self.assertNotIn("mapq_a", data.files)

    def test_build_cache_sorts_unsorted_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            cache_dir = tmp_path / "cache"
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t20\tchr1\t40\t-\t+\tUU\t32\t33",
                        "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            build_npz_cache(pairs, cache_dir, source="touche", prefix="sample")

            chr1 = load_npz_cache(cache_dir / "sample.chr1.npz")
            self.assertEqual(chr1.pos_a.tolist(), [10, 20])
            self.assertEqual(chr1.mapq_a.tolist(), [30, 32])

    def test_build_cache_parses_distiller_and_touche_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            touche_pairs = tmp_path / "sample.touche.pairs"
            distiller_pairs = tmp_path / "sample.distiller.pairs"
            touche_cache = tmp_path / "touche-cache"
            distiller_cache = tmp_path / "distiller-cache"
            touche_pairs.write_text(
                "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n",
                encoding="utf-8",
            )
            distiller_pairs.write_text(
                ".\tchr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n",
                encoding="utf-8",
            )

            build_npz_cache(touche_pairs, touche_cache, source="touche", prefix="sample")
            build_npz_cache(
                distiller_pairs,
                distiller_cache,
                source="distiller",
                prefix="sample",
            )

            touche_index = load_npz_cache(touche_cache / "sample.chr1.npz")
            distiller_index = load_npz_cache(distiller_cache / "sample.chr1.npz")
            self.assertEqual(touche_index.pos_a.tolist(), distiller_index.pos_a.tolist())
            self.assertEqual(touche_index.pos_b.tolist(), distiller_index.pos_b.tolist())


if __name__ == "__main__":
    unittest.main()
