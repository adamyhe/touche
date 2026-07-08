from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from touche.contacts import build_npz_cache, load_npz_cache


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

            self.assertEqual(len(written), 3)
            manifest = json.loads((cache_dir / "sample.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["chromosomes"]["chr1"]["rows"], 2)
            chr1 = load_npz_cache(cache_dir / "sample.chr1.npz")
            self.assertEqual(chr1.chrom, "chr1")
            self.assertEqual(chr1.pos_a.tolist(), [10, 20])

    def test_build_cache_can_write_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            cache_dir = tmp_path / "cache"
            qc_out = tmp_path / "sample.qc.json"
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

            self.assertIn(qc_out, written)
            payload = json.loads(qc_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["stats"]["parsed_rows"], 2)
            self.assertEqual(payload["stats"]["cis_rows"], 1)
            self.assertEqual(payload["stats"]["trans_rows"], 1)


if __name__ == "__main__":
    unittest.main()
