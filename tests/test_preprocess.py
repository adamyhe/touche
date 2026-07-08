from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from touche.preprocess import convert_pairs, filter_pairs, write_qc


class PreprocessTests(unittest.TestCase):
    def test_filter_pairs_matches_reference_filter_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs"
            out = tmp_path / "filtered.pairs"
            pairs.write_text(
                "\n".join(
                    [
                        ".\tchr1\t10\tchr1\t30\t+\t-\tUU\t30\t31",
                        ".\tchr1\t10\tchr2\t30\t+\t-\tUU\t30\t31",
                        ".\tchr1\t10\tchr1\t40\t+\t-\tUU\t29\t31",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stats = filter_pairs(pairs, out)

            self.assertEqual(stats.written_rows, 1)
            self.assertEqual(
                out.read_text(encoding="utf-8"),
                "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n",
            )

    def test_convert_pairs_supports_gzip_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.pairs.gz"
            out = tmp_path / "converted.pairs.gz"
            with gzip.open(pairs, "wt", encoding="utf-8") as handle:
                handle.write(".\tchr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n")

            stats = convert_pairs(pairs, out, source="distiller")

            self.assertEqual(stats.written_rows, 1)
            with gzip.open(out, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n")

    def test_write_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "sample.touche.pairs"
            out = tmp_path / "qc.json"
            pairs.write_text("chr1\t10\tchr1\t30\t+\t-\tUU\t30\t31\n", encoding="utf-8")

            stats = write_qc(pairs, out, source="touche")
            payload = json.loads(out.read_text(encoding="utf-8"))

            self.assertEqual(stats.cis_rows, 1)
            self.assertEqual(payload["stats"]["per_chromosome"], {"chr1": 1})


if __name__ == "__main__":
    unittest.main()
