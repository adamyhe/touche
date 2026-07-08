from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from touche.background import count_ep_and_background


class BackgroundCountTests(unittest.TestCase):
    def test_count_ep_and_background_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = tmp_path / "pairs.tsv"
            baits = tmp_path / "baits.bed"
            preys = tmp_path / "preys.bed"
            out = tmp_path / "EP_and_BG_contacts.tsv"

            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t100\tchr1\t300\t+\t-\tUU\t30\t30",  # EP
                        "chr1\t100\tchr1\t250\t+\t-\tUU\t30\t30",  # bait to prey background
                        "chr1\t150\tchr1\t300\t+\t-\tUU\t30\t30",  # prey to bait background
                        "chr1\t100\tchr1\t101\t+\t-\tUU\t30\t30",  # intra bait, ignored
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            baits.write_text("chr1\t95\t105\t+\n", encoding="utf-8")
            preys.write_text("chr1\t295\t305\t+\n", encoding="utf-8")

            result = count_ep_and_background(
                pairs,
                baits,
                preys,
                out,
                min_distance=150,
                max_distance=250,
                window=10,
                min_bg_distance=40,
                max_bg_distance=60,
                source="touche",
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["EP_contacts"], 1)
            self.assertEqual(result.iloc[0]["BG_contacts"], 2)
            written = pd.read_csv(out, sep="\t", names=["chr", "bait", "prey", "ep", "bg"])
            self.assertEqual(written.iloc[0]["ep"], 1)
            self.assertEqual(written.iloc[0]["bg"], 2)


if __name__ == "__main__":
    unittest.main()
