"""The default path must stay numerically identical to the reference workflow.

Every statistical addition is opt-in. These tests pin the properties that
make that claim checkable: the legacy layout's exact shape and column order,
that opting in to tidy output does not perturb the shared values, and that
the legacy path never materializes the tidy-only columns it would discard.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from touche.apa import compute_apa
from touche.background import compute_ep_and_background
from touche.local_decay import LOCAL_DECAY_OUTPUT_COLUMNS, call_local_decay, compute_local_decay
from touche.models import ContactIndex

CALL_KWARGS = dict(dist=500_000, cap=2_000, min_distance=5_000)


def _write_fixture(tmp: Path) -> None:
    rng = np.random.default_rng(7)
    rows = []
    for chrom in ("chr1", "chr2"):
        positions = np.sort(rng.integers(0, 2_000_000, 20_000))
        rows += [
            f"{chrom}\t{a}\t{chrom}\t{a + d}\t+\t-\tUU\t30\t30"
            for a, d in zip(positions, rng.integers(5_000, 400_000, positions.size))
        ]
    (tmp / "contacts.pairs").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp / "baits.tsv").write_text(
        "".join(f"{c}\t{s + 500}\n" for c in ("chr1", "chr2") for s in range(0, 2_000_000, 40_000)),
        encoding="utf-8",
    )
    (tmp / "preys.tsv").write_text(
        "".join(
            f"{c}\t{s + 500}\n" for c in ("chr1", "chr2") for s in range(17_000, 2_017_000, 40_000)
        ),
        encoding="utf-8",
    )


def _indexes() -> dict[str, ContactIndex]:
    rng = np.random.default_rng(11)
    indexes = {}
    for chrom in ("chr1", "chr2"):
        n = 20_000
        pos_a = np.sort(rng.integers(0, 1_000_000, n))
        indexes[chrom] = ContactIndex(
            chrom=chrom,
            pos_a=pos_a,
            pos_b=pos_a + rng.integers(20_000, 300_000, n),
            strand_a=np.ones(n, dtype=np.int8),
            strand_b=-np.ones(n, dtype=np.int8),
            mapq_a=np.full(n, 30, dtype=np.int16),
            mapq_b=np.full(n, 30, dtype=np.int16),
        )
    return indexes


def _anchors() -> tuple[pl.DataFrame, pl.DataFrame]:
    starts = list(range(0, 1_000_000, 40_000))
    baits = pl.DataFrame(
        {"chr": [c for c in ("chr1", "chr2") for _ in starts], "start": starts * 2}
    ).with_columns(
        (pl.col("start") + 1_000).alias("end"),
        pl.lit("+").alias("strand"),
        (pl.col("start") + 500).alias("center"),
    )
    preys = baits.with_columns(
        pl.lit("-").alias("strand"), (pl.col("center") + 17_000).alias("center")
    )
    return baits, preys


class LegacyLocalDecayTests(unittest.TestCase):
    def test_legacy_output_keeps_the_reference_layout_and_no_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fixture(tmp_path)

            calls = call_local_decay(
                tmp_path / "baits.tsv", tmp_path / "preys.tsv", tmp_path / "contacts.pairs",
                tmp_path / "calls.tsv", cache_dir=tmp_path / "cache", **CALL_KWARGS,
            )
            written = pl.read_csv(tmp_path / "calls.tsv", separator="\t", has_header=False)
            has_sidecar = (tmp_path / "calls.tsv.meta.json").exists()

        self.assertEqual(calls.columns, LOCAL_DECAY_OUTPUT_COLUMNS)
        self.assertEqual(written.width, 9)
        self.assertEqual(written.height, calls.height)
        self.assertGreater(calls.height, 0)
        self.assertFalse(has_sidecar, "the legacy layout must not gain a metadata sidecar")

    def test_legacy_path_does_not_materialize_tidy_only_columns(self) -> None:
        # Not cosmetic: pair_id/bait_id/prey_id are string columns costing
        # roughly a second and half a gigabyte per five million rows, which the
        # legacy layout would then throw away.
        baits, preys = _anchors()
        legacy = compute_local_decay(_indexes(), baits, preys, **CALL_KWARGS)

        self.assertEqual(legacy.columns, LOCAL_DECAY_OUTPUT_COLUMNS)
        for column in ("pair_id", "bait_id", "prey_id", "log2_oe", "method"):
            self.assertNotIn(column, legacy.columns)

    def test_tidy_and_legacy_agree_on_every_shared_column(self) -> None:
        baits, preys = _anchors()
        legacy = compute_local_decay(_indexes(), baits, preys, **CALL_KWARGS)
        tidy = compute_local_decay(_indexes(), baits, preys, schema="tidy", **CALL_KWARGS)

        self.assertEqual(legacy.height, tidy.height)
        self.assertEqual(legacy["chr"].to_list(), tidy["chrom"].to_list())
        for column in LOCAL_DECAY_OUTPUT_COLUMNS[1:]:
            np.testing.assert_allclose(
                legacy[column].cast(pl.Float64).to_numpy(),
                tidy[column].cast(pl.Float64).to_numpy(),
                err_msg=column,
            )

    def test_expected_is_the_product_of_the_reported_trials_and_null_probability(self) -> None:
        # The calibrated test's inputs must reconstruct the legacy expectation
        # exactly; if they do not, one of the two is not what it claims to be.
        baits, preys = _anchors()
        tidy = compute_local_decay(_indexes(), baits, preys, schema="tidy", **CALL_KWARGS)

        np.testing.assert_allclose(
            tidy["expected"].to_numpy(),
            tidy["n_trials"].to_numpy() * tidy["p_null"].to_numpy(),
            rtol=1e-12,
        )

    def test_choosing_a_method_does_not_change_the_counts_it_tests(self) -> None:
        baits, preys = _anchors()
        fisher = compute_local_decay(_indexes(), baits, preys, schema="tidy", **CALL_KWARGS)
        binomial = compute_local_decay(
            _indexes(), baits, preys, schema="tidy", method="binomial", **CALL_KWARGS
        )

        for column in ("observed", "expected", "n_trials", "p_null"):
            np.testing.assert_allclose(
                fisher[column].cast(pl.Float64).to_numpy(),
                binomial[column].cast(pl.Float64).to_numpy(),
                err_msg=column,
            )
        self.assertFalse(np.allclose(fisher["p_value"].to_numpy(), binomial["p_value"].to_numpy()))


class LegacyApaAndBackgroundTests(unittest.TestCase):
    def test_keep_chromosomes_does_not_change_the_pooled_matrix(self) -> None:
        baits, preys = _anchors()
        kwargs = dict(min_distance=25_000, max_distance=300_000, window=10_000, pixels=50)

        pooled = compute_apa(_indexes(), baits, preys, **kwargs)
        split = compute_apa(_indexes(), baits, preys, keep_chromosomes=True, **kwargs)

        self.assertTrue(pooled.matrix.equals(split.matrix))
        self.assertTrue(pooled.bait_signal.equals(split.bait_signal))
        self.assertIsNone(pooled.chrom_matrices)

    def test_background_counts_are_unchanged_without_a_pair_list(self) -> None:
        baits, preys = _anchors()
        kwargs = dict(
            min_distance=25_000, max_distance=300_000, window=2_500,
            min_bg_distance=10_000, max_bg_distance=300_000,
        )

        counts = compute_ep_and_background(_indexes(), baits, preys, **kwargs)

        self.assertEqual(
            counts.columns, ["chr", "promoter", "enhancer", "EP_contacts", "BG_contacts"]
        )
        self.assertGreater(counts.height, 0)


if __name__ == "__main__":
    unittest.main()
