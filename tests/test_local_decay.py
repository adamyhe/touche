from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from touche.backends import has_numba
from touche.cli.local_decay import _assign_pair_types
from touche.contacts import build_contact_indexes, build_npz_cache
from touche.local_decay import (
    assign_pair_types,
    call_local_decay,
    compute_local_decay,
    fit_distance_decay_model,
    fit_zero_inflation_model,
    plot_pair_type_distribution,
    read_center_anchors,
)


class LocalDecayTests(unittest.TestCase):
    def test_call_local_decay_writes_reference_columns_and_counts_observed_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            out = tmp_path / "calls.tsv"
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

            calls = call_local_decay(
                baits,
                preys,
                pairs,
                out,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
            )

            self.assertEqual(calls.shape, (2, 9))
            self.assertEqual(calls["chr"].to_list(), ["chr1", "chr1"])
            self.assertEqual(calls["observed"].to_list(), [1, 2])
            self.assertTrue((calls["expected"] >= 0).all())
            self.assertTrue(out.exists())
            self.assertTrue((tmp_path / "contact_index_cache" / "contacts.manifest.json").exists())
            written = pl.read_csv(out, separator="\t", has_header=False)
            self.assertEqual(written.shape, (2, 9))

    def test_call_local_decay_chromosome_index_strategy_matches_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            all_out = tmp_path / "all.tsv"
            chrom_out = tmp_path / "chromosome.tsv"
            baits.write_text("chr1\t10000\nchr2\t20000\n", encoding="utf-8")
            preys.write_text("chr1\t6000\nchr1\t14000\nchr2\t24000\n", encoding="utf-8")
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t9950\tchr1\t14020\t+\t-\tUU\t30\t30",
                        "chr1\t9990\tchr1\t14080\t+\t-\tUU\t30\t30",
                        "chr1\t6050\tchr1\t10020\t+\t-\tUU\t30\t30",
                        "chr2\t19950\tchr2\t24020\t+\t-\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            all_calls = call_local_decay(
                baits,
                preys,
                pairs,
                all_out,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
                index_strategy="all",
            )
            chrom_calls = call_local_decay(
                baits,
                preys,
                pairs,
                chrom_out,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
                index_strategy="chromosome",
            )

            assert_frame_equal(chrom_calls, all_calls)

    def test_call_local_decay_cache_index_strategy_matches_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            cache_dir = tmp_path / "cache"
            all_out = tmp_path / "all.tsv"
            cache_out = tmp_path / "cache.tsv"
            baits.write_text("chr1\t10000\nchr2\t20000\n", encoding="utf-8")
            preys.write_text("chr1\t6000\nchr1\t14000\nchr2\t24000\n", encoding="utf-8")
            pairs.write_text(
                "\n".join(
                    [
                        "chr1\t9950\tchr1\t14020\t+\t-\tUU\t30\t30",
                        "chr1\t9990\tchr1\t14080\t+\t-\tUU\t30\t30",
                        "chr1\t6050\tchr1\t10020\t+\t-\tUU\t30\t30",
                        "chr2\t19950\tchr2\t24020\t+\t-\tUU\t30\t30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            build_npz_cache(pairs, cache_dir, source="touche", prefix="sample")

            all_calls = call_local_decay(
                baits,
                preys,
                pairs,
                all_out,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
                index_strategy="all",
            )
            cache_calls = call_local_decay(
                baits,
                preys,
                pairs,
                cache_out,
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
                index_strategy="cache",
                cache_dir=cache_dir,
                cache_prefix="sample",
            )

            assert_frame_equal(cache_calls, all_calls)

    def test_call_local_decay_require_cache_rejects_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            out = tmp_path / "calls.tsv"
            baits.write_text("chr1\t10000\n", encoding="utf-8")
            preys.write_text("chr1\t14000\n", encoding="utf-8")
            pairs.write_text("chr1\t9950\tchr1\t14020\t+\t-\tUU\t30\t30\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                call_local_decay(
                    baits,
                    preys,
                    pairs,
                    out,
                    dist=10_000,
                    cap=100,
                    min_distance=1_000,
                    lowess_window=500,
                    index_strategy="cache",
                    cache_dir=tmp_path / "missing-cache",
                    require_cache=True,
                )

    def test_compute_local_decay_accepts_in_memory_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
            baits.write_text("chr1\t10000\n", encoding="utf-8")
            preys.write_text("chr1\t14000\n", encoding="utf-8")
            pairs.write_text(
                "chr1\t9950\tchr1\t14020\t+\t-\tUU\t30\t30\n",
                encoding="utf-8",
            )

            calls = compute_local_decay(
                build_contact_indexes(pairs, source="touche"),
                read_center_anchors(baits),
                read_center_anchors(preys),
                dist=10_000,
                cap=100,
                min_distance=1_000,
                lowess_window=500,
            )

            self.assertEqual(calls.shape, (1, 9))
            self.assertEqual(calls[0, "observed"], 1)

    @unittest.skipUnless(has_numba(), "numba extra is not installed")
    def test_numba_compute_local_decay_matches_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
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
            indexes = build_contact_indexes(pairs, source="touche")
            bait_anchors = read_center_anchors(baits)
            prey_anchors = read_center_anchors(preys)
            kwargs = {
                "dist": 10_000,
                "cap": 100,
                "min_distance": 1_000,
                "lowess_window": 500,
            }

            numpy_calls = compute_local_decay(
                indexes, bait_anchors, prey_anchors, backend="numpy", **kwargs
            )
            numba_calls = compute_local_decay(
                indexes, bait_anchors, prey_anchors, backend="numba", **kwargs
            )

            assert_frame_equal(numba_calls, numpy_calls)

    @unittest.skipUnless(has_numba(), "numba extra is not installed")
    def test_fisher_backend_numba_matches_scipy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits = tmp_path / "baits.tsv"
            preys = tmp_path / "preys.tsv"
            pairs = tmp_path / "pairs.tsv"
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
            indexes = build_contact_indexes(pairs, source="touche")
            bait_anchors = read_center_anchors(baits)
            prey_anchors = read_center_anchors(preys)
            kwargs = {
                "dist": 10_000,
                "cap": 100,
                "min_distance": 1_000,
                "lowess_window": 500,
            }

            scipy_calls = compute_local_decay(
                indexes, bait_anchors, prey_anchors, fisher_backend="scipy", **kwargs
            )
            numba_calls = compute_local_decay(
                indexes, bait_anchors, prey_anchors, fisher_backend="numba", **kwargs
            )

            assert_frame_equal(
                numba_calls.drop("p_value"),
                scipy_calls.drop("p_value"),
            )
            assert_series_equal(
                numba_calls["p_value"],
                scipy_calls["p_value"],
                abs_tol=1e-6,
                check_exact=False,
            )

    @unittest.skipUnless(has_numba(), "numba extra is not installed")
    def test_numba_lowess_backend_returns_finite_values(self) -> None:
        counts = pl.Series([0, 1, 0, 0, 2, 0, 1, 0, 0, 1] * 20).cast(pl.Float64).to_numpy()

        smoothed = fit_zero_inflation_model(
            counts,
            dist=100,
            winsize=50,
            backend="numba",
        )

        self.assertEqual(smoothed.shape, (100,))
        self.assertTrue(pl.Series(smoothed).is_not_nan().all())

    @unittest.skipUnless(has_numba(), "numba extra is not installed")
    def test_numba_lowess_backend_matches_statsmodels_wrappers(self) -> None:
        counts = pl.Series([0, 1, 0, 0, 2, 0, 1, 0, 0, 1] * 200).cast(pl.Float64).to_numpy()
        statsmodels_zero = fit_zero_inflation_model(
            counts,
            dist=1_000,
            winsize=500,
            delta=16,
            backend="statsmodels",
        )
        numba_zero = fit_zero_inflation_model(
            counts,
            dist=1_000,
            winsize=500,
            delta=16,
            backend="numba",
        )
        assert_series_equal(
            pl.Series(numba_zero),
            pl.Series(statsmodels_zero),
            check_names=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

        distances = pl.Series(range(len(counts))).cast(pl.Float64).to_numpy()
        statsmodels_decay = fit_distance_decay_model(
            counts,
            statsmodels_zero,
            distances,
            dist=1_000,
            winsize=500,
            delta=16,
            backend="statsmodels",
        )
        numba_decay = fit_distance_decay_model(
            counts,
            numba_zero,
            distances,
            dist=1_000,
            winsize=500,
            delta=16,
            backend="numba",
        )
        assert_series_equal(
            pl.Series(numba_decay),
            pl.Series(statsmodels_decay),
            check_names=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    @unittest.skipUnless(has_numba(), "numba extra is not installed")
    def test_batched_lowess_numba_matches_per_chunk_calls(self) -> None:
        """Batching chunks into one prange launch must not change any fitted value.

        fit_zero_inflation_model/fit_distance_decay_model call a batched numba
        kernel (one parallel launch over all of a bait's LOWESS chunks)
        instead of one small parallel launch per chunk purely for speed --
        this pins down that the two are numerically identical, not just close,
        per the project's no-numeric-deviation policy.
        """
        from touche.numba_kernels import lowess_evenly_spaced_batched_numba, lowess_evenly_spaced_numba

        rng = np.random.default_rng(0)
        chunk_lengths = [5000, 5000, 3000, 1, 2, 5300, 0]
        chunks = [
            rng.poisson(0.4, size=n).astype(np.float64) if n > 0 else np.zeros(0, dtype=np.float64)
            for n in chunk_lengths
        ]
        offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
        np.cumsum([len(chunk) for chunk in chunks], out=offsets[1:])
        flat = np.concatenate(chunks)

        batched = lowess_evenly_spaced_batched_numba(flat, offsets, 0.01, 3, 16.0)
        per_chunk = np.concatenate(
            [lowess_evenly_spaced_numba(chunk, 0.01, 3, 16.0) for chunk in chunks]
        )
        np.testing.assert_array_equal(batched, per_chunk)

    def test_compute_local_decay_rejects_negative_lowess_iterations(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowess_iterations"):
            compute_local_decay(
                {},
                pl.DataFrame(schema=["chr", "center"]),
                pl.DataFrame(schema=["chr", "center"]),
                lowess_iterations=-1,
            )

    def test_assign_pair_types_uses_functional_and_nonfunctional_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contacts = tmp_path / "contacts.tsv"
            functional = tmp_path / "functional.csv"
            nonfunctional = tmp_path / "nonfunctional.csv"
            out = tmp_path / "assigned.tsv"
            contacts.write_text(
                "\n".join(
                    [
                        "chr1\t100\t200\t100\t0.01\t5\t2\t0\t0",
                        "chr1\t110\t210\t100\t0.02\t4\t2\t0\t0",
                        "chr1\t120\t220\t100\t0.03\t3\t2\t0\t0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            functional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,200,100\n",
                encoding="utf-8",
            )
            nonfunctional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,210,110\n",
                encoding="utf-8",
            )

            assigned = assign_pair_types(contacts, functional, nonfunctional, out)

            self.assertEqual(assigned["PosNeg"].to_list(), ["positive", "negative", "other"])
            written = pl.read_csv(out, separator="\t")
            self.assertEqual(written["distance"].to_list(), [100, 100, 100])

    def test_cli_assign_pair_types_prints_sorted_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contacts = tmp_path / "contacts.tsv"
            functional = tmp_path / "functional.csv"
            nonfunctional = tmp_path / "nonfunctional.csv"
            out = tmp_path / "assigned.tsv"
            contacts.write_text(
                "\n".join(
                    [
                        "chr1\t100\t200\t100\t0.01\t5\t2\t0\t0",
                        "chr1\t110\t210\t100\t0.02\t4\t2\t0\t0",
                        "chr1\t120\t220\t100\t0.03\t3\t2\t0\t0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            functional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,200,100\n",
                encoding="utf-8",
            )
            nonfunctional.write_text(
                "target_site.chr,target_promoter.center,target_site.center\n"
                "chr1,210,110\n",
                encoding="utf-8",
            )

            args = argparse.Namespace(
                contacts=contacts, functional=functional, nonfunctional=nonfunctional, out=out
            )
            captured = io.StringIO()
            with redirect_stdout(captured):
                _assign_pair_types(args)

            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["rows"], 3)
            self.assertEqual(payload["counts"], {"negative": 1, "other": 1, "positive": 1})

    def test_plot_pair_type_distribution_writes_plot_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            assignments = tmp_path / "assigned.tsv"
            figure = tmp_path / "plot.svg"
            table = tmp_path / "plot-data.tsv"
            pl.DataFrame(
                {
                    "target_site.chr": ["chr1", "chr1", "chr1"],
                    "target_site.center": [100, 110, 120],
                    "target_promoter.center": [200, 210, 220],
                    "directional_distance": [100, 100, 100],
                    "observed": [8, 4, 6],
                    "expected": [2, 2, 3],
                    "distance": [20_000, 20_000, 10_000],
                    "PosNeg": ["positive", "negative", "other"],
                }
            ).write_csv(assignments, separator="\t")

            plot_data, fig = plot_pair_type_distribution(
                assignments,
                figure,
                min_contacts=1,
                min_distance=15_000,
                plot_table_out=table,
            )

            self.assertEqual(len(plot_data), 2)
            self.assertTrue(hasattr(fig, "savefig"))
            self.assertTrue(figure.exists())
            self.assertTrue(table.exists())
            import matplotlib.pyplot as plt

            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
