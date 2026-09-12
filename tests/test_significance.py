from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binom

from touche.local_decay import LOCAL_DECAY_OUTPUT_COLUMNS, call_local_decay
from touche.significance import assess_calibration, read_local_decay_calls, test_contacts
from touche.stats import adjust_pvalues


def _calls(observed: list[int], trials: list[int], p_null: list[float]) -> pl.DataFrame:
    expected = [n * p for n, p in zip(trials, p_null)]
    return pl.DataFrame(
        {
            "pair_id": [f"chr1:{i}|chr1:{i + 10000}" for i in range(len(observed))],
            "chrom": ["chr1"] * len(observed),
            "bait_center": list(range(len(observed))),
            "prey_center": [i + 10000 for i in range(len(observed))],
            "observed": observed,
            "expected": expected,
            "n_trials": trials,
            "p_null": p_null,
            "observed_background": [1000 - o for o in observed],
            "expected_background": [1000.0 - e for e in expected],
        }
    )


class TestContactsTests(unittest.TestCase):
    def test_binomial_p_values_match_the_declared_model(self) -> None:
        calls = _calls([5, 0, 20], [100, 50, 100], [0.02, 0.1, 0.05])

        result = test_contacts(calls, method="binomial")

        np.testing.assert_allclose(
            result.table["p_value"].to_numpy(),
            binom.sf(np.array([5, 0, 20]) - 1, [100, 50, 100], [0.02, 0.1, 0.05]),
        )
        self.assertEqual(result.info.inference_class, "technical")
        self.assertEqual(result.info.n_tested, 3)

    def test_q_values_are_bh_over_the_declared_family(self) -> None:
        calls = _calls([5, 6, 7, 8], [100] * 4, [0.02] * 4)

        result = test_contacts(calls, method="binomial", fdr="bh")

        np.testing.assert_allclose(
            result.table["q_value"].to_numpy(), adjust_pvalues(result.table["p_value"].to_numpy())
        )
        self.assertEqual(result.info.fdr_family, "all called pairs")
        self.assertEqual(result.table.columns.index("q_value"), result.table.columns.index("p_value") + 1)

    def test_fdr_scope_records_a_stratified_family(self) -> None:
        calls = _calls([5, 6, 7, 8], [100] * 4, [0.02] * 4).with_columns(
            pl.Series("chrom", ["chr1", "chr1", "chr2", "chr2"])
        )

        result = test_contacts(calls, method="binomial", fdr_scope="chrom")

        self.assertIn("chrom", result.info.fdr_family)

    def test_zero_trial_pairs_are_untestable_and_excluded_from_the_family(self) -> None:
        calls = _calls([0, 5], [0, 100], [0.1, 0.02])

        result = test_contacts(calls, method="binomial")

        self.assertTrue(np.isnan(result.table["p_value"][0]))
        self.assertTrue(np.isnan(result.table["q_value"][0]))
        self.assertEqual(result.info.n_tested, 1)
        self.assertEqual(result.info.filtered["untestable"], 1)
        self.assertTrue(any("untestable" in warning for warning in result.info.warnings))

    def test_legacy_fisher_carries_a_non_calibration_warning(self) -> None:
        result = test_contacts(_calls([5], [100], [0.02]), method="legacy_fisher")

        self.assertEqual(result.info.inference_class, "descriptive")
        self.assertTrue(any("not calibrated" in warning for warning in result.info.warnings))

    def test_recompute_false_keeps_existing_p_values(self) -> None:
        calls = _calls([5], [100], [0.02]).with_columns(pl.lit(0.125).alias("p_value"))

        result = test_contacts(calls, method="binomial", recompute=False)

        self.assertAlmostEqual(result.table["p_value"][0], 0.125)

    def test_rejects_unknown_method_and_fdr(self) -> None:
        calls = _calls([5], [100], [0.02])
        with self.assertRaises(ValueError):
            test_contacts(calls, method="beta_binomial")
        with self.assertRaises(ValueError):
            test_contacts(calls, fdr="storey")

    def test_binomial_on_legacy_input_raises_instead_of_guessing_a_trial_total(self) -> None:
        legacy = _calls([5], [100], [0.02]).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("n_trials"), pl.lit(None, dtype=pl.Float64).alias("p_null")
        )

        with self.assertRaises(ValueError) as raised:
            test_contacts(legacy, method="binomial")
        self.assertIn("--schema tidy", str(raised.exception))


class NegativeBinomialTests(unittest.TestCase):
    def _overdispersed_calls(self, n: int = 20_000, phi: float = 2.5, seed: int = 0) -> pl.DataFrame:
        """Null pairs whose counts are `phi` times more variable than Poisson allows."""
        rng = np.random.default_rng(seed)
        trials = rng.integers(200, 2_000, n)
        p_null = rng.uniform(0.005, 0.05, n)
        mean = trials * p_null
        observed = rng.negative_binomial(mean / (phi - 1.0), 1.0 / phi)
        return _calls(observed.tolist(), trials.tolist(), p_null.tolist())

    def test_dispersion_one_reduces_to_the_poisson_test(self) -> None:
        calls = _calls([5, 0, 20], [100, 50, 100], [0.02, 0.1, 0.05])

        nb = test_contacts(calls, method="negative_binomial", dispersion=1.0)
        poisson = test_contacts(calls, method="poisson")

        np.testing.assert_allclose(
            nb.table["p_value"].to_numpy(), poisson.table["p_value"].to_numpy(), equal_nan=True
        )

    def test_it_controls_type_one_error_where_the_binomial_does_not(self) -> None:
        calls = self._overdispersed_calls(phi=2.5)

        binomial = assess_calibration(test_contacts(calls, method="binomial").table)
        nb = assess_calibration(
            test_contacts(calls, method="negative_binomial", dispersion=2.5).table
        )

        # The binomial over-rejects by roughly the dispersion factor; the NB
        # with the right dispersion does not.
        self.assertGreater(binomial["reject_rate_at_0.05"][0], 0.10)
        self.assertLessEqual(nb["reject_rate_at_0.05"][0], 0.055)

    def test_estimated_dispersion_recovers_the_simulated_one(self) -> None:
        calls = self._overdispersed_calls(phi=2.5)

        result = test_contacts(calls, method="negative_binomial", dispersion="pearson")

        self.assertAlmostEqual(result.info.parameters["dispersion"], 2.5, delta=0.3)
        self.assertTrue(any("estimated from the pairs" in w for w in result.info.warnings))

    def test_dispersion_is_recorded_and_validated(self) -> None:
        calls = self._overdispersed_calls(n=500)

        self.assertEqual(
            test_contacts(calls, method="negative_binomial", dispersion=3.0).info.parameters["dispersion"],
            3.0,
        )
        with self.assertRaises(ValueError):
            test_contacts(calls, method="negative_binomial", dispersion=0.5)
        with self.assertRaises(ValueError):
            test_contacts(calls, method="negative_binomial", dispersion="mle")

    def test_the_per_bait_path_refuses_the_method_with_an_actionable_error(self) -> None:
        from touche.local_decay import _contact_p_values

        with self.assertRaises(ValueError) as raised:
            _contact_p_values(
                np.array([1.0]), np.array([1.0]), np.array([10.0]), np.array([0.1]), 1000,
                method="negative_binomial", fisher_backend="numba",
            )
        self.assertIn("local-decay test", str(raised.exception))


class NullCalibrationTests(unittest.TestCase):
    """The plan's acceptance criterion: under the declared null, p-values are uniform."""

    def _null_calls(self, n: int = 20_000, seed: int = 0) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        trials = rng.integers(200, 2000, n)
        p_null = rng.uniform(0.005, 0.05, n)
        observed = rng.binomial(trials, p_null)
        return _calls(observed.tolist(), trials.tolist(), p_null.tolist())

    def test_binomial_type_one_error_is_controlled_under_the_null(self) -> None:
        n = 20_000
        result = test_contacts(self._null_calls(n=n), method="binomial")

        calibration = assess_calibration(result.table)
        for alpha in (0.01, 0.05):
            rate = calibration[f"reject_rate_at_{alpha:g}"][0]
            # A discrete upper-tail test is conservative in expectation, so the
            # rejection rate should sit at or below alpha -- allowing three
            # Monte Carlo standard errors for the finite simulation.
            tolerance = 3 * np.sqrt(alpha * (1 - alpha) / n)
            self.assertLessEqual(rate, alpha + tolerance, f"anticonservative at alpha={alpha}")
            self.assertGreater(rate, alpha / 3, f"implausibly conservative at alpha={alpha}")

    def test_bh_controls_the_false_discovery_proportion_under_the_null(self) -> None:
        result = test_contacts(self._null_calls(seed=1), method="binomial")

        discoveries = result.table.filter(pl.col("q_value") <= 0.05)
        # Every pair here is null, so any discovery is false; BH at 0.05 over a
        # wholly null family should make discoveries rare.
        self.assertLess(discoveries.height / result.table.height, 0.01)

    def test_calibration_can_be_stratified(self) -> None:
        calls = self._null_calls(n=2000).with_columns(
            pl.when(pl.col("n_trials") < 1000).then(pl.lit("low")).otherwise(pl.lit("high")).alias("depth")
        )
        result = test_contacts(calls, method="binomial")

        calibration = assess_calibration(result.table, strata="depth")

        self.assertEqual(sorted(calibration["depth"].to_list()), ["high", "low"])
        self.assertTrue((calibration["n_tested"] > 0).all())

    def test_legacy_fisher_is_not_uniform_on_the_same_null_data(self) -> None:
        calls = self._null_calls(seed=2)
        legacy = assess_calibration(test_contacts(calls, method="legacy_fisher").table)
        binomial = assess_calibration(test_contacts(calls, method="binomial").table)

        self.assertLess(binomial["ks_statistic"][0], legacy["ks_statistic"][0])

    def test_assess_calibration_rejects_a_missing_column(self) -> None:
        with self.assertRaises(ValueError):
            assess_calibration(pl.DataFrame({"x": [1.0]}))
        with self.assertRaises(ValueError):
            assess_calibration(_calls([1], [10], [0.1]), trials_col="absent", probability_col="p_null")

    def test_attainable_rate_bounds_the_observed_rate_for_a_discrete_test(self) -> None:
        result = test_contacts(self._null_calls(n=20_000), method="binomial")

        calibration = assess_calibration(
            result.table, trials_col="n_trials", probability_col="p_null"
        ).row(0, named=True)

        # Discreteness makes the attainable size strictly below alpha, so the
        # observed rate must be read against it rather than against alpha.
        attainable = calibration["attainable_rate_at_0.05"]
        self.assertLess(attainable, 0.05)
        self.assertGreater(attainable, 0.0)
        self.assertAlmostEqual(calibration["reject_rate_at_0.05"], attainable, delta=0.01)

    def test_discreteness_is_reported_so_the_ks_statistic_is_not_misread(self) -> None:
        # Zero-observed pairs give p = 1 exactly; that atom drives KS, not
        # any calibration failure.
        calls = _calls([0, 0, 0, 5], [50, 50, 50, 100], [0.02, 0.02, 0.02, 0.02])

        calibration = assess_calibration(test_contacts(calls, method="binomial").table).row(0, named=True)

        self.assertAlmostEqual(calibration["fraction_at_one"], 0.0, delta=1.0)
        self.assertIn("fraction_at_one", calibration)


class CallOutputIntegrationTests(unittest.TestCase):
    def _fixture(self, tmp: Path) -> tuple[Path, Path, Path]:
        baits, preys, pairs = tmp / "baits.tsv", tmp / "preys.tsv", tmp / "pairs.tsv"
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
        return baits, preys, pairs

    def test_legacy_schema_output_is_unchanged_by_the_tidy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits, preys, pairs = self._fixture(tmp_path)

            calls = call_local_decay(
                baits, preys, pairs, tmp_path / "legacy.tsv", dist=10_000, cap=100,
                min_distance=1_000, lowess_window=500,
            )
            written = pl.read_csv(tmp_path / "legacy.tsv", separator="\t", has_header=False)

        self.assertEqual(calls.columns, LOCAL_DECAY_OUTPUT_COLUMNS)
        self.assertEqual(calls.shape, (2, 9))
        self.assertEqual(written.shape, (2, 9))

    def test_tidy_schema_writes_q_values_and_a_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits, preys, pairs = self._fixture(tmp_path)
            out = tmp_path / "tidy.tsv"

            calls = call_local_decay(
                baits, preys, pairs, out, dist=10_000, cap=100, min_distance=1_000,
                lowess_window=500, method="binomial", schema="tidy",
            )
            metadata = json.loads((tmp_path / "tidy.tsv.meta.json").read_text(encoding="utf-8"))

        self.assertIn("q_value", calls.columns)
        self.assertIn("pair_id", calls.columns)
        self.assertEqual(calls["method"].unique().to_list(), ["binomial"])
        self.assertEqual(metadata["method"], "binomial")
        self.assertEqual(metadata["fdr_method"], "bh")
        self.assertEqual(metadata["inference_class"], "technical")
        self.assertIn("touche_version", metadata)

    def test_legacy_and_tidy_agree_on_the_shared_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits, preys, pairs = self._fixture(tmp_path)
            kwargs = dict(dist=10_000, cap=100, min_distance=1_000, lowess_window=500)

            legacy = call_local_decay(baits, preys, pairs, tmp_path / "a.tsv", **kwargs)
            tidy = call_local_decay(
                baits, preys, pairs, tmp_path / "b.tsv", schema="tidy", cache_dir=tmp_path / "c2", **kwargs
            )

        for column in ("observed", "expected", "p_value", "directional_distance"):
            np.testing.assert_allclose(
                legacy[column].cast(pl.Float64).to_numpy(), tidy[column].cast(pl.Float64).to_numpy()
            )

    def test_read_local_decay_calls_accepts_both_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baits, preys, pairs = self._fixture(tmp_path)
            kwargs = dict(dist=10_000, cap=100, min_distance=1_000, lowess_window=500)
            call_local_decay(baits, preys, pairs, tmp_path / "a.tsv", **kwargs)
            call_local_decay(
                baits, preys, pairs, tmp_path / "b.tsv", schema="tidy", cache_dir=tmp_path / "c2", **kwargs
            )

            from_legacy = read_local_decay_calls(tmp_path / "a.tsv")
            from_tidy = read_local_decay_calls(tmp_path / "b.tsv")

        self.assertEqual(from_legacy["pair_id"].to_list(), from_tidy["pair_id"].to_list())
        self.assertTrue(from_legacy["n_trials"].is_null().all())
        self.assertFalse(from_tidy["n_trials"].is_null().any())


if __name__ == "__main__":
    unittest.main()
