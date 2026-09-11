from __future__ import annotations

import unittest

import numpy as np
import polars as pl
from scipy.stats import binom, false_discovery_control, fisher_exact, poisson

from touche.backends import has_numba
from touche.stats import (
    adjust_pvalue_column,
    adjust_pvalues,
    binom_sf_greater,
    bootstrap_ci,
    bootstrap_difference_ci,
    cliffs_delta,
    fisher_greater_batch,
    log2_fold_change,
    log_odds_ratio,
    median_difference,
    poisson_sf_greater,
    rank_biserial,
)


class StatsTests(unittest.TestCase):
    def test_fisher_greater_batch_rounds_inputs(self) -> None:
        rounded = fisher_greater_batch(
            np.array([5.0]), np.array([1.0]), np.array([10.0]), np.array([20.0])
        )
        fractional = fisher_greater_batch(
            np.array([5.2]), np.array([1.2]), np.array([10.1]), np.array([19.9])
        )
        np.testing.assert_allclose(fractional, rounded)

    def test_fisher_greater_batch_returns_probability(self) -> None:
        p_value = fisher_greater_batch(
            np.array([5.0]), np.array([1.0]), np.array([10.0]), np.array([20.0])
        )
        self.assertGreaterEqual(p_value[0], 0.0)
        self.assertLessEqual(p_value[0], 1.0)

    def test_fisher_greater_batch_matches_scipy_one_sided(self) -> None:
        for table in [
            [[5, 1], [10, 20]],
            [[0, 2], [8, 9]],
            [[10, 10], [10, 10]],
            [[25, 3], [100, 200]],
        ]:
            expected = fisher_exact(table, alternative="greater").pvalue
            observed = fisher_greater_batch(
                np.array([table[0][0]]),
                np.array([table[0][1]]),
                np.array([table[1][0]]),
                np.array([table[1][1]]),
                backend="scipy",
            )
            self.assertAlmostEqual(observed[0], expected)

    def test_fisher_greater_batch_rejects_negative_entries(self) -> None:
        with self.assertRaises(ValueError):
            fisher_greater_batch(
                np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([-1.0])
            )

    @unittest.skipUnless(has_numba(), "numba is not installed")
    def test_fisher_greater_batch_numba_matches_scipy(self) -> None:
        rng = np.random.default_rng(0)

        # Random small tables, plus tables shaped like local-decay's actual
        # usage: a huge population/draws (M, N in the millions from a
        # genome-scale background histogram) with a small successes count
        # (a + b, bounded by observed/expected contact counts).
        small_a1 = rng.integers(0, 100, 200).astype(float)
        small_a2 = rng.uniform(0, 100, 200)
        small_b1 = rng.integers(0, 1000, 200).astype(float)
        small_b2 = rng.uniform(0, 1000, 200)

        histogram_bins = rng.integers(1_000, 2_000_000, 200)
        observed = rng.integers(0, 50, 200).astype(float)
        expected = rng.uniform(0, 50, 200)
        large_a1 = observed
        large_a2 = expected
        large_b1 = histogram_bins - observed
        large_b2 = histogram_bins - expected

        a1 = np.concatenate([small_a1, large_a1])
        a2 = np.concatenate([small_a2, large_a2])
        b1 = np.concatenate([small_b1, large_b1])
        b2 = np.concatenate([small_b2, large_b2])

        scipy_result = fisher_greater_batch(a1, a2, b1, b2, backend="scipy")
        numba_result = fisher_greater_batch(a1, a2, b1, b2, backend="numba")
        np.testing.assert_allclose(numba_result, scipy_result, atol=1e-6)


class AdjustPvaluesTests(unittest.TestCase):
    def test_bh_and_by_match_scipy_including_ties(self) -> None:
        rng = np.random.default_rng(0)
        # Rounding to three decimals guarantees ties, the case a naive
        # step-up implementation gets wrong.
        p_values = np.round(rng.random(500), 3)

        for method in ("bh", "by"):
            np.testing.assert_allclose(
                adjust_pvalues(p_values, method=method),
                false_discovery_control(p_values, method=method),
                rtol=1e-12,
            )

    def test_tied_pvalues_get_identical_qvalues(self) -> None:
        adjusted = adjust_pvalues(np.array([0.01, 0.01, 0.02, 0.5, 0.5, 1.0]))
        self.assertEqual(adjusted[0], adjusted[1])
        self.assertEqual(adjusted[3], adjusted[4])

    def test_holm_and_bonferroni_match_r_p_adjust(self) -> None:
        p_values = np.array([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjust_pvalues(p_values, method="holm"), [0.03, 0.06, 0.06, 0.02])
        np.testing.assert_allclose(adjust_pvalues(p_values, method="bonferroni"), [0.04, 0.16, 0.12, 0.02])

    def test_nan_pvalues_pass_through_and_do_not_inflate_family_size(self) -> None:
        with_nan = adjust_pvalues(np.array([0.01, np.nan, 0.02, np.nan]))
        without = adjust_pvalues(np.array([0.01, 0.02]))

        self.assertTrue(np.isnan(with_nan[1]) and np.isnan(with_nan[3]))
        np.testing.assert_allclose(with_nan[[0, 2]], without)

    def test_empty_and_all_nan_inputs(self) -> None:
        self.assertEqual(adjust_pvalues(np.array([])).size, 0)
        self.assertTrue(np.isnan(adjust_pvalues(np.array([np.nan, np.nan]))).all())

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            adjust_pvalues(np.array([0.5]), method="storey")

    def test_adjust_pvalue_column_groupby_corrects_each_stratum_separately(self) -> None:
        table = pl.DataFrame({"sample": ["a", "a", "b", "b"], "p_value": [0.01, 0.02, 0.01, 0.02]})

        grouped = adjust_pvalue_column(table, groupby="sample")
        pooled = adjust_pvalue_column(table)

        np.testing.assert_allclose(grouped["q_value"].to_numpy(), [0.02, 0.02, 0.02, 0.02])
        np.testing.assert_allclose(pooled["q_value"].to_numpy(), [0.02, 0.02, 0.02, 0.02])
        self.assertGreater(
            adjust_pvalue_column(table.head(2))["q_value"][0], 0.0
        )

    def test_adjust_pvalue_column_on_empty_frame(self) -> None:
        empty = pl.DataFrame({"p_value": []}, schema={"p_value": pl.Float64})
        self.assertEqual(adjust_pvalue_column(empty)["q_value"].len(), 0)


class TailTestTests(unittest.TestCase):
    def test_binom_sf_greater_matches_scipy(self) -> None:
        observed = np.array([0, 3, 10, 25])
        trials = np.array([10, 10, 100, 50])
        probability = np.array([0.1, 0.1, 0.05, 0.5])

        np.testing.assert_allclose(
            binom_sf_greater(observed, trials, probability), binom.sf(observed - 1, trials, probability)
        )

    def test_binom_sf_greater_is_nan_without_trials(self) -> None:
        result = binom_sf_greater(np.array([0, 1]), np.array([0, 5]), np.array([0.1, 0.1]))
        self.assertTrue(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[1]))

    def test_poisson_sf_greater_matches_scipy_and_is_nan_at_zero_expectation(self) -> None:
        observed = np.array([2, 5])
        expected = np.array([1.5, 0.0])

        result = poisson_sf_greater(observed, expected)
        self.assertAlmostEqual(result[0], float(poisson.sf(1, 1.5)))
        self.assertTrue(np.isnan(result[1]))


class EffectSizeTests(unittest.TestCase):
    def test_log2_fold_change_preserves_infinities_without_pseudocount(self) -> None:
        result = log2_fold_change(np.array([4.0, 1.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0, 0.0]))
        self.assertEqual(result[0], 2.0)
        self.assertTrue(np.isposinf(result[1]))
        self.assertTrue(np.isneginf(result[2]))
        self.assertTrue(np.isnan(result[3]))

    def test_log2_fold_change_pseudocount_makes_zero_pairs_finite(self) -> None:
        self.assertTrue(np.isfinite(log2_fold_change(np.array([1.0]), np.array([0.0]), pseudocount=1.0)).all())

    def test_log2_fold_change_rejects_negative_pseudocount(self) -> None:
        with self.assertRaises(ValueError):
            log2_fold_change(np.array([1.0]), np.array([1.0]), pseudocount=-1.0)

    def test_log_odds_ratio_matches_hand_computation(self) -> None:
        estimate, standard_error = log_odds_ratio(
            np.array([10.0]), np.array([20.0]), np.array([30.0]), np.array([40.0])
        )
        self.assertAlmostEqual(estimate[0], np.log((10 * 40) / (20 * 30)))
        self.assertAlmostEqual(standard_error[0], np.sqrt(1 / 10 + 1 / 20 + 1 / 30 + 1 / 40))

    def test_log_odds_ratio_correction_makes_zero_cells_finite(self) -> None:
        raw, raw_se = log_odds_ratio(np.array([0.0]), np.array([5.0]), np.array([5.0]), np.array([5.0]))
        corrected, corrected_se = log_odds_ratio(
            np.array([0.0]), np.array([5.0]), np.array([5.0]), np.array([5.0]), correction=0.5
        )

        self.assertTrue(np.isneginf(raw[0]))
        self.assertTrue(np.isinf(raw_se[0]))
        self.assertTrue(np.isfinite(corrected[0]) and np.isfinite(corrected_se[0]))

    def test_cliffs_delta_spans_full_range(self) -> None:
        self.assertAlmostEqual(cliffs_delta(np.array([4.0, 5.0]), np.array([1.0, 2.0])), 1.0)
        self.assertAlmostEqual(cliffs_delta(np.array([1.0, 2.0]), np.array([4.0, 5.0])), -1.0)
        self.assertAlmostEqual(cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 0.0)

    def test_rank_biserial_ignores_exact_ties(self) -> None:
        self.assertAlmostEqual(rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)
        self.assertAlmostEqual(rank_biserial(np.array([1.0, 2.0, 3.0, 0.0, 0.0])), 1.0)
        self.assertTrue(np.isnan(rank_biserial(np.array([0.0, 0.0]))))

    def test_median_difference_ignores_non_finite(self) -> None:
        self.assertAlmostEqual(median_difference(np.array([1.0, 3.0, np.inf]), np.array([1.0, 1.0])), 1.0)


class BootstrapTests(unittest.TestCase):
    def test_percentile_interval_brackets_the_estimate_and_is_reproducible(self) -> None:
        values = np.random.default_rng(0).normal(5.0, 1.0, 200)

        first = bootstrap_ci(values, statistic=np.mean, n_resamples=500, seed=7)
        second = bootstrap_ci(values, statistic=np.mean, n_resamples=500, seed=7)

        self.assertEqual(first, second)
        self.assertLess(first["ci_low"], first["estimate"])
        self.assertGreater(first["ci_high"], first["estimate"])
        self.assertEqual(first["n_clusters"], 200)

    def test_cluster_bootstrap_widens_interval_for_correlated_rows(self) -> None:
        rng = np.random.default_rng(1)
        labels = np.repeat(np.arange(20), 25)
        values = rng.normal(0.0, 1.0, 20)[labels] + rng.normal(0.0, 0.01, 500)

        by_row = bootstrap_ci(values, statistic=np.mean, n_resamples=500, seed=0)
        by_cluster = bootstrap_ci(values, statistic=np.mean, clusters=labels, n_resamples=500, seed=0)

        self.assertEqual(by_cluster["n_clusters"], 20)
        self.assertEqual(by_cluster["cluster_unit"], "cluster")
        self.assertGreater(
            by_cluster["ci_high"] - by_cluster["ci_low"], 3 * (by_row["ci_high"] - by_row["ci_low"])
        )

    def test_bca_interval_is_finite_and_brackets_the_estimate(self) -> None:
        values = np.random.default_rng(2).gamma(2.0, 2.0, 150)
        result = bootstrap_ci(values, statistic=np.mean, n_resamples=500, method="bca", seed=0)

        self.assertTrue(np.isfinite(result["ci_low"]) and np.isfinite(result["ci_high"]))
        self.assertLess(result["ci_low"], result["estimate"])
        self.assertGreater(result["ci_high"], result["estimate"])

    def test_degenerate_inputs_return_nan_bounds_not_errors(self) -> None:
        self.assertTrue(np.isnan(bootstrap_ci(np.array([1.0]))["ci_low"]))
        self.assertTrue(np.isnan(bootstrap_ci(np.array([]))["estimate"]))

    def test_rejects_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([1.0, 2.0]), method="jackknife")
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([1.0, 2.0]), n_resamples=0)
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([1.0, 2.0]), confidence=1.5)
        with self.assertRaises(ValueError):
            bootstrap_ci(np.array([1.0, 2.0]), clusters=np.array([1]))

    def test_difference_interval_recovers_a_known_shift(self) -> None:
        rng = np.random.default_rng(3)
        result = bootstrap_difference_ci(
            rng.normal(2.0, 1.0, 300), rng.normal(0.0, 1.0, 300), statistic=np.mean, n_resamples=500
        )

        self.assertLess(result["ci_low"], 2.0)
        self.assertGreater(result["ci_high"], 2.0)
        self.assertGreater(result["ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
