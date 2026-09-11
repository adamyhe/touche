from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from touche.compare import balance_table, compare_groups, compare_paired, correlate, match_pairs


def _pair_table(seed: int = 0, n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    promoter_index = rng.integers(0, 30, n)
    group = np.where(rng.random(n) < 0.4, "positive", "negative")
    value = rng.normal(0.0, 1.0, n) + (group == "positive") * 0.8
    return pl.DataFrame(
        {
            "promoter": [f"p{i}" for i in promoter_index],
            "PosNeg": group,
            "log2_oe": value,
            "log_distance": rng.uniform(4.0, 6.0, n),
            "coverage": rng.uniform(0.0, 10.0, n),
        }
    )


class CompareGroupsTests(unittest.TestCase):
    def test_recovers_a_known_shift_with_an_interval_that_excludes_zero(self) -> None:
        result = compare_groups(_pair_table(), value_col="log2_oe", group_col="PosNeg", bootstrap=300)

        row = result.table.row(0, named=True)
        self.assertEqual((row["group_x"], row["group_y"]), ("negative", "positive"))
        self.assertLess(row["difference"], 0.0)
        self.assertLess(row["ci_high"], 0.0)
        self.assertLess(row["p_value"], 0.01)
        self.assertLess(row["cliffs_delta"], 0.0)

    def test_non_finite_values_are_excluded_and_counted_not_dropped_silently(self) -> None:
        table = _pair_table().with_columns(
            pl.when(pl.int_range(pl.len()) < 3).then(float("inf")).otherwise(pl.col("log2_oe")).alias("log2_oe")
        )

        result = compare_groups(table, value_col="log2_oe", group_col="PosNeg", bootstrap=100)

        excluded = result.info.filtered["non_finite_x"] + result.info.filtered["non_finite_y"]
        self.assertEqual(excluded, 3)
        self.assertEqual(result.info.n_tested, table.height - 3)

    def test_zero_counts_are_reported_per_group(self) -> None:
        table = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [0.0, 1.0, 0.0, 0.0]})

        row = compare_groups(table, value_col="v", group_col="g", bootstrap=50).table.row(0, named=True)

        self.assertEqual((row["n_zero_x"], row["n_zero_y"]), (1, 2))

    def test_unclustered_result_warns_that_pairs_are_not_independent(self) -> None:
        result = compare_groups(_pair_table(), value_col="log2_oe", group_col="PosNeg", bootstrap=50)

        self.assertEqual(result.info.inference_class, "descriptive")
        self.assertTrue(any("cluster_by" in warning for warning in result.info.warnings))

    def test_cluster_bootstrap_uses_clusters_as_the_resampling_unit(self) -> None:
        result = compare_groups(
            _pair_table(), value_col="log2_oe", group_col="PosNeg", cluster_by="promoter", bootstrap=200
        )

        row = result.table.row(0, named=True)
        self.assertLessEqual(row["n_clusters_x"], 30)
        self.assertEqual(result.info.cluster_unit, "promoter")
        self.assertFalse(any("cluster_by" in warning for warning in result.info.warnings))

    def test_too_few_clusters_is_warned_about(self) -> None:
        table = pl.DataFrame(
            {"g": ["a"] * 20 + ["b"] * 20, "v": list(range(40)), "c": [f"c{i % 3}" for i in range(40)]}
        )

        result = compare_groups(table, value_col="v", group_col="g", cluster_by="c", bootstrap=50)

        self.assertTrue(any("clusters" in warning for warning in result.info.warnings))

    def test_explicit_groups_and_other_levels(self) -> None:
        table = pl.DataFrame({"g": ["a", "a", "b", "b", "other"], "v": [1.0, 2.0, 5.0, 6.0, 99.0]})

        result = compare_groups(table, value_col="v", group_col="g", groups=("a", "b"), bootstrap=50)

        self.assertEqual(result.info.filtered["other_groups"], 1)

    def test_rejects_missing_columns_unknown_statistic_and_absent_group(self) -> None:
        table = _pair_table()
        with self.assertRaises(ValueError):
            compare_groups(table, value_col="nope", group_col="PosNeg")
        with self.assertRaises(ValueError):
            compare_groups(table, value_col="log2_oe", group_col="PosNeg", statistic="mode")
        with self.assertRaises(ValueError):
            compare_groups(table, value_col="log2_oe", group_col="PosNeg", groups=("positive", "missing"))

    def test_single_level_group_column_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compare_groups(pl.DataFrame({"g": ["a", "a"], "v": [1.0, 2.0]}), value_col="v", group_col="g")


class ComparePairedTests(unittest.TestCase):
    def test_recovers_a_paired_shift(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, 120)
        table = pl.DataFrame({"a": base + 0.5, "b": base, "c": [f"c{i % 12}" for i in range(120)]})

        row = compare_paired(table, x_col="a", y_col="b", cluster_by="c", bootstrap=200).table.row(0, named=True)

        self.assertAlmostEqual(row["median_difference"], 0.5, places=6)
        self.assertLess(row["p_value"], 1e-6)
        self.assertAlmostEqual(row["rank_biserial"], 1.0)
        self.assertEqual(row["n_clusters"], 12)

    def test_pairwise_drops_rows_where_either_value_is_missing(self) -> None:
        table = pl.DataFrame({"a": [1.0, 2.0, float("nan")], "b": [0.0, float("inf"), 1.0]})

        result = compare_paired(table, x_col="a", y_col="b", bootstrap=50)

        self.assertEqual(result.table["n_pairs"][0], 1)
        self.assertEqual(result.info.filtered["non_finite"], 2)

    def test_all_ties_returns_nan_instead_of_raising(self) -> None:
        table = pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})

        row = compare_paired(table, x_col="a", y_col="b", bootstrap=50).table.row(0, named=True)

        self.assertTrue(np.isnan(row["p_value"]))
        self.assertEqual(row["n_ties"], 2)


class CorrelateTests(unittest.TestCase):
    def test_recovers_a_strong_correlation(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 200)
        table = pl.DataFrame({"x": x, "y": x * 2 + rng.normal(0.0, 0.1, 200)})

        row = correlate(table, x_col="x", y_col="y", bootstrap=200).table.row(0, named=True)

        self.assertGreater(row["correlation"], 0.95)
        self.assertGreater(row["ci_low"], 0.9)
        self.assertLess(row["p_value"], 1e-12)

    def test_pearson_and_spearman_differ_on_a_monotone_nonlinear_relation(self) -> None:
        x = np.linspace(0.1, 5.0, 200)
        table = pl.DataFrame({"x": x, "y": np.exp(x)})

        pearson = correlate(table, x_col="x", y_col="y", method="pearson", bootstrap=50).table["correlation"][0]
        spearman = correlate(table, x_col="x", y_col="y", method="spearman", bootstrap=50).table["correlation"][0]

        self.assertAlmostEqual(spearman, 1.0)
        self.assertLess(pearson, spearman)

    def test_too_few_usable_rows_returns_nan(self) -> None:
        table = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})

        row = correlate(table, x_col="x", y_col="y", bootstrap=50).table.row(0, named=True)

        self.assertTrue(np.isnan(row["correlation"]))

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            correlate(pl.DataFrame({"x": [1.0], "y": [1.0]}), x_col="x", y_col="y", method="kendall")


class MatchingTests(unittest.TestCase):
    def test_matching_improves_covariate_balance(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        # Positives are systematically closer and better covered -- exactly the
        # confounding an unmatched functional comparison would attribute to
        # regulatory function.
        group = np.where(rng.random(n) < 0.3, "positive", "negative")
        table = pl.DataFrame(
            {
                "PosNeg": group,
                "log_distance": rng.normal(0.0, 1.0, n) - (group == "positive") * 0.8,
                "coverage": rng.normal(0.0, 1.0, n) + (group == "positive") * 0.5,
            }
        )
        covariates = ["log_distance", "coverage"]

        before = balance_table(table, group_col="PosNeg", covariates=covariates)
        matched = match_pairs(
            table, group_col="PosNeg", covariates={"log_distance": 0.1, "coverage": 0.25},
            groups=("positive", "negative"),
        )
        after = balance_table(matched, group_col="PosNeg", covariates=covariates)

        self.assertGreater(before["standardized_difference"].abs().max(), 0.4)
        self.assertLess(after["standardized_difference"].abs().max(), 0.1)

    def test_matched_output_pairs_each_case_with_a_control_sharing_a_match_id(self) -> None:
        table = pl.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "x": list(range(10)) * 2})

        matched = match_pairs(table, group_col="g", covariates={"x": 0.5}, groups=("a", "b"))

        self.assertEqual(matched["match_role"].value_counts().sort("match_role")["count"].to_list(), [10, 10])
        self.assertEqual(matched.group_by("match_id").len()["len"].unique().to_list(), [2])

    def test_controls_are_used_at_most_once(self) -> None:
        table = pl.DataFrame({"g": ["a"] * 5 + ["b"] * 2, "x": [0.0] * 7})

        matched = match_pairs(table, group_col="g", covariates={"x": 1.0}, groups=("a", "b"))

        self.assertEqual(matched.filter(pl.col("match_role") == "control").height, 2)

    def test_no_eligible_control_yields_an_empty_but_typed_frame(self) -> None:
        table = pl.DataFrame({"g": ["a", "b"], "x": [0.0, 100.0]})

        matched = match_pairs(table, group_col="g", covariates={"x": 0.1}, groups=("a", "b"))

        self.assertEqual(matched.height, 0)
        self.assertIn("match_id", matched.columns)

    def test_matching_is_reproducible_for_a_fixed_seed(self) -> None:
        table = _pair_table()
        kwargs = dict(group_col="PosNeg", covariates={"log_distance": 0.2, "coverage": 1.0})

        first = match_pairs(table, seed=3, **kwargs)
        second = match_pairs(table, seed=3, **kwargs)

        self.assertEqual(first.to_dicts(), second.to_dicts())

    def test_rejects_bad_matching_arguments(self) -> None:
        table = _pair_table()
        with self.assertRaises(ValueError):
            match_pairs(table, group_col="PosNeg", covariates={})
        with self.assertRaises(ValueError):
            match_pairs(table, group_col="PosNeg", covariates={"coverage": 0.0})
        with self.assertRaises(ValueError):
            match_pairs(table, group_col="PosNeg", covariates={"coverage": 1.0}, ratio=0)


if __name__ == "__main__":
    unittest.main()
