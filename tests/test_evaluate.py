from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from touche.evaluate import average_precision, evaluate_scores, precision_recall_curve, roc_auc


class MetricTests(unittest.TestCase):
    def test_perfect_and_inverted_rankings(self) -> None:
        labels = np.array([True, True, False, False])

        self.assertAlmostEqual(roc_auc(labels, np.array([4.0, 3.0, 2.0, 1.0])), 1.0)
        self.assertAlmostEqual(average_precision(labels, np.array([4.0, 3.0, 2.0, 1.0])), 1.0)
        self.assertAlmostEqual(roc_auc(labels, np.array([1.0, 2.0, 3.0, 4.0])), 0.0)

    def test_all_tied_scores_give_chance_performance(self) -> None:
        labels = np.array([True, False, True, False])
        tied = np.zeros(4)

        self.assertAlmostEqual(roc_auc(labels, tied), 0.5)
        # With one tie group, precision is the prevalence at full recall.
        self.assertAlmostEqual(average_precision(labels, tied), 0.5)

    def test_average_precision_matches_a_hand_computed_case(self) -> None:
        # Ranked: +, -, +, -  -> precision at the two hits is 1/1 and 2/3,
        # each contributing a recall step of 1/2.
        labels = np.array([True, False, True, False])
        scores = np.array([4.0, 3.0, 2.0, 1.0])

        self.assertAlmostEqual(average_precision(labels, scores), 0.5 * 1.0 + 0.5 * (2 / 3))

    def test_average_precision_baseline_is_the_prevalence(self) -> None:
        rng = np.random.default_rng(0)
        labels = rng.random(20_000) < 0.05

        self.assertAlmostEqual(average_precision(labels, rng.random(20_000)), 0.05, places=2)

    def test_metrics_are_invariant_to_monotone_rescaling(self) -> None:
        rng = np.random.default_rng(1)
        labels = rng.random(500) < 0.2
        scores = rng.random(500)

        for transform in (lambda x: 3 * x + 1, np.exp, lambda x: -(-x) ** 1):
            np.testing.assert_allclose(roc_auc(labels, transform(scores)), roc_auc(labels, scores))
            np.testing.assert_allclose(
                average_precision(labels, transform(scores)), average_precision(labels, scores)
            )

    def test_degenerate_inputs_return_nan_rather_than_raising(self) -> None:
        self.assertTrue(np.isnan(roc_auc(np.array([], dtype=bool), np.array([]))))
        self.assertTrue(np.isnan(roc_auc(np.ones(5, dtype=bool), np.arange(5.0))))
        self.assertTrue(np.isnan(average_precision(np.zeros(5, dtype=bool), np.arange(5.0))))

    def test_mismatched_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            roc_auc(np.array([True, False]), np.array([1.0]))

    def test_precision_recall_curve_collapses_ties_to_one_point(self) -> None:
        labels = np.array([True, False, True, False])
        precision, recall, thresholds = precision_recall_curve(labels, np.array([1.0, 1.0, 0.0, 0.0]))

        self.assertEqual(thresholds.tolist(), [1.0, 0.0])
        self.assertEqual(recall.tolist(), [0.5, 1.0])


class EvaluateScoresTests(unittest.TestCase):
    def _table(self, n: int = 400, seed: int = 0) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        positive = rng.random(n) < 0.25
        return pl.DataFrame(
            {
                "label": np.where(positive, "positive", "negative"),
                "chrom": [f"chr{i % 8 + 1}" for i in range(n)],
                "good": rng.normal(0, 1, n) + positive * 2.0,
                "useless": rng.normal(0, 1, n),
            }
        )

    def test_ranks_an_informative_score_above_an_uninformative_one(self) -> None:
        result = evaluate_scores(
            self._table(), label_col="label", score_cols=["good", "useless"],
            positive_label="positive", bootstrap=100,
        )

        rows = {row["score"]: row for row in result.table.iter_rows(named=True)}
        self.assertGreater(rows["good"]["auprc"], 0.7)
        self.assertLess(rows["useless"]["auprc"], 0.45)
        self.assertAlmostEqual(rows["good"]["baseline_auprc"], rows["useless"]["baseline_auprc"])

    def test_unlabelled_rows_are_excluded_not_treated_as_negatives(self) -> None:
        table = self._table().with_columns(
            pl.when(pl.int_range(pl.len()) < 100).then(None).otherwise(pl.col("label")).alias("label")
        )

        result = evaluate_scores(
            table, label_col="label", score_cols=["good"], positive_label="positive", bootstrap=50
        )

        self.assertEqual(result.table["n"][0], 300)
        self.assertEqual(result.info.n_tested, 300)

    def test_missing_scores_rank_last_and_are_counted(self) -> None:
        table = self._table().with_columns(
            pl.when(pl.int_range(pl.len()) < 10).then(None).otherwise(pl.col("good")).alias("good")
        )

        worst = evaluate_scores(
            table, label_col="label", score_cols=["good"], positive_label="positive",
            nan_policy="worst", bootstrap=50,
        )

        self.assertEqual(worst.table["n_missing_score"][0], 10)
        # Every row is still evaluated, so methods stay comparable on one set.
        self.assertEqual(worst.table["n"][0], table.height)

    def test_held_out_summary_reports_groups_and_an_interval(self) -> None:
        result = evaluate_scores(
            self._table(n=800), label_col="label", score_cols=["good"],
            positive_label="positive", held_out_col="chrom", bootstrap=200,
        )

        row = result.table.row(0, named=True)
        self.assertEqual(row["n_groups"], 8)
        self.assertLessEqual(row["held_out_ci_low"], row["held_out_auprc_mean"])
        self.assertGreaterEqual(row["held_out_ci_high"], row["held_out_auprc_mean"])
        self.assertEqual(result.info.cluster_unit, "chrom")

    def test_small_or_single_class_groups_are_skipped_and_counted(self) -> None:
        table = pl.DataFrame(
            {
                "label": ["positive"] * 30 + ["negative"] * 30 + ["positive"] * 5,
                "chrom": ["chr1"] * 60 + ["chr2"] * 5,
                "score": list(np.arange(65.0)),
            }
        )

        result = evaluate_scores(
            table, label_col="label", score_cols=["score"], positive_label="positive",
            held_out_col="chrom", bootstrap=50,
        )

        self.assertEqual(result.table["n_groups"][0], 1)
        self.assertEqual(result.table["n_groups_skipped"][0], 1)

    def test_pooled_evaluation_warns_that_it_has_no_uncertainty(self) -> None:
        result = evaluate_scores(
            self._table(), label_col="label", score_cols=["good"], positive_label="positive", bootstrap=50
        )

        self.assertTrue(any("No held_out_col" in warning for warning in result.info.warnings))
        self.assertEqual(result.info.inference_class, "descriptive")

    def test_too_few_positives_is_warned_about(self) -> None:
        table = pl.DataFrame(
            {"label": ["positive"] * 5 + ["negative"] * 95, "score": list(np.arange(100.0))}
        )

        result = evaluate_scores(
            table, label_col="label", score_cols=["score"], positive_label="positive", bootstrap=50
        )

        self.assertTrue(any("positive pairs" in warning for warning in result.info.warnings))

    def test_rejects_missing_columns_and_unknown_nan_policy(self) -> None:
        table = self._table()
        with self.assertRaises(ValueError):
            evaluate_scores(table, label_col="label", score_cols=["absent"])
        with self.assertRaises(ValueError):
            evaluate_scores(table, label_col="label", score_cols=["good"], nan_policy="impute")
        with self.assertRaises(ValueError):
            evaluate_scores(table, label_col="label", score_cols=["good"], held_out_col="absent")


if __name__ == "__main__":
    unittest.main()
