from __future__ import annotations

import unittest

from scipy.stats import fisher_exact

from touche.stats import fisher_greater


class StatsTests(unittest.TestCase):
    def test_fisher_greater_rounds_inputs(self) -> None:
        rounded = fisher_greater(5, 1, 10, 20)
        fractional = fisher_greater(5.2, 1.2, 10.1, 19.9)
        self.assertEqual(fractional, rounded)

    def test_fisher_greater_returns_probability(self) -> None:
        p_value = fisher_greater(5, 1, 10, 20)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_fisher_greater_matches_scipy_one_sided(self) -> None:
        for table in [
            [[5, 1], [10, 20]],
            [[0, 2], [8, 9]],
            [[10, 10], [10, 10]],
            [[25, 3], [100, 200]],
        ]:
            expected = fisher_exact(table, alternative="greater").pvalue
            observed = fisher_greater(table[0][0], table[0][1], table[1][0], table[1][1])
            self.assertAlmostEqual(observed, expected)

    def test_fisher_greater_rejects_negative_entries(self) -> None:
        with self.assertRaises(ValueError):
            fisher_greater(1, 2, 3, -1)


if __name__ == "__main__":
    unittest.main()
