from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
