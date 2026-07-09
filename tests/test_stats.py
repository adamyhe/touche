from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import fisher_exact

from touche.backends import has_numba
from touche.stats import fisher_greater, fisher_greater_batch


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

    def test_fisher_greater_batch_scipy_backend_matches_scalar(self) -> None:
        a1 = np.array([5.0, 0.0, 10.0, 25.0])
        a2 = np.array([1.0, 2.0, 10.0, 3.0])
        b1 = np.array([10.0, 8.0, 10.0, 100.0])
        b2 = np.array([20.0, 9.0, 10.0, 200.0])
        batch = fisher_greater_batch(a1, a2, b1, b2, backend="scipy")
        scalar = [
            fisher_greater(a1[i], a2[i], b1[i], b2[i]) for i in range(len(a1))
        ]
        np.testing.assert_allclose(batch, scalar)

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


if __name__ == "__main__":
    unittest.main()
