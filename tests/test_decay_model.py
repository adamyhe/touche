"""The legacy background model must stay bit-identical; the corrected one must integrate to 1.

`fit_distance_decay_model`'s default reproduces the reference
implementation's scale error, in which the fitted background integrates to
roughly half of one rather than one. That is a defect in the reference
method, not in the port -- `ReferenceParityTests` proves the port is exact
whenever the reference sources are checked out. The corrected model is
opt-in, and these tests pin both sides of that contract.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from touche.backends import has_statsmodels
from touche.local_decay import (
    DECAY_MODELS,
    compute_local_decay,
    fit_distance_decay_model,
    fit_zero_inflation_model,
)
from touche.models import ContactIndex

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "_reference/E-P_contacts/Contact_normalization_by_local_decay/ContactCaller_microC.py"
)
DIST = 100_000
WINSIZE = 5_000


def _histogram(seed: int = 0, n: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """A sparse, right-skewed 1 bp distance histogram.

    The default `n` gives ~90% empty 1 bp bins, which is the regime a real
    Micro-C bait sits in over a megabase window and where the reference
    model's scale error is severe. Raising `n` densifies it.
    """
    rng = np.random.default_rng(seed)
    distances = np.clip(rng.exponential(30_000, n).astype(int), 1, 2 * DIST - 1)
    counts = np.bincount(distances, minlength=2 * DIST)[: 2 * DIST].astype(float)
    return counts, distances


def _models(counts: np.ndarray, distances: np.ndarray, **kwargs: object) -> np.ndarray:
    zero = fit_zero_inflation_model(np.where(counts != 0, 0.0, 1.0), dist=DIST, winsize=WINSIZE)
    return fit_distance_decay_model(counts, zero, distances, dist=DIST, winsize=WINSIZE, **kwargs)


class LegacyScaleErrorTests(unittest.TestCase):
    def test_legacy_model_loses_most_of_its_mass(self) -> None:
        counts, distances = _histogram()

        legacy = _models(counts, distances)

        # Pinned deliberately: this is the reference's documented defect, and
        # a change here means the legacy path stopped reproducing it.
        self.assertLess(legacy.sum(), 0.8)
        self.assertGreater(legacy.sum(), 0.2)

    def test_robust_reweighting_is_what_drives_the_mass_down(self) -> None:
        counts, distances = _histogram()

        masses = [_models(counts, distances, iterations=it).sum() for it in (0, 1, 3)]

        self.assertGreater(masses[0], masses[1])
        self.assertGreater(masses[1], masses[2])

    def test_the_error_depends_on_coverage_so_it_is_not_a_fixed_factor(self) -> None:
        # This is why the defect matters beyond a constant rescale: a
        # low-coverage bait is biased far more than a high-coverage one, so
        # the distortion tracks exactly the covariate being controlled for.
        sparse = _models(*_histogram(n=10_000)).sum()
        dense = _models(*_histogram(n=1_000_000)).sum()

        self.assertLess(sparse, 0.8)
        self.assertAlmostEqual(dense, 1.0, delta=0.15)
        self.assertGreater(dense - sparse, 0.3)

    def test_normalized_model_integrates_to_one(self) -> None:
        counts, distances = _histogram()

        normalized = _models(counts, distances, iterations=0, normalize=True)

        self.assertAlmostEqual(float(normalized.sum()), 1.0, places=10)
        self.assertTrue((normalized >= 0).all())

    def test_normalization_is_monotone_so_it_cannot_reorder_pairs(self) -> None:
        counts, distances = _histogram()

        legacy = _models(counts, distances, iterations=0)
        normalized = _models(counts, distances, iterations=0, normalize=True)

        usable = legacy > 0
        ratios = normalized[usable] / legacy[usable]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)

    def test_an_empty_window_yields_no_density_in_either_mode(self) -> None:
        counts = np.zeros(2 * DIST)
        empty = np.array([], dtype=int)

        # The zero-inflation pedestal would otherwise normalize to a flat
        # distribution, which looks like information and is not.
        self.assertTrue(np.all(_models(counts, empty, iterations=0, normalize=True) == 0.0))
        self.assertTrue(np.all(_models(counts, empty) == 0.0))


@unittest.skipUnless(REFERENCE.exists(), "reference sources are not checked out")
@unittest.skipUnless(has_statsmodels(), "reference implementation requires statsmodels")
class ReferenceParityTests(unittest.TestCase):
    """The legacy path must equal the reference implementation exactly.

    The reference functions are extracted from its source with `ast` rather
    than transcribed, so this cannot drift from what the reference actually
    does. `_reference/` is gitignored, so this skips in CI and runs for
    anyone who has cloned the upstream workflows.
    """

    @staticmethod
    def _reference_functions() -> dict[str, object]:
        import statsmodels.api as sm

        tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
        wanted = {"optimize_lowess", "optimize_lowess2"}
        module = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
            type_ignores=[],
        )
        namespace: dict[str, object] = {"np": np, "sm": sm}
        exec(compile(module, str(REFERENCE), "exec"), namespace)
        return namespace

    def test_legacy_background_model_matches_the_reference_exactly(self) -> None:
        counts, distances = _histogram()
        reference = self._reference_functions()

        reference_zero = reference["optimize_lowess"](
            np.where(counts != 0, 0, 1), WINSIZE, 16, DIST
        )
        reference_bg = np.asarray(
            reference["optimize_lowess2"](
                counts, reference_zero, np.arange(1, int(distances.max()) + 1, 1),
                WINSIZE, 16, distances, DIST,
            )
        )

        touche_zero = fit_zero_inflation_model(
            np.where(counts != 0, 0.0, 1.0), dist=DIST, winsize=WINSIZE, backend="statsmodels"
        )
        touche_bg = fit_distance_decay_model(
            counts, touche_zero, distances, dist=DIST, winsize=WINSIZE, backend="statsmodels"
        )

        np.testing.assert_array_equal(touche_zero, reference_zero)
        self.assertEqual(len(touche_bg), len(reference_bg))
        np.testing.assert_array_equal(touche_bg, reference_bg)

    def test_the_reference_itself_loses_the_mass(self) -> None:
        counts, distances = _histogram()
        reference = self._reference_functions()

        reference_zero = reference["optimize_lowess"](np.where(counts != 0, 0, 1), WINSIZE, 16, DIST)
        reference_bg = np.asarray(
            reference["optimize_lowess2"](
                counts, reference_zero, np.arange(1, int(distances.max()) + 1, 1),
                WINSIZE, 16, distances, DIST,
            )
        )

        # The defect is upstream, not in the port.
        self.assertLess(reference_bg.sum(), 0.8)


def _simulated_chromosome(
    *, length: int = 6_000_000, n: int = 500_000, exponent: float = 1.0, seed: int = 0
) -> ContactIndex:
    """Contacts drawn from a known `P(s) ~ s**-exponent` with no focal structure.

    Every bait-prey pair over this index is null by construction, so a
    correctly specified background model must predict the observed counts:
    `mean(observed) / mean(expected)` has to come out at 1.
    """
    rng = np.random.default_rng(seed)
    start = rng.integers(0, length, n)
    u = rng.random(n)
    low, high = 1_000.0, 400_000.0
    if exponent == 1.0:
        distance = low * (high / low) ** u
    else:
        power = 1.0 - exponent
        distance = (low**power + u * (high**power - low**power)) ** (1.0 / power)
    left = np.minimum(start, length - 1)
    right = np.minimum(start + distance.astype(np.int64), length - 1)
    keep = right > left
    left, right = left[keep], right[keep]
    order = np.argsort(left, kind="mergesort")
    left, right = left[order], right[order]
    return ContactIndex(
        chrom="chr1", pos_a=left, pos_b=right,
        strand_a=np.ones(left.size, dtype=np.int8), strand_b=-np.ones(left.size, dtype=np.int8),
        mapq_a=np.full(left.size, 30, dtype=np.int16), mapq_b=np.full(left.size, 30, dtype=np.int16),
    )


def _null_anchors(dist: int, length: int = 6_000_000) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Baits and preys spread over the simulated chromosome, away from its edges."""
    def frame(values: list[int]) -> pl.DataFrame:
        return pl.DataFrame({"chr": ["chr1"] * len(values), "center": values})

    baits = list(range(2 * dist, length - 2 * dist, max(dist // 4, 1)))
    preys = sorted({b + off for b in baits for off in (dist // 10, dist // 3, dist // 2, int(dist * 0.85))})
    return frame(baits), frame(preys)


def _bias(decay_model: str, *, dist: int = 200_000, **kwargs: object) -> float:
    """`mean(observed) / mean(expected)` over null pairs; 1.0 means unbiased."""
    index = _simulated_chromosome(**kwargs)
    baits, preys = _null_anchors(dist)
    calls = compute_local_decay(
        {"chr1": index}, baits, preys, dist=dist, cap=2_000, min_distance=5_000,
        schema="tidy", decay_model=decay_model,
    )
    return float(calls["observed"].mean() / calls["expected"].mean())


class UnbiasedExpectationTests(unittest.TestCase):
    """`anchored` must predict observed counts on pairs that are null by construction."""

    def test_anchored_is_unbiased_where_the_others_are_not(self) -> None:
        anchored = _bias("anchored")
        normalized = _bias("normalized")

        self.assertAlmostEqual(anchored, 1.0, delta=0.10)
        # `normalized` fixes the total mass but not the window's shape bias,
        # so it over-predicts expected counts by roughly a quarter.
        self.assertLess(normalized, 0.9)

    def test_anchored_stays_unbiased_across_decay_shapes_and_window_sizes(self) -> None:
        for label, kwargs in (
            ("steeper P(s)", {"exponent": 1.5}),
            ("shallower P(s)", {"exponent": 0.5}),
            ("sparser", {"n": 150_000}),
        ):
            self.assertAlmostEqual(_bias("anchored", **kwargs), 1.0, delta=0.12, msg=label)
        for dist in (100_000, 400_000):
            self.assertAlmostEqual(_bias("anchored", dist=dist), 1.0, delta=0.12, msg=f"dist={dist}")

    def test_geometry_weight_divides_out_the_window_inclusion_measure(self) -> None:
        # A contact spanning d has 2*dist + d qualifying positions, so a flat
        # histogram must come back tilted by exactly that factor.
        dist, span = 50_000, 100_000
        counts = np.ones(span)
        zero = np.zeros(span)
        distances = np.arange(1, span)

        plain = fit_distance_decay_model(counts, zero, distances, dist=dist, winsize=5_000, iterations=0, normalize=True)
        weighted = fit_distance_decay_model(
            counts, zero, distances, dist=dist, winsize=5_000, iterations=0, normalize=True, geometry_weight=True
        )

        offsets = np.arange(plain.size, dtype=float)
        expected_ratio = 1.0 / (2.0 * dist + offsets)
        usable = (offsets > 1_000) & (offsets < dist - 1_000) & (plain > 0)
        observed_ratio = weighted[usable] / plain[usable]
        observed_ratio /= observed_ratio[0]
        reference = expected_ratio[usable] / expected_ratio[usable][0]
        np.testing.assert_allclose(observed_ratio, reference, rtol=0.02)

    def test_anchored_restricts_trials_to_the_modelled_range(self) -> None:
        index = _simulated_chromosome()
        baits, preys = _null_anchors(200_000)
        kwargs = dict(dist=200_000, cap=2_000, min_distance=5_000, schema="tidy")

        anchored = compute_local_decay({"chr1": index}, baits, preys, decay_model="anchored", **kwargs)
        normalized = compute_local_decay({"chr1": index}, baits, preys, decay_model="normalized", **kwargs)

        # p_null carries no mass beyond `dist`, so trials there cannot belong
        # in its denominator.
        self.assertLess(anchored["n_trials"].sum(), normalized["n_trials"].sum())
        self.assertEqual(anchored["observed"].to_list(), normalized["observed"].to_list())


class DecayModelWiringTests(unittest.TestCase):
    def _fixture(self) -> tuple[dict[str, ContactIndex], pl.DataFrame, pl.DataFrame]:
        rng = np.random.default_rng(3)
        # Sparse enough that each bait's 1 bp histogram is mostly empty, the
        # regime where the reference model's scale error actually bites.
        n = 12_000
        pos_a = np.sort(rng.integers(0, 400_000, n))
        index = ContactIndex(
            chrom="chr1",
            pos_a=pos_a,
            pos_b=pos_a + np.clip(rng.exponential(20_000, n).astype(int), 1_000, 150_000),
            strand_a=np.ones(n, dtype=np.int8),
            strand_b=-np.ones(n, dtype=np.int8),
            mapq_a=np.full(n, 30, dtype=np.int16),
            mapq_b=np.full(n, 30, dtype=np.int16),
        )
        baits = pl.DataFrame({"chr": ["chr1"] * 8, "center": list(range(50_000, 370_000, 40_000))})
        preys = baits.with_columns((pl.col("center") + 22_000).alias("center"))
        return {"chr1": index}, baits, preys

    def test_normalized_raises_expected_counts_without_touching_observed(self) -> None:
        indexes, baits, preys = self._fixture()
        kwargs = dict(dist=100_000, cap=2_000, min_distance=5_000, schema="tidy")

        legacy = compute_local_decay(indexes, baits, preys, decay_model="legacy", **kwargs)
        normalized = compute_local_decay(indexes, baits, preys, decay_model="normalized", **kwargs)

        self.assertEqual(legacy["observed"].to_list(), normalized["observed"].to_list())
        self.assertEqual(legacy["n_trials"].to_list(), normalized["n_trials"].to_list())
        self.assertGreater(
            normalized["expected"].sum(), 1.5 * legacy["expected"].sum(),
            "the corrected model should roughly double the expected counts",
        )

    def test_correcting_the_model_makes_the_binomial_test_less_extreme(self) -> None:
        indexes, baits, preys = self._fixture()
        kwargs = dict(dist=100_000, cap=2_000, min_distance=5_000, schema="tidy", method="binomial")

        legacy = compute_local_decay(indexes, baits, preys, decay_model="legacy", **kwargs)
        normalized = compute_local_decay(indexes, baits, preys, decay_model="normalized", **kwargs)

        self.assertLess(
            float(np.nanmean(legacy["p_value"].to_numpy())),
            float(np.nanmean(normalized["p_value"].to_numpy())),
        )

    def test_unknown_decay_model_is_rejected(self) -> None:
        indexes, baits, preys = self._fixture()
        with self.assertRaises(ValueError):
            compute_local_decay(indexes, baits, preys, decay_model="spline")
        self.assertEqual(DECAY_MODELS, {"legacy", "normalized", "anchored"})


if __name__ == "__main__":
    unittest.main()
