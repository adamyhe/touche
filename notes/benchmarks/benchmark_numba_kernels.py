from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from touche.background import compute_ep_and_background
from touche.apa import compute_apa
from touche.local_decay import compute_local_decay
from touche.models import ContactIndex
from touche.stats import fisher_greater


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow", choices=["background", "apa", "local-decay", "fisher"], default="background"
    )
    parser.add_argument("--contacts", type=int, default=100_000)
    parser.add_argument("--baits", type=int, default=300)
    parser.add_argument("--preys", type=int, default=300)
    parser.add_argument("--genome-size", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare-kernels", action="store_true")
    parser.add_argument(
        "--lowess-backend",
        choices=["statsmodels", "numba"],
        default="statsmodels",
    )
    parser.add_argument("--lowess-iterations", type=int, default=3)
    args = parser.parse_args()

    if args.workflow == "fisher":
        return benchmark_fisher(repeats=args.repeats)

    indexes, baits, preys = make_background_inputs(
        contacts=args.contacts,
        baits=args.baits,
        preys=args.preys,
        genome_size=args.genome_size,
    )
    if args.workflow == "apa":
        return benchmark_apa(indexes, baits, preys, repeats=args.repeats)
    if args.workflow == "local-decay":
        return benchmark_local_decay(
            indexes,
            baits,
            preys,
            repeats=args.repeats,
            lowess_backend=args.lowess_backend,
            lowess_iterations=args.lowess_iterations,
        )

    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "min_bg_distance": 10_000,
        "max_bg_distance": 150_000,
    }

    numpy_result, numpy_times = run_repeats(
        indexes, baits, preys, backend="numpy", repeats=args.repeats, kwargs=kwargs
    )
    try:
        numba_result, numba_times = run_repeats(
            indexes, baits, preys, backend="numba", repeats=args.repeats, kwargs=kwargs
        )
    except RuntimeError as exc:
        print(f"numba unavailable: {exc}")
        return 0

    pd.testing.assert_frame_equal(numba_result, numpy_result)
    print_result("numpy", numpy_times)
    print_result("numba", numba_times)
    if median(numba_times) > 0:
        print(f"median_speedup_excluding_compile={median(numpy_times) / median(numba_times):.2f}x")
    print(f"rows={len(numpy_result)}")
    if args.compare_kernels:
        benchmark_kernel_variants(indexes, baits, preys, repeats=args.repeats, kwargs=kwargs)
    return 0


def benchmark_fisher(*, repeats: int) -> int:
    rng = np.random.default_rng(20260708)
    tables = rng.integers(0, 500, size=(50_000, 4))
    tables[:, 3] += 1

    scipy_times: list[float] = []
    direct_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        scipy_values = [
            fisher_exact([[a, b], [c, d]], alternative="greater").pvalue
            for a, b, c, d in tables
        ]
        scipy_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        direct_values = [fisher_greater(a, b, c, d) for a, b, c, d in tables]
        direct_times.append(time.perf_counter() - started)

    np.testing.assert_allclose(direct_values, scipy_values)
    print_result("scipy_fisher_exact", scipy_times)
    print_result("direct_hypergeom_sf", direct_times)
    if median(direct_times) > 0:
        print(f"median_speedup={median(scipy_times) / median(direct_times):.2f}x")
    return 0


def benchmark_apa(
    indexes: dict[str, ContactIndex],
    baits: pd.DataFrame,
    preys: pd.DataFrame,
    *,
    repeats: int,
    lowess_backend: str,
    lowess_iterations: int,
) -> int:
    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "pixels": 50,
        "shift": 0,
    }
    numpy_result, numpy_times = run_apa_repeats(
        indexes, baits, preys, backend="numpy", repeats=repeats, kwargs=kwargs
    )
    try:
        numba_result, numba_times = run_apa_repeats(
            indexes, baits, preys, backend="numba", repeats=repeats, kwargs=kwargs
        )
    except RuntimeError as exc:
        print(f"numba unavailable: {exc}")
        return 0

    pd.testing.assert_frame_equal(numba_result.matrix, numpy_result.matrix)
    pd.testing.assert_frame_equal(numba_result.bait_signal, numpy_result.bait_signal)
    pd.testing.assert_frame_equal(numba_result.prey_signal, numpy_result.prey_signal)
    print_result("numpy", numpy_times)
    print_result("numba", numba_times)
    if median(numba_times) > 0:
        print(f"median_speedup_excluding_compile={median(numpy_times) / median(numba_times):.2f}x")
    print(f"matrix_sum={int(numba_result.matrix.to_numpy().sum())}")
    return 0


def benchmark_local_decay(
    indexes: dict[str, ContactIndex],
    baits: pd.DataFrame,
    preys: pd.DataFrame,
    *,
    repeats: int,
    lowess_backend: str,
    lowess_iterations: int,
) -> int:
    local_baits = baits[["chr", "center"]].copy()
    bait_centers = local_baits["center"].to_numpy(dtype=np.int64)
    offsets = np.tile(np.asarray([-50_000, 50_000], dtype=np.int64), len(bait_centers))
    local_preys = pd.DataFrame(
        {
            "chr": "chr1",
            "center": np.repeat(bait_centers, 2) + offsets,
        }
    )
    kwargs = {
        "dist": 100_000,
        "cap": 500,
        "min_distance": 1_000,
        "lowess_window": 500,
        "lowess_backend": lowess_backend,
        "lowess_iterations": lowess_iterations,
    }
    numpy_result, numpy_times = run_local_decay_repeats(
        indexes,
        local_baits,
        local_preys,
        backend="numpy",
        repeats=repeats,
        kwargs=kwargs,
    )
    try:
        numba_result, numba_times = run_local_decay_repeats(
            indexes,
            local_baits,
            local_preys,
            backend="numba",
            repeats=repeats,
            kwargs=kwargs,
        )
    except RuntimeError as exc:
        print(f"numba unavailable: {exc}")
        return 0

    pd.testing.assert_frame_equal(numba_result, numpy_result)
    print_result("numpy", numpy_times)
    print_result("numba", numba_times)
    if median(numba_times) > 0:
        print(f"median_speedup_excluding_compile={median(numpy_times) / median(numba_times):.2f}x")
    print(f"rows={len(numba_result)}")
    return 0


def make_background_inputs(
    *, contacts: int, baits: int, preys: int, genome_size: int
) -> tuple[dict[str, ContactIndex], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260708)
    pos_a = rng.integers(1, genome_size - 200_000, size=contacts, dtype=np.int64)
    distances = rng.integers(5_000, 250_000, size=contacts, dtype=np.int64)
    pos_b = pos_a + distances
    indexes = {
        "chr1": ContactIndex(
            chrom="chr1",
            pos_a=np.minimum(pos_a, pos_b),
            pos_b=np.maximum(pos_a, pos_b),
            strand_a=np.full(contacts, "+"),
            strand_b=np.full(contacts, "-"),
            mapq_a=np.full(contacts, 30, dtype=np.int64),
            mapq_b=np.full(contacts, 30, dtype=np.int64),
        )
    }

    bait_centers = np.linspace(500_000, genome_size - 500_000, baits, dtype=np.int64)
    offsets = rng.integers(25_000, 150_000, size=preys, dtype=np.int64)
    prey_base = np.linspace(525_000, genome_size - 350_000, preys, dtype=np.int64)
    prey_centers = np.minimum(prey_base + offsets, genome_size - 100_000)
    baits_df = anchors("chr1", bait_centers)
    preys_df = anchors("chr1", prey_centers)
    return indexes, baits_df, preys_df


def anchors(chrom: str, centers: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chr": chrom,
            "start": centers - 50,
            "end": centers + 50,
            "strand": "+",
            "center": centers,
        }
    )


def run_repeats(
    indexes: dict[str, ContactIndex],
    baits: pd.DataFrame,
    preys: pd.DataFrame,
    *,
    backend: str,
    repeats: int,
    kwargs: dict[str, int],
) -> tuple[pd.DataFrame, list[float]]:
    result = compute_ep_and_background(indexes, baits, preys, backend=backend, **kwargs)
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compute_ep_and_background(indexes, baits, preys, backend=backend, **kwargs)
        times.append(time.perf_counter() - started)
    return result, times


def run_apa_repeats(
    indexes: dict[str, ContactIndex],
    baits: pd.DataFrame,
    preys: pd.DataFrame,
    *,
    backend: str,
    repeats: int,
    kwargs: dict[str, int],
):
    result = compute_apa(indexes, baits, preys, backend=backend, **kwargs)
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compute_apa(indexes, baits, preys, backend=backend, **kwargs)
        times.append(time.perf_counter() - started)
    return result, times


def run_local_decay_repeats(
    indexes: dict[str, ContactIndex],
    baits: pd.DataFrame,
    preys: pd.DataFrame,
    *,
    backend: str,
    repeats: int,
    kwargs: dict[str, int],
):
    result = compute_local_decay(indexes, baits, preys, backend=backend, **kwargs)
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compute_local_decay(indexes, baits, preys, backend=backend, **kwargs)
        times.append(time.perf_counter() - started)
    return result, times


def print_result(name: str, times: list[float]) -> None:
    print(
        f"{name}: min={min(times):.4f}s median={median(times):.4f}s "
        f"mean={statistics.fmean(times):.4f}s repeats={len(times)}"
    )


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def benchmark_kernel_variants(
    indexes: dict[str, ContactIndex],
    baits: pd.DataFrame,
    preys: pd.DataFrame,
    *,
    repeats: int,
    kwargs: dict[str, int],
) -> None:
    from touche.numba_kernels import (
        count_ep_background_pairs_eager_numba,
        count_ep_background_pairs_numba,
    )

    index = indexes["chr1"]
    bait_centers = baits["center"].to_numpy(dtype=np.int64)
    prey_centers = preys["center"].to_numpy(dtype=np.int64)
    pair_bait_index, pair_prey_index = make_pair_indexes(
        bait_centers,
        prey_centers,
        min_distance=kwargs["min_distance"],
        max_distance=kwargs["max_distance"],
    )
    kernel_args = (
        index.pos_a.astype(np.int64, copy=False),
        index.pos_b.astype(np.int64, copy=False),
        bait_centers,
        prey_centers,
        pair_bait_index,
        pair_prey_index,
        kwargs["window"],
        kwargs["min_bg_distance"],
        kwargs["max_bg_distance"],
    )

    eager = count_ep_background_pairs_eager_numba(*kernel_args)
    optimized = count_ep_background_pairs_numba(*kernel_args)
    np.testing.assert_array_equal(optimized[0], eager[0])
    np.testing.assert_array_equal(optimized[1], eager[1])

    eager_times = time_kernel(count_ep_background_pairs_eager_numba, kernel_args, repeats)
    optimized_times = time_kernel(count_ep_background_pairs_numba, kernel_args, repeats)
    print_result("kernel_eager", eager_times)
    print_result("kernel_optimized", optimized_times)
    if median(optimized_times) > 0:
        print(
            "kernel_optimized_speedup="
            f"{median(eager_times) / median(optimized_times):.2f}x"
        )


def make_pair_indexes(
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    *,
    min_distance: int,
    max_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair_bait_indexes: list[int] = []
    pair_prey_indexes: list[int] = []
    for bait_index, bait_center in enumerate(bait_centers):
        distances = np.abs(prey_centers - bait_center)
        candidate_indexes = np.flatnonzero(
            (distances >= min_distance) & (distances <= max_distance)
        )
        pair_bait_indexes.extend([bait_index] * len(candidate_indexes))
        pair_prey_indexes.extend(candidate_indexes.tolist())
    return (
        np.asarray(pair_bait_indexes, dtype=np.int64),
        np.asarray(pair_prey_indexes, dtype=np.int64),
    )


def time_kernel(function, args: tuple[object, ...], repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - started)
    return times


if __name__ == "__main__":
    raise SystemExit(main())
