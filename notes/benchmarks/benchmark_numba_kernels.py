"""Synthetic-data numpy vs. numba speedup benchmark.

Orchestrates itself as a subprocess per (workflow, backend) combination via
the hidden `_run-single` mode, reusing `_report.py`'s profiling/report
infrastructure so this produces the same CSV/Markdown/HTML/plot report shape
(wall time, peak RSS, CPU utilization, and a numba-vs-numpy speedup chart) as
`benchmark_reference_real_data.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import fisher_exact

from _report import (
    BenchmarkStep,
    result_to_record,
    run_profiled_step,
    write_profile_report,
)

from touche.apa import compute_apa
from touche.backends import has_numba
from touche.background import compute_ep_and_background
from touche.local_decay import compute_local_decay
from touche.models import ContactIndex
from touche.stats import fisher_greater

WORKFLOWS = ["background", "apa", "local-decay"]


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-single":
        return main_run_single(sys.argv[2:])
    return main_orchestrate(sys.argv[1:])


# --------------------------------------------------------------------------
# Orchestrator: builds one subprocess step per (workflow, backend), profiles
# each with _report.run_profiled_step, and writes the shared report.
# --------------------------------------------------------------------------


def main_orchestrate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark touche's numpy vs. numba backends on synthetic data."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark/numba-kernels"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--contacts", type=int, default=100_000)
    parser.add_argument("--baits", type=int, default=300)
    parser.add_argument("--preys", type=int, default=300)
    parser.add_argument("--genome-size", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare-kernels", action="store_true")
    parser.add_argument("--lowess-backend", choices=["statsmodels", "numba"], default="statsmodels")
    parser.add_argument("--fisher-backend", choices=["scipy", "numba"], default="scipy")
    parser.add_argument("--lowess-iterations", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    work_dir = args.work_dir
    logs_dir = work_dir / "logs"
    report_dir = args.report_dir or work_dir / "report"
    results_jsonl = work_dir / "benchmark-results.jsonl"
    logs_dir.mkdir(parents=True, exist_ok=True)

    backends = ["numpy", "numba"] if has_numba() else ["numpy"]
    if not has_numba():
        print(
            "numba is not installed; falling back to numpy-only "
            "(install with `uv sync --extra fast` to get a speedup comparison)",
            file=sys.stderr,
        )

    steps = build_steps(
        python=args.python,
        backends=backends,
        contacts=args.contacts,
        baits=args.baits,
        preys=args.preys,
        genome_size=args.genome_size,
        repeats=args.repeats,
        lowess_backend=args.lowess_backend,
        fisher_backend=args.fisher_backend,
        lowess_iterations=args.lowess_iterations,
        compare_kernels=args.compare_kernels and "numba" in backends,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "backends": backends,
                    "steps": [
                        {"name": s.name, "group": s.group, "command": s.command} for s in steps
                    ],
                },
                indent=2,
            )
        )
        return 0

    results = []
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for step in steps:
            if args.progress:
                print(f"running {step.name}", file=sys.stderr, flush=True)
            result = run_profiled_step(
                step, logs_dir=logs_dir, poll_interval=args.poll_interval, live_stderr=args.progress
            )
            results.append(result)
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            if result.returncode != 0:
                print(
                    f"{step.name} failed (returncode={result.returncode}); see {result.stderr_log}",
                    file=sys.stderr,
                )

    records = [result_to_record(item) for item in results]
    warn_on_result_mismatch(records, backends)

    if not args.no_report:
        speedup_pairs = build_speedup_pairs(records, backends)
        write_profile_report(records, report_dir=report_dir, speedup_pairs=speedup_pairs)

    return 0 if all(result.returncode == 0 for result in results) else 1


def build_steps(
    *,
    python: str,
    backends: list[str],
    contacts: int,
    baits: int,
    preys: int,
    genome_size: int,
    repeats: int,
    lowess_backend: str,
    fisher_backend: str,
    lowess_iterations: int,
    compare_kernels: bool,
) -> list[BenchmarkStep]:
    script = str(Path(__file__).resolve())
    common = [
        "--contacts",
        str(contacts),
        "--baits",
        str(baits),
        "--preys",
        str(preys),
        "--genome-size",
        str(genome_size),
        "--repeats",
        str(repeats),
    ]
    steps: list[BenchmarkStep] = []
    for workflow in WORKFLOWS:
        for backend in backends:
            extra = (
                [
                    "--lowess-backend",
                    lowess_backend,
                    "--fisher-backend",
                    fisher_backend,
                    "--lowess-iterations",
                    str(lowess_iterations),
                ]
                if workflow == "local-decay"
                else []
            )
            suffix = f"-{backend}" if len(backends) > 1 else ""
            steps.append(
                BenchmarkStep(
                    name=f"{workflow}{suffix}",
                    group=workflow,
                    command=[
                        python,
                        script,
                        "_run-single",
                        "--workflow",
                        workflow,
                        "--backend",
                        backend,
                        *common,
                        *extra,
                    ],
                )
            )
    steps.append(
        BenchmarkStep(
            name="fisher",
            group="fisher",
            command=[python, script, "_run-single", "--workflow", "fisher", "--repeats", str(repeats)],
        )
    )
    if compare_kernels:
        steps.append(
            BenchmarkStep(
                name="kernel-variants",
                group="background",
                command=[
                    python,
                    script,
                    "_run-single",
                    "--workflow",
                    "kernel-variants",
                    *common,
                ],
            )
        )
    return steps


def build_speedup_pairs(
    records: list[dict[str, Any]], backends: list[str]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if "numpy" not in backends or "numba" not in backends:
        return []
    by_name = {record.get("name"): record for record in records}
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for workflow in WORKFLOWS:
        numpy_record = by_name.get(f"{workflow}-numpy")
        numba_record = by_name.get(f"{workflow}-numba")
        if numpy_record is not None and numba_record is not None:
            pairs.append(
                (workflow, _with_median_elapsed(numpy_record), _with_median_elapsed(numba_record))
            )
    return pairs


def _with_median_elapsed(record: dict[str, Any]) -> dict[str, Any]:
    """Use the in-process warm median (excludes subprocess/import/JIT startup) for speedup math."""

    command_json = record.get("command_json")
    median_seconds = command_json.get("median_seconds") if isinstance(command_json, dict) else None
    patched = dict(record)
    if isinstance(median_seconds, (int, float)):
        patched["elapsed_seconds"] = median_seconds
    return patched


def warn_on_result_mismatch(records: list[dict[str, Any]], backends: list[str]) -> None:
    if "numpy" not in backends or "numba" not in backends:
        return
    by_name = {record.get("name"): record for record in records}
    for workflow in WORKFLOWS:
        numpy_record = by_name.get(f"{workflow}-numpy")
        numba_record = by_name.get(f"{workflow}-numba")
        if numpy_record is None or numba_record is None:
            continue
        numpy_json = numpy_record.get("command_json") or {}
        numba_json = numba_record.get("command_json") or {}
        for key in ("rows", "matrix_sum"):
            if key in numpy_json and numpy_json.get(key) != numba_json.get(key):
                print(
                    f"WARNING: {workflow} {key} mismatch between backends: "
                    f"numpy={numpy_json.get(key)!r} numba={numba_json.get(key)!r}",
                    file=sys.stderr,
                )


# --------------------------------------------------------------------------
# _run-single: runs one (workflow, backend) combination in-process, prints a
# JSON summary to stdout. Invoked as a subprocess by the orchestrator above.
# --------------------------------------------------------------------------


def main_run_single(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow",
        required=True,
        choices=["background", "apa", "local-decay", "fisher", "kernel-variants"],
    )
    parser.add_argument("--backend", default="numpy", choices=["numpy", "numba"])
    parser.add_argument("--contacts", type=int, default=100_000)
    parser.add_argument("--baits", type=int, default=300)
    parser.add_argument("--preys", type=int, default=300)
    parser.add_argument("--genome-size", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lowess-backend", default="statsmodels", choices=["statsmodels", "numba"])
    parser.add_argument("--fisher-backend", default="scipy", choices=["scipy", "numba"])
    parser.add_argument("--lowess-iterations", type=int, default=3)
    args = parser.parse_args(argv)

    if args.workflow == "fisher":
        summary = run_fisher_single(repeats=args.repeats)
    elif args.workflow == "kernel-variants":
        summary = run_kernel_variants_single(
            contacts=args.contacts,
            baits=args.baits,
            preys=args.preys,
            genome_size=args.genome_size,
            repeats=args.repeats,
        )
    else:
        indexes, baits, preys = make_background_inputs(
            contacts=args.contacts, baits=args.baits, preys=args.preys, genome_size=args.genome_size
        )
        if args.workflow == "background":
            summary = run_background_single(indexes, baits, preys, backend=args.backend, repeats=args.repeats)
        elif args.workflow == "apa":
            summary = run_apa_single(indexes, baits, preys, backend=args.backend, repeats=args.repeats)
        else:
            summary = run_local_decay_single(
                indexes,
                baits,
                preys,
                backend=args.backend,
                repeats=args.repeats,
                lowess_backend=args.lowess_backend,
                fisher_backend=args.fisher_backend,
                lowess_iterations=args.lowess_iterations,
            )

    print(json.dumps(summary))
    return 0


def run_background_single(
    indexes: dict[str, ContactIndex], baits: pl.DataFrame, preys: pl.DataFrame, *, backend: str, repeats: int
) -> dict[str, Any]:
    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "min_bg_distance": 10_000,
        "max_bg_distance": 150_000,
    }
    result = compute_ep_and_background(indexes, baits, preys, backend=backend, **kwargs)  # warm up
    times = timed_repeats(
        lambda: compute_ep_and_background(indexes, baits, preys, backend=backend, **kwargs), repeats
    )
    return {"rows": len(result), **timing_summary(times)}


def run_apa_single(
    indexes: dict[str, ContactIndex], baits: pl.DataFrame, preys: pl.DataFrame, *, backend: str, repeats: int
) -> dict[str, Any]:
    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "pixels": 50,
        "shift": 0,
    }
    result = compute_apa(indexes, baits, preys, backend=backend, **kwargs)  # warm up
    times = timed_repeats(lambda: compute_apa(indexes, baits, preys, backend=backend, **kwargs), repeats)
    matrix_sum = int(result.matrix.drop("bin_label").to_numpy().sum())
    return {"matrix_sum": matrix_sum, **timing_summary(times)}


def run_local_decay_single(
    indexes: dict[str, ContactIndex],
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    backend: str,
    repeats: int,
    lowess_backend: str,
    fisher_backend: str,
    lowess_iterations: int,
) -> dict[str, Any]:
    local_baits = baits.select(["chr", "center"])
    bait_centers = local_baits["center"].to_numpy().astype(np.int64)
    offsets = np.tile(np.asarray([-50_000, 50_000], dtype=np.int64), len(bait_centers))
    prey_centers = np.repeat(bait_centers, 2) + offsets
    local_preys = pl.DataFrame({"chr": ["chr1"] * len(prey_centers), "center": prey_centers})

    kwargs = {
        "dist": 100_000,
        "cap": 500,
        "min_distance": 1_000,
        "lowess_window": 500,
        "lowess_backend": lowess_backend,
        "fisher_backend": fisher_backend,
        "lowess_iterations": lowess_iterations,
    }
    result = compute_local_decay(
        indexes, local_baits, local_preys, backend=backend, **kwargs
    )  # warm up
    times = timed_repeats(
        lambda: compute_local_decay(indexes, local_baits, local_preys, backend=backend, **kwargs),
        repeats,
    )
    return {"rows": len(result), **timing_summary(times)}


def run_fisher_single(*, repeats: int) -> dict[str, Any]:
    rng = np.random.default_rng(20260708)
    tables = rng.integers(0, 500, size=(50_000, 4))
    tables[:, 3] += 1

    scipy_times: list[float] = []
    direct_times: list[float] = []
    scipy_values: list[float] = []
    direct_values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        scipy_values = [
            fisher_exact([[a, b], [c, d]], alternative="greater").pvalue for a, b, c, d in tables
        ]
        scipy_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        direct_values = [fisher_greater(a, b, c, d) for a, b, c, d in tables]
        direct_times.append(time.perf_counter() - started)

    np.testing.assert_allclose(direct_values, scipy_values)
    return {
        "rows": len(tables),
        "median_seconds": median(direct_times),
        "min_seconds": min(direct_times),
        "mean_seconds": statistics.fmean(direct_times),
        "timings": [
            {"step": "scipy_fisher_exact", "elapsed_seconds": median(scipy_times)},
            {"step": "direct_hypergeom_sf", "elapsed_seconds": median(direct_times)},
        ],
    }


def run_kernel_variants_single(
    *, contacts: int, baits: int, preys: int, genome_size: int, repeats: int
) -> dict[str, Any]:
    from touche.numba_kernels import (
        count_ep_background_pairs_eager_numba,
        count_ep_background_pairs_numba,
    )

    indexes, baits_df, preys_df = make_background_inputs(
        contacts=contacts, baits=baits, preys=preys, genome_size=genome_size
    )
    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "min_bg_distance": 10_000,
        "max_bg_distance": 150_000,
    }
    index = indexes["chr1"]
    bait_centers = baits_df["center"].to_numpy().astype(np.int64)
    prey_centers = preys_df["center"].to_numpy().astype(np.int64)
    pair_bait_index, pair_prey_index = make_pair_indexes(
        bait_centers, prey_centers, min_distance=kwargs["min_distance"], max_distance=kwargs["max_distance"]
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

    eager_times = timed_repeats(lambda: count_ep_background_pairs_eager_numba(*kernel_args), repeats)
    optimized_times = timed_repeats(lambda: count_ep_background_pairs_numba(*kernel_args), repeats)
    return {
        "rows": len(pair_bait_index),
        "median_seconds": median(optimized_times),
        "min_seconds": min(optimized_times),
        "mean_seconds": statistics.fmean(optimized_times),
        "timings": [
            {"step": "kernel_eager", "elapsed_seconds": median(eager_times)},
            {"step": "kernel_optimized", "elapsed_seconds": median(optimized_times)},
        ],
    }


def make_pair_indexes(
    bait_centers: np.ndarray, prey_centers: np.ndarray, *, min_distance: int, max_distance: int
) -> tuple[np.ndarray, np.ndarray]:
    pair_bait_indexes: list[int] = []
    pair_prey_indexes: list[int] = []
    for bait_index, bait_center in enumerate(bait_centers):
        distances = np.abs(prey_centers - bait_center)
        candidate_indexes = np.flatnonzero((distances >= min_distance) & (distances <= max_distance))
        pair_bait_indexes.extend([bait_index] * len(candidate_indexes))
        pair_prey_indexes.extend(candidate_indexes.tolist())
    return (
        np.asarray(pair_bait_indexes, dtype=np.int64),
        np.asarray(pair_prey_indexes, dtype=np.int64),
    )


def make_background_inputs(
    *, contacts: int, baits: int, preys: int, genome_size: int
) -> tuple[dict[str, ContactIndex], pl.DataFrame, pl.DataFrame]:
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


def anchors(chrom: str, centers: np.ndarray) -> pl.DataFrame:
    n = len(centers)
    return pl.DataFrame(
        {
            "chr": [chrom] * n,
            "start": centers - 50,
            "end": centers + 50,
            "strand": ["+"] * n,
            "center": centers,
        }
    )


def timed_repeats(fn, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return times


def timing_summary(times: list[float]) -> dict[str, float]:
    return {
        "median_seconds": median(times),
        "min_seconds": min(times),
        "mean_seconds": statistics.fmean(times),
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values))


if __name__ == "__main__":
    raise SystemExit(main())
