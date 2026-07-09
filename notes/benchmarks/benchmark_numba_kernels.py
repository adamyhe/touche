"""Synthetic-data profiling for touche's numba-accelerated counting kernels.

Orchestrates itself as a subprocess per workflow via the hidden `_run-single`
mode, reusing `scripts/_report.py`'s profiling/report infrastructure so this
produces the same CSV/Markdown/HTML/plot report shape (wall time, peak RSS,
CPU utilization) as `scripts/reference_replication.py`.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from _report import (  # noqa: E402
    BenchmarkStep,
    result_to_record,
    run_profiled_step,
    write_profile_report,
)

from touche.apa import compute_apa
from touche.background import compute_ep_and_background
from touche.local_decay import compute_local_decay
from touche.models import ContactIndex

WORKFLOWS = ["background", "apa", "local-decay"]


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-single":
        return main_run_single(sys.argv[2:])
    return main_orchestrate(sys.argv[1:])


# --------------------------------------------------------------------------
# Orchestrator: builds one subprocess step per workflow, profiles each with
# _report.run_profiled_step, and writes the shared report.
# --------------------------------------------------------------------------


def main_orchestrate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Profile touche's numba-accelerated counting kernels on synthetic data."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark/numba-kernels"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--contacts", type=int, default=100_000)
    parser.add_argument("--baits", type=int, default=300)
    parser.add_argument("--preys", type=int, default=300)
    parser.add_argument("--genome-size", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare-kernels", action="store_true")
    parser.add_argument("--lowess-backend", choices=["statsmodels", "numba"], default="numba")
    parser.add_argument("--fisher-backend", choices=["scipy", "numba"], default="numba")
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="local-decay n_jobs -- baits to process concurrently (default: 1, sequential).",
    )
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

    steps = build_steps(
        python=args.python,
        contacts=args.contacts,
        baits=args.baits,
        preys=args.preys,
        genome_size=args.genome_size,
        repeats=args.repeats,
        lowess_backend=args.lowess_backend,
        fisher_backend=args.fisher_backend,
        jobs=args.jobs,
        lowess_iterations=args.lowess_iterations,
        compare_kernels=args.compare_kernels,
    )

    if args.dry_run:
        print(
            json.dumps(
                {"steps": [{"name": s.name, "group": s.group, "command": s.command} for s in steps]},
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

    if not args.no_report:
        write_profile_report(records, report_dir=report_dir)

    return 0 if all(result.returncode == 0 for result in results) else 1


def build_steps(
    *,
    python: str,
    contacts: int,
    baits: int,
    preys: int,
    genome_size: int,
    repeats: int,
    lowess_backend: str,
    fisher_backend: str,
    jobs: int,
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
        extra = (
            [
                "--lowess-backend",
                lowess_backend,
                "--fisher-backend",
                fisher_backend,
                "--jobs",
                str(jobs),
                "--lowess-iterations",
                str(lowess_iterations),
            ]
            if workflow == "local-decay"
            else []
        )
        steps.append(
            BenchmarkStep(
                name=workflow,
                group=workflow,
                command=[python, script, "_run-single", "--workflow", workflow, *common, *extra],
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


# --------------------------------------------------------------------------
# _run-single: runs one workflow in-process, prints a JSON summary to
# stdout. Invoked as a subprocess by the orchestrator above.
# --------------------------------------------------------------------------


def main_run_single(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow",
        required=True,
        choices=["background", "apa", "local-decay", "kernel-variants"],
    )
    parser.add_argument("--contacts", type=int, default=100_000)
    parser.add_argument("--baits", type=int, default=300)
    parser.add_argument("--preys", type=int, default=300)
    parser.add_argument("--genome-size", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lowess-backend", default="numba", choices=["statsmodels", "numba"])
    parser.add_argument("--fisher-backend", default="numba", choices=["scipy", "numba"])
    parser.add_argument("--jobs", "-j", type=int, default=1)
    parser.add_argument("--lowess-iterations", type=int, default=3)
    args = parser.parse_args(argv)

    if args.workflow == "kernel-variants":
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
            summary = run_background_single(indexes, baits, preys, repeats=args.repeats)
        elif args.workflow == "apa":
            summary = run_apa_single(indexes, baits, preys, repeats=args.repeats)
        else:
            summary = run_local_decay_single(
                indexes,
                baits,
                preys,
                repeats=args.repeats,
                lowess_backend=args.lowess_backend,
                fisher_backend=args.fisher_backend,
                n_jobs=args.jobs,
                lowess_iterations=args.lowess_iterations,
            )

    print(json.dumps(summary))
    return 0


def run_background_single(
    indexes: dict[str, ContactIndex], baits: pl.DataFrame, preys: pl.DataFrame, *, repeats: int
) -> dict[str, Any]:
    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "min_bg_distance": 10_000,
        "max_bg_distance": 150_000,
    }
    result = compute_ep_and_background(indexes, baits, preys, **kwargs)  # warm up
    times = timed_repeats(lambda: compute_ep_and_background(indexes, baits, preys, **kwargs), repeats)
    return {"rows": len(result), **timing_summary(times)}


def run_apa_single(
    indexes: dict[str, ContactIndex], baits: pl.DataFrame, preys: pl.DataFrame, *, repeats: int
) -> dict[str, Any]:
    kwargs = {
        "min_distance": 25_000,
        "max_distance": 150_000,
        "window": 10_000,
        "pixels": 50,
        "shift": 0,
    }
    result = compute_apa(indexes, baits, preys, **kwargs)  # warm up
    times = timed_repeats(lambda: compute_apa(indexes, baits, preys, **kwargs), repeats)
    matrix_sum = int(result.matrix.drop("bin_label").to_numpy().sum())
    return {"matrix_sum": matrix_sum, **timing_summary(times)}


def run_local_decay_single(
    indexes: dict[str, ContactIndex],
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    repeats: int,
    lowess_backend: str,
    fisher_backend: str,
    n_jobs: int,
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
        "n_jobs": n_jobs,
        "lowess_iterations": lowess_iterations,
    }
    result = compute_local_decay(indexes, local_baits, local_preys, **kwargs)  # warm up
    times = timed_repeats(
        lambda: compute_local_decay(indexes, local_baits, local_preys, **kwargs), repeats
    )
    return {"rows": len(result), **timing_summary(times)}


def run_kernel_variants_single(
    *, contacts: int, baits: int, preys: int, genome_size: int, repeats: int
) -> dict[str, Any]:
    """Compare the current binary-search kernel against the naive linear-scan one it replaced."""

    from touche.numba.background import (
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
