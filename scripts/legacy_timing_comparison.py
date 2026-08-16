"""Time touche against the original Danko-Lab E-P_contacts reference pipeline.

Downloads the same real inputs `reference_replication.py` uses, then runs one
representative touche command and its legacy `.bsh`/`.py` counterpart for each
of the three workflow families (local-decay, APA, EP/background) on identical
inputs and parameters, profiling both with the same wall-time/RSS/CPU harness.

See `legacy_timing_comparison.md` for prerequisites (a local clone of
Danko-Lab/E-P_contacts, plus its R/rpy2 conda environment for the local-decay
step) and usage.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _report import (
    BenchmarkResult,
    BenchmarkStep,
    plot_metric,
    result_to_record,
    run_profiled_step,
    write_csv,
    write_profile_report,
)
from reference_replication import (
    CORNELL_FTP_BASE,
    GEO_BASE,
    LOCAL_DECAY_BAITS,
    LOCAL_DECAY_PREYS,
    MESC_BAITS,
    MESC_PREYS,
    REFERENCE_RAW_BASE,
    Download,
    build_cache_steps,
    build_steps,
    download_reference_inputs,
    missing_output_paths_local,
)

REFERENCE_REPO_URL = "https://github.com/Danko-Lab/E-P_contacts.git"


def available_cores() -> int:
    """Cores actually usable by this process.

    Prefers `NUMBA_NUM_THREADS` (what numba's own kernels will use, if set),
    then `os.sched_getaffinity` (respects scheduler/cgroup CPU pinning on
    Linux -- unlike `os.cpu_count()`, which reports the whole node's core
    count regardless of what a shared-HPC job was actually allocated), then
    falls back to `os.cpu_count()` on platforms without `sched_getaffinity`
    (e.g. macOS).
    """
    env_threads = os.environ.get("NUMBA_NUM_THREADS")
    if env_threads and env_threads.isdigit():
        return max(1, int(env_threads))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1

SAMPLE_PAIRS = {
    "dmso": "mESCs_DMSO_30_intra.mm10.nodups.pairs.gz",
    "flv": "mESCs_FLV_30_intra.mm10.nodups.pairs.gz",
    "trp": "mESCs_TRP_30_intra.mm10.nodups.pairs.gz",
}
K562_PAIRS = "GSE206131_K562_cis_mapq30_pairs.txt.gz"

LOCAL_DECAY_BSH = "Contact_normalization_by_local_decay/ContactCaller_microC.bsh"
APA_BSH = "APA_and_inter-sample_APA/MicroC_Stranded_Aggregation_pipeline_with_1D_signal.bsh"
BACKGROUND_BSH = "EP_contacts_compared_to_local_background/MicroC_EP_and_BG_contacts.bsh"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched touche vs. legacy E-P_contacts steps on identical real "
            "inputs/parameters and compare wall time, peak RSS, and CPU usage."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Local clone of https://github.com/Danko-Lab/E-P_contacts (see --clone-reference).",
    )
    parser.add_argument(
        "--clone-reference",
        action="store_true",
        help="git clone the reference repo into --reference-dir if it doesn't exist yet.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark/legacy-timing-comparison"))
    parser.add_argument("--touche-python", default=sys.executable, help="Python executable for -m touche")
    parser.add_argument(
        "--legacy-shell-prefix",
        default="",
        help=(
            "Command prefix to invoke bash/python inside the legacy repo's own environment, "
            "as a single shell-quoted string, e.g. "
            '--legacy-shell-prefix "conda run -n EP-contacts --no-capture-output". '
            "Local-decay's ContactCaller_microC.py needs R (>=4.1) + rpy2 on this path. "
            "(Passed as one string, not nargs=\"*\", so option-looking tokens like -n "
            "inside it don't get parsed as this script's own flags.)"
        ),
    )
    parser.add_argument(
        "--legacy-ld-preload",
        default="",
        help=(
            "Path to a shared library to LD_PRELOAD before running legacy steps, e.g. "
            'a conda env\'s own libstdc++.so.6: --legacy-ld-preload '
            '"$(conda run -n EP-contacts python -c \'import sys; print(sys.prefix)\')/lib/libstdc++.so.6". '
            "ContactCaller_microC.bsh unconditionally overwrites LD_LIBRARY_PATH (not appends), "
            "which can wipe out a conda env's newer libstdc++ and expose the HPC node's older "
            "system one instead -- symptom: ImportError: .../libstdc++.so.6: version "
            "'GLIBCXX_3.4.30' not found, from scipy/statsmodels/rpy2's compiled extensions. "
            "LD_PRELOAD is a separate env var that line never touches, so it survives."
        ),
    )
    parser.add_argument(
        "--legacy-cpu",
        type=int,
        default=available_cores(),
        help="Passed through to ContactCaller_microC.bsh's [CPU.threads] argument (default: "
        "detected available cores). See the Core Usage section of legacy_timing_comparison.md "
        "-- that script only echoes this value and doesn't actually use it to throttle.",
    )
    parser.add_argument("--lowess-backend", choices=["statsmodels", "numba"], default="numba")
    parser.add_argument("--fisher-backend", choices=["scipy", "numba"], default="numba")
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=available_cores(),
        help=(
            "touche local-decay call --jobs (bait-level thread pool; default: detected "
            "available cores, see --legacy-cpu). Only local-decay's LOWESS kernel is "
            "numba-parallel; most of its wall time at real-data scale is single-threaded "
            "per-bait glue code that only --jobs parallelizes, not NUMBA_NUM_THREADS -- pass "
            "--jobs 1 to see the sequential-baits baseline instead."
        ),
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=["local-decay", "apa", "background"],
        default=["local-decay", "apa", "background"],
    )
    parser.add_argument("--apa-sample", choices=["dmso", "flv", "trp"], default="dmso")
    parser.add_argument("--background-sample", choices=["dmso", "flv", "trp"], default="dmso")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--fail-on-missing-output",
        action="store_true",
        help="Return a non-zero exit code when a successful step misses expected outputs.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Path to a benchmark-results.jsonl from a previous (e.g. killed or crashed) run. "
            "Steps that already completed successfully there are reused instead of re-run -- "
            "important on a remote box where a single legacy step can take hours."
        ),
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    logs_dir = work_dir / "logs"
    results_jsonl = work_dir / "benchmark-results.jsonl"
    manifest_json = work_dir / "benchmark-manifest.json"
    report_dir = args.report_dir or work_dir / "report"
    for path in [data_dir, output_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if args.clone_reference and not args.reference_dir.exists():
        clone_reference(args.reference_dir)
    if not args.dry_run and not required_reference_scripts_exist(args.reference_dir, args.workflows):
        raise SystemExit(
            f"--reference-dir {args.reference_dir} is missing the E-P_contacts scripts needed "
            f"for {args.workflows}. Clone {REFERENCE_REPO_URL} there, or pass --clone-reference. "
            "See legacy_timing_comparison.md for the R/rpy2 environment the legacy local-decay "
            "step also needs."
        )

    downloads = needed_downloads(
        data_dir,
        workflows=args.workflows,
        apa_sample=args.apa_sample,
        background_sample=args.background_sample,
    )
    cache_steps, cache_paths = build_cache_steps(
        python=args.touche_python, data_dir=data_dir, output_dir=output_dir
    )
    cache_steps = [
        step
        for step in cache_steps
        if step.name == "preprocess-cache-k562" and "local-decay" in args.workflows
    ]
    touche_steps = build_steps(
        python=args.touche_python,
        data_dir=data_dir,
        output_dir=output_dir,
        k562_cache_dir=cache_paths["k562"],
        lowess_backend=args.lowess_backend,
        fisher_backend=args.fisher_backend,
        jobs=args.jobs,
        lowess_iterations=3,
        progress=args.progress,
    )
    wanted_touche_names = touche_step_names(
        workflows=args.workflows, apa_sample=args.apa_sample, background_sample=args.background_sample
    )
    touche_steps = [step for step in touche_steps if step.name in wanted_touche_names]

    legacy_steps = build_legacy_steps(
        reference_dir=args.reference_dir.resolve(),
        data_dir=data_dir.resolve(),
        output_dir=(output_dir / "legacy").resolve(),
        shell_prefix=shlex.split(args.legacy_shell_prefix),
        legacy_ld_preload=args.legacy_ld_preload,
        legacy_cpu=args.legacy_cpu,
        workflows=args.workflows,
        apa_sample=args.apa_sample,
        background_sample=args.background_sample,
    )

    all_steps = [*cache_steps, *interleave(touche_steps, legacy_steps)]

    if args.dry_run:
        print_plan(downloads, all_steps)
        return 0

    download_records: list[dict[str, Any]] = []
    if not args.skip_download:
        download_records = download_reference_inputs(downloads)

    reusable_results = load_resumable_results(args.resume_from) if args.resume_from else {}

    results: list[BenchmarkResult] = []
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for index, step in enumerate(all_steps, start=1):
            reused = reusable_results.get(step.name)
            if reused is not None:
                print(f"[{index}/{len(all_steps)}] reusing {step.name} from --resume-from", file=sys.stderr)
                result = reused
            else:
                if args.progress:
                    print(f"[{index}/{len(all_steps)}] running {step.name}", file=sys.stderr, flush=True)
                result = run_profiled_step(
                    step, logs_dir=logs_dir, poll_interval=args.poll_interval, live_stderr=args.progress
                )
            results.append(result)
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            handle.flush()
            missing_outputs = missing_output_paths_local(result_to_record(result))
            missing_output_failure = (
                args.fail_on_missing_output and result.returncode == 0 and missing_outputs
            )
            if (result.returncode != 0 or missing_output_failure) and not args.keep_going:
                write_manifest(manifest_json, args=args, downloads=download_records, results=results)
                if not args.no_report:
                    write_reports(results, cache_steps=cache_steps, report_dir=report_dir)
                return result.returncode if result.returncode != 0 else 2

    write_manifest(manifest_json, args=args, downloads=download_records, results=results)
    if not args.no_report:
        write_reports(results, cache_steps=cache_steps, report_dir=report_dir)
    return 0


def load_resumable_results(path: Path) -> dict[str, BenchmarkResult]:
    """Load successful step results from a previous run's `benchmark-results.jsonl`."""
    if not path.exists():
        raise SystemExit(f"--resume-from path does not exist: {path}")
    reusable: dict[str, BenchmarkResult] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSONL record in {path} line {line_number}") from exc
            if record.get("returncode") == 0:
                reusable[record["name"]] = BenchmarkResult(**record)
    return reusable


def clone_reference(reference_dir: Path) -> None:
    import subprocess

    reference_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {REFERENCE_REPO_URL} -> {reference_dir}", file=sys.stderr, flush=True)
    subprocess.run(["git", "clone", REFERENCE_REPO_URL, str(reference_dir)], check=True)


def required_reference_scripts_exist(reference_dir: Path, workflows: list[str]) -> bool:
    script_by_workflow = {
        "local-decay": LOCAL_DECAY_BSH,
        "apa": APA_BSH,
        "background": BACKGROUND_BSH,
    }
    return all((reference_dir / script_by_workflow[workflow]).exists() for workflow in workflows)


def touche_step_names(*, workflows: list[str], apa_sample: str, background_sample: str) -> set[str]:
    names: set[str] = set()
    if "local-decay" in workflows:
        names.add("local-decay-call")
    if "apa" in workflows:
        names.add(f"apa-aggregate-{apa_sample}")
    if "background" in workflows:
        names.add(f"background-count-{background_sample}")
    return names


def _ld_preload_prefix(legacy_ld_preload: str) -> list[str]:
    """`export LD_PRELOAD=...` as a leading command, or `[]` if unset.

    A separate env var from `LD_LIBRARY_PATH`, so it survives
    `ContactCaller_microC.bsh`'s unconditional `export LD_LIBRARY_PATH=...`
    overwrite -- see `--legacy-ld-preload`'s help text.
    """
    if not legacy_ld_preload:
        return []
    return [f"export LD_PRELOAD={shlex.quote(legacy_ld_preload)}"]


_UV_ENV_LEAK_VARS = [
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "UV_RUN_RECURSION_DEPTH",
    "UV_PYTHON",
    "UV_PROJECT_ENVIRONMENT",
    "UV_INDEX",
]


def _env_unset_prefix() -> list[str]:
    """`unset ...` for vars `uv run` sets that legacy subprocesses shouldn't inherit.

    `uv run python legacy_timing_comparison.py` sets VIRTUAL_ENV (pointing at
    touche's own venv) and UV_RUN_RECURSION_DEPTH, which then leak into every
    subprocess this script spawns -- including through `conda run` into the
    legacy repo's own Python/R. Confirmed on real hardware: the identical
    generated command succeeded when run directly (no leaked vars) but failed
    with a `rpy2` ModuleNotFoundError when run through this script (leaked
    vars present), even though `rpy2` was independently confirmed installed
    and importable in that same conda env. Unsetting these defensively
    (whether or not each one is actually set) is harmless and removes touche's
    own Python environment from a process tree that should reflect the
    legacy repo's own environment instead.
    """
    return [f"unset {' '.join(_UV_ENV_LEAK_VARS)}"]


def build_legacy_steps(
    *,
    reference_dir: Path,
    data_dir: Path,
    output_dir: Path,
    shell_prefix: list[str],
    legacy_ld_preload: str,
    legacy_cpu: int,
    workflows: list[str],
    apa_sample: str,
    background_sample: str,
) -> list[BenchmarkStep]:
    steps: list[BenchmarkStep] = []
    input_dir = data_dir / "Input_files"
    baits_local = input_dir / LOCAL_DECAY_BAITS
    preys_local = input_dir / LOCAL_DECAY_PREYS
    baits_mesc = input_dir / MESC_BAITS
    preys_mesc = input_dir / MESC_PREYS
    env_prefix = [*_env_unset_prefix(), *_ld_preload_prefix(legacy_ld_preload)]

    if "local-decay" in workflows:
        outdir = output_dir / "local-decay"
        combined_out = outdir / "ContactCaller_microC_output.txt"
        inner = " && ".join(
            [
                *env_prefix,
                f"mkdir -p {shlex.quote(str(outdir))}",
                f"cd {shlex.quote(str(reference_dir))}",
                "bash "
                + shlex.join(
                    [
                        LOCAL_DECAY_BSH,
                        str(baits_local),
                        str(preys_local),
                        str(data_dir / K562_PAIRS),
                        str(outdir),
                        str(legacy_cpu),
                        "1000000",
                        "2000",
                    ]
                ),
                f"cat {shlex.quote(str(outdir))}/chr* > {shlex.quote(str(combined_out))}",
            ]
        )
        steps.append(
            BenchmarkStep(
                name="local-decay-call-legacy",
                group="legacy",
                command=[*shell_prefix, "bash", "-c", inner],
                outputs=[combined_out],
            )
        )

    if "apa" in workflows:
        outdir = output_dir / "apa" / apa_sample.upper()
        inner = " && ".join(
            [
                *env_prefix,
                f"cd {shlex.quote(str(reference_dir))}",
                "bash "
                + shlex.join(
                    [
                        APA_BSH,
                        str(data_dir / SAMPLE_PAIRS[apa_sample]),
                        str(baits_mesc),
                        str(preys_mesc),
                        "25000",
                        "150000",
                        "10000",
                        "50",
                        str(outdir),
                    ]
                ),
            ]
        )
        steps.append(
            BenchmarkStep(
                name=f"apa-aggregate-{apa_sample}-legacy",
                group="legacy",
                command=[*shell_prefix, "bash", "-c", inner],
                outputs=[
                    outdir / "AggMat.csv",
                    outdir / "AggHeatmap.svg",
                    outdir / "baits_genome_wide_contacts.csv",
                    outdir / "preys_genome_wide_contacts.csv",
                ],
            )
        )

    if "background" in workflows:
        outdir = output_dir / "background" / background_sample.upper()
        inner = " && ".join(
            [
                *env_prefix,
                f"cd {shlex.quote(str(reference_dir))}",
                "bash "
                + shlex.join(
                    [
                        BACKGROUND_BSH,
                        str(data_dir / SAMPLE_PAIRS[background_sample]),
                        str(baits_mesc),
                        str(preys_mesc),
                        "25000",
                        "150000",
                        "2500",
                        str(outdir),
                        "10000",
                        "150000",
                    ]
                ),
            ]
        )
        steps.append(
            BenchmarkStep(
                name=f"background-count-{background_sample}-legacy",
                group="legacy",
                command=[*shell_prefix, "bash", "-c", inner],
                outputs=[outdir / "EP_and_BG_contacts.txt"],
            )
        )
    return steps


def interleave(touche_steps: list[BenchmarkStep], legacy_steps: list[BenchmarkStep]) -> list[BenchmarkStep]:
    """Run each family's touche step immediately before its legacy counterpart."""
    legacy_by_prefix = {step.name.removesuffix("-legacy"): step for step in legacy_steps}
    ordered: list[BenchmarkStep] = []
    used_legacy: set[str] = set()
    for touche_step in touche_steps:
        ordered.append(touche_step)
        legacy_step = legacy_by_prefix.get(touche_step.name)
        if legacy_step is not None:
            ordered.append(legacy_step)
            used_legacy.add(legacy_step.name)
    ordered.extend(step for step in legacy_steps if step.name not in used_legacy)
    return ordered


def needed_downloads(
    data_dir: Path, *, workflows: list[str], apa_sample: str, background_sample: str
) -> list[Download]:
    pairs_files: set[str] = set()
    if "local-decay" in workflows:
        pairs_files.add(K562_PAIRS)
    if "apa" in workflows:
        pairs_files.add(SAMPLE_PAIRS[apa_sample])
    if "background" in workflows:
        pairs_files.add(SAMPLE_PAIRS[background_sample])

    downloads: list[Download] = []
    for name in sorted(pairs_files):
        if name == K562_PAIRS:
            downloads.append(Download(name, f"{GEO_BASE}/{name}", data_dir / name))
        else:
            downloads.append(Download(name, f"{CORNELL_FTP_BASE}/{name}", data_dir / name))

    input_files: set[str] = set()
    if "local-decay" in workflows:
        input_files.update([LOCAL_DECAY_BAITS, LOCAL_DECAY_PREYS])
    if "apa" in workflows or "background" in workflows:
        input_files.update([MESC_BAITS, MESC_PREYS])
    for name in sorted(input_files):
        downloads.append(Download(name, f"{REFERENCE_RAW_BASE}/{name}", data_dir / "Input_files" / name))
    return downloads


def print_plan(downloads: list[Download], steps: list[BenchmarkStep]) -> None:
    payload = {
        "downloads": [{"name": item.name, "url": item.url, "path": str(item.path)} for item in downloads],
        "steps": [{"name": step.name, "group": step.group, "command": step.command} for step in steps],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_manifest(
    path: Path, *, args: argparse.Namespace, downloads: list[dict[str, Any]], results: list[BenchmarkResult]
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "argv": sys.argv,
                "reference_dir": str(args.reference_dir),
                "workflows": args.workflows,
                "apa_sample": args.apa_sample,
                "background_sample": args.background_sample,
                "downloads": downloads,
                "results": [asdict(result) for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_reports(
    results: list[BenchmarkResult],
    *,
    cache_steps: list[BenchmarkStep],
    report_dir: Path,
) -> None:
    records = [result_to_record(result) for result in results]
    write_profile_report(records, report_dir=report_dir)
    write_speedup_report(records, cache_step_names={step.name for step in cache_steps}, report_dir=report_dir)


def write_speedup_report(
    records: list[dict[str, Any]], *, cache_step_names: set[str], report_dir: Path
) -> None:
    by_name = {record["name"]: record for record in records}
    cache_seconds = sum(
        float(by_name[name]["elapsed_seconds"]) for name in cache_step_names if name in by_name
    )

    rows: list[dict[str, Any]] = []
    for legacy_name, legacy_record in sorted(by_name.items()):
        if not legacy_name.endswith("-legacy") or legacy_record["returncode"] != 0:
            continue
        touche_name = legacy_name.removesuffix("-legacy")
        touche_record = by_name.get(touche_name)
        if touche_record is None or touche_record["returncode"] != 0:
            continue
        legacy_seconds = float(legacy_record["elapsed_seconds"])
        touche_seconds = float(touche_record["elapsed_seconds"])
        touche_cold_seconds = touche_seconds + (cache_seconds if touche_name == "local-decay-call" else 0.0)
        rows.append(
            {
                "family": touche_name,
                "touche_seconds": round(touche_seconds, 3),
                "touche_cold_seconds": round(touche_cold_seconds, 3),
                "legacy_seconds": round(legacy_seconds, 3),
                "speedup_warm": round(legacy_seconds / touche_seconds, 2) if touche_seconds else None,
                "speedup_cold": round(legacy_seconds / touche_cold_seconds, 2) if touche_cold_seconds else None,
            }
        )

    if not rows:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "speedup.csv", rows)

    lines = [
        "# touche vs. legacy speedup",
        "",
        "`touche_cold_seconds` adds one-time NPZ cache construction where touche uses a "
        "persistent cache the legacy pipeline has no equivalent to (local-decay only); "
        "`touche_seconds` is the cache-reuse ('warm') cost.",
        "",
        "| Workflow | touche (warm) s | touche (cold) s | legacy s | Speedup (warm) | Speedup (cold) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {touche_seconds} | {touche_cold_seconds} | {legacy_seconds} | "
            "{speedup_warm}x | {speedup_cold}x |".format(**row)
        )
    (report_dir / "speedup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    chart_rows = []
    for row in rows:
        chart_rows.append({"name": f"{row['family']} (touche)", "group": "touche", "seconds": row["touche_seconds"]})
        chart_rows.append({"name": f"{row['family']} (legacy)", "group": "legacy", "seconds": row["legacy_seconds"]})
    plot_metric(
        chart_rows,
        metric="seconds",
        title="touche vs. legacy wall time (warm cache)",
        xlabel="seconds",
        out=report_dir / "speedup.svg",
    )


if __name__ == "__main__":
    raise SystemExit(main())
