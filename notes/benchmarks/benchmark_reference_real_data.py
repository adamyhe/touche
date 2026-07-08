from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REFERENCE_RAW_BASE = "https://raw.githubusercontent.com/Danko-Lab/E-P_contacts/main/Input_files"
GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206131/suppl"
CORNELL_FTP_BASE = "ftp://cbsuftp.tc.cornell.edu/danko/hub/MicroC_pairs_files"

K562_PAIRS = "GSE206131_K562_cis_mapq30_pairs.txt.gz"
DMSO_PAIRS = "mESCs_DMSO_30_intra.mm10.nodups.pairs.gz"
FLV_PAIRS = "mESCs_FLV_30_intra.mm10.nodups.pairs.gz"
TRP_PAIRS = "mESCs_TRP_30_intra.mm10.nodups.pairs.gz"

LOCAL_DECAY_BAITS = "Gasperini_dREG_based_TRE_baits_hg38.txt"
LOCAL_DECAY_PREYS = "Gasperini_dREG_based_promoter_preys_hg38.txt"
LOCAL_DECAY_FUNCTIONAL = "Gasperini_dREG_based_functional.csv"
LOCAL_DECAY_NONFUNCTIONAL = "Gasperini_dREG_based_nonfunctional.csv"
MESC_BAITS = "dREG_based_promoters_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed"
MESC_PREYS = "dREG_based_TREs_with_STARTseq_based_maxTSS_mm10_200bp_centered_on_maxTSS_chr_start_end_strand.bed"


@dataclass(frozen=True, slots=True)
class Download:
    name: str
    url: str
    path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkStep:
    name: str
    group: str
    command: list[str]
    outputs: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    group: str
    command: list[str]
    returncode: int
    elapsed_seconds: float
    peak_rss_mb: float | None
    stdout_log: str
    stderr_log: str
    outputs: dict[str, int | None]
    command_json: Any | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the real E-P_contacts example inputs and profile touche "
            "preprocessing, local-decay, APA, and background steps."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark/reference-real-data"))
    parser.add_argument("--python", default=sys.executable, help="Python executable for -m touche")
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    parser.add_argument("--lowess-backend", choices=["statsmodels", "numba"], default="statsmodels")
    parser.add_argument("--lowess-iterations", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Pass --progress to touche commands")
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Optional step-name subset. Use --dry-run to list all planned step names.",
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

    if args.plot_only:
        result_dicts = read_results_jsonl(results_jsonl)
        if not args.no_report:
            write_profile_report(result_dicts, report_dir=report_dir)
        return 0

    downloads = reference_downloads(data_dir)
    steps = reference_steps(
        python=args.python,
        data_dir=data_dir,
        output_dir=output_dir,
        backend=args.backend,
        lowess_backend=args.lowess_backend,
        lowess_iterations=args.lowess_iterations,
        progress=args.progress,
    )
    if args.steps:
        requested = set(args.steps)
        known = {step.name for step in steps}
        unknown = sorted(requested - known)
        if unknown:
            raise SystemExit(f"Unknown step(s): {', '.join(unknown)}")
        steps = [step for step in steps if step.name in requested]

    if args.dry_run:
        print_plan(downloads, steps)
        return 0

    download_records: list[dict[str, Any]] = []
    if not args.skip_download:
        download_records = download_reference_inputs(downloads)
    if args.download_only:
        write_manifest(
            manifest_json,
            args=args,
            downloads=download_records,
            results=[],
            results_jsonl=results_jsonl,
        )
        return 0

    results: list[BenchmarkResult] = []
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for step in steps:
            result = run_profiled_step(
                step,
                logs_dir=logs_dir,
                poll_interval=args.poll_interval,
            )
            results.append(result)
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            handle.flush()
            if result.returncode != 0 and not args.keep_going:
                write_manifest(
                    manifest_json,
                    args=args,
                    downloads=download_records,
                    results=results,
                    results_jsonl=results_jsonl,
                )
                if not args.no_report:
                    write_profile_report(
                        [result_to_record(item) for item in results],
                        report_dir=report_dir,
                    )
                return result.returncode

    write_manifest(
        manifest_json,
        args=args,
        downloads=download_records,
        results=results,
        results_jsonl=results_jsonl,
    )
    if not args.no_report:
        write_profile_report([result_to_record(item) for item in results], report_dir=report_dir)
    return 0


def reference_downloads(data_dir: Path) -> list[Download]:
    return [
        Download(K562_PAIRS, f"{GEO_BASE}/{K562_PAIRS}", data_dir / K562_PAIRS),
        Download(DMSO_PAIRS, f"{CORNELL_FTP_BASE}/{DMSO_PAIRS}", data_dir / DMSO_PAIRS),
        Download(FLV_PAIRS, f"{CORNELL_FTP_BASE}/{FLV_PAIRS}", data_dir / FLV_PAIRS),
        Download(TRP_PAIRS, f"{CORNELL_FTP_BASE}/{TRP_PAIRS}", data_dir / TRP_PAIRS),
        *[
            Download(name, f"{REFERENCE_RAW_BASE}/{name}", data_dir / "Input_files" / name)
            for name in [
                LOCAL_DECAY_BAITS,
                LOCAL_DECAY_PREYS,
                LOCAL_DECAY_FUNCTIONAL,
                LOCAL_DECAY_NONFUNCTIONAL,
                MESC_BAITS,
                MESC_PREYS,
            ]
        ],
    ]


def reference_steps(
    *,
    python: str,
    data_dir: Path,
    output_dir: Path,
    backend: str,
    lowess_backend: str,
    lowess_iterations: int,
    progress: bool,
) -> list[BenchmarkStep]:
    input_dir = data_dir / "Input_files"
    k562_pairs = data_dir / K562_PAIRS
    dmso_pairs = data_dir / DMSO_PAIRS
    flv_pairs = data_dir / FLV_PAIRS
    trp_pairs = data_dir / TRP_PAIRS
    baits_local = input_dir / LOCAL_DECAY_BAITS
    preys_local = input_dir / LOCAL_DECAY_PREYS
    functional = input_dir / LOCAL_DECAY_FUNCTIONAL
    nonfunctional = input_dir / LOCAL_DECAY_NONFUNCTIONAL
    baits_mesc = input_dir / MESC_BAITS
    preys_mesc = input_dir / MESC_PREYS
    common_profile = ["--profile", *(("--progress",) if progress else ())]

    local_dir = output_dir / "local-decay"
    apa_dir = output_dir / "apa"
    background_dir = output_dir / "background"
    cache_dir = output_dir / "caches"

    steps: list[BenchmarkStep] = []
    for label, pairs in [
        ("k562", k562_pairs),
        ("dmso", dmso_pairs),
        ("flv", flv_pairs),
        ("trp", trp_pairs),
    ]:
        qc_out = output_dir / "preprocess" / f"{label}.qc.json"
        cache_out = cache_dir / label
        steps.append(
            BenchmarkStep(
                name=f"preprocess-qc-{label}",
                group="preprocess",
                command=touche_cmd(python, "preprocess", "qc", "--pairs", pairs, "--source", "auto", "--out", qc_out),
                outputs=[qc_out],
            )
        )
        steps.append(
            BenchmarkStep(
                name=f"preprocess-cache-{label}",
                group="preprocess",
                command=touche_cmd(
                    python,
                    "preprocess",
                    "build-cache",
                    "--pairs",
                    pairs,
                    "--source",
                    "auto",
                    "--cache-dir",
                    cache_out,
                    "--prefix",
                    label,
                ),
                outputs=[cache_out],
            )
        )

    local_calls = local_dir / "ContactCaller_microC_output.tsv"
    local_assignments = (
        local_dir / "ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv"
    )
    local_plot = local_dir / "Violinplot_for_normalized_contacts_by_pair_type.svg"
    local_plot_table = local_dir / "Violinplot_for_normalized_contacts_by_pair_type.tsv"
    steps.extend(
        [
            BenchmarkStep(
                name="local-decay-call",
                group="local-decay",
                command=touche_cmd(
                    python,
                    "local-decay",
                    "call",
                    "--baits",
                    baits_local,
                    "--preys",
                    preys_local,
                    "--pairs",
                    k562_pairs,
                    "--out",
                    local_calls,
                    "--dist",
                    "1000000",
                    "--cap",
                    "2000",
                    "--backend",
                    backend,
                    "--lowess-backend",
                    lowess_backend,
                    "--lowess-iterations",
                    str(lowess_iterations),
                    *common_profile,
                ),
                outputs=[local_calls],
            ),
            BenchmarkStep(
                name="local-decay-assign-pair-types",
                group="local-decay",
                command=touche_cmd(
                    python,
                    "local-decay",
                    "assign-pair-types",
                    "--contacts",
                    local_calls,
                    "--functional",
                    functional,
                    "--nonfunctional",
                    nonfunctional,
                    "--out",
                    local_assignments,
                ),
                outputs=[local_assignments],
            ),
            BenchmarkStep(
                name="local-decay-plot",
                group="local-decay",
                command=touche_cmd(
                    python,
                    "local-decay",
                    "plot",
                    "--assignments",
                    local_assignments,
                    "--min-contacts",
                    "1",
                    "--min-distance",
                    "15000",
                    "--plot-table-out",
                    local_plot_table,
                    "--out",
                    local_plot,
                ),
                outputs=[local_plot_table, local_plot],
            ),
        ]
    )

    apa_outputs = {
        "dmso": apa_dir / "DMSO",
        "flv": apa_dir / "FLV",
        "trp": apa_dir / "TRP",
    }
    for label, pairs in [("dmso", dmso_pairs), ("flv", flv_pairs), ("trp", trp_pairs)]:
        sample_dir = apa_outputs[label]
        steps.append(
            BenchmarkStep(
                name=f"apa-aggregate-{label}",
                group="apa",
                command=touche_cmd(
                    python,
                    "apa",
                    "aggregate",
                    "--pairs",
                    pairs,
                    "--baits",
                    baits_mesc,
                    "--preys",
                    preys_mesc,
                    "--min-distance",
                    "25000",
                    "--max-distance",
                    "150000",
                    "--window",
                    "10000",
                    "--pixels",
                    "50",
                    "--out-dir",
                    sample_dir,
                    "--backend",
                    backend,
                    *common_profile,
                ),
                outputs=[
                    sample_dir / "AggMat.csv",
                    sample_dir / "AggHeatmap.svg",
                    sample_dir / "baits_genome_wide_contacts.csv",
                    sample_dir / "preys_genome_wide_contacts.csv",
                ],
            )
        )

    for label in ["flv", "trp"]:
        compare_dir = apa_dir / f"{label.upper()}_vs_DMSO"
        steps.append(
            BenchmarkStep(
                name=f"apa-compare-{label}-vs-dmso",
                group="apa",
                command=touche_cmd(
                    python,
                    "apa",
                    "compare",
                    "--control-apa",
                    apa_outputs["dmso"] / "AggMat.csv",
                    "--treatment-apa",
                    apa_outputs[label] / "AggMat.csv",
                    "--control-baits",
                    apa_outputs["dmso"] / "baits_genome_wide_contacts.csv",
                    "--control-preys",
                    apa_outputs["dmso"] / "preys_genome_wide_contacts.csv",
                    "--treatment-baits",
                    apa_outputs[label] / "baits_genome_wide_contacts.csv",
                    "--treatment-preys",
                    apa_outputs[label] / "preys_genome_wide_contacts.csv",
                    "--bait-count",
                    "10530",
                    "--prey-count",
                    "27900",
                    "--window",
                    "10000",
                    "--pixels",
                    "50",
                    "--matrix-out",
                    compare_dir / "ObsOverExp.csv",
                    "--out",
                    compare_dir / "ObsOverExp.svg",
                ),
                outputs=[compare_dir / "ObsOverExp.csv", compare_dir / "ObsOverExp.svg"],
            )
        )

    background_counts = {
        "dmso": background_dir / "counts" / "DMSO_EP_and_BG_contacts.tsv",
        "flv": background_dir / "counts" / "FLV_EP_and_BG_contacts.tsv",
        "trp": background_dir / "counts" / "TRP_EP_and_BG_contacts.tsv",
    }
    for label, pairs in [("dmso", dmso_pairs), ("flv", flv_pairs), ("trp", trp_pairs)]:
        steps.append(
            BenchmarkStep(
                name=f"background-count-{label}",
                group="background",
                command=touche_cmd(
                    python,
                    "background",
                    "count",
                    "--pairs",
                    pairs,
                    "--baits",
                    baits_mesc,
                    "--preys",
                    preys_mesc,
                    "--min-distance",
                    "25000",
                    "--max-distance",
                    "150000",
                    "--window",
                    "2500",
                    "--min-bg-distance",
                    "10000",
                    "--max-bg-distance",
                    "150000",
                    "--out",
                    background_counts[label],
                    "--backend",
                    backend,
                    *common_profile,
                ),
                outputs=[background_counts[label]],
            )
        )
    steps.append(
        BenchmarkStep(
            name="background-compare",
            group="background",
            command=touche_cmd(
                python,
                "background",
                "compare",
                "--control",
                f"DMSO={background_counts['dmso']}",
                "--treatments",
                f"FLV={background_counts['flv']}",
                f"TRP={background_counts['trp']}",
                "--depths",
                "DMSO=53226768",
                "FLV=362862200",
                "TRP=410040533",
                "--min-ep-cpb",
                "8",
                "--out-dir",
                background_dir / "plots",
                "--table-out",
                background_dir / "background_comparison.tsv",
            ),
            outputs=[background_dir / "background_comparison.tsv", background_dir / "plots"],
        )
    )
    return steps


def touche_cmd(python: str, *args: object) -> list[str]:
    return [python, "-m", "touche", *[str(arg) for arg in args]]


def download_reference_inputs(downloads: list[Download]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in downloads:
        item.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if not item.path.exists():
            download_file(item.url, item.path)
        elapsed = time.perf_counter() - started
        records.append(
            {
                "name": item.name,
                "url": item.url,
                "path": str(item.path),
                "bytes": item.path.stat().st_size,
                "sha256": sha256_file(item.path),
                "elapsed_seconds": round(elapsed, 6),
                "already_present": elapsed < 0.001,
            }
        )
    return records


def download_file(url: str, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    tmp_path.replace(path)


def run_profiled_step(
    step: BenchmarkStep,
    *,
    logs_dir: Path,
    poll_interval: float,
) -> BenchmarkResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / f"{step.name}.stdout"
    stderr_log = logs_dir / f"{step.name}.stderr"
    for output in step.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    peak_rss_kb: int | None = None
    with stdout_log.open("w", encoding="utf-8") as stdout_handle:
        with stderr_log.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                step.command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            while process.poll() is None:
                rss_kb = process_tree_rss_kb(process.pid)
                if rss_kb is not None:
                    peak_rss_kb = max(peak_rss_kb or 0, rss_kb)
                time.sleep(poll_interval)
            final_rss_kb = process_tree_rss_kb(process.pid)
            if final_rss_kb is not None:
                peak_rss_kb = max(peak_rss_kb or 0, final_rss_kb)
            returncode = process.returncode

    elapsed = time.perf_counter() - started
    return BenchmarkResult(
        name=step.name,
        group=step.group,
        command=step.command,
        returncode=int(returncode),
        elapsed_seconds=round(elapsed, 6),
        peak_rss_mb=round(peak_rss_kb / 1024, 3) if peak_rss_kb is not None else None,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        outputs={str(output): path_size(output) for output in step.outputs},
        command_json=read_stdout_json(stdout_log),
    )


def process_tree_rss_kb(pid: int) -> int | None:
    pids = [pid, *child_pids(pid)]
    values: list[int] = []
    for current_pid in pids:
        value = process_rss_kb(current_pid)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values)


def child_pids(pid: int) -> list[int]:
    try:
        completed = subprocess.run(
            ["pgrep", "-P", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    children = [int(value) for value in completed.stdout.split() if value.isdigit()]
    descendants: list[int] = []
    for child in children:
        descendants.extend(child_pids(child))
    return [*children, *descendants]


def process_rss_kb(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        return int(value.splitlines()[-1].strip())
    except ValueError:
        return None


def path_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def read_stdout_json(path: Path) -> Any | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_plan(downloads: list[Download], steps: list[BenchmarkStep]) -> None:
    payload = {
        "downloads": [asdict(item) | {"path": str(item.path)} for item in downloads],
        "steps": [
            {
                "name": step.name,
                "group": step.group,
                "command": step.command,
                "outputs": [str(output) for output in step.outputs],
            }
            for step in steps
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    downloads: list[dict[str, Any]],
    results: list[BenchmarkResult],
    results_jsonl: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "argv": sys.argv,
                "work_dir": str(args.work_dir),
                "backend": args.backend,
                "lowess_backend": args.lowess_backend,
                "lowess_iterations": args.lowess_iterations,
                "downloads": downloads,
                "results_jsonl": str(results_jsonl),
                "results": [asdict(result) for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_results_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Benchmark results not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSONL record in {path} line {line_number}") from exc
    return records


def result_to_record(result: BenchmarkResult) -> dict[str, Any]:
    return asdict(result)


def write_profile_report(records: list[dict[str, Any]], *, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = report_dir / ".matplotlib-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir = report_dir / ".cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
    summary_rows = [summary_row(record) for record in records]
    timing_rows = profile_timing_rows(records)

    write_csv(report_dir / "summary.csv", summary_rows)
    if timing_rows:
        write_csv(report_dir / "command-timings.csv", timing_rows)
    write_markdown_summary(report_dir / "summary.md", summary_rows, timing_rows)
    write_html_index(report_dir / "index.html", timing_rows=bool(timing_rows))

    plot_metric(
        summary_rows,
        metric="elapsed_seconds",
        title="Wall Time by Benchmark Step",
        xlabel="seconds",
        out=report_dir / "wall-time.svg",
    )
    plot_metric(
        [row for row in summary_rows if row["peak_rss_mb"] != ""],
        metric="peak_rss_mb",
        title="Peak RSS by Benchmark Step",
        xlabel="MiB",
        out=report_dir / "peak-rss.svg",
    )
    plot_metric(
        [row for row in summary_rows if row["output_mb"] != ""],
        metric="output_mb",
        title="Output Size by Benchmark Step",
        xlabel="MiB",
        out=report_dir / "output-size.svg",
    )
    if timing_rows:
        plot_profile_timings(timing_rows, out=report_dir / "command-timings.svg")


def summary_row(record: dict[str, Any]) -> dict[str, Any]:
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        outputs = {}
    output_bytes = sum(size for size in outputs.values() if isinstance(size, int))
    missing_outputs = [path for path, size in outputs.items() if size is None]
    zero_byte_outputs = [path for path, size in outputs.items() if size == 0]
    command_json = record.get("command_json") if isinstance(record.get("command_json"), dict) else {}
    rows = command_json.get("rows")
    return {
        "name": record.get("name", ""),
        "group": record.get("group", ""),
        "returncode": record.get("returncode", ""),
        "elapsed_seconds": record.get("elapsed_seconds", ""),
        "peak_rss_mb": record.get("peak_rss_mb") or "",
        "output_mb": round(output_bytes / (1024 * 1024), 3),
        "output_count": len(outputs),
        "missing_outputs": len(missing_outputs),
        "zero_byte_outputs": len(zero_byte_outputs),
        "rows": rows if rows is not None else "",
        "stdout_log": record.get("stdout_log", ""),
        "stderr_log": record.get("stderr_log", ""),
    }


def profile_timing_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        command_json = record.get("command_json")
        if not isinstance(command_json, dict):
            continue
        timings = command_json.get("timings")
        if not isinstance(timings, list):
            continue
        for timing in timings:
            if not isinstance(timing, dict):
                continue
            rows.append(
                {
                    "step": record.get("name", ""),
                    "group": record.get("group", ""),
                    "timing_step": timing.get("step", ""),
                    "elapsed_seconds": timing.get("elapsed_seconds", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    path: Path, summary_rows: list[dict[str, Any]], timing_rows: list[dict[str, Any]]
) -> None:
    lines = [
        "# Reference benchmark profile",
        "",
        "## Overview",
        "",
        "| Step | Group | Status | Wall time (s) | Peak RSS (MiB) | Output (MiB) | Outputs | Missing | Empty | Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {group} | {returncode} | {elapsed_seconds} | {peak_rss_mb} | {output_mb} | {output_count} | {missing_outputs} | {zero_byte_outputs} | {rows} |".format(
                **row
            )
        )

    if timing_rows:
        lines.extend(
            [
                "",
                "## CLI Profile Timings",
                "",
                "| Step | Group | Internal step | Wall time (s) |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for row in timing_rows:
            lines.append(
                "| {step} | {group} | {timing_step} | {elapsed_seconds} |".format(**row)
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- [Wall time](wall-time.svg)",
            "- [Peak RSS](peak-rss.svg)",
            "- [Output size](output-size.svg)",
        ]
    )
    if timing_rows:
        lines.append("- [CLI profile timings](command-timings.svg)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_index(path: Path, *, timing_rows: bool) -> None:
    timing_image = (
        '<h2>CLI Profile Timings</h2><img src="command-timings.svg" alt="CLI profile timings">'
        if timing_rows
        else ""
    )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>touche reference benchmark profile</title>",
                "<style>",
                "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}",
                "img{display:block;max-width:100%;margin:1rem 0 2rem}",
                "a{color:#075985}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>touche reference benchmark profile</h1>",
                '<p><a href="summary.md">Markdown summary</a> | <a href="summary.csv">CSV summary</a></p>',
                '<h2>Wall Time</h2><img src="wall-time.svg" alt="Wall time by benchmark step">',
                '<h2>Peak RSS</h2><img src="peak-rss.svg" alt="Peak RSS by benchmark step">',
                '<h2>Output Size</h2><img src="output-size.svg" alt="Output size by benchmark step">',
                timing_image,
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def plot_metric(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    xlabel: str,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = [str(row["name"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    colors = [group_color(str(row["group"])) for row in rows]
    height = max(4.0, 0.36 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(11, height))
    positions = range(len(rows))
    ax.barh(list(positions), values, color=colors)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_profile_timings(rows: list[dict[str, Any]], *, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['step']}: {row['timing_step']}" for row in rows]
    values = [float(row["elapsed_seconds"]) for row in rows]
    colors = [group_color(str(row["group"])) for row in rows]
    height = max(4.0, 0.32 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(12, height))
    positions = range(len(rows))
    ax.barh(list(positions), values, color=colors)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Nested CLI --profile Timings")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def group_color(group: str) -> str:
    return {
        "preprocess": "#3b82f6",
        "local-decay": "#ef4444",
        "apa": "#22c55e",
        "background": "#a855f7",
    }.get(group, "#64748b")


if __name__ == "__main__":
    raise SystemExit(main())
