from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _report import (
    BenchmarkResult,
    BenchmarkStep,
    result_to_record,
    run_profiled_step,
    write_profile_report,
)

from touche.backends import has_numba

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


class Download:
    __slots__ = ("name", "url", "path")

    def __init__(self, name: str, url: str, path: Path) -> None:
        self.name = name
        self.url = url
        self.path = path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the real E-P_contacts example inputs and profile touche "
            "preprocessing, local-decay, APA, and background steps."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark/reference-real-data"))
    parser.add_argument("--python", default=sys.executable, help="Python executable for -m touche")
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "both"],
        default="both",
        help=(
            "Which backend(s) to benchmark. 'both' (default) runs numpy and numba side by "
            "side and adds a speedup report; falls back to numpy-only if numba isn't installed."
        ),
    )
    parser.add_argument("--lowess-backend", choices=["statsmodels", "numba"], default="statsmodels")
    parser.add_argument("--lowess-iterations", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--fail-on-missing-output",
        action="store_true",
        help="Return a non-zero exit code when a successful step misses expected outputs.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Pass --progress to profiled touche commands and stream their stderr "
            "progress bars live while still writing log files."
        ),
    )
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

    backends = resolve_backends(args.backend)

    if args.plot_only:
        result_dicts = read_results_jsonl(results_jsonl)
        if not args.no_report:
            speedup_pairs = speedup_pairs_from_records(result_dicts, backends)
            plot_gallery = plot_gallery_from_records(result_dicts)
            write_profile_report(
                result_dicts,
                report_dir=report_dir,
                speedup_pairs=speedup_pairs,
                plot_gallery=plot_gallery,
            )
        return 0

    downloads = reference_downloads(data_dir)
    cache_steps, cache_paths = build_cache_steps(python=args.python, data_dir=data_dir, output_dir=output_dir)
    backend_step_lists: dict[str, list[BenchmarkStep]] = {}
    for backend in backends:
        backend_step_lists[backend] = build_backend_steps(
            python=args.python,
            data_dir=data_dir,
            output_dir=output_dir / backend if len(backends) > 1 else output_dir,
            backend=backend,
            k562_cache_dir=cache_paths["k562"],
            lowess_backend=args.lowess_backend,
            lowess_iterations=args.lowess_iterations,
            progress=args.progress,
            name_suffix=f"-{backend}" if len(backends) > 1 else "",
        )
    steps = list(cache_steps)
    for backend in backends:
        steps.extend(backend_step_lists[backend])

    if args.steps:
        requested = set(args.steps)
        known = {step.name for step in steps}
        unknown = sorted(requested - known)
        if unknown:
            raise SystemExit(f"Unknown step(s): {', '.join(unknown)}")
        steps = [step for step in steps if step.name in requested]

    if args.dry_run:
        print_plan(downloads, steps, backends)
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
    validate_benchmark_cache_requirements(steps, cache_paths)

    results: list[BenchmarkResult] = []
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for index, step in enumerate(steps, start=1):
            if args.progress:
                print(
                    f"[{index}/{len(steps)}] running {step.name}",
                    file=sys.stderr,
                    flush=True,
                )
            result = run_profiled_step(
                step,
                logs_dir=logs_dir,
                poll_interval=args.poll_interval,
                live_stderr=args.progress,
            )
            if args.progress:
                print(
                    f"[{index}/{len(steps)}] finished {step.name} "
                    f"returncode={result.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
            results.append(result)
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            handle.flush()
            missing_outputs = missing_output_paths_local(result_to_record(result))
            missing_output_failure = (
                args.fail_on_missing_output and result.returncode == 0 and missing_outputs
            )
            if missing_output_failure:
                print(
                    f"{step.name} completed but missed expected outputs: "
                    f"{', '.join(missing_outputs)}",
                    file=sys.stderr,
                )
            if (result.returncode != 0 or missing_output_failure) and not args.keep_going:
                write_manifest(
                    manifest_json,
                    args=args,
                    downloads=download_records,
                    results=results,
                    results_jsonl=results_jsonl,
                )
                if not args.no_report:
                    write_final_report(results, backends, report_dir=report_dir, no_report=False)
                return result.returncode if result.returncode != 0 else 2

    write_manifest(
        manifest_json,
        args=args,
        downloads=download_records,
        results=results,
        results_jsonl=results_jsonl,
    )
    write_final_report(results, backends, report_dir=report_dir, no_report=args.no_report)
    return 0


def resolve_backends(requested: str) -> list[str]:
    if requested in {"numpy", "numba"}:
        return [requested]
    if not has_numba():
        print(
            "numba is not installed; falling back to numpy-only "
            "(install with `uv sync --extra fast` to get a numpy vs. numba comparison)",
            file=sys.stderr,
        )
        return ["numpy"]
    return ["numpy", "numba"]


def write_final_report(
    results: list[BenchmarkResult], backends: list[str], *, report_dir: Path, no_report: bool
) -> None:
    if no_report:
        return
    records = [result_to_record(item) for item in results]
    speedup_pairs = speedup_pairs_from_records(records, backends)
    plot_gallery = plot_gallery_from_records(records)
    write_profile_report(
        records, report_dir=report_dir, speedup_pairs=speedup_pairs, plot_gallery=plot_gallery
    )


def speedup_pairs_from_records(
    records: list[dict[str, Any]], backends: list[str]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if "numpy" not in backends or "numba" not in backends:
        return []
    by_name = {record.get("name"): record for record in records}
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name, record in by_name.items():
        if not isinstance(name, str) or not name.endswith("-numpy"):
            continue
        base = name[: -len("-numpy")]
        numba_record = by_name.get(f"{base}-numba")
        if numba_record is not None:
            pairs.append((base, record, numba_record))
    return pairs


def plot_gallery_from_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    gallery: list[dict[str, str]] = []
    for record in records:
        outputs = record.get("outputs") or {}
        if not isinstance(outputs, dict):
            continue
        for output_path, size in outputs.items():
            if isinstance(size, int) and size > 0 and output_path.endswith(".svg"):
                gallery.append(
                    {
                        "title": f"{record.get('name', '')}: {Path(output_path).name}",
                        "group": str(record.get("group", "")),
                        "src_svg": output_path,
                    }
                )
    return gallery


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


def build_cache_steps(
    *, python: str, data_dir: Path, output_dir: Path
) -> tuple[list[BenchmarkStep], dict[str, Path]]:
    """Steps that don't depend on backend: build NPZ caches shared by every backend run."""

    cache_dir = output_dir / "caches"
    steps: list[BenchmarkStep] = []
    cache_paths: dict[str, Path] = {}
    for label, pairs in [
        ("k562", data_dir / K562_PAIRS),
        ("dmso", data_dir / DMSO_PAIRS),
        ("flv", data_dir / FLV_PAIRS),
        ("trp", data_dir / TRP_PAIRS),
    ]:
        cache_out = cache_dir / label
        cache_paths[label] = cache_out
        qc_out = cache_out / f"{label}.qc.json"
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
                    *(("--no-metadata",) if label == "k562" else ()),
                ),
                outputs=[cache_out, qc_out],
            )
        )
    return steps, cache_paths


def build_backend_steps(
    *,
    python: str,
    data_dir: Path,
    output_dir: Path,
    backend: str,
    k562_cache_dir: Path,
    lowess_backend: str,
    lowess_iterations: int,
    progress: bool,
    name_suffix: str,
) -> list[BenchmarkStep]:
    input_dir = data_dir / "Input_files"
    dmso_pairs = data_dir / DMSO_PAIRS
    flv_pairs = data_dir / FLV_PAIRS
    trp_pairs = data_dir / TRP_PAIRS
    k562_pairs = data_dir / K562_PAIRS
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

    steps: list[BenchmarkStep] = []

    local_calls = local_dir / "ContactCaller_microC_output.tsv"
    local_assignments = (
        local_dir / "ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv"
    )
    local_plot = local_dir / "Violinplot_for_normalized_contacts_by_pair_type.svg"
    local_plot_table = local_dir / "Violinplot_for_normalized_contacts_by_pair_type.tsv"
    steps.extend(
        [
            BenchmarkStep(
                name=f"local-decay-call{name_suffix}",
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
                    "--index-strategy",
                    "cache",
                    "--cache-dir",
                    k562_cache_dir,
                    "--cache-prefix",
                    "k562",
                    "--require-cache",
                    "--lowess-backend",
                    lowess_backend,
                    "--lowess-iterations",
                    str(lowess_iterations),
                    *common_profile,
                ),
                outputs=[local_calls],
            ),
            BenchmarkStep(
                name=f"local-decay-assign-pair-types{name_suffix}",
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
                name=f"local-decay-plot{name_suffix}",
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
                name=f"apa-aggregate-{label}{name_suffix}",
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
                name=f"apa-compare-{label}-vs-dmso{name_suffix}",
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
                name=f"background-count-{label}{name_suffix}",
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
            name=f"background-compare{name_suffix}",
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


def validate_benchmark_cache_requirements(steps: list[BenchmarkStep], cache_paths: dict[str, Path]) -> None:
    step_names = {step.name for step in steps}
    if not any(name.startswith("local-decay-call") for name in step_names):
        return
    if any(name == "preprocess-cache-k562" for name in step_names):
        return
    manifest_path = cache_paths["k562"] / "k562.manifest.json"
    if manifest_path.exists():
        return
    raise SystemExit(
        "local-decay-call benchmarks require the K562 NPZ cache. "
        "Run with `--steps preprocess-cache-k562 local-decay-call ...` first, "
        "or run preprocess-cache-k562 in an earlier benchmark invocation."
    )


def touche_cmd(python: str, *args: object) -> list[str]:
    return [python, "-m", "touche", *[str(arg) for arg in args]]


def download_reference_inputs(downloads: list[Download]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in downloads:
        item.path.parent.mkdir(parents=True, exist_ok=True)
        started_present = item.path.exists()
        started = time.perf_counter()
        if not started_present:
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
                "already_present": started_present,
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


def missing_output_paths_local(record: dict[str, Any]) -> list[str]:
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        return []
    return [path for path, size in outputs.items() if size is None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_plan(downloads: list[Download], steps: list[BenchmarkStep], backends: list[str]) -> None:
    payload = {
        "backends": backends,
        "downloads": [
            {"name": item.name, "url": item.url, "path": str(item.path)} for item in downloads
        ],
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
                "schema_version": 2,
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


if __name__ == "__main__":
    raise SystemExit(main())
