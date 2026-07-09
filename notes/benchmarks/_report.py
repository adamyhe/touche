"""Shared subprocess-profiling and report-writing helpers for the benchmark scripts.

Both `benchmark_reference_real_data.py` and `benchmark_numba_kernels.py` run a
list of steps as subprocesses, measure wall time / peak RSS / CPU time for
each, and write the same CSV/Markdown/HTML/plot report shape. This module is
the single source of truth for that plumbing so the two scripts stay
consistent as they evolve.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    signal_name: str | None
    elapsed_seconds: float
    peak_rss_mb: float | None
    cpu_seconds: float | None
    cpu_percent: float | None
    stdout_log: str
    stderr_log: str
    outputs: dict[str, int | None]
    command_json: Any | None = None


def run_profiled_step(
    step: BenchmarkStep,
    *,
    logs_dir: Path,
    poll_interval: float,
    live_stderr: bool = False,
) -> BenchmarkResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / f"{step.name}.stdout"
    stderr_log = logs_dir / f"{step.name}.stderr"
    for output in step.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    # RUSAGE_CHILDREN accumulates as children are reaped; a before/after delta
    # isolates this step's CPU time. The RSS-polling helper subprocesses
    # (`ps`/`pgrep`, spawned below) are also reaped in between and add a small
    # amount of noise to this delta -- negligible next to real workloads.
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_kb: int | None = None
    with stdout_log.open("w", encoding="utf-8") as stdout_handle:
        with stderr_log.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                step.command,
                stdout=stdout_handle,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stderr_thread = threading.Thread(
                target=tee_stream,
                args=(process.stderr, stderr_handle),
                kwargs={"live": live_stderr},
                daemon=True,
            )
            stderr_thread.start()
            while process.poll() is None:
                rss_kb = process_tree_rss_kb(process.pid)
                if rss_kb is not None:
                    peak_rss_kb = max(peak_rss_kb or 0, rss_kb)
                time.sleep(poll_interval)
            stderr_thread.join()
            final_rss_kb = process_tree_rss_kb(process.pid)
            if final_rss_kb is not None:
                peak_rss_kb = max(peak_rss_kb or 0, final_rss_kb)
            returncode = process.returncode

    elapsed = time.perf_counter() - started
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (cpu_after.ru_utime + cpu_after.ru_stime) - (
        cpu_before.ru_utime + cpu_before.ru_stime
    )
    cpu_seconds = max(0.0, cpu_seconds)
    cpu_percent = 100.0 * cpu_seconds / elapsed if elapsed > 0 else None

    return BenchmarkResult(
        name=step.name,
        group=step.group,
        command=step.command,
        returncode=int(returncode),
        signal_name=return_signal_name(returncode),
        elapsed_seconds=round(elapsed, 6),
        peak_rss_mb=round(peak_rss_kb / 1024, 3) if peak_rss_kb is not None else None,
        cpu_seconds=round(cpu_seconds, 6),
        cpu_percent=round(cpu_percent, 1) if cpu_percent is not None else None,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        outputs={str(output): path_size(output) for output in step.outputs},
        command_json=read_stdout_json(stdout_log),
    )


def tee_stream(stream, log_handle, *, live: bool) -> None:
    if stream is None:
        return
    while True:
        chunk = stream.read(1)
        if chunk == "":
            break
        log_handle.write(chunk)
        log_handle.flush()
        if live:
            sys.stderr.write(chunk)
            sys.stderr.flush()


def return_signal_name(returncode: int) -> str | None:
    if returncode >= 0:
        return None
    signal_number = -returncode
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"signal {signal_number}"


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
    except (FileNotFoundError, PermissionError):
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
    except (FileNotFoundError, PermissionError):
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


def result_to_record(result: BenchmarkResult) -> dict[str, Any]:
    return asdict(result)


def missing_output_paths(record: dict[str, Any]) -> list[str]:
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        return []
    return [path for path, size in outputs.items() if size is None]


def zero_byte_output_paths(record: dict[str, Any]) -> list[str]:
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        return []
    return [path for path, size in outputs.items() if size == 0]


def signal_name_from_record(record: dict[str, Any]) -> str | None:
    returncode = record.get("returncode")
    if not isinstance(returncode, int):
        return None
    return return_signal_name(returncode)


def summary_row(record: dict[str, Any]) -> dict[str, Any]:
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        outputs = {}
    output_bytes = sum(size for size in outputs.values() if isinstance(size, int))
    missing_outputs = missing_output_paths(record)
    zero_byte_outputs = zero_byte_output_paths(record)
    command_json = record.get("command_json") if isinstance(record.get("command_json"), dict) else {}
    rows = command_json.get("rows")
    return {
        "name": record.get("name", ""),
        "group": record.get("group", ""),
        "returncode": record.get("returncode", ""),
        "signal_name": record.get("signal_name") or signal_name_from_record(record) or "",
        "elapsed_seconds": record.get("elapsed_seconds", ""),
        "peak_rss_mb": record.get("peak_rss_mb") or "",
        "cpu_seconds": record.get("cpu_seconds") or "",
        "cpu_percent": record.get("cpu_percent") or "",
        "output_mb": round(output_bytes / (1024 * 1024), 3),
        "output_count": len(outputs),
        "missing_outputs": len(missing_outputs),
        "missing_output_paths": "; ".join(missing_outputs),
        "zero_byte_outputs": len(zero_byte_outputs),
        "zero_byte_output_paths": "; ".join(zero_byte_outputs),
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


def speedup_table(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build speedup rows from (name, baseline_record, accelerated_record) triples."""

    rows: list[dict[str, Any]] = []
    for name, baseline, accelerated in pairs:
        baseline_seconds = baseline.get("elapsed_seconds")
        accelerated_seconds = accelerated.get("elapsed_seconds")
        speedup = None
        if isinstance(baseline_seconds, (int, float)) and isinstance(
            accelerated_seconds, (int, float)
        ) and accelerated_seconds > 0:
            speedup = baseline_seconds / accelerated_seconds
        baseline_rss = baseline.get("peak_rss_mb")
        accelerated_rss = accelerated.get("peak_rss_mb")
        rss_delta = None
        if isinstance(baseline_rss, (int, float)) and isinstance(accelerated_rss, (int, float)):
            rss_delta = accelerated_rss - baseline_rss
        rows.append(
            {
                "name": name,
                "baseline_seconds": baseline_seconds or "",
                "accelerated_seconds": accelerated_seconds or "",
                "speedup": round(speedup, 3) if speedup is not None else "",
                "baseline_peak_rss_mb": baseline_rss or "",
                "accelerated_peak_rss_mb": accelerated_rss or "",
                "peak_rss_delta_mb": round(rss_delta, 3) if rss_delta is not None else "",
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
    path: Path,
    summary_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    *,
    speedup_rows: list[dict[str, Any]] | None = None,
    extra_figures: list[str] | None = None,
) -> None:
    lines = [
        "# Benchmark profile",
        "",
        "## Overview",
        "",
        "| Step | Group | Status | Signal | Wall time (s) | Peak RSS (MiB) | CPU (s) | CPU % | Output (MiB) | Outputs | Missing | Empty | Rows |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {group} | {returncode} | {signal_name} | {elapsed_seconds} | "
            "{peak_rss_mb} | {cpu_seconds} | {cpu_percent} | {output_mb} | {output_count} | "
            "{missing_outputs} | {zero_byte_outputs} | {rows} |".format(**row)
        )

    if speedup_rows:
        lines.extend(
            [
                "",
                "## Speedup (baseline / accelerated)",
                "",
                "| Step | Baseline (s) | Accelerated (s) | Speedup | Baseline RSS (MiB) | Accelerated RSS (MiB) | RSS delta (MiB) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in speedup_rows:
            lines.append(
                "| {name} | {baseline_seconds} | {accelerated_seconds} | {speedup} | "
                "{baseline_peak_rss_mb} | {accelerated_peak_rss_mb} | {peak_rss_delta_mb} |".format(
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
            "- [CPU utilization](cpu-percent.svg)",
            "- [Output size](output-size.svg)",
        ]
    )
    if speedup_rows:
        lines.append("- [Speedup](speedup.svg)")
    if timing_rows:
        lines.append("- [CLI profile timings](command-timings.svg)")
    for figure in extra_figures or []:
        lines.append(f"- [{figure}]({figure})")

    missing_rows = [row for row in summary_rows if row["missing_outputs"]]
    if missing_rows:
        lines.extend(["", "## Missing Outputs", ""])
        for row in missing_rows:
            lines.append(f"- `{row['name']}`")
            for output_path in str(row["missing_output_paths"]).split("; "):
                if output_path:
                    lines.append(f"  - `{output_path}`")
    zero_rows = [row for row in summary_rows if row["zero_byte_outputs"]]
    if zero_rows:
        lines.extend(["", "## Zero-Byte Outputs", ""])
        for row in zero_rows:
            lines.append(f"- `{row['name']}`")
            for output_path in str(row["zero_byte_output_paths"]).split("; "):
                if output_path:
                    lines.append(f"  - `{output_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_index(
    path: Path,
    *,
    timing_rows: bool,
    speedup_rows: bool = False,
    plot_gallery: list[dict[str, str]] | None = None,
) -> None:
    timing_image = (
        '<h2>CLI Profile Timings</h2><img src="command-timings.svg" alt="CLI profile timings">'
        if timing_rows
        else ""
    )
    speedup_image = (
        '<h2>Speedup</h2><img src="speedup.svg" alt="Speedup by benchmark step">'
        if speedup_rows
        else ""
    )
    gallery_html = ""
    if plot_gallery:
        groups: dict[str, list[dict[str, str]]] = {}
        for item in plot_gallery:
            groups.setdefault(item["group"], []).append(item)
        sections = []
        for group, items in groups.items():
            figures = "".join(
                f'<figure><img src="{item["src"]}" alt="{item["title"]}">'
                f"<figcaption>{item['title']}</figcaption></figure>"
                for item in items
            )
            sections.append(f"<h3>{group}</h3><div class=\"gallery\">{figures}</div>")
        gallery_html = (
            "<h2>Generated Plots</h2>"
            "<p>Compare these against the published Danko-lab reference figures by eye.</p>"
            + "".join(sections)
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>touche benchmark profile</title>",
                "<style>",
                "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}",
                "img{display:block;max-width:100%;margin:1rem 0 2rem}",
                "a{color:#075985}",
                ".gallery{display:flex;flex-wrap:wrap;gap:1rem}",
                ".gallery figure{margin:0;max-width:480px}",
                ".gallery img{margin:0}",
                "figcaption{font-size:0.85rem;color:#475569}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>touche benchmark profile</h1>",
                '<p><a href="summary.md">Markdown summary</a> | <a href="summary.csv">CSV summary</a></p>',
                '<h2>Wall Time</h2><img src="wall-time.svg" alt="Wall time by benchmark step">',
                '<h2>Peak RSS</h2><img src="peak-rss.svg" alt="Peak RSS by benchmark step">',
                '<h2>CPU Utilization</h2><img src="cpu-percent.svg" alt="CPU percent by benchmark step">',
                speedup_image,
                '<h2>Output Size</h2><img src="output-size.svg" alt="Output size by benchmark step">',
                timing_image,
                gallery_html,
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


def plot_speedup(rows: list[dict[str, Any]], *, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plottable = [row for row in rows if row["speedup"] != ""]
    if not plottable:
        return
    labels = [str(row["name"]) for row in plottable]
    values = [float(row["speedup"]) for row in plottable]
    height = max(4.0, 0.36 * len(plottable) + 1.2)
    fig, ax = plt.subplots(figsize=(11, height))
    positions = range(len(plottable))
    ax.barh(list(positions), values, color="#f59e0b")
    ax.axvline(1.0, color="#334155", linestyle="--", linewidth=1)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("speedup (baseline seconds / accelerated seconds)")
    ax.set_title("Numba Speedup Over Numpy")
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
        "fisher": "#64748b",
    }.get(group, "#64748b")


def write_profile_report(
    records: list[dict[str, Any]],
    *,
    report_dir: Path,
    speedup_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] | None = None,
    plot_gallery: list[dict[str, str]] | None = None,
) -> None:
    """Write the shared CSV/Markdown/HTML/plot report for a list of step records."""

    report_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = report_dir / ".matplotlib-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir = report_dir / ".cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    summary_rows = [summary_row(record) for record in records]
    timing_rows = profile_timing_rows(records)
    speedup_rows = speedup_table(speedup_pairs) if speedup_pairs else []

    write_csv(report_dir / "summary.csv", summary_rows)
    if timing_rows:
        write_csv(report_dir / "command-timings.csv", timing_rows)
    if speedup_rows:
        write_csv(report_dir / "speedup.csv", speedup_rows)
    write_markdown_summary(
        report_dir / "summary.md", summary_rows, timing_rows, speedup_rows=speedup_rows
    )

    resolved_gallery: list[dict[str, str]] = []
    if plot_gallery:
        gallery_dir = report_dir / "plots"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        for item in plot_gallery:
            src_path = Path(item["src_svg"])
            if not src_path.exists():
                continue
            dest_name = f"{item['group']}__{src_path.stem}.svg".replace(" ", "_")
            dest_path = gallery_dir / dest_name
            shutil.copyfile(src_path, dest_path)
            resolved_gallery.append(
                {"group": item["group"], "title": item["title"], "src": f"plots/{dest_name}"}
            )

    write_html_index(
        report_dir / "index.html",
        timing_rows=bool(timing_rows),
        speedup_rows=bool(speedup_rows),
        plot_gallery=resolved_gallery,
    )

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
        [row for row in summary_rows if row["cpu_percent"] != ""],
        metric="cpu_percent",
        title="CPU Utilization by Benchmark Step",
        xlabel="% of one core",
        out=report_dir / "cpu-percent.svg",
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
    if speedup_rows:
        plot_speedup(speedup_rows, out=report_dir / "speedup.svg")
