"""End-to-end command-group workflows, each writing a `manifest.json` alongside its outputs.

Public API: `run_local_decay_pipeline`, `run_background_pipeline`,
`run_apa_pipeline` -- each chains a domain module's file-driven wrappers
(e.g. call -> assign -> plot for local-decay) and records inputs,
parameters, outputs, metrics, and timings. `_base_manifest`/`_write_manifest`
are the shared internal manifest-building helpers; `_close_figure` releases
a plot's matplotlib figure after saving.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from touche import __version__
from touche.anchors import read_bed_anchors
from touche.apa import aggregate_apa, compare_apa_change
from touche.backends import DEFAULT_FISHER_BACKEND, DEFAULT_LOWESS_BACKEND
from touche.background import compare_background_ratios, count_ep_and_background
from touche.instrumentation import Instrumentation, make_instrumentation
from touche.local_decay import assign_pair_types, call_local_decay, plot_pair_type_distribution
from touche.models import NamedPath


def run_local_decay_pipeline(
    baits_path: str | Path,
    preys_path: str | Path,
    pairs_path: str | Path,
    functional_path: str | Path,
    nonfunctional_path: str | Path,
    out_dir: str | Path,
    *,
    dist: int = 1_000_000,
    cap: int = 2_000,
    min_distance: int = 5_000,
    source: str = "auto",
    lowess_window: int = 5_000,
    lowess_delta: float = 16.0,
    lowess_iterations: int = 3,
    plot_min_contacts: int = 1,
    plot_min_distance: int = 15_000,
    reference_style: bool = True,
    lowess_backend: str = DEFAULT_LOWESS_BACKEND,
    fisher_backend: str = DEFAULT_FISHER_BACKEND,
    n_jobs: int = 1,
    index_strategy: str = "cache",
    cache_dir: str | Path | None = None,
    cache_prefix: str = "contacts",
    require_cache: bool = False,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> dict[str, Any]:
    """Run local-decay call, pair assignment, and violin plotting."""

    started = perf_counter()
    instrument = make_instrumentation(progress, profile=profile)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contacts_out = out_dir / "ContactCaller_microC_output.tsv"
    assignments_out = (
        out_dir
        / "ContactCaller_microC_output_W_functional_nonfunctional_and_other_pair_assignments.tsv"
    )
    plot_table_out = out_dir / "Violinplot_for_normalized_contacts_by_pair_type.tsv"
    figure_out = out_dir / "Violinplot_for_normalized_contacts_by_pair_type.svg"

    with instrument.step("call local-decay"):
        calls = call_local_decay(
            baits_path,
            preys_path,
            pairs_path,
            contacts_out,
            dist=dist,
            cap=cap,
            min_distance=min_distance,
            source=source,
            lowess_window=lowess_window,
            lowess_delta=lowess_delta,
            lowess_backend=lowess_backend,
            fisher_backend=fisher_backend,
            n_jobs=n_jobs,
            index_strategy=index_strategy,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            require_cache=require_cache,
            lowess_iterations=lowess_iterations,
            progress=instrument,
        )
    with instrument.step("assign pair types"):
        assignments = assign_pair_types(
            contacts_out, functional_path, nonfunctional_path, assignments_out
        )
    with instrument.step("plot pair type distribution"):
        plot_data, fig = plot_pair_type_distribution(
            assignments_out,
            figure_out,
            min_contacts=plot_min_contacts,
            min_distance=plot_min_distance,
            plot_table_out=plot_table_out,
            reference_style=reference_style,
        )
        _close_figure(fig)

    manifest = _base_manifest(
        "local-decay run",
        inputs={
            "baits": str(baits_path),
            "preys": str(preys_path),
            "pairs": str(pairs_path),
            "functional": str(functional_path),
            "nonfunctional": str(nonfunctional_path),
        },
        parameters={
            "dist": dist,
            "cap": cap,
            "min_distance": min_distance,
            "source": source,
            "lowess_window": lowess_window,
            "lowess_delta": lowess_delta,
            "lowess_iterations": lowess_iterations,
            "plot_min_contacts": plot_min_contacts,
            "plot_min_distance": plot_min_distance,
            "reference_style": reference_style,
            "lowess_backend": lowess_backend,
            "fisher_backend": fisher_backend,
            "n_jobs": n_jobs,
            "index_strategy": index_strategy,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "cache_prefix": cache_prefix,
            "require_cache": require_cache,
        },
        outputs={
            "contacts": str(contacts_out),
            "assignments": str(assignments_out),
            "plot_table": str(plot_table_out),
            "figure": str(figure_out),
        },
        metrics={
            "called_rows": int(len(calls)),
            "assigned_rows": int(len(assignments)),
            "plotted_rows": int(len(plot_data)),
        },
        started=started,
        timings=instrument.timings,
    )
    return _write_manifest(out_dir / "manifest.json", manifest)


def run_background_pipeline(
    control: NamedPath,
    treatments: list[NamedPath],
    depths: dict[str, int],
    baits_path: str | Path,
    preys_path: str | Path,
    out_dir: str | Path,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
    source: str = "auto",
    min_ep_cpb: float = 8.0,
    reference_style: bool = True,
    index_strategy: str = "all",
    cache_dir: str | Path | None = None,
    require_cache: bool = False,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> dict[str, Any]:
    """Run per-sample EP/background counts and treatment comparisons.

    `cache_dir`, if given, is a base directory namespaced per sample
    (`cache_dir/sample.name`) so samples' NPZ caches never collide -- each
    defaults to a `contact_index_cache/` directory next to that sample's own
    count output.
    """

    started = perf_counter()
    instrument = make_instrumentation(progress, profile=profile)
    out_dir = Path(out_dir)
    counts_dir = out_dir / "counts"
    plots_dir = out_dir / "plots"
    counts_dir.mkdir(parents=True, exist_ok=True)
    samples = [control, *treatments]
    count_paths: dict[str, Path] = {}
    count_rows: dict[str, int] = {}
    sample_iter = instrument.iter(samples, total=len(samples), desc="background samples", unit="sample")
    for sample in sample_iter:
        sample_out = counts_dir / f"{sample.name}_EP_and_BG_contacts.tsv"
        with instrument.step(f"count background {sample.name}"):
            counts = count_ep_and_background(
                sample.path,
                baits_path,
                preys_path,
                sample_out,
                min_distance=min_distance,
                max_distance=max_distance,
                window=window,
                min_bg_distance=min_bg_distance,
                max_bg_distance=max_bg_distance,
                source=source,
                index_strategy=index_strategy,
                cache_dir=Path(cache_dir) / sample.name if cache_dir is not None else None,
                require_cache=require_cache,
                progress=instrument,
            )
        count_paths[sample.name] = sample_out
        count_rows[sample.name] = int(len(counts))

    with instrument.step("compare background ratios"):
        merged_table = out_dir / "background_comparison.tsv"
        merged, plot_paths = compare_background_ratios(
            NamedPath(control.name, count_paths[control.name]),
            [NamedPath(sample.name, count_paths[sample.name]) for sample in treatments],
            depths,
            min_ep_cpb=min_ep_cpb,
            out_dir=plots_dir,
            table_out=merged_table,
            reference_style=reference_style,
        )

    manifest = _base_manifest(
        "background run",
        inputs={
            "control": {control.name: str(control.path)},
            "treatments": {sample.name: str(sample.path) for sample in treatments},
            "baits": str(baits_path),
            "preys": str(preys_path),
        },
        parameters={
            "depths": depths,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "window": window,
            "min_bg_distance": min_bg_distance,
            "max_bg_distance": max_bg_distance,
            "source": source,
            "min_ep_cpb": min_ep_cpb,
            "reference_style": reference_style,
            "index_strategy": index_strategy,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "require_cache": require_cache,
        },
        outputs={
            "counts": {name: str(path) for name, path in count_paths.items()},
            "comparison_table": str(merged_table),
            "plots": {name: str(path) for name, path in plot_paths.items()},
        },
        metrics={
            "count_rows": count_rows,
            "comparison_rows": int(len(merged)),
        },
        started=started,
        timings=instrument.timings,
    )
    return _write_manifest(out_dir / "manifest.json", manifest)


def run_apa_pipeline(
    control: NamedPath,
    treatment: NamedPath,
    baits_path: str | Path,
    preys_path: str | Path,
    out_dir: str | Path,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    pixels: int,
    source: str = "auto",
    shift: int = 75,
    bait_count: int | None = None,
    prey_count: int | None = None,
    reference_style: bool = True,
    index_strategy: str = "all",
    cache_dir: str | Path | None = None,
    require_cache: bool = False,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> dict[str, Any]:
    """Run aggregate APA for two samples and compare treatment to control.

    `cache_dir`, if given, is a base directory namespaced per sample
    (`cache_dir/control.name`, `cache_dir/treatment.name`) so the two
    samples' NPZ caches never collide -- each defaults to a
    `contact_index_cache/` directory next to that sample's own output dir.
    """

    started = perf_counter()
    instrument = make_instrumentation(progress, profile=profile)
    out_dir = Path(out_dir)
    control_dir = out_dir / control.name
    treatment_dir = out_dir / treatment.name
    compare_dir = out_dir / f"{treatment.name}_vs_{control.name}"

    with instrument.step(f"aggregate apa {control.name}"):
        control_outputs = aggregate_apa(
            control.path,
            baits_path,
            preys_path,
            control_dir,
            min_distance=min_distance,
            max_distance=max_distance,
            window=window,
            pixels=pixels,
            source=source,
            shift=shift,
            reference_style=reference_style,
            index_strategy=index_strategy,
            cache_dir=Path(cache_dir) / control.name if cache_dir is not None else None,
            require_cache=require_cache,
            progress=instrument,
        )
    with instrument.step(f"aggregate apa {treatment.name}"):
        treatment_outputs = aggregate_apa(
            treatment.path,
            baits_path,
            preys_path,
            treatment_dir,
            min_distance=min_distance,
            max_distance=max_distance,
            window=window,
            pixels=pixels,
            source=source,
            shift=shift,
            reference_style=reference_style,
            index_strategy=index_strategy,
            cache_dir=Path(cache_dir) / treatment.name if cache_dir is not None else None,
            require_cache=require_cache,
            progress=instrument,
        )
    compare_dir.mkdir(parents=True, exist_ok=True)
    inferred_bait_count = (
        bait_count if bait_count is not None else int(len(read_bed_anchors(baits_path)))
    )
    inferred_prey_count = (
        prey_count if prey_count is not None else int(len(read_bed_anchors(preys_path)))
    )
    matrix_out = compare_dir / "ObsOverExp.csv"
    heatmap_out = compare_dir / "ObsOverExp.svg"
    with instrument.step("compare apa change"):
        comparison = compare_apa_change(
            control_outputs["matrix"],
            treatment_outputs["matrix"],
            control_outputs["baits_signal"],
            control_outputs["preys_signal"],
            treatment_outputs["baits_signal"],
            treatment_outputs["preys_signal"],
            bait_count=inferred_bait_count,
            prey_count=inferred_prey_count,
            out=heatmap_out,
            matrix_out=matrix_out,
            window=window,
            pixels=pixels,
            reference_style=reference_style,
        )

    manifest = _base_manifest(
        "apa run",
        inputs={
            "control": {control.name: str(control.path)},
            "treatment": {treatment.name: str(treatment.path)},
            "baits": str(baits_path),
            "preys": str(preys_path),
        },
        parameters={
            "min_distance": min_distance,
            "max_distance": max_distance,
            "window": window,
            "pixels": pixels,
            "source": source,
            "shift": shift,
            "bait_count": inferred_bait_count,
            "prey_count": inferred_prey_count,
            "reference_style": reference_style,
            "index_strategy": index_strategy,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "require_cache": require_cache,
        },
        outputs={
            "control": {key: str(value) for key, value in control_outputs.items()},
            "treatment": {key: str(value) for key, value in treatment_outputs.items()},
            "comparison_matrix": str(matrix_out),
            "comparison_heatmap": str(heatmap_out),
        },
        metrics={
            "comparison_rows": int(comparison.shape[0]),
            "comparison_columns": int(comparison.shape[1]),
        },
        started=started,
        timings=instrument.timings,
    )
    return _write_manifest(out_dir / "manifest.json", manifest)


def _base_manifest(
    command: str,
    *,
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    metrics: dict[str, Any],
    started: float,
    timings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest fields common to every pipeline (version, timing, inputs/outputs/metrics)."""
    manifest = {
        "schema_version": 1,
        "touche_version": __version__,
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 6),
        "inputs": inputs,
        "parameters": parameters,
        "outputs": outputs,
        "metrics": metrics,
    }
    if timings:
        manifest["timings"] = timings
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write the manifest JSON to `path`, recording its own path in the returned dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest"] = str(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _close_figure(fig: Any) -> None:
    """Release a matplotlib figure's memory once it's been saved."""
    import matplotlib.pyplot as plt

    plt.close(fig)
