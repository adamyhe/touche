"""`touche apa` subcommands: aggregate, compare, and run APA workflows.

`add_apa_parser` is the public entry point called from `cli/main.py`. Every
`_`-prefixed function below is an argparse `func=` callback, not meant to be
called directly -- it unpacks `args` and forwards to the matching
`touche.apa`/`touche.pipelines` function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from touche.apa import aggregate_apa, compare_apa_change
from touche.background import parse_named_path
from touche.cli.utils import add_instrumentation_args, add_timings, make_cli_instrumentation, print_json
from touche.pipelines import run_apa_pipeline


def add_apa_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `apa aggregate/compare/run` on `subparsers`."""
    apa = subparsers.add_parser("apa", help="Aggregated peak analysis workflows")
    apa_sub = apa.add_subparsers(dest="apa_command", required=True)

    apa_aggregate = apa_sub.add_parser(
        "aggregate",
        help="Aggregate APA matrix and 1D anchor signals for one sample",
        description=(
            "Build one sample's aggregate peak-analysis matrix plus bait/prey 1D "
            "signal tracks from pairs and anchor files."
        ),
    )
    apa_aggregate.add_argument("--pairs", required=True, type=Path, help="Input pairs file.")
    apa_aggregate.add_argument("--baits", required=True, type=Path, help="BED-like bait anchors.")
    apa_aggregate.add_argument("--preys", required=True, type=Path, help="BED-like prey anchors.")
    apa_aggregate.add_argument(
        "--min-distance",
        required=True,
        type=int,
        help="Minimum bait-prey center distance included in the APA aggregate.",
    )
    apa_aggregate.add_argument(
        "--max-distance",
        required=True,
        type=int,
        help="Maximum bait-prey center distance included in the APA aggregate.",
    )
    apa_aggregate.add_argument(
        "--window",
        required=True,
        type=int,
        help="Total genomic window around each bait/prey pair; must be divisible by --pixels.",
    )
    apa_aggregate.add_argument(
        "--pixels",
        required=True,
        type=int,
        help="Number of APA bins/pixels across each axis.",
    )
    apa_aggregate.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for AggMat.csv, AggHeatmap.svg, and 1D signal CSVs.",
    )
    apa_aggregate.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input pairs layout.",
    )
    apa_aggregate.add_argument(
        "--shift",
        default=75,
        type=int,
        help="Shift contact endpoints before APA counting to match reference coordinates.",
    )
    apa_aggregate.add_argument(
        "--index-strategy",
        choices=["all", "cache"],
        default="all",
        help=(
            "Contact-index strategy. cache reads a persistent NPZ ContactIndex cache "
            "(building it first if missing) instead of re-parsing --pairs -- useful when "
            "background count also runs against the same sample."
        ),
    )
    apa_aggregate.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory containing or receiving chromosome-sharded contact cache files.",
    )
    apa_aggregate.add_argument("--cache-prefix", default="contacts", help="Cache manifest/shard prefix.")
    apa_aggregate.add_argument(
        "--require-cache",
        action="store_true",
        help="Require an existing cache instead of building one implicitly.",
    )
    add_instrumentation_args(apa_aggregate)
    apa_aggregate.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    apa_aggregate.set_defaults(func=_aggregate_apa)

    apa_compare = apa_sub.add_parser(
        "compare",
        help="Calculate and plot 1D-normalized inter-sample APA change",
        description=(
            "Compare two precomputed APA aggregates after normalizing by their bait/prey "
            "1D signal tracks."
        ),
    )
    apa_compare.add_argument("--control-apa", required=True, type=Path, help="Control AggMat.csv.")
    apa_compare.add_argument("--treatment-apa", required=True, type=Path, help="Treatment AggMat.csv.")
    apa_compare.add_argument(
        "--control-baits",
        required=True,
        type=Path,
        help="Control baits_genome_wide_contacts.csv.",
    )
    apa_compare.add_argument(
        "--control-preys",
        required=True,
        type=Path,
        help="Control preys_genome_wide_contacts.csv.",
    )
    apa_compare.add_argument(
        "--treatment-baits",
        required=True,
        type=Path,
        help="Treatment baits_genome_wide_contacts.csv.",
    )
    apa_compare.add_argument(
        "--treatment-preys",
        required=True,
        type=Path,
        help="Treatment preys_genome_wide_contacts.csv.",
    )
    apa_compare.add_argument(
        "--bait-count",
        required=True,
        type=int,
        help="Number of bait anchors used to normalize the comparison.",
    )
    apa_compare.add_argument(
        "--prey-count",
        required=True,
        type=int,
        help="Number of prey anchors used to normalize the comparison.",
    )
    apa_compare.add_argument("--out", type=Path, help="Optional comparison heatmap SVG output.")
    apa_compare.add_argument("--matrix-out", type=Path, help="Optional comparison matrix CSV output.")
    apa_compare.add_argument("--window", default=10_000, type=int, help="APA window size used for labels.")
    apa_compare.add_argument("--pixels", default=50, type=int, help="APA pixel count used for labels.")
    apa_compare.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    apa_compare.set_defaults(func=_compare_apa)

    apa_run = apa_sub.add_parser(
        "run",
        help="Run aggregate APA for control/treatment and compare them",
        description=(
            "Aggregate APA outputs for a control and treatment sample, compare the "
            "1D-normalized change, and write a manifest."
        ),
    )
    apa_run.add_argument(
        "--control",
        required=True,
        help="Control sample pairs file as NAME=PAIRS, for example DMSO=dmso.pairs.gz.",
    )
    apa_run.add_argument(
        "--treatment",
        required=True,
        help="Treatment sample pairs file as NAME=PAIRS, for example FLV=flv.pairs.gz.",
    )
    apa_run.add_argument("--baits", required=True, type=Path, help="BED-like bait anchors.")
    apa_run.add_argument("--preys", required=True, type=Path, help="BED-like prey anchors.")
    apa_run.add_argument(
        "--min-distance",
        required=True,
        type=int,
        help="Minimum bait-prey center distance included in APA aggregates.",
    )
    apa_run.add_argument(
        "--max-distance",
        required=True,
        type=int,
        help="Maximum bait-prey center distance included in APA aggregates.",
    )
    apa_run.add_argument(
        "--window",
        required=True,
        type=int,
        help="Total genomic window around each bait/prey pair; must be divisible by --pixels.",
    )
    apa_run.add_argument("--pixels", required=True, type=int, help="Number of APA bins/pixels per axis.")
    apa_run.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for per-sample APA outputs, comparison outputs, and manifest.",
    )
    apa_run.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input pairs layout.",
    )
    apa_run.add_argument(
        "--shift",
        default=75,
        type=int,
        help="Shift contact endpoints before APA counting to match reference coordinates.",
    )
    apa_run.add_argument(
        "--bait-count",
        type=int,
        help="Optional bait count for normalization. Defaults to the number of bait anchors.",
    )
    apa_run.add_argument(
        "--prey-count",
        type=int,
        help="Optional prey count for normalization. Defaults to the number of prey anchors.",
    )
    apa_run.add_argument(
        "--index-strategy",
        choices=["all", "cache"],
        default="all",
        help=(
            "Contact-index strategy. cache reads a persistent NPZ ContactIndex cache per "
            "sample (building it first if missing) instead of re-parsing each pairs file -- "
            "useful when background run also runs against the same samples."
        ),
    )
    apa_run.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Base directory for per-sample contact caches (one subdirectory per sample name). "
            "Defaults to a contact_index_cache/ directory next to each sample's own output."
        ),
    )
    apa_run.add_argument(
        "--require-cache",
        action="store_true",
        help="Require existing caches instead of building them implicitly.",
    )
    add_instrumentation_args(apa_run)
    apa_run.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    apa_run.set_defaults(func=_run_apa)


def _aggregate_apa(args: argparse.Namespace) -> None:
    instrument = make_cli_instrumentation(args)
    outputs = aggregate_apa(
        args.pairs,
        args.baits,
        args.preys,
        args.out_dir,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        window=args.window,
        pixels=args.pixels,
        source=args.source,
        shift=args.shift,
        reference_style=args.reference_style,
        index_strategy=args.index_strategy,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        require_cache=args.require_cache,
        progress=instrument,
    )
    print_json(add_timings({key: str(value) for key, value in outputs.items()}, instrument))


def _compare_apa(args: argparse.Namespace) -> None:
    matrix = compare_apa_change(
        args.control_apa,
        args.treatment_apa,
        args.control_baits,
        args.control_preys,
        args.treatment_baits,
        args.treatment_preys,
        bait_count=args.bait_count,
        prey_count=args.prey_count,
        out=args.out,
        matrix_out=args.matrix_out,
        window=args.window,
        pixels=args.pixels,
        reference_style=args.reference_style,
    )
    print_json(
        {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "out": str(args.out) if args.out else None,
            "matrix_out": str(args.matrix_out) if args.matrix_out else None,
        }
    )


def _run_apa(args: argparse.Namespace) -> None:
    instrument = make_cli_instrumentation(args)
    manifest = run_apa_pipeline(
        parse_named_path(args.control),
        parse_named_path(args.treatment),
        args.baits,
        args.preys,
        args.out_dir,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        window=args.window,
        pixels=args.pixels,
        source=args.source,
        shift=args.shift,
        bait_count=args.bait_count,
        prey_count=args.prey_count,
        reference_style=args.reference_style,
        index_strategy=args.index_strategy,
        cache_dir=args.cache_dir,
        require_cache=args.require_cache,
        progress=instrument,
    )
    print_json(manifest)
