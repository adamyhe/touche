"""`touche background` subcommands: count, compare, and run EP/background workflows.

`add_background_parser` is the public entry point called from `cli/main.py`.
Every `_`-prefixed function below is an argparse `func=` callback, not meant
to be called directly -- it unpacks `args` and forwards to the matching
`touche.background`/`touche.pipelines` function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from touche.background import count_ep_and_background, compare_background_ratios, parse_named_depth, parse_named_path
from touche.cli.utils import add_instrumentation_args, add_timings, make_cli_instrumentation, print_json
from touche.pipelines import run_background_pipeline


def add_background_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `background count/compare/run` on `subparsers`."""
    background = subparsers.add_parser("background", help="EP/background contact analyses")
    background_sub = background.add_subparsers(dest="background_command", required=True)

    count_parser = background_sub.add_parser(
        "count",
        help="Count EP contacts and local background contacts for bait/prey pairs",
        description=(
            "Count enhancer-promoter contacts and local background contacts for one "
            "sample. Use background run to count multiple samples and compare them."
        ),
    )
    count_parser.add_argument("--pairs", required=True, type=Path, help="Input pairs file for one sample.")
    count_parser.add_argument("--baits", required=True, type=Path, help="BED-like promoter/bait anchors.")
    count_parser.add_argument("--preys", required=True, type=Path, help="BED-like enhancer/prey anchors.")
    count_parser.add_argument("--out", required=True, type=Path, help="Output count TSV path.")
    count_parser.add_argument(
        "--min-distance",
        required=True,
        type=int,
        help="Minimum bait-prey center distance to count as a candidate pair.",
    )
    count_parser.add_argument(
        "--max-distance",
        required=True,
        type=int,
        help="Maximum bait-prey center distance to count as a candidate pair.",
    )
    count_parser.add_argument(
        "--window",
        required=True,
        type=int,
        help="Half-window around each bait/prey center used for EP contact counts.",
    )
    count_parser.add_argument(
        "--min-bg-distance",
        required=True,
        type=int,
        help="Inner distance from an anchor center for local background windows.",
    )
    count_parser.add_argument(
        "--max-bg-distance",
        required=True,
        type=int,
        help="Outer distance from an anchor center for local background windows.",
    )
    count_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input pairs layout.",
    )
    add_instrumentation_args(count_parser)
    count_parser.set_defaults(func=_count_background)

    compare_parser = background_sub.add_parser(
        "compare",
        help="Compare EP/background ratios across control and treatment samples",
        description="Join per-sample background count tables and write comparison plots/tables.",
    )
    compare_parser.add_argument(
        "--control",
        required=True,
        help="Control count table as NAME=PATH, for example DMSO=counts/DMSO.tsv.",
    )
    compare_parser.add_argument(
        "--treatments",
        required=True,
        nargs="+",
        help="Treatment count tables as NAME=PATH values, for example FLV=counts/FLV.tsv.",
    )
    compare_parser.add_argument(
        "--depths",
        required=True,
        nargs="+",
        help="Sequencing depths as NAME=INTEGER values for control and treatments.",
    )
    compare_parser.add_argument(
        "--min-ep-cpb",
        default=8.0,
        type=float,
        help="Minimum EP contacts per billion contacts required for plotting/comparison.",
    )
    compare_parser.add_argument("--out-dir", type=Path, help="Directory for comparison SVG plots.")
    compare_parser.add_argument("--table-out", type=Path, help="Optional merged comparison TSV output.")
    compare_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    compare_parser.set_defaults(func=_compare_background)

    run_parser = background_sub.add_parser(
        "run",
        help="Run per-sample EP/background counting and comparisons",
        description=(
            "Count EP/background contacts for control and treatment pairs files, compare "
            "normalized ratios, write plots/tables, and record a manifest."
        ),
    )
    run_parser.add_argument(
        "--control",
        required=True,
        help="Control sample pairs file as NAME=PAIRS, for example DMSO=dmso.pairs.gz.",
    )
    run_parser.add_argument(
        "--treatments",
        required=True,
        nargs="+",
        help="Treatment sample pairs files as NAME=PAIRS values.",
    )
    run_parser.add_argument(
        "--depths",
        required=True,
        nargs="+",
        help="Sequencing depths as NAME=INTEGER values for control and treatments.",
    )
    run_parser.add_argument("--baits", required=True, type=Path, help="BED-like promoter/bait anchors.")
    run_parser.add_argument("--preys", required=True, type=Path, help="BED-like enhancer/prey anchors.")
    run_parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for counts, plots, merged table, and manifest.",
    )
    run_parser.add_argument(
        "--min-distance",
        required=True,
        type=int,
        help="Minimum bait-prey center distance to count as a candidate pair.",
    )
    run_parser.add_argument(
        "--max-distance",
        required=True,
        type=int,
        help="Maximum bait-prey center distance to count as a candidate pair.",
    )
    run_parser.add_argument(
        "--window",
        required=True,
        type=int,
        help="Half-window around each bait/prey center used for EP contact counts.",
    )
    run_parser.add_argument(
        "--min-bg-distance",
        required=True,
        type=int,
        help="Inner distance from an anchor center for local background windows.",
    )
    run_parser.add_argument(
        "--max-bg-distance",
        required=True,
        type=int,
        help="Outer distance from an anchor center for local background windows.",
    )
    run_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input pairs layout.",
    )
    run_parser.add_argument(
        "--min-ep-cpb",
        default=8.0,
        type=float,
        help="Minimum EP contacts per billion contacts required for plotting/comparison.",
    )
    add_instrumentation_args(run_parser)
    run_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    run_parser.set_defaults(func=_run_background)


def _count_background(args: argparse.Namespace) -> None:
    instrument = make_cli_instrumentation(args)
    result = count_ep_and_background(
        args.pairs,
        args.baits,
        args.preys,
        args.out,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        window=args.window,
        min_bg_distance=args.min_bg_distance,
        max_bg_distance=args.max_bg_distance,
        source=args.source,
        progress=instrument,
    )
    print_json(add_timings({"rows": int(len(result)), "out": str(args.out)}, instrument))


def _compare_background(args: argparse.Namespace) -> None:
    control = parse_named_path(args.control)
    treatments = [parse_named_path(value) for value in args.treatments]
    depths = {named.name: named.depth for named in (parse_named_depth(value) for value in args.depths)}
    merged, plot_paths = compare_background_ratios(
        control,
        treatments,
        depths,
        min_ep_cpb=args.min_ep_cpb,
        out_dir=args.out_dir,
        table_out=args.table_out,
        reference_style=args.reference_style,
    )
    print_json(
        {
            "rows": int(len(merged)),
            "plots": {key: str(value) for key, value in plot_paths.items()},
            "table_out": str(args.table_out) if args.table_out else None,
        }
    )


def _run_background(args: argparse.Namespace) -> None:
    instrument = make_cli_instrumentation(args)
    control = parse_named_path(args.control)
    treatments = [parse_named_path(value) for value in args.treatments]
    depths = {named.name: named.depth for named in (parse_named_depth(value) for value in args.depths)}
    manifest = run_background_pipeline(
        control,
        treatments,
        depths,
        args.baits,
        args.preys,
        args.out_dir,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        window=args.window,
        min_bg_distance=args.min_bg_distance,
        max_bg_distance=args.max_bg_distance,
        source=args.source,
        min_ep_cpb=args.min_ep_cpb,
        reference_style=args.reference_style,
        progress=instrument,
    )
    print_json(manifest)
