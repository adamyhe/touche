from __future__ import annotations

import argparse
from pathlib import Path

from touche.backends import DEFAULT_BACKEND
from touche.background import count_ep_and_background, compare_background_ratios, parse_named_depth, parse_named_path
from touche.cli.utils import add_instrumentation_args, add_timings, make_cli_instrumentation, print_json
from touche.pipelines import run_background_pipeline


def add_background_parser(subparsers: argparse._SubParsersAction) -> None:
    background = subparsers.add_parser("background", help="EP/background contact analyses")
    background_sub = background.add_subparsers(dest="background_command", required=True)

    count_parser = background_sub.add_parser(
        "count",
        help="Count EP contacts and local background contacts for bait/prey pairs",
    )
    count_parser.add_argument("--pairs", required=True, type=Path)
    count_parser.add_argument("--baits", required=True, type=Path)
    count_parser.add_argument("--preys", required=True, type=Path)
    count_parser.add_argument("--out", required=True, type=Path)
    count_parser.add_argument("--min-distance", required=True, type=int)
    count_parser.add_argument("--max-distance", required=True, type=int)
    count_parser.add_argument("--window", required=True, type=int)
    count_parser.add_argument("--min-bg-distance", required=True, type=int)
    count_parser.add_argument("--max-bg-distance", required=True, type=int)
    count_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    count_parser.add_argument("--backend", choices=["numpy", "numba"], default=DEFAULT_BACKEND)
    add_instrumentation_args(count_parser)
    count_parser.set_defaults(func=_count_background)

    compare_parser = background_sub.add_parser(
        "compare",
        help="Compare EP/background ratios across control and treatment samples",
    )
    compare_parser.add_argument("--control", required=True, help="Control sample as NAME=PATH")
    compare_parser.add_argument(
        "--treatments",
        required=True,
        nargs="+",
        help="Treatment samples as NAME=PATH",
    )
    compare_parser.add_argument(
        "--depths",
        required=True,
        nargs="+",
        help="Sequencing depths as NAME=INTEGER",
    )
    compare_parser.add_argument("--min-ep-cpb", default=8.0, type=float)
    compare_parser.add_argument("--out-dir", type=Path)
    compare_parser.add_argument("--table-out", type=Path)
    compare_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    compare_parser.set_defaults(func=_compare_background)

    run_parser = background_sub.add_parser(
        "run",
        help="Run per-sample EP/background counting and comparisons",
    )
    run_parser.add_argument("--control", required=True, help="Control sample as NAME=PAIRS")
    run_parser.add_argument(
        "--treatments",
        required=True,
        nargs="+",
        help="Treatment samples as NAME=PAIRS",
    )
    run_parser.add_argument(
        "--depths",
        required=True,
        nargs="+",
        help="Sequencing depths as NAME=INTEGER",
    )
    run_parser.add_argument("--baits", required=True, type=Path)
    run_parser.add_argument("--preys", required=True, type=Path)
    run_parser.add_argument("--out-dir", required=True, type=Path)
    run_parser.add_argument("--min-distance", required=True, type=int)
    run_parser.add_argument("--max-distance", required=True, type=int)
    run_parser.add_argument("--window", required=True, type=int)
    run_parser.add_argument("--min-bg-distance", required=True, type=int)
    run_parser.add_argument("--max-bg-distance", required=True, type=int)
    run_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    run_parser.add_argument("--min-ep-cpb", default=8.0, type=float)
    run_parser.add_argument("--backend", choices=["numpy", "numba"], default=DEFAULT_BACKEND)
    add_instrumentation_args(run_parser)
    run_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        backend=args.backend,
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
        backend=args.backend,
        progress=instrument,
    )
    print_json(manifest)
