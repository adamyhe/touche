from __future__ import annotations

import argparse
from pathlib import Path

from touche.apa import aggregate_apa, compare_apa_change
from touche.background import parse_named_path
from touche.cli.utils import add_instrumentation_args, add_timings, make_cli_instrumentation, print_json
from touche.pipelines import run_apa_pipeline


def add_apa_parser(subparsers: argparse._SubParsersAction) -> None:
    apa = subparsers.add_parser("apa", help="Aggregated peak analysis workflows")
    apa_sub = apa.add_subparsers(dest="apa_command", required=True)

    apa_aggregate = apa_sub.add_parser(
        "aggregate",
        help="Aggregate APA matrix and 1D anchor signals for one sample",
    )
    apa_aggregate.add_argument("--pairs", required=True, type=Path)
    apa_aggregate.add_argument("--baits", required=True, type=Path)
    apa_aggregate.add_argument("--preys", required=True, type=Path)
    apa_aggregate.add_argument("--min-distance", required=True, type=int)
    apa_aggregate.add_argument("--max-distance", required=True, type=int)
    apa_aggregate.add_argument("--window", required=True, type=int)
    apa_aggregate.add_argument("--pixels", required=True, type=int)
    apa_aggregate.add_argument("--out-dir", required=True, type=Path)
    apa_aggregate.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    apa_aggregate.add_argument("--shift", default=75, type=int)
    add_instrumentation_args(apa_aggregate)
    apa_aggregate.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    apa_aggregate.set_defaults(func=_aggregate_apa)

    apa_compare = apa_sub.add_parser(
        "compare",
        help="Calculate and plot 1D-normalized inter-sample APA change",
    )
    apa_compare.add_argument("--control-apa", required=True, type=Path)
    apa_compare.add_argument("--treatment-apa", required=True, type=Path)
    apa_compare.add_argument("--control-baits", required=True, type=Path)
    apa_compare.add_argument("--control-preys", required=True, type=Path)
    apa_compare.add_argument("--treatment-baits", required=True, type=Path)
    apa_compare.add_argument("--treatment-preys", required=True, type=Path)
    apa_compare.add_argument("--bait-count", required=True, type=int)
    apa_compare.add_argument("--prey-count", required=True, type=int)
    apa_compare.add_argument("--out", type=Path)
    apa_compare.add_argument("--matrix-out", type=Path)
    apa_compare.add_argument("--window", default=10_000, type=int)
    apa_compare.add_argument("--pixels", default=50, type=int)
    apa_compare.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    apa_compare.set_defaults(func=_compare_apa)

    apa_run = apa_sub.add_parser(
        "run",
        help="Run aggregate APA for control/treatment and compare them",
    )
    apa_run.add_argument("--control", required=True, help="Control sample as NAME=PAIRS")
    apa_run.add_argument("--treatment", required=True, help="Treatment sample as NAME=PAIRS")
    apa_run.add_argument("--baits", required=True, type=Path)
    apa_run.add_argument("--preys", required=True, type=Path)
    apa_run.add_argument("--min-distance", required=True, type=int)
    apa_run.add_argument("--max-distance", required=True, type=int)
    apa_run.add_argument("--window", required=True, type=int)
    apa_run.add_argument("--pixels", required=True, type=int)
    apa_run.add_argument("--out-dir", required=True, type=Path)
    apa_run.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    apa_run.add_argument("--shift", default=75, type=int)
    apa_run.add_argument("--bait-count", type=int)
    apa_run.add_argument("--prey-count", type=int)
    add_instrumentation_args(apa_run)
    apa_run.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        progress=instrument,
    )
    print_json(manifest)
