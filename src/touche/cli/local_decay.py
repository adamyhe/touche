from __future__ import annotations

import argparse
from pathlib import Path

from touche.cli.utils import print_json
from touche.local_decay import assign_pair_types, call_local_decay, plot_pair_type_distribution
from touche.pipelines import run_local_decay_pipeline


def add_local_decay_parser(subparsers: argparse._SubParsersAction) -> None:
    local_decay = subparsers.add_parser("local-decay", help="Local-decay contact analyses")
    local_decay_sub = local_decay.add_subparsers(dest="local_decay_command", required=True)

    call_parser = local_decay_sub.add_parser(
        "call",
        help="Call bait-prey contacts normalized by local distance decay",
    )
    call_parser.add_argument("--baits", required=True, type=Path)
    call_parser.add_argument("--preys", required=True, type=Path)
    call_parser.add_argument("--pairs", required=True, type=Path)
    call_parser.add_argument("--out", required=True, type=Path)
    call_parser.add_argument("--dist", default=1_000_000, type=int)
    call_parser.add_argument("--cap", default=2_000, type=int)
    call_parser.add_argument("--min-distance", default=5_000, type=int)
    call_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    call_parser.add_argument("--lowess-window", default=5_000, type=int)
    call_parser.add_argument("--lowess-delta", default=16.0, type=float)
    call_parser.set_defaults(func=_call_local_decay)

    run_parser = local_decay_sub.add_parser(
        "run",
        help="Run local-decay calling, pair assignment, and plotting",
    )
    run_parser.add_argument("--baits", required=True, type=Path)
    run_parser.add_argument("--preys", required=True, type=Path)
    run_parser.add_argument("--pairs", required=True, type=Path)
    run_parser.add_argument("--functional", required=True, type=Path)
    run_parser.add_argument("--nonfunctional", required=True, type=Path)
    run_parser.add_argument("--out-dir", required=True, type=Path)
    run_parser.add_argument("--dist", default=1_000_000, type=int)
    run_parser.add_argument("--cap", default=2_000, type=int)
    run_parser.add_argument("--min-distance", default=5_000, type=int)
    run_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    run_parser.add_argument("--lowess-window", default=5_000, type=int)
    run_parser.add_argument("--lowess-delta", default=16.0, type=float)
    run_parser.add_argument("--plot-min-contacts", default=1, type=int)
    run_parser.add_argument("--plot-min-distance", default=15_000, type=int)
    run_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.set_defaults(func=_run_local_decay)

    assign_parser = local_decay_sub.add_parser(
        "assign-pair-types",
        help="Assign local-decay contact rows to functional/nonfunctional/other classes",
    )
    assign_parser.add_argument("--contacts", required=True, type=Path)
    assign_parser.add_argument("--functional", required=True, type=Path)
    assign_parser.add_argument("--nonfunctional", required=True, type=Path)
    assign_parser.add_argument("--out", required=True, type=Path)
    assign_parser.set_defaults(func=_assign_pair_types)

    plot_parser = local_decay_sub.add_parser(
        "plot",
        help="Plot observed/expected local-decay contacts by pair type",
    )
    plot_parser.add_argument("--assignments", required=True, type=Path)
    plot_parser.add_argument("--out", required=True, type=Path)
    plot_parser.add_argument("--min-contacts", default=1, type=int)
    plot_parser.add_argument("--min-distance", default=15_000, type=int)
    plot_parser.add_argument("--plot-table-out", type=Path)
    plot_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    plot_parser.set_defaults(func=_plot_pair_type_distribution)


def _call_local_decay(args: argparse.Namespace) -> None:
    calls = call_local_decay(
        args.baits,
        args.preys,
        args.pairs,
        args.out,
        dist=args.dist,
        cap=args.cap,
        min_distance=args.min_distance,
        source=args.source,
        lowess_window=args.lowess_window,
        lowess_delta=args.lowess_delta,
    )
    print_json(
        {
            "rows": int(len(calls)),
            "out": str(args.out),
        }
    )


def _run_local_decay(args: argparse.Namespace) -> None:
    manifest = run_local_decay_pipeline(
        args.baits,
        args.preys,
        args.pairs,
        args.functional,
        args.nonfunctional,
        args.out_dir,
        dist=args.dist,
        cap=args.cap,
        min_distance=args.min_distance,
        source=args.source,
        lowess_window=args.lowess_window,
        lowess_delta=args.lowess_delta,
        plot_min_contacts=args.plot_min_contacts,
        plot_min_distance=args.plot_min_distance,
        reference_style=args.reference_style,
    )
    print_json(manifest)


def _assign_pair_types(args: argparse.Namespace) -> None:
    assignments = assign_pair_types(
        args.contacts,
        args.functional,
        args.nonfunctional,
        args.out,
    )
    print_json(
        {
            "rows": int(len(assignments)),
            "counts": assignments["PosNeg"].value_counts().sort_index().to_dict(),
            "out": str(args.out),
        }
    )


def _plot_pair_type_distribution(args: argparse.Namespace) -> None:
    plot_data = plot_pair_type_distribution(
        args.assignments,
        args.out,
        min_contacts=args.min_contacts,
        min_distance=args.min_distance,
        plot_table_out=args.plot_table_out,
        reference_style=args.reference_style,
    )
    print_json(
        {
            "rows": int(len(plot_data)),
            "out": str(args.out),
            "plot_table_out": str(args.plot_table_out) if args.plot_table_out else None,
        }
    )
