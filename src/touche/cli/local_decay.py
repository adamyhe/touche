"""`touche local-decay` subcommands: call, run, assign-pair-types, and plot.

`add_local_decay_parser` is the public entry point called from
`cli/main.py`. Every `_`-prefixed function below is an argparse `func=`
callback, not meant to be called directly -- it unpacks `args` and forwards
to the matching `touche.local_decay`/`touche.pipelines` function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from touche.backends import DEFAULT_FISHER_BACKEND, DEFAULT_LOWESS_BACKEND
from touche.cli.utils import add_instrumentation_args, add_timings, make_cli_instrumentation, print_json
from touche.local_decay import assign_pair_types, call_local_decay, plot_pair_type_distribution
from touche.pipelines import run_local_decay_pipeline


def add_local_decay_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `local-decay call/run/assign-pair-types/plot` on `subparsers`."""
    local_decay = subparsers.add_parser("local-decay", help="Local-decay contact analyses")
    local_decay_sub = local_decay.add_subparsers(dest="local_decay_command", required=True)

    call_parser = local_decay_sub.add_parser(
        "call",
        help="Call bait-prey contacts normalized by local distance decay",
        description=(
            "Call candidate bait/prey contacts, estimate local distance-decay expected "
            "contacts, and write the ContactCaller-style TSV."
        ),
    )
    call_parser.add_argument("--baits", required=True, type=Path, help="Two-column chr/center bait file.")
    call_parser.add_argument("--preys", required=True, type=Path, help="Two-column chr/center prey file.")
    call_parser.add_argument("--pairs", required=True, type=Path, help="Input pairs file.")
    call_parser.add_argument("--out", required=True, type=Path, help="Output contact-call TSV path.")
    call_parser.add_argument(
        "--dist",
        default=1_000_000,
        type=int,
        help="Maximum bait-prey distance considered around each bait.",
    )
    call_parser.add_argument(
        "--cap",
        default=2_000,
        type=int,
        help="Maximum observed count cap used by local-decay modeling.",
    )
    call_parser.add_argument(
        "--min-distance",
        default=5_000,
        type=int,
        help="Minimum bait-prey distance to report.",
    )
    call_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input pairs layout.",
    )
    call_parser.add_argument(
        "--lowess-window",
        default=5_000,
        type=int,
        help="Distance-bin window used by LOWESS smoothing.",
    )
    call_parser.add_argument(
        "--lowess-delta",
        default=16.0,
        type=float,
        help="LOWESS delta parameter controlling interpolation between fits.",
    )
    call_parser.add_argument(
        "--lowess-iterations",
        default=3,
        type=int,
        help="Robust LOWESS residual reweighting iterations. Lower is faster.",
    )
    call_parser.add_argument(
        "--index-strategy",
        choices=["cache", "all", "chromosome"],
        default="cache",
        help=(
            "Contact-index strategy. Use cache for real/repeated runs; all and chromosome "
            "are small-data or diagnostic modes."
        ),
    )
    call_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory containing or receiving chromosome-sharded contact cache files.",
    )
    call_parser.add_argument("--cache-prefix", default="contacts", help="Cache manifest/shard prefix.")
    call_parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Require an existing cache instead of building one implicitly.",
    )
    call_parser.add_argument(
        "--lowess-backend",
        choices=["statsmodels", "numba"],
        default=DEFAULT_LOWESS_BACKEND,
        help="LOWESS implementation. numba is default; statsmodels requires the legacy extra.",
    )
    call_parser.add_argument(
        "--fisher-backend",
        choices=["scipy", "numba"],
        default=DEFAULT_FISHER_BACKEND,
        help="Fisher exact-test implementation for local-decay p-values.",
    )
    call_parser.add_argument(
        "--jobs",
        "-j",
        default=1,
        type=int,
        help="Number of baits to process concurrently. Use 1 for sequential processing.",
    )
    add_instrumentation_args(call_parser)
    call_parser.set_defaults(func=_call_local_decay)

    run_parser = local_decay_sub.add_parser(
        "run",
        help="Run local-decay calling, pair assignment, and plotting",
        description=(
            "Run contact calling, assign functional/nonfunctional/other labels, plot "
            "observed/expected contacts, and write a manifest."
        ),
    )
    run_parser.add_argument("--baits", required=True, type=Path, help="Two-column chr/center bait file.")
    run_parser.add_argument("--preys", required=True, type=Path, help="Two-column chr/center prey file.")
    run_parser.add_argument("--pairs", required=True, type=Path, help="Input pairs file.")
    run_parser.add_argument(
        "--functional",
        required=True,
        type=Path,
        help="Functional bait/prey pair annotation file.",
    )
    run_parser.add_argument(
        "--nonfunctional",
        required=True,
        type=Path,
        help="Nonfunctional bait/prey pair annotation file.",
    )
    run_parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for calls, assignments, plot files, and manifest.",
    )
    run_parser.add_argument(
        "--dist",
        default=1_000_000,
        type=int,
        help="Maximum bait-prey distance considered around each bait.",
    )
    run_parser.add_argument(
        "--cap",
        default=2_000,
        type=int,
        help="Maximum observed count cap used by local-decay modeling.",
    )
    run_parser.add_argument(
        "--min-distance",
        default=5_000,
        type=int,
        help="Minimum bait-prey distance to report.",
    )
    run_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input pairs layout.",
    )
    run_parser.add_argument(
        "--lowess-window",
        default=5_000,
        type=int,
        help="Distance-bin window used by LOWESS smoothing.",
    )
    run_parser.add_argument(
        "--lowess-delta",
        default=16.0,
        type=float,
        help="LOWESS delta parameter controlling interpolation between fits.",
    )
    run_parser.add_argument(
        "--lowess-iterations",
        default=3,
        type=int,
        help="Robust LOWESS residual reweighting iterations. Lower is faster.",
    )
    run_parser.add_argument(
        "--index-strategy",
        choices=["cache", "all", "chromosome"],
        default="cache",
        help=(
            "Contact-index strategy. Use cache for real/repeated runs; all and chromosome "
            "are small-data or diagnostic modes."
        ),
    )
    run_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory containing or receiving chromosome-sharded contact cache files.",
    )
    run_parser.add_argument("--cache-prefix", default="contacts", help="Cache manifest/shard prefix.")
    run_parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Require an existing cache instead of building one implicitly.",
    )
    run_parser.add_argument(
        "--lowess-backend",
        choices=["statsmodels", "numba"],
        default=DEFAULT_LOWESS_BACKEND,
        help="LOWESS implementation. numba is default; statsmodels requires the legacy extra.",
    )
    run_parser.add_argument(
        "--fisher-backend",
        choices=["scipy", "numba"],
        default=DEFAULT_FISHER_BACKEND,
        help="Fisher exact-test implementation for local-decay p-values.",
    )
    run_parser.add_argument(
        "--jobs",
        "-j",
        default=1,
        type=int,
        help="Number of baits to process concurrently. Use 1 for sequential processing.",
    )
    add_instrumentation_args(run_parser)
    run_parser.add_argument(
        "--plot-min-contacts",
        default=1,
        type=int,
        help="Minimum observed contacts required for rows included in the violin plot.",
    )
    run_parser.add_argument(
        "--plot-min-distance",
        default=15_000,
        type=int,
        help="Minimum absolute bait-prey distance included in the violin plot.",
    )
    run_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    run_parser.set_defaults(func=_run_local_decay)

    assign_parser = local_decay_sub.add_parser(
        "assign-pair-types",
        help="Assign local-decay contact rows to functional/nonfunctional/other classes",
    )
    assign_parser.add_argument("--contacts", required=True, type=Path, help="Contact-call TSV from call.")
    assign_parser.add_argument(
        "--functional",
        required=True,
        type=Path,
        help="Functional bait/prey pair annotation file.",
    )
    assign_parser.add_argument(
        "--nonfunctional",
        required=True,
        type=Path,
        help="Nonfunctional bait/prey pair annotation file.",
    )
    assign_parser.add_argument("--out", required=True, type=Path, help="Output assignment TSV path.")
    assign_parser.set_defaults(func=_assign_pair_types)

    plot_parser = local_decay_sub.add_parser(
        "plot",
        help="Plot observed/expected local-decay contacts by pair type",
    )
    plot_parser.add_argument(
        "--assignments",
        required=True,
        type=Path,
        help="Assignment TSV from assign-pair-types or local-decay run.",
    )
    plot_parser.add_argument("--out", required=True, type=Path, help="Output SVG path.")
    plot_parser.add_argument(
        "--min-contacts",
        default=1,
        type=int,
        help="Minimum observed contacts required for rows included in the plot.",
    )
    plot_parser.add_argument(
        "--min-distance",
        default=15_000,
        type=int,
        help="Minimum absolute bait-prey distance included in the plot.",
    )
    plot_parser.add_argument("--plot-table-out", type=Path, help="Optional TSV of plot input rows.")
    plot_parser.add_argument(
        "--reference-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference-style plot formatting.",
    )
    plot_parser.set_defaults(func=_plot_pair_type_distribution)


def _call_local_decay(args: argparse.Namespace) -> None:
    instrument = make_cli_instrumentation(args)
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
        lowess_backend=args.lowess_backend,
        fisher_backend=args.fisher_backend,
        lowess_iterations=args.lowess_iterations,
        n_jobs=args.jobs,
        index_strategy=args.index_strategy,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        require_cache=args.require_cache,
        progress=instrument,
    )
    print_json(
        add_timings(
            {
                "rows": int(len(calls)),
                "out": str(args.out),
            },
            instrument,
        )
    )


def _run_local_decay(args: argparse.Namespace) -> None:
    instrument = make_cli_instrumentation(args)
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
        lowess_backend=args.lowess_backend,
        fisher_backend=args.fisher_backend,
        lowess_iterations=args.lowess_iterations,
        n_jobs=args.jobs,
        index_strategy=args.index_strategy,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        require_cache=args.require_cache,
        plot_min_contacts=args.plot_min_contacts,
        plot_min_distance=args.plot_min_distance,
        reference_style=args.reference_style,
        progress=instrument,
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
            "counts": dict(assignments["PosNeg"].value_counts().sort("PosNeg").iter_rows()),
            "out": str(args.out),
        }
    )


def _plot_pair_type_distribution(args: argparse.Namespace) -> None:
    plot_data, fig = plot_pair_type_distribution(
        args.assignments,
        args.out,
        min_contacts=args.min_contacts,
        min_distance=args.min_distance,
        plot_table_out=args.plot_table_out,
        reference_style=args.reference_style,
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    print_json(
        {
            "rows": int(len(plot_data)),
            "out": str(args.out),
            "plot_table_out": str(args.plot_table_out) if args.plot_table_out else None,
        }
    )
