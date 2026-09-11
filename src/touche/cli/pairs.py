"""`touche pairs` subcommands: build, import-calls, and annotate.

`add_pairs_parser` is the public entry point called from `cli/main.py`.
Every `_`-prefixed function below is an argparse `func=` callback that
unpacks `args` and forwards to `touche.pairs`/`touche.adapters`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from touche.adapters import LOOP_FORMATS, annotate_pairs, read_loop_calls, write_loop_calls
from touche.anchors import read_bed_anchors
from touche.cli.utils import print_json
from touche.pairs import build_pair_table, read_bedpe, write_bedpe


def add_pairs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `pairs build/import-calls/annotate` on `subparsers`."""
    pairs = subparsers.add_parser(
        "pairs",
        help="Build, import, and annotate explicit bait/prey pair lists",
        description=(
            "Materialize the bait/prey pair universe as BEDPE, import external loop or "
            "significance calls onto the same schema, and attach those calls to a pair list."
        ),
    )
    pairs_sub = pairs.add_subparsers(dest="pairs_command", required=True)

    build_parser = pairs_sub.add_parser(
        "build",
        help="Write the bait/prey pair universe as a BEDPE pair list",
        description=(
            "Expand two anchor BED files into the explicit cis pair list that apa and "
            "background would otherwise construct implicitly, so it can be inspected, "
            "filtered, annotated, and passed back in with --pairs-list."
        ),
    )
    build_parser.add_argument("--baits", required=True, type=Path, help="Bait anchor BED file.")
    build_parser.add_argument("--preys", required=True, type=Path, help="Prey anchor BED file.")
    build_parser.add_argument("--out", required=True, type=Path, help="Output BEDPE path.")
    build_parser.add_argument(
        "--min-distance", default=0, type=int, help="Minimum bait-prey center distance to include."
    )
    build_parser.add_argument(
        "--max-distance", type=int, help="Maximum bait-prey center distance to include."
    )
    build_parser.add_argument(
        "--id-style",
        choices=["coord", "digest"],
        default="coord",
        help="pair_id format: readable coordinates or a 16-character digest.",
    )
    build_parser.set_defaults(func=_build_pairs)

    import_parser = pairs_sub.add_parser(
        "import-calls",
        help="Import external loop or significance calls onto the touche pair schema",
        description=(
            "Read FitHiC2, HiC-DC+, MaxHiC, Mustache, Chromosight, Peakachu, HiCCUPS, or "
            "generic BEDPE calls into one table with touche pair_ids. Coordinates and "
            "chromosome names are taken as written; touche does not translate genome builds."
        ),
    )
    import_parser.add_argument("--input", required=True, type=Path, help="External call file.")
    import_parser.add_argument(
        "--format", required=True, choices=sorted(LOOP_FORMATS), help="Source tool's output format."
    )
    import_parser.add_argument("--out", required=True, type=Path, help="Output TSV path.")
    import_parser.add_argument(
        "--method", help="Method label recorded on every row. Defaults to the format name."
    )
    import_parser.add_argument(
        "--column",
        action="append",
        default=[],
        metavar="FIELD=COLUMN",
        help=(
            "Override one column mapping, e.g. --column q_value=FDR. Repeatable. Use when a "
            "tool's header differs from the built-in mapping."
        ),
    )
    import_parser.add_argument(
        "--cis-only", action="store_true", help="Drop trans calls."
    )
    import_parser.add_argument(
        "--bedpe-out", type=Path, help="Also write the imported calls as BEDPE."
    )
    import_parser.set_defaults(func=_import_calls)

    annotate_parser = pairs_sub.add_parser(
        "annotate",
        help="Attach imported calls to a pair list by anchor overlap",
        description=(
            "Join external calls onto a BEDPE pair list. touche anchors are points and "
            "external callers report bins, so matching is by anchor containment in either "
            "anchor order rather than exact pair_id equality."
        ),
    )
    annotate_parser.add_argument("--pairs", required=True, type=Path, help="BEDPE pair list.")
    annotate_parser.add_argument(
        "--calls", required=True, type=Path, help="Call file to attach."
    )
    annotate_parser.add_argument(
        "--format",
        default="bedpe",
        choices=sorted(LOOP_FORMATS),
        help="Format of --calls.",
    )
    annotate_parser.add_argument("--out", required=True, type=Path, help="Output annotated TSV path.")
    annotate_parser.add_argument(
        "--slop",
        default=0,
        type=int,
        help="Bases to widen each call anchor by before testing containment.",
    )
    annotate_parser.add_argument(
        "--prefix", default="loop_", help="Prefix for the columns added from the calls."
    )
    annotate_parser.set_defaults(func=_annotate_pairs)


def _build_pairs(args: argparse.Namespace) -> None:
    pairs = build_pair_table(
        read_bed_anchors(args.baits),
        read_bed_anchors(args.preys),
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        id_style=args.id_style,
    )
    write_bedpe(pairs, args.out)
    print_json(
        {
            "pairs": int(pairs.height),
            "chromosomes": int(pairs["chrom"].n_unique()),
            "out": str(args.out),
        }
    )


def _import_calls(args: argparse.Namespace) -> None:
    calls = read_loop_calls(
        args.input,
        format=args.format,
        columns=_parse_column_overrides(args.column),
        method=args.method,
        cis_only=args.cis_only,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    calls.write_csv(args.out, separator="\t")
    if args.bedpe_out is not None:
        write_loop_calls(calls, args.bedpe_out)
    print_json(
        {
            "calls": int(calls.height),
            "cis": int(calls["is_cis"].sum()),
            "with_q_value": int(calls["q_value"].is_not_null().sum()),
            "format": args.format,
            "out": str(args.out),
            "bedpe_out": str(args.bedpe_out) if args.bedpe_out else None,
        }
    )


def _annotate_pairs(args: argparse.Namespace) -> None:
    pairs = read_bedpe(args.pairs)
    calls = read_loop_calls(args.calls, format=args.format)
    annotated = annotate_pairs(pairs, calls, prefix=args.prefix, slop=args.slop)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    annotated.write_csv(args.out, separator="\t")
    print_json(
        {
            "pairs": int(annotated.height),
            "calls": int(calls.height),
            "matched": int(annotated[f"{args.prefix}matched"].sum()),
            "out": str(args.out),
        }
    )


def _parse_column_overrides(values: list[str]) -> dict[str, str | int] | None:
    """Parse repeated `--column FIELD=COLUMN` arguments; a numeric COLUMN is a 0-based index."""
    if not values:
        return None
    overrides: dict[str, str | int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected FIELD=COLUMN, got {value!r}")
        field, column = value.split("=", 1)
        overrides[field] = int(column) if column.lstrip("-").isdigit() else column
    return overrides
