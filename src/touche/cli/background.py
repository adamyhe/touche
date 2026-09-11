"""`touche background` subcommands: count, compare, diff, and run EP/background workflows.

`add_background_parser` is the public entry point called from `cli/main.py`.
Every `_`-prefixed function below is an argparse `func=` callback, not meant
to be called directly -- it unpacks `args` and forwards to the matching
`touche.background`/`touche.pipelines` function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from touche.background import (
    CPB_SCALES,
    compare_background_ratios,
    count_ep_and_background,
    parse_named_depth,
    parse_named_path,
)
from touche.cli.utils import (
    add_instrumentation_args,
    add_timings,
    make_cli_instrumentation,
    print_json,
    require_anchors_or_pair_list,
)
from touche.differential import test_background_change
from touche.pipelines import run_background_pipeline
from touche.stats import ADJUST_METHODS


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
    count_parser.add_argument("--baits", type=Path, help="BED-like promoter/bait anchors.")
    count_parser.add_argument("--preys", type=Path, help="BED-like enhancer/prey anchors.")
    count_parser.add_argument(
        "--pairs-list",
        type=Path,
        help=(
            "BEDPE of explicit promoter/enhancer pairs to count, replacing --baits/--preys. "
            "The list is the pair universe exactly as given; the distance window does not "
            "prune it."
        ),
    )
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
    count_parser.add_argument(
        "--index-strategy",
        choices=["all", "cache"],
        default="all",
        help=(
            "Contact-index strategy. cache reads a persistent NPZ ContactIndex cache "
            "(building it first if missing) instead of re-parsing --pairs -- useful when "
            "apa aggregate also runs against the same sample."
        ),
    )
    count_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory containing or receiving chromosome-sharded contact cache files.",
    )
    count_parser.add_argument("--cache-prefix", default="contacts", help="Cache manifest/shard prefix.")
    count_parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Require an existing cache instead of building one implicitly.",
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
    compare_parser.add_argument(
        "--scale",
        choices=sorted(CPB_SCALES),
        default="legacy",
        help=(
            "Divisor behind EP_CPB_*. legacy is depth/1e10 (contacts per ten billion, the "
            "reference workflow's unit despite the CPB name); per_billion is depth/1e9."
        ),
    )
    compare_parser.add_argument(
        "--zero-policy",
        choices=["drop", "keep"],
        default="drop",
        help=(
            "drop reproduces the reference plot by requiring positive EP signal in every "
            "sample. That discards complete gains and losses, so use keep (and "
            "`background diff`) for anything inferential."
        ),
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

    diff_parser = background_sub.add_parser(
        "diff",
        help="Test the EP-versus-background change between two libraries",
        description=(
            "Per-pair log odds ratio of enhancer-promoter against matched local background "
            "between a control and a treatment library, with BH-adjusted q-values. Zeros "
            "are kept: a complete gain or loss is an observation, not a missing value. "
            "This is technical inference conditional on the two observed libraries, not "
            "evidence of biological variation between conditions."
        ),
    )
    diff_parser.add_argument(
        "--control", required=True, help="Control count table as NAME=PATH."
    )
    diff_parser.add_argument(
        "--treatment", required=True, help="Treatment count table as NAME=PATH."
    )
    diff_parser.add_argument("--out", required=True, type=Path, help="Output TSV path.")
    diff_parser.add_argument(
        "--method",
        choices=["wald", "fisher"],
        default="wald",
        help="wald is the vectorized normal approximation; fisher is exact and slower.",
    )
    diff_parser.add_argument(
        "--fdr",
        choices=sorted(ADJUST_METHODS),
        default="bh",
        help="Multiple-testing correction applied across all shared pairs.",
    )
    diff_parser.add_argument(
        "--correction",
        default=0.5,
        type=float,
        help="Continuity correction added to the tested 2x2 table. 0 leaves zero cells infinite.",
    )
    diff_parser.add_argument(
        "--display-pseudocount",
        default=1.0,
        type=float,
        help="Pseudocount used only for the finite log2_ratio_change_display column.",
    )
    diff_parser.set_defaults(func=_diff_background)

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
    run_parser.add_argument(
        "--index-strategy",
        choices=["all", "cache"],
        default="all",
        help=(
            "Contact-index strategy. cache reads a persistent NPZ ContactIndex cache per "
            "sample (building it first if missing) instead of re-parsing each pairs file -- "
            "useful when apa run also runs against the same samples."
        ),
    )
    run_parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Base directory for per-sample contact caches (one subdirectory per sample name). "
            "Defaults to a contact_index_cache/ directory next to each sample's own output."
        ),
    )
    run_parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Require existing caches instead of building them implicitly.",
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
    require_anchors_or_pair_list(args)
    result = count_ep_and_background(
        args.pairs,
        args.baits,
        args.preys,
        args.out,
        pairs_list=args.pairs_list,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        window=args.window,
        min_bg_distance=args.min_bg_distance,
        max_bg_distance=args.max_bg_distance,
        source=args.source,
        index_strategy=args.index_strategy,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        require_cache=args.require_cache,
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
        scale=args.scale,
        zero_policy=args.zero_policy,
        out_dir=args.out_dir,
        table_out=args.table_out,
        reference_style=args.reference_style,
    )
    print_json(
        {
            "rows": int(len(merged)),
            "scale": args.scale,
            "zero_policy": args.zero_policy,
            "plots": {key: str(value) for key, value in plot_paths.items()},
            "table_out": str(args.table_out) if args.table_out else None,
        }
    )


def _diff_background(args: argparse.Namespace) -> None:
    result = test_background_change(
        parse_named_path(args.control),
        parse_named_path(args.treatment),
        method=args.method,
        fdr=args.fdr,
        correction=args.correction,
        display_pseudocount=args.display_pseudocount,
        out_path=args.out,
    )
    print_json({**result.to_dict(), "out": str(args.out)})


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
        index_strategy=args.index_strategy,
        cache_dir=args.cache_dir,
        require_cache=args.require_cache,
        progress=instrument,
    )
    print_json(manifest)
