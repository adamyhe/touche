"""`touche local-decay` subcommands: call, run, test, compare-groups, calibration, assign-pair-types, and plot.

`add_local_decay_parser` is the public entry point called from
`cli/main.py`. Every `_`-prefixed function below is an argparse `func=`
callback, not meant to be called directly -- it unpacks `args` and forwards
to the matching `touche.local_decay`/`touche.pipelines` function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from touche.backends import DEFAULT_FISHER_BACKEND, DEFAULT_LOWESS_BACKEND
from touche.cli.utils import add_instrumentation_args, add_timings, make_cli_instrumentation, print_json
from touche.compare import compare_groups, correlate
from touche.local_decay import (
    DECAY_MODELS,
    SIGNIFICANCE_METHODS,
    assign_pair_types,
    call_local_decay,
    plot_pair_type_distribution,
)
from touche.pipelines import run_local_decay_pipeline
from touche.significance import assess_calibration, read_local_decay_calls, test_contacts
from touche.stats import ADJUST_METHODS


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
    _add_significance_args(call_parser)
    _add_decay_model_arg(call_parser)
    call_parser.add_argument(
        "--schema",
        choices=["legacy", "tidy"],
        default="legacy",
        help=(
            "Output layout. legacy writes the reference nine-column headerless TSV, which has "
            "no room for a q_value; tidy writes the canonical pair schema with pair_id, "
            "n_trials, p_null, log2_oe, and q_value. Both write a .meta.json sidecar unless "
            "--method legacy_fisher leaves the output identical to the reference workflow's."
        ),
    )
    add_instrumentation_args(call_parser)
    call_parser.set_defaults(func=_call_local_decay)

    test_parser = local_decay_sub.add_parser(
        "test",
        help="Recompute per-pair significance and attach FDR-adjusted q-values",
        description=(
            "Retest an existing contact-call table under a named null and add q_value. "
            "method=binomial needs n_trials/p_null, so the calls must have been written "
            "with --schema tidy; the legacy nine-column layout does not carry them."
        ),
    )
    test_parser.add_argument("--calls", required=True, type=Path, help="Contact-call TSV from call.")
    test_parser.add_argument("--out", required=True, type=Path, help="Output tidy TSV path.")
    _add_significance_args(test_parser)
    test_parser.set_defaults(func=_test_contacts)

    calibration_parser = local_decay_sub.add_parser(
        "calibration",
        help="Check whether a p-value column is uniform under the null",
        description=(
            "Report KS uniformity and empirical rejection rates per stratum. Run this on "
            "null pairs (distance- and coverage-matched shifted or random pairs), not on "
            "real enhancer-promoter pairs, which are not expected to be null."
        ),
    )
    calibration_parser.add_argument("--calls", required=True, type=Path, help="Contact-call TSV.")
    calibration_parser.add_argument("--out", type=Path, help="Optional output TSV path.")
    calibration_parser.add_argument(
        "--strata",
        action="append",
        default=[],
        help="Column to stratify calibration by, e.g. --strata chrom. Repeatable.",
    )
    calibration_parser.set_defaults(func=_assess_calibration)

    compare_parser = local_decay_sub.add_parser(
        "compare-groups",
        help="Compare a value between two labelled groups of pairs",
        description=(
            "Effect size, rank test, and clustered bootstrap interval between two groups "
            "of pairs. This is a descriptive comparison: pairs are not biological "
            "replicates, so pass --cluster to resample the unit pairs actually share."
        ),
    )
    compare_parser.add_argument(
        "--table", required=True, type=Path, help="Headed TSV of pair-level rows."
    )
    compare_parser.add_argument("--out", type=Path, help="Optional output TSV path.")
    compare_parser.add_argument(
        "--value-col", required=True, help="Numeric column to compare between groups."
    )
    compare_parser.add_argument("--group-col", required=True, help="Column holding the group labels.")
    compare_parser.add_argument(
        "--groups",
        nargs=2,
        metavar=("X", "Y"),
        help="The two group levels to compare. Defaults to the two most frequent.",
    )
    compare_parser.add_argument(
        "--cluster",
        help=(
            "Column to resample whole clusters of for the confidence interval "
            "(e.g. a promoter or enhancer id). Strongly recommended."
        ),
    )
    compare_parser.add_argument(
        "--correlate-with",
        help="Instead of a group comparison, correlate --value-col against this column.",
    )
    compare_parser.add_argument(
        "--correlation-method", choices=["spearman", "pearson"], default="spearman",
        help="Correlation used by --correlate-with.",
    )
    compare_parser.add_argument(
        "--statistic", choices=["median", "mean"], default="median",
        help="Group summary whose difference is estimated.",
    )
    compare_parser.add_argument(
        "--bootstrap", default=1000, type=int, help="Bootstrap resamples for the interval."
    )
    compare_parser.add_argument(
        "--confidence", default=0.95, type=float, help="Confidence level for the interval."
    )
    compare_parser.add_argument("--seed", default=0, type=int, help="Random seed for the bootstrap.")
    compare_parser.set_defaults(func=_compare_groups)

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
    _add_method_arg(run_parser)
    _add_decay_model_arg(run_parser)
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


def _add_decay_model_arg(parser: argparse.ArgumentParser) -> None:
    """Register the `--decay-model` flag choosing how the background density is scaled."""
    parser.add_argument(
        "--decay-model",
        choices=sorted(DECAY_MODELS),
        default="normalized",
        help=(
            "How the per-bait distance-decay background is turned into a density. normalized "
            "rescales the fit to integrate to 1 and drops the robust reweighting that biases "
            "a sparse count histogram downward. legacy reproduces the reference "
            "implementation including its scale error, which makes expected counts roughly "
            "half what they should be on sparse data."
        ),
    )


def _add_method_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--method` flag and its guidance."""
    parser.add_argument(
        "--method",
        choices=sorted(SIGNIFICANCE_METHODS),
        default="binomial",
        help=(
            "Per-pair null. binomial tests the observed count against the model's own trial "
            "total and null probability. legacy_fisher reproduces the reference workflow's "
            "numbers but is a reproducibility mode, not a calibrated test."
        ),
    )


def _add_significance_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--method`/`--fdr`/`--fdr-scope` significance flags."""
    _add_method_arg(parser)
    parser.add_argument(
        "--fdr",
        choices=sorted(ADJUST_METHODS),
        default="bh",
        help="Multiple-testing correction applied to p_value to produce q_value.",
    )
    parser.add_argument(
        "--fdr-scope",
        action="append",
        default=[],
        help=(
            "Column defining a stratified FDR family, e.g. --fdr-scope chrom. Repeatable. "
            "Omit for one global family over every called pair."
        ),
    )


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
        method=args.method,
        decay_model=args.decay_model,
        schema=args.schema,
        fdr=args.fdr,
        fdr_scope=args.fdr_scope or None,
        progress=instrument,
    )
    print_json(
        add_timings(
            {
                "rows": int(len(calls)),
                "method": args.method,
                "decay_model": args.decay_model,
                "schema": args.schema,
                "out": str(args.out),
            },
            instrument,
        )
    )


def _test_contacts(args: argparse.Namespace) -> None:
    result = test_contacts(
        read_local_decay_calls(args.calls),
        method=args.method,
        fdr=args.fdr,
        fdr_scope=args.fdr_scope or None,
    )
    result.write(args.out)
    print_json({**result.to_dict(), "out": str(args.out)})


def _assess_calibration(args: argparse.Namespace) -> None:
    calibration = assess_calibration(
        read_local_decay_calls(args.calls), strata=args.strata or None
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_csv(args.out, separator="\t")
    print_json({"strata": calibration.to_dicts(), "out": str(args.out) if args.out else None})


def _compare_groups(args: argparse.Namespace) -> None:
    table = pl.read_csv(args.table, separator="\t")
    if args.correlate_with:
        result = correlate(
            table,
            x_col=args.value_col,
            y_col=args.correlate_with,
            method=args.correlation_method,
            cluster_by=args.cluster,
            bootstrap=args.bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
    else:
        result = compare_groups(
            table,
            value_col=args.value_col,
            group_col=args.group_col,
            groups=tuple(args.groups) if args.groups else None,
            cluster_by=args.cluster,
            statistic=args.statistic,
            bootstrap=args.bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
    if args.out is not None:
        result.write(args.out)
    print_json({**result.to_dict(), "result": result.table.to_dicts()[0], "out": str(args.out) if args.out else None})


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
        method=args.method,
        decay_model=args.decay_model,
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
