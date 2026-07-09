"""`touche preprocess` subcommands: filter/convert pairs, QC, and NPZ cache building.

`add_preprocess_parser` is the public entry point called from `cli/main.py`.
Every `_`-prefixed function below is an argparse `func=` callback, not
meant to be called directly -- it unpacks `args` and forwards to the
matching `touche.preprocess`/`touche.contacts` function.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from touche.cli.utils import print_json
from touche.contacts import build_npz_cache
from touche.preprocess import convert_pairs, filter_pairs, summarize_pairs, write_qc


def add_preprocess_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `preprocess filter-pairs/convert-pairs/qc/summarize/build-cache` on `subparsers`."""
    preprocess = subparsers.add_parser("preprocess", help="Prepare Micro-C pairs for analysis")
    preprocess_sub = preprocess.add_subparsers(dest="preprocess_command", required=True)

    filter_parser = preprocess_sub.add_parser(
        "filter-pairs",
        help="Filter pairs into the canonical touche format",
        description=(
            "Read a distiller/pairtools or canonical touche pairs file, apply MAPQ and "
            "cis/trans filters, and write an analysis-ready pairs file."
        ),
    )
    filter_parser.add_argument("--pairs", required=True, type=Path, help="Input .pairs/.pairs.gz file.")
    filter_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path. Use a .gz suffix to write gzip-compressed pairs.",
    )
    filter_parser.add_argument(
        "--min-mapq",
        default=30,
        type=int,
        help="Minimum MAPQ required on both sides of a pair.",
    )
    filter_parser.add_argument(
        "--cis-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only same-chromosome pairs. Use --no-cis-only to keep trans pairs.",
    )
    filter_parser.add_argument(
        "--keep-read-id",
        action="store_true",
        help="Retain the leading read_id column when present instead of writing canonical 9-column output.",
    )
    filter_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input layout. auto infers distiller vs canonical touche rows from the first data row.",
    )
    filter_parser.set_defaults(func=_filter_pairs)

    convert_parser = preprocess_sub.add_parser(
        "convert-pairs",
        help="Convert pairs formats without filtering",
    )
    convert_parser.add_argument("--pairs", required=True, type=Path, help="Input .pairs/.pairs.gz file.")
    convert_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path. Use a .gz suffix to write gzip-compressed pairs.",
    )
    convert_parser.add_argument(
        "--from",
        dest="source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input layout to convert from.",
    )
    convert_parser.add_argument(
        "--to",
        dest="target",
        choices=["touche"],
        default="touche",
        help="Output layout. Currently only canonical touche format is supported.",
    )
    convert_parser.set_defaults(func=_convert_pairs)

    qc_parser = preprocess_sub.add_parser("qc", help="Write pair QC summary JSON")
    qc_parser.add_argument("--pairs", required=True, type=Path, help="Input pairs file to summarize.")
    qc_parser.add_argument("--out", required=True, type=Path, help="QC JSON output path.")
    qc_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input layout.",
    )
    qc_parser.set_defaults(func=_qc)

    summarize_parser = preprocess_sub.add_parser("summarize", help="Print pair summary JSON")
    summarize_parser.add_argument(
        "--pairs",
        required=True,
        type=Path,
        help="Input pairs file to summarize.",
    )
    summarize_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input layout.",
    )
    summarize_parser.set_defaults(func=_summarize)

    cache_parser = preprocess_sub.add_parser(
        "build-cache",
        help="Build chromosome-sharded NPZ contact caches",
        description=(
            "Build reusable contact-index shards for local-decay and repeated analyses. "
            "Use --no-metadata for smaller position-only caches when strand/MAPQ arrays "
            "are not needed."
        ),
    )
    cache_parser.add_argument("--pairs", required=True, type=Path, help="Input pairs file.")
    cache_parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Directory where cache shards and manifest will be written.",
    )
    cache_parser.add_argument(
        "--source",
        choices=["auto", "distiller", "touche"],
        default="auto",
        help="Input layout.",
    )
    cache_parser.add_argument(
        "--prefix",
        default="contacts",
        help="Prefix for the cache manifest, QC JSON, and chromosome shard files.",
    )
    cache_parser.add_argument(
        "--compressed",
        action="store_true",
        help="Write compressed NPZ shards. Smaller on disk, slower to build/load.",
    )
    cache_parser.add_argument(
        "--cis-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache only same-chromosome pairs.",
    )
    cache_parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store strand and MAPQ arrays in cache shards. Use --no-metadata for position-only caches.",
    )
    cache_parser.add_argument(
        "--index-strategy",
        choices=["chromosome", "all"],
        default="chromosome",
        help=(
            "Cache build strategy. chromosome bounds memory by spooling and writing one "
            "chromosome at a time; all holds the whole genome in memory."
        ),
    )
    cache_parser.add_argument(
        "--qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write pair QC JSON while building the cache.",
    )
    cache_parser.add_argument(
        "--qc-out",
        type=Path,
        help="Override the QC JSON path. Defaults to CACHE_DIR/PREFIX.qc.json.",
    )
    cache_parser.set_defaults(func=_build_cache)


def _filter_pairs(args: argparse.Namespace) -> None:
    stats = filter_pairs(
        args.pairs,
        args.out,
        min_mapq=args.min_mapq,
        cis_only=args.cis_only,
        keep_read_id=args.keep_read_id,
        source=args.source,
    )
    print_json(asdict(stats))


def _convert_pairs(args: argparse.Namespace) -> None:
    stats = convert_pairs(args.pairs, args.out, source=args.source, target=args.target)
    print_json(asdict(stats))


def _qc(args: argparse.Namespace) -> None:
    stats = write_qc(args.pairs, args.out, source=args.source)
    print_json(asdict(stats))


def _summarize(args: argparse.Namespace) -> None:
    stats = summarize_pairs(args.pairs, source=args.source)
    print_json(asdict(stats))


def _build_cache(args: argparse.Namespace) -> None:
    paths = build_npz_cache(
        args.pairs,
        args.cache_dir,
        source=args.source,
        prefix=args.prefix,
        compressed=args.compressed,
        cis_only=args.cis_only,
        include_metadata=args.metadata,
        index_strategy=args.index_strategy,
        qc_out=args.qc_out,
        write_qc=args.qc,
    )
    print_json({"written": [str(path) for path in paths]})
