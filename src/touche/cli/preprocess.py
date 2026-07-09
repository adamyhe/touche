from __future__ import annotations

import argparse
from pathlib import Path

from touche.cli.utils import print_dataclass, print_json
from touche.contacts import build_npz_cache
from touche.preprocess import convert_pairs, filter_pairs, summarize_pairs, write_qc


def add_preprocess_parser(subparsers: argparse._SubParsersAction) -> None:
    preprocess = subparsers.add_parser("preprocess", help="Prepare Micro-C pairs for analysis")
    preprocess_sub = preprocess.add_subparsers(dest="preprocess_command", required=True)

    filter_parser = preprocess_sub.add_parser("filter-pairs", help="Filter pairs into touche format")
    filter_parser.add_argument("--pairs", required=True, type=Path)
    filter_parser.add_argument("--out", required=True, type=Path)
    filter_parser.add_argument("--min-mapq", default=30, type=int)
    filter_parser.add_argument("--cis-only", action=argparse.BooleanOptionalAction, default=True)
    filter_parser.add_argument("--keep-read-id", action="store_true")
    filter_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    filter_parser.set_defaults(func=_filter_pairs)

    convert_parser = preprocess_sub.add_parser("convert-pairs", help="Convert pairs formats")
    convert_parser.add_argument("--pairs", required=True, type=Path)
    convert_parser.add_argument("--out", required=True, type=Path)
    convert_parser.add_argument("--from", dest="source", choices=["auto", "distiller", "touche"], default="auto")
    convert_parser.add_argument("--to", dest="target", choices=["touche"], default="touche")
    convert_parser.set_defaults(func=_convert_pairs)

    qc_parser = preprocess_sub.add_parser("qc", help="Write pair QC summary JSON")
    qc_parser.add_argument("--pairs", required=True, type=Path)
    qc_parser.add_argument("--out", required=True, type=Path)
    qc_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    qc_parser.set_defaults(func=_qc)

    summarize_parser = preprocess_sub.add_parser("summarize", help="Print pair summary JSON")
    summarize_parser.add_argument("--pairs", required=True, type=Path)
    summarize_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    summarize_parser.set_defaults(func=_summarize)

    cache_parser = preprocess_sub.add_parser("build-cache", help="Build chromosome-sharded NPZ caches")
    cache_parser.add_argument("--pairs", required=True, type=Path)
    cache_parser.add_argument("--cache-dir", required=True, type=Path)
    cache_parser.add_argument("--source", choices=["auto", "distiller", "touche"], default="auto")
    cache_parser.add_argument("--prefix", default="contacts")
    cache_parser.add_argument("--compressed", action="store_true")
    cache_parser.add_argument("--cis-only", action=argparse.BooleanOptionalAction, default=True)
    cache_parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store strand and MAPQ arrays in cache shards. Use --no-metadata for position-only caches.",
    )
    cache_parser.add_argument("--index-strategy", choices=["chromosome", "all"], default="chromosome")
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
    print_dataclass(stats)


def _convert_pairs(args: argparse.Namespace) -> None:
    stats = convert_pairs(args.pairs, args.out, source=args.source, target=args.target)
    print_dataclass(stats)


def _qc(args: argparse.Namespace) -> None:
    stats = write_qc(args.pairs, args.out, source=args.source)
    print_dataclass(stats)


def _summarize(args: argparse.Namespace) -> None:
    stats = summarize_pairs(args.pairs, source=args.source)
    print_dataclass(stats)


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
