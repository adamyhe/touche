from __future__ import annotations

from pathlib import Path

import polars as pl

from touche.io import DISTILLER_COLUMNS, TOUCHE_COLUMNS, scan_pairs
from touche.models import PairStats
from touche.pair_stats import compute_pair_stats, write_qc_payload

_STATS_COLUMNS = ["chrom_a", "chrom_b", "pos_a", "pos_b", "mapq_a", "mapq_b"]


def _write_pairs_csv(frame: pl.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    compression = "gzip" if out_path.suffix == ".gz" else "uncompressed"
    frame.write_csv(out_path, include_header=False, separator="\t", compression=compression)


def filter_pairs(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    min_mapq: int = 30,
    cis_only: bool = True,
    keep_read_id: bool = False,
    source: str = "auto",
) -> PairStats:
    """Filter pairs into the canonical touche 9-column format by default."""

    lf = scan_pairs(pairs_path, source=source)
    has_read_id = "read_id" in lf.collect_schema().names()

    pass_expr = (pl.col("mapq_a") >= min_mapq) & (pl.col("mapq_b") >= min_mapq)
    if cis_only:
        pass_expr = pass_expr & (pl.col("chrom_a") == pl.col("chrom_b"))

    out_columns = DISTILLER_COLUMNS[:10] if (keep_read_id and has_read_id) else TOUCHE_COLUMNS
    filtered = lf.filter(pass_expr).select(out_columns).collect(engine="streaming")
    _write_pairs_csv(filtered, out_path)

    stats = compute_pair_stats(
        lf.select(_STATS_COLUMNS), min_mapq=min_mapq, written_rows=filtered.height
    )
    return stats


def convert_pairs(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    source: str = "auto",
    target: str = "touche",
) -> PairStats:
    """Convert supported pair formats.

    The initial implementation supports conversion into the canonical 9-column
    touche format. It intentionally does not filter rows.
    """

    if target != "touche":
        raise ValueError(f"Unsupported conversion target: {target}")

    lf = scan_pairs(pairs_path, source=source)
    converted = lf.select(TOUCHE_COLUMNS).collect(engine="streaming")
    _write_pairs_csv(converted, out_path)

    stats = compute_pair_stats(lf.select(_STATS_COLUMNS), written_rows=converted.height)
    return stats


def summarize_pairs(pairs_path: str | Path, *, source: str = "auto") -> PairStats:
    lf = scan_pairs(pairs_path, source=source).select(_STATS_COLUMNS)
    return compute_pair_stats(lf)


def write_qc(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    source: str = "auto",
) -> PairStats:
    stats = summarize_pairs(pairs_path, source=source)
    write_qc_payload(out_path, stats=stats, source=pairs_path)
    return stats
