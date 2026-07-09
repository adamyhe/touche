"""QC counter accumulation shared by `preprocess qc`, `preprocess summarize`, and cache building.

Public API: `compute_pair_stats`, `write_qc_payload`, `distance_bin`,
`DISTANCE_BINS`. `_distance_bin_expr` is an internal Polars-expression
mirror of `distance_bin` used inside lazy aggregations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from touche import __version__
from touche.models import PairStats

DISTANCE_BINS: list[tuple[int, str]] = [
    (1_000, "<1kb"),
    (10_000, "1kb-10kb"),
    (100_000, "10kb-100kb"),
    (1_000_000, "100kb-1Mb"),
    (10_000_000, "1Mb-10Mb"),
]


def distance_bin(distance: int) -> str:
    """Label a contact distance using the same buckets as the distance histogram."""
    for upper, label in DISTANCE_BINS:
        if distance < upper:
            return label
    return ">=10Mb"


def _distance_bin_expr(distance: pl.Expr) -> pl.Expr:
    """`distance_bin`, expressed as a chained `pl.when`/`.then` for use inside a lazy aggregation."""
    expr = pl.when(distance < DISTANCE_BINS[0][0]).then(pl.lit(DISTANCE_BINS[0][1]))
    for upper, label in DISTANCE_BINS[1:]:
        expr = expr.when(distance < upper).then(pl.lit(label))
    return expr.otherwise(pl.lit(">=10Mb"))


def compute_pair_stats(
    lf: pl.LazyFrame, *, min_mapq: int = 30, written_rows: int = 0
) -> PairStats:
    """Compute QC stats from a lazy frame of pairs.

    `lf` must have `chrom_a`/`chrom_b`/`pos_a`/`pos_b`/`mapq_a`/`mapq_b` columns.
    Aggregations run against the lazy plan with the streaming engine so peak
    memory is bounded by the number of distinct chromosomes/distance buckets,
    not the row count.
    """

    lf = lf.with_columns(
        (pl.col("chrom_a") == pl.col("chrom_b")).alias("is_cis"),
        ((pl.col("mapq_a") >= min_mapq) & (pl.col("mapq_b") >= min_mapq)).alias("mapq_pass"),
    )
    totals = (
        lf.select(
            pl.len().alias("total_rows"),
            pl.col("is_cis").sum().alias("cis_rows"),
            (~pl.col("is_cis")).sum().alias("trans_rows"),
            pl.col("mapq_pass").sum().alias("mapq_pass_rows"),
            (~pl.col("mapq_pass")).sum().alias("mapq_fail_rows"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )

    cis_lf = lf.filter(pl.col("is_cis")).with_columns(
        _distance_bin_expr((pl.col("pos_b") - pl.col("pos_a")).abs()).alias("distance_bin")
    )
    per_chrom = cis_lf.group_by("chrom_a").agg(pl.len().alias("n")).collect(engine="streaming")
    per_chromosome = dict(sorted(zip(per_chrom["chrom_a"].to_list(), per_chrom["n"].to_list())))

    hist = cis_lf.group_by("distance_bin").agg(pl.len().alias("n")).collect(engine="streaming")
    distance_histogram = dict(zip(hist["distance_bin"].to_list(), hist["n"].to_list()))

    return PairStats(
        total_rows=int(totals["total_rows"]),
        parsed_rows=int(totals["total_rows"]),
        written_rows=written_rows,
        cis_rows=int(totals["cis_rows"]),
        trans_rows=int(totals["trans_rows"]),
        mapq_pass_rows=int(totals["mapq_pass_rows"]),
        mapq_fail_rows=int(totals["mapq_fail_rows"]),
        per_chromosome=per_chromosome,
        distance_histogram=distance_histogram,
    )


def write_qc_payload(
    out_path: str | Path,
    *,
    stats: PairStats,
    source: str | Path,
) -> None:
    """Write `stats` as the versioned JSON payload used by `preprocess qc`/`summarize` and caching."""
    payload = {
        "schema_version": 1,
        "touche_version": __version__,
        "source": str(source),
        "stats": asdict(stats),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
