"""BED anchor file loading. Public API: `read_bed_anchors`."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_bed_anchors(path: str | Path) -> pl.DataFrame:
    """Read BED3/BED4 anchors and add an integer center column."""

    data = pl.read_csv(path, separator="\t", has_header=False, comment_prefix="#")
    if data.width < 3:
        raise ValueError(f"Expected at least three BED columns in {path}")
    if data.width >= 4:
        anchors = data.select(data.columns[:4])
        anchors.columns = ["chr", "start", "end", "strand"]
    else:
        anchors = data.select(data.columns[:3])
        anchors.columns = ["chr", "start", "end"]
        anchors = anchors.with_columns(pl.lit(".").alias("strand"))
    anchors = anchors.with_columns(
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
    )
    return anchors.with_columns(((pl.col("start") + pl.col("end")) // 2).alias("center"))
