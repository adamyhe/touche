"""Lazy scanning of pairs files. Public API: `scan_pairs`, `open_text`, `iter_noncomment_lines`.

`_resolve_pair_columns` is internal -- it backs `scan_pairs`'s `source="auto"`
sniffing and isn't meant to be called directly.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import polars as pl

PAIR_SOURCES = {"auto", "distiller", "touche"}

TOUCHE_COLUMNS = [
    "chrom_a",
    "pos_a",
    "chrom_b",
    "pos_b",
    "strand_a",
    "strand_b",
    "read_type",
    "mapq_a",
    "mapq_b",
]
DISTILLER_COLUMNS = ["read_id", *TOUCHE_COLUMNS]

PAIR_DTYPES: dict[str, pl.DataType] = {
    "read_id": pl.Utf8,
    "chrom_a": pl.Utf8,
    "pos_a": pl.Int64,
    "chrom_b": pl.Utf8,
    "pos_b": pl.Int64,
    "strand_a": pl.Utf8,
    "strand_b": pl.Utf8,
    "read_type": pl.Utf8,
    "mapq_a": pl.Int16,
    "mapq_b": pl.Int16,
}


@contextmanager
def open_text(path: str | Path, mode: str = "rt") -> Iterator[TextIO]:
    """Open plain text or gzip-compressed text using UTF-8."""

    path = Path(path)
    if "b" in mode:
        raise ValueError("open_text only supports text modes")
    if path.suffix == ".gz":
        with gzip.open(path, mode, encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open(mode, encoding="utf-8", newline="") as handle:
            yield handle


def iter_noncomment_lines(handle: TextIO) -> Iterator[tuple[int, str]]:
    """Yield `(1-based line number, stripped line)` pairs, skipping blank lines and `#` comments."""
    for line_number, line in enumerate(handle, start=1):
        stripped = line.rstrip("\n")
        if not stripped or stripped.startswith("#"):
            continue
        yield line_number, stripped


def _resolve_pair_columns(path: str | Path, source: str) -> list[str]:
    """Resolve the canonical touche vs. distiller/pairtools column layout.

    `source="auto"` sniffs the layout from the first data row's field count,
    matching the CLI's documented `--source` semantics.
    """

    if source not in PAIR_SOURCES:
        raise ValueError(f"Unsupported pair source: {source}")
    if source == "touche":
        return TOUCHE_COLUMNS
    if source == "distiller":
        return DISTILLER_COLUMNS

    with open_text(path, "rt") as handle:
        for _, line in iter_noncomment_lines(handle):
            field_count = line.count("\t") + 1
            break
        else:
            return TOUCHE_COLUMNS

    if field_count == 9:
        return TOUCHE_COLUMNS
    if field_count >= 10:
        return DISTILLER_COLUMNS
    raise ValueError(f"Expected 9 canonical or 10+ distiller columns, found {field_count}")


def scan_pairs(path: str | Path, *, source: str = "auto") -> pl.LazyFrame:
    """Lazily scan a plain or gzip-compressed pairs file.

    Supports the canonical 9-column `touche` layout and the 10+ column
    `distiller`/pairtools layout (leading `read_id` column); any trailing
    columns beyond the 10th are ignored.
    """

    columns = _resolve_pair_columns(path, source)
    schema = {name: PAIR_DTYPES[name] for name in columns}
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        schema=schema,
        truncate_ragged_lines=True,
    )
