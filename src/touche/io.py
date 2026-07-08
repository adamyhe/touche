from __future__ import annotations

import gzip
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from touche.models import ContactPair

PairRecord = tuple[str, int, str, int, str, str, int, int]


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
    for line_number, line in enumerate(handle, start=1):
        stripped = line.rstrip("\n")
        if not stripped or stripped.startswith("#"):
            continue
        yield line_number, stripped


def parse_pair_fields(fields: list[str], *, source: str = "auto") -> ContactPair:
    """Parse distiller/pairtools-style or canonical touche pair fields.

    Supported data rows:
    - 10+ columns: read_id, chrom1, pos1, chrom2, pos2, strand1, strand2, type, mapq1, mapq2
    - 9 columns: chrom1, pos1, chrom2, pos2, strand1, strand2, type, mapq1, mapq2
    """

    if source not in {"auto", "distiller", "touche"}:
        raise ValueError(f"Unsupported pair source: {source}")

    if source == "touche" or (source == "auto" and len(fields) == 9):
        offset = 0
    elif source == "distiller" or (source == "auto" and len(fields) >= 10):
        offset = 1
    else:
        raise ValueError(f"Expected 9 canonical or 10+ distiller columns, found {len(fields)}")

    try:
        return ContactPair(
            chrom_a=fields[offset],
            pos_a=int(fields[offset + 1]),
            chrom_b=fields[offset + 2],
            pos_b=int(fields[offset + 3]),
            strand_a=fields[offset + 4],
            strand_b=fields[offset + 5],
            read_type=fields[offset + 6],
            mapq_a=int(fields[offset + 7]),
            mapq_b=int(fields[offset + 8]),
        )
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed pair row with {len(fields)} columns") from exc


def parse_pair_record(fields: list[str], *, source: str = "auto") -> PairRecord:
    """Parse only the columns needed for QC and numeric contact indexes."""

    if source not in {"auto", "distiller", "touche"}:
        raise ValueError(f"Unsupported pair source: {source}")

    if source == "touche" or (source == "auto" and len(fields) == 9):
        offset = 0
    elif source == "distiller" or (source == "auto" and len(fields) >= 10):
        offset = 1
    else:
        raise ValueError(f"Expected 9 canonical or 10+ distiller columns, found {len(fields)}")

    try:
        return (
            fields[offset],
            int(fields[offset + 1]),
            fields[offset + 2],
            int(fields[offset + 3]),
            fields[offset + 4],
            fields[offset + 5],
            int(fields[offset + 7]),
            int(fields[offset + 8]),
        )
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed pair row with {len(fields)} columns") from exc


def iter_pairs(
    path: str | Path, *, source: str = "auto"
) -> Iterator[tuple[int, ContactPair, list[str]]]:
    """Yield parsed contact pairs from plain or gzipped text."""

    with open_text(path, "rt") as handle:
        for line_number, line in iter_noncomment_lines(handle):
            fields = line.split("\t")
            yield line_number, parse_pair_fields(fields, source=source), fields


def iter_pair_records(path: str | Path, *, source: str = "auto") -> Iterator[tuple[int, PairRecord]]:
    """Yield lightweight parsed pair records without constructing ContactPair objects."""

    with open_text(path, "rt") as handle:
        for line_number, line in iter_noncomment_lines(handle):
            fields = line.split("\t", 10)
            yield line_number, parse_pair_record(fields, source=source)
