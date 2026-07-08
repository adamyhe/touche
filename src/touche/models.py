from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ContactPair:
    """Canonical 9-column contact pair used by touche workflows."""

    chrom_a: str
    pos_a: int
    chrom_b: str
    pos_b: int
    strand_a: str
    strand_b: str
    read_type: str
    mapq_a: int
    mapq_b: int

    def as_tsv(self) -> str:
        return "\t".join(
            [
                self.chrom_a,
                str(self.pos_a),
                self.chrom_b,
                str(self.pos_b),
                self.strand_a,
                self.strand_b,
                self.read_type,
                str(self.mapq_a),
                str(self.mapq_b),
            ]
        )


@dataclass(frozen=True, slots=True)
class ContactIndex:
    """In-memory numeric contact index for one chromosome."""

    chrom: str
    pos_a: np.ndarray
    pos_b: np.ndarray
    strand_a: np.ndarray
    strand_b: np.ndarray
    mapq_a: np.ndarray
    mapq_b: np.ndarray

    @property
    def size(self) -> int:
        return int(self.pos_a.shape[0])

    def sorted_by_pos_a(self) -> "ContactIndex":
        order = np.argsort(self.pos_a, kind="mergesort")
        return ContactIndex(
            chrom=self.chrom,
            pos_a=self.pos_a[order],
            pos_b=self.pos_b[order],
            strand_a=self.strand_a[order],
            strand_b=self.strand_b[order],
            mapq_a=self.mapq_a[order],
            mapq_b=self.mapq_b[order],
        )


@dataclass(frozen=True, slots=True)
class FilterSettings:
    min_mapq: int = 30
    cis_only: bool = True
    keep_read_id: bool = False
    source: str = "auto"


@dataclass(frozen=True, slots=True)
class PairStats:
    total_rows: int
    parsed_rows: int
    written_rows: int
    cis_rows: int
    trans_rows: int
    mapq_pass_rows: int
    mapq_fail_rows: int
    per_chromosome: dict[str, int]
    distance_histogram: dict[str, int]


@dataclass(frozen=True, slots=True)
class NamedPath:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class NamedDepth:
    name: str
    depth: int
