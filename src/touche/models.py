from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
