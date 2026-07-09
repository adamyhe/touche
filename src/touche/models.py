"""Shared frozen dataclasses used across `touche`'s domain modules and CLI.

Public API: `ContactIndex`, `PairStats`, `NamedPath`, `NamedDepth`. Nothing
here is internal -- every dataclass is constructed directly by callers in
`touche.contacts`, `touche.pair_stats`, and the CLI argument parsers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ContactIndex:
    """In-memory numeric contact index for one chromosome.

    Each array is one column of the pairs file restricted to `chrom`, so
    `pos_a[i]`/`strand_a[i]`/`mapq_a[i]` etc. all describe the same contact.
    Built by `touche.contacts.build_contact_indexes`.
    """

    chrom: str
    pos_a: np.ndarray
    pos_b: np.ndarray
    strand_a: np.ndarray
    strand_b: np.ndarray
    mapq_a: np.ndarray
    mapq_b: np.ndarray

    @property
    def size(self) -> int:
        """Number of contacts (rows) in this index."""
        return int(self.pos_a.shape[0])

    def sorted_by_pos_a(self) -> "ContactIndex":
        """Return a copy with every array reordered by ascending `pos_a`.

        Local-decay and APA counting rely on `pos_a` being sorted so they can
        binary-search/slice windows instead of scanning every contact.
        """
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
    """QC counters accumulated over a pairs file by `touche.pair_stats`."""

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
    """One `NAME=PATH` CLI argument (e.g. a named treatment/control pairs file)."""

    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class NamedDepth:
    """One `NAME=INTEGER` CLI argument (e.g. a named sequencing depth)."""

    name: str
    depth: int
