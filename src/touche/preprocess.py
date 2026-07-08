from __future__ import annotations

from pathlib import Path

from touche.io import iter_pair_records, iter_pairs, open_text
from touche.models import ContactPair, FilterSettings, PairStats
from touche.pair_stats import (
    freeze_pair_stats,
    new_pair_stats,
    observe_contact_pair,
    observe_parsed_pair,
    write_qc_payload,
)


def _passes(pair: ContactPair, settings: FilterSettings) -> bool:
    if settings.cis_only and pair.chrom_a != pair.chrom_b:
        return False
    return pair.mapq_a >= settings.min_mapq and pair.mapq_b >= settings.min_mapq


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

    settings = FilterSettings(
        min_mapq=min_mapq,
        cis_only=cis_only,
        keep_read_id=keep_read_id,
        source=source,
    )
    stats = new_pair_stats()
    with open_text(out_path, "wt") as out_handle:
        for _, pair, fields in iter_pairs(pairs_path, source=source):
            observe_contact_pair(stats, pair, min_mapq=min_mapq)
            if not _passes(pair, settings):
                continue
            stats["written_rows"] += 1
            if keep_read_id and len(fields) >= 10:
                out_handle.write("\t".join(fields[:10]) + "\n")
            else:
                out_handle.write(pair.as_tsv() + "\n")
    return freeze_pair_stats(stats)


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

    stats = new_pair_stats()
    with open_text(out_path, "wt") as out_handle:
        for _, pair, _ in iter_pairs(pairs_path, source=source):
            observe_contact_pair(stats, pair)
            stats["written_rows"] += 1
            out_handle.write(pair.as_tsv() + "\n")
    return freeze_pair_stats(stats)


def summarize_pairs(pairs_path: str | Path, *, source: str = "auto") -> PairStats:
    stats = new_pair_stats()
    for _, record in iter_pair_records(pairs_path, source=source):
        observe_parsed_pair(stats, record)
    return freeze_pair_stats(stats)


def write_qc(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    source: str = "auto",
) -> PairStats:
    stats = summarize_pairs(pairs_path, source=source)
    write_qc_payload(out_path, stats=stats, source=pairs_path)
    return stats
