from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from touche import __version__
from touche.io import PairRecord, iter_pair_records, iter_pairs, open_text
from touche.models import ContactPair, FilterSettings, PairStats


def _distance_bin(distance: int) -> str:
    bins = [
        (1_000, "<1kb"),
        (10_000, "1kb-10kb"),
        (100_000, "10kb-100kb"),
        (1_000_000, "100kb-1Mb"),
        (10_000_000, "1Mb-10Mb"),
    ]
    for upper, label in bins:
        if distance < upper:
            return label
    return ">=10Mb"


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
    stats = _new_stats()
    with open_text(out_path, "wt") as out_handle:
        for _, pair, fields in iter_pairs(pairs_path, source=source):
            _observe_pair(stats, pair, min_mapq=min_mapq)
            if not _passes(pair, settings):
                continue
            stats["written_rows"] += 1
            if keep_read_id and len(fields) >= 10:
                out_handle.write("\t".join(fields[:10]) + "\n")
            else:
                out_handle.write(pair.as_tsv() + "\n")
    return _freeze_stats(stats)


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

    stats = _new_stats()
    with open_text(out_path, "wt") as out_handle:
        for _, pair, _ in iter_pairs(pairs_path, source=source):
            _observe_pair(stats, pair)
            stats["written_rows"] += 1
            out_handle.write(pair.as_tsv() + "\n")
    return _freeze_stats(stats)


def summarize_pairs(pairs_path: str | Path, *, source: str = "auto") -> PairStats:
    stats = _new_stats()
    for _, record in iter_pair_records(pairs_path, source=source):
        _observe_record(stats, record)
    return _freeze_stats(stats)


def write_qc(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    source: str = "auto",
) -> PairStats:
    stats = summarize_pairs(pairs_path, source=source)
    write_qc_payload(out_path, stats=stats, source=pairs_path)
    return stats


def write_qc_payload(
    out_path: str | Path,
    *,
    stats: PairStats,
    source: str | Path,
) -> None:
    payload = {
        "schema_version": 1,
        "touche_version": __version__,
        "source": str(source),
        "stats": asdict(stats),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _new_stats() -> dict[str, object]:
    return {
        "total_rows": 0,
        "parsed_rows": 0,
        "written_rows": 0,
        "cis_rows": 0,
        "trans_rows": 0,
        "mapq_pass_rows": 0,
        "mapq_fail_rows": 0,
        "per_chromosome": Counter(),
        "distance_histogram": Counter(),
    }


def _observe_pair(stats: dict[str, object], pair: ContactPair, *, min_mapq: int = 30) -> None:
    stats["total_rows"] += 1
    stats["parsed_rows"] += 1
    if pair.chrom_a == pair.chrom_b:
        stats["cis_rows"] += 1
        stats["per_chromosome"][pair.chrom_a] += 1
        stats["distance_histogram"][_distance_bin(abs(pair.pos_b - pair.pos_a))] += 1
    else:
        stats["trans_rows"] += 1
    if pair.mapq_a >= min_mapq and pair.mapq_b >= min_mapq:
        stats["mapq_pass_rows"] += 1
    else:
        stats["mapq_fail_rows"] += 1


def _observe_record(stats: dict[str, object], record: PairRecord, *, min_mapq: int = 30) -> None:
    chrom_a, pos_a, chrom_b, pos_b, _strand_a, _strand_b, mapq_a, mapq_b = record
    stats["total_rows"] += 1
    stats["parsed_rows"] += 1
    if chrom_a == chrom_b:
        stats["cis_rows"] += 1
        stats["per_chromosome"][chrom_a] += 1
        stats["distance_histogram"][_distance_bin(abs(pos_b - pos_a))] += 1
    else:
        stats["trans_rows"] += 1
    if mapq_a >= min_mapq and mapq_b >= min_mapq:
        stats["mapq_pass_rows"] += 1
    else:
        stats["mapq_fail_rows"] += 1


def _freeze_stats(stats: dict[str, object]) -> PairStats:
    return PairStats(
        total_rows=int(stats["total_rows"]),
        parsed_rows=int(stats["parsed_rows"]),
        written_rows=int(stats["written_rows"]),
        cis_rows=int(stats["cis_rows"]),
        trans_rows=int(stats["trans_rows"]),
        mapq_pass_rows=int(stats["mapq_pass_rows"]),
        mapq_fail_rows=int(stats["mapq_fail_rows"]),
        per_chromosome=dict(sorted(stats["per_chromosome"].items())),
        distance_histogram=dict(stats["distance_histogram"]),
    )
