from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from touche import __version__
from touche.io import ParsedPairRecord
from touche.models import ContactPair, PairStats

PairStatsAccumulator = dict[str, object]


def distance_bin(distance: int) -> str:
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


def new_pair_stats() -> PairStatsAccumulator:
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


def observe_contact_pair(
    stats: PairStatsAccumulator, pair: ContactPair, *, min_mapq: int = 30
) -> None:
    observe_pair_values(
        stats,
        chrom_a=pair.chrom_a,
        pos_a=pair.pos_a,
        chrom_b=pair.chrom_b,
        pos_b=pair.pos_b,
        mapq_a=pair.mapq_a,
        mapq_b=pair.mapq_b,
        min_mapq=min_mapq,
    )


def observe_parsed_pair(
    stats: PairStatsAccumulator, record: ParsedPairRecord, *, min_mapq: int = 30
) -> None:
    observe_pair_values(
        stats,
        chrom_a=record.chrom_a,
        pos_a=record.pos_a,
        chrom_b=record.chrom_b,
        pos_b=record.pos_b,
        mapq_a=record.mapq_a,
        mapq_b=record.mapq_b,
        min_mapq=min_mapq,
    )


def observe_pair_values(
    stats: PairStatsAccumulator,
    *,
    chrom_a: str,
    pos_a: int,
    chrom_b: str,
    pos_b: int,
    mapq_a: int,
    mapq_b: int,
    min_mapq: int = 30,
) -> None:
    stats["total_rows"] += 1
    stats["parsed_rows"] += 1
    if chrom_a == chrom_b:
        stats["cis_rows"] += 1
        stats["per_chromosome"][chrom_a] += 1
        stats["distance_histogram"][distance_bin(abs(pos_b - pos_a))] += 1
    else:
        stats["trans_rows"] += 1
    if mapq_a >= min_mapq and mapq_b >= min_mapq:
        stats["mapq_pass_rows"] += 1
    else:
        stats["mapq_fail_rows"] += 1


def freeze_pair_stats(stats: PairStatsAccumulator) -> PairStats:
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
